from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from safe_rl.modules import ActorCritic
from safe_rl.storage import RolloutStorageCMDP


def _pcgrad(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Project-and-average two gradients (PCGrad), following MACPO's trpo.py."""
    g11 = torch.dot(g1, g1).item()
    g12 = torch.dot(g1, g2).item()
    g22 = torch.dot(g2, g2).item()
    if g12 < 0:
        return ((1 - g12 / (g11 + 1e-8)) * g1 + (1 - g12 / (g22 + 1e-8)) * g2) / 2
    return (g1 + g2) / 2


class PCRPO:
    """
    Projected-based Constrained Reward Policy Optimization (PCRPO).

    Adapted (single-agent) from the MACPO reference implementation in
    https://github.com/xiaoBOSS97/Multi-Agent-Constrained-Policy-Optimisation.
    Each update performs (1) a TRPO natural-gradient step on the reward
    surrogate, (2) a TRPO natural-gradient step on the cost surrogate, and
    (3) either combines the two via PCGrad when a cost violation trigger
    fires or falls back to the reward-only step.
    """

    policy: ActorCritic

    def __init__(
        self,
        policy: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 40.0,
        use_clipped_value_loss: bool = True,
        clip_param: float = 0.2,
        schedule: str = "fixed",
        desired_kl: Optional[float] = None,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # PCRPO specific parameters
        cost_limits: Optional[List[float]] = None,
        cost_loss_coef: float = 1.0,
        use_clipped_cost_loss: bool = True,
        kl_threshold: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        ls_step: int = 10,
        line_search_fraction: float = 0.5,
        fraction_coef: float = 0.5,
        projection_threshold: Optional[float] = None,
        # Backward compatibility
        rnd_cfg: Optional[Dict[str, Any]] = None,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        multi_gpu_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        if cost_limits is None:
            raise ValueError("cost_limits must be provided for PCRPO")
        self.cost_limits = cost_limits
        self.num_costs = len(cost_limits)

        self.policy = policy
        self.policy.to(self.device)
        self._validate_cost_critic()

        self.storage: Optional[RolloutStorageCMDP] = None
        self.transition = RolloutStorageCMDP.Transition()

        # Critic-only optimizer: actor is updated via natural-gradient steps.
        critic_params = list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=learning_rate)

        # PPO-style hyperparameters retained for value-loss shape only.
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.cost_loss_coef = cost_loss_coef
        self.use_clipped_cost_loss = use_clipped_cost_loss

        # TRPO / PCRPO knobs
        self.kl_threshold = kl_threshold
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.ls_step = ls_step
        self.line_search_fraction = line_search_fraction
        self.fraction_coef = fraction_coef
        # Trigger projection once mean episode cost exceeds this threshold.
        # Default: total cost budget — matches the "2S/4S" soft-region idea in the PCRPO paper.
        self.projection_threshold = (
            projection_threshold if projection_threshold is not None else float(sum(self.cost_limits))
        )

        # Runner expects these attributes even though PCRPO does not use them.
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None
        self.is_multi_gpu = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0
        self.symmetry = None

    # ------------------------------------------------------------------ storage / rollout
    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple,
        critic_obs_shape: Tuple,
        action_shape: Tuple,
    ) -> None:
        cost_shape = (self.num_costs,) if self.num_costs > 1 else None
        self.storage = RolloutStorageCMDP(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            training_type=training_type,
            cost_shape=cost_shape,
            device=self.device,
        )

    def test_mode(self) -> None:
        self.policy.eval()

    def train_mode(self) -> None:
        self.policy.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self, rewards: torch.Tensor, costs: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.costs = self._format_costs_tensor(costs)
        self.transition.dones = dones

        if "time_outs" in infos:
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * timeout_mask, 1)
            if self.transition.cost_values is not None:
                expanded_mask = timeout_mask.expand(-1, self.num_costs)
                self.transition.costs += self.gamma * self.transition.cost_values * expanded_mask

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _format_costs_tensor(self, costs: torch.Tensor) -> torch.Tensor:
        if isinstance(costs, list):
            return torch.stack([cost.clone() for cost in costs], dim=1)
        if costs.dim() == 1:
            if self.num_costs == 1:
                return costs.unsqueeze(1).clone()
            return costs.unsqueeze(1).expand(-1, self.num_costs).clone()
        return costs.clone()

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values,
            self.gamma,
            self.lam,
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch,
        )

    # ------------------------------------------------------------------ parameter utils
    @staticmethod
    def _flat_params(module: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in module.parameters()])

    @staticmethod
    def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
        idx = 0
        for p in module.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n

    @staticmethod
    def _flat_grad(grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        flat = []
        for g in grads:
            if g is None:
                continue
            flat.append(g.contiguous().view(-1))
        return torch.cat(flat)

    # ------------------------------------------------------------------ KL / Fisher
    def _kl_gaussian(self, mu_new: torch.Tensor, std_new: torch.Tensor,
                     mu_old: torch.Tensor, std_old: torch.Tensor) -> torch.Tensor:
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        log_std_new = torch.log(std_new + 1e-8)
        log_std_old = torch.log(std_old + 1e-8)
        kl = log_std_old - log_std_new + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2) + 1e-8) - 0.5
        return kl.sum(dim=-1)

    def _current_policy_kl(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """KL of current policy vs. a detached copy of itself — used for Fisher-vector products."""
        self.policy.act(obs_batch)
        mu = self.policy.action_mean
        std = self.policy.action_std
        return self._kl_gaussian(mu, std, mu.detach(), std.detach()).mean()

    def _fisher_vector_product(self, obs_batch: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        kl = self._current_policy_kl(obs_batch)
        grads = torch.autograd.grad(kl, self.policy.actor.parameters(), create_graph=True, allow_unused=True)
        flat_grad_kl = self._flat_grad(grads)
        kl_v = (flat_grad_kl * vector).sum()
        hvp = torch.autograd.grad(kl_v, self.policy.actor.parameters(), retain_graph=True, allow_unused=True)
        flat_hvp = self._flat_grad(hvp)
        return flat_hvp + self.cg_damping * vector

    def _conjugate_gradient(self, obs_batch: torch.Tensor, b: torch.Tensor, residual_tol: float = 1e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.cg_iters):
            Avp = self._fisher_vector_product(obs_batch, p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr
        return x

    # ------------------------------------------------------------------ critics
    def _value_loss(self, values: torch.Tensor, value_preds: torch.Tensor, returns: torch.Tensor,
                    use_clipped: bool) -> torch.Tensor:
        if use_clipped:
            clipped = value_preds + (values - value_preds).clamp(-self.clip_param, self.clip_param)
            errors = (values - returns).pow(2)
            errors_clipped = (clipped - returns).pow(2)
            return torch.max(errors, errors_clipped).mean()
        return (returns - values).pow(2).mean()

    def _update_critics(self, batch: Tuple) -> Tuple[float, float]:
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
         returns_cost_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

        value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        cost_value_batch = self.policy.evaluate_cost(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

        value_loss = self._value_loss(value_batch, target_values_batch, returns_batch, self.use_clipped_value_loss)
        if self.use_clipped_cost_loss:
            cost_clipped = target_cost_values_batch + (cost_value_batch - target_cost_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            cost_losses = (cost_value_batch - returns_cost_batch).pow(2)
            cost_losses_clipped = (cost_clipped - returns_cost_batch).pow(2)
            cost_loss = torch.max(cost_losses, cost_losses_clipped).mean(dim=0).sum()
        else:
            cost_loss = (returns_cost_batch - cost_value_batch).pow(2).mean(dim=0).sum()

        total_critic_loss = self.value_loss_coef * value_loss + self.cost_loss_coef * cost_loss
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()
        return value_loss.item(), cost_loss.item()

    # ------------------------------------------------------------------ TRPO sub-steps
    def _reward_surrogate(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor,
                          old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        self.policy.act(obs_batch)
        log_probs = self.policy.get_actions_log_prob(actions_batch)
        ratio = torch.exp(log_probs - torch.squeeze(old_log_probs))
        return -(ratio * torch.squeeze(advantages)).mean()

    def _cost_surrogate(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor,
                        old_log_probs: torch.Tensor, cost_advantages: torch.Tensor) -> torch.Tensor:
        self.policy.act(obs_batch)
        log_probs = self.policy.get_actions_log_prob(actions_batch)
        ratio = torch.exp(log_probs - torch.squeeze(old_log_probs))
        # Collapse multi-cost advantage to a single direction by summing over costs.
        cost_adv = cost_advantages.view(ratio.shape[0], -1).sum(dim=-1)
        return (ratio * cost_adv).mean()

    def _trpo_step_reward(self, obs_batch, actions_batch, old_log_probs, advantages,
                          old_mu, old_sigma) -> Tuple[torch.Tensor, float]:
        """Run a TRPO natural-gradient step targeting the reward surrogate and return the
        parameter delta (best_params - pre_params) along with the accepted loss improvement."""
        reward_loss = self._reward_surrogate(obs_batch, actions_batch, old_log_probs, advantages)
        grads = torch.autograd.grad(reward_loss, self.policy.actor.parameters(), retain_graph=True, allow_unused=True)
        flat_grad = self._flat_grad(grads).detach()

        step_dir = self._conjugate_gradient(obs_batch, flat_grad)
        shs = 0.5 * torch.dot(step_dir, self._fisher_vector_product(obs_batch, step_dir))
        lm = torch.sqrt(shs / self.kl_threshold + 1e-8)
        fullstep = step_dir / (lm + 1e-8)

        pre_params = self._flat_params(self.policy.actor)
        reward_loss_np = reward_loss.detach().cpu().item()
        best_delta = torch.zeros_like(pre_params)
        best_improve = 0.0

        for i in range(self.ls_step):
            new_params = pre_params - self.fraction_coef * (self.line_search_fraction ** i) * fullstep
            self._set_flat_params(self.policy.actor, new_params)
            with torch.no_grad():
                new_reward_loss = self._reward_surrogate(obs_batch, actions_batch, old_log_probs, advantages)
                self.policy.act(obs_batch)
                kl = self._kl_gaussian(
                    self.policy.action_mean, self.policy.action_std, old_mu, old_sigma
                ).mean()
            loss_improve = new_reward_loss.item() - reward_loss_np
            if kl.item() < self.kl_threshold and loss_improve < best_improve:
                best_improve = loss_improve
                best_delta = new_params - pre_params

        self._set_flat_params(self.policy.actor, pre_params)
        return best_delta, best_improve

    def _trpo_step_cost(self, obs_batch, actions_batch, old_log_probs, cost_advantages,
                        old_mu, old_sigma) -> Tuple[torch.Tensor, float]:
        cost_loss = self._cost_surrogate(obs_batch, actions_batch, old_log_probs, cost_advantages)
        grads = torch.autograd.grad(cost_loss, self.policy.actor.parameters(), retain_graph=True, allow_unused=True)
        flat_grad = self._flat_grad(grads).detach()

        step_dir = self._conjugate_gradient(obs_batch, flat_grad)
        shs = 0.5 * torch.dot(step_dir, self._fisher_vector_product(obs_batch, step_dir))
        lm = torch.sqrt(shs / self.kl_threshold + 1e-8)
        fullstep = step_dir / (lm + 1e-8)

        pre_params = self._flat_params(self.policy.actor)
        cost_loss_np = cost_loss.detach().cpu().item()
        best_delta = torch.zeros_like(pre_params)
        best_improve = 0.0

        for i in range(self.ls_step):
            new_params = pre_params - self.fraction_coef * (self.line_search_fraction ** i) * fullstep
            self._set_flat_params(self.policy.actor, new_params)
            with torch.no_grad():
                new_cost_loss = self._cost_surrogate(obs_batch, actions_batch, old_log_probs, cost_advantages)
                self.policy.act(obs_batch)
                kl = self._kl_gaussian(
                    self.policy.action_mean, self.policy.action_std, old_mu, old_sigma
                ).mean()
            loss_improve = new_cost_loss.item() - cost_loss_np
            if kl.item() < self.kl_threshold and loss_improve < best_improve:
                best_improve = loss_improve
                best_delta = new_params - pre_params
                break

        self._set_flat_params(self.policy.actor, pre_params)
        return best_delta, best_improve

    # ------------------------------------------------------------------ main loop
    def update(self, current_costs: Optional[List[float]] = None) -> Dict[str, float]:
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_reward_improve = 0.0
        mean_cost_improve = 0.0
        projection_count = 0
        num_updates = 0

        trigger_cost = float(sum(current_costs)) if current_costs is not None else 0.0
        violated = trigger_cost > self.projection_threshold

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
             returns_cost_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            value_loss, cost_value_loss = self._update_critics(batch)

            pre_actor_params = self._flat_params(self.policy.actor)

            reward_delta, reward_improve = self._trpo_step_reward(
                obs_batch, actions_batch, old_actions_log_prob_batch, advantages_batch,
                old_mu_batch, old_sigma_batch,
            )

            if violated:
                cost_delta, cost_improve = self._trpo_step_cost(
                    obs_batch, actions_batch, old_actions_log_prob_batch, cost_advantages_batch,
                    old_mu_batch, old_sigma_batch,
                )
                final_delta = _pcgrad(reward_delta, cost_delta)
                projection_count += 1
            else:
                cost_improve = 0.0
                final_delta = reward_delta

            self._set_flat_params(self.policy.actor, pre_actor_params + final_delta)

            mean_value_loss += value_loss
            mean_cost_value_loss += cost_value_loss
            mean_reward_improve += reward_improve
            mean_cost_improve += cost_improve
            num_updates += 1

        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_cost_value_loss /= num_updates
            mean_reward_improve /= num_updates
            mean_cost_improve /= num_updates

        self.storage.clear()
        return {
            "value_function": mean_value_loss,
            "cost_function": mean_cost_value_loss,
            "surrogate": mean_reward_improve,
            "cost_surrogate": mean_cost_improve,
            "entropy": 0.0,
            "projection_fraction": projection_count / max(num_updates, 1),
            "trigger_cost": trigger_cost,
        }

    def get_penalty_info(self) -> Dict[str, Any]:
        return {
            "kappa": 0.0,
            "kappa_list": [0.0] * self.num_costs,
            "cost_limits": self.cost_limits,
            "projection_threshold": self.projection_threshold,
        }

    def _validate_cost_critic(self) -> None:
        if not hasattr(self.policy, "cost_critic"):
            raise ValueError("ActorCritic with a cost_critic is required for PCRPO")
        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, nn.Linear):
                last_layer = module
                break
        if last_layer is None:
            raise ValueError("Could not find linear layer in cost critic")
        if last_layer.out_features != self.num_costs:
            print(
                f"WARNING: Cost critic outputs {last_layer.out_features} values but PCRPO expects {self.num_costs}. "
                f"Configure ActorCritic with num_costs={self.num_costs}."
            )
