from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from safe_rl.modules import ActorCriticCost
from safe_rl.storage import RolloutStorageCMDP
from safe_rl.utils import (
    conjugate_gradients,
    flatten_tensor_sequence,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_parameters,
    trainable_parameters,
)


class CPO:
    """Constrained Policy Optimization (Achiam et al. 2017) for safe RL.

    Trust-region update on the actor with a per-iteration constraint: picks the most-violated
    constraint, computes the Lagrangian step direction via CG + case analysis, then backtracks
    until reward improves, KL <= target_kl, and cost does not regress past the violation budget.
    Value and cost-value critics are updated with standard PPO-style clipped value loss.
    """

    policy: ActorCriticCost

    def __init__(
        self,
        policy: ActorCriticCost,
        num_learning_epochs: int = 10,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        cost_value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # CPO specifics
        cost_limits: Optional[List[float]] = None,
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 15,
        fvp_sample_freq: int = 1,
        # Compatibility with runner plumbing
        schedule: str = "fixed",
        desired_kl: Optional[float] = None,
        rnd_cfg: Optional[Dict[str, Any]] = None,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        multi_gpu_cfg: Optional[Dict[str, Any]] = None,
    ):
        if cost_limits is None or len(cost_limits) == 0:
            raise ValueError("CPO requires non-empty cost_limits.")
        if multi_gpu_cfg is not None:
            raise NotImplementedError("CPO does not support multi-GPU training.")

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        # Constraint bookkeeping
        self.cost_limits = list(cost_limits)
        self.num_costs = len(self.cost_limits)

        # Storage placeholder (populated in init_storage)
        self.storage: Optional[RolloutStorageCMDP] = None
        self.transition = RolloutStorageCMDP.Transition()

        # Hyperparameters shared with PPO-family
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # CPO trust-region parameters
        self.target_kl = float(target_kl)
        self.cg_iters = max(int(cg_iters), 1)
        self.cg_damping = float(cg_damping)
        self.backtrack_coeff = float(backtrack_coeff)
        self.max_backtracks = max(int(max_backtracks), 1)
        self.fvp_sample_freq = max(int(fvp_sample_freq), 1)

        # Parameter partitions
        self._policy_parameters = self._collect_policy_parameters()
        self._value_parameters = self._collect_value_parameters()

        # Value-side optimizer (Adam). Actor is updated via the trust-region step, no optimizer.
        # self.optimizer is pointed at the value optimizer so runner save/load keeps working.
        self.value_optimizer = optim.Adam(self._value_parameters, lr=learning_rate)
        self.optimizer = self.value_optimizer

        # RND / intrinsic-reward stubs expected by the runner.
        self.rnd = None
        self.intrinsic_rewards = None

        # Most recent step diagnostics (exposed through get_penalty_info / loss_dict).
        self._last_lambda: List[float] = [0.0] * self.num_costs
        self._last_active_idx: int = 0
        self._last_optim_case: int = -1

    def _collect_policy_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.policy.actor.parameters())
        std_param = getattr(self.policy, "std", None)
        log_std_param = getattr(self.policy, "log_std", None)
        if isinstance(std_param, torch.nn.Parameter):
            params.append(std_param)
        if isinstance(log_std_param, torch.nn.Parameter):
            params.append(log_std_param)
        return trainable_parameters(params)

    def _collect_value_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        params.extend(self.policy.critic.parameters())
        params.extend(self.policy.cost_critic.parameters())
        return trainable_parameters(params)

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

    def _format_costs_tensor(self, costs: torch.Tensor) -> torch.Tensor:
        if isinstance(costs, list):
            return torch.stack([c.clone() for c in costs], dim=1)
        if costs.dim() == 1:
            if self.num_costs == 1:
                return costs.unsqueeze(1).clone()
            return costs.unsqueeze(1).expand(-1, self.num_costs).clone()
        return costs.clone()

    def process_env_step(
        self,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        infos: Dict[str, Any],
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.costs = self._format_costs_tensor(costs)
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device).expand(-1, self.num_costs)
            if self.transition.cost_values is not None:
                self.transition.costs += self.gamma * self.transition.cost_values * timeout_mask

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def compute_cost_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values,
            self.gamma,
            self.lam,
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch,
        )

    # ---------- CPO actor update ----------

    def _make_distribution(self, obs_batch: torch.Tensor) -> Normal:
        self.policy.update_distribution(obs_batch)
        return self.policy.distribution

    def _loss_pi_reward(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_logp: torch.Tensor,
        reward_adv: torch.Tensor,
    ) -> torch.Tensor:
        self._make_distribution(obs_batch)
        logp = self.policy.get_actions_log_prob(actions_batch)
        ratio = torch.exp(logp - old_logp.squeeze(-1))
        return -(ratio * reward_adv).mean()

    def _loss_pi_cost(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_logp: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> torch.Tensor:
        self._make_distribution(obs_batch)
        logp = self.policy.get_actions_log_prob(actions_batch)
        ratio = torch.exp(logp - old_logp.squeeze(-1))
        return (ratio * cost_adv).mean()

    def _fisher_vector_product(self, obs_batch: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        sub_obs = obs_batch[:: self.fvp_sample_freq] if self.fvp_sample_freq > 1 else obs_batch
        with torch.enable_grad():
            self.policy.update_distribution(sub_obs)
            mean = self.policy.distribution.mean
            std = self.policy.distribution.stddev
            q_dist = Normal(mean, std)
            p_dist = Normal(mean.detach(), std.detach())
            kl = torch.distributions.kl_divergence(p_dist, q_dist).sum(dim=-1).mean()
            grads = torch.autograd.grad(
                kl, self._policy_parameters, create_graph=True, allow_unused=True
            )
            flat_grad_kl = flatten_tensor_sequence(grads, self._policy_parameters)
            kl_p = torch.dot(flat_grad_kl, vector)
            hvp = torch.autograd.grad(
                kl_p, self._policy_parameters, retain_graph=False, allow_unused=True
            )
            flat_hvp = flatten_tensor_sequence(hvp, self._policy_parameters)
        return flat_hvp + self.cg_damping * vector

    def _active_constraint_index(self, current_costs: Optional[List[float]]) -> Tuple[int, float]:
        if current_costs is None:
            mean_costs = self.storage.get_mean_episode_costs().detach().cpu().tolist()
        else:
            mean_costs = [float(c.item()) if torch.is_tensor(c) else float(c) for c in current_costs]
        violations = [mean_costs[k] - self.cost_limits[k] for k in range(self.num_costs)]
        active_idx = int(max(range(self.num_costs), key=lambda k: violations[k]))
        return active_idx, float(violations[active_idx])

    def _determine_case(
        self,
        b_grads: torch.Tensor,
        c_hat: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=self.device)
        if b_grads.dot(b_grads) <= 1e-6 and c_hat < 0:
            return 4, zero, zero

        A = q - r**2 / (s + 1e-8)
        B = 2 * self.target_kl - c_hat**2 / (s + 1e-8)
        if c_hat < 0 and B < 0:
            optim_case = 3
        elif c_hat < 0 <= B:
            optim_case = 2
        elif c_hat >= 0 and B >= 0:
            optim_case = 1
        else:
            optim_case = 0
        return optim_case, A, B

    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        c_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros((), device=self.device)
            lambda_star = 1.0 / (alpha + 1e-8)
            step = alpha * x
        elif optim_case in (1, 2):
            lambda_a = torch.sqrt(torch.clamp_min(A / (B + 1e-8), 0.0))
            lambda_b = torch.sqrt(torch.clamp_min(q / (2 * self.target_kl + 1e-8), 0.0))
            inf_t = torch.tensor(float("inf"), device=self.device)
            zero_t = torch.zeros((), device=self.device)
            boundary = r / (c_hat + 1e-8)
            if c_hat < 0:
                lambda_a_star = torch.clamp(lambda_a, min=zero_t, max=boundary)
                lambda_b_star = torch.clamp(lambda_b, min=boundary, max=inf_t)
            else:
                lambda_a_star = torch.clamp(lambda_a, min=boundary, max=inf_t)
                lambda_b_star = torch.clamp(lambda_b, min=zero_t, max=boundary)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * c_hat / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self.target_kl * lam)

            lambda_star = lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            nu_star = torch.clamp(lambda_star * c_hat - r, min=0.0) / (s + 1e-8)
            step = (x - nu_star * p) / (lambda_star + 1e-8)
        else:  # case 0: infeasible, take pure recovery step along cost gradient
            lambda_star = torch.zeros((), device=self.device)
            nu_star = torch.sqrt(2 * self.target_kl / (s + 1e-8))
            step = -nu_star * p
        return step, lambda_star, nu_star

    def _cpo_line_search(
        self,
        step_direction: torch.Tensor,
        old_dist: Normal,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_logp: torch.Tensor,
        reward_adv: torch.Tensor,
        cost_adv: torch.Tensor,
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        c_hat: float,
        optim_case: int,
    ) -> Tuple[torch.Tensor, int, float]:
        step_frac = 1.0
        theta_old = get_flat_params_from(self._policy_parameters)
        final_kl = 0.0
        accepted = 0

        for step_idx in range(self.max_backtracks):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_parameters(self._policy_parameters, new_theta)

            with torch.no_grad():
                loss_reward = self._loss_pi_reward(obs_batch, actions_batch, old_logp, reward_adv)
                loss_cost = self._loss_pi_cost(obs_batch, actions_batch, old_logp, cost_adv)
                self.policy.update_distribution(obs_batch)
                new_dist = Normal(self.policy.distribution.mean, self.policy.distribution.stddev)
                kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean()

            reward_improve = (loss_reward_before - loss_reward).item()
            cost_diff = (loss_cost - loss_cost_before).item()
            kl_val = kl.item()

            if not torch.isfinite(loss_reward) or not torch.isfinite(loss_cost) or not torch.isfinite(kl):
                pass
            elif optim_case > 1 and reward_improve < 0:
                pass
            elif cost_diff > max(-c_hat, 0.0):
                pass
            elif kl_val > self.target_kl:
                pass
            else:
                final_kl = kl_val
                accepted = step_idx + 1
                break
            step_frac *= self.backtrack_coeff
        else:
            # No acceptable step — revert to theta_old
            step_frac = 0.0
            step_direction = torch.zeros_like(step_direction)

        set_param_values_to_parameters(self._policy_parameters, theta_old)
        return step_frac * step_direction, accepted, final_kl

    def _update_actor(self, current_costs: Optional[List[float]]) -> Dict[str, float]:
        # Full batch in one shot
        batch = next(self.storage.mini_batch_generator(1, 1))
        (
            obs_batch,
            _critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            _returns_batch,
            _target_cost_values_batch,
            cost_advantages_batch,
            _returns_cost_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            _rnd_state_batch,
        ) = batch

        reward_adv = advantages_batch.squeeze(-1)
        active_idx, violation = self._active_constraint_index(current_costs)
        if cost_advantages_batch.dim() == 2:
            cost_adv_active = cost_advantages_batch[:, active_idx]
        else:
            cost_adv_active = cost_advantages_batch.squeeze(-1)
        c_hat_tensor = torch.tensor(violation, device=self.device, dtype=torch.float32)
        old_dist = Normal(old_mu_batch, old_sigma_batch)

        # Reward-loss gradient and CG direction x = H^-1 g
        self.policy.zero_grad(set_to_none=True)
        loss_reward = self._loss_pi_reward(obs_batch, actions_batch, old_actions_log_prob_batch, reward_adv)
        loss_reward_before = loss_reward.detach()
        loss_reward.backward()
        grads = -get_flat_gradients_from(self._policy_parameters)

        fvp = lambda vec: self._fisher_vector_product(obs_batch, vec)  # noqa: E731
        x = conjugate_gradients(fvp, grads, num_steps=self.cg_iters)
        xHx = torch.dot(x, fvp(x))
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))

        # Cost-loss gradient and CG direction p = H^-1 b
        self.policy.zero_grad(set_to_none=True)
        loss_cost = self._loss_pi_cost(obs_batch, actions_batch, old_actions_log_prob_batch, cost_adv_active)
        loss_cost_before = loss_cost.detach()
        loss_cost.backward()
        b_grads = get_flat_gradients_from(self._policy_parameters)
        p = conjugate_gradients(fvp, b_grads, num_steps=self.cg_iters)

        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        optim_case, A, B = self._determine_case(b_grads, c_hat_tensor, q, r, s)
        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case, xHx=xHx, x=x, A=A, B=B, q=q, p=p, r=r, s=s, c_hat=c_hat_tensor
        )

        accepted_direction, accepted, final_kl = self._cpo_line_search(
            step_direction=step_direction,
            old_dist=old_dist,
            obs_batch=obs_batch,
            actions_batch=actions_batch,
            old_logp=old_actions_log_prob_batch,
            reward_adv=reward_adv,
            cost_adv=cost_adv_active,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            c_hat=float(c_hat_tensor.item()),
            optim_case=optim_case,
        )
        theta_old = get_flat_params_from(self._policy_parameters)
        theta_new = theta_old + accepted_direction
        set_param_values_to_parameters(self._policy_parameters, theta_new)

        with torch.no_grad():
            final_reward_loss = self._loss_pi_reward(
                obs_batch, actions_batch, old_actions_log_prob_batch, reward_adv
            )
            final_cost_loss = self._loss_pi_cost(
                obs_batch, actions_batch, old_actions_log_prob_batch, cost_adv_active
            )
            self._make_distribution(obs_batch)
            entropy = self.policy.entropy.mean().item()

        # Record diagnostics for get_penalty_info() and the returned loss dict
        self._last_lambda = [0.0] * self.num_costs
        self._last_lambda[active_idx] = float(lambda_star.item()) if lambda_star.numel() == 1 else float(lambda_star)
        self._last_active_idx = active_idx
        self._last_optim_case = optim_case

        return {
            "surrogate": float(final_reward_loss.item()),
            "cost_surrogate": float(final_cost_loss.item()),
            "entropy": entropy,
            "kl": final_kl,
            "acceptance_step": float(accepted),
            "alpha": float(alpha.item()),
            "lambda_star": float(lambda_star.item()) if lambda_star.numel() == 1 else float(lambda_star),
            "nu_star": float(nu_star.item()) if nu_star.numel() == 1 else float(nu_star),
            "optim_case": float(optim_case),
            "active_constraint_index": float(active_idx),
            "cost_violation": float(c_hat_tensor.item()),
        }

    # ---------- Value-function update (reward + cost critics) ----------

    def _update_value_functions(self) -> Tuple[float, float]:
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0

        if self.policy.is_recurrent:
            raise NotImplementedError("CPO currently supports feed-forward policies only.")
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            _obs_batch,
            critic_obs_batch,
            _actions_batch,
            target_values_batch,
            _advantages_batch,
            returns_batch,
            target_cost_values_batch,
            _cost_advantages_batch,
            returns_cost_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            _rnd_state_batch,
        ) in generator:

            value_batch = self.policy.evaluate(critic_obs_batch)
            cost_value_batch = self.policy.evaluate_cost(critic_obs_batch)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                cost_clipped = target_cost_values_batch + (
                    cost_value_batch - target_cost_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                cost_losses = (cost_value_batch - returns_cost_batch).pow(2)
                cost_losses_clipped = (cost_clipped - returns_cost_batch).pow(2)
                cost_value_loss = torch.max(cost_losses, cost_losses_clipped).mean(dim=0).sum()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                cost_value_loss = (returns_cost_batch - cost_value_batch).pow(2).mean(dim=0).sum()

            loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss

            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._value_parameters, self.max_grad_norm)
            self.value_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return mean_value_loss / num_updates, mean_cost_value_loss / num_updates

    def update(self, current_costs: Optional[List[float]] = None) -> Dict[str, float]:
        actor_metrics = self._update_actor(current_costs)
        mean_value_loss, mean_cost_value_loss = self._update_value_functions()
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": actor_metrics["surrogate"],
            "cost_surrogate": actor_metrics["cost_surrogate"],
            "entropy": actor_metrics["entropy"],
            "kl": actor_metrics["kl"],
            "acceptance_step": actor_metrics["acceptance_step"],
            "lambda_star": actor_metrics["lambda_star"],
            "nu_star": actor_metrics["nu_star"],
            "optim_case": actor_metrics["optim_case"],
            "active_constraint_index": actor_metrics["active_constraint_index"],
            "cost_violation": actor_metrics["cost_violation"],
        }
        return loss_dict

    def get_penalty_info(self) -> Dict[str, Any]:
        """Logger-compatible summary (mirrors P3O's ``get_penalty_info`` keys)."""
        return {
            "kappa": float(sum(self._last_lambda) / max(len(self._last_lambda), 1)),
            "cost_limit": float(sum(self.cost_limits) / max(len(self.cost_limits), 1)),
            "kappa_list": list(self._last_lambda),
            "cost_limits": list(self.cost_limits),
            "active_constraint_index": int(self._last_active_idx),
            "optim_case": int(self._last_optim_case),
        }
