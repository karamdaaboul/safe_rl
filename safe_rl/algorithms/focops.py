from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

from safe_rl.modules import ActorCritic
from safe_rl.storage import RolloutStorageCMDP


class FOCOPS:
    """
    First Order Constrained Optimization in Policy Space (FOCOPS).

    Reference: Zhang et al., "First Order Constrained Optimization in Policy Space",
    NeurIPS 2020. https://arxiv.org/abs/2002.06506

    Single-phase first-order update with a KL-indicator-gated policy loss and a
    Lagrange multiplier ν updated by gradient ascent on the cost-violation.
    """

    policy: ActorCritic

    def __init__(
        self,
        policy,
        num_learning_epochs=10,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # Safe RL parameters
        cost_limits: Optional[List[float]] = None,
        cost_loss_coef=1.0,
        use_clipped_cost_loss=True,
        # FOCOPS-specific parameters
        focops_lam=1.5,
        focops_eta=0.02,
        nu_lr=0.01,
        nu_max=2.0,
        nu_init=0.0,
        c_gamma=0.99,
        c_gae_lam=0.95,
        l2_reg=0.0,
        # Backward compatibility
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        if cost_limits is not None:
            self.cost_limits = cost_limits
            self.num_costs = len(cost_limits)
        else:
            self.cost_limits = [0.0]
            self.num_costs = 1

        self.policy = policy
        self.policy.to(self.device)
        self._validate_and_fix_cost_critic()

        self.storage = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCMDP.Transition()

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

        self.focops_lam = focops_lam
        self.focops_eta = focops_eta
        self.nu_lr = nu_lr
        self.nu_max = nu_max
        self.nu_init = nu_init
        self.c_gamma = c_gamma
        self.c_gae_lam = c_gae_lam
        self.l2_reg = l2_reg

        self.nu = [float(nu_init)] * self.num_costs
        self.cost_history = [[] for _ in range(self.num_costs)]

        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None

        self.is_multi_gpu = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0

        self.symmetry = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
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

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs):
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

    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()

        if isinstance(costs, list):
            self.transition.costs = torch.stack([cost.clone() for cost in costs], dim=1)
        else:
            if costs.dim() == 1 and self.num_costs == 1:
                self.transition.costs = costs.unsqueeze(1).clone()
            else:
                self.transition.costs = costs.clone()

        self.transition.dones = dones

        if "time_outs" in infos:
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * timeout_mask, 1
            )
            timeout_mask_expanded = timeout_mask.expand(-1, self.num_costs)
            cost_bootstrap = self.gamma * self.transition.cost_values * timeout_mask_expanded
            self.transition.costs += cost_bootstrap

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs):
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values,
            self.c_gamma,
            self.c_gae_lam,
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def _kl_per_sample(self, old_mu, old_sigma, new_mu, new_sigma):
        """Per-sample analytic KL between two diagonal Gaussians, summed over action dim."""
        return torch.sum(
            torch.log(new_sigma / old_sigma + 1.0e-5)
            + (torch.square(old_sigma) + torch.square(old_mu - new_mu))
            / (2.0 * torch.square(new_sigma))
            - 0.5,
            dim=-1,
        )

    def _update_lagrange_multipliers(self, current_costs):
        for cost_idx in range(self.num_costs):
            if current_costs is not None:
                cost_violation = current_costs[cost_idx] - self.cost_limits[cost_idx]
                self.nu[cost_idx] = max(
                    0.0,
                    min(self.nu_max, self.nu[cost_idx] + self.nu_lr * cost_violation),
                )
            self.cost_history[cost_idx].append(
                current_costs[cost_idx] if current_costs is not None else 0.0
            )

    def update(self, current_costs=None):
        # Update Lagrange multipliers from the last rollout's episode-level costs.
        if current_costs is not None:
            self._update_lagrange_multipliers(current_costs)

        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_gated_fraction = 0.0
        num_updates = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            target_cost_values_batch,
            cost_advantages_batch,
            returns_cost_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                    for cost_idx in range(self.num_costs):
                        col = cost_advantages_batch[:, cost_idx]
                        cost_advantages_batch[:, cost_idx] = (col - col.mean()) / (col.std() + 1e-8)

            # Forward pass through current policy and critics.
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Per-sample KL (FOCOPS gates the loss per-sample with this).
            kl_per_sample = self._kl_per_sample(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
            kl_mean = kl_per_sample.mean()

            # Adaptive learning-rate schedule, matching CUP/PPO behaviour.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean_detached = kl_mean.detach()
                    if self.is_multi_gpu:
                        kl_mean_detached = kl_mean_detached.clone()
                        torch.distributed.all_reduce(kl_mean_detached, op=torch.distributed.ReduceOp.SUM)
                        kl_mean_detached /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean_detached > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean_detached < self.desired_kl / 2.0 and kl_mean_detached > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

            # Combined Lagrangian advantage: A_r - Σ_i ν_i * A_c_i.
            adv_r = torch.squeeze(advantages_batch)
            cost_term = torch.zeros_like(adv_r)
            for cost_idx in range(self.num_costs):
                cost_term = cost_term + self.nu[cost_idx] * cost_advantages_batch[:, cost_idx]
            adv_combined = adv_r - cost_term

            # FOCOPS first-order policy loss with KL-indicator gate.
            gate = (kl_per_sample.detach() <= self.focops_eta).float()
            policy_loss_per_sample = (
                kl_per_sample - (1.0 / self.focops_lam) * ratio * adv_combined
            ) * gate
            policy_loss = policy_loss_per_sample.mean() - self.entropy_coef * entropy_batch.mean()

            # Reward value loss (clipped, matching CUP).
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Cost value loss, clipped per constraint.
            cost_value_loss = torch.tensor(0.0, device=self.device)
            for cost_idx in range(self.num_costs):
                cv = cost_value_batch[:, cost_idx]
                cv_target = target_cost_values_batch[:, cost_idx]
                cv_returns = returns_cost_batch[:, cost_idx]
                if self.use_clipped_cost_loss:
                    cv_clipped = cv_target + (cv - cv_target).clamp(-self.clip_param, self.clip_param)
                    cv_losses = (cv - cv_returns).pow(2)
                    cv_losses_clipped = (cv_clipped - cv_returns).pow(2)
                    cost_value_loss = cost_value_loss + torch.max(cv_losses, cv_losses_clipped).mean()
                else:
                    cost_value_loss = cost_value_loss + (cv - cv_returns).pow(2).mean()

            total_loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                + self.cost_loss_coef * cost_value_loss
            )

            if self.l2_reg > 0.0:
                l2_term = torch.tensor(0.0, device=self.device)
                for param in self.policy.critic.parameters():
                    l2_term = l2_term + param.pow(2).sum()
                if hasattr(self.policy, "cost_critic"):
                    for param in self.policy.cost_critic.parameters():
                        l2_term = l2_term + param.pow(2).sum()
                total_loss = total_loss + self.l2_reg * l2_term

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += policy_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl += kl_mean.item()
            mean_gated_fraction += gate.mean().item()
            num_updates += 1

        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_cost_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            mean_kl /= num_updates
            mean_gated_fraction /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_mean": mean_kl,
            "kl_gated_fraction": mean_gated_fraction,
            "lagrange_nu_mean": sum(self.nu) / len(self.nu) if len(self.nu) > 0 else 0.0,
        }
        for i, nu_val in enumerate(self.nu):
            loss_dict[f"lagrange_nu_{i}"] = nu_val

        return loss_dict

    def get_penalty_info(self):
        """Compatibility with OnPolicyRunner's safe-RL logging path."""
        return {
            "kappa": sum(self.nu) / len(self.nu) if len(self.nu) > 0 else 0.0,
            "kappa_list": self.nu,
            "cost_limits": self.cost_limits,
            "nu_lr": self.nu_lr,
            "focops_eta": self.focops_eta,
            "focops_lam": self.focops_lam,
        }

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                offset += numel

    def _validate_and_fix_cost_critic(self):
        if not hasattr(self.policy, "cost_critic"):
            raise ValueError("ActorCritic must have a cost_critic attribute for FOCOPS algorithm")

        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, torch.nn.Linear):
                last_layer = module
                break

        if last_layer is None:
            raise ValueError("Could not find linear layer in cost critic")

        current_outputs = last_layer.out_features
        if current_outputs != self.num_costs:
            print(f"WARNING: Cost critic outputs {current_outputs} values but FOCOPS expects {self.num_costs}.")
            print("This mismatch will cause runtime errors. Please configure ActorCritic with num_costs parameter.")
            print(f"Example: ActorCritic(..., num_costs={self.num_costs})")
