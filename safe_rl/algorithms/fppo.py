from __future__ import annotations

import collections
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from safe_rl.modules import ActorCritic
from safe_rl.storage import RolloutStorageCMDP
from safe_rl.utils.torch_utils import (
    get_flat_params_from,
    set_param_values_to_parameters,
)


class FPPO:
    """
    Feasibility-guided PPO (FPPO) for safe reinforcement learning.

    Uses a predictor-corrector approach:
    - Predictor: Standard PPO update with per-batch KL monitoring and a hard abort threshold
    - Corrector: Gradient projection onto the feasibility set — solves a small QP over
      constraint gradients then applies a backtracking line search step

    Optionally applies an adaptive constraint curriculum that starts from relaxed limits
    and progressively tightens them toward cost_limits as the agent satisfies them.
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
        cost_loss_coef=1.0,
        use_clipped_cost_loss=True,
        entropy_coef=0.0,
        learning_rate=3e-4,
        max_grad_norm=40.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        # Predictor phase
        delta_kl=0.01,
        predictor_desired_kl=0.01,
        predictor_kl_hard_limit=0.05,
        # Corrector phase
        step_size=1.0,
        backtrack_coeff=0.5,
        max_backtracks=10,
        projection_eps=1e-8,
        # Constraints
        cost_limits: Optional[List[float]] = None,
        # Curriculum
        adaptive_constraint_curriculum=True,
        constraint_limits_start: Optional[List[float]] = None,
        constraint_curriculum_alpha=0.8,
        constraint_curriculum_shrink=0.97,
        constraint_curriculum_ema_decay=0.95,
        constraint_curriculum_check_interval=20,
        # Runner compatibility stubs
        device="cpu",
        rnd_cfg=None,
        symmetry_cfg=None,
        multi_gpu_cfg=None,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Constraint setup
        if cost_limits is not None:
            self.cost_limits = list(cost_limits)
            self.num_costs = len(cost_limits)
        else:
            self.cost_limits = [0.0]
            self.num_costs = 1

        # Policy
        self.policy = policy
        self.policy.to(self.device)
        self._validate_and_fix_cost_critic()

        self.storage = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCMDP.Transition()

        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_loss_coef = cost_loss_coef
        self.use_clipped_cost_loss = use_clipped_cost_loss
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Predictor hyperparameters
        self.delta_kl = delta_kl
        self.predictor_desired_kl = predictor_desired_kl
        self.predictor_kl_hard_limit = predictor_kl_hard_limit

        # Corrector hyperparameters
        self.step_size = step_size
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks
        self.projection_eps = projection_eps

        # Curriculum hyperparameters
        self.adaptive_constraint_curriculum = adaptive_constraint_curriculum
        self.constraint_curriculum_alpha = constraint_curriculum_alpha
        self.constraint_curriculum_shrink = constraint_curriculum_shrink
        self.constraint_curriculum_ema_decay = constraint_curriculum_ema_decay
        self.constraint_curriculum_check_interval = constraint_curriculum_check_interval

        # Curriculum state — start from relaxed limits, tighten toward cost_limits
        if constraint_limits_start is not None:
            self._effective_limits = list(constraint_limits_start)
        else:
            self._effective_limits = [
                2.0 * l if l > 0.0 else 1.0 for l in self.cost_limits
            ]
        self._cost_ema = [0.0] * self.num_costs
        self._iteration_counter = 0

        # Corrector state
        self._corrector_step_size = step_size
        self._recent_acceptances: collections.deque = collections.deque(maxlen=20)

        # Runner compatibility stubs (runner checks these attributes)
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None
        self.symmetry = None
        self.is_multi_gpu = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0

    # ------------------------------------------------------------------
    # Interface methods — same signatures as CUP / P3O
    # ------------------------------------------------------------------

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
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
            self.transition.costs = torch.stack([c.clone() for c in costs], dim=1)
        elif costs.dim() == 1 and self.num_costs == 1:
            self.transition.costs = costs.unsqueeze(1).clone()
        else:
            self.transition.costs = costs.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * timeout_mask, 1)
            timeout_mask_cost = timeout_mask.expand(-1, self.num_costs)
            self.transition.costs = self.transition.costs + self.gamma * self.transition.cost_values * timeout_mask_cost
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
            self.gamma,
            self.lam,
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self, current_costs=None):
        self._iteration_counter += 1

        # Snapshot θ_anchor before predictor — defines the KL trust region for the corrector
        self._theta_anchor = get_flat_params_from(self.policy.actor.parameters())

        # Phase 1: predictor — PPO update with KL abort
        predictor_dict = self._predictor_phase()

        # Snapshot θ_predictor — starting point that gets projected onto the safe set
        self._theta_predictor = get_flat_params_from(self.policy.actor.parameters())

        # Update cost EMA for curriculum decisions
        self._update_cost_ema(current_costs)

        # Tighten curriculum limits when constraints are consistently satisfied
        if (
            self.adaptive_constraint_curriculum
            and self._iteration_counter % self.constraint_curriculum_check_interval == 0
        ):
            self._maybe_update_curriculum()

        # Phase 2: corrector — gradient projection onto feasibility set
        corrector_dict = self._corrector_phase(current_costs)

        self.storage.clear()

        loss_dict = {**predictor_dict, **corrector_dict}
        for i, lim in enumerate(self._effective_limits):
            loss_dict[f"curriculum_limit_{i}"] = lim

        return loss_dict

    def get_penalty_info(self):
        return {
            "kappa": sum(self._effective_limits) / len(self._effective_limits),
            "kappa_list": list(self._effective_limits),
            "cost_limits": self.cost_limits,
        }

    # ------------------------------------------------------------------
    # Phase 1: Predictor (PPO + cost critic, with per-batch KL abort)
    # ------------------------------------------------------------------

    def _predictor_phase(self):
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        max_kl = 0.0
        mean_kl_post = 0.0
        max_kl_post = 0.0
        mean_grad_norm = 0.0
        max_grad_norm_seen = 0.0
        num_updates = 0
        aborted = 0.0

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

            # Forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL divergence: monitor and abort if hard limit exceeded
            kl = torch.sum(
                torch.log(sigma_batch / (old_sigma_batch + 1e-5) + 1e-5)
                + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2)) / (2.0 * sigma_batch.pow(2))
                - 0.5,
                dim=-1,
            )
            kl_mean = kl.mean().item()
            mean_kl += kl_mean
            max_kl = max(max_kl, kl_mean)

            if kl_mean > self.predictor_kl_hard_limit:
                aborted = 1.0
                break

            # Adaptive learning rate
            if self.schedule == "adaptive" and self.gpu_global_rank == 0:
                if kl_mean > self.predictor_desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.predictor_desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
            surrogate = -advantages_batch.squeeze() * ratio
            surrogate_clipped = -advantages_batch.squeeze() * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Cost critic loss (trains cost critic to improve future cost GAE estimates)
            cost_value_loss = torch.tensor(0.0, device=self.device)
            if target_cost_values_batch is not None and returns_cost_batch is not None:
                cost_value_batch = self.policy.evaluate_cost(critic_obs_batch)
                if self.use_clipped_cost_loss:
                    cost_value_clipped = target_cost_values_batch + (
                        cost_value_batch - target_cost_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    cost_value_loss = torch.max(
                        (cost_value_batch - returns_cost_batch).pow(2),
                        (cost_value_clipped - returns_cost_batch).pow(2),
                    ).mean()
                else:
                    cost_value_loss = (returns_cost_batch - cost_value_batch).pow(2).mean()

            total_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Post-step KL vs the rollout policy — captures actual policy displacement per update.
            with torch.no_grad():
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                new_mu = self.policy.action_mean
                new_sigma = self.policy.action_std
                kl_post = torch.sum(
                    torch.log(new_sigma / (old_sigma_batch + 1e-5) + 1e-5)
                    + (old_sigma_batch.pow(2) + (old_mu_batch - new_mu).pow(2)) / (2.0 * new_sigma.pow(2))
                    - 0.5,
                    dim=-1,
                ).mean().item()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_post += kl_post
            max_kl_post = max(max_kl_post, kl_post)
            gn = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)
            mean_grad_norm += gn
            max_grad_norm_seen = max(max_grad_norm_seen, gn)
            num_updates += 1

        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_cost_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            mean_kl /= num_updates
            mean_kl_post /= num_updates
            mean_grad_norm /= num_updates

        return {
            "value_function": mean_value_loss,
            "cost_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "predictor_kl": mean_kl,
            "predictor_kl_max": max_kl,
            "predictor_kl_post": mean_kl_post,
            "predictor_kl_post_max": max_kl_post,
            "predictor_grad_norm": mean_grad_norm,
            "predictor_grad_norm_max": max_grad_norm_seen,
            "predictor_lr": self.learning_rate,
            "predictor_num_updates": float(num_updates),
            "predictor_aborted": aborted,
        }

    # ------------------------------------------------------------------
    # Phase 2: Corrector (first-order active-set projection + KL backtracking)
    # ------------------------------------------------------------------

    def _corrector_phase(self, current_costs):
        actor_params = list(self.policy.actor.parameters())

        # Full rollout batch via indexed generator — strips the runner's inference-mode flag.
        batch = next(self.storage.mini_batch_generator(1, 1))
        (obs_b, _, actions_b, _, _, _, _, cost_adv_b, _, old_logp_b,
         old_mu_b, old_sigma_b, _, _, _) = batch

        # Per-constraint cost-surrogate gradients at θ_predictor, stacked as columns.
        a_mat, any_nonzero = self._compute_constraint_gradients(
            actor_params, obs_b, actions_b, old_logp_b, cost_adv_b
        )
        if not any_nonzero:
            return self._corrector_skip_dict(active_count=0)

        # Active-set soft projection: identify which constraints the predictor step violates.
        delta = self._theta_predictor - self._theta_anchor
        b_budget = self._constraint_budget(current_costs)
        violation = a_mat.t() @ delta - b_budget                  # [num_costs]
        active = violation > 0.0

        if not bool(active.any()):
            return self._corrector_skip_dict(active_count=0)

        a_active = a_mat[:, active]
        v_active = violation[active]
        q_mat = a_active.t() @ a_active
        lamb = self._solve_nonnegative_qp(q_mat, v_active)
        theta_projected = self._theta_predictor - a_active @ lamb

        # Per-coordinate RMS cap: ‖limited_delta‖ ≤ corrector_step_size · √|θ|.
        # Matches the reference's `step_budget` guard against small-‖A‖ blowup.
        projected_delta = theta_projected - self._theta_anchor
        projected_norm = projected_delta.norm()
        step_budget = self._corrector_step_size * math.sqrt(max(projected_delta.numel(), 1))
        if not torch.isfinite(projected_norm) or projected_norm.item() <= 0.0:
            step_ratio = 0.0
            limited_delta = torch.zeros_like(projected_delta)
        else:
            step_ratio = min(1.0, step_budget / max(projected_norm.item(), 1e-12))
            limited_delta = projected_delta * step_ratio

        # KL-only backtracking line search between θ_anchor and θ_anchor + limited_delta.
        accepted, n_back, eta, final_kl = self._corrector_line_search(
            actor_params, limited_delta, obs_b, old_mu_b, old_sigma_b,
        )
        self._recent_acceptances.append(float(accepted))

        # Adapt base step size from rolling acceptance rate (target ~70%).
        if len(self._recent_acceptances) >= 5:
            rate = sum(self._recent_acceptances) / len(self._recent_acceptances)
            if rate > 0.7:
                self._corrector_step_size = min(self._corrector_step_size * 1.05, self.step_size * 10.0)
            elif rate < 0.4:
                self._corrector_step_size = max(self._corrector_step_size * 0.95, self.step_size * 0.001)

        return {
            "corrector_step_size": self._corrector_step_size,
            "corrector_accepted": float(accepted),
            "corrector_backtracks": float(n_back),
            "corrector_active_constraints": float(int(active.sum().item())),
            "corrector_eta": float(eta),
            "corrector_kl": float(final_kl),
            "corrector_step_ratio": float(step_ratio),
        }

    def _corrector_skip_dict(self, active_count=0):
        return {
            "corrector_step_size": self._corrector_step_size,
            "corrector_accepted": 1.0,
            "corrector_backtracks": 0.0,
            "corrector_active_constraints": float(active_count),
            "corrector_eta": 1.0,
            "corrector_kl": 0.0,
            "corrector_step_ratio": 1.0,
        }

    def _constraint_budget(self, current_costs):
        """Per-constraint slack b_i = d_tight_i - J_cost_i (positive means feasible)."""
        budget = torch.zeros(self.num_costs, device=self.device)
        for i in range(self.num_costs):
            d_tight = self._effective_limits[i]
            if current_costs is not None:
                ci = current_costs[i]
                j_cost = float(ci) if hasattr(ci, "__float__") else ci.item()
            else:
                j_cost = self._cost_ema[i]
            budget[i] = d_tight - j_cost
        return budget

    def _compute_constraint_gradients(self, actor_params, obs, actions, old_log_prob, cost_adv):
        """Stack flat gradients ∇_θ E[ratio · A_cost_i] as columns of [|θ|, num_costs]."""
        old_logp_flat = old_log_prob.squeeze(-1).detach()
        cols = []
        any_nonzero = False
        for i in range(self.num_costs):
            self.policy.update_distribution(obs)
            new_log_prob = self.policy.get_actions_log_prob(actions)
            ratio = torch.exp(new_log_prob - old_logp_flat)
            adv_col = cost_adv[:, i] if cost_adv.dim() > 1 else cost_adv.squeeze(-1)
            cost_surrogate = (ratio * adv_col.detach()).mean()

            grads = torch.autograd.grad(
                cost_surrogate,
                actor_params,
                retain_graph=(i < self.num_costs - 1),
                allow_unused=True,
            )
            flat = torch.cat(
                [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=self.device)
                 for g, p in zip(grads, actor_params)]
            ).detach()
            if flat.abs().max().item() > 0.0:
                any_nonzero = True
            cols.append(flat)

        return torch.stack(cols, dim=1), any_nonzero

    def _solve_nonnegative_qp(self, q_mat, v_vec, max_iters=50, tol=1e-6):
        """Proximal-gradient solver for min_{λ ≥ 0} ½ λᵀ Q λ − vᵀ λ (first-order, on-device)."""
        q_reg = q_mat + self.projection_eps * torch.eye(
            q_mat.shape[0], device=q_mat.device, dtype=q_mat.dtype
        )
        max_eig = torch.linalg.eigvalsh(q_reg).max()
        step = 1.0 / (max_eig + self.projection_eps)
        lamb = torch.zeros_like(v_vec)
        for _ in range(max_iters):
            grad = q_reg @ lamb - v_vec
            new_lamb = torch.clamp(lamb - step * grad, min=0.0)
            if (new_lamb - lamb).abs().max().item() <= tol:
                lamb = new_lamb
                break
            lamb = new_lamb
        return lamb

    def _corrector_line_search(self, actor_params, limited_delta, obs, old_mu, old_sigma):
        """Backtrack η from 1.0 toward 0 until KL(old || candidate) ≤ hard limit."""
        kl_limit = self.predictor_kl_hard_limit
        eta = 1.0
        final_kl = float("inf")
        accepted = False
        n_back = 0
        for _ in range(self.max_backtracks):
            theta_candidate = self._theta_anchor + eta * limited_delta
            set_param_values_to_parameters(actor_params, theta_candidate)
            with torch.no_grad():
                self.policy.update_distribution(obs)
                new_mu = self.policy.action_mean
                new_sigma = self.policy.action_std
                kl = torch.sum(
                    torch.log(new_sigma / (old_sigma + 1e-5) + 1e-5)
                    + (old_sigma.pow(2) + (old_mu - new_mu).pow(2)) / (2.0 * new_sigma.pow(2))
                    - 0.5,
                    dim=-1,
                ).mean()
            final_kl = kl.item() if torch.isfinite(kl) else float("inf")
            if final_kl <= kl_limit:
                accepted = True
                break
            eta *= self.backtrack_coeff
            n_back += 1

        if not accepted:
            # Reference-faithful: revert to θ_anchor on line-search failure.
            set_param_values_to_parameters(actor_params, self._theta_anchor)

        return accepted, n_back, eta, final_kl

    # ------------------------------------------------------------------
    # Curriculum helpers
    # ------------------------------------------------------------------

    def _update_cost_ema(self, current_costs):
        if current_costs is None:
            return
        for i in range(self.num_costs):
            val = float(current_costs[i]) if hasattr(current_costs[i], "__float__") else current_costs[i].item()
            self._cost_ema[i] = (
                self.constraint_curriculum_ema_decay * self._cost_ema[i]
                + (1.0 - self.constraint_curriculum_ema_decay) * val
            )

    def _maybe_update_curriculum(self):
        for i in range(self.num_costs):
            if self._cost_ema[i] < self.constraint_curriculum_alpha * self._effective_limits[i]:
                new_limit = self._effective_limits[i] * self.constraint_curriculum_shrink
                self._effective_limits[i] = max(new_limit, self.cost_limits[i])

    # ------------------------------------------------------------------
    # Validation and multi-GPU stubs
    # ------------------------------------------------------------------

    def _validate_and_fix_cost_critic(self):
        if not hasattr(self.policy, "cost_critic"):
            raise ValueError("Policy must have a cost_critic attribute for FPPO.")
        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, torch.nn.Linear):
                last_layer = module
                break
        if last_layer is None:
            raise ValueError("Could not find a linear layer in cost critic.")
        if last_layer.out_features != self.num_costs:
            print(
                f"WARNING: Cost critic outputs {last_layer.out_features} values but FPPO expects "
                f"{self.num_costs}. Configure ActorCritic with num_costs={self.num_costs}."
            )

    def reduce_parameters(self):
        grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self.policy.parameters():
            if p.grad is not None:
                numel = p.grad.numel()
                p.grad.copy_(all_grads[offset: offset + numel].view_as(p.grad))
                offset += numel
