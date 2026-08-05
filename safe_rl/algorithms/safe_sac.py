from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_rl.algorithms.sac import SAC
from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic
from safe_rl.utils.torch_utils import resolve_optimizer


class SafeSAC(SAC):
    """Safe Soft Actor-Critic with PID Lagrangian constraint handling.

    Extends :class:`SAC` and inherits all non-safety mechanics verbatim — the
    replay buffer (including in-buffer n-step returns), the twin reward-critic
    update, the bootstrap channel, entropy auto-tuning and the optimizer factory
    — so the reward path cannot drift from plain SAC / MPO. On top of that it
    adds only the safety layer:

    - Cost Q-network(s) for constraint estimation (costs are n-step aggregated
      in the buffer exactly like rewards)
    - PID controller for adaptive Lagrangian multiplier updates
    - Safety-augmented actor objective: maximize ``Q_r - α·log π - λ·Q_c``

    References:
    - SAC: https://arxiv.org/abs/1801.01290
    - Safe RL with PID Lagrangian: https://arxiv.org/abs/2007.03964
    """

    policy: SafeSACActorCritic
    """The actor critic module."""

    _extra_critic_keys = ("cost_critic",)

    def __init__(
        self,
        policy: SafeSACActorCritic,
        cost_critic_lr: float = 3e-4,
        cost_limits: list[float] | None = None,
        lagrangian_pid: tuple[float, float, float] = (0.1, 0.01, 0.01),  # (Kp, Ki, Kd)
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float = 0.95,
        pid_d_delay: int = 10,
        lambda_init: list[float] | None = None,
        lambda_max: float = 100.0,
        sum_norm: bool = True,
        diff_norm: bool = False,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize Safe SAC algorithm.

        All non-safety arguments (``actor_lr``, ``critic_lr``, ``alpha_lr``,
        ``gamma``, ``tau``, ``alpha``, ``auto_entropy_tuning``, ``target_entropy``,
        ``batch_size``, ``num_updates_per_step``, ``policy_frequency``,
        ``max_grad_norm``, ``n_step``, ``multi_gpu_cfg``, ...) are forwarded to
        :class:`SAC` via ``**kwargs``.

        Args:
            policy: SafeSACActorCritic module.
            cost_critic_lr: Learning rate for the cost critics.
            cost_limits: Cost limits for each constraint (required).
            lagrangian_pid: PID gains (Kp, Ki, Kd) for Lagrangian multiplier updates.
            pid_delta_p_ema_alpha: EMA alpha for proportional term smoothing.
            pid_delta_d_ema_alpha: EMA alpha for derivative term smoothing.
            pid_d_delay: Delay steps for derivative calculation.
            lambda_init: Initial Lagrangian multipliers.
            lambda_max: Maximum Lagrangian multiplier value.
            sum_norm: Apply sum normalization for lambda.
            diff_norm: Apply diff normalization (clips to [0, 1]).
            optimizer: Optimizer name ("adam" or "adamw"), also used for the cost critics.
            weight_decay: Weight decay for the actor/critic/cost-critic optimizers.
            betas: Adam/AdamW beta coefficients.
            device: Device to run on.
        """
        if cost_limits is None:
            raise ValueError("cost_limits must be provided for Safe SAC")

        super().__init__(
            policy, optimizer=optimizer, weight_decay=weight_decay, betas=betas, device=device, **kwargs
        )

        self.cost_limits = cost_limits
        self.num_costs = len(cost_limits)
        # Hazard-stratified replay diagnostics, refreshed each cost-critic update.
        self._last_replay_info: dict[str, float] = {}
        if hasattr(policy, 'num_costs') and policy.num_costs != self.num_costs:
            print(f"WARNING: Policy num_costs ({policy.num_costs}) doesn't match cost_limits ({self.num_costs})")

        self.kp, self.ki, self.kd = lagrangian_pid
        self.lambda_max = lambda_max
        self.sum_norm = sum_norm
        self.diff_norm = diff_norm
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self.pid_d_delay = pid_d_delay
        # Anti-windup: cap the integral term at 50% of lambda_max.
        self.pid_i_max = lambda_max * 0.5

        if lambda_init is None:
            self.lambdas = [0.001] * self.num_costs
        elif len(lambda_init) == 1 and self.num_costs > 1:
            self.lambdas = list(lambda_init) * self.num_costs
        else:
            self.lambdas = list(lambda_init)

        self.pid_i = [self.lambdas[i] for i in range(self.num_costs)]
        self.delta_p = [0.0] * self.num_costs
        self.cost_ema = [0.0] * self.num_costs
        self.cost_delay_queue: list[deque] = [
            deque([0.0], maxlen=pid_d_delay) for _ in range(self.num_costs)
        ]

        print(f"Safe SAC initialized with {self.num_costs} cost constraints")
        print(f"PID gains: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        print(f"Cost limits: {self.cost_limits}")
        print(f"Initial lambdas: {self.lambdas}")
        print(f"Lambda max: {self.lambda_max}, sum_norm: {self.sum_norm}, diff_norm: {self.diff_norm}")

        optimizer_cls = resolve_optimizer(optimizer)
        cost_critic_params = []
        for critic in policy.cost_critics:
            cost_critic_params.extend(list(critic.parameters()))
        self.cost_critic_optimizer = optimizer_cls(
            cost_critic_params, lr=cost_critic_lr, weight_decay=weight_decay, betas=betas
        )

    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        cost: torch.Tensor | None = None,
        critic_obs: torch.Tensor | None = None,
        next_critic_obs: torch.Tensor | None = None,
        bootstrap: torch.Tensor | None = None,
    ) -> None:
        """Store a transition in the replay buffer, including its cost.

        Args:
            obs: Current observations.
            action: Actions taken.
            reward: Rewards received.
            done: Done flags (truthful episode end when the bootstrap channel is used).
            next_obs: Next observations.
            cost: Costs received; zeros are stored when absent.
            critic_obs: Optional critic observations.
            next_critic_obs: Optional next critic observations.
            bootstrap: Optional timeout flag (1 = truncation) for the bootstrap channel.
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage() first.")

        extras = {}
        if cost is not None:
            cost = self._format_costs_tensor(cost)
        else:
            cost = torch.zeros(obs.shape[0], self.num_costs, device=self.device)
        extras["costs"] = cost
        if critic_obs is not None:
            extras["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            extras["next_critic_observations"] = next_critic_obs
        if bootstrap is not None:
            extras["bootstrap"] = bootstrap.view(-1, 1) if bootstrap.dim() == 1 else bootstrap
        self.storage.add(obs, action, reward, done, next_obs, **extras)

    def _format_costs_tensor(self, costs: torch.Tensor) -> torch.Tensor:
        """Format costs tensor to have shape (batch_size, num_costs)."""
        if isinstance(costs, list):
            return torch.stack([cost.clone() for cost in costs], dim=1)

        if costs.dim() == 1:
            if self.num_costs == 1:
                return costs.unsqueeze(1).clone()
            else:
                return costs.unsqueeze(1).expand(-1, self.num_costs).clone()

        return costs.clone()

    def update_lagrangian_multipliers(self, current_costs: list[float]) -> None:
        """Update Lagrangian multipliers using PID controller.

        Based on: "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
        https://arxiv.org/abs/2007.03964

        Args:
            current_costs: Current mean episode costs for each constraint.
        """
        for cost_idx in range(self.num_costs):
            current_cost = float(current_costs[cost_idx])
            cost_limit = self.cost_limits[cost_idx]

            # Positive delta means constraint violation.
            delta = current_cost - cost_limit

            # Integral term with anti-windup.
            self.pid_i[cost_idx] = max(0.0, self.pid_i[cost_idx] + delta * self.ki)
            if self.diff_norm:
                self.pid_i[cost_idx] = max(0.0, min(1.0, self.pid_i[cost_idx]))
            else:
                self.pid_i[cost_idx] = min(self.pid_i[cost_idx], self.pid_i_max)

            # Proportional term with EMA smoothing.
            alpha_p = self.pid_delta_p_ema_alpha
            self.delta_p[cost_idx] = alpha_p * self.delta_p[cost_idx] + (1 - alpha_p) * delta

            # Derivative term with EMA smoothing and delay.
            alpha_d = self.pid_delta_d_ema_alpha
            self.cost_ema[cost_idx] = alpha_d * self.cost_ema[cost_idx] + (1 - alpha_d) * current_cost
            if len(self.cost_delay_queue[cost_idx]) > 0:
                pid_d = max(0.0, self.cost_ema[cost_idx] - self.cost_delay_queue[cost_idx][0])
            else:
                pid_d = 0.0

            pid_output = self.kp * self.delta_p[cost_idx] + self.pid_i[cost_idx] + self.kd * pid_d
            self.lambdas[cost_idx] = max(0.0, pid_output)
            if self.diff_norm:
                self.lambdas[cost_idx] = min(1.0, self.lambdas[cost_idx])
            else:
                self.lambdas[cost_idx] = min(self.lambdas[cost_idx], self.lambda_max)

            self.cost_delay_queue[cost_idx].append(self.cost_ema[cost_idx])

    def update(
        self,
        current_costs: list[float] | None = None,
        obs_normalizer=None,
        critic_obs_normalizer=None,
        reward_normalizer=None,
    ) -> dict[str, float]:
        """Perform Safe SAC update step.

        Updates the PID Lagrangian multipliers (if ``current_costs`` is given),
        then delegates the gradient updates to :meth:`SAC.update`; the cost
        critics are trained inside that loop via :meth:`_update_extra_critics`.

        Args:
            current_costs: Current mean episode costs for the Lagrangian update.
                If None, the PID update is skipped.
            obs_normalizer: Optional normalizer for actor observations.
            critic_obs_normalizer: Optional normalizer for critic observations.
            reward_normalizer: Optional normalizer for rewards.

        Returns:
            Dictionary containing loss values for logging (includes ``cost_critic``).
        """
        if current_costs is not None:
            self.update_lagrangian_multipliers(current_costs)

        return super().update(
            obs_normalizer=obs_normalizer,
            critic_obs_normalizer=critic_obs_normalizer,
            reward_normalizer=reward_normalizer,
        )

    def _update_extra_critics(
        self,
        batch: dict[str, torch.Tensor],
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        bootstrap: torch.Tensor | None = None,
        effective_n_steps: torch.Tensor | None = None,
        obs_normalizer=None,
        critic_obs_normalizer=None,
    ) -> dict[str, float]:
        """Train the cost critics inside SAC's update loop (see :meth:`SAC.update`).

        With ``hazard_fraction > 0`` the cost critic trains on its OWN hazard-stratified
        batch, drawn from the same storage. The batch this hook is handed stays uniform
        and continues to feed the reward critic, the CVPO E-step, the eta dual, the
        lambda controller and the M-step -- all of which take plain means over the
        sampled states and would be biased by a stratified batch.
        """
        cost_is_weights = None
        if self.hazard_fraction > 0.0 and self.storage is not None:
            cost_batch = self.storage.sample(self.batch_size, stratified=True)
            costs = cost_batch.get("costs")
            obs = cost_batch["observations"]
            critic_obs = cost_batch.get("critic_observations", obs)
            actions = cost_batch["actions"]
            dones = cost_batch["dones"]
            next_obs = cost_batch["next_observations"]
            next_critic_obs = cost_batch.get("next_critic_observations", next_obs)
            bootstrap = cost_batch.get("bootstrap")
            effective_n_steps = cost_batch.get("effective_n_steps")
            obs, critic_obs, next_obs, next_critic_obs = self._normalize_obs_tensors(
                obs, critic_obs, next_obs, next_critic_obs, obs_normalizer, critic_obs_normalizer
            )
            cost_is_weights = cost_batch["cost_is_weights"]
            # Exact stratum composition from the sampler itself. Recomputing it from the
            # sampled costs would be wrong under n_step > 1, where a safe-stratum start
            # can still carry a positive aggregated cost from later in its window.
            self._last_replay_info = dict(self.storage.last_stratified_info)
        else:
            costs = batch.get("costs")

        if costs is None:
            costs = torch.zeros(actions.shape[0], self.num_costs, device=self.device)
        cost_critic_loss = self._update_cost_critic(
            obs, critic_obs, actions, costs, dones, next_obs, next_critic_obs,
            bootstrap=bootstrap, effective_n_steps=effective_n_steps,
            cost_is_weights=cost_is_weights,
        )
        return {"cost_critic": cost_critic_loss}

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        bootstrap: torch.Tensor | None = None,
        effective_n_steps: torch.Tensor | None = None,
        cost_is_weights: torch.Tensor | None = None,
    ) -> float:
        """Update cost Q-networks.

        Args:
            obs: Current actor observations (for sampling next actions).
            critic_obs: Current critic observations (for cost Q-value evaluation).
            actions: Actions taken.
            costs: Costs received [batch_size, num_costs] (n-step aggregated when
                sampled with n_step > 1, mirroring the reward channel).
            dones: Done flags.
            next_obs: Next actor observations.
            next_critic_obs: Next critic observations.
            bootstrap: Optional timeout flag for the bootstrap mask.
            effective_n_steps: Optional per-sample n-step horizon for the discount.
            cost_is_weights: Optional per-transition importance weights [batch_size, 1]
                from hazard-stratified replay. ``None`` reduces to the plain unweighted
                mean, which is arithmetically identical to the previous ``F.mse_loss``.
                These weights must reach NO other loss.

        Returns:
            Cost critic loss value.
        """
        if getattr(self.policy, "is_distributional_cost_critic", False):
            return self._update_cost_critic_distributional(
                obs, critic_obs, actions, costs, dones, next_obs, next_critic_obs,
                bootstrap=bootstrap, effective_n_steps=effective_n_steps,
                cost_is_weights=cost_is_weights,
            )

        with torch.no_grad():
            next_actions, _ = self.policy.sample_with_log_prob(next_obs)
            cost_q_target = self.policy.evaluate_cost_q_target(next_critic_obs, next_actions)

            # Cost Bellman backup: Q_c_target = c + γ^n * mask * Q_c_target.
            # No entropy term for cost critics.
            mask = self._bootstrap_mask(dones, bootstrap)
            discount = self._bootstrap_discount(effective_n_steps)
            target_cost_q = costs + discount * mask * cost_q_target

        cost_q = self.policy.evaluate_cost_q(critic_obs, actions)
        # Per-transition Bellman error so hazard-stratified replay can reweight it.
        cost_td_error = cost_q - target_cost_q  # [batch, num_costs]
        per_sample_cost_loss = cost_td_error.pow(2)
        if cost_is_weights is None:
            cost_critic_loss = per_sample_cost_loss.mean()  # == F.mse_loss, exactly
        else:
            # view(-1, 1) is the explicit guard against a [batch]-shaped weight
            # broadcasting into a [batch, batch] matrix. One weight per transition,
            # applied to every constraint before reducing.
            weights = cost_is_weights.view(-1, 1)
            cost_critic_loss = (weights * per_sample_cost_loss).mean()

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        cost_critic_params = []
        for critic in self.policy.cost_critics:
            cost_critic_params.extend(list(critic.parameters()))
        nn.utils.clip_grad_norm_(cost_critic_params, self.max_grad_norm)
        self.cost_critic_optimizer.step()

        return cost_critic_loss.item()

    def _update_cost_critic_distributional(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        bootstrap: torch.Tensor | None = None,
        effective_n_steps: torch.Tensor | None = None,
        cost_is_weights: torch.Tensor | None = None,
    ) -> float:
        """Categorical (C51) cost-critic update.

        Mirrors :meth:`SAC._update_critic_distributional` with two differences: no entropy
        term (cost carries no entropy bonus) and no min-over-twins pessimism (understating
        cost is the unsafe direction, and the cost channel uses a single critic by default).
        """
        costs = costs.squeeze(-1)  # [batch, 1] -> [batch]; distributional cost critic is single-constraint
        bootstrap_mask = self._bootstrap_mask(dones, bootstrap).squeeze(-1)
        discount = self._bootstrap_discount(effective_n_steps)
        if isinstance(discount, torch.Tensor):
            discount = discount.reshape(-1)

        with torch.no_grad():
            next_actions, _ = self.policy.sample_with_log_prob(next_obs)
            next_obs_norm = self.policy.critic_obs_normalizer(next_critic_obs)
            target_dists = []
            for target in self.policy.cost_critic_targets:
                dist = target.get_dist(target(next_obs_norm, next_actions))
                target_dists.append(
                    target.project(
                        next_dist=dist, rewards=costs, bootstrap=bootstrap_mask, discount=discount
                    )
                )

        obs_normalized = self.policy.critic_obs_normalizer(critic_obs)
        # Per-transition weights from hazard-stratified replay; [batch] to match the
        # per-sample cross-entropy, and 1.0 when stratification is off.
        weights = 1.0 if cost_is_weights is None else cost_is_weights.view(-1)
        cost_critic_loss = 0.0
        for critic, target_dist in zip(self.policy.cost_critics, target_dists):
            logits = critic(obs_normalized, actions)
            per_sample_cost_loss = -torch.sum(target_dist * F.log_softmax(logits, dim=-1), dim=-1)
            cost_critic_loss = cost_critic_loss + (weights * per_sample_cost_loss).mean()

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        cost_critic_params = []
        for critic in self.policy.cost_critics:
            cost_critic_params.extend(list(critic.parameters()))
        nn.utils.clip_grad_norm_(cost_critic_params, self.max_grad_norm)
        self.cost_critic_optimizer.step()

        return cost_critic_loss.item()

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> tuple[float, float]:
        """Update actor with the safety-augmented objective ``Q_r - α·log π - λ·Q_c``.

        Args:
            obs: Current actor observations.
            critic_obs: Current critic observations. If None, uses actor observations.

        Returns:
            Tuple of (actor_loss, alpha_loss).
        """
        actions, log_prob = self.policy.sample_with_log_prob(obs)
        critic_obs = obs if critic_obs is None else critic_obs

        q1, q2 = self.policy.evaluate_q(critic_obs, actions)
        q_min = torch.min(q1, q2)

        cost_q = self.policy.evaluate_cost_q(critic_obs, actions)  # [batch, num_costs]
        total_lambda = sum(self.lambdas)
        lambda_tensor = torch.tensor(self.lambdas, device=self.device, dtype=cost_q.dtype)
        cost_penalty = (cost_q * lambda_tensor).sum(dim=-1, keepdim=True)  # [batch, 1]

        # sum_norm (OmniSafe-style): divide only the cost penalty by (1 + λ) so a large
        # multiplier moderates the penalty instead of drowning the reward signal.
        if self.sum_norm and total_lambda > 0:
            cost_penalty = cost_penalty / (1.0 + total_lambda)

        actor_loss = (self.alpha.detach() * log_prob - (q_min - cost_penalty)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss_value = 0.0
        if self.auto_entropy_tuning and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss_value = alpha_loss.item()

        return actor_loss.item(), alpha_loss_value

    def get_penalty_info(self) -> dict[str, Any]:
        """Get Lagrangian multiplier information for logging."""
        return {
            "lambda_mean": sum(self.lambdas) / len(self.lambdas) if self.lambdas else 0.0,
            "lambda_max": max(self.lambdas) if self.lambdas else 0.0,
            "lambda_min": min(self.lambdas) if self.lambdas else 0.0,
            "lambda_list": self.lambdas.copy(),
            "cost_limits": self.cost_limits.copy(),
            "pid_gains": (self.kp, self.ki, self.kd),
            "pid_i": self.pid_i.copy(),
            "delta_p": self.delta_p.copy(),
            "alpha": self.alpha.item(),
        }

    def get_lagrangian_info(self) -> dict[str, Any]:
        """Get Lagrangian multiplier information for logging (alias)."""
        return self.get_penalty_info()

    def get_actual_action_std(self) -> float:
        """Mean action std from the stochastic actor's log_std head (rough estimate
        from a zero observation; 1.0 when the actor has no log_std head)."""
        with torch.no_grad():
            if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'log_std_head'):
                dummy_obs = torch.zeros(1, self.policy.actor.backbone[0].in_features, device=self.device)
                _, log_std = self.policy.actor(dummy_obs)
                return log_std.exp().mean().item()
        return 1.0
