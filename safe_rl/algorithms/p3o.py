from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple

from safe_rl.modules import ActorCritic
from safe_rl.storage import RolloutStorageCMDP


class P3O:
    """
    Penalized Proximal Policy Optimization (P3O) for Safe Reinforcement Learning
    
    Based on the paper: "Penalized Proximal Policy Optimization for Safe Reinforcement Learning"
    https://arxiv.org/pdf/2205.11814.pdf
    """
    policy: ActorCritic

    def __init__(
        self,
        policy: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        gamma_cost: Optional[float] = None,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        cost_critic_lr: Optional[float] = None,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # P3O specific parameters
        cost_limits: Optional[List[float]] = None,
        kappa: Optional[List[float]] = None,
        cost_loss_coef: float = 1.0,
        use_clipped_cost_loss: bool = True,
        rho: float = 1.5,
        kappa_max: float = 1000.0,
        adaptive_penalty: bool = True,
        constraint_margin: float = 0.85,
        cvar_alpha: Optional[float] = None,
        cvar_alpha_start: Optional[float] = None,
        cvar_alpha_end: Optional[float] = None,
        cvar_alpha_schedule: str = "constant",
        cvar_alpha_warmup_iters: int = 0,
        cvar_alpha_anneal_iters: int = 0,
        use_cvar_in_gate: bool = False,
        cvar_gate_coef: float = 1.0,
        cvar_gate_clip: Optional[float] = None,
        # Backward compatibility
        rnd_cfg: Optional[Dict[str, Any]] = None,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        multi_gpu_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Initialize cost constraints
        if cost_limits is None:
            raise ValueError("cost_limits must be provided")
        
        self.cost_limits = cost_limits
        self.num_costs = len(cost_limits)

        # Initialize penalty factors
        if kappa is None:
            self.kappa = [1.0] * self.num_costs
        elif len(kappa) == 1 and self.num_costs > 1:
            self.kappa = kappa * self.num_costs
        else:
            self.kappa = kappa

        print(f"P3O initialized with {self.num_costs} cost constraints and kappa: {self.kappa}")

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        
        # Validate and fix cost critic output dimensions
        self._validate_and_fix_cost_critic()

        # CVaR curriculum + gate anchor. cvar_alpha is the constant fallback when
        # cvar_alpha_start is not set, so existing configs keep their behavior.
        # alpha=1 reduces cvar_value to expected_value, so a curriculum starting at
        # 1.0 is mathematically equivalent to today's expected-value-only training.
        is_hlgauss = getattr(self.policy, "cost_critic_loss_type", None) == "hlgauss"
        self.cvar_alpha_start = cvar_alpha_start if cvar_alpha_start is not None else cvar_alpha
        self.cvar_alpha_end = cvar_alpha_end if cvar_alpha_end is not None else self.cvar_alpha_start
        self.cvar_alpha_schedule = cvar_alpha_schedule
        self.cvar_alpha_warmup_iters = int(cvar_alpha_warmup_iters)
        self.cvar_alpha_anneal_iters = int(cvar_alpha_anneal_iters)
        if self.cvar_alpha_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                f"cvar_alpha_schedule must be 'constant', 'linear', or 'cosine'; "
                f"got {self.cvar_alpha_schedule!r}"
            )
        for _name, _val in (("cvar_alpha_start", self.cvar_alpha_start),
                            ("cvar_alpha_end", self.cvar_alpha_end)):
            if _val is not None and not (0.0 < _val <= 1.0):
                raise ValueError(f"{_name} must be in (0, 1]; got {_val}")

        self.use_cvar_in_gate = bool(use_cvar_in_gate)
        self.cvar_gate_coef = float(cvar_gate_coef)
        self.cvar_gate_clip = cvar_gate_clip
        if self.use_cvar_in_gate and not is_hlgauss:
            raise ValueError(
                "use_cvar_in_gate=True requires an HL-Gauss cost critic "
                "(set policy.cost_critic_kwargs.loss_type=hlgauss)."
            )

        self._current_cvar_alpha = self.cvar_alpha_start
        if self._current_cvar_alpha is not None and is_hlgauss:
            self.policy.cost_critic.cvar_alpha = self._current_cvar_alpha
            print(
                f"[P3O] CVaR enabled: alpha={self._current_cvar_alpha} "
                f"schedule={self.cvar_alpha_schedule} "
                f"warmup={self.cvar_alpha_warmup_iters} anneal={self.cvar_alpha_anneal_iters} "
                f"gate_anchor={'on' if self.use_cvar_in_gate else 'off'}"
            )

        self.storage = None  # initialized later

        # Two-group optimizer: actor + value critic share the (KL-adaptive) actor LR;
        # the cost critic gets its own (typically higher, fixed) LR so it can chase the
        # moving target without being throttled by the actor's adaptive schedule.
        self.cost_critic_lr = cost_critic_lr if cost_critic_lr is not None else learning_rate
        if hasattr(self.policy, "cost_critic") and self.policy.cost_critic is not None:
            self.cost_critic_params = list(self.policy.cost_critic.parameters())
            cost_critic_param_ids = {id(p) for p in self.cost_critic_params}
            self.actor_value_params = [
                p for p in self.policy.parameters() if id(p) not in cost_critic_param_ids
            ]
            self.optimizer = optim.Adam([
                {"params": self.actor_value_params, "lr": learning_rate},
                {"params": self.cost_critic_params, "lr": self.cost_critic_lr},
            ])
            print(f"[P3O] Optimizer split: actor/value lr={learning_rate}, cost_critic lr={self.cost_critic_lr}")
        else:
            self.actor_value_params = list(self.policy.parameters())
            self.cost_critic_params = []
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCMDP.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        # Cost can use a shorter horizon than reward — sharper targets, easier to learn.
        self.gamma_cost = gamma_cost if gamma_cost is not None else gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.cost_loss_coef = cost_loss_coef
        self.use_clipped_cost_loss = use_clipped_cost_loss
        if getattr(self.policy, "cost_critic_loss_type", None) == "hlgauss" and self.use_clipped_cost_loss:
            print(
                "[P3O] use_clipped_cost_loss=True is incompatible with HL-Gauss cost critic; "
                "forcing use_clipped_cost_loss=False."
            )
            self.use_clipped_cost_loss = False
        
        # Adaptive penalty parameters
        self.rho = rho  # Penalty factor multiplier (ρ > 1)
        self.kappa_max = kappa_max  # Maximum penalty factor
        self.adaptive_penalty = adaptive_penalty
        self.constraint_margin = constraint_margin  # Margin for early constraint violation detection

        # Cost history per cost constraint
        self.cost_history = [[] for _ in range(self.num_costs)]
        self.history_length = 10
        
        # RND compatibility (P3O doesn't use RND, but runner checks for it)
        self.rnd = None
        self.intrinsic_rewards = None

    def init_storage(self, training_type: str, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: Tuple, critic_obs_shape: Tuple, action_shape: Tuple) -> None:
        # Cost shape for multiple cost constraints
        cost_shape = (self.num_costs,) if self.num_costs > 1 else None
        self.storage = RolloutStorageCMDP(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
            training_type=training_type, cost_shape=cost_shape, device=self.device
        )

    def test_mode(self) -> None:
        self.policy.eval()

    def train_mode(self) -> None:
        self.policy.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        
        # Cost critic returns tensor of shape (batch_size, num_costs)
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # Record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def _adjust_learning_rate(self, kl_mean: torch.Tensor) -> None:
        """
        Adjust the learning rate based on the KL divergence.

        Only updates the actor/value param group; the cost critic keeps its own
        (typically higher, fixed) LR so it can chase the moving target without
        being throttled by the actor's KL-driven schedule.
        """
        if kl_mean > self.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2 and kl_mean > 0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        # param_groups[0] is always actor+value; cost critic (if present) is in group 1.
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

    def _compute_cvar_alpha(self, iteration: int) -> Optional[float]:
        a0 = self.cvar_alpha_start
        if a0 is None:
            return None
        a1 = self.cvar_alpha_end
        w, T = self.cvar_alpha_warmup_iters, self.cvar_alpha_anneal_iters
        if self.cvar_alpha_schedule == "constant" or T <= 0:
            return a0
        if iteration < w:
            return a0
        if iteration >= w + T:
            return a1
        progress = (iteration - w) / T
        if self.cvar_alpha_schedule == "linear":
            return a0 + (a1 - a0) * progress
        return a1 + 0.5 * (a0 - a1) * (1.0 + math.cos(math.pi * progress))

    def constraint_violation_detected(
        self, current_costs: List[torch.Tensor],
        cost_advantages_batch: Optional[List[torch.Tensor]] = None
    ) -> List[bool]:
        """
        Detect constraint violations for each cost constraint.
        
        Args:
            current_costs: List of current episode costs
            cost_advantages_batch: Batch of cost advantages for each cost (optional)
            
        Returns:
            List[bool]: True if constraint violation is detected for each cost
        """
        violations = []
        for cost_idx in range(self.num_costs):
            current_cost = current_costs[cost_idx]
            
            # Update cost history
            cost_val = current_cost.item() if isinstance(current_cost, torch.Tensor) else current_cost
            self.cost_history[cost_idx].append(cost_val)
            if len(self.cost_history[cost_idx]) > self.history_length:
                self.cost_history[cost_idx].pop(0)
            
            # Primary condition: direct violation
            if current_cost > self.cost_limits[cost_idx]:
                violations.append(True)
                continue
            
            # Secondary condition: approaching limit with positive cost trend
            if cost_advantages_batch is not None:
                mean_cost_advantage = cost_advantages_batch[cost_idx].mean().item()
                if (current_cost > self.constraint_margin * self.cost_limits[cost_idx]
                        and mean_cost_advantage > 0.1):
                    violations.append(True)
                    continue
            
            # Tertiary condition: trending upward toward violation
            if len(self.cost_history[cost_idx]) >= 6:
                recent_trend = sum(self.cost_history[cost_idx][-3:]) / 3
                older_trend = sum(self.cost_history[cost_idx][-6:-3]) / 3
                if (recent_trend > older_trend
                        and current_cost > self.constraint_margin * self.cost_limits[cost_idx]):
                    violations.append(True)
                    continue
            
            violations.append(False)
        
        return violations

    def update_penalty_factor(
        self, current_costs: List[torch.Tensor],
        cost_advantages_batch: Optional[List[torch.Tensor]] = None
    ) -> None:
        """
        Symmetric multiplicative κ update:
            violation  → κ ← min(κ · ρ,   κ_max)
            safe       → κ ← max(κ / ρ,   0.1)

        Growth and decay use the same ρ, so alternating states cancel exactly
        (ρ · 1/ρ = 1). This prevents the ratchet explosion of the previous
        asymmetric tiered update.
        """
        if not self.adaptive_penalty:
            return

        violations = self.constraint_violation_detected(current_costs, cost_advantages_batch)

        for cost_idx, violation_detected in enumerate(violations):
            if violation_detected:
                self.kappa[cost_idx] = min(self.kappa[cost_idx] * self.rho, self.kappa_max)
            else:
                self.kappa[cost_idx] = max(self.kappa[cost_idx] / self.rho, 0.1)

    def _update_policy(
        self, batch: Tuple, current_costs: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Handle multiple costs with single cost critic having n output neurons.
        """
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
         returns_cost_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

        # Capture pre-normalization advantage stats for W&B diagnostics. These reveal
        # whether the cost critic is producing a usable signal (large abs_max + std)
        # versus being washed out (near-zero std), and whether κ is meaningfully
        # affecting the actor (cost_penalty_term should scale ~linearly with κ).
        with torch.no_grad():
            batch_metrics: Dict[str, float] = {
                "adv_reward_mean": advantages_batch.mean().item(),
                "adv_reward_std": advantages_batch.std().item(),
                "adv_reward_abs_max": advantages_batch.abs().max().item(),
                "adv_cost_mean": cost_advantages_batch.mean().item(),
                "adv_cost_std": cost_advantages_batch.std().item(),
                "adv_cost_abs_max": cost_advantages_batch.abs().max().item(),
                "adv_cost_positive_frac": (cost_advantages_batch > 0).float().mean().item(),
            }

        # Normalize advantages per mini-batch if enabled
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                # Cost advantages: scale by std only — mean-centering would flip the sign of
                # roughly half the unsafe samples and tell the policy they are "less unsafe than
                # average", neutralizing the κ penalty. This is safe to do here because
                # `evaluate_cost` always decodes V_cost as the expected value (not CVaR), so
                # cost advantages are already naturally centered around zero.
                for cost_idx in range(self.num_costs):
                    cost_advantages_batch[:, cost_idx] = (
                        cost_advantages_batch[:, cost_idx]
                        / (cost_advantages_batch[:, cost_idx].std() + 1e-8)
                    )

        # Perform the action and value predictions
        self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        value_batch = self.policy.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        
        # Single call to cost critic returns tensor with shape (batch_size, num_costs)
        cost_value_batch_all = self.policy.evaluate_cost(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )

        # CVaR-anchored gate: decode the worst-α tail off the cached HL-Gauss logits
        # without a second forward pass. The anchor is detached and clamped to [0, ...]
        # so it can only ever open the ReLU gate further, never close it; closing the
        # gate via a biased-upward signal would replicate the §4.1 silent-failure mode.
        cvar_anchor_per_cost: Optional[torch.Tensor] = None
        cvar_minus_exp_mean = 0.0
        if self.use_cvar_in_gate and self._current_cvar_alpha is not None:
            with torch.no_grad():
                cvar_batch = self.policy.cost_critic.cvar_value(
                    self.policy._cost_logits, alpha=self._current_cvar_alpha
                )
                cvar_minus_exp = cvar_batch - cost_value_batch_all
                anchor = cvar_minus_exp.mean(dim=0)
                if self.cvar_gate_clip is not None:
                    anchor = anchor.clamp(min=0.0, max=self.cvar_gate_clip)
                else:
                    anchor = anchor.clamp(min=0.0)
                cvar_anchor_per_cost = anchor
                cvar_minus_exp_mean = float(cvar_minus_exp.mean().item())

        mu_batch = self.policy.action_mean
        sigma_batch = self.policy.action_std
        entropy_batch = self.policy.entropy

        # Calculate KL-divergence and adjust learning rate if needed
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                self._adjust_learning_rate(kl_mean)

        # Calculate the importance sampling ratio
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        
        # Cost surrogate calculation for multiple costs
        total_cost_loss = 0
        current_costs_computed = []
        cost_penalty_acc = 0.0
        gate_open_acc = 0.0
        active_cost_heads = 0

        for cost_idx in range(self.num_costs):
            if current_costs is None:
                mean_cost = self.storage.get_mean_episode_costs()[cost_idx]
            else:
                mean_cost = current_costs[cost_idx]
            current_costs_computed.append(mean_cost)
            Jc = mean_cost - self.cost_limits[cost_idx]

            if Jc > 0:
                # Unclipped cost surrogate — matches paper and both external reference implementations.
                # Gate anchor uses gamma_cost (not gamma): the (1-γ)·Jc term amortizes the discounted
                # cost violation over the cost horizon, and Jc is in the cost-return space, so it must
                # use the cost discount. With gamma=0.99 vs gamma_cost=0.97 the anchor is 3× larger,
                # which is what makes the ReLU gate open reliably during violations.
                surrogate_cost_loss = (torch.squeeze(cost_advantages_batch[:, cost_idx]) * ratio).mean()
                gate_input = surrogate_cost_loss + (1.0 - self.gamma_cost) * Jc
                if cvar_anchor_per_cost is not None:
                    gate_input = gate_input + self.cvar_gate_coef * cvar_anchor_per_cost[cost_idx]
                cost_penalty = self.kappa[cost_idx] * F.relu(gate_input)
                total_cost_loss += cost_penalty
                with torch.no_grad():
                    cost_penalty_acc += cost_penalty.item()
                    gate_open_acc += float(gate_input.item() > 0.0)
                    active_cost_heads += 1

        batch_metrics["cost_penalty_term"] = (
            cost_penalty_acc / active_cost_heads if active_cost_heads else 0.0
        )
        batch_metrics["cost_gate_open_frac"] = (
            gate_open_acc / active_cost_heads if active_cost_heads else 0.0
        )
        if cvar_anchor_per_cost is not None:
            batch_metrics["cvar_anchor"] = float(cvar_anchor_per_cost.mean().item())
            batch_metrics["cvar_minus_exp_mean"] = cvar_minus_exp_mean

        # Standard PPO reward surrogate (Equation 6)
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        
        # Total P3O policy loss
        policy_loss = surrogate_loss + total_cost_loss
        
        # Value function loss (standard PPO)
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # No L2 weight decay here: PPO's value loss is plain (clipped) MSE, so
        # matching it keeps P3O's reward/value path identical to PPO when the
        # penalty gate is closed.

        # Cost value function losses with vectorized operations
        total_cost_critic_loss = 0

        if getattr(self.policy, "cost_critic_loss_type", None) == "hlgauss":
            # _cost_logits cached by evaluate_cost call above
            cost_losses_all = self.policy.cost_critic.loss(
                self.policy._cost_logits, returns_cost_batch
            ).mean(dim=0)
        elif self.use_clipped_cost_loss:
            # Vectorized operations across all costs simultaneously
            cost_clipped = target_cost_values_batch + (cost_value_batch_all - target_cost_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            cost_value_losses = (cost_value_batch_all - returns_cost_batch).pow(2)
            cost_value_losses_clipped = (cost_clipped - returns_cost_batch).pow(2)
            cost_losses_all = torch.max(cost_value_losses, cost_value_losses_clipped).mean(dim=0)
        else:
            cost_losses_all = (returns_cost_batch - cost_value_batch_all).pow(2).mean(dim=0)

        # Sum across all cost critics. Plain (clipped) MSE — no L2 weight decay,
        # mirroring the value critic above.
        total_cost_critic_loss = self.cost_loss_coef * cost_losses_all.sum()

        # Composite loss
        critic_loss = self.value_loss_coef * value_loss
        total_loss = policy_loss + critic_loss + total_cost_critic_loss - self.entropy_coef * entropy_batch.mean()

        # Skip non-finite mini-batches instead of letting NaN poison the network.
        # max_grad_norm + advantage normalization should prevent this; if it fires
        # repeatedly, lower kappa_max or cvar_alpha rather than ignoring the warning.
        if not torch.isfinite(total_loss):
            print(f"[P3O] Skipping non-finite loss at update: {total_loss.item()}")
            self.optimizer.zero_grad()
            batch_metrics.update({
                "value_loss": value_loss.item() if torch.isfinite(value_loss) else 0.0,
                "cost_loss": 0.0,
                "surrogate_loss": surrogate_loss.item() if torch.isfinite(surrogate_loss) else 0.0,
                "entropy": entropy_batch.mean().item(),
            })
            return batch_metrics

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip the actor+value and the cost critic separately. A single
        # clip over self.policy.parameters() lets the cost critic's regression
        # gradients inflate the global norm and shrink the actor's effective
        # step — so P3O would not reduce to PPO even when the penalty gate is
        # closed. Clipping the groups independently keeps the actor update
        # identical to PPO whenever total_cost_loss == 0.
        nn.utils.clip_grad_norm_(self.actor_value_params, self.max_grad_norm)
        if self.cost_critic_params:
            nn.utils.clip_grad_norm_(self.cost_critic_params, self.max_grad_norm)
        self.optimizer.step()

        # Don't update penalty factors here - moved to update() method to avoid too frequent updates

        batch_metrics.update({
            "value_loss": value_loss.item(),
            "cost_loss": total_cost_critic_loss.item(),
            "surrogate_loss": surrogate_loss.item(),
            "entropy": entropy_batch.mean().item(),
        })
        return batch_metrics

    def update(
        self,
        current_costs: Optional[List[torch.Tensor]] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Main update function that coordinates the update of actor-critic networks.
        """
        if iteration is not None and self.cvar_alpha_start is not None:
            self._current_cvar_alpha = self._compute_cvar_alpha(iteration)
            if hasattr(self.policy, "cost_critic") and self.policy.cost_critic is not None:
                self.policy.cost_critic.cvar_alpha = self._current_cvar_alpha

        # Determine the appropriate generator based on whether the model is recurrent
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        metrics_sum: Dict[str, float] = {}
        n_batches = 0
        for batch in generator:
            batch_metrics = self._update_policy(batch, current_costs)
            for key, value in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            n_batches += 1

        # Average across mini-batches.
        mean_metrics = {key: value / max(n_batches, 1) for key, value in metrics_sum.items()}

        # Update penalty factors once per iteration (not per mini-batch)
        if current_costs is not None:
            self.update_penalty_factor(current_costs, cost_advantages_batch=None)

        self.storage.clear()

        # Construct the loss dictionary. Legacy keys are renamed for the runner;
        # advantage diagnostics flow through unchanged and land under W&B `Loss/adv_*`.
        loss_dict = {
            "value_function": mean_metrics.pop("value_loss", 0.0),
            "cost_function": mean_metrics.pop("cost_loss", 0.0),
            "surrogate": mean_metrics.pop("surrogate_loss", 0.0),
            "entropy": mean_metrics.pop("entropy", 0.0),
        }
        loss_dict.update(mean_metrics)
        if self._current_cvar_alpha is not None:
            loss_dict["cvar_alpha_current"] = float(self._current_cvar_alpha)

        return loss_dict

    def process_env_step(self, rewards: torch.Tensor, costs: torch.Tensor,
                         dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """
        Process environment step with multiple costs.
        """
        self.transition.rewards = rewards.clone()
        
        # Handle costs tensor formatting
        self.transition.costs = self._format_costs_tensor(costs)
        
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            # Bootstrap multiple costs with tensor operations
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            # Expand timeout mask to match cost dimensions
            timeout_mask = timeout_mask.expand(-1, self.num_costs)
            
            if self.transition.cost_values is not None:
                cost_bootstrap = self.gamma_cost * self.transition.cost_values * timeout_mask
                self.transition.costs += cost_bootstrap

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _format_costs_tensor(self, costs: torch.Tensor) -> torch.Tensor:
        """
        Format costs tensor to have shape (batch_size, num_costs).
        """
        if isinstance(costs, list):
            return torch.stack([cost.clone() for cost in costs], dim=1)
        
        if costs.dim() == 1:
            if self.num_costs == 1:
                return costs.unsqueeze(1).clone()
            else:
                return costs.unsqueeze(1).expand(-1, self.num_costs).clone()
        
        return costs.clone()

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """
        Compute GAE returns for rewards.
        """
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs: torch.Tensor) -> None:
        """
        Compute GAE returns for multiple costs with single critic call.
        """
        # Single call returns tensor with shape (batch_size, num_costs)
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        # P3O never mean-centers cost advantages: the sign of the cost advantage is what tells
        # the κ-penalty whether a sample is unsafe. Mean-centering would zero out the penalty in
        # expectation. (CPO/PCPO use the storage's centering for their trust-region logic.)
        self.storage.compute_cost_returns(
            last_cost_values, self.gamma_cost, self.lam,
            normalize_cost_advantage=False,
        )

    def get_penalty_info(self) -> Dict[str, Any]:
        """
        Get penalty information for all costs.
        For logging purposes, returns scalar values (mean of lists for multiple costs).
        """
        return {
            # Convert lists to scalars for logging
            "kappa": torch.tensor(self.kappa).mean().item(),
            "cost_limit": torch.tensor(self.cost_limits).mean().item(),
            
            # Original lists for detailed analysis
            "kappa_list": self.kappa,
            "cost_limits": self.cost_limits,
            "constraint_margin": self.constraint_margin,
            "recent_costs": [hist[-5:] if len(hist) >= 5 else hist for hist in self.cost_history]
        }

    def _validate_and_fix_cost_critic(self) -> None:
        """
        Validate that the cost critic outputs the correct number of cost values.
        If not, recreate the cost critic with the correct output dimension.
        """
        # Check if the policy has a cost_critic attribute
        if not hasattr(self.policy, 'cost_critic'):
            raise ValueError("ActorCritic must have a cost_critic attribute for P3O algorithm")

        # HL-Gauss head outputs num_costs * num_bins; skip the linear-layer heuristic.
        if getattr(self.policy, "cost_critic_loss_type", None) == "hlgauss":
            return

        # Get the last layer of the cost critic to check output dimension
        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, torch.nn.Linear):
                last_layer = module
                break

        if last_layer is None:
            raise ValueError("Could not find linear layer in cost critic")

        current_outputs = last_layer.out_features
        
        if current_outputs != self.num_costs:
            print(f"WARNING: Cost critic outputs {current_outputs} values but P3O expects {self.num_costs}.")
            print("This mismatch will cause runtime errors. Please configure ActorCritic with num_costs parameter.")
            print(f"Example: ActorCritic(..., num_costs={self.num_costs})")