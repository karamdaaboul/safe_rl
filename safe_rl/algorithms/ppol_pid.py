from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple

from safe_rl.modules import ActorCriticCost
from safe_rl.storage import RolloutStorageCMDP


class PPOL_PID:
    """
    PPO Lagrangian with PID Controller for Safe Reinforcement Learning

    Combines PPO with Lagrangian constraint handling using PID controllers
    to adaptively update Lagrangian multipliers based on constraint violations.

    Based on OmniSafe's CPPOPID implementation:
    "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
    https://arxiv.org/abs/2007.03964

    Key features (aligned with OmniSafe):
    - EMA smoothing on proportional and derivative terms for stability
    - Delayed derivative calculation to reduce noise
    - PID output directly sets lambda (not accumulated)
    - Normalized surrogate: (adv_r - λ * adv_c) / (1 + λ)
    """
    policy: ActorCriticCost

    def __init__(
        self,
        policy: ActorCriticCost,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # PPOL-PID specific parameters (aligned with OmniSafe CPPOPID)
        cost_limits: Optional[List[float]] = None,
        lagrangian_pid: Tuple[float, float, float] = (0.1, 0.01, 0.01),  # (Kp, Ki, Kd) OmniSafe defaults
        cost_loss_coef: float = 1.0,
        use_clipped_cost_loss: bool = True,
        # PID EMA smoothing parameters (OmniSafe defaults)
        pid_delta_p_ema_alpha: float = 0.95,  # EMA alpha for proportional term
        pid_delta_d_ema_alpha: float = 0.95,  # EMA alpha for derivative term
        pid_d_delay: int = 10,  # Delay steps for derivative calculation
        # Constraint parameters
        lambda_init: Optional[List[float]] = None,  # Initial Lagrangian multipliers
        lambda_max: float = 100.0,  # Maximum Lagrangian multiplier value
        sum_norm: bool = True,  # Apply sum normalization (OmniSafe default)
        diff_norm: bool = False,  # Apply diff normalization (clips to [0, 1])
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

        # PID controller parameters (OmniSafe-style)
        self.kp, self.ki, self.kd = lagrangian_pid
        self.lambda_max = lambda_max
        self.sum_norm = sum_norm
        self.diff_norm = diff_norm

        # EMA smoothing parameters
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self.pid_d_delay = pid_d_delay

        # Initialize Lagrangian multipliers (these are the PID outputs, not accumulated)
        if lambda_init is None:
            init_val = 0.001  # OmniSafe default
            self.lambdas = [init_val] * self.num_costs
        elif len(lambda_init) == 1 and self.num_costs > 1:
            self.lambdas = list(lambda_init) * self.num_costs
        else:
            self.lambdas = list(lambda_init)

        # PID state variables for each cost (OmniSafe-style)
        # Integral term: accumulates Ki * delta directly
        self.pid_i = [self.lambdas[i] for i in range(self.num_costs)]
        # EMA-smoothed proportional term (delta_p)
        self.delta_p = [0.0] * self.num_costs
        # EMA-smoothed cost for derivative calculation
        self.cost_ema = [0.0] * self.num_costs
        # Delay queue for derivative calculation
        self.cost_delay_queue: List[deque] = [
            deque([0.0], maxlen=pid_d_delay) for _ in range(self.num_costs)
        ]

        print(f"PPOL-PID initialized with {self.num_costs} cost constraints (OmniSafe-style)")
        print(f"PID gains: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        print(f"EMA alphas: P={self.pid_delta_p_ema_alpha}, D={self.pid_delta_d_ema_alpha}, delay={self.pid_d_delay}")
        print(f"Initial lambdas: {self.lambdas}")

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        
        # Validate cost critic
        self._validate_and_fix_cost_critic()
        
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCMDP.Transition()

        # PPO parameters
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
        
        # RND compatibility
        self.rnd = None
        self.intrinsic_rewards = None

    def init_storage(self, training_type: str, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: Tuple, critic_obs_shape: Tuple, action_shape: Tuple) -> None:
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
        
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def _adjust_learning_rate(self, kl_mean: torch.Tensor) -> None:
        """Adjust the learning rate based on the KL divergence."""
        if kl_mean > self.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2 and kl_mean > 0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def update_lagrangian_multipliers(self, current_costs: List[torch.Tensor]) -> None:
        """
        Update Lagrangian multipliers using PID controller (OmniSafe-style).

        Based on: "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
        https://arxiv.org/abs/2007.03964

        The PID controller computes lambda directly (not accumulated):
        λ = Kp * delta_p + pid_i + Kd * pid_d

        Where:
        - delta_p: EMA-smoothed error (proportional term)
        - pid_i: Accumulated Ki * delta (integral term)
        - pid_d: Delayed difference of EMA-smoothed costs (derivative term)
        """
        for cost_idx in range(self.num_costs):
            # Get current cost value
            if isinstance(current_costs[cost_idx], torch.Tensor):
                current_cost = float(current_costs[cost_idx].item())
            else:
                current_cost = float(current_costs[cost_idx])

            cost_limit = self.cost_limits[cost_idx]

            # Calculate error (delta): positive means violation
            delta = current_cost - cost_limit

            # === Integral term (I) ===
            # Accumulate Ki * delta directly (OmniSafe style)
            self.pid_i[cost_idx] = max(0.0, self.pid_i[cost_idx] + delta * self.ki)
            # Apply diff_norm clipping if enabled
            if self.diff_norm:
                self.pid_i[cost_idx] = max(0.0, min(1.0, self.pid_i[cost_idx]))

            # === Proportional term (P) with EMA smoothing ===
            alpha_p = self.pid_delta_p_ema_alpha
            self.delta_p[cost_idx] = alpha_p * self.delta_p[cost_idx] + (1 - alpha_p) * delta

            # === Derivative term (D) with EMA smoothing and delay ===
            alpha_d = self.pid_delta_d_ema_alpha
            self.cost_ema[cost_idx] = alpha_d * self.cost_ema[cost_idx] + (1 - alpha_d) * current_cost

            # Derivative uses delayed cost difference (reduces noise)
            if len(self.cost_delay_queue[cost_idx]) > 0:
                pid_d = max(0.0, self.cost_ema[cost_idx] - self.cost_delay_queue[cost_idx][0])
            else:
                pid_d = 0.0

            # === Compute PID output (this IS the new lambda, not added to it) ===
            pid_output = self.kp * self.delta_p[cost_idx] + self.pid_i[cost_idx] + self.kd * pid_d

            # Apply constraints
            old_lambda = self.lambdas[cost_idx]
            self.lambdas[cost_idx] = max(0.0, pid_output)

            if self.diff_norm:
                self.lambdas[cost_idx] = min(1.0, self.lambdas[cost_idx])
            elif not self.sum_norm:
                self.lambdas[cost_idx] = min(self.lambdas[cost_idx], self.lambda_max)
            else:
                # sum_norm enabled: apply lambda_max
                self.lambdas[cost_idx] = min(self.lambdas[cost_idx], self.lambda_max)

            # Update delay queue for next iteration
            self.cost_delay_queue[cost_idx].append(self.cost_ema[cost_idx])

    def _update_policy(self, batch: Tuple, current_costs: Optional[List[torch.Tensor]] = None) -> Tuple[float, float, float]:
        """
        PPO policy update with Lagrangian constraint handling (OmniSafe-style).

        Uses normalized surrogate advantage: (adv_r - λ * adv_c) / (1 + λ)
        This prevents the cost term from completely dominating when λ is large.
        """
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
         returns_cost_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

        # Normalize advantages per mini-batch if enabled
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                for cost_idx in range(self.num_costs):
                    cost_advantages_batch[:, cost_idx] = (
                        (cost_advantages_batch[:, cost_idx] - cost_advantages_batch[:, cost_idx].mean())
                        / (cost_advantages_batch[:, cost_idx].std() + 1e-8)
                    )

        # Forward pass
        self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        value_batch = self.policy.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        cost_value_batch_all = self.policy.evaluate_cost(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )

        mu_batch = self.policy.action_mean
        sigma_batch = self.policy.action_std
        entropy_batch = self.policy.entropy

        # KL divergence and learning rate adjustment
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

        # Importance sampling ratio
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

        # === OmniSafe-style normalized surrogate computation ===
        # Compute combined advantage: (adv_r - λ * adv_c) / (1 + λ)
        # This is the key difference from standard PPO-Lagrangian

        # Sum of Lagrangian multipliers for normalization
        total_lambda = sum(self.lambdas)

        # Compute weighted cost advantage
        weighted_cost_advantage = torch.zeros_like(torch.squeeze(advantages_batch))
        for cost_idx in range(self.num_costs):
            cost_advantage = torch.squeeze(cost_advantages_batch[:, cost_idx])
            weighted_cost_advantage += self.lambdas[cost_idx] * cost_advantage

        # Combined normalized advantage (OmniSafe Equation)
        # adv_combined = (adv_r - λ * adv_c) / (1 + λ)
        combined_advantage = (torch.squeeze(advantages_batch) - weighted_cost_advantage) / (1.0 + total_lambda)

        # PPO clipped surrogate with normalized advantage
        surrogate = -combined_advantage * ratio
        surrogate_clipped = -combined_advantage * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Policy loss is just the normalized surrogate (no separate constraint term needed)
        policy_loss = surrogate_loss

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        
        # Cost value function losses
        if self.use_clipped_cost_loss:
            cost_clipped = target_cost_values_batch + (cost_value_batch_all - target_cost_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            cost_value_losses = (cost_value_batch_all - returns_cost_batch).pow(2)
            cost_value_losses_clipped = (cost_clipped - returns_cost_batch).pow(2)
            cost_losses_all = torch.max(cost_value_losses, cost_value_losses_clipped).mean(dim=0)
        else:
            cost_losses_all = (returns_cost_batch - cost_value_batch_all).pow(2).mean(dim=0)
        
        total_cost_critic_loss = self.cost_loss_coef * cost_losses_all.sum()

        # Total loss
        critic_loss = self.value_loss_coef * value_loss
        total_loss = policy_loss + critic_loss + total_cost_critic_loss - self.entropy_coef * entropy_batch.mean()

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Don't update Lagrangian multipliers here - moved to update() method to avoid too frequent updates

        return value_loss.item(), total_cost_critic_loss.item(), surrogate_loss.item()

    def update(self, current_costs: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """Main update function."""
        # Update Lagrangian multipliers ONCE per iteration BEFORE policy updates
        # This matches OmniSafe's approach and prevents multipliers from jumping to extremes
        if current_costs is not None:
            self.update_lagrangian_multipliers(current_costs)
        elif self.storage is not None:
            # Fall back to storage costs if not provided
            current_costs_from_storage = self.storage.get_mean_episode_costs()
            self.update_lagrangian_multipliers(current_costs_from_storage)

        mean_value_loss = 0
        mean_cost_loss = 0
        mean_surrogate_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            value_loss, cost_loss, surrogate_loss = self._update_policy(batch, current_costs)
            mean_value_loss += value_loss
            mean_cost_loss += cost_loss
            mean_surrogate_loss += surrogate_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_function": mean_cost_loss,
            "surrogate": mean_surrogate_loss,
        }

        return loss_dict

    def process_env_step(self, rewards: torch.Tensor, costs: torch.Tensor,
                         dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """Process environment step with costs."""
        self.transition.rewards = rewards.clone()
        self.transition.costs = self._format_costs_tensor(costs)
        self.transition.dones = dones
        
        # Bootstrapping on timeouts
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            timeout_mask = timeout_mask.expand(-1, self.num_costs)
            
            if self.transition.cost_values is not None:
                cost_bootstrap = self.gamma * self.transition.cost_values * timeout_mask
                self.transition.costs += cost_bootstrap

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

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

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute GAE returns for rewards."""
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute GAE returns for costs."""
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values, self.gamma, self.lam, 
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch
        )

    def get_penalty_info(self) -> Dict[str, Any]:
        """
        Get Lagrangian multiplier information for logging.
        Uses same interface as P3O for compatibility with runner logging.
        """
        return {
            # Convert lists to scalars for logging (compatible with P3O interface)
            "kappa": torch.tensor(self.lambdas).mean().item(),  # Use lambda as "kappa" for logging compatibility
            "cost_limit": torch.tensor(self.cost_limits).mean().item(),

            # Original lists for detailed analysis
            "kappa_list": self.lambdas,  # Lambdas mapped to kappa_list for logging compatibility
            "cost_limits": self.cost_limits,

            # PPOL-PID specific information
            "lambda_mean": torch.tensor(self.lambdas).mean().item(),
            "lambda_max": max(self.lambdas),
            "lambda_min": min(self.lambdas),
            "lambda_list": self.lambdas,
            "pid_gains": (self.kp, self.ki, self.kd),
            "pid_i": self.pid_i,  # Integral term
            "delta_p": self.delta_p,  # Smoothed proportional term
            "integral_errors": self.pid_i,  # Alias for runner compatibility
        }

    def get_lagrangian_info(self) -> Dict[str, Any]:
        """Get Lagrangian multiplier information for logging (legacy method)."""
        return self.get_penalty_info()

    def _validate_and_fix_cost_critic(self) -> None:
        """Validate cost critic output dimensions."""
        if not hasattr(self.policy, 'cost_critic'):
            raise ValueError("ActorCritic must have a cost_critic attribute for PPOL-PID algorithm")
        
        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, torch.nn.Linear):
                last_layer = module
                break
        
        if last_layer is None:
            raise ValueError("Could not find linear layer in cost critic")
        
        current_outputs = last_layer.out_features
        
        if current_outputs != self.num_costs:
            print(f"WARNING: Cost critic outputs {current_outputs} values but PPOL-PID expects {self.num_costs}.")
            print("This mismatch will cause runtime errors. Please configure ActorCriticCost with num_costs parameter.")
            print(f"Example: ActorCriticCost(..., num_costs={self.num_costs})")