from __future__ import annotations

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
    
    Based on the approach from SKRL PPOL but adapted to the P3O infrastructure.
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
        # PPOL-PID specific parameters
        cost_limits: Optional[List[float]] = None,
        lagrangian_pid: Tuple[float, float, float] = (0.05, 0.0005, 0.1),  # (Kp, Ki, Kd)
        cost_loss_coef: float = 1.0,
        use_clipped_cost_loss: bool = True,
        constraint_margin: float = 0.95,  # Margin for constraint activation
        pid_scale: float = 1.0,  # Scaling factor for PID output
        lambda_init: Optional[List[float]] = None,  # Initial Lagrangian multipliers
        lambda_max: float = 100.0,  # Maximum Lagrangian multiplier value
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

        # Initialize Lagrangian multipliers
        if lambda_init is None:
            self.lambdas = [0.1] * self.num_costs  # Small initial values
        elif len(lambda_init) == 1 and self.num_costs > 1:
            self.lambdas = lambda_init * self.num_costs
        else:
            self.lambdas = lambda_init

        # PID controller parameters
        self.kp, self.ki, self.kd = lagrangian_pid
        self.pid_scale = pid_scale
        self.lambda_max = lambda_max
        self.constraint_margin = constraint_margin

        # PID state variables for each cost
        self.integral_errors = [0.0] * self.num_costs
        self.previous_errors = [0.0] * self.num_costs
        self.error_history = [[] for _ in range(self.num_costs)]
        self.history_length = 10

        print(f"PPOL-PID initialized with {self.num_costs} cost constraints")
        print(f"PID gains: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
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
        Update Lagrangian multipliers using PID controller.
        
        The PID controller adjusts lambda based on constraint violation:
        - P (Proportional): Responds to current constraint violation
        - I (Integral): Responds to accumulated violations over time  
        - D (Derivative): Responds to rate of change in violations
        """
        for cost_idx in range(self.num_costs):
            current_cost = current_costs[cost_idx]
            cost_limit = self.cost_limits[cost_idx]
            
            # Calculate constraint violation (error)
            error = float(current_cost - cost_limit)  # Positive if violation
            
            # Update error history
            self.error_history[cost_idx].append(error)
            if len(self.error_history[cost_idx]) > self.history_length:
                self.error_history[cost_idx].pop(0)
            
            # PID components
            # Proportional term: current error
            p_term = self.kp * max(0.0, error)  # Only positive violations matter
            
            # Integral term: accumulated error over time
            self.integral_errors[cost_idx] += error
            # Prevent integral windup
            self.integral_errors[cost_idx] = max(-10.0, min(10.0, self.integral_errors[cost_idx]))
            i_term = self.ki * self.integral_errors[cost_idx]
            
            # Derivative term: rate of change of error
            if len(self.error_history[cost_idx]) >= 2:
                error_derivative = error - self.previous_errors[cost_idx]
                d_term = self.kd * error_derivative
            else:
                d_term = 0.0
                
            self.previous_errors[cost_idx] = error
            
            # PID output (change in lambda)
            pid_output = (p_term + i_term + d_term) * self.pid_scale
            
            # Update Lagrangian multiplier
            old_lambda = self.lambdas[cost_idx]
            self.lambdas[cost_idx] = max(0.0, min(
                self.lambdas[cost_idx] + pid_output, 
                self.lambda_max
            ))
            
            # Optional: gradual decay when well within constraints
            if error < -0.2 * cost_limit:  # Well within constraint
                self.lambdas[cost_idx] *= 0.99  # Gradual decay
                
            #print(f"Cost {cost_idx}: error={error:.3f}, PID={pid_output:.3f}, "
            #      f"λ: {old_lambda:.3f} → {self.lambdas[cost_idx]:.3f}")

    def _update_policy(self, batch: Tuple, current_costs: Optional[List[torch.Tensor]] = None) -> Tuple[float, float, float]:
        """
        PPO policy update with Lagrangian constraint handling.
        """
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
         returns_cost_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

        # Normalize advantages
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
        
        # Standard PPO reward objective
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        
        # Lagrangian constraint terms
        total_constraint_loss = 0
        current_costs_computed = []
        
        for cost_idx in range(self.num_costs):
            # Cost advantage for this constraint
            cost_advantage = torch.squeeze(cost_advantages_batch[:, cost_idx])
            
            # Cost surrogate (similar to reward surrogate)
            cost_surrogate = cost_advantage * ratio
            cost_surrogate_clipped = cost_advantage * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            cost_surrogate_loss = torch.max(cost_surrogate, cost_surrogate_clipped).mean()
            
            # Current cost for Lagrangian update
            if current_costs is None:
                mean_cost = self.storage.get_mean_episode_costs()[cost_idx]
            else:
                mean_cost = current_costs[cost_idx]
            current_costs_computed.append(mean_cost)
            
            # Lagrangian constraint loss: λ * (cost_surrogate)
            # This penalizes policy changes that increase constraint violations
            constraint_loss = self.lambdas[cost_idx] * cost_surrogate_loss
            total_constraint_loss += constraint_loss

        # Total policy loss: PPO objective + Lagrangian constraints
        policy_loss = surrogate_loss + total_constraint_loss
        
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

        # Update Lagrangian multipliers after optimization
        self.update_lagrangian_multipliers(current_costs_computed)

        return value_loss.item(), total_cost_critic_loss.item(), surrogate_loss.item()

    def update(self, current_costs: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """Main update function."""
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

    def get_lagrangian_info(self) -> Dict[str, Any]:
        """Get Lagrangian multiplier information for logging."""
        return {
            "lambda_mean": torch.tensor(self.lambdas).mean().item(),
            "lambda_max": max(self.lambdas),
            "lambda_min": min(self.lambdas),
            "lambda_list": self.lambdas,
            "cost_limits": self.cost_limits,
            "pid_gains": (self.kp, self.ki, self.kd),
            "integral_errors": self.integral_errors,
            "recent_errors": [hist[-3:] if len(hist) >= 3 else hist for hist in self.error_history]
        }

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