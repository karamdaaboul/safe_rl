from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple

from safe_rl.modules import ActorCriticCost
from safe_rl.storage import RolloutStorageCMDP


class P3O:
    """
    Penalized Proximal Policy Optimization (P3O) for Safe Reinforcement Learning
    
    Based on the paper: "Penalized Proximal Policy Optimization for Safe Reinforcement Learning"
    https://arxiv.org/pdf/2205.11814.pdf
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
        # P3O specific parameters
        cost_limits: Optional[List[float]] = None,
        kappa: Optional[List[float]] = None,
        cost_loss_coef: float = 1.0,
        use_clipped_cost_loss: bool = True,
        rho: float = 1.5,
        kappa_max: float = 1000.0,
        adaptive_penalty: bool = True,
        constraint_margin: float = 0.85,
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
        """
        if kl_mean > self.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2 and kl_mean > 0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

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
        Update penalty factors for each cost constraint based on violations.
        """
        if not self.adaptive_penalty:
            return
            
        violations = self.constraint_violation_detected(current_costs, cost_advantages_batch)
        
        for cost_idx, violation_detected in enumerate(violations):
            old_kappa = self.kappa[cost_idx]
            current_cost = current_costs[cost_idx]
            
            if violation_detected:
                # Current: κ = min(ρ * κ, κ_max)
                # Improved: Different update rates based on violation severity
                violation_ratio = current_cost / self.cost_limits[cost_idx]
                if violation_ratio > 2.0:  # Severe violation
                    self.kappa[cost_idx] *= 2.0  # Aggressive increase
                elif violation_ratio > 1.5:
                    self.kappa[cost_idx] *= 1.5
                else:
                    # Default behavior for moderate violations (>1.0)
                    self.kappa[cost_idx] = min(self.rho * self.kappa[cost_idx], self.kappa_max)
            else:
                # No violation detected - consider decreasing kappa
                cost_ratio = current_cost / self.cost_limits[cost_idx]
                if cost_ratio < 0.8:  # Well within limit
                    self.kappa[cost_idx] *= 0.95  # Slight decrease
                elif cost_ratio < 0.6:  # Very safe
                    self.kappa[cost_idx] *= 0.9   # More aggressive decrease
                # If cost_ratio >= 0.8, keep kappa unchanged (close to limit but no violation)
            
            # Apply constraints: minimum 0.1, maximum kappa_max
            self.kappa[cost_idx] = max(0.1, min(self.kappa[cost_idx], self.kappa_max))
            
            #print(f"Cost {cost_idx}: ratio={current_cost / self.cost_limits[cost_idx]:.3f}, "
            #      f"κ: {old_kappa:.3f} → {self.kappa[cost_idx]:.3f}")

    def _update_policy(
        self, batch: Tuple, current_costs: Optional[List[torch.Tensor]] = None
    ) -> Tuple[float, float, float, float]:
        """
        Handle multiple costs with single cost critic having n output neurons.
        """
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
         advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
         returns_cost_batch, old_actions_log_prob_batch,
         old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

        # Normalize advantages per mini-batch if enabled
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                # Normalize cost advantages for each cost
                for cost_idx in range(self.num_costs):
                    cost_advantages_batch[:, cost_idx] = (
                        (cost_advantages_batch[:, cost_idx] - cost_advantages_batch[:, cost_idx].mean())
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
        
        for cost_idx in range(self.num_costs):
            # Cost surrogate calculation (Equation 7)
            surrogate_cost = torch.squeeze(cost_advantages_batch[:, cost_idx]) * ratio
            surrogate_cost_clipped = torch.squeeze(cost_advantages_batch[:, cost_idx]) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # P3O extends PPO's clipped surrogate to costs
            surrogate_cost_loss = torch.max(surrogate_cost, surrogate_cost_clipped).mean()
            
            # Calculate the cost constraint term using mean episode costs
            if current_costs is None:
                # Use mean episode costs from storage (like statistics.mean(cost_buffer) in runner)
                mean_cost = self.storage.get_mean_episode_costs()[cost_idx]
            else:
                mean_cost = current_costs[cost_idx]
            current_costs_computed.append(mean_cost)
            Jc = mean_cost - self.cost_limits[cost_idx]
            
            # P3O penalty loss for this cost (Equation 3)
            cost_penalty = self.kappa[cost_idx] * F.relu(surrogate_cost_loss + (1.0 - self.gamma) * Jc)
            total_cost_loss += cost_penalty

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
        
        # Cost value function losses with vectorized operations
        total_cost_critic_loss = 0
        
        if self.use_clipped_cost_loss:
            # Vectorized operations across all costs simultaneously
            cost_clipped = target_cost_values_batch + (cost_value_batch_all - target_cost_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            cost_value_losses = (cost_value_batch_all - returns_cost_batch).pow(2)
            cost_value_losses_clipped = (cost_clipped - returns_cost_batch).pow(2)
            cost_losses_all = torch.max(cost_value_losses, cost_value_losses_clipped).mean(dim=0)
        else:
            cost_losses_all = (returns_cost_batch - cost_value_batch_all).pow(2).mean(dim=0)
        
        # Sum across all cost critics
        total_cost_critic_loss = self.cost_loss_coef * cost_losses_all.sum()

        # Composite loss
        critic_loss = self.value_loss_coef * value_loss
        total_loss = policy_loss + critic_loss + total_cost_critic_loss - self.entropy_coef * entropy_batch.mean()

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Don't update penalty factors here - moved to update() method to avoid too frequent updates

        return value_loss.item(), total_cost_critic_loss.item(), surrogate_loss.item(), entropy_batch.mean().item()

    def update(self, current_costs: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Main update function that coordinates the update of actor-critic networks.
        """
        mean_value_loss = 0
        mean_cost_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # Determine the appropriate generator based on whether the model is recurrent
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            value_loss, cost_loss, surrogate_loss, entropy = self._update_policy(batch, current_costs)
            mean_value_loss += value_loss
            mean_cost_loss += cost_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy += entropy

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # Update penalty factors once per iteration (not per mini-batch)
        if current_costs is not None:
            self.update_penalty_factor(current_costs, cost_advantages_batch=None)

        self.storage.clear()

        # Construct the loss dictionary (similar to PPO)
        loss_dict = {
            "value_function": mean_value_loss,
            "cost_function": mean_cost_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

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
                cost_bootstrap = self.gamma * self.transition.cost_values * timeout_mask
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
        self.storage.compute_cost_returns(
            last_cost_values, self.gamma, self.lam, 
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch
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
            print("This mismatch will cause runtime errors. Please configure ActorCriticCost with num_costs parameter.")
            print(f"Example: ActorCriticCost(..., num_costs={self.num_costs})")