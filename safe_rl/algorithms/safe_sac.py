# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic
from safe_rl.storage.replay_storage import ReplayStorage


class SafeSAC:
    """Safe Soft Actor-Critic algorithm with PID Lagrangian constraint handling.

    This implementation combines SAC with constrained RL using PID Lagrangian
    multipliers for adaptive constraint handling, similar to PPOL_PID but for
    off-policy learning.

    Key features:
    - Twin Q-networks for reward estimation (standard SAC)
    - Cost Q-network(s) for constraint estimation
    - PID controller for adaptive Lagrangian multiplier updates
    - Automatic entropy coefficient (alpha) tuning
    - Soft target updates (Polyak averaging)

    References:
    - SAC: https://arxiv.org/abs/1801.01290
    - Safe RL with PID Lagrangian: https://arxiv.org/abs/2007.03964
    """

    policy: SafeSACActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy: SafeSACActorCritic,
        # Learning rates
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        cost_critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # SAC parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float | None = None,
        # Training parameters
        batch_size: int = 256,
        num_updates_per_step: int = 1,
        policy_frequency: int = 1,
        max_grad_norm: float = 1.0,
        # Safe RL parameters (aligned with PPOL_PID / OmniSafe)
        cost_limits: list[float] | None = None,
        lagrangian_pid: tuple[float, float, float] = (0.1, 0.01, 0.01),  # (Kp, Ki, Kd)
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float = 0.95,
        pid_d_delay: int = 10,
        lambda_init: list[float] | None = None,
        lambda_max: float = 100.0,
        sum_norm: bool = True,
        diff_norm: bool = False,
        # Device
        device: str = "cpu",
        # Multi-GPU (for compatibility)
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize Safe SAC algorithm.

        Args:
            policy: SafeSACActorCritic module.
            actor_lr: Learning rate for the actor.
            critic_lr: Learning rate for the reward critics.
            cost_critic_lr: Learning rate for the cost critics.
            alpha_lr: Learning rate for the entropy coefficient.
            gamma: Discount factor.
            tau: Soft target update coefficient.
            alpha: Initial entropy regularization coefficient.
            auto_entropy_tuning: Whether to automatically tune alpha.
            target_entropy: Target entropy for auto-tuning.
            batch_size: Mini-batch size for updates.
            num_updates_per_step: Number of gradient updates per environment step.
            policy_frequency: Frequency of actor/alpha updates relative to critic updates.
            max_grad_norm: Maximum gradient norm for clipping.
            cost_limits: Cost limits for each constraint (required).
            lagrangian_pid: PID gains (Kp, Ki, Kd) for Lagrangian multiplier updates.
            pid_delta_p_ema_alpha: EMA alpha for proportional term smoothing.
            pid_delta_d_ema_alpha: EMA alpha for derivative term smoothing.
            pid_d_delay: Delay steps for derivative calculation.
            lambda_init: Initial Lagrangian multipliers.
            lambda_max: Maximum Lagrangian multiplier value.
            sum_norm: Apply sum normalization for lambda.
            diff_norm: Apply diff normalization (clips to [0, 1]).
            device: Device to run on.
            multi_gpu_cfg: Multi-GPU configuration (for compatibility).
        """
        if kwargs:
            print(f"SafeSAC.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")

        if cost_limits is None:
            raise ValueError("cost_limits must be provided for Safe SAC")

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        # SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_updates_per_step = num_updates_per_step
        self.policy_frequency = max(1, policy_frequency)
        self.max_grad_norm = max_grad_norm

        # Safe RL parameters
        self.cost_limits = cost_limits
        self.num_costs = len(cost_limits)

        # Validate policy has matching num_costs
        if hasattr(policy, 'num_costs') and policy.num_costs != self.num_costs:
            print(f"WARNING: Policy num_costs ({policy.num_costs}) doesn't match cost_limits ({self.num_costs})")

        # PID controller parameters
        self.kp, self.ki, self.kd = lagrangian_pid
        self.lambda_max = lambda_max
        self.sum_norm = sum_norm
        self.diff_norm = diff_norm
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self.pid_d_delay = pid_d_delay
        # Integral anti-windup: prevent integral term from growing too large
        self.pid_i_max = lambda_max * 0.5  # Cap integral at 50% of lambda_max

        # Initialize Lagrangian multipliers
        if lambda_init is None:
            init_val = 0.001
            self.lambdas = [init_val] * self.num_costs
        elif len(lambda_init) == 1 and self.num_costs > 1:
            self.lambdas = list(lambda_init) * self.num_costs
        else:
            self.lambdas = list(lambda_init)

        # PID state variables (similar to PPOL_PID)
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

        # Entropy coefficient (alpha)
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -policy.num_actions
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.tensor([math.log(alpha)], requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor([alpha], device=device).log()
            self.target_entropy = None
            self.alpha_optimizer = None

        # Optimizers
        # Actor optimizer
        actor_params = list(policy.actor.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)

        # Reward critic optimizer (all reward critics)
        reward_critic_params = []
        for critic in policy.reward_critics:
            reward_critic_params.extend(list(critic.parameters()))
        self.critic_optimizer = optim.Adam(reward_critic_params, lr=critic_lr)

        # Cost critic optimizer
        cost_critic_params = []
        for critic in policy.cost_critics:
            cost_critic_params.extend(list(critic.parameters()))
        self.cost_critic_optimizer = optim.Adam(cost_critic_params, lr=cost_critic_lr)

        # Unified optimizer for compatibility
        self.optimizer = self.actor_optimizer

        # Storage (initialized later)
        self.storage: ReplayStorage | None = None

        # Multi-GPU (for compatibility)
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # For compatibility with PPO interface
        self.rnd = None
        self.learning_rate = actor_lr

        # Episode cost tracking for PID updates
        self._episode_costs: list[list[float]] = [[] for _ in range(self.num_costs)]

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient."""
        return self.log_alpha.exp()

    def init_storage(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: list[int],
        act_shape: list[int],
    ) -> None:
        """Initialize the replay buffer with cost storage support.

        Args:
            buffer_size: Maximum number of transitions to store.
            num_envs: Number of parallel environments.
            obs_shape: Shape of observations.
            act_shape: Shape of actions.
        """
        self.storage = ReplayStorage(
            num_envs=num_envs,
            max_size=buffer_size,
            obs_shape=obs_shape,
            action_shape=act_shape,
            device=self.device,
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
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: Current observations.
            action: Actions taken.
            reward: Rewards received.
            done: Done flags.
            next_obs: Next observations.
            cost: Costs received (for safe RL).
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage() first.")

        # Format cost tensor
        extras = {}
        if cost is not None:
            cost = self._format_costs_tensor(cost)
        else:
            # If no cost provided, use zeros
            cost = torch.zeros(obs.shape[0], self.num_costs, device=self.device)
        extras["costs"] = cost
        if critic_obs is not None:
            extras["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            extras["next_critic_observations"] = next_critic_obs
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

            # Calculate error (delta): positive means violation
            delta = current_cost - cost_limit

            # === Integral term (I) with anti-windup ===
            self.pid_i[cost_idx] = max(0.0, self.pid_i[cost_idx] + delta * self.ki)
            if self.diff_norm:
                self.pid_i[cost_idx] = max(0.0, min(1.0, self.pid_i[cost_idx]))
            else:
                # Prevent integral windup by capping at pid_i_max
                self.pid_i[cost_idx] = min(self.pid_i[cost_idx], self.pid_i_max)

            # === Proportional term (P) with EMA smoothing ===
            alpha_p = self.pid_delta_p_ema_alpha
            self.delta_p[cost_idx] = alpha_p * self.delta_p[cost_idx] + (1 - alpha_p) * delta

            # === Derivative term (D) with EMA smoothing and delay ===
            alpha_d = self.pid_delta_d_ema_alpha
            self.cost_ema[cost_idx] = alpha_d * self.cost_ema[cost_idx] + (1 - alpha_d) * current_cost

            if len(self.cost_delay_queue[cost_idx]) > 0:
                pid_d = max(0.0, self.cost_ema[cost_idx] - self.cost_delay_queue[cost_idx][0])
            else:
                pid_d = 0.0

            # === Compute PID output ===
            pid_output = self.kp * self.delta_p[cost_idx] + self.pid_i[cost_idx] + self.kd * pid_d

            # Apply constraints
            self.lambdas[cost_idx] = max(0.0, pid_output)

            if self.diff_norm:
                self.lambdas[cost_idx] = min(1.0, self.lambdas[cost_idx])
            else:
                self.lambdas[cost_idx] = min(self.lambdas[cost_idx], self.lambda_max)

            # Update delay queue
            self.cost_delay_queue[cost_idx].append(self.cost_ema[cost_idx])

    def update(self, current_costs: list[float] | None = None, obs_normalizer=None, critic_obs_normalizer=None, reward_normalizer=None) -> dict[str, float]:
        """Perform Safe SAC update step.

        Args:
            current_costs: Current mean episode costs for Lagrangian update.
                          If None, PID update is skipped.
            obs_normalizer: Optional normalizer for actor observations.
                Applied at sample time (FastSAC-style) to keep raw obs in buffer.
            critic_obs_normalizer: Optional normalizer for critic observations.

        Returns:
            Dictionary containing loss values for logging.
        """
        if self.storage is None or len(self.storage) < self.batch_size:
            return {"critic": 0.0, "cost_critic": 0.0, "actor": 0.0, "alpha": 0.0}

        # Update Lagrangian multipliers if costs provided
        if current_costs is not None:
            self.update_lagrangian_multipliers(current_costs)

        total_critic_loss = 0.0
        total_cost_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0

        actor_updates = 0

        for update_idx in range(self.num_updates_per_step):
            # Sample batch (raw obs from buffer)
            batch = self.storage.sample(self.batch_size)
            obs = batch["observations"]
            critic_obs = batch.get("critic_observations", obs)
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_observations"]
            next_critic_obs = batch.get("next_critic_observations", next_obs)
            costs = batch.get("costs", torch.zeros(self.batch_size, self.num_costs, device=self.device))

            # Normalize at sample time (FastSAC-style, eval mode to avoid updating stats)
            if obs_normalizer is not None:
                with torch.no_grad():
                    obs = (obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
                    next_obs = (next_obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
            if critic_obs_normalizer is not None:
                with torch.no_grad():
                    critic_obs = (critic_obs - critic_obs_normalizer._mean) / (critic_obs_normalizer._std + critic_obs_normalizer.eps)
                    next_critic_obs = (next_critic_obs - critic_obs_normalizer._mean) / (critic_obs_normalizer._std + critic_obs_normalizer.eps)
            # Normalize rewards by running std (don't shift mean)
            if reward_normalizer is not None:
                with torch.no_grad():
                    rewards = rewards / (reward_normalizer._std + reward_normalizer.eps)

            # Update reward critic
            critic_loss = self._update_reward_critic(critic_obs, actions, rewards, dones, next_critic_obs)
            total_critic_loss += critic_loss

            # Update cost critic
            cost_critic_loss = self._update_cost_critic(critic_obs, actions, costs, dones, next_critic_obs)
            total_cost_critic_loss += cost_critic_loss

            # Update actor and alpha
            if update_idx % self.policy_frequency == 0:
                actor_loss, alpha_loss = self._update_actor_and_alpha(obs, critic_obs)
                total_actor_loss += actor_loss
                total_alpha_loss += alpha_loss
                actor_updates += 1

            # Soft update target networks
            self.policy.soft_update_targets(self.tau)

        # Average losses
        num_updates = self.num_updates_per_step
        actor_denominator = actor_updates if actor_updates > 0 else 1
        return {
            "critic": total_critic_loss / num_updates,
            "cost_critic": total_cost_critic_loss / num_updates,
            "actor": total_actor_loss / actor_denominator,
            "alpha": total_alpha_loss / actor_denominator,
        }

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Update reward Q-networks with standard SAC loss.

        Args:
            obs: Current observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Done flags.
            next_obs: Next observations.

        Returns:
            Reward critic loss value.
        """
        with torch.no_grad():
            # Sample next actions and compute log probs
            next_actions, next_log_prob = self.policy.sample_with_log_prob(next_obs)

            # Compute target Q-values (min of twin critics)
            q1_target, q2_target = self.policy.evaluate_q_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # Soft Bellman backup: Q_target = r + γ * (1 - d) * (min Q_target - α * log π)
            target_q = rewards + self.gamma * (1 - dones) * (q_target - self.alpha.detach() * next_log_prob)

        # Compute current Q-values
        q1, q2 = self.policy.evaluate_q(obs, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Optimize reward critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        reward_critic_params = []
        for critic in self.policy.reward_critics:
            reward_critic_params.extend(list(critic.parameters()))
        nn.utils.clip_grad_norm_(reward_critic_params, self.max_grad_norm)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Update cost Q-networks.

        Args:
            obs: Current observations.
            actions: Actions taken.
            costs: Costs received [batch_size, num_costs].
            dones: Done flags.
            next_obs: Next observations.

        Returns:
            Cost critic loss value.
        """
        with torch.no_grad():
            # Sample next actions
            next_actions, _ = self.policy.sample_with_log_prob(next_obs)

            # Compute target cost Q-values
            cost_q_target = self.policy.evaluate_cost_q_target(next_obs, next_actions)

            # Cost Bellman backup: Q_c_target = c + γ * (1 - d) * Q_c_target
            # Note: No entropy term for cost critics
            target_cost_q = costs + self.gamma * (1 - dones) * cost_q_target

        # Compute current cost Q-values
        cost_q = self.policy.evaluate_cost_q(obs, actions)

        # Cost critic loss (MSE)
        cost_critic_loss = F.mse_loss(cost_q, target_cost_q)

        # Optimize cost critics
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        cost_critic_params = []
        for critic in self.policy.cost_critics:
            cost_critic_params.extend(list(critic.parameters()))
        nn.utils.clip_grad_norm_(cost_critic_params, self.max_grad_norm)
        self.cost_critic_optimizer.step()

        return cost_critic_loss.item()

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> tuple[float, float]:
        """Update actor with safety-augmented objective.

        The actor maximizes: Q_r - α * log π - λ * Q_c
        Where:
        - Q_r is the reward Q-value
        - α * log π is the entropy regularization
        - λ * Q_c is the Lagrangian penalty for constraint violation

        Args:
            obs: Current actor observations.
            critic_obs: Current critic observations. If None, uses actor observations.

        Returns:
            Tuple of (actor_loss, alpha_loss).
        """
        # Sample actions and log probs for current policy
        actions, log_prob = self.policy.sample_with_log_prob(obs)
        critic_obs = obs if critic_obs is None else critic_obs

        # Compute reward Q-values
        q1, q2 = self.policy.evaluate_q(critic_obs, actions)
        q_min = torch.min(q1, q2)

        # Compute cost Q-values
        cost_q = self.policy.evaluate_cost_q(critic_obs, actions)  # [batch, num_costs]

        # Compute weighted cost penalty: sum(λ_i * Q_c_i)
        total_lambda = sum(self.lambdas)
        lambda_tensor = torch.tensor(self.lambdas, device=self.device, dtype=cost_q.dtype)
        cost_penalty = (cost_q * lambda_tensor).sum(dim=-1, keepdim=True)  # [batch, 1]

        # Actor loss with Lagrangian constraint handling
        # The actor maximizes: Q_r - λ * Q_c - α * log π
        # which is equivalent to minimizing: α * log π - Q_r + λ * Q_c
        #
        # With sum_norm=True (default, OmniSafe-style):
        #   Normalize ONLY the cost penalty by (1 + λ) to prevent it from dominating when λ is very large
        #   This keeps the reward signal strong while moderating the cost penalty
        #   actor_loss = α * log π - Q_r + (λ * Q_c) / (1 + λ)
        #
        # With sum_norm=False:
        #   Use raw Lagrangian without normalization (stronger cost gradients)
        #   actor_loss = α * log π - Q_r + λ * Q_c
        if self.sum_norm and total_lambda > 0:
            # Normalize only the cost penalty, not the reward Q
            normalized_cost_penalty = cost_penalty / (1.0 + total_lambda)
        else:
            # No normalization - full cost signal
            normalized_cost_penalty = cost_penalty

        normalized_q = q_min - normalized_cost_penalty
        actor_loss = (self.alpha.detach() * log_prob - normalized_q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update entropy coefficient (alpha)
        alpha_loss_value = 0.0
        if self.auto_entropy_tuning and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss_value = alpha_loss.item()

        return actor_loss.item(), alpha_loss_value

    def get_penalty_info(self) -> dict[str, Any]:
        """Get Lagrangian multiplier information for logging.

        Returns:
            Dictionary with penalty/constraint information.
        """
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
        """Get the actual action std from the stochastic actor (not the placeholder).

        Returns:
            Mean action standard deviation from the policy's log_std_head.
        """
        # Sample a dummy observation to get the actual std
        with torch.no_grad():
            if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'log_std_head'):
                # Get the log_std bias (when input is zeros, output is roughly the bias)
                # This gives a rough estimate of the policy's learned std
                dummy_obs = torch.zeros(1, self.policy.actor.backbone[0].in_features, device=self.device)
                _, log_std = self.policy.actor(dummy_obs)
                std = log_std.exp().mean().item()
                return std
        return 1.0  # Fallback

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (for multi-GPU training)."""
        if not self.is_multi_gpu:
            return

        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
