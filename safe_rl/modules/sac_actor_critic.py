# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

from safe_rl.utils import resolve_nn_activation


class SACActorCritic(nn.Module):
    """Actor-Critic module for Soft Actor-Critic (SAC).

    This module implements:
    - A squashed Gaussian policy (actor) with tanh activation
    - Twin Q-networks (critics) for the double Q-learning trick
    - Target networks for stable training

    The policy outputs actions in [-1, 1] via tanh squashing.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256],
        critic_hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        """Initialize SAC Actor-Critic.

        Args:
            num_actor_obs: Dimension of actor observations.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            actor_hidden_dims: Hidden layer dimensions for actor.
            critic_hidden_dims: Hidden layer dimensions for each critic.
            activation: Activation function name.
            log_std_min: Minimum log standard deviation.
            log_std_max: Maximum log standard deviation.
            init_noise_std: Initial noise standard deviation (unused, for compatibility).
        """
        if kwargs:
            print(
                "SACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.num_actions = num_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        activation_fn = resolve_nn_activation(activation)

        # Actor network (outputs mean and log_std)
        actor_layers = []
        actor_input_dim = num_actor_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(actor_input_dim, hidden_dim))
            actor_layers.append(activation_fn)
            actor_input_dim = hidden_dim
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(actor_input_dim, num_actions)
        self.actor_log_std = nn.Linear(actor_input_dim, num_actions)

        # Twin Q-networks (critics)
        # Q1
        critic1_layers = []
        critic_input_dim = num_critic_obs + num_actions
        for hidden_dim in critic_hidden_dims:
            critic1_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic1_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        critic1_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic_1 = nn.Sequential(*critic1_layers)

        # Q2
        critic2_layers = []
        critic_input_dim = num_critic_obs + num_actions
        for hidden_dim in critic_hidden_dims:
            critic2_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic2_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        critic2_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic_2 = nn.Sequential(*critic2_layers)

        # Target networks (initialized as copies)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Freeze target networks (they are updated via polyak averaging)
        for param in self.critic_1_target.parameters():
            param.requires_grad = False
        for param in self.critic_2_target.parameters():
            param.requires_grad = False

        # For compatibility with runners that expect std attribute
        self.std = nn.Parameter(torch.ones(num_actions) * init_noise_std, requires_grad=False)

        print(f"SAC Actor backbone: {self.actor_backbone}")
        print(f"SAC Critic 1: {self.critic_1}")
        print(f"SAC Critic 2: {self.critic_2}")

    def reset(self, dones=None):
        """Reset method for compatibility with recurrent policies."""
        pass

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor to get mean and log_std.

        Args:
            obs: Observations [batch_size, num_actor_obs]

        Returns:
            mean: Action mean [batch_size, num_actions]
            log_std: Action log standard deviation [batch_size, num_actions]
        """
        features = self.actor_backbone(obs)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions from the policy.

        Args:
            obs: Observations [batch_size, num_actor_obs]
            deterministic: If True, return mean action without sampling.

        Returns:
            actions: Squashed actions in [-1, 1] [batch_size, num_actions]
        """
        mean, log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        std = log_std.exp()
        dist = Normal(mean, std)
        # Reparameterization trick
        x = dist.rsample()
        actions = torch.tanh(x)

        return actions

    def act_with_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions with exploration noise (used during training).

        This is the same as act() with deterministic=False.
        """
        return self.act(obs, deterministic=False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference/evaluation."""
        return self.act(obs, deterministic=True)

    def sample_with_log_prob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.

        This is used for policy optimization in SAC.

        Args:
            obs: Observations [batch_size, num_actor_obs]

        Returns:
            actions: Squashed actions in [-1, 1] [batch_size, num_actions]
            log_prob: Log probability of actions [batch_size, 1]
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparameterization trick
        x = dist.rsample()
        actions = torch.tanh(x)

        # Compute log probability with correction for tanh squashing
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(uᵢ))
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        # Squashing correction (numerically stable version)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - x - nn.functional.softplus(-2 * x))).sum(dim=-1, keepdim=True)

        return actions, log_prob

    def evaluate_q(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q-values for state-action pairs using both critics.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            q1: Q-values from critic 1 [batch_size, 1]
            q2: Q-values from critic 2 [batch_size, 1]
        """
        critic_input = torch.cat([obs, actions], dim=-1)
        q1 = self.critic_1(critic_input)
        q2 = self.critic_2(critic_input)
        return q1, q2

    def evaluate_q_target(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q-values using target networks.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            q1_target: Q-values from target critic 1 [batch_size, 1]
            q2_target: Q-values from target critic 2 [batch_size, 1]
        """
        critic_input = torch.cat([obs, actions], dim=-1)
        q1_target = self.critic_1_target(critic_input)
        q2_target = self.critic_2_target(critic_input)
        return q1_target, q2_target

    def soft_update_targets(self, tau: float) -> None:
        """Polyak averaging update for target networks.

        θ_target = τ * θ + (1 - τ) * θ_target

        Args:
            tau: Interpolation factor (typically small, e.g., 0.005)
        """
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_random_action(self, num_envs: int) -> torch.Tensor:
        """Sample random actions for initial exploration.

        Args:
            num_envs: Number of environments.

        Returns:
            Random actions in [-1, 1] [num_envs, num_actions]
        """
        device = next(self.parameters()).device
        return torch.rand(num_envs, self.num_actions, device=device) * 2 - 1

    @property
    def action_std(self) -> torch.Tensor:
        """Return action standard deviation for logging compatibility."""
        return self.std
