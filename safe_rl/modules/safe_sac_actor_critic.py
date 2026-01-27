# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.actor import StochasticActor
from safe_rl.modules.normalizer import EmpiricalNormalization
from safe_rl.utils import resolve_nn_activation


class SafeSACActorCritic(nn.Module):
    """Actor-Critic module for Safe Soft Actor-Critic (Safe SAC).

    This module extends the standard SAC architecture with cost critics for
    safe reinforcement learning with constraints. It implements:
    - A squashed Gaussian policy (actor) with tanh activation
    - Twin Q-networks (reward critics) for the double Q-learning trick
    - Cost Q-networks for estimating constraint violations
    - Target networks for stable training

    The policy outputs actions in [-1, 1] via tanh squashing.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_costs: int = 1,
        actor_type: str = "stochastic",
        critic_type: str = "standard",
        num_reward_critics: int = 2,
        num_cost_critics: int = 1,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        cost_critic_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize Safe SAC Actor-Critic.

        Args:
            num_actor_obs: Dimension of actor observations.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            num_costs: Number of cost constraints to estimate.
            actor_type: Type of actor - "stochastic" (default for SAC).
            critic_type: Type of critic - "standard" only for Safe SAC.
            num_reward_critics: Number of reward critic networks (default 2 for twin Q-learning).
            num_cost_critics: Number of cost critic networks (default 1).
            actor_obs_normalization: Whether to normalize actor observations.
            critic_obs_normalization: Whether to normalize critic observations.
            actor_kwargs: Actor-specific parameters.
            critic_kwargs: Reward critic-specific parameters.
            cost_critic_kwargs: Cost critic-specific parameters.
        """
        super().__init__()

        # Deep copy to avoid modifying original dicts
        actor_kwargs = deepcopy(actor_kwargs) if actor_kwargs is not None else {}
        critic_kwargs = deepcopy(critic_kwargs) if critic_kwargs is not None else {}

        # Handle top-level config keys for backward compatibility
        # Move actor_hidden_dims -> actor_kwargs['hidden_dims']
        if "actor_hidden_dims" in kwargs and "hidden_dims" not in actor_kwargs:
            actor_kwargs["hidden_dims"] = kwargs.pop("actor_hidden_dims")
        if "critic_hidden_dims" in kwargs and "hidden_dims" not in critic_kwargs:
            critic_kwargs["hidden_dims"] = kwargs.pop("critic_hidden_dims")
        if "activation" in kwargs:
            if "activation" not in actor_kwargs:
                actor_kwargs["activation"] = kwargs.get("activation")
            if "activation" not in critic_kwargs:
                critic_kwargs["activation"] = kwargs.pop("activation")
        if "log_std_min" in kwargs and "log_std_min" not in actor_kwargs:
            actor_kwargs["log_std_min"] = kwargs.pop("log_std_min")
        if "log_std_max" in kwargs and "log_std_max" not in actor_kwargs:
            actor_kwargs["log_std_max"] = kwargs.pop("log_std_max")

        if kwargs:
            print(
                "SafeSACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        cost_critic_kwargs = deepcopy(cost_critic_kwargs) if cost_critic_kwargs is not None else {}

        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self.num_costs = num_costs
        self.actor_type = actor_type
        self.critic_type = critic_type
        self.num_reward_critics = num_reward_critics
        self.num_cost_critics = num_cost_critics

        # ==================== Actor ====================
        init_noise_std = actor_kwargs.pop("init_noise_std", 1.0)

        if actor_type == "stochastic":
            self.actor = StochasticActor(
                num_actor_obs,
                num_actions,
                **actor_kwargs,
            )
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}. Safe SAC requires 'stochastic' actor.")

        print(f"Safe SAC Actor: {self.actor}")

        # For compatibility with runners that expect std attribute
        self.std = nn.Parameter(torch.ones(num_actions) * init_noise_std, requires_grad=False)

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # ==================== Reward Critics (Twin Q-networks) ====================
        if critic_type != "standard":
            raise ValueError(f"Safe SAC only supports 'standard' critic_type, got: {critic_type}")

        self.is_distributional_critic = False
        hidden_dims = critic_kwargs.get("hidden_dims", [256, 256])
        activation = critic_kwargs.get("activation", "relu")
        activation_fn = resolve_nn_activation(activation)

        reward_critics = []
        for _ in range(num_reward_critics):
            layers = []
            input_dim = num_critic_obs + num_actions
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activation_fn)
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            reward_critics.append(nn.Sequential(*layers))
        self.reward_critics = nn.ModuleList(reward_critics)

        print(f"Safe SAC Reward Critic: {self.reward_critics[0]}")

        # Create target networks for reward critics
        self.reward_critic_targets = nn.ModuleList(
            deepcopy(critic) for critic in self.reward_critics
        )

        # Freeze target networks
        for target in self.reward_critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        # For backward compatibility
        self.critic_1 = self.reward_critics[0]
        self.critic_2 = self.reward_critics[1] if num_reward_critics > 1 else self.reward_critics[0]
        self.critic_1_target = self.reward_critic_targets[0]
        self.critic_2_target = self.reward_critic_targets[1] if num_reward_critics > 1 else self.reward_critic_targets[0]

        # ==================== Cost Critics ====================
        cost_hidden_dims = cost_critic_kwargs.get("hidden_dims", hidden_dims)
        cost_activation = cost_critic_kwargs.get("activation", activation)
        cost_activation_fn = resolve_nn_activation(cost_activation)

        cost_critics = []
        for _ in range(num_cost_critics):
            layers = []
            input_dim = num_critic_obs + num_actions
            for hidden_dim in cost_hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(cost_activation_fn)
                input_dim = hidden_dim
            # Output num_costs values (one for each constraint)
            layers.append(nn.Linear(input_dim, num_costs))
            cost_critics.append(nn.Sequential(*layers))
        self.cost_critics = nn.ModuleList(cost_critics)

        print(f"Safe SAC Cost Critic (num_costs={num_costs}): {self.cost_critics[0]}")

        # Create target networks for cost critics
        self.cost_critic_targets = nn.ModuleList(
            deepcopy(critic) for critic in self.cost_critics
        )

        # Freeze cost target networks
        for target in self.cost_critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset method for compatibility with recurrent policies."""
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_std(self) -> torch.Tensor:
        """Return action standard deviation for logging compatibility."""
        return self.std

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions from the policy.

        Args:
            obs: Observations [batch_size, num_actor_obs]
            deterministic: If True, return mean action without sampling.

        Returns:
            actions: Squashed actions in [-1, 1] [batch_size, num_actions]
        """
        obs = self.actor_obs_normalizer(obs)
        return self.actor.act(obs, deterministic=deterministic)

    def act_with_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions with exploration noise (used during training)."""
        return self.act(obs, deterministic=False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference/evaluation."""
        return self.act(obs, deterministic=True)

    def sample_with_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.

        Args:
            obs: Observations [batch_size, num_actor_obs]

        Returns:
            actions: Squashed actions in [-1, 1] [batch_size, num_actions]
            log_prob: Log probability of actions [batch_size, 1]
        """
        obs = self.actor_obs_normalizer(obs)
        return self.actor.sample(obs)

    def evaluate_q(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q-values for state-action pairs using both reward critics.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            q1: Q-values from reward critic 1
            q2: Q-values from reward critic 2
        """
        obs = self.critic_obs_normalizer(obs)
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
            q1_target: Q-values from target reward critic 1
            q2_target: Q-values from target reward critic 2
        """
        obs = self.critic_obs_normalizer(obs)
        critic_input = torch.cat([obs, actions], dim=-1)
        q1_target = self.critic_1_target(critic_input)
        q2_target = self.critic_2_target(critic_input)
        return q1_target, q2_target

    def evaluate_cost_q(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate cost Q-values for state-action pairs.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            cost_q: Cost Q-values [batch_size, num_costs]
        """
        obs = self.critic_obs_normalizer(obs)
        critic_input = torch.cat([obs, actions], dim=-1)
        # Use first cost critic (or average if multiple)
        if self.num_cost_critics == 1:
            return self.cost_critics[0](critic_input)
        else:
            cost_qs = [critic(critic_input) for critic in self.cost_critics]
            return torch.stack(cost_qs, dim=0).mean(dim=0)

    def evaluate_cost_q_target(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate cost Q-values using target networks.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            cost_q_target: Cost Q-values from target [batch_size, num_costs]
        """
        obs = self.critic_obs_normalizer(obs)
        critic_input = torch.cat([obs, actions], dim=-1)
        # Use first cost critic target (or average if multiple)
        if self.num_cost_critics == 1:
            return self.cost_critic_targets[0](critic_input)
        else:
            cost_qs = [target(critic_input) for target in self.cost_critic_targets]
            return torch.stack(cost_qs, dim=0).mean(dim=0)

    def soft_update_targets(self, tau: float) -> None:
        """Polyak averaging update for all target networks.

        θ_target = τ * θ + (1 - τ) * θ_target
        """
        # Update reward critic targets
        for critic, target in zip(self.reward_critics, self.reward_critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update cost critic targets
        for critic, target in zip(self.cost_critics, self.cost_critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_random_action(self, num_envs: int) -> torch.Tensor:
        """Sample random actions for initial exploration."""
        device = next(self.parameters()).device
        return torch.rand(num_envs, self.num_actions, device=device) * 2 - 1

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        """Update observation normalizers."""
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)
