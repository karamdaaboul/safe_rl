from __future__ import annotations

from copy import deepcopy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.actor import StochasticActor
from safe_rl.modules.distributional_critic import DistributionalCritic
from safe_rl.modules.normalizer import EmpiricalNormalization
from safe_rl.utils import resolve_nn_activation


class SACActorCritic(nn.Module):
    """Actor-Critic module for Soft Actor-Critic (SAC).

    This module implements:
    - A squashed Gaussian policy (actor) with tanh activation
    - Twin Q-networks (critics) for the double Q-learning trick
    - Target networks for stable training
    - Optional distributional critics for better value estimation

    The policy outputs actions in [-1, 1] via tanh squashing.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_type: str = "stochastic",
        critic_type: str = "standard",
        num_critics: int = 2,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize SAC Actor-Critic.

        Args:
            num_actor_obs: Dimension of actor observations.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            actor_type: Type of actor - "stochastic" (default for SAC).
            critic_type: Type of critic - "standard" or "distributional".
            num_critics: Number of critic networks (default 2 for twin Q-learning).
            actor_obs_normalization: Whether to normalize actor observations.
            critic_obs_normalization: Whether to normalize critic observations.
            actor_kwargs: Actor-specific parameters.
            critic_kwargs: Critic-specific parameters.
        """
        if kwargs:
            print(
                "SACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        # Deep copy to avoid modifying original dicts
        actor_kwargs = deepcopy(actor_kwargs) if actor_kwargs is not None else {}
        critic_kwargs = deepcopy(critic_kwargs) if critic_kwargs is not None else {}

        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self.actor_type = actor_type
        self.critic_type = critic_type
        self.num_critics = num_critics

        # ==================== Actor ====================
        init_noise_std = actor_kwargs.pop("init_noise_std", 1.0)

        if actor_type == "stochastic":
            self.actor = StochasticActor(
                num_actor_obs,
                num_actions,
                **actor_kwargs,
            )
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}. SAC requires 'stochastic' actor.")

        print(f"SAC Actor: {self.actor}")

        # For compatibility with runners that expect std attribute
        self.std = nn.Parameter(torch.ones(num_actions) * init_noise_std, requires_grad=False)

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # ==================== Critics ====================
        if critic_type == "standard":
            self.is_distributional_critic = False
            hidden_dims = critic_kwargs.get("hidden_dims", [256, 256])
            activation = critic_kwargs.get("activation", "relu")

            critics = []
            for _ in range(num_critics):
                layers = []
                input_dim = num_critic_obs + num_actions
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(resolve_nn_activation(activation))
                    input_dim = hidden_dim
                layers.append(nn.Linear(input_dim, 1))
                critics.append(nn.Sequential(*layers))
            self.critics = nn.ModuleList(critics)

        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critics = nn.ModuleList(
                DistributionalCritic(
                    num_obs=num_critic_obs,
                    num_actions=num_actions,
                    **critic_kwargs,
                )
                for _ in range(num_critics)
            )
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'standard' or 'distributional'.")

        print(f"SAC Critic: {self.critics[0]}")

        # Create target networks
        self.critic_targets = nn.ModuleList(
            deepcopy(critic) for critic in self.critics
        )

        # Freeze target networks
        for target in self.critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        # For backward compatibility
        self.critic_1 = self.critics[0]
        self.critic_2 = self.critics[1] if num_critics > 1 else self.critics[0]
        self.critic_1_target = self.critic_targets[0]
        self.critic_2_target = self.critic_targets[1] if num_critics > 1 else self.critic_targets[0]

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

    @property
    def logits(self) -> torch.Tensor:
        """Return cached logits from last evaluate call (distributional critic only)."""
        if not self.is_distributional_critic:
            raise RuntimeError("logits only available for distributional critic")
        return self._logits

    @property
    def value_dist(self) -> torch.Tensor:
        """Return cached value distribution from last evaluate call (distributional critic only)."""
        if not self.is_distributional_critic:
            raise RuntimeError("value_dist only available for distributional critic")
        return self._value_dist

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
        """Evaluate Q-values for state-action pairs using both critics.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            q1: Q-values from critic 1
            q2: Q-values from critic 2
        """
        obs = self.critic_obs_normalizer(obs)

        if self.is_distributional_critic:
            logits_1 = self.critic_1(obs, actions)
            logits_2 = self.critic_2(obs, actions)
            dist_1 = self.critic_1.get_dist(logits_1)
            dist_2 = self.critic_2.get_dist(logits_2)
            q1 = self.critic_1.get_value(dist_1).unsqueeze(-1)
            q2 = self.critic_2.get_value(dist_2).unsqueeze(-1)

            # Cache for properties
            self._logits = torch.stack([logits_1, logits_2], dim=1)
            self._value_dist = dist_1
        else:
            critic_input = torch.cat([obs, actions], dim=-1)
            q1 = self.critic_1(critic_input)
            q2 = self.critic_2(critic_input)

        return q1, q2

    def evaluate_q_dist(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate distributional Q-values and return raw logits.

        Only available when critic_type='distributional'.
        """
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_dist only available for distributional critics")

        obs = self.critic_obs_normalizer(obs)
        logits_1 = self.critic_1(obs, actions)
        logits_2 = self.critic_2(obs, actions)
        return logits_1, logits_2

    def evaluate_q_target(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q-values using target networks.

        Args:
            obs: Observations [batch_size, num_critic_obs]
            actions: Actions [batch_size, num_actions]

        Returns:
            q1_target: Q-values from target critic 1
            q2_target: Q-values from target critic 2
        """
        obs = self.critic_obs_normalizer(obs)

        if self.is_distributional_critic:
            logits_1 = self.critic_1_target(obs, actions)
            logits_2 = self.critic_2_target(obs, actions)
            dist_1 = self.critic_1_target.get_dist(logits_1)
            dist_2 = self.critic_2_target.get_dist(logits_2)
            q1_target = self.critic_1_target.get_value(dist_1).unsqueeze(-1)
            q2_target = self.critic_2_target.get_value(dist_2).unsqueeze(-1)
        else:
            critic_input = torch.cat([obs, actions], dim=-1)
            q1_target = self.critic_1_target(critic_input)
            q2_target = self.critic_2_target(critic_input)

        return q1_target, q2_target

    def evaluate_q_target_dist(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate distributional Q-values from target networks.

        Only available when critic_type='distributional'.
        """
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_target_dist only available for distributional critics")

        obs = self.critic_obs_normalizer(obs)
        logits_1 = self.critic_1_target(obs, actions)
        logits_2 = self.critic_2_target(obs, actions)
        return logits_1, logits_2

    def soft_update_targets(self, tau: float) -> None:
        """Polyak averaging update for target networks.

        θ_target = τ * θ + (1 - τ) * θ_target
        """
        for critic, target in zip(self.critics, self.critic_targets):
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
