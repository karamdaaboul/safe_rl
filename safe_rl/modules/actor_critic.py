from __future__ import annotations

from copy import deepcopy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.actor import DeterministicActor, GaussianActor
from safe_rl.modules.critic import DistributionalCritic, StandardCritic
from safe_rl.modules.normalizer import EmpiricalNormalization


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_type: str = "gaussian",
        critic_type: str = "standard",
        num_critics: int = 1,
        num_costs: int = 0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        cost_critic_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize Actor-Critic.

        Args:
            num_actor_obs: Dimension of actor observations.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            actor_type: Type of actor - "deterministic" or "gaussian".
            critic_type: Type of critic - "standard" or "distributional".
            num_critics: Number of critic networks.
            num_costs: Number of constraint cost heads. If > 0, a `cost_critic` submodule is built and
                `evaluate_cost()` becomes available. The runner injects this from `len(cost_limits)`.
            actor_obs_normalization: Whether to normalize actor observations.
            critic_obs_normalization: Whether to normalize critic observations.
            actor_kwargs: Actor-specific parameters.
            critic_kwargs: Critic-specific parameters.
            cost_critic_kwargs: Cost-critic MLP parameters (hidden_dims, activation, ...). Only read when num_costs > 0.
        """
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        # Deep copy to avoid modifying original dicts
        actor_kwargs = deepcopy(actor_kwargs) if actor_kwargs is not None else {}
        critic_kwargs = deepcopy(critic_kwargs) if critic_kwargs is not None else {}
        cost_critic_kwargs = deepcopy(cost_critic_kwargs) if cost_critic_kwargs is not None else {}

        self.actor_type = actor_type
        self.critic_type = critic_type
        self.num_critics = num_critics
        self.num_costs = num_costs

        # ==================== Actor ====================
        if actor_type == "deterministic":
            self.is_deterministic_actor = True
            # Extract noise parameters for deterministic actor
            self.std_max = actor_kwargs.pop("std_max", 1.0)
            self.std_min = actor_kwargs.pop("std_min", 0.1)
            init_noise_std = actor_kwargs.pop("init_noise_std", 1.0)

            self.actor = DeterministicActor(
                num_actor_obs,
                num_actions,
                **actor_kwargs,
            )
            # For compatibility
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        elif actor_type == "gaussian":
            self.is_deterministic_actor = False
            self.actor = GaussianActor(num_actor_obs, num_actions, **actor_kwargs)
            self.distribution = None
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}. Must be 'deterministic' or 'gaussian'.")

        print(f"Actor: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # ==================== Critic ====================
        if critic_type == "standard":
            self.is_distributional_critic = False
            self.critics = nn.ModuleList(
                StandardCritic(num_obs=num_critic_obs, num_actions=0, **critic_kwargs)
                for _ in range(num_critics)
            )

        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critics = nn.ModuleList(
                DistributionalCritic(
                    num_obs=num_critic_obs,
                    num_actions=0,  # V(s) for on-policy, no actions
                    **critic_kwargs,
                )
                for _ in range(num_critics)
            )
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'standard' or 'distributional'.")

        print(f"Critic: {self.critics[0]}")

        # For backward compatibility
        self.critic = self.critics[0]

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # ==================== Cost critic (optional) ====================
        # Built only for safe RL algorithms; runner injects num_costs from len(cost_limits).
        if num_costs > 0:
            self.cost_critic = StandardCritic(
                num_obs=num_critic_obs,
                num_actions=0,
                output_dim=num_costs,
                **cost_critic_kwargs,
            )
            print(f"Cost critic: {self.cost_critic}")
        else:
            self.cost_critic = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self.is_deterministic_actor:
            raise RuntimeError("action_mean not available for deterministic actor")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.is_deterministic_actor:
            return self.std
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.is_deterministic_actor:
            raise RuntimeError("entropy not available for deterministic actor")
        return self.distribution.entropy().sum(dim=-1)

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

    def update_distribution(self, observations: torch.Tensor) -> None:
        """Update the action distribution (for Gaussian actor only)."""
        if self.is_deterministic_actor:
            raise RuntimeError("update_distribution not available for deterministic actor")
        self.distribution = self.actor.distribution(observations)

    def act(self, observations: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample actions from the policy."""
        observations = self.actor_obs_normalizer(observations)

        if self.is_deterministic_actor:
            action = self.actor(observations)
            noise = torch.randn_like(action) * self.std
            return action + noise
        else:
            self.update_distribution(observations)
            return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Get deterministic actions for inference."""
        observations = self.actor_obs_normalizer(observations)

        if self.is_deterministic_actor:
            return self.actor(observations)
        else:
            return self.actor(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions (Gaussian actor only)."""
        if self.is_deterministic_actor:
            raise RuntimeError("get_actions_log_prob not available for deterministic actor")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the value function.

        Args:
            critic_observations: Observations for the critic [batch_size, num_critic_obs]

        Returns:
            value: Value estimates [batch_size, 1] or [batch_size, num_critics] for distributional
        """
        critic_observations = self.critic_obs_normalizer(critic_observations)

        if self.is_distributional_critic:
            # For V(s), we use dummy empty actions
            dummy_actions = torch.zeros(
                critic_observations.shape[0], 0, device=critic_observations.device
            )
            out = torch.stack(
                [critic(critic_observations, dummy_actions) for critic in self.critics], dim=1
            )
            self._logits = out
            self._value_dist = self.critics[0].get_dist(self._logits[:, 0])
            value = self.critics[0].get_value(self._value_dist)
        else:
            if self.num_critics == 1:
                value = self.critics[0](critic_observations)
            else:
                value = torch.stack(
                    [critic(critic_observations) for critic in self.critics], dim=1
                )
        return value

    def evaluate_cost(self, critic_observations: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the cost-value function (safe RL only).

        Returns:
            cost_value: [batch_size, num_costs].
        """
        if self.cost_critic is None:
            raise RuntimeError("evaluate_cost requires num_costs > 0 (no cost_critic was built)")
        critic_observations = self.critic_obs_normalizer(critic_observations)
        return self.cost_critic(critic_observations)

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        """Update observation normalizers."""
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Legacy checkpoints (from the old ActorCritic / ActorCriticCost classes) stored the reward
        critic under ``critic.network.*``. The unified class stores it under ``critics.0.network.*``
        (via ``nn.ModuleList`` to support ``num_critics > 1``). Remap old keys on the fly so
        pre-refactor checkpoints load without manual conversion.
        """
        has_new = any(k.startswith("critics.0.") for k in state_dict)
        has_old = any(k.startswith("critic.network.") for k in state_dict)
        if has_old and not has_new:
            # self.critic = self.critics[0] registers both paths, so state_dict() emits both.
            # Mirror the legacy critic.* keys under critics.0.* so strict load is satisfied.
            remap = {}
            for k, v in state_dict.items():
                if k.startswith("critic.network."):
                    remap[k.replace("critic.network.", "critics.0.network.", 1)] = v
            state_dict = {**state_dict, **remap}
        super().load_state_dict(state_dict, strict=strict)
        return True
