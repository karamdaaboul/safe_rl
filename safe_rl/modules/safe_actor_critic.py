from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from safe_rl.modules.critic import DistributionalCritic, QuantileCritic
from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic


class SafeActorCritic(SafeSACActorCritic):
    """Safe actor-critic accepting either standard or distributional reward critics.

    Same interface as :class:`SafeSACActorCritic` (twin reward critics, cost critics,
    target networks, tanh-squashed Gaussian actor) with ``critic_type="distributional"``
    additionally supported, so constrained algorithms such as :class:`~safe_rl.algorithms.CVPO`
    can use the C51 critic stack that :class:`~safe_rl.algorithms.MPO` benefits from.

    Cost critics stay scalar in both modes. The constrained E-step needs ``Q_c`` as a plain
    expectation to form ``exp((Q_r - lambda Q_c)/eta)``, and the cost scale is set by the
    episodic budget rather than the reward support, so a shared categorical support would
    not fit both. ``is_distributional_critic`` therefore refers to the reward critics only.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        critic_type: str = "standard",
        critic_kwargs: dict[str, Any] | None = None,
        num_reward_critics: int = 2,
        cost_critic_type: str = "standard",
        cost_critic_kwargs: dict[str, Any] | None = None,
        num_costs: int = 1,
        num_cost_critics: int = 1,
        **kwargs: Any,
    ) -> None:
        if critic_type not in ("standard", "distributional", "quantile"):
            raise ValueError(
                f"critic_type must be 'standard', 'distributional' or 'quantile', got: {critic_type!r}"
            )
        if cost_critic_type not in ("standard", "distributional", "quantile"):
            raise ValueError(
                f"cost_critic_type must be 'standard', 'distributional' or 'quantile', "
                f"got: {cost_critic_type!r}"
            )
        if cost_critic_type in ("distributional", "quantile") and num_costs != 1:
            # One distribution head per critic; m > 1 would need m separate heads.
            raise ValueError(
                f"cost_critic_type={cost_critic_type!r} supports a single constraint, got num_costs={num_costs}."
            )
        distributional = critic_type == "distributional"
        cost_distributional = cost_critic_type == "distributional"
        quantile = critic_type == "quantile"
        cost_quantile = cost_critic_type == "quantile"

        # The base class builds standard reward critics; for the distributional case they are
        # replaced below (they are small MLPs, so the discarded build is negligible).
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            critic_type="standard",
            critic_kwargs=None if (distributional or quantile) else critic_kwargs,
            num_reward_critics=num_reward_critics,
            cost_critic_kwargs=None if (cost_distributional or cost_quantile) else cost_critic_kwargs,
            num_costs=num_costs,
            num_cost_critics=num_cost_critics,
            **kwargs,
        )
        self.critic_type = critic_type
        self.cost_critic_type = cost_critic_type
        self.is_distributional_cost_critic = cost_distributional
        # Separate flags, deliberately not folded into `is_distributional_*`: those select the
        # categorical cross-entropy loss, and a quantile critic routed there would be trained
        # against a projection its output does not represent.
        self.is_quantile_critic = quantile
        self.is_quantile_cost_critic = cost_quantile

        if distributional:
            self._replace_reward_critics_with_distributional(
                num_critic_obs, num_actions, num_reward_critics, deepcopy(critic_kwargs) or {}
            )
        if cost_distributional:
            self._replace_cost_critics_with_distributional(
                num_critic_obs, num_actions, num_cost_critics, deepcopy(cost_critic_kwargs) or {}
            )
        if quantile:
            self._replace_reward_critics_with_quantile(
                num_critic_obs, num_actions, num_reward_critics, deepcopy(critic_kwargs) or {}
            )
        if cost_quantile:
            self._replace_cost_critics_with_quantile(
                num_critic_obs, num_actions, num_cost_critics, deepcopy(cost_critic_kwargs) or {}
            )

    def _replace_reward_critics_with_distributional(
        self, num_critic_obs: int, num_actions: int, num_reward_critics: int, critic_kwargs: dict[str, Any]
    ) -> None:
        critic_kwargs.setdefault("num_atoms", 101)
        critic_kwargs.setdefault("v_min", -10.0)
        critic_kwargs.setdefault("v_max", 10.0)

        self.reward_critics = nn.ModuleList(
            DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
            for _ in range(num_reward_critics)
        )
        self.reward_critic_targets = nn.ModuleList(deepcopy(c) for c in self.reward_critics)
        for target in self.reward_critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        self.critic_1 = self.reward_critics[0]
        self.critic_2 = self.reward_critics[1] if num_reward_critics > 1 else self.reward_critics[0]
        self.critic_1_target = self.reward_critic_targets[0]
        self.critic_2_target = self.reward_critic_targets[1] if num_reward_critics > 1 else self.reward_critic_targets[0]

        self.is_distributional_critic = True
        print(f"Safe Reward Critic (distributional): {self.reward_critics[0]}")

    def _replace_cost_critics_with_distributional(
        self, num_critic_obs: int, num_actions: int, num_cost_critics: int, cost_critic_kwargs: dict[str, Any]
    ) -> None:
        """Categorical (C51) cost critics.

        The support is one-sided by construction: cost is non-negative, so ``v_min`` defaults
        to 0 and the critic cannot emit a negative ``Q_c`` at all — the invariant that
        ``cost_critic_nonneg`` patches on the scalar head is structural here. ``v_max`` must
        cover the discounted cost return; on SafetyPointGoal1 the measured MC maximum is ~38.
        """
        cost_critic_kwargs.setdefault("num_atoms", 101)
        cost_critic_kwargs.setdefault("v_min", 0.0)
        cost_critic_kwargs.setdefault("v_max", 50.0)

        self.cost_critics = nn.ModuleList(
            DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **cost_critic_kwargs)
            for _ in range(num_cost_critics)
        )
        self.cost_critic_targets = nn.ModuleList(deepcopy(c) for c in self.cost_critics)
        for target in self.cost_critic_targets:
            for param in target.parameters():
                param.requires_grad = False
        print(f"Safe Cost Critic (distributional): {self.cost_critics[0]}")

    def _replace_reward_critics_with_quantile(
        self, num_critic_obs: int, num_actions: int, num_reward_critics: int, critic_kwargs: dict[str, Any]
    ) -> None:
        critic_kwargs.setdefault("n_quantiles", 64)
        critic_kwargs.setdefault("nonneg", False)  # reward returns are signed

        self.reward_critics = nn.ModuleList(
            QuantileCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
            for _ in range(num_reward_critics)
        )
        self.reward_critic_targets = nn.ModuleList(deepcopy(c) for c in self.reward_critics)
        for target in self.reward_critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        self.critic_1 = self.reward_critics[0]
        self.critic_2 = self.reward_critics[1] if num_reward_critics > 1 else self.reward_critics[0]
        self.critic_1_target = self.reward_critic_targets[0]
        self.critic_2_target = self.reward_critic_targets[1] if num_reward_critics > 1 else self.reward_critic_targets[0]

        print(f"Safe Reward Critic (quantile): {self.reward_critics[0]}")

    def _replace_cost_critics_with_quantile(
        self, num_critic_obs: int, num_actions: int, num_cost_critics: int, cost_critic_kwargs: dict[str, Any]
    ) -> None:
        """Quantile (QR-DQN) cost critics.

        ``nonneg`` defaults to True here, mirroring the categorical cost critic's one-sided
        support (``v_min=0``): cost is non-negative, so a softplus head makes a negative
        ``Q_c`` structurally impossible rather than merely improbable. That property is what
        the repo credits for the constraint-satisfaction improvement, so it carries over.
        """
        cost_critic_kwargs.setdefault("n_quantiles", 64)
        cost_critic_kwargs.setdefault("nonneg", True)

        self.cost_critics = nn.ModuleList(
            QuantileCritic(num_obs=num_critic_obs, num_actions=num_actions, **cost_critic_kwargs)
            for _ in range(num_cost_critics)
        )
        self.cost_critic_targets = nn.ModuleList(deepcopy(c) for c in self.cost_critics)
        for target in self.cost_critic_targets:
            for param in target.parameters():
                param.requires_grad = False
        print(f"Safe Cost Critic (quantile): {self.cost_critics[0]}")

    @property
    def _cost_critic_has_distribution(self) -> bool:
        """True when the cost critic emits a distribution rather than a scalar.

        Both the categorical and quantile critics scalarize through the same
        ``get_value(get_dist(out))`` call -- for the quantile critic ``get_dist`` is the
        identity and ``get_value`` the mean over quantiles -- so every consumer below needs
        only this one predicate, not a per-type branch.
        """
        return self.is_distributional_cost_critic or self.is_quantile_cost_critic

    @property
    def _reward_critic_has_distribution(self) -> bool:
        """True when the reward critics emit a distribution rather than a scalar."""
        return self.is_distributional_critic or self.is_quantile_critic

    def _scalar_qc(self, critic: nn.Module, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Expected cost Q from a cost critic. Shape [batch, num_costs]."""
        out = critic(obs, actions)
        if not self._cost_critic_has_distribution:
            return self._cost_head(out)
        return critic.get_value(critic.get_dist(out)).unsqueeze(-1)

    def evaluate_cost_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self._cost_critic_has_distribution:
            return super().evaluate_cost_q(obs, actions)
        obs = self.critic_obs_normalizer(obs)
        if self.num_cost_critics == 1:
            return self._scalar_qc(self.cost_critics[0], obs, actions)
        qs = [self._scalar_qc(c, obs, actions) for c in self.cost_critics]
        return torch.stack(qs, dim=0).mean(dim=0)

    def evaluate_cost_q_target(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self._cost_critic_has_distribution:
            return super().evaluate_cost_q_target(obs, actions)
        obs = self.critic_obs_normalizer(obs)
        if self.num_cost_critics == 1:
            return self._scalar_qc(self.cost_critic_targets[0], obs, actions)
        qs = [self._scalar_qc(c, obs, actions) for c in self.cost_critic_targets]
        return torch.stack(qs, dim=0).mean(dim=0)

    def _scalar_q(self, critic: nn.Module, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Expected Q from a critic, for all critic types. Shape [batch, 1]."""
        out = critic(obs, actions)
        if not self._reward_critic_has_distribution:
            return out
        return critic.get_value(critic.get_dist(out)).unsqueeze(-1)

    def evaluate_q(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(obs)
        return self._scalar_q(self.critic_1, obs, actions), self._scalar_q(self.critic_2, obs, actions)

    def evaluate_q_target(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(obs)
        return (
            self._scalar_q(self.critic_1_target, obs, actions),
            self._scalar_q(self.critic_2_target, obs, actions),
        )
