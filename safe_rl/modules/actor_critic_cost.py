from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from safe_rl.modules.actor_critic import ActorCritic
from safe_rl.modules.critic import CategoricalCostCritic, HLGaussCostCritic, StandardCritic


class ActorCriticCost(ActorCritic):
    """ActorCritic with a state-only cost-value head V_C(s) for constrained (safe) RL.

    Every on-policy safe RL algorithm (P3O, PPOL-PID, CPO, PCPO, CUP, FOCOPS, PCRPO, FPPO,
    RCPPO) needs a cost critic to build the cost advantages its constraint machinery acts on.
    Splitting it into its own class means the config states which one it is: plain
    ``ActorCritic`` is reward-only, ``ActorCriticCost`` carries the constraint head. The
    algorithms can then rely on ``cost_critic`` existing rather than testing a ``None``
    sentinel that used to make every guard vacuously true.

    ``num_costs`` is injected by the runner from ``len(cost_limits)`` and must be > 0.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_costs: int = 1,
        cost_critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Actor-Critic with a cost critic.

        Args:
            num_actor_obs: Dimension of actor observations.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            num_costs: Number of constraint cost heads. The runner injects this from
                `len(cost_limits)`. Must be > 0.
            cost_critic_kwargs: Cost-critic MLP parameters (loss_type, hidden_dims, activation, ...).
            **kwargs: Forwarded to `ActorCritic` (actor/critic type, normalization, ...).
        """
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        if num_costs <= 0:
            raise ValueError(
                f"ActorCriticCost requires num_costs > 0, got {num_costs}. Use it with a safe RL "
                "algorithm and cost_limits set so the runner injects num_costs; for reward-only "
                "training use ActorCritic instead."
            )

        # Deep copy to avoid modifying original dicts
        cost_critic_kwargs = deepcopy(cost_critic_kwargs) if cost_critic_kwargs is not None else {}

        self.num_costs = num_costs

        # ==================== Cost critic ====================
        # Shares the reward critic's observation normalizer and input width, so it sees exactly
        # the same (normalized) critic observations as V_R(s).
        loss_type = cost_critic_kwargs.pop("loss_type", "mse")
        self.cost_critic_loss_type = loss_type
        if loss_type == "mse":
            self.cost_critic = StandardCritic(
                num_obs=num_critic_obs,
                num_actions=0,
                output_dim=num_costs,
                **cost_critic_kwargs,
            )
        elif loss_type == "hlgauss":
            self.cost_critic = HLGaussCostCritic(
                num_obs=num_critic_obs,
                num_costs=num_costs,
                **cost_critic_kwargs,
            )
        elif loss_type == "categorical":
            self.cost_critic = CategoricalCostCritic(
                num_obs=num_critic_obs,
                num_costs=num_costs,
                **cost_critic_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown cost_critic loss_type: {loss_type!r}. Must be 'mse', 'hlgauss' or 'categorical'."
            )
        print(f"Cost critic ({loss_type}): {self.cost_critic}")

    @property
    def is_distributional_cost_critic(self) -> bool:
        """True for classification-based cost critics (HL-Gauss / categorical), which output
        per-cost logits over a fixed support instead of a scalar and are trained with
        cross-entropy. Callers use this to select the distributional loss/decoding path."""
        return self.cost_critic_loss_type in ("hlgauss", "categorical")

    @property
    def cost_logits(self) -> torch.Tensor:
        """Return cached cost logits from last evaluate_cost call (HL-Gauss cost critic only)."""
        if self.cost_critic_loss_type != "hlgauss":
            raise RuntimeError("cost_logits only available when cost_critic loss_type is 'hlgauss'")
        return self._cost_logits

    def evaluate_cost(self, critic_observations: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the cost-value function.

        Returns:
            cost_value: [batch_size, num_costs].
        """
        critic_observations = self.critic_obs_normalizer(critic_observations)
        if self.is_distributional_cost_critic:
            logits = self.cost_critic(critic_observations)
            self._cost_logits = logits
            # Always decode as the expected value here. GAE requires an unbiased
            # baseline, so the standard `evaluate_cost` path must not return CVaR
            # (which is the upper-tail expectation and biases A_C globally negative,
            # closing the P3O ReLU gate). Risk-sensitive callers should call
            # `evaluate_cost_cvar` explicitly.
            return self.cost_critic.expected_value(logits)
        return self.cost_critic(critic_observations)

    def evaluate_cost_cvar(
        self,
        critic_observations: torch.Tensor,
        alpha: float | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Risk-sensitive cost-value evaluation (HL-Gauss critic only).

        Decodes the predicted cost-return distribution as CVaR over the worst
        ``alpha`` tail. Used by safety-gate / κ-adaptation logic that wants to
        react to predicted catastrophic outcomes; *not* used as the GAE baseline.

        Args:
            critic_observations: same shape contract as ``evaluate_cost``.
            alpha: tail fraction in (0, 1]. Defaults to ``self.cost_critic.cvar_alpha``
                if set on the critic; raises if neither is provided.
        """
        if self.cost_critic_loss_type != "hlgauss":
            raise RuntimeError(
                f"evaluate_cost_cvar requires the HL-Gauss cost critic; got loss_type={self.cost_critic_loss_type!r}"
            )
        if alpha is None:
            alpha = getattr(self.cost_critic, "cvar_alpha", None)
        if alpha is None:
            raise ValueError(
                "evaluate_cost_cvar called without an alpha; either pass alpha=... or set cost_critic.cvar_alpha"
            )
        critic_observations = self.critic_obs_normalizer(critic_observations)
        logits = self.cost_critic(critic_observations)
        self._cost_logits = logits
        return self.cost_critic.cvar_value(logits, alpha)
