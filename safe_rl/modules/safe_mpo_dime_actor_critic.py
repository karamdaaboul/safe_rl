"""DIME/TruDi diffusion actor paired with a quantile cost critic.

:class:`~safe_rl.modules.mpo_dime_actor_critic.MPODIMEActorCritic` gives the diffusion actor and
the reward critics; :class:`~safe_rl.modules.safe_actor_critic.SafeActorCritic` gives the cost
critics but only for a Gaussian actor. Nothing paired the two, which is the only structural gap
between MPO-DIME and a *constrained* diffusion policy.

Deliberately quantile-only on the cost side. FH-DCMPO's cost critic is undiscounted and episodic,
so its support runs to the episode budget and beyond; the categorical head hard-clips at ``v_max``
and would silently truncate exactly the tail the constraint reads. ``QuantileCritic`` has no
support bound.

**The cost critic never sees the denoising chain.** It is an ordinary ``Q_c(s, a)`` on the *final*
action, and the constraint reaches the policy only through the E-step's per-sample weights. That is
the structural property CLAUDE.md's rule 3 exists to protect: no gradient is ever chained from a
cost critic back through ``T`` denoising steps.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from safe_rl.modules.critic import QuantileCritic
from safe_rl.modules.mpo_dime_actor_critic import MPODIMEActorCritic


class SafeMPODIMEActorCritic(MPODIMEActorCritic):
    """MPO-DIME plus ``num_cost_critics`` quantile cost critics and their frozen targets.

    Args:
        num_costs: Number of constraints. FH-DCMPO is single-constraint, so this must be 1.
        num_cost_critics: Independent cost critics. Default 1 -- the cost channel deliberately
            runs a single critic (no min/max ensemble), so conservatism comes from the risk
            statistic rather than from an ensemble reduction.
        cost_critic_kwargs: Forwarded to :class:`QuantileCritic`.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_costs: int = 1,
        num_cost_critics: int = 1,
        cost_critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        if num_costs != 1:
            raise ValueError(f"SafeMPODIMEActorCritic is single-constraint; got num_costs={num_costs}.")
        self.num_costs = num_costs
        self.num_cost_critics = int(num_cost_critics)

        ckw = deepcopy(cost_critic_kwargs) or {}
        ckw.setdefault("n_quantiles", 64)
        ckw.setdefault("nonneg", True)  # cost is non-negative: softplus head, as in SafeActorCritic
        self.cost_critics = nn.ModuleList(
            QuantileCritic(num_obs=num_critic_obs, num_actions=num_actions, **ckw)
            for _ in range(self.num_cost_critics)
        )
        self.cost_critic_targets = nn.ModuleList(deepcopy(c) for c in self.cost_critics)
        for target in self.cost_critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        # Predicates the SafeSAC/CVPO lineage branches on. Quantile, never categorical.
        self.is_quantile_cost_critic = True
        self.is_distributional_cost_critic = False
        print(f"Safe MPO-DIME Cost Critic (quantile): {self.cost_critics[0]}")

    @property
    def _cost_critic_has_distribution(self) -> bool:
        return True

    def _scalar_qc(self, critic: QuantileCritic, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mean of the represented cost distribution, shape ``[batch, 1]``."""
        return critic.get_value(critic.get_dist(critic(obs, actions))).unsqueeze(-1)

    def evaluate_cost_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = self.critic_obs_normalizer(obs)
        qs = [self._scalar_qc(c, obs, actions) for c in self.cost_critics]
        return torch.stack(qs, dim=0).mean(dim=0)

    def evaluate_cost_q_target(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = self.critic_obs_normalizer(obs)
        qs = [self._scalar_qc(c, obs, actions) for c in self.cost_critic_targets]
        return torch.stack(qs, dim=0).mean(dim=0)
