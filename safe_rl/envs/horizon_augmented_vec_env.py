"""Safety-Gymnasium vec env that appends finite-horizon state to the observation.

FH-DCMPO's cost critic learns the **undiscounted** cost-to-go over the remaining episode,
``sum_{t'=t}^{T-1} c_t'``. With ``gamma_c = 1`` that quantity is not a function of ``(s, a)``
alone -- it depends on how much episode is left -- so the regression is ill-posed until the
remaining horizon is part of the state. This wrapper supplies it.

Why a subclass and not an edit to :class:`SafetyGymnasiumVecEnv`: the base class already has
exactly the right seam. ``_append_risk`` is the single funnel through which every observation
leaves the env -- ``reset`` (:283), ``step`` (:341), and the ``final_observation`` path in
``_build_extras`` (:499, :510) -- and ``risk_obs_dim`` is the single place the appended width is
declared. Overriding those two is enough, with no changes to the base file.

**The step ordering already gives the correct value for free**, which is worth stating because it
is easy to get wrong and hard to notice. In ``SafetyGymnasiumVecEnv.step``:

1. ``episode_length_buf += 1`` (:303)
2. ``_build_extras`` (:307) -- so ``final_observation`` is stamped while the counter reads ``T``,
   i.e. ``u = 0``, the correct terminal remaining-horizon and the boundary condition that pins the
   undiscounted cost-to-go to zero;
3. the done block zeroes the counter for finished envs (:336);
4. ``_append_risk`` (:341) -- so the returned observation carries ``u = 1`` for an auto-reset
   episode and ``u_{t+1}`` for a continuing one.

Since ``critic_obs`` on iteration ``t+1`` *is* the previous ``next_critic_obs``, the ``(u_t,
u_{t+1})`` pairing the bootstrap needs falls out with no boundary handling of our own, and
``OffPolicyRunner._terminal_observations`` patches truncations with the ``u = 0``
``final_observation``.

Column order is ``[obs, u, (b), risk]``: the horizon columns go **before** the risk level so that
``critic_obs[..., -1]`` still reads the risk level, which ``cvpo.py:871`` depends on.
"""

from __future__ import annotations

import torch
from typing import Any

from safe_rl.common.fh_cost import horizon_feature_dim, normalized_remaining_budget, normalized_remaining_horizon

from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv


class HorizonAugmentedVecEnv(SafetyGymnasiumVecEnv):
    """Appends ``u_t = (T - t)/T`` and optionally ``b_t = (d - J_c^{<t})/d`` to the observation.

    Args:
        horizon_feature: Append the normalized remaining horizon. This is the one FH-DCMPO
            actually requires; without it the undiscounted value function is not Markov.
        budget_feature: Also append the normalized remaining cost budget (Sauté RL's safety
            state). Off by default -- it makes the *policy* budget-conditional, which is a
            behavioural change on top of the critic change and deserves its own ablation.
        budget_limit: The ``d`` used to normalize the budget feature. Defaults to the env's
            first cost limit.
    """

    def __init__(
        self,
        *args: Any,
        horizon_feature: bool = True,
        budget_feature: bool = False,
        budget_limit: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.horizon_feature = bool(horizon_feature)
        self.budget_feature = bool(budget_feature)
        if not self.horizon_feature and self.budget_feature:
            raise ValueError(
                "budget_feature requires horizon_feature: the remaining budget alone does not "
                "make the undiscounted cost-to-go Markov, since it says nothing about how many "
                "steps are left to spend it over."
            )
        limit = budget_limit
        if limit is None:
            limits = getattr(self, "cost_limits", None)
            limit = float(limits[0]) if limits else None
        if self.budget_feature and (limit is None or limit <= 0.0):
            raise ValueError("budget_feature needs a positive budget_limit (or env cost_limits) to normalize by.")
        self.budget_limit = limit
        print(
            f"HorizonAugmentedVecEnv: appending {self._horizon_dim} finite-horizon column(s) "
            f"(horizon={self.horizon_feature}, budget={self.budget_feature}, "
            f"T={self.max_episode_length}, d={self.budget_limit})"
        )

    @property
    def _horizon_dim(self) -> int:
        if not self.horizon_feature:
            return 0
        return horizon_feature_dim(self.budget_feature)

    @property
    def risk_obs_dim(self) -> int:
        """Total appended width. Read at ``_build_extras`` (:503) to rebuild the raw obs shape.

        The base class names this after the risk conditioning because that was the only appended
        column; it is used purely as "how many columns did we add", so widening it here is what
        keeps the ``final_observation`` zero-fill the right shape.
        """
        return super().risk_obs_dim + self._horizon_dim

    def _append_horizon(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate the finite-horizon columns. Read from the counters, never from ``obs``."""
        if not self.horizon_feature:
            return obs
        cols = [obs, normalized_remaining_horizon(self.episode_length_buf, self.max_episode_length).to(obs.dtype)]
        if self.budget_feature:
            cols.append(normalized_remaining_budget(self._cost_in_episode, self.budget_limit).to(obs.dtype))
        return torch.cat(cols, dim=-1)

    def _append_risk(self, obs: torch.Tensor) -> torch.Tensor:
        """Horizon columns first, then the base class's risk level, so risk stays at ``[..., -1]``."""
        return super()._append_risk(self._append_horizon(obs))
