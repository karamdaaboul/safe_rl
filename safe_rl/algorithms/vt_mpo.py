"""VT-MPO: MPO on a survival-shaped MDP, with no Lagrange multiplier at all.

Implements the virtual-termination instantiation of *Stochastic Decision Horizons for Constrained
Reinforcement Learning* (Milosevic et al., arXiv:2602.04599). Constraint violations do not consume
a budget policed by a dual variable; they shorten the effective planning horizon:

    alpha(s, a) = exp(-lam * sum_i c_i(s, a)),    r~ = alpha r,    gamma~ = gamma alpha

and the critic learns ``Q_surv`` under those shaped quantities. **The E-step and M-step are
inherited from MPO unchanged** -- that is the paper's central claim, and the reason this class is
nearly empty. Everything the method needs happens in two places already built:

* :mod:`safe_rl.common.continuation` -- the alpha mapping and the scale schedule;
* :meth:`safe_rl.storage.ReplayStorage._gather_n_step` -- the survival-shaped n-step return
  ``R^(n) = sum_k u_k r~_k`` and the bootstrap factor ``u_{t+n}``, computed at SAMPLE time from
  the stored per-step costs so a scheduled ``lam`` is never stale in replay.

What this arm gives up, stated plainly
--------------------------------------
It is **not** a CMDP. ``cost_limits`` is not enforced and is carried only so the runner's logging
and the shared evaluation path keep working; there is no budget, no feasibility certificate and no
lambda. The operating point is set entirely by ``sdh_lambda_final``, which has to be swept. Compare
arms at MATCHED realized cost, never at "the same constraint".

Why it is worth running here
----------------------------
Measured on this repo: ``std_a(Q_c)`` is 0.03-0.16 against a level of ~16, so the additive
``- lambda Q_c`` term carries under 1% of the across-action spread that the per-state E-step softmax
can actually use. SDH replaces that additive term with a multiplicative attenuation of the entire
return, which compounds along the horizon. If the diagnosis is right this is the arm that fixes it;
if cost does not fall at matched reward, the diagnosis was incomplete and that is worth knowing.
"""

from __future__ import annotations

from typing import Any

from safe_rl.common.continuation import continuation_scale_at
from safe_rl.algorithms.mpo import MPO


class VTMPO(MPO):
    """MPO whose reward critic is trained on the survival-shaped ``(r~, gamma~)``.

    Args:
        sdh_lambda_final: continuation scale reached at the end of the ramp. This is THE knob:
            0 recovers plain MPO exactly, larger values attenuate harder.
        sdh_lambda_init: scale before the ramp starts.
        sdh_warmup: updates held at ``sdh_lambda_init``. A large scale from step 0 attenuates the
            return before the critic has learned anything to attenuate.
        sdh_ramp: updates spent ramping linearly to ``sdh_lambda_final``. Set both warmup and ramp
            to 0 for a fixed scale, which is what the lam sweep uses.
    """

    def __init__(
        self,
        policy,
        sdh_lambda_final: float = 0.5,
        sdh_lambda_init: float = 0.0,
        sdh_warmup: int = 0,
        sdh_ramp: int = 0,
        **kwargs: Any,
    ) -> None:
        cost_limits = kwargs.pop("cost_limits", None)
        super().__init__(policy, **kwargs)
        self._init_cost_plumbing(cost_limits)
        if sdh_lambda_final < 0.0:
            raise ValueError(f"sdh_lambda_final must be >= 0, got {sdh_lambda_final}")
        self.sdh_lambda_final = float(sdh_lambda_final)
        self.sdh_lambda_init = float(sdh_lambda_init)
        self.sdh_warmup = int(sdh_warmup)
        self.sdh_ramp = int(sdh_ramp)
        self._sdh_updates = 0
        self._sdh_lambda = float(sdh_lambda_init)
        self._last_alpha_diag: dict[str, float] = {}
        print(
            f"VT-MPO: survival-shaped critic, continuation alpha = exp(-lam * sum_i c_i), "
            f"lam {self.sdh_lambda_init} -> {self.sdh_lambda_final} over "
            f"[{self.sdh_warmup}, {self.sdh_warmup + self.sdh_ramp}] updates. "
            f"NO lambda dual and NO cost budget -- compare at matched realized cost."
        )

    # The runner gates cost plumbing on `num_costs > 0` (off_policy_runner.py:191). VT-MPO needs
    # the cost SIGNAL (alpha is a function of it) but has no cost critic and no dual, so it opts
    # into the plumbing and nothing else. `cost_limits` is accepted and stored for logging and for
    # the shared evaluation path -- it is NOT enforced anywhere in this algorithm.
    def _init_cost_plumbing(self, cost_limits) -> None:
        limits = [25.0] if cost_limits is None else list(cost_limits)
        self.cost_limits = [float(x) for x in limits]
        self.num_costs = len(self.cost_limits)

    def init_storage(self, *args: Any, **kwargs: Any) -> None:
        """Storage must carry per-step costs: the shaping reads them at sample time."""
        super().init_storage(*args, **kwargs)
        self._sync_survival_lambda()

    def store_transition(
        self,
        obs,
        action,
        reward,
        done,
        next_obs,
        cost=None,
        critic_obs=None,
        next_critic_obs=None,
        bootstrap=None,
        behavior_log_prob=None,
        policy_version=None,
    ) -> None:
        """Accept and store the per-step cost. Mirrors SafeSAC's override, minus the cost critic.

        Without this the buffer holds no costs, `alpha` has nothing to read, and the shaping would
        raise at the first sample -- which is the correct loud failure, but this is the fix.
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage() first.")
        import torch

        if cost is None:
            cost = torch.zeros(obs.shape[0], self.num_costs, device=self.device)
        cost = cost.view(-1, self.num_costs) if cost.dim() == 1 else cost
        extras = {"costs": cost}
        if critic_obs is not None:
            extras["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            extras["next_critic_observations"] = next_critic_obs
        if bootstrap is not None:
            extras["bootstrap"] = bootstrap.view(-1, 1) if bootstrap.dim() == 1 else bootstrap
        if behavior_log_prob is not None:
            extras["behavior_log_prob"] = behavior_log_prob.view(-1, 1)
        if policy_version is not None:
            extras["policy_version"] = policy_version.view(-1, 1)
        self.storage.add(obs, action, reward, done, next_obs, **extras)

    def _sync_survival_lambda(self) -> None:
        """Publish the current scale to the buffer. Called every update, because the buffer
        recomputes alpha per sample and must use the live value, not the one at insertion."""
        self._sdh_lambda = continuation_scale_at(
            self._sdh_updates, self.sdh_lambda_final, self.sdh_warmup, self.sdh_ramp, self.sdh_lambda_init
        )
        if self.storage is not None:
            self.storage.survival_lambda = self._sdh_lambda

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        # The runner passes `current_costs` to every algorithm it considers safe RL, for a PID
        # Lagrangian update. VT-MPO has no multiplier, so it is accepted and discarded -- the
        # constraint acts through alpha in the critic target, not through a dual.
        kwargs.pop("current_costs", None)
        self._sync_survival_lambda()
        self._sdh_updates += 1
        losses = super().update(*args, **kwargs)
        losses.update(self._survival_diagnostics())
        return losses

    def _survival_diagnostics(self) -> dict[str, float]:
        """``alpha`` statistics over a fresh batch.

        Logged because the silent failure of this arm is ``alpha == 1`` everywhere -- a run that
        looks healthy and is simply unshaped MPO wearing VT-MPO's name.
        """
        info = {"sdh_lambda": self._sdh_lambda}
        if self.storage is None or not self.storage._initialized:
            return info
        import torch

        from safe_rl.common.continuation import exponential_continuation

        with torch.no_grad():
            batch = self.storage.sample(self.batch_size)
            costs = batch.get("costs")
            if costs is None:
                return info
            alpha = exponential_continuation(costs, self._sdh_lambda)
            info["sdh_alpha_mean"] = float(alpha.mean())
            info["sdh_alpha_min"] = float(alpha.min())
            sd = batch.get("survival_discount")
            if sd is not None:
                info["sdh_survival_discount_mean"] = float(sd.mean())
        return info

    def get_penalty_info(self) -> dict[str, Any]:
        """No dual to report -- surface the shaping instead, under the same hook the runner logs."""
        info = dict(getattr(super(), "get_penalty_info", lambda: {})())
        info.update(self._last_alpha_diag)
        info["sdh_lambda"] = self._sdh_lambda
        return info
