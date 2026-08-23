"""FH-DCMPO with a per-state cost multiplier and one shared temperature.

CLAUDE.md's M5-A, applied to the finite-horizon method. Pure composition again:

* :class:`~safe_rl.algorithms.cvpo_per_state.CVPOPerState` owns the **dual**: one global ``eta``
  for the batch, and ``lambda_b`` solved separately for every state by vectorised 1-D
  root-finding (``safe_rl.common.per_state_dual``). No SciPy, no per-batch host sync beyond one
  scalar.
* :class:`~safe_rl.algorithms.fhdcmpo.FHDCMPO` owns the **cost statistic**: the undiscounted
  finite-horizon ``rho = E[Z_c] + kappa (CVaR_alpha[Z_c] - E[Z_c])`` in true episodic units, plus
  the ``gamma_c = 1`` critic contract.

The seam is ``_estep_cost``. ``CVPOPerState._estep_weights`` calls ``self._estep_cost(...)``, so the
MRO ``FHDCMPOPerState -> CVPOPerState -> FHDCMPO -> CVPO`` gives the per-state dual the
finite-horizon cost signal with no plumbing.

**Why per-state lambda is worth trying here specifically.** A single shared ``lambda`` applies the
same pressure in a corridor between hazards and in open space. Measured on the shared-lambda arms,
the E-step reduces predicted cost by only 1-5% while sitting exactly on its KL budget
(``kl_q/eps = 1.000``) -- it spends its whole trust region everywhere, including where there is no
cost to save. Per-state ``lambda_b`` concentrates that budget where cost is actually reducible.

**What would make it a failure, stated up front** (CLAUDE.md M5): ``lambda_b`` is fitted from that
state's ``N`` samples alone, so it is noisier than a batch-wide fit. If ``ess_per_state`` collapses
-- weights piling onto one or two actions -- the M-step target becomes noise and the honest move is
to revert to shared ``lambda``. That is a legitimate result, not a failure; CVPO's authors reached
it first.
"""

from __future__ import annotations

from typing import Any

from safe_rl.common.fh_cost import kappa_at

from .cvpo_per_state import CVPOPerState
from .fhdcmpo import FHDCMPO


class FHDCMPOPerState(CVPOPerState, FHDCMPO):
    """Global ``eta``, per-state ``lambda_b``, finite-horizon cost statistic."""

    def __init__(self, policy, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)

        w = type(self)._estep_weights.__qualname__.split(".")[0]
        c = type(self)._estep_cost.__qualname__.split(".")[0]
        if w not in ("FHDCMPOPerState", "CVPOPerState") or c != "FHDCMPO":
            raise RuntimeError(
                f"MRO regression: _estep_weights from {w}, _estep_cost from {c}"
                "; expected the per-state dual with the finite-horizon cost statistic."
            )
        print(
            f"FH-DCMPO per-state: global eta, per-state lambda (cap {self._current_lambda_max():.3f})"
            f", qc_thres = {self.qc_thres:.3f}"
            f", estep_sample_std_scale = {self.estep_sample_std_scale}"
        )

    def _estep_weights(self, q, actions, critic_obs):
        """Advance the kappa ramp, then run the per-state dual.

        Necessary because the ramp lives in ``FHDCMPO._estep_weights``, which this class's MRO
        bypasses in favour of ``CVPOPerState``'s. Without this the ramp would silently never
        advance and every ``cvar`` arm would quietly run as a ``mean`` arm.
        """
        self._fh_kappa = kappa_at(self._fh_updates, self.fh_kappa_target, self.fh_kappa_warmup, self.fh_kappa_ramp)
        self._fh_updates += 1
        weights = super()._estep_weights(q, actions, critic_obs)
        self._last_estep_info["fh_kappa"] = self._fh_kappa
        return weights
