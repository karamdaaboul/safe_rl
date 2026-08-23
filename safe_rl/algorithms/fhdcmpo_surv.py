"""FH-DCMPO with a survival-shaped reward critic: the hybrid arm.

Where :class:`~safe_rl.algorithms.vt_mpo.VTMPO` drops the CMDP entirely, this keeps all of it --
the finite-horizon quantile cost critic, the constrained E-step, ``eta``, ``lambda``, the budget --
and changes exactly one thing: the REWARD critic is trained on the survival-shaped ``(r~, gamma~)``
of arXiv:2602.04599, so the E-step scores actions with ``Q_surv`` instead of ``Q_r``.

The point of running both
-------------------------
VT-MPO changes two things at once relative to our arms: it adds survival shaping AND removes the
dual. If VT-MPO wins we would not know which half did the work. This arm holds the dual fixed and
adds only the shaping, so the pair separates "shaping helps" from "dropping the dual helps".

It also keeps every metric we already report meaningful -- violation rate against the budget of 25,
realized vs predicted CVaR_0.9, VaR_0.9 coverage -- because the constraint and its threshold are
untouched. VT-MPO has no budget, so those numbers only become comparable after matching realized
cost.

Note the two horizons do NOT interact: the cost channel stays undiscounted (``gamma_c = 1``,
``cost_gamma`` on the buffer) with its own TD(lambda) window, while the shaping applies to the
reward channel's discount. ``ReplayStorage`` keeps them separate by construction, and
:meth:`FHDCMPO._cost_bootstrap_discount` still returns 1.0.
"""

from __future__ import annotations

from typing import Any

from safe_rl.common.continuation import continuation_scale_at
from safe_rl.algorithms.fhdcmpo import FHDCMPO


class FHDCMPOSurv(FHDCMPO):
    """FH-DCMPO whose reward critic sees ``r~ = alpha r`` and ``gamma~ = gamma alpha``.

    Args mirror :class:`~safe_rl.algorithms.vt_mpo.VTMPO` so the two arms can be swept with the
    same config block.
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
        super().__init__(policy, **kwargs)
        if sdh_lambda_final < 0.0:
            raise ValueError(f"sdh_lambda_final must be >= 0, got {sdh_lambda_final}")
        self.sdh_lambda_final = float(sdh_lambda_final)
        self.sdh_lambda_init = float(sdh_lambda_init)
        self.sdh_warmup = int(sdh_warmup)
        self.sdh_ramp = int(sdh_ramp)
        self._sdh_updates = 0
        self._sdh_lambda = float(sdh_lambda_init)
        print(
            f"FH-DCMPO + survival shaping: reward critic on (r~, gamma~), lam "
            f"{self.sdh_lambda_init} -> {self.sdh_lambda_final} over "
            f"[{self.sdh_warmup}, {self.sdh_warmup + self.sdh_ramp}] updates. "
            f"The cost critic, the budget and the lambda dual are UNCHANGED."
        )

    def init_storage(self, *args: Any, **kwargs: Any) -> None:
        super().init_storage(*args, **kwargs)
        self._sync_survival_lambda()

    def _sync_survival_lambda(self) -> None:
        self._sdh_lambda = continuation_scale_at(
            self._sdh_updates, self.sdh_lambda_final, self.sdh_warmup, self.sdh_ramp, self.sdh_lambda_init
        )
        if self.storage is not None:
            self.storage.survival_lambda = self._sdh_lambda

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self._sync_survival_lambda()
        self._sdh_updates += 1
        return super().update(*args, **kwargs)

    def get_penalty_info(self) -> dict[str, Any]:
        info = dict(super().get_penalty_info())
        info["sdh_lambda"] = self._sdh_lambda
        return info
