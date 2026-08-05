"""Quantile recalibration of a predicted distribution (Kuleshov et al., 2018, arXiv:1807.00263).

A categorical cost critic can rank risk correctly while being badly *calibrated*: measured on
SafetyPointGoal1 the predicted std was 3.21 against a realized 5.55, the PIT was U-shaped
(KS 0.381 against a 0.008 critical value), and ``VaR_0.9`` covered only 0.858 of realized
returns instead of 0.90. A CVaR read off such a distribution understates tail risk by 25-30%.

The fix is a monotone map ``R: [0, 1] -> [0, 1]`` learned from the empirical distribution of
PIT values ``u = F_pred(y_realized)``. If the forecast were calibrated, ``u`` would be uniform;
``R`` is the empirical CDF of ``u``, fitted with isotonic regression so it stays monotone, and
``R(F_pred)`` is used as the recalibrated CDF. Unlike a variance rescaling this assumes no
shape, which matters here: the true law is a spike at zero plus a heavy right tail, and a
symmetric widening cannot represent that (measured: it left PIT KS at 0.183 vs 0.064 here).

Pure functions plus a small stateful buffer; no training-loop imports.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


class PITRecalibrator:
    """FIFO buffer of PIT values with a periodically refitted isotonic map.

    Args:
        capacity: max PIT samples retained (FIFO, so the map tracks the current policy).
        min_samples: refits are refused below this, and :meth:`apply` stays the identity —
            an under-fitted map is worse than none.
    """

    def __init__(self, capacity: int = 20000, min_samples: int = 500) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")
        self.capacity = int(capacity)
        self.min_samples = int(min_samples)
        self._buf: deque[float] = deque(maxlen=self.capacity)
        self._x: np.ndarray | None = None   # sorted PIT values
        self._y: np.ndarray | None = None   # isotonic-fitted targets

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return self._x is not None

    def update(self, pits: Iterable[float] | np.ndarray) -> None:
        """Add PIT observations. Values outside [0, 1] are clipped, not dropped."""
        arr = np.asarray(list(pits) if not isinstance(pits, np.ndarray) else pits, dtype=np.float64)
        if arr.size == 0:
            return
        self._buf.extend(np.clip(arr.reshape(-1), 0.0, 1.0).tolist())

    def refit(self) -> bool:
        """Refit the monotone map. Returns False (leaving the map unchanged) if too few samples."""
        if len(self._buf) < self.min_samples:
            return False
        from sklearn.isotonic import IsotonicRegression

        u = np.sort(self.values)
        # Target: where each sample SHOULD sit if the PIT were uniform.
        ranks = (np.arange(1, u.size + 1) - 0.5) / u.size
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self._y = np.asarray(iso.fit_transform(u, ranks), dtype=np.float64)
        self._x = u
        return True

    def apply(self, cdf_values: np.ndarray) -> np.ndarray:
        """Map predicted CDF values through R. Identity until fitted."""
        arr = np.asarray(cdf_values, dtype=np.float64)
        if self._x is None or self._y is None:
            return arr
        return np.interp(np.clip(arr, 0.0, 1.0), self._x, self._y, left=0.0, right=1.0)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "min_samples": self.min_samples,
                "buf": list(self._buf),
                "x": None if self._x is None else self._x.tolist(),
                "y": None if self._y is None else self._y.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.min_samples = int(state["min_samples"])
        self._buf = deque(state["buf"], maxlen=self.capacity)
        self._x = None if state["x"] is None else np.asarray(state["x"], dtype=np.float64)
        self._y = None if state["y"] is None else np.asarray(state["y"], dtype=np.float64)


def recalibrate_cdf(cdf: np.ndarray, recalibrator: PITRecalibrator) -> np.ndarray:
    """Apply R to a CDF, restoring monotonicity and the right endpoint."""
    out = recalibrator.apply(cdf)
    out = np.maximum.accumulate(out, axis=-1)
    out[..., -1] = 1.0
    return out


def recalibrate_distribution(
    atoms: np.ndarray, probs: np.ndarray, recalibrator: PITRecalibrator
) -> np.ndarray:
    """Remap a categorical distribution's CDF through R and return the new probabilities.

    ``atoms`` is accepted for interface symmetry with the critic (the support is unchanged;
    only the mass on it moves).
    """
    del atoms  # support is fixed; only the probabilities are recalibrated
    if not recalibrator.is_fitted:
        return np.asarray(probs, dtype=np.float64)

    p = np.asarray(probs, dtype=np.float64)
    new_cdf = recalibrate_cdf(np.cumsum(p, axis=-1), recalibrator)
    out = np.diff(new_cdf, axis=-1, prepend=0.0)
    out = np.clip(out, 0.0, None)
    total = out.sum(axis=-1, keepdims=True)
    # A degenerate map can annihilate all mass; fall back to the input rather than emit NaN.
    return np.where(total > 0, out / np.maximum(total, 1e-300), p)
