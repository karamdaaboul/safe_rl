"""Lagrange-multiplier controllers for a constrained E-step.

The default ``sgd`` mode is the existing CVPO update, ``lambda <- clip(lambda + lr*delta, 0,
lam_max)``, kept bit-identical so switching modes is opt-in.

``pid`` adds proportional and derivative terms (Stooke et al. 2020, arXiv:2007.03964) plus
anti-windup. Anti-windup is the reason this module exists: when ``lambda`` saturates at
``lam_max`` the integral keeps accumulating a reserve that must be paid back before the
multiplier can fall again. Measured in the CVaR arms, ``lambda`` sat at its cap for roughly
13000 of 15000 iterations, during which the E-step weight ``exp((Q_r - lambda*Q_c)/eta)`` is
effectively "minimise cost, ignore reward" -- the observed reward collapse was a controller
stuck at its bound, not risk-sensitivity.

``rescale_advantage`` implements the CPPOPID-style ``1/(1+lambda)`` normalisation, which
bounds the combined advantage in ``lambda`` without changing its sign or its ordering across
actions (the E-step softmax only cares about relative differences).
"""

from __future__ import annotations

import numpy as np


class LambdaController:
    """Projected multiplier update, in ``sgd`` (default) or ``pid`` mode.

    Args:
        mode: ``"sgd"`` reproduces the existing integral-only update; ``"pid"`` adds Kp/Kd.
        lr: step size for ``sgd`` mode.
        kp, ki, kd: PID gains (``ki`` plays the role of ``lr``).
        lam_max: upper bound; ``lambda`` is projected onto ``[0, lam_max]``.
        anti_windup: freeze integral accumulation while the output is saturated *and* the
            error would push it further into the bound. Ignored in ``sgd`` mode.
    """

    def __init__(
        self,
        mode: str = "sgd",
        lr: float = 0.03,
        kp: float = 0.0,
        ki: float = 0.03,
        kd: float = 0.0,
        lam_max: float = 100.0,
        anti_windup: bool = True,
    ) -> None:
        if mode not in ("sgd", "pid"):
            raise ValueError(f"mode must be 'sgd' or 'pid', got {mode!r}")
        if lam_max < 0.0:
            raise ValueError(f"lam_max must be non-negative, got {lam_max}")
        self.mode = mode
        self.lr = float(lr)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.lam_max = float(lam_max)
        self.anti_windup = bool(anti_windup)
        self.reset()

    def reset(self) -> None:
        self.lam = 0.0
        self.integral = 0.0
        self._prev_delta: float | None = None

    def update(self, delta: float) -> float:
        """One controller step. ``delta = E[Q_c] - qc_thres`` (positive == violating)."""
        delta = float(delta)
        if not np.isfinite(delta):
            return self.lam

        if self.mode == "sgd":
            self.lam = float(np.clip(self.lam + self.lr * delta, 0.0, self.lam_max))
            self.integral = self.lam
            self._prev_delta = delta
            return self.lam

        # Freeze the integral when saturated and the error pushes further into the bound.
        # Without this the integral banks an arbitrarily large reserve during a long
        # violation and lambda cannot come down until it is unwound.
        saturated_high = self.lam >= self.lam_max - 1e-12 and delta > 0.0
        saturated_low = self.lam <= 1e-12 and delta < 0.0
        if not (self.anti_windup and (saturated_high or saturated_low)):
            self.integral += self.ki * delta

        derivative = 0.0 if self._prev_delta is None else (delta - self._prev_delta)
        self._prev_delta = delta

        raw = self.kp * delta + self.integral + self.kd * derivative
        self.lam = float(np.clip(raw, 0.0, self.lam_max))
        return self.lam

    def state_dict(self) -> dict:
        return {"mode": self.mode, "lr": self.lr, "kp": self.kp, "ki": self.ki, "kd": self.kd,
                "lam_max": self.lam_max, "anti_windup": self.anti_windup,
                "lam": self.lam, "integral": self.integral, "prev_delta": self._prev_delta}

    def load_state_dict(self, state: dict) -> None:
        for key in ("mode", "lr", "kp", "ki", "kd", "lam_max", "anti_windup"):
            setattr(self, key, state[key])
        self.lam = float(state["lam"])
        self.integral = float(state["integral"])
        self._prev_delta = state["prev_delta"]


def rescale_advantage(q_r, q_c, lam: float):
    """``(Q_r - lambda*Q_c) / (1 + lambda)`` -- bounded in lambda, sign- and order-preserving.

    Dividing by a positive constant cannot flip a sign or reorder actions, so the E-step
    softmax sees the same preference ordering; only the effective temperature changes.
    """
    lam = float(lam)
    if lam < 0.0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return (q_r - lam * q_c) / (1.0 + lam)
