"""Control-Lyapunov-function helpers for CLF-RL reward shaping.

Implements the constant CLF matrix ``P`` and the reward normalization constants
used by :class:`safe_rl.envs.clf_reward_wrapper.CLFRewardWrapper`.

Reference: Li, Olkin et al., "CLF-RL: Control Lyapunov Function Guided
Reinforcement Learning" (RA-L 2025), Sec. III-B and Prop. 1. The output-tracking
error system linearizes to a double integrator, so ``V(eta) = eta^T P eta`` with
``P`` the solution of the continuous-time algebraic Riccati equation (CARE) is a
valid CLF for the full nonlinear dynamics during single support.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.linalg import solve_continuous_are
except ImportError:  # pragma: no cover - exercised only when scipy is missing
    solve_continuous_are = None


def build_clf_P(
    n_out: int,
    q_pos: float = 1.0,
    q_vel: float = 1.0,
    r: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Solve the CARE for the double-integrator eta-system and return ``P``.

    The tracking error ``eta = [e; e_dot]`` (positions stacked above velocities,
    each block of dim ``n_out``) follows a double integrator with
    ``A = [[0, I], [0, 0]]`` and ``B = [[0], [I]]``. We solve

        ``A^T P + P A - P B R^{-1} B^T P + Q = 0``

    once at construction with ``Q = diag(q_pos I, q_vel I)`` and ``R = r I`` and
    return ``P`` as a ``(2 n_out, 2 n_out)`` torch tensor. ``V(eta) = eta^T P eta``
    is then a valid (and positive-definite) CLF.

    Args:
        n_out: Number of tracked outputs (the dimension of ``e``).
        q_pos: Position-error weight on the Q diagonal.
        q_vel: Velocity-error weight on the Q diagonal.
        r: Control weight (R = r I).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.

    Returns:
        The ``(2 n_out, 2 n_out)`` symmetric positive-definite CLF matrix ``P``.
    """
    if solve_continuous_are is None:
        raise ImportError("scipy is required to build the CLF P matrix (pip install scipy).")
    n = int(n_out)
    zero = np.zeros((n, n))
    eye = np.eye(n)
    a = np.block([[zero, eye], [zero, zero]])
    b = np.block([[zero], [eye]])
    q = np.block([[q_pos * eye, zero], [zero, q_vel * eye]])
    r_mat = r * eye

    p_np = solve_continuous_are(a, b, q, r_mat)
    p_np = 0.5 * (p_np + p_np.T)  # symmetrize away numerical asymmetry

    min_eig = float(np.linalg.eigvalsh(p_np).min())
    if min_eig <= 0.0:
        raise ValueError(f"CARE solution is not positive definite (min eigenvalue={min_eig:.3e}).")

    p = torch.as_tensor(p_np, dtype=dtype)
    if device is not None:
        p = p.to(device)
    return p


def clf_sigma(
    p: torch.Tensor,
    eta_max: float,
    eta_dot_max: float | None = None,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward normalization constants ``sigma_v`` and ``sigma_vdot``.

    Following CLF-RL Sec. III-B::

        sigma_v    = mu_max(P) * eta_max**2
        sigma_vdot = 2 * ||P|| * eta_max * eta_dot_max + lam * mu_max(P) * eta_max**2

    where ``mu_max(P)`` is the largest eigenvalue of ``P`` and ``eta_max`` /
    ``eta_dot_max`` are empirical bounds on the tracking error and its rate.

    Returns:
        ``(sigma_v, sigma_vdot)`` as 0-d tensors on ``p``'s device.
    """
    mu_max = torch.linalg.eigvalsh(p).max()
    p_norm = torch.linalg.matrix_norm(p, ord=2)
    if eta_dot_max is None:
        eta_dot_max = eta_max
    sigma_v = mu_max * (eta_max**2)
    sigma_vdot = 2.0 * p_norm * eta_max * eta_dot_max + lam * mu_max * (eta_max**2)
    return sigma_v, sigma_vdot
