"""Analytic H-LIP reference generator for CLF-RL reward shaping.

Produces velocity-conditioned reference outputs ``yd`` and their time
derivatives ``yd_dot`` from the Hybrid Linear Inverted Pendulum (H-LIP) model,
fully vectorized over environments. This is the LIP-CLF reference of Li, Olkin
et al., "CLF-RL: Control Lyapunov Function Guided Reinforcement Learning"
(RA-L 2025), Sec. III-A. The reference is closed-form, generated online, and
needs no offline trajectory optimization.

The default output spec is velocity-based so it can be tracked robustly from the
robot base state alone (no stance/swing-foot bookkeeping): forward velocity from
the analytic LIP profile, plus commanded lateral / vertical / yaw rates. A
5th-order Bezier swing-foot height output is also available (``swing_foot_z``)
for setups that expose a swing-foot frame.

Each output is treated as a "position" in the CLF tracking error
``eta = [yd - y; yd_dot - y_dot]``; the consuming wrapper forms the velocity
block by finite-differencing, matching the paper's finite-difference treatment
of the Lyapunov derivative.
"""

from __future__ import annotations

import math

import torch

# Output kinds the generator and the CLF wrapper agree on. The wrapper switches
# on the same strings to extract the matching actual values from the robot.
VELOCITY_OUTPUTS = ("lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_z")
DEFAULT_OUTPUTS = VELOCITY_OUTPUTS


def lip_velocity_profile(
    vd: torch.Tensor,
    phase: torch.Tensor,
    step_period: float,
    com_height: float,
    gravity: float = 9.81,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic H-LIP forward CoM velocity and acceleration over a step.

    Solves the period-1 H-LIP orbit whose average forward speed equals the
    commanded ``vd`` and returns the instantaneous velocity ``v(t)`` and
    acceleration ``a(t) = v_dot(t)`` at the given gait ``phase`` in ``[0, 1)``.

    With stance foot at the origin the single-support LIP obeys
    ``p_dot = v``, ``v_dot = lambda^2 p`` (``lambda = sqrt(g / z0)``). The
    period-1 orbit at step length ``u* = vd * T`` has ``p0 = -u*/2`` and
    ``v0 = (u*/2) * lambda * coth(lambda T / 2)``; the mean of ``v(t)`` over the
    step equals ``vd`` by construction.

    Args:
        vd: Commanded forward velocity, shape ``(num_envs,)``.
        phase: Gait phase in ``[0, 1)``, shape ``(num_envs,)``.
        step_period: Single-support step duration ``T`` in seconds.
        com_height: Constant CoM height ``z0`` in meters.
        gravity: Gravitational acceleration.

    Returns:
        ``(v, a)`` each of shape ``(num_envs,)``.
    """
    lam = math.sqrt(gravity / com_height)
    t = phase * step_period
    half = 0.5 * lam * step_period
    coth_half = math.cosh(half) / math.sinh(half)

    u_star = vd * step_period
    p0 = -0.5 * u_star
    v0 = 0.5 * u_star * lam * coth_half

    sinh = torch.sinh(lam * t)
    cosh = torch.cosh(lam * t)
    v = p0 * lam * sinh + v0 * cosh
    a = p0 * (lam**2) * cosh + v0 * lam * sinh
    return v, a


def bezier5(control_points: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Evaluate a 5th-order Bezier curve at parameter ``s`` in ``[0, 1]``.

    Args:
        control_points: Six control points, shape ``(..., 6)`` (broadcast over
            leading dims).
        s: Curve parameter, shape broadcastable to ``control_points[..., 0]``.

    Returns:
        Curve value, shape matching the broadcast of the inputs.
    """
    coeffs = [1.0, 5.0, 10.0, 10.0, 5.0, 1.0]
    out = torch.zeros_like(s)
    for i, c in enumerate(coeffs):
        out = out + c * (s**i) * ((1.0 - s) ** (5 - i)) * control_points[..., i]
    return out


def bezier5_deriv(control_points: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Time-normalized derivative ``d/ds`` of :func:`bezier5`.

    Returns ``dB/ds``; multiply by ``ds/dt = 1 / step_period`` for the time
    derivative.
    """
    # B'(s) = 5 * sum_{i=0..4} C(4,i) s^i (1-s)^(4-i) (P_{i+1} - P_i)
    coeffs = [1.0, 4.0, 6.0, 4.0, 1.0]
    diffs = control_points[..., 1:] - control_points[..., :-1]
    out = torch.zeros_like(s)
    for i, c in enumerate(coeffs):
        out = out + c * (s**i) * ((1.0 - s) ** (4 - i)) * diffs[..., i]
    return 5.0 * out


class HLIPReference:
    """Vectorized H-LIP analytic reference generator.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device for produced tensors.
        outputs: Ordered tuple of output kinds (see :data:`VELOCITY_OUTPUTS` and
            ``"swing_foot_z"``). Determines the dimension and meaning of ``yd``.
        step_period: Single-support step duration ``T`` (s). Half of the full
            gait cycle (paper uses a 0.8 s cycle -> 0.4 s step).
        com_height: Nominal CoM height ``z0`` (m).
        gravity: Gravitational acceleration.
        swing_height: Peak swing-foot clearance (m), used by ``swing_foot_z``.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        outputs: tuple[str, ...] = DEFAULT_OUTPUTS,
        step_period: float = 0.4,
        com_height: float = 0.74,
        gravity: float = 9.81,
        swing_height: float = 0.08,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.outputs = tuple(outputs)
        self.step_period = float(step_period)
        self.com_height = float(com_height)
        self.gravity = float(gravity)
        self.swing_height = float(swing_height)

        known = set(VELOCITY_OUTPUTS) | {"swing_foot_z"}
        unknown = [o for o in self.outputs if o not in known]
        if unknown:
            raise ValueError(f"Unknown reference output kinds: {unknown}. Known: {sorted(known)}.")

    @property
    def n_out(self) -> int:
        return len(self.outputs)

    def compute(self, command: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reference outputs and their time derivatives.

        Args:
            command: Commanded twist ``[vx, vy, wz]``, shape ``(num_envs, 3)``.
            phase: Gait phase in ``[0, 1)``, shape ``(num_envs,)``.

        Returns:
            ``(yd, yd_dot)`` each of shape ``(num_envs, n_out)``.
        """
        vx, vy, wz = command[:, 0], command[:, 1], command[:, 2]
        zero = torch.zeros(self.num_envs, device=self.device)

        v_lip, a_lip = lip_velocity_profile(vx, phase, self.step_period, self.com_height, self.gravity)

        cols_y: list[torch.Tensor] = []
        cols_yd: list[torch.Tensor] = []
        for kind in self.outputs:
            if kind == "lin_vel_x":
                cols_y.append(v_lip)
                cols_yd.append(a_lip)
            elif kind == "lin_vel_y":
                cols_y.append(vy)
                cols_yd.append(zero)
            elif kind == "lin_vel_z":
                cols_y.append(zero)
                cols_yd.append(zero)
            elif kind == "ang_vel_z":
                cols_y.append(wz)
                cols_yd.append(zero)
            elif kind == "swing_foot_z":
                # 5th-order Bezier height profile: rises to clearance mid-swing,
                # returns to ground at touchdown. Control points in meters.
                h = self.swing_height
                ctrl = torch.tensor(
                    [0.0, 0.0, 1.5 * h, 1.5 * h, 0.0, 0.0], device=self.device
                ).expand(self.num_envs, 6)
                z = bezier5(ctrl, phase)
                zdot = bezier5_deriv(ctrl, phase) / self.step_period
                cols_y.append(z)
                cols_yd.append(zdot)

        yd = torch.stack(cols_y, dim=1)
        yd_dot = torch.stack(cols_yd, dim=1)
        return yd, yd_dot
