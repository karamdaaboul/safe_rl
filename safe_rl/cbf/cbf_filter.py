from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv


class SafetyGymnasiumCBFFilter:
    """Analytic CBF-QP action filter for Safety-Gymnasium environments.

    For each parallel env, computes the CBF constraint for every sphere hazard
    and analytically projects the nominal action[0] (forward speed) onto the
    safe halfspace. Steering (action[1]) is left unchanged.

    Safety constraint (``velocity_aware=True``, the default):
        h(x)   = ||p - p_obs||^2 - d_safe^2
        one-step lookahead on the displacement, using the MEASURED velocity:
            dp  ~= (v + a_scale * u0 * heading) * dt
            h+  ~= h + 2 (p - p_obs) . dp
        CBF:   h+ - h >= -alpha * h   =>   A*u0 >= b   with
            A = 2 * a_scale * dt * (p - p_obs) . heading
            b = -alpha * h - 2 * dt * (p - p_obs) . v

    The legacy form (``velocity_aware=False``) instead assumed
    ``h_dot = 2 (p - p_obs) . heading * v_scale * action[0]``, i.e. that action[0] is a
    VELOCITY command. Safety-Gymnasium's Point robot is force-actuated, and measurement on
    SafetyPointGoal2 showed action[0] carries essentially no linear information about
    per-step displacement (R^2 = -0.000, corr 0.014) -- so that constraint's gradient points
    in an arbitrary direction and no v_scale can fix it. Evaluated on a reward-greedy level-2
    checkpoint it cut reward 24.2 -> 2.5 while cost stayed 44.8 -> 35.9, and at other gains
    it RAISED cost to 126. Kept only for reproducing those runs.

    Steering (action[1]) is left unchanged in both modes.

    Analytic solution for a single linear constraint: if A*u0_ref < b, set u0 = b/A (clipped
    to action limits). Iterates over all hazards for up to ``max_iter`` passes.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        d_min: float = 0.35,
        v_scale: float = 1.0,
        max_iter: int = 5,
        device: str | torch.device = "cpu",
        velocity_aware: bool = True,
        a_scale: float = 0.00629,   # measured action[0] -> delta-velocity gain, PointGoal2
        dt: float = 0.02,
    ) -> None:
        self.alpha = float(alpha)
        self.d_min = float(d_min)
        self.v_scale = float(v_scale)
        self.max_iter = int(max_iter)
        self.velocity_aware = bool(velocity_aware)
        self.a_scale = float(a_scale)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.last_solve_ms: float = 0.0

    def filter(self, actions: torch.Tensor, env: SafetyGymnasiumVecEnv) -> torch.Tensor:
        """Project actions onto the CBF-safe set.

        Args:
            actions: (num_envs, action_dim) tensor on ``self.device``
            env:     SafetyGymnasiumVecEnv whose sub-envs are wrapped with
                     SGCBFStateWrapper

        Returns:
            Safe actions tensor with same shape/dtype/device as input.
        """
        t0 = time.perf_counter()
        states = env.env.call("get_cbf_state")
        actions_np = actions.detach().cpu().numpy().copy()
        for i, state in enumerate(states):
            actions_np[i] = self._filter_single(actions_np[i], state)
        self.last_solve_ms = (time.perf_counter() - t0) * 1e3
        return torch.as_tensor(actions_np, dtype=actions.dtype, device=self.device)

    def _filter_single(self, action: np.ndarray, state: dict) -> np.ndarray:
        pos = state["pos"]              # (2,)
        vel = state["vel"]              # (2,) measured velocity -- ignored by the legacy form
        heading = state["heading"]      # (2,) unit forward vector
        hazards_pos = state["hazards_pos"]  # (N, 2)
        hazard_size = state["hazard_size"]
        d_safe = self.d_min + hazard_size

        if len(hazards_pos) == 0:
            return action

        action = action.copy()
        for _ in range(self.max_iter):
            changed = False
            for haz in hazards_pos:
                diff = pos - haz          # vector from hazard center to robot
                d_sq = float(np.dot(diff, diff))
                h = d_sq - d_safe**2

                # constraint coefficient: A * action[0] >= b
                if self.velocity_aware:
                    # one-step lookahead with the measured velocity; the action enters only
                    # through the acceleration it produces over the step.
                    A = 2.0 * self.a_scale * self.dt * float(np.dot(diff, heading))
                    b = -self.alpha * h - 2.0 * self.dt * float(np.dot(diff, vel))
                else:
                    A = 2.0 * float(np.dot(diff, heading)) * self.v_scale
                    b = -self.alpha * h

                if abs(A) < 1e-6:
                    continue  # no velocity component toward/away this hazard
                if A * action[0] < b:
                    # project onto constraint boundary: A * u0 = b
                    action[0] = float(np.clip(b / A, -1.0, 1.0))
                    changed = True
            if not changed:
                break
        return action
