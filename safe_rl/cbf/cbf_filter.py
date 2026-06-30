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

    Safety constraint:
        h(x)  = ||p - p_obs||^2 - d_safe^2
        h_dot = 2*(p - p_obs) . heading * v_scale * action[0]
        CBF:   h_dot + alpha*h >= 0  =>  A*u0 >= b

    Analytic solution for single linear constraint: if A*u0_ref < b,
    set u0 = b/A (clipped to action limits).  Iterates over all hazards for
    up to ``max_iter`` passes.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        d_min: float = 0.35,
        v_scale: float = 1.0,
        max_iter: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.alpha = float(alpha)
        self.d_min = float(d_min)
        self.v_scale = float(v_scale)
        self.max_iter = int(max_iter)
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
