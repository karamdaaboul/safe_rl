from __future__ import annotations

import numpy as np
import gymnasium


class SGCBFStateWrapper(gymnasium.Wrapper):
    """Gymnasium wrapper exposing privileged CBF state for Safety-Gymnasium envs.

    Intended for use with the async vector env: each worker process wraps its
    sub-env with this class so the main process can retrieve per-env physical
    state via ``env.call("get_cbf_state")``.
    """

    def get_cbf_state(self) -> dict:
        """Return dict with robot state and obstacle positions needed by CBF-QP."""
        task = self.env.unwrapped.task
        agent = task.agent

        pos = np.array(agent.pos[:2], dtype=np.float64)
        vel = np.array(agent.vel[:2], dtype=np.float64)
        # agent.mat is a 3x3 rotation matrix; row 0 is the forward (local x) axis
        heading = np.array(agent.mat[0, :2], dtype=np.float64)
        heading_norm = float(np.linalg.norm(heading))
        if heading_norm > 1e-6:
            heading = heading / heading_norm

        haz = task.hazards
        hazards_pos = np.array([p[:2] for p in haz.pos], dtype=np.float64)  # (N, 2)
        hazard_size = float(haz.size)

        return {
            "pos": pos,
            "vel": vel,
            "heading": heading,
            "hazards_pos": hazards_pos,
            "hazard_size": hazard_size,
        }
