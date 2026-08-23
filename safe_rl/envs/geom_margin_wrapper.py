from __future__ import annotations

import numpy as np

import gymnasium


class GeometricMarginWrapper(gymnasium.Wrapper):
    """Replace the sparse hazard cost with a continuous geometric safety margin.

    Reachability-style safe RL (RCPPO) learns V_h(s) = max_t h(s_t) from the
    per-step margin h. Safety-Gymnasium's hazard cost is zero everywhere outside
    a hazard, so h carries no state information in the safe region and V_h
    saturates to a near-constant — the PointGoal failure mode. This wrapper
    replaces the cost channel with a *signed distance* margin

        h(s) = d_safe - min_i dist_xy(agent, hazard_i center),

    clipped below at ``margin_min``. h > 0 means the agent is within ``d_safe``
    of a hazard center (unsafe buffer zone, since d_safe > hazard radius);
    h <= 0 means it keeps a safe distance, more negative = safer. The margin is
    continuous in the agent position, so V_h gets a real gradient and a
    meaningful zero level set. Pair with ``RCPPO(signed_margin=True)`` so the
    algorithm does not clamp the negative (safe) part away.

    The env's original cost is preserved in ``info["original_cost"]`` and the
    per-episode *true* cost sum is emitted as ``info["true_episode_cost"]`` on
    the terminal step (the vector env forwards it via ``final_info`` for
    logging), so safety remains comparable to unwrapped runs.
    """

    def __init__(self, env: gymnasium.Env, d_safe: float = 0.4, margin_min: float | None = None) -> None:
        super().__init__(env)
        task = env.unwrapped.task
        if not hasattr(task, "hazards") or task.hazards.num == 0:
            raise ValueError(
                f"GeometricMarginWrapper needs a task with hazards; {type(task).__name__} has none."
            )
        if d_safe <= task.hazards.size:
            raise ValueError(
                f"d_safe={d_safe} must exceed the hazard radius {task.hazards.size} "
                "so that h > 0 strictly contains the true unsafe set."
            )
        self.d_safe = float(d_safe)
        # Default: symmetric range [-d_safe, d_safe] (plus the interior up to
        # d_safe at a hazard center) so deep-safe states don't dominate E[V_h].
        self.margin_min = float(margin_min) if margin_min is not None else -float(d_safe)
        self._ep_true_cost = 0.0

    def _margin(self) -> float:
        task = self.env.unwrapped.task
        min_dist = min(task.agent.dist_xy(pos) for pos in task.hazards.pos)
        return max(self.d_safe - min_dist, self.margin_min)

    def reset(self, *, seed: int | None = None, options=None):
        self._ep_true_cost = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self._ep_true_cost += float(np.asarray(cost).sum())
        info["original_cost"] = cost
        if terminated or truncated:
            info["true_episode_cost"] = self._ep_true_cost
            self._ep_true_cost = 0.0
        return obs, reward, self._margin(), terminated, truncated, info
