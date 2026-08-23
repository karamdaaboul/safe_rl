"""Tests for the continuous geometric margin cost wrapper (Direction 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")


@pytest.fixture(scope="module")
def wrapped_env():
    import safety_gymnasium

    from safe_rl.envs.geom_margin_wrapper import GeometricMarginWrapper

    env = GeometricMarginWrapper(
        safety_gymnasium.make("SafetyPointGoal1-v0"), d_safe=0.4
    )
    yield env
    env.close()


def test_margin_matches_hazard_geometry(wrapped_env) -> None:
    wrapped_env.reset(seed=0)
    task = wrapped_env.unwrapped.task
    for _ in range(5):
        action = wrapped_env.action_space.sample()
        _, _, margin, _, _, info = wrapped_env.step(action)
        min_dist = min(task.agent.dist_xy(pos) for pos in task.hazards.pos)
        expected = max(0.4 - min_dist, -0.4)
        assert margin == pytest.approx(expected, abs=1e-6)
        # Bounded: [margin_min, d_safe] (dist >= 0).
        assert -0.4 <= margin <= 0.4
        # Original binary/contact cost preserved for logging.
        assert "original_cost" in info
        # Positive margin strictly contains the true unsafe set (radius 0.2 < d_safe).
        if np.asarray(info["original_cost"]).sum() > 0:
            assert margin > 0.0


def test_true_episode_cost_reported_on_terminal_step(wrapped_env) -> None:
    wrapped_env.reset(seed=1)
    total = 0.0
    for _ in range(1000):
        _, _, _, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
        total += float(np.asarray(info["original_cost"]).sum())
        if terminated or truncated:
            assert info["true_episode_cost"] == pytest.approx(total, abs=1e-6)
            break
    else:
        pytest.fail("episode did not end within 1000 steps")


def test_rejects_d_safe_inside_hazard_radius() -> None:
    import safety_gymnasium

    from safe_rl.envs.geom_margin_wrapper import GeometricMarginWrapper

    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    try:
        with pytest.raises(ValueError, match="hazard radius"):
            GeometricMarginWrapper(env, d_safe=0.1)
    finally:
        env.close()
