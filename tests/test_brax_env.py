"""Tests for the BraxVecEnv wrapper (skipped when brax is not installed)."""

import pytest
import torch

brax = pytest.importorskip("brax")

from safe_rl.envs import make_env  # noqa: E402

N = 4


@pytest.fixture(scope="module")
def env():
    return make_env(env_id="BraxAnt", num_envs=N, device="cpu", seed=0, episode_length=50)


def test_reset_and_shapes(env):
    obs, extras = env.reset()
    assert obs.shape == (N, env.num_obs)
    assert torch.isfinite(obs).all()
    assert "observations" in extras
    assert env.num_actions == 8  # brax ant


def test_step_contract(env):
    env.reset()
    actions = torch.zeros(N, env.num_actions)
    obs, rew, dones, extras = env.step(actions)
    assert obs.shape == (N, env.num_obs)
    assert rew.shape == (N,) and torch.isfinite(rew).all()
    assert dones.dtype == torch.bool
    assert "time_outs" in extras and extras["time_outs"].shape == (N,)


def test_time_out_fires_at_episode_length():
    # Fresh env (not the shared fixture): the truncation counter must start
    # from a clean slate regardless of test ordering.
    e = make_env(env_id="BraxAnt", num_envs=N, device="cpu", seed=1, episode_length=12)
    e.reset()
    actions = torch.zeros(N, e.num_actions)
    fired = False
    for _ in range(14):
        _, _, dones, extras = e.step(actions)
        if extras["time_outs"].any():
            fired = True
            assert bool(dones[extras["time_outs"]].all())
            break
    assert fired, "truncation never fired within episode_length steps"
