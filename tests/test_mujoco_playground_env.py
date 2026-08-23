"""Tests for MujocoPlaygroundVecEnv (skipped when mujoco_playground is not installed).

Beyond the usual VecEnv contract, two regressions are pinned here because each would
silently corrupt the comparison against the paper's DMC numbers rather than crash:

* ``get_observations()`` must not reset. The underlying
  ``RSLRLBraxWrapper.get_observations()`` *does*, and our runner calls it at learn start
  and after every periodic eval.
* ``extras["observations"]`` must not carry a ``None`` ``critic`` entry. The wrapper
  emits that key for every symmetric-observation task, and ``OnPolicyRunner`` treats the
  key's presence as "privileged observations exist".
"""

import pytest
import torch

pytest.importorskip("mujoco_playground")

from safe_rl.envs import make_env  # noqa: E402

N = 4
TASK = "MjxCartpoleBalance"  # smallest DMC sim; obs 5, act 1
EPISODE_LENGTH = 12


@pytest.fixture(scope="module")
def env():
    return make_env(env_id=TASK, num_envs=N, device="cpu", seed=0, episode_length=EPISODE_LENGTH)


def test_reset_and_shapes(env):
    obs, extras = env.reset()
    assert obs.shape == (N, env.num_obs)
    assert torch.isfinite(obs).all()
    assert "observations" in extras
    assert env.num_actions == 1
    assert env.max_episode_length == EPISODE_LENGTH


def test_step_contract(env):
    env.reset()
    obs, rew, dones, extras = env.step(torch.zeros(N, env.num_actions))
    assert obs.shape == (N, env.num_obs)
    assert rew.shape == (N,) and torch.isfinite(rew).all()
    assert dones.dtype == torch.bool and dones.shape == (N,)
    assert "time_outs" in extras and extras["time_outs"].shape == (N,)


def test_no_none_observation_entries(env):
    """A None `critic` entry would be read as a privileged-obs tensor by the runner."""
    _, extras = env.reset()
    assert all(v is not None for v in extras["observations"].values())
    _, _, _, extras = env.step(torch.zeros(N, env.num_actions))
    assert all(v is not None for v in extras["observations"].values())
    if not env.asymmetric_obs:
        assert "critic" not in extras["observations"]


def test_get_observations_does_not_reset_or_step(env):
    """The regression that would silently reset training ~20x per run."""
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(N, env.num_actions))
    lengths_before = env.episode_length_buf.clone()

    obs_a, _ = env.get_observations()
    obs_b, _ = env.get_observations()

    assert torch.equal(obs_a, obs_b), "get_observations must be idempotent"
    assert torch.equal(env.episode_length_buf, lengths_before), "it must not advance the env"
    assert int(env.episode_length_buf[0]) == 3, "the episode counter must be preserved"


def test_time_out_fires_at_episode_length():
    e = make_env(env_id=TASK, num_envs=N, device="cpu", seed=1, episode_length=EPISODE_LENGTH)
    e.reset()
    fired = -1
    for t in range(1, EPISODE_LENGTH + 3):
        _, _, _, extras = e.step(torch.zeros(N, e.num_actions))
        if extras["time_outs"].any():
            fired = t
            break
    assert fired == EPISODE_LENGTH, f"truncation fired at {fired}, want {EPISODE_LENGTH}"
    assert int(e.episode_length_buf[0]) == 0, "counter must reset on truncation"


def test_episode_length_buf_tracks_steps(env):
    env.reset()
    assert int(env.episode_length_buf[0]) == 0
    env.step(torch.zeros(N, env.num_actions))
    assert int(env.episode_length_buf[0]) == 1


def test_eval_twin_is_separate_and_optional():
    plain = make_env(env_id=TASK, num_envs=N, device="cpu", seed=0, episode_length=EPISODE_LENGTH)
    assert plain.eval_env is None, "no twin unless num_eval_envs is set"

    with_twin = make_env(
        env_id=TASK, num_envs=N, device="cpu", seed=0, episode_length=EPISODE_LENGTH, num_eval_envs=2
    )
    twin = with_twin.eval_env
    assert twin is not None and twin.num_envs == 2
    assert twin is with_twin.eval_env, "the twin must be built once and cached"
    assert twin.eval_env is None, "the twin must not build a twin of its own"

    # Stepping the twin must leave the training env untouched.
    with_twin.reset()
    twin.reset()
    twin.step(torch.zeros(2, twin.num_actions))
    assert int(with_twin.episode_length_buf[0]) == 0
