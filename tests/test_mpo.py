"""Tests for MPO (Maximum a Posteriori Policy Optimization)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS = 6
NUM_ACT = 2


def _make_policy(num_obs: int = NUM_OBS, num_actions: int = NUM_ACT):
    from safe_rl.modules import SACActorCritic

    return SACActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
    )


def _make_mpo(**overrides):
    from safe_rl.algorithms import MPO

    kwargs = dict(
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        device="cpu",
    )
    kwargs.update(overrides)
    return MPO(_make_policy(), **kwargs)


def _fill_buffer(alg, n: int = 128) -> None:
    alg.init_storage(buffer_size=1000, num_envs=1, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(n):
        obs = torch.randn(1, NUM_OBS)
        action = torch.rand(1, NUM_ACT) * 2 - 1
        reward = torch.randn(1)
        done = torch.zeros(1)
        next_obs = torch.randn(1, NUM_OBS)
        alg.store_transition(obs, action, reward, done, next_obs)


def test_mpo_builds_with_frozen_target_actor() -> None:
    alg = _make_mpo()
    # A dedicated frozen target actor must exist for the E-step / M-step KL.
    assert hasattr(alg, "actor_target")
    assert all(not p.requires_grad for p in alg.actor_target.parameters())
    # SAC entropy is disabled — exploration comes from the E-step KL trust region.
    assert not alg.auto_entropy_tuning


def test_mpo_eta_solver_returns_positive() -> None:
    import numpy as np

    alg = _make_mpo()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(16, 32))
    eta = alg._solve_eta(q)
    assert eta > 0
    assert np.isfinite(eta)


def test_mpo_estep_weights_are_a_distribution() -> None:
    # The variational weights are a softmax over the N candidate actions per state,
    # so each state's column must sum to 1.
    alg = _make_mpo(sample_action_num=16)
    obs = torch.randn(32, NUM_OBS)
    with torch.no_grad():
        mean_old, log_std_old = alg.actor_target(obs)
        dist_old = torch.distributions.Normal(mean_old, log_std_old.exp())
        x = dist_old.sample((16,))
        actions = torch.tanh(x)
        cobs = obs.unsqueeze(0).expand(16, -1, -1).reshape(16 * 32, -1)
        q1, q2 = alg.policy.evaluate_q(cobs, actions.reshape(16 * 32, -1))
        q = torch.min(q1, q2).reshape(16, 32)
        eta = alg._solve_eta(q.numpy().astype("float64"))
        weights = torch.softmax(q / eta, dim=0)
    assert torch.allclose(weights.sum(dim=0), torch.ones(32), atol=1e-5)


def test_mpo_update_step_runs_and_is_finite() -> None:
    alg = _make_mpo(num_updates_per_step=2)
    _fill_buffer(alg, n=128)
    info = alg.update()
    for key in ("critic", "actor"):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key]))
    penalty = alg.get_penalty_info()
    assert penalty["eta"] > 0
    for key in ("kl_mean", "kl_var", "alpha_mean", "alpha_var"):
        assert key in penalty
        assert torch.isfinite(torch.tensor(penalty[key]))


def test_mpo_update_moves_actor_params() -> None:
    alg = _make_mpo(num_updates_per_step=3)
    _fill_buffer(alg, n=128)
    before = [p.detach().clone() for p in alg.policy.actor.parameters()]
    alg.update()
    after = list(alg.policy.actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
