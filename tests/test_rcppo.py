"""Tests for RCPPO (reachability-constrained PPO) and the reachability safety filter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make_reach_policy(num_obs: int = 8, num_actions: int = 2):
    from safe_rl.modules import ActorCriticReachQ

    return ActorCriticReachQ(
        num_obs,
        num_obs,
        num_actions,
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
        reach_critic_kwargs={"hidden_dims": [32, 32]},
    )


# -- Reachability max-backup ---------------------------------------------------


def _make_storage(num_steps: int = 4):
    from safe_rl.storage import RolloutStorageReach

    return RolloutStorageReach(1, num_steps, [3], [3], [2], training_type="saferl", cost_shape=(1,))


def test_reachability_backup_hand_computed() -> None:
    # V_h[t] = (1-g)*c[t] + g*max(c[t], V_h[t+1]); bootstrap V_h[T] = 3.
    storage = _make_storage()
    for t, c in enumerate([0.0, 2.0, 0.0, 1.0]):
        storage.costs[t, 0, 0] = c
    storage.compute_reachability_returns(torch.tensor([[3.0]]), gamma_h=0.9)
    # t3 = 0.1*1 + 0.9*max(1,3) = 2.8; t2 = 0.9*2.8 = 2.52;
    # t1 = 0.1*2 + 0.9*max(2,2.52) = 2.468; t0 = 0.9*2.468 = 2.2212
    expected = torch.tensor([2.2212, 2.468, 2.52, 2.8])
    assert torch.allclose(storage.cost_returns.view(-1), expected, atol=1e-5)
    # Advantages are targets minus (zero-initialized) values, never mean-centered.
    assert torch.allclose(storage.cost_advantages.view(-1), expected, atol=1e-5)


def test_reachability_backup_true_done_grounds_at_cost() -> None:
    storage = _make_storage()
    for t, c in enumerate([0.0, 2.0, 0.0, 1.0]):
        storage.costs[t, 0, 0] = c
    storage.dones[1, 0, 0] = 1  # true termination, no timeout
    storage.compute_reachability_returns(torch.tensor([[3.0]]), gamma_h=0.9)
    # Terminal state: V_h = h(s); recursion does not cross the episode boundary.
    expected = torch.tensor([1.8, 2.0, 2.52, 2.8])
    assert torch.allclose(storage.cost_returns.view(-1), expected, atol=1e-5)


def test_reachability_backup_timeout_bootstraps_from_critic() -> None:
    storage = _make_storage()
    for t, c in enumerate([0.0, 2.0, 0.0, 1.0]):
        storage.costs[t, 0, 0] = c
    storage.dones[2, 0, 0] = 1
    storage.time_outs[2, 0, 0] = 1.0  # truncation: bootstrap from the critic value
    storage.reach_bootstrap[2, 0, 0] = 5.0
    storage.compute_reachability_returns(torch.tensor([[3.0]]), gamma_h=0.9)
    # t2 = 0.1*0 + 0.9*max(0, 5) = 4.5; t1 = 0.1*2 + 0.9*max(2,4.5) = 4.25; t0 = 0.9*4.25
    expected = torch.tensor([3.825, 4.25, 4.5, 2.8])
    assert torch.allclose(storage.cost_returns.view(-1), expected, atol=1e-5)


# -- RCPPO update --------------------------------------------------------------


def test_rcppo_update_smoke() -> None:
    from safe_rl.algorithms import RCPPO

    torch.manual_seed(0)
    num_envs, num_steps, num_obs, num_actions = 4, 8, 8, 2
    policy = _make_reach_policy(num_obs, num_actions)
    alg = RCPPO(
        policy,
        cost_limits=[0.1],
        num_learning_epochs=1,
        num_mini_batches=1,
        gamma_h=0.9,
        schedule="fixed",
        desired_kl=None,
    )
    alg.init_storage("saferl", num_envs, num_steps, [num_obs], [num_obs], [num_actions])

    obs = torch.randn(num_envs, num_obs)
    for _ in range(num_steps):
        alg.act(obs, obs)
        rewards = torch.randn(num_envs)
        costs = torch.rand(num_envs)
        dones = torch.zeros(num_envs)
        alg.process_env_step(rewards, costs, dones, {"time_outs": torch.zeros(num_envs)})
        obs = torch.randn(num_envs, num_obs)

    alg.compute_returns(obs)
    alg.compute_cost_returns(obs)
    loss_dict = alg.update(iteration=0)

    assert "reach_q" in loss_dict
    assert "reach_level_mean" in loss_dict
    for key, value in loss_dict.items():
        assert torch.isfinite(torch.tensor(float(value))), f"non-finite loss {key}={value}"
    assert alg.lambdas[0] >= 0.0


def test_rcppo_signed_margin_keeps_negative_costs() -> None:
    from safe_rl.algorithms import RCPPO

    torch.manual_seed(0)
    num_envs, num_obs, num_actions = 4, 8, 2
    margins = torch.tensor([-0.4, -0.1, 0.0, 0.3])

    for signed in (False, True):
        policy = _make_reach_policy(num_obs, num_actions)
        alg = RCPPO(policy, cost_limits=[0.05], signed_margin=signed, gamma_h=0.9)
        alg.init_storage("saferl", num_envs, 2, [num_obs], [num_obs], [num_actions])
        obs = torch.randn(num_envs, num_obs)
        alg.act(obs, obs)
        alg.process_env_step(torch.zeros(num_envs), margins, torch.zeros(num_envs),
                             {"time_outs": torch.zeros(num_envs)})
        stored = alg.storage.costs[0, :, 0]
        expected = margins if signed else margins.clamp(min=0.0)
        assert torch.allclose(stored, expected), f"signed={signed}: {stored} vs {expected}"


def test_rcppo_scale_cost_advantage_unit_std() -> None:
    from safe_rl.algorithms import RCPPO

    torch.manual_seed(0)
    num_envs, num_steps, num_obs, num_actions = 4, 8, 8, 2
    policy = _make_reach_policy(num_obs, num_actions)
    alg = RCPPO(
        policy, cost_limits=[0.1], scale_cost_advantage=True, gamma_h=0.9,
        num_learning_epochs=1, num_mini_batches=1, schedule="fixed", desired_kl=None,
    )
    alg.init_storage("saferl", num_envs, num_steps, [num_obs], [num_obs], [num_actions])
    obs = torch.randn(num_envs, num_obs)
    for _ in range(num_steps):
        alg.act(obs, obs)
        alg.process_env_step(torch.randn(num_envs), torch.rand(num_envs), torch.zeros(num_envs),
                             {"time_outs": torch.zeros(num_envs)})
        obs = torch.randn(num_envs, num_obs)
    alg.compute_returns(obs)
    alg.compute_cost_returns(obs)

    raw_std = alg.storage.cost_advantages.flatten(0, 1).std(dim=0)
    loss_dict = alg.update(iteration=0)

    # The reported scale is the pre-division std; after division the (consumed)
    # advantages had unit scale.
    assert loss_dict["cost_adv_std"] == pytest.approx(float(raw_std.mean()), rel=1e-5)
    for key, value in loss_dict.items():
        assert torch.isfinite(torch.tensor(float(value))), f"non-finite loss {key}={value}"


def test_rcppo_requires_reach_head_when_training_q() -> None:
    from safe_rl.algorithms import RCPPO
    from safe_rl.modules import ActorCriticCost

    policy = ActorCriticCost(
        8, 8, 2, num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )
    with pytest.raises(ValueError, match="reach_critic"):
        RCPPO(policy, cost_limits=[0.1])


# -- Reachability safety filter ------------------------------------------------


def test_reachability_filter_least_restrictive() -> None:
    from safe_rl.filters import ReachabilitySafetyFilter

    torch.manual_seed(0)
    policy = _make_reach_policy()
    obs = torch.randn(4, 8)
    # In-bounds proposals so the (clamped) proposal candidate equals the input and the
    # argmin-Q_h guarantee below is exact.
    actions = policy.act(obs).clamp(-1.0, 1.0)

    # Threshold too high to trigger: identity pass-through.
    passive = ReachabilitySafetyFilter(policy, threshold=1e9)
    out = passive.filter(actions, obs)
    assert torch.equal(out, actions)
    assert passive.last_intervention_frac == 0.0

    # Threshold low enough that every env is flagged: all actions filtered, in bounds.
    active = ReachabilitySafetyFilter(policy, threshold=-1e9, num_candidates=8)
    out = active.filter(actions, obs)
    assert out.shape == actions.shape
    assert active.last_intervention_frac == 1.0
    assert (out >= -1.0).all() and (out <= 1.0).all()
    assert active.last_solve_ms > 0.0
    # Replaced actions carry the argmin-Q_h candidate: never worse than the proposal.
    q_before = policy.evaluate_reach_q(obs, actions)[:, 0]
    q_after = policy.evaluate_reach_q(obs, out)[:, 0]
    assert (q_after <= q_before + 1e-6).all()


def test_reachability_filter_blend_mode_within_bounds() -> None:
    from safe_rl.filters import ReachabilitySafetyFilter

    torch.manual_seed(1)
    policy = _make_reach_policy()
    obs = torch.randn(4, 8)
    actions = policy.act(obs)
    filt = ReachabilitySafetyFilter(policy, threshold=-1e9, mode="blend", blend_temp=0.5)
    out = filt.filter(actions, obs)
    assert out.shape == actions.shape
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_reachability_filter_requires_reach_head() -> None:
    from safe_rl.filters import ReachabilitySafetyFilter
    from safe_rl.modules import ActorCriticCost

    policy = ActorCriticCost(
        8, 8, 2, num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )
    with pytest.raises(ValueError, match="evaluate_reach_q"):
        ReachabilitySafetyFilter(policy)
