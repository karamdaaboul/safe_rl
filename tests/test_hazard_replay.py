"""Tests for hazard-stratified replay of the CVPO cost critic.

The cost critic on SafetyPointGoal1 sees a cost signal on ~3.4% of transitions, giving a
Bellman residual whose mean (+0.010) is 36x below its own noise (0.356); it converges to a
near-constant Q_c (calibration slope 0.165 against a target of 1.0). See
codex/cvpo-cost-critic-investigation.md.

``hazard_fraction`` adds a second *sampling view* over the same ReplayStorage -- a hazard
index pool and a safe index pool -- so the cost critic's batch can be held at a chosen
hazard fraction. Transitions are stored exactly once; the pools hold indices. Per-transition
importance weights ``(N_c/N)/(B_c/B)`` undo the resulting distribution shift, so the
objective being minimized is still the uniform-replay one.

These tests pin: the requested composition, the exactness of the correction, pool
bookkeeping across circular overwrite, the degenerate strata, and -- critically -- that the
weights reach the cost critic and nothing else.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS = 6
NUM_ACT = 2


def _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25, **kwargs):
    from safe_rl.storage.replay_storage import ReplayStorage

    return ReplayStorage(
        num_envs=num_envs,
        max_size=max_size,
        obs_shape=[1],
        action_shape=[1],
        device="cpu",
        hazard_fraction=hazard_fraction,
        **kwargs,
    )


def _add(storage, costs, obs_value=0.0):
    """Add one vectorized step whose per-env costs are ``costs``."""
    costs = torch.as_tensor(costs, dtype=torch.float32).view(-1, 1)
    n = costs.shape[0]
    return storage.add(
        torch.full((n, 1), float(obs_value)),
        torch.zeros(n, 1),
        torch.zeros(n),
        torch.zeros(n),
        torch.zeros(n, 1),
        costs=costs,
    )


def _fill(storage, steps, hazard_prob, num_envs=4, generator=None):
    for _ in range(steps):
        costs = (torch.rand(num_envs, 1, generator=generator) < hazard_prob).float()
        _add(storage, costs)


# ---------------------------------------------------------------------------
# A: the requested stratum ratio
# ---------------------------------------------------------------------------


def test_stratified_batch_hits_the_requested_hazard_fraction() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=2000, hazard_fraction=0.25)
    _fill(storage, steps=500, hazard_prob=0.05)  # buffer ~5% hazard

    assert storage.hazard_pool_size + storage.safe_pool_size == storage.size
    assert storage.hazard_pool_size >= 25  # enough to serve the request

    batch = storage.sample(100, stratified=True)
    is_hazard = (batch["costs"] > 0).squeeze(-1)
    assert int(is_hazard.sum()) == 25
    assert int((~is_hazard).sum()) == 75


# ---------------------------------------------------------------------------
# B: the correction is exact for binary costs
# ---------------------------------------------------------------------------


def test_is_weights_exactly_recover_the_buffer_hazard_fraction() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25)
    _fill(storage, steps=100, hazard_prob=0.05)

    true_fraction = storage.hazard_pool_size / storage.size
    # Safe cost is 0 and hazard cost is exactly 1, so the weighted batch mean is exact
    # per draw, not merely unbiased across draws.
    for _ in range(5):
        batch = storage.sample(64, stratified=True)
        estimate = (batch["cost_is_weights"] * batch["costs"]).mean().item()
        assert estimate == pytest.approx(true_fraction, abs=1e-6)


def test_is_weights_average_to_one() -> None:
    """E[w] == 1: the reweighting changes composition, not the objective's scale."""
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25)
    _fill(storage, steps=100, hazard_prob=0.1)

    batch = storage.sample(64, stratified=True)
    assert batch["cost_is_weights"].mean().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.isfinite(batch["cost_is_weights"]).all()
    assert bool((batch["cost_is_weights"] >= 0).all())


# ---------------------------------------------------------------------------
# C: general (non-binary) costs converge rather than matching per batch
# ---------------------------------------------------------------------------


def test_weighted_batch_cost_converges_to_the_buffer_mean() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25)
    for i in range(100):
        # Hazard costs of differing magnitudes -> the per-batch estimate is unbiased but
        # no longer exact, so only the average over many draws is asserted.
        magnitude = float(1 + (i % 4))
        costs = torch.where(torch.rand(4, 1) < 0.1, magnitude, 0.0)
        _add(storage, costs)

    true_mean = storage._data["costs"][: storage.size].mean().item()
    estimates = [
        (b["cost_is_weights"] * b["costs"]).mean().item()
        for b in (storage.sample(64, stratified=True) for _ in range(400))
    ]
    assert sum(estimates) / len(estimates) == pytest.approx(true_mean, rel=0.05)


# ---------------------------------------------------------------------------
# D: circular overwrite in both directions
# ---------------------------------------------------------------------------


def test_pools_track_circular_overwrite_in_both_directions() -> None:
    storage = _make_storage(num_envs=2, max_size=8, hazard_fraction=0.5)

    _add(storage, [1.0, 0.0])
    _add(storage, [0.0, 0.0])
    _add(storage, [1.0, 1.0])
    _add(storage, [0.0, 0.0])
    assert storage.size == 8  # exactly full, no wrap yet
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (3, 5)
    storage.check_pool_invariants()

    # Wrap: slot 0 hazard -> safe and slot 1 safe -> hazard in the same add().
    _add(storage, [0.0, 1.0])
    storage.check_pool_invariants()
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (3, 5)

    # Wrap again: slots 2,3 safe -> hazard.
    _add(storage, [1.0, 1.0])
    storage.check_pool_invariants()
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (5, 3)

    # And back: slots 4,5 hazard -> safe.
    _add(storage, [0.0, 0.0])
    storage.check_pool_invariants()
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (3, 5)

    # Sampled classifications still agree with the stored costs. The request is
    # round(8 * 0.5) = 4 hazards but only 3 exist, so all 3 are taken once and safe
    # fills the rest -- no duplication to reach the target.
    batch = storage.sample(8, stratified=True)
    assert int((batch["costs"] > 0).sum()) == 3
    assert int((batch["costs"] == 0).sum()) == 5


def test_unchanged_class_overwrite_keeps_pools_consistent() -> None:
    """The add() fast path skips slots whose class does not flip -- it must still be right."""
    storage = _make_storage(num_envs=2, max_size=6, hazard_fraction=0.5)
    for _ in range(10):  # many wraps, every write safe -> safe
        _add(storage, [0.0, 0.0])
        storage.check_pool_invariants()
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (0, 6)


def test_pools_rebuild_on_load_state_dict() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=80, hazard_fraction=0.25)
    _fill(storage, steps=30, hazard_prob=0.2)
    expected = (storage.hazard_pool_size, storage.safe_pool_size)

    restored = _make_storage(num_envs=4, max_size=80, hazard_fraction=0.25)
    restored.load_state_dict(storage.state_dict())
    assert (restored.hazard_pool_size, restored.safe_pool_size) == expected
    restored.check_pool_invariants()


# ---------------------------------------------------------------------------
# E: no-hazard burn-in
# ---------------------------------------------------------------------------


def test_no_hazard_buffer_falls_back_to_uniform_with_unit_weights() -> None:
    storage = _make_storage(num_envs=4, max_size=64, hazard_fraction=0.25)
    for _ in range(20):
        _add(storage, [0.0] * 4)

    assert storage.hazard_pool_size == 0
    batch = storage.sample(16, stratified=True)
    assert bool((batch["cost_is_weights"] == 1.0).all())
    assert batch["observations"].shape == (16, 1)


def test_hazard_fraction_zero_preserves_uniform_replay() -> None:
    """Disabled is byte-identical to the old behaviour, and allocates no metadata."""
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=64, hazard_fraction=0.0)
    _fill(storage, steps=20, hazard_prob=0.5)

    assert storage.hazard_pool_size == 0 and storage.safe_pool_size == 0
    assert storage._slot_class is None  # nothing allocated

    torch.manual_seed(123)
    batch = storage.sample(16)
    torch.manual_seed(123)
    reference_indices = torch.randint(0, storage.size, (16,))
    assert torch.equal(batch["observations"], storage._data["observations"][reference_indices])
    assert bool((batch["cost_is_weights"] == 1.0).all())


def test_all_hazard_buffer_falls_back_with_unit_weights() -> None:
    storage = _make_storage(num_envs=4, max_size=64, hazard_fraction=0.25)
    for _ in range(20):
        _add(storage, [1.0] * 4)

    assert storage.safe_pool_size == 0
    batch = storage.sample(16, stratified=True)
    assert bool((batch["cost_is_weights"] == 1.0).all())


# ---------------------------------------------------------------------------
# F: insufficient hazard pool -- no duplication, weights use the ACTUAL composition
# ---------------------------------------------------------------------------


def test_short_hazard_pool_is_not_duplicated_and_weights_use_actual_counts() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=2, max_size=200, hazard_fraction=0.25)
    for i in range(50):
        # Exactly two hazardous transitions in the whole buffer, at distinct obs values.
        _add(storage, [1.0 if i < 2 else 0.0, 0.0], obs_value=i)
    assert storage.hazard_pool_size == 2

    batch = storage.sample(40, stratified=True)  # target 10 hazard, only 2 exist
    is_hazard = (batch["costs"] > 0).squeeze(-1)
    assert int(is_hazard.sum()) == 2  # all available, taken once each

    sampled = batch["observations"][is_hazard].flatten().tolist()
    assert len(sampled) == len(set(sampled))  # no duplication

    n_hazard, n_safe, total = 2, storage.size - 2, storage.size
    expected_hazard = (n_hazard / total) / (2 / 40)  # ACTUAL 2/40, not the configured 0.25
    expected_safe = (n_safe / total) / (38 / 40)
    assert batch["cost_is_weights"][is_hazard].unique().item() == pytest.approx(expected_hazard)
    assert batch["cost_is_weights"][~is_hazard].unique().item() == pytest.approx(expected_safe)


def test_sampling_is_without_replacement() -> None:
    """Drawing the whole buffer must yield each stored transition exactly once."""
    torch.manual_seed(0)
    storage = _make_storage(num_envs=1, max_size=40, hazard_fraction=0.25)
    for i in range(40):
        # obs uniquely identifies the slot, so duplicates would be visible.
        _add(storage, [1.0 if i % 5 == 0 else 0.0], obs_value=i)
    assert (storage.hazard_pool_size, storage.safe_pool_size) == (8, 32)

    for _ in range(10):
        batch = storage.sample(40, stratified=True)
        drawn = sorted(batch["observations"].flatten().tolist())
        assert drawn == [float(i) for i in range(40)]


def test_draw_positions_are_distinct_and_cover_the_pool() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=2, max_size=100, hazard_fraction=0.25)
    for _ in range(50):
        _add(storage, [0.0, 0.0])

    for k in (1, 7, 40, 100):
        positions = storage._draw_positions_wor(100, k)
        assert positions.numel() == k
        assert torch.unique(positions).numel() == k  # no repeats
        assert bool(((positions >= 0) & (positions < 100)).all())

    # k >= pool_size returns the whole pool.
    assert torch.equal(
        torch.sort(storage._draw_positions_wor(10, 25))[0], torch.arange(10)
    )


def test_batch_larger_than_buffer_falls_back_to_uniform() -> None:
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25)
    _fill(storage, steps=5, hazard_prob=0.5)  # only 20 transitions
    batch = storage.sample(64, stratified=True)
    assert batch["observations"].shape[0] == 64
    assert bool((batch["cost_is_weights"] == 1.0).all())


def test_fallback_does_not_report_a_stale_stratified_composition() -> None:
    """A fallback draw must not leave the previous stratified numbers to be logged."""
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25)
    _fill(storage, steps=100, hazard_prob=0.2)

    storage.sample(64, stratified=True)
    assert storage.last_stratified_info["replay_cost_is_weight_hazard"] != 1.0

    storage.clear()
    _fill(storage, steps=2, hazard_prob=0.5)  # 8 transitions, batch of 64 cannot be served
    batch = storage.sample(64, stratified=True)
    assert bool((batch["cost_is_weights"] == 1.0).all())
    info = storage.last_stratified_info
    assert info["replay_cost_is_weight_hazard"] == 1.0
    assert info["replay_cost_is_weight_safe"] == 1.0


# ---------------------------------------------------------------------------
# n-step interaction
# ---------------------------------------------------------------------------


def test_stratified_nstep_batch_matches_uniform_nstep_shape() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25, n_step=10, gamma=0.99)
    _fill(storage, steps=100, hazard_prob=0.05)

    uniform = storage.sample(64)
    stratified = storage.sample(64, stratified=True)
    assert set(uniform) == set(stratified)
    assert "effective_n_steps" in stratified
    assert stratified["effective_n_steps"].shape == (64, 1)
    assert stratified["cost_is_weights"].mean().item() == pytest.approx(1.0, abs=1e-4)
    # The stratum is the START step, but the sampled cost is window-aggregated, so it can
    # exceed the 1-step maximum of 1.0. This is why tests B/C pin n_step=1.
    assert stratified["costs"].max().item() > 1.0


def test_nstep_stratified_repairs_invalid_starts() -> None:
    """Most starts straddle the write head here, forcing the redraw/rebalance path."""
    torch.manual_seed(0)
    storage = _make_storage(num_envs=2, max_size=20, hazard_fraction=0.25, n_step=5, gamma=0.99)
    for i in range(40):  # several wraps of a deliberately tiny ring
        _add(storage, [1.0 if i % 3 == 0 else 0.0, 0.0], obs_value=i)

    valid_rows = storage._valid_start_t()
    assert valid_rows.numel() < storage.max_size // storage.num_envs  # repair is exercised

    for _ in range(5):
        batch = storage.sample(6, stratified=True)
        weights = batch["cost_is_weights"]
        assert torch.isfinite(weights).all()
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-4)
        assert batch["effective_n_steps"].shape == (6, 1)
    storage.check_pool_invariants()


def test_nstep_stratified_reports_start_step_composition() -> None:
    torch.manual_seed(0)
    storage = _make_storage(num_envs=4, max_size=400, hazard_fraction=0.25, n_step=10, gamma=0.99)
    _fill(storage, steps=100, hazard_prob=0.05)

    storage.sample(64, stratified=True)
    info = storage.last_stratified_info
    # Composition is reported from the sampler, not re-derived from aggregated costs.
    assert info["replay_hazard_fraction_batch"] == pytest.approx(0.25)
    assert info["replay_hazard_pool_size"] + info["replay_safe_pool_size"] == storage.size


# ---------------------------------------------------------------------------
# G: the weights reach the cost critic and nothing else
# ---------------------------------------------------------------------------


def _make_cvpo(**overrides):
    from safe_rl.algorithms import CVPO
    from safe_rl.modules import SafeSACActorCritic

    policy = SafeSACActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )
    kwargs = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=2,
        device="cpu",
    )
    kwargs.update(overrides)
    return CVPO(policy, **kwargs)


def _fill_alg(alg, num_envs=4, steps=60, seed=0):
    alg.init_storage(buffer_size=1000, num_envs=num_envs, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    torch.manual_seed(seed)
    for _ in range(steps):
        alg.store_transition(
            torch.randn(num_envs, NUM_OBS),
            torch.rand(num_envs, NUM_ACT) * 2 - 1,
            torch.randn(num_envs),
            torch.zeros(num_envs),
            torch.randn(num_envs, NUM_OBS),
            cost=(torch.rand(num_envs) < 0.1).float(),
        )
    return alg


def test_cost_is_weights_change_only_the_cost_critic_loss() -> None:
    """The same transitions, two weightings: the cost loss moves and nothing else does."""
    import copy

    alg = _fill_alg(_make_cvpo())
    batch = alg.storage.sample(32)
    obs = batch["observations"]

    policy_state = copy.deepcopy(alg.policy.state_dict())
    optim_state = copy.deepcopy(alg.cost_critic_optimizer.state_dict())

    def cost_loss(weights):
        # Restore everything the update mutates so the weighting is the only difference.
        alg.policy.load_state_dict(policy_state)
        alg.cost_critic_optimizer.load_state_dict(optim_state)
        torch.manual_seed(5)
        return alg._update_cost_critic(
            obs, obs, batch["actions"], batch["costs"], batch["dones"],
            batch["next_observations"], obs, cost_is_weights=weights,
        )

    ones = torch.ones(32, 1)
    skewed = torch.cat([torch.full((16, 1), 0.25), torch.full((16, 1), 1.75)])
    assert cost_loss(skewed) != pytest.approx(cost_loss(ones), rel=1e-6)

    # The real isolation claim: a cost-critic update leaves the reward critics and the
    # actor bit-identical. Restore first so the comparison starts from a clean state.
    alg.policy.load_state_dict(policy_state)
    alg.cost_critic_optimizer.load_state_dict(optim_state)
    untouched = {
        name: param.detach().clone()
        for name, param in alg.policy.named_parameters()
        if not name.startswith("cost_critic")
    }
    torch.manual_seed(5)
    alg._update_cost_critic(
        obs, obs, batch["actions"], batch["costs"], batch["dones"],
        batch["next_observations"], obs, cost_is_weights=skewed,
    )
    for name, before in untouched.items():
        assert torch.equal(before, dict(alg.policy.named_parameters())[name]), name


def test_unit_weights_reproduce_the_unweighted_cost_loss_exactly() -> None:
    alg = _fill_alg(_make_cvpo())
    batch = alg.storage.sample(32)
    obs = batch["observations"]

    import copy

    # The update samples next actions stochastically and takes an optimizer step, so
    # both the RNG and the optimizer state must be restored for an exact comparison.
    policy_state = copy.deepcopy(alg.policy.state_dict())
    optim_state = copy.deepcopy(alg.cost_critic_optimizer.state_dict())

    def run(weights):
        alg.policy.load_state_dict(policy_state)
        alg.cost_critic_optimizer.load_state_dict(optim_state)
        torch.manual_seed(11)
        return alg._update_cost_critic(
            obs, obs, batch["actions"], batch["costs"], batch["dones"],
            batch["next_observations"], obs, cost_is_weights=weights,
        )

    assert run(torch.ones(32, 1)) == pytest.approx(run(None), rel=1e-6)


def test_stratification_leaves_the_reward_critic_batch_uniform() -> None:
    """The cost critic gets its own batch, so the reward critic sees identical data."""
    # Seed the policy construction too, or the two runs start from different weights.
    torch.manual_seed(3)
    plain = _fill_alg(_make_cvpo())
    torch.manual_seed(3)
    stratified = _fill_alg(_make_cvpo(hazard_fraction=0.25))

    torch.manual_seed(7)
    plain_info = plain.update(current_costs=[30.0])
    torch.manual_seed(7)
    stratified_info = stratified.update(current_costs=[30.0])

    # Same seed, same stored transitions, and the reward critic draws first from the same
    # uniform sampler -> identical reward-critic loss.
    assert stratified_info["critic"] == pytest.approx(plain_info["critic"], rel=1e-6)
    # The cost critic trained on a different (stratified, reweighted) batch.
    assert stratified_info["cost_critic"] != pytest.approx(plain_info["cost_critic"], rel=1e-6)


def test_cvpo_logs_replay_diagnostics() -> None:
    alg = _fill_alg(_make_cvpo(hazard_fraction=0.25))
    alg.update(current_costs=[30.0])
    info = alg.get_penalty_info()
    for key in (
        "replay_hazard_fraction_buffer",
        "replay_hazard_fraction_batch",
        "replay_cost_is_weight_hazard",
        "replay_cost_is_weight_safe",
        "replay_hazard_pool_size",
        "replay_safe_pool_size",
    ):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key]))
    assert info["replay_hazard_fraction_batch"] == pytest.approx(0.25)


def test_cvpo_without_hazard_fraction_logs_no_replay_keys() -> None:
    alg = _fill_alg(_make_cvpo())
    alg.update(current_costs=[30.0])
    info = alg.get_penalty_info()
    assert not [key for key in info if key.startswith("replay_")]


def test_replay_diagnostics_route_to_the_replay_namespace() -> None:
    from safe_rl.utils.logger import Logger

    route = Logger._route_key
    assert route(None, "replay_hazard_pool_size") == "Replay/hazard_pool_size"
    assert route(None, "replay_hazard_fraction_batch") == "Replay/hazard_fraction_batch"
    # Existing conventions are unaffected.
    assert route(None, "cost_critic_loss") == "Loss/cost_critic"
    assert route(None, "lambda_mean") == "SafeRL/lambda_mean"
    assert route(None, "eta") == "Train/eta"
