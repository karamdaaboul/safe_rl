"""Tests for the CVPO feasible threshold homotopy, measured q_target, and the
infeasibility / lambda-saturation diagnostics."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

NUM_OBS = 6
NUM_ACT = 2


def _make_policy(num_obs: int = NUM_OBS, num_actions: int = NUM_ACT):
    from safe_rl.modules import SafeSACActorCritic

    return SafeSACActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )


def _make_cvpo(**overrides):
    from safe_rl.algorithms import CVPO

    kwargs = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_horizon=1000,
        device="cpu",
    )
    kwargs.update(overrides)
    return CVPO(_make_policy(), **kwargs)


def _fill_buffer(alg, n: int = 128) -> None:
    alg.init_storage(buffer_size=1000, num_envs=1, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(n):
        alg.store_transition(
            torch.randn(1, NUM_OBS),
            torch.rand(1, NUM_ACT) * 2 - 1,
            torch.randn(1),
            torch.zeros(1),
            torch.randn(1, NUM_OBS),
            cost=torch.rand(1),
        )


def _report_costs(alg, value: float, times: int) -> None:
    for _ in range(times):
        alg.update_lagrangian_multipliers([value])


# --------------------------------------------------------------------------------------
# 1. The off-path is inert
# --------------------------------------------------------------------------------------


def test_homotopy_off_leaves_every_threshold_path_on_qc_thres() -> None:
    alg = _make_cvpo()
    assert alg._effective_thres() == alg.qc_thres
    # Even after being fed cost levels, nothing moves while the switch is off.
    alg._update_homotopy_threshold(c_now=99.0)
    assert alg._effective_thres() == alg.qc_thres
    assert alg._current_q_target() == alg.qc_thres


def test_homotopy_off_keeps_the_actor_update_bit_identical() -> None:
    def run(seed: int, **overrides):
        torch.manual_seed(seed)
        alg = _make_cvpo(**overrides)
        _fill_buffer(alg, n=96)
        torch.manual_seed(seed + 1)
        alg.update(current_costs=[40.0])
        return [p.detach().clone() for p in alg.policy.actor.parameters()]

    baseline = run(0)
    again = run(0)
    for a, b in zip(baseline, again):
        assert torch.equal(a, b), "harness itself is not deterministic"

    # Explicitly-off flags must reproduce the default path exactly.
    off = run(0, qc_thres_homotopy=False, qc_target_ema=False)
    for a, b in zip(baseline, off):
        assert torch.equal(a, b)


def test_homotopy_loosens_the_near_term_ask_when_the_policy_is_over_budget() -> None:
    # qc_thres well below what the (freshly initialised) cost critic reads, so the policy is
    # "over budget" and the static threshold is the unreachable ask the homotopy exists to fix.
    def run(seed: int, **overrides):
        torch.manual_seed(seed)
        alg = _make_cvpo(qc_thres=1e-3, **overrides)
        _fill_buffer(alg, n=96)
        torch.manual_seed(seed + 1)
        for _ in range(3):
            alg.update(current_costs=[40.0])
        return alg

    base = run(0)
    homo = run(0, qc_thres_homotopy=True, homotopy_cnow_ema=0.0)

    assert base._effective_thres() == pytest.approx(1e-3), "baseline holds the static ask"
    info = homo.get_penalty_info()
    assert homo._effective_thres() > 1e-3, "the homotopy must loosen the near-term ask"
    # And the ask it settles on is just under where the policy actually sits.
    assert info["c_now_over_thres"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------------------
# 2-5. The homotopy rule itself
# --------------------------------------------------------------------------------------


def test_threshold_asks_for_a_reachable_reduction_when_far_over_budget() -> None:
    alg = _make_cvpo(qc_thres_homotopy=True, homotopy_beta=0.005, homotopy_cnow_ema=0.0)
    c_now = 50.0 * alg.qc_thres  # hopelessly over the static target
    alg._update_homotopy_threshold(c_now)
    thresh = alg._effective_thres()
    assert thresh == pytest.approx(0.995 * c_now)
    assert thresh < c_now, "the ask must be below where the policy sits"
    assert thresh > alg.qc_thres, "and above the real target, or it is not a homotopy"


def test_threshold_clamps_to_q_target_once_the_policy_is_feasible() -> None:
    alg = _make_cvpo(qc_thres_homotopy=True, homotopy_cnow_ema=0.0)
    alg._update_homotopy_threshold(c_now=0.1 * alg.qc_thres)
    # Below the target the real budget applies; the homotopy must not ask for less.
    assert alg._effective_thres() == pytest.approx(alg.qc_thres)


def test_ratchet_is_monotone_while_cost_is_falling() -> None:
    # Monotone *while feasible* is the guarantee. It is deliberately not monotone when C_now
    # rises: beta_max then loosens the threshold just enough to keep the ask reachable, which
    # is the whole point of clamping the ask rather than the level.
    alg = _make_cvpo(qc_thres_homotopy=True, homotopy_ratchet=True, homotopy_cnow_ema=0.0)
    target = alg.qc_thres
    seen = []
    for c_now in [10.0 * target, 8.0 * target, 4.0 * target, 3.0 * target]:
        alg._update_homotopy_threshold(c_now)
        seen.append(alg._effective_thres())
    assert seen == sorted(seen, reverse=True), f"ratchet rose again: {seen}"


def test_ratchet_never_asks_for_less_than_q_target() -> None:
    alg = _make_cvpo(qc_thres_homotopy=True, homotopy_ratchet=True, homotopy_cnow_ema=0.0)
    target = alg.qc_thres
    for c_now in [10.0 * target, 8.0 * target, 4.0 * target, 9.0 * target, 0.01 * target]:
        alg._update_homotopy_threshold(c_now)
        assert alg._effective_thres() >= target - 1e-12


def test_without_the_ratchet_the_threshold_follows_cost_back_up() -> None:
    alg = _make_cvpo(qc_thres_homotopy=True, homotopy_ratchet=False, homotopy_cnow_ema=0.0)
    target = alg.qc_thres
    alg._update_homotopy_threshold(4.0 * target)
    low = alg._effective_thres()
    alg._update_homotopy_threshold(20.0 * target)
    assert alg._effective_thres() > low


def test_ratchet_floor_tracks_a_rising_q_target() -> None:
    # The floor is applied outside the min(): min(prev, max(q_target, .)) alone would leave
    # the threshold stranded below a q_target that has since risen.
    alg = _make_cvpo(
        qc_thres_homotopy=True,
        qc_target_ema=True,
        qc_target_warmup_reports=1,
        qc_target_ema_horizon=1,
        qc_target_jc_ema=1.0,
        qc_target_max_frac=1e6,
    )
    _report_costs(alg, 25.0, 2)  # J_c == the limit, so q_target == EMA[C_now]
    used = alg._update_homotopy_threshold(c_now=1.0)
    used = alg._update_homotopy_threshold(c_now=1.0)
    assert alg._effective_thres() >= used - 1e-12

    # A large jump in C_now raises q_target with it; the ratchet must not sit underneath it.
    used = alg._update_homotopy_threshold(c_now=100.0)
    assert alg._effective_thres() >= used - 1e-12


def test_cnow_ema_smooths_the_ratchet_against_a_single_noisy_batch() -> None:
    raw = _make_cvpo(qc_thres_homotopy=True, homotopy_cnow_ema=0.0)
    smoothed = _make_cvpo(qc_thres_homotopy=True, homotopy_cnow_ema=0.01)
    target = raw.qc_thres
    for alg in (raw, smoothed):
        alg._update_homotopy_threshold(20.0 * target)
        alg._update_homotopy_threshold(5.0 * target)  # one-off dip
    # The raw ratchet locks onto the dip; the smoothed one barely moves.
    assert raw._effective_thres() < smoothed._effective_thres()


def test_beta_max_keeps_the_ratchet_ask_reachable_when_cost_rises() -> None:
    # The failure a bare min(thresh_prev, .) allows: the threshold is monotone, but once C_now
    # climbs away from it the per-update demand grows without bound and the E-step is asked for
    # something the trust region cannot deliver -- the original bug, arriving slowly.
    alg = _make_cvpo(
        qc_thres_homotopy=True,
        homotopy_beta=0.005,
        homotopy_beta_max=0.02,
        homotopy_cnow_ema=0.0,
    )
    target = alg.qc_thres
    alg._update_homotopy_threshold(4.0 * target)  # ratchet locks near 4*target
    alg._update_homotopy_threshold(40.0 * target)  # cost blows up tenfold
    ask = 1.0 - alg._effective_thres() / (40.0 * target)
    assert ask <= 0.02 + 1e-9, f"per-update ask grew to {ask:.4f}, past beta_max"


def test_beta_max_does_not_loosen_the_ratchet_while_cost_is_falling() -> None:
    alg = _make_cvpo(
        qc_thres_homotopy=True, homotopy_beta=0.005, homotopy_beta_max=0.02, homotopy_cnow_ema=0.0
    )
    target = alg.qc_thres
    seen = []
    for c_now in [20.0 * target, 15.0 * target, 10.0 * target, 5.0 * target]:
        alg._update_homotopy_threshold(c_now)
        seen.append(alg._effective_thres())
    assert seen == sorted(seen, reverse=True), f"monotone while feasible violated: {seen}"


def test_beta_max_below_beta_is_rejected() -> None:
    with pytest.raises(ValueError, match="homotopy_beta_max must be >="):
        _make_cvpo(homotopy_beta=0.05, homotopy_beta_max=0.01)


# --------------------------------------------------------------------------------------
# 6-7. Measured q_target
# --------------------------------------------------------------------------------------


def test_jc_ema_horizon_is_derived_to_match_the_cnow_horizon() -> None:
    # The two EMAs tick in different clocks: C_now once per gradient update, J_c once per
    # update() call. Left unset, the J_c weight must be derived so the horizons agree.
    alg = _make_cvpo(qc_target_ema=True, qc_target_ema_horizon=200, num_updates_per_step=8)
    assert alg.qc_target_jc_ema == pytest.approx(8 / 200)
    # An explicit value is still honoured.
    explicit = _make_cvpo(qc_target_ema=True, qc_target_ema_horizon=200, qc_target_jc_ema=0.05)
    assert explicit.qc_target_jc_ema == pytest.approx(0.05)


def test_q_target_rise_is_rate_limited_but_falls_are_not() -> None:
    alg = _make_cvpo(
        qc_target_ema=True,
        qc_target_ema_horizon=1,
        qc_target_jc_ema=1.0,
        qc_target_warmup_reports=1,
        qc_target_max_rise=0.01,
        qc_target_min_frac=1e-6,
        qc_target_max_frac=1e6,
    )
    _report_costs(alg, 25.0, 2)
    first = alg._update_homotopy_threshold(c_now=1.0)  # first valid value accepted outright
    assert first == pytest.approx(1.0)

    # A 100x jump in the raw target may only come through 1% at a time.
    second = alg._update_homotopy_threshold(c_now=100.0)
    assert second == pytest.approx(1.01)
    # Falls are unclamped: tightening is the direction the method wants.
    third = alg._update_homotopy_threshold(c_now=0.05)
    assert third == pytest.approx(0.05)


def test_q_target_rise_clamp_can_be_disabled() -> None:
    alg = _make_cvpo(
        qc_target_ema=True,
        qc_target_ema_horizon=1,
        qc_target_jc_ema=1.0,
        qc_target_warmup_reports=1,
        qc_target_max_rise=0.0,
        qc_target_max_frac=1e6,
    )
    _report_costs(alg, 25.0, 2)
    alg._update_homotopy_threshold(c_now=1.0)
    assert alg._update_homotopy_threshold(c_now=100.0) == pytest.approx(100.0)


# --------------------------------------------------------------------------------------
# Spread diagnostics: is the policy asked to trade off, or only to duck?
# --------------------------------------------------------------------------------------


def test_spread_ratio_measures_action_spread_not_level() -> None:
    # Only the spread across candidate actions survives the per-state softmax: adding a
    # constant to Q_c multiplies the weights by a per-state constant that cancels in the
    # normalisation. The diagnostic must be blind to that constant.
    def spread_of(qc_offset: float) -> float:
        torch.manual_seed(0)
        alg = _make_cvpo()
        _fill_buffer(alg, n=96)
        torch.manual_seed(1)
        alg.update(current_costs=[40.0])
        info = alg.get_penalty_info()
        return info["estep_spread_ratio"]

    assert spread_of(0.0) == pytest.approx(spread_of(1000.0))


def test_lambda_balanced_is_the_multiplier_that_equalises_the_two_terms() -> None:
    torch.manual_seed(0)
    alg = _make_cvpo()
    _fill_buffer(alg, n=96)
    torch.manual_seed(1)
    alg.update(current_costs=[40.0])
    info = alg.get_penalty_info()
    # At lambda == lambda_balanced the spread ratio would be ~1. Not an exact identity: a
    # median of ratios is not the reciprocal of the median of the reciprocals once the sample
    # count is even and the median averages two neighbours.
    assert info["estep_spread_ratio"] == pytest.approx(info["lambda_over_balanced"], rel=1e-2)
    assert info["lambda_balanced"] > 0.0


# --------------------------------------------------------------------------------------
# (continued) Measured q_target
# --------------------------------------------------------------------------------------


def test_q_target_converges_to_cost_lim_times_cbar_over_jbar() -> None:
    alg = _make_cvpo(
        qc_target_ema=True,
        qc_target_ema_horizon=1,  # alpha = 1 -> the EMA is the last value
        qc_target_jc_ema=1.0,
        qc_target_warmup_reports=1,
        qc_target_max_frac=1e6,
    )
    _report_costs(alg, 50.0, 2)  # EMA[J_c] = 50
    alg._update_homotopy_threshold(c_now=4.0)  # EMA[C_now] = 4
    # 25 * 4 / 50 = 2.0
    assert alg._current_q_target() == pytest.approx(2.0)


def test_q_target_reproduces_the_episodic_overshoot_ratio() -> None:
    # The property the critic-based target exists for: C_now / q_target == J_c / cost_lim.
    alg = _make_cvpo(
        qc_target_ema=True,
        qc_target_ema_horizon=1,
        qc_target_jc_ema=1.0,
        qc_target_warmup_reports=1,
        qc_target_max_frac=1e6,
    )
    j_c, c_now = 46.475, 2.123  # the measured SafetyPointGoal1 baseline numbers
    _report_costs(alg, j_c, 2)
    alg._update_homotopy_threshold(c_now=c_now)
    assert c_now / alg._current_q_target() == pytest.approx(j_c / 25.0)


def test_q_target_falls_back_to_the_static_threshold_during_warmup() -> None:
    alg = _make_cvpo(qc_target_ema=True, qc_target_warmup_reports=10, qc_target_jc_ema=1.0)
    _report_costs(alg, 50.0, 9)
    alg._update_homotopy_threshold(c_now=4.0)
    assert alg._current_q_target() == pytest.approx(alg.qc_thres)
    _report_costs(alg, 50.0, 1)
    assert alg._current_q_target() != pytest.approx(alg.qc_thres)


def test_q_target_is_clamped_at_both_band_edges() -> None:
    def target_for(c_now: float) -> float:
        alg = _make_cvpo(
            qc_target_ema=True,
            qc_target_ema_horizon=1,
            qc_target_jc_ema=1.0,
            qc_target_warmup_reports=1,
            qc_target_min_frac=0.5,
            qc_target_max_frac=2.0,
        )
        _report_costs(alg, 25.0, 2)
        alg._update_homotopy_threshold(c_now)
        return alg._current_q_target(), alg._qc_thres_initial

    low, static = target_for(1e-6)
    assert low == pytest.approx(0.5 * static)
    high, static = target_for(1e6)
    assert high == pytest.approx(2.0 * static)


def test_q_target_ema_is_inert_when_off() -> None:
    alg = _make_cvpo(qc_target_ema=False)
    _report_costs(alg, 50.0, 20)
    alg._update_homotopy_threshold(c_now=4.0)
    assert alg._current_q_target() == pytest.approx(alg.qc_thres)
    assert alg._realized_cost_ema is None, "the cost EMA must stay off with both flags off"


def test_q_target_ema_and_frozen_qc_scale_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_cvpo(qc_target_ema=True, use_measured_qc_scale=True)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"homotopy_beta": 1.0}, "homotopy_beta"),
        ({"homotopy_cnow_ema": 1.5}, "homotopy_cnow_ema"),
        ({"qc_target_ema_horizon": 0}, "qc_target_ema_horizon"),
        ({"qc_target_jc_ema": 0.0}, "qc_target_jc_ema"),
        ({"qc_target_min_frac": 0.0}, "qc_target_min_frac"),
        ({"qc_target_min_frac": 2.0, "qc_target_max_frac": 1.0}, "qc_target_min_frac"),
        ({"feasibility_probe_interval": -1}, "feasibility_probe_interval"),
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _make_cvpo(**kwargs)


# --------------------------------------------------------------------------------------
# 8. Lambda saturation accounting
# --------------------------------------------------------------------------------------


def test_lambda_saturation_fraction_tracks_time_spent_at_the_cap() -> None:
    alg = _make_cvpo(lambda_max=4.0)
    alg.lam = alg.lambda_max
    for _ in range(10):
        alg._track_lambda_saturation()
    assert alg.get_penalty_info()["lambda_at_cap_frac"] == pytest.approx(1.0)
    assert alg._lam_at_cap == 1.0

    alg.lam = 0.5
    for _ in range(10):
        alg._track_lambda_saturation()
    info = alg.get_penalty_info()
    # The cumulative figure is a running total and cannot fall to zero...
    assert info["lambda_at_cap_frac"] == pytest.approx(0.5)
    # ...but the EMA must have started decaying, which is why both are reported.
    assert info["lambda_at_cap_frac_ema"] < 1.0
    assert info["lambda_at_cap"] == 0.0


def test_lambda_max_zero_is_not_reported_as_saturation() -> None:
    # lambda_max = 0 pins lambda at zero to disable the constraint (the passive arms).
    alg = _make_cvpo(lambda_max=0.0)
    alg.lam = 0.0
    alg._track_lambda_saturation()
    assert alg.get_penalty_info()["lambda_at_cap_frac"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# 9-10. Sampled-support feasibility probe
# --------------------------------------------------------------------------------------


def test_reachable_min_is_between_the_per_state_min_and_the_mean() -> None:
    alg = _make_cvpo(dual_constraint=0.1)
    rng = np.random.default_rng(0)
    qc = np.abs(rng.normal(loc=3.0, scale=1.0, size=(32, 64)))
    reachable, kl_at = alg._reachable_qc_min(qc)
    assert float(qc.min(axis=0).mean()) <= reachable <= float(qc.mean())
    assert kl_at <= alg.eps_dual + 1e-6, "the probe must stay inside the KL budget"


def test_reachable_min_is_the_constant_when_cost_does_not_vary_across_actions() -> None:
    # No reweighting can move a constant, so the floor is that constant regardless of eps.
    alg = _make_cvpo()
    qc = np.full((16, 8), 3.7)
    reachable, _ = alg._reachable_qc_min(qc)
    assert reachable == pytest.approx(3.7)


def test_reachable_min_falls_as_the_kl_budget_grows() -> None:
    rng = np.random.default_rng(1)
    qc = np.abs(rng.normal(loc=3.0, scale=1.0, size=(32, 64)))
    tight, _ = _make_cvpo(dual_constraint=0.01)._reachable_qc_min(qc)
    loose, _ = _make_cvpo(dual_constraint=1.0)._reachable_qc_min(qc)
    assert loose < tight


def test_infeasible_e_step_is_flagged() -> None:
    alg = _make_cvpo(dual_constraint=0.1)
    rng = np.random.default_rng(2)
    qc = np.abs(rng.normal(loc=10.0, scale=0.5, size=(32, 64)))  # every candidate is costly
    alg._probe_feasibility(qc, thres=1.0)  # unreachable inside the trust region
    info = alg.get_penalty_info()
    assert info["estep_feasible"] == 0.0
    assert info["estep_feasibility_margin"] < 0.0


def test_feasible_e_step_is_flagged() -> None:
    alg = _make_cvpo(dual_constraint=0.1)
    rng = np.random.default_rng(3)
    qc = np.abs(rng.normal(loc=10.0, scale=0.5, size=(32, 64)))
    reachable, _ = alg._reachable_qc_min(qc)
    alg._probe_feasibility(qc, thres=reachable + 0.5)
    info = alg.get_penalty_info()
    assert info["estep_feasible"] == 1.0
    assert info["estep_feasibility_margin"] > 0.0


def test_feasibility_probe_can_be_disabled() -> None:
    alg = _make_cvpo(feasibility_probe_interval=0)
    alg._probe_feasibility(np.ones((8, 4)), thres=1.0)
    assert "estep_feasible" not in alg.get_penalty_info()


# --------------------------------------------------------------------------------------
# End-to-end: the diagnostics the request asked to be logged
# --------------------------------------------------------------------------------------


def test_update_reports_the_requested_diagnostics() -> None:
    alg = _make_cvpo(
        qc_thres_homotopy=True,
        qc_target_ema=True,
        qc_target_warmup_reports=1,
        num_updates_per_step=2,
    )
    _fill_buffer(alg, n=128)
    alg.update(current_costs=[45.0])
    info = alg.get_penalty_info()
    for key in (
        "lambda_mean",
        "lambda_at_cap_frac",
        "c_now",
        "c_now_over_thres",
        "qc_thres_eff",
        "qc_thres_target",
        "dual_residual_lambda",
        "estep_feasible",
    ):
        assert key in info, f"missing diagnostic {key}"
        assert np.isfinite(info[key]), f"non-finite diagnostic {key}"
    assert info["c_now_over_thres"] == pytest.approx(info["c_now"] / info["qc_thres_eff"])
    assert info["dual_residual_lambda"] == pytest.approx(info["qc_thres_eff"] - info["eqc"])


# --------------------------------------------------------------------------------------
# Fixed-lambda pinning (the front-tracing sweep)
# --------------------------------------------------------------------------------------


def test_lambda_init_pins_lambda_when_the_controller_step_is_zero() -> None:
    alg = _make_cvpo(lambda_init=0.78, lambda_lr=0.0, lambda_max=4.0)
    assert alg.lam == pytest.approx(0.78)
    # Drive a large violation and a large slack; neither may move a pinned multiplier.
    for delta in (50.0, -50.0, 1e4):
        alg._update_lambda(alg._effective_thres() + delta)
        assert alg.lam == pytest.approx(0.78)


def test_lambda_init_defaults_to_the_previous_behaviour() -> None:
    assert _make_cvpo().lam == pytest.approx(1.0)
    assert _make_cvpo(cost_critic_passive=True).lam == pytest.approx(0.0)


def test_lambda_init_above_lambda_max_is_rejected() -> None:
    # Silently clipping would make the pin a lie and the sweep point mislabelled.
    with pytest.raises(ValueError, match="exceeds lambda_max"):
        _make_cvpo(lambda_init=5.0, lambda_max=4.0)


def test_lambda_init_zero_removes_the_cost_term_entirely() -> None:
    alg = _make_cvpo(lambda_init=0.0, lambda_lr=0.0, lambda_max=1e-6)
    _fill_buffer(alg, n=96)
    alg.update(current_costs=[40.0])
    info = alg.get_penalty_info()
    assert info["lambda_mean"] == pytest.approx(0.0)
    assert info["estep_spread_ratio"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# PID-Lagrangian on realized episodic cost (lambda_source="episodic")
# --------------------------------------------------------------------------------------


def _pid_cvpo(**kw):
    base = dict(
        lambda_source="episodic", lambda_update="pid", lambda_kp=0.25, lambda_lr=0.002,
        lambda_max=3.0, lambda_init=0.0, lambda_episodic_warmup=1, cost_limits=[25.0],
    )
    base.update(kw)
    return _make_cvpo(**base)



def _over_budget_reports(n: int, level: float = 48.0):
    """`n` distinct reports all at the same violation level.

    The episodic controller steps once per NEW measurement, so a literal constant would
    register as a single report. The runner's cost-buffer mean likewise moves every time an
    episode completes; the jitter here is that movement, too small to change delta.
    """
    return [level + i * 1e-9 for i in range(n)]

def test_episodic_lambda_integrates_up_when_over_budget() -> None:
    alg = _pid_cvpo()
    lams = []
    for c in _over_budget_reports(400):  # 1.9x the limit, as measured
        alg.update_lagrangian_multipliers([c])
        lams.append(alg.lam)
    assert lams == sorted(lams), "integral term must be monotone under a constant violation"
    assert lams[-1] > 0.5, f"lambda failed to integrate: {lams[-1]:.4f}"
    # delta is dimensionless: (48-25)/25
    assert alg.get_penalty_info()["lambda_delta"] == pytest.approx((48.0 - 25.0) / 25.0)


def test_episodic_lambda_falls_back_when_under_budget() -> None:
    alg = _pid_cvpo()
    for _ in range(400):
        alg.update_lagrangian_multipliers([48.0])
    high = alg.lam
    for _ in range(400):
        alg.update_lagrangian_multipliers([10.0])
    assert alg.lam < high, "lambda must release once the policy is inside budget"


def test_episodic_lambda_is_scale_invariant_in_the_cost_limit() -> None:
    # Stooke's normalisation: delta = (J_c - d)/d, so the SAME relative violation gives the
    # same lambda trajectory whatever the limit is -- gains never need retuning per limit.
    a = _pid_cvpo(cost_limits=[25.0])
    b = _pid_cvpo(cost_limits=[100.0])
    for _ in range(200):
        a.update_lagrangian_multipliers([50.0])   # 2x limit
        b.update_lagrangian_multipliers([200.0])  # 2x limit
    assert a.lam == pytest.approx(b.lam, rel=1e-9)


def test_empty_cost_buffer_cannot_wind_the_integral_negative() -> None:
    # The runner reports mean(costbuffer) with a 0.0 fallback before any episode completes.
    # Taken literally that is delta = -1 (max slack) for hundreds of iterations.
    alg = _pid_cvpo()
    for _ in range(500):
        alg.update_lagrangian_multipliers([0.0])  # "no data", not "zero cost"
    assert alg.lam == pytest.approx(0.0)
    assert alg.get_penalty_info()["lambda_episodic_reports"] == 0.0, "0.0 must not count as a report"
    # and a real over-budget signal afterwards still drives lambda up from a clean integral
    for c in _over_budget_reports(400):
        alg.update_lagrangian_multipliers([c])
    assert alg.lam > 0.5


def test_episodic_warmup_defers_engagement() -> None:
    alg = _pid_cvpo(lambda_episodic_warmup=50)
    reports = _over_budget_reports(449)
    for c in reports[:49]:
        alg.update_lagrangian_multipliers([c])
    assert alg.lam == pytest.approx(0.0)
    for c in reports[49:]:
        alg.update_lagrangian_multipliers([c])
    assert alg.lam > 0.5


def test_episodic_source_does_not_read_the_critic_level() -> None:
    # The whole point: the E-step must not touch lambda when the source is episodic, so a
    # collapsed Q_c level (which broke both threshold schemes) cannot zero the multiplier.
    alg = _pid_cvpo()
    for _ in range(400):
        alg.update_lagrangian_multipliers([48.0])
    before = alg.lam
    for eqc in (0.0, 1e-6, 500.0):   # absurd Q-space readings
        alg._update_lambda(eqc)
    assert alg.lam == pytest.approx(before), "Q-space values must not move an episodic lambda"


def test_lambda_source_validation() -> None:
    with pytest.raises(ValueError, match="lambda_source"):
        _make_cvpo(lambda_source="bogus")


def test_qspace_source_is_the_default_and_unchanged() -> None:
    alg = _make_cvpo()
    assert alg.lambda_source == "qspace"
    before = alg.lam
    alg._update_lambda(alg._effective_thres() + 10.0)
    assert alg.lam != before, "default path must still be driven by the Q-space residual"
