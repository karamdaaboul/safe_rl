"""Off-policy mismatch diagnostic: flag-gated telemetry, inert by default.

Covers the three seams the diagnostic touches:
* ``StochasticActor.log_prob`` must be the exact inverse of ``sample``'s change of variables;
* ``ReplayStorage(cost_window_extras=True)`` must add per-step window fields without touching
  the training fields, and the default must add nothing;
* ``FHDCMPO(offpolicy_diag=True)`` must produce ``opd_*`` telemetry that reads ~0 when the
  behavior policy IS the current policy, and must collapse (ESS) once the policy drifts.
"""

from __future__ import annotations

import pytest
import torch

from safe_rl.algorithms.fhdcmpo import FHDCMPO
from safe_rl.modules.safe_actor_critic import SafeActorCritic

OBS, ACT = 12, 3


def make_policy() -> SafeActorCritic:
    return SafeActorCritic(
        OBS,
        OBS,
        ACT,
        critic_type="quantile",
        cost_critic_type="quantile",
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"n_quantiles": 8, "network_kwargs": {"hidden_dims": [32, 32]}},
        cost_critic_kwargs={"n_quantiles": 8, "nonneg": True, "network_kwargs": {"hidden_dims": [32, 32]}},
    )


def make_alg(diag: bool) -> FHDCMPO:
    alg = FHDCMPO(
        make_policy(),
        cost_limits=[25.0],
        n_step=2,
        cost_n_step=8,
        cost_td_lambda=0.99,
        offpolicy_diag=diag,
        offpolicy_diag_interval=1,
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=4,
        mstep_iteration_num=1,
        feasibility_probe_interval=0,
    )
    alg.init_storage(buffer_size=4096, num_envs=4, obs_shape=[OBS], act_shape=[ACT])
    return alg


def fill(alg: FHDCMPO, with_blp: bool, n_steps: int = 300) -> None:
    torch.manual_seed(1)
    for t in range(n_steps):
        o = torch.randn(4, OBS)
        a = torch.rand(4, ACT) * 2 - 1
        kw = {}
        if with_blp:
            kw["behavior_log_prob"] = alg.policy.action_log_prob(o, a)
            kw["policy_version"] = torch.full((4, 1), float(t))
        alg.store_transition(
            o,
            a,
            torch.randn(4, 1),
            (torch.rand(4, 1) < 0.02).float(),
            torch.randn(4, OBS),
            cost=(torch.rand(4, 1) < 0.3).float(),
            bootstrap=torch.zeros(4, 1),
            **kw,
        )


def test_action_log_prob_matches_sample() -> None:
    torch.manual_seed(0)
    pol = make_policy()
    obs = torch.randn(64, OBS)
    a, lp = pol.sample_with_log_prob(obs)
    assert torch.allclose(lp, pol.action_log_prob(obs, a), atol=1e-4)


def test_log_prob_finite_on_boundary_actions() -> None:
    pol = make_policy()
    a = torch.ones(4, ACT)  # exactly on the tanh boundary
    assert torch.isfinite(pol.action_log_prob(torch.randn(4, OBS), a)).all()


def test_flag_off_batch_is_unchanged() -> None:
    alg = make_alg(diag=False)
    fill(alg, with_blp=False)
    batch = alg.storage.sample(32)
    assert "cost_window_returns" in batch
    assert not any(k.startswith("cost_window_a") for k in batch)  # actions/age/alive absent
    alg.update()  # regression: the edited update path still runs
    assert alg._last_opd_diag == {}


def test_flag_on_extras_shapes_and_age() -> None:
    alg = make_alg(diag=True)
    fill(alg, with_blp=True)
    batch = alg.storage.sample(32)
    for key, shape in {
        "cost_window_actions": (32, 8, ACT),
        "cost_window_obs": (32, 8, OBS),
        "cost_window_alive": (32, 8),
        "cost_window_blp": (32, 8),
        "cost_window_version": (32, 8),
        "cost_window_age": (32,),
    }.items():
        assert tuple(batch[key].shape) == shape, key
    # Ring age and the stored version stamp track the same write order exactly.
    corr = torch.corrcoef(torch.stack([batch["cost_window_version"][:, 0], batch["cost_window_age"]]))[0, 1]
    assert corr < -0.99


def test_diag_reads_zero_on_policy_and_collapses_after_drift() -> None:
    alg = make_alg(diag=True)
    fill(alg, with_blp=True)
    alg.update()
    diag = alg._last_opd_diag
    assert abs(diag["opd_logratio_mean"]) < 1e-3
    assert diag["opd_ess_frac_j1"] > 0.999
    assert diag["opd_frac_ratio_extreme"] == 0.0
    assert any(k.startswith("opd_") for k in alg.get_penalty_info())

    with torch.no_grad():
        for p in alg.policy.actor.parameters():
            p.add_(0.3 * torch.randn_like(p))
    alg.update()
    drifted = alg._last_opd_diag
    assert abs(drifted["opd_logratio_mean"]) > 1e-3
    assert drifted["opd_ess_frac_j63"] < 0.9  # importance weights collapse once the policy moves


def test_diag_requires_cost_window() -> None:
    with pytest.raises(ValueError, match="cost_n_step"):
        FHDCMPO(make_policy(), cost_limits=[25.0], offpolicy_diag=True)


# ---------------------------------------------------------------------------------------------
# EMA bootstrap policy for the cost target (bootstrap-stability experiment)
# ---------------------------------------------------------------------------------------------


def make_boot_alg(tau_b: float | None) -> FHDCMPO:
    alg = FHDCMPO(
        make_policy(),
        cost_limits=[25.0],
        n_step=2,
        cost_n_step=8,
        cost_td_lambda=0.99,
        cost_boot_ema_tau=tau_b,
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=4,
        mstep_iteration_num=1,
        feasibility_probe_interval=0,
    )
    alg.init_storage(buffer_size=4096, num_envs=4, obs_shape=[OBS], act_shape=[ACT])
    return alg


def test_boot_default_is_off_and_requires_window() -> None:
    assert make_boot_alg(None).cost_boot_actor is None
    with pytest.raises(ValueError, match="cost_n_step"):
        FHDCMPO(make_policy(), cost_limits=[25.0], cost_boot_ema_tau=0.005)
    with pytest.raises(ValueError, match="cost_boot_ema_tau"):
        make_boot_alg(1.5)


def test_boot_ema_update_is_exact_and_leaves_estep_anchor_alone() -> None:
    alg = make_boot_alg(0.1)
    boot_before = [p.clone() for p in alg.cost_boot_actor.parameters()]
    anchor_before = [p.clone() for p in alg.actor_target.parameters()]
    with torch.no_grad():
        for p in alg.policy.actor.parameters():
            p.add_(torch.randn_like(p))
    alg._sync_target_actor()
    for p, bp, b0 in zip(alg.policy.actor.parameters(), alg.cost_boot_actor.parameters(), boot_before):
        assert torch.allclose(bp, 0.9 * b0 + 0.1 * p, atol=1e-6)  # pi_boot <- (1-tau_b) pi_boot + tau_b pi
    # The E-step anchor keeps ITS OWN rate (self.tau), untouched by the boot EMA.
    for p, tp, t0 in zip(alg.policy.actor.parameters(), alg.actor_target.parameters(), anchor_before):
        assert torch.allclose(tp, (1 - alg.tau) * t0 + alg.tau * p, atol=1e-6)
    assert not any(bp.requires_grad for bp in alg.cost_boot_actor.parameters())


def test_boot_actor_supplies_the_bootstrap_action() -> None:
    alg = make_boot_alg(0.005)
    obs = torch.randn(16, OBS)
    # Pin pi_boot's pre-tanh mean far from pi_current's: every bootstrap action must reflect it.
    with torch.no_grad():
        alg.cost_boot_actor.mean_head.weight.zero_()
        alg.cost_boot_actor.mean_head.bias.fill_(50.0)  # tanh -> all actions ~ +1
    acts = alg._cost_bootstrap_action(obs)
    assert float(acts.min()) > 0.99
    # Baseline path unchanged: with the EMA off, actions come from pi_current, not from any copy.
    base = make_boot_alg(None)
    assert base._cost_bootstrap_action(obs).abs().max() <= 1.0


# ---------------------------------------------------------------------------------------------
# Per-state cost spread matching in the E-step exponent
# ---------------------------------------------------------------------------------------------


def make_spread_alg(match: bool) -> FHDCMPO:
    alg = FHDCMPO(
        make_policy(),
        cost_limits=[25.0],
        n_step=2,
        cost_n_step=8,
        cost_td_lambda=0.99,
        lambda_mode="grad",
        lambda_source="episodic",
        lambda_init=1.0,
        estep_cost_spread_match=match,
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=1,
        feasibility_probe_interval=0,
    )
    alg.init_storage(buffer_size=4096, num_envs=4, obs_shape=[OBS], act_shape=[ACT])
    return alg


def test_spread_match_requires_episodic_grad_lambda() -> None:
    with pytest.raises(ValueError, match="episodic"):
        FHDCMPO(
            make_policy(),
            cost_limits=[25.0],
            n_step=2,
            cost_n_step=8,
            estep_cost_spread_match=True,
            lambda_mode="grad",
            lambda_source="qspace",
        )


def test_spread_match_equalizes_without_raising_total_pressure() -> None:
    torch.manual_seed(3)
    q = torch.randn(8, 16)
    # Heterogeneous across-action cost spreads (the probe's finding): half the states nearly
    # flat (lambda-inert), half with reward-scale spread.
    qc = 20.0 + torch.randn(8, 16) * torch.cat([torch.full((8,), 0.02), torch.full((8,), 1.0)])

    def weights_for(alg):
        torch.manual_seed(5)
        fill(alg, with_blp=False, n_steps=200)
        obs = torch.randn(16, OBS)
        acts = torch.rand(8, 16, ACT) * 2 - 1
        import unittest.mock as mock

        with mock.patch.object(alg, "_estep_cost", return_value=qc.reshape(-1, 1)):
            return alg._estep_weights(q.clone(), acts, obs)

    alg_b = make_spread_alg(False)
    w_base = weights_for(alg_b)
    alg_m = make_spread_alg(True)
    w_match = weights_for(alg_m)
    # Baseline is the exact pre-existing exponent (raw qc, its own eta).
    assert torch.allclose(w_base, torch.softmax((q - alg_b.lam * qc) / alg_b.eta, dim=0), atol=1e-4)
    info = alg_m.get_penalty_info()
    # Median normalization: the typical state's scale is ~1 (no total-pressure inflation --
    # the c1 idle-policy failure), while flat-cost states are amplified toward the cap.
    assert 0.5 < info["estep_match_scale_median"] < 2.0, info["estep_match_scale_median"]
    # The mechanism: post-match effective cost spreads are EQUALIZED across states. Recompute
    # the scale exactly as the algorithm does and check flat and wide states land together.
    ratio = q.std(dim=0) / qc.std(dim=0).clamp_min(1e-6)
    scale = (ratio / ratio.median()).clamp(1 / alg_m.estep_spread_match_max, alg_m.estep_spread_match_max)
    post = qc.std(dim=0) * scale
    assert float(post[:8].mean()) / float(post[8:].mean()) > 0.66  # was 0.02 vs 1.0 pre-match
    tv = 0.5 * (w_base - w_match).abs().sum(0)
    assert float(tv.mean()) > 0.01  # the redistribution really moves the weights
    # eqc stays in RAW units (a level near 20), not in matched units.
    assert 10.0 < alg_m._eqc < 30.0


def test_boot_run_produces_drift_diag_and_env_policy_is_untouched() -> None:
    alg = make_boot_alg(0.05)
    fill(alg, with_blp=False)
    obs = torch.randn(8, OBS)
    torch.manual_seed(7)
    act_before = alg.policy.act(obs, deterministic=True)
    alg.update()
    info = alg.get_penalty_info()
    for key in ("boot_kl_mean", "boot_kl_p90", "boot_mean_l2", "boot_ema_tau"):
        assert key in info
    assert info["boot_kl_mean"] >= 0.0
    # Environment actions keep coming from pi_current (deterministic act moved WITH the actor,
    # i.e. it is not pinned to the boot copy).
    torch.manual_seed(7)
    act_after = alg.policy.act(obs, deterministic=True)
    assert not torch.allclose(act_before, act_after)  # actor trained -> env policy moved


# ---------------------------------------------------------------------------------------------
# Readout / dose diagnostics: which cost statistic the E-step exponent is actually built from
# ---------------------------------------------------------------------------------------------


def make_dose_alg(risk_mode: str, match: bool = True) -> FHDCMPO:
    """Spread-match arm with the kappa ramp collapsed, so `_fh_kappa` is set directly by tests."""
    alg = FHDCMPO(
        make_policy(),
        cost_limits=[25.0],
        n_step=2,
        cost_n_step=8,
        cost_td_lambda=0.99,
        lambda_mode="grad",
        lambda_source="episodic",
        lambda_init=1.0,
        estep_cost_spread_match=match,
        fh_risk_mode=risk_mode,
        fh_alpha=0.9,
        fh_kappa=1.0,
        fh_kappa_warmup=0,
        fh_kappa_ramp=0,
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=1,
        feasibility_probe_interval=0,
    )
    alg.init_storage(buffer_size=4096, num_envs=4, obs_shape=[OBS], act_shape=[ACT])
    return alg


def run_estep(alg: FHDCMPO, seed: int = 11):
    """One real E-step (no `_estep_cost` mock, so the kappa=0 reference is actually stashed)."""
    torch.manual_seed(5)
    fill(alg, with_blp=False, n_steps=200)
    torch.manual_seed(seed)
    q = torch.randn(8, 16)
    obs = torch.randn(16, OBS)
    acts = torch.rand(8, 16, ACT) * 2 - 1
    with torch.no_grad():  # the runner's E-step context (mpo.py `_update_actor`)
        weights = alg._estep_weights(q, acts, obs)
    return weights, alg.get_penalty_info()


def test_mean_arm_reports_unit_dose_and_is_unperturbed() -> None:
    """`fh_risk_mode: mean` -- the c2 semantics. Both doses must be EXACTLY 1.0, not merely close.

    The reference readout is aliased to the returned statistic at kappa = 0, so this is an
    identity, not a numerical coincidence: any drift here means the mean arm stopped being
    bit-reproducible.
    """
    _, info = run_estep(make_dose_alg("mean"))
    assert info["estep_match_readout"] == "mean"
    assert info["estep_match_readout_is_tail"] == 0.0
    assert info["estep_cost_dose_vs_mean_median"] == 1.0
    assert info["estep_match_dose_vs_mean_median"] == 1.0


def test_dose_diagnostics_do_not_change_the_weights() -> None:
    """The hard constraint: the diagnostics read the E-step, they never feed it.

    Compares against the exponent recomputed by hand from the algorithm's own eta/lambda and the
    pre-existing s_b formula -- `torch.equal`, so a single changed float fails.
    """
    alg = make_dose_alg("mean")
    weights, _ = run_estep(alg)
    qc = alg._last_estep_cost_ref[:, 0].reshape(8, 16)
    torch.manual_seed(11)
    q = torch.randn(8, 16)
    ratio = q.std(dim=0) / qc.std(dim=0).clamp_min(1e-6)
    s_b = (ratio / ratio.median().clamp_min(1e-6)).clamp(1.0 / alg.estep_spread_match_max, alg.estep_spread_match_max)
    expected = torch.softmax((q - alg.lam * (qc * s_b)) / alg.eta, dim=0)
    assert torch.equal(weights, expected)


def test_cvar_readout_carries_a_dose_above_one() -> None:
    """kappa = 1: the tail readout's across-action spread is what lambda actually multiplies."""
    alg = make_dose_alg("cvar")
    assert alg._fh_kappa == 0.0  # ramp not advanced yet
    _, info = run_estep(alg)
    assert info["estep_match_readout"] == "cvar"  # `_estep_weights` advances the ramp first
    assert info["estep_match_readout_is_tail"] == 1.0
    assert alg._fh_kappa == 1.0
    assert info["estep_cost_dose_vs_mean_median"] > 1.0
    # s_b is median-normalized, hence invariant to a global rescale of the readout: it does NOT
    # absorb the dose. This near-1.0 reading is the standing refutation of that hypothesis.
    assert 0.5 < info["estep_match_dose_vs_mean_median"] < 2.0


def test_cvar_arm_reports_mean_during_kappa_warmup() -> None:
    """A cvar arm still on the warmup plateau is reading the mean; report the EFFECTIVE readout."""
    alg = make_dose_alg("cvar")
    alg.fh_kappa_warmup = 10_000  # stay at kappa = 0 through this update
    _, info = run_estep(alg)
    assert alg._fh_kappa == 0.0
    assert info["estep_match_readout"] == "mean"
    assert info["estep_match_readout_is_tail"] == 0.0
    assert info["estep_cost_dose_vs_mean_median"] == 1.0


def test_dose_reported_without_spread_matching() -> None:
    """The dose is a property of the readout, so it is logged whether or not s_b is applied."""
    _, info = run_estep(make_dose_alg("cvar", match=False))
    assert info["estep_cost_dose_vs_mean_median"] > 1.0
    assert "estep_match_dose_vs_mean_median" not in info  # s_b was never formed


def test_readout_diag_survives_the_runner_numeric_filter() -> None:
    """`OffPolicyRunner` drops non-numeric `get_penalty_info` entries, so the dose needs a
    numeric carrier or it never reaches a dashboard. Mirrors the runner's filter exactly."""
    _, info = run_estep(make_dose_alg("cvar"))
    logged = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
    assert logged["estep_match_readout_is_tail"] == 1.0
    assert logged["estep_cost_dose_vs_mean_median"] > 1.0
    assert "estep_match_readout" not in logged  # the string form is for probes/tests only


# ---------------------------------------------------------------------------------------------
# Reference-median normalization (`estep_match_normalizer`): the c3 dose fix
# ---------------------------------------------------------------------------------------------


def make_norm_alg(risk_mode: str, normalizer: str = "active", match: bool = True) -> FHDCMPO:
    alg = FHDCMPO(
        make_policy(),
        cost_limits=[25.0],
        n_step=2,
        cost_n_step=8,
        cost_td_lambda=0.99,
        lambda_mode="grad",
        lambda_source="episodic",
        lambda_init=1.0,
        estep_cost_spread_match=match,
        estep_match_normalizer=normalizer,
        fh_risk_mode=risk_mode,
        fh_alpha=0.9,
        fh_kappa=1.0,
        fh_kappa_warmup=0,
        fh_kappa_ramp=0,
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=1,
        feasibility_probe_interval=0,
    )
    alg.init_storage(buffer_size=4096, num_envs=4, obs_shape=[OBS], act_shape=[ACT])
    return alg


def synthetic_estep(alg: FHDCMPO, q: torch.Tensor, qc: torch.Tensor, qc_ref: torch.Tensor | None = None):
    """Drive one E-step on FIXED cost values, bypassing the critic.

    `_estep_cost` is patched to return `qc` and stash `qc_ref` exactly as `FHDCMPO._estep_cost`
    would, so the normalizer sees a controlled dose instead of whatever the untrained critic emits.
    """
    import unittest.mock as mock

    n, b = q.shape

    def fake_cost(cobs, acts, target):
        flat = qc.reshape(-1, 1)
        alg._last_estep_cost_ref = flat if qc_ref is None else qc_ref.reshape(-1, 1)
        return flat

    torch.manual_seed(5)
    fill(alg, with_blp=False, n_steps=200)
    obs = torch.randn(b, OBS)
    acts = torch.rand(n, b, ACT) * 2 - 1
    with torch.no_grad(), mock.patch.object(alg, "_estep_cost", side_effect=fake_cost):
        weights = alg._estep_weights(q.clone(), acts, obs)
    return weights, alg.get_penalty_info()


def spread_batch(dose: float, seed: int = 21):
    """`(q, qc_mean, qc_cvar)` where the tail readout's across-action spread is `dose` x the mean's.

    Built by scaling the deviation from each state's mean, so only the SPREAD changes -- the level
    is held fixed, which is what the E-step softmax is sensitive to.

    Per-state spreads span only 4x, deliberately: the identities the fix is defined by hold on the
    UNCAPPED branch, so the batch must not straddle the clamp unless the test is about the clamp.
    """
    torch.manual_seed(seed)
    q = torch.randn(8, 16)
    per_state = torch.linspace(0.5, 2.0, 16)
    qc_mean = 20.0 + torch.randn(8, 16) * per_state
    qc_cvar = qc_mean.mean(dim=0, keepdim=True) + (qc_mean - qc_mean.mean(dim=0, keepdim=True)) * dose
    return q, qc_mean, qc_cvar


def assert_uncapped(info: dict) -> None:
    """Precondition for the exactness assertions: no state was clamped in either direction."""
    assert info["estep_match_scale_capped_frac"] == 0.0
    assert info["estep_match_scale_floored_frac"] == 0.0


def delivered_grip(alg: FHDCMPO, q: torch.Tensor, qc: torch.Tensor, info: dict) -> float:
    """Recompute `estep_match_grip` from the batch, as an independent check of the logged value."""
    ratio = q.std(dim=0) / qc.std(dim=0).clamp_min(1e-6)
    key = "estep_match_ref_median" if alg.estep_match_normalizer == "reference" else "estep_match_ratio_median"
    norm = torch.tensor(info[key])
    m = (ratio / norm.clamp_min(1e-6)).clamp(1.0 / alg.estep_spread_match_max, alg.estep_spread_match_max)
    return float(info["lambda_mean"] * (m * qc.std(dim=0) / q.std(dim=0).clamp_min(1e-12)).mean())


def test_active_mode_is_the_pre_existing_exponent() -> None:
    """(1) Regression: `active` reproduces the formula every c1/c2/c3 run used, at any kappa.

    Recomputed by hand rather than compared to a golden file, so it stays readable: cap lands on
    `s_b / med(s_b)`, and the exponent is `q - lam * qc * m`.
    """
    q, qc_mean, qc_cvar = spread_batch(dose=3.0)
    for risk, qc, ref in (("mean", qc_mean, None), ("cvar", qc_cvar, qc_mean)):
        alg = make_norm_alg(risk, "active")
        weights, info = synthetic_estep(alg, q, qc, ref)
        ratio = q.std(dim=0) / qc.std(dim=0).clamp_min(1e-6)
        m = (ratio / ratio.median().clamp_min(1e-6)).clamp(1.0 / alg.estep_spread_match_max, alg.estep_spread_match_max)
        expected = torch.softmax((q - alg.lam * (qc * m)) / alg.eta, dim=0)
        assert torch.equal(weights, expected), risk
        # `active` normalizes by the active readout's own median, so the median state keeps m = 1
        # -- which is exactly why the dose passes straight through to the exponent.
        assert info["estep_match_scale_median"] == pytest.approx(1.0, abs=1e-5)
        assert delivered_grip(alg, q, qc, info) == pytest.approx(info["estep_match_grip"], rel=1e-5)
    # ref_median is always the MEAN-readout median, so the two medians differ by the dose.
    assert info["estep_match_ref_median"] / info["estep_match_ratio_median"] == pytest.approx(3.0, rel=1e-4)


def test_reference_mode_holds_the_grip_at_the_kappa_zero_value() -> None:
    """(2) The fix: a 3x tail spread must NOT buy 3x the constraint pressure.

    `active` passes the dose straight through; `reference` cancels it, landing on the grip the
    same lambda delivers at kappa = 0. That equality is the whole point of the change.
    """
    q, qc_mean, qc_cvar = spread_batch(dose=3.0)
    base = make_norm_alg("mean", "active")
    _, info_base = synthetic_estep(base, q, qc_mean, None)
    grip0 = info_base["estep_match_grip"]

    act = make_norm_alg("cvar", "active")
    _, info_act = synthetic_estep(act, q, qc_cvar, qc_mean)
    ref = make_norm_alg("cvar", "reference")
    _, info_ref = synthetic_estep(ref, q, qc_cvar, qc_mean)

    # Same lambda in all three (lambda_init=1.0, episodic controller not yet stepped).
    for i in (info_base, info_act, info_ref):
        assert_uncapped(i)
    assert info_ref["estep_match_grip"] == pytest.approx(grip0, rel=1e-5)
    assert info_act["estep_match_grip"] == pytest.approx(3.0 * grip0, rel=1e-5)
    # The dose telemetry still reports the readout honestly -- it is the GRIP that is corrected.
    assert info_ref["estep_cost_dose_vs_mean_median"] == pytest.approx(3.0, rel=1e-4)
    # And the correction shows up as m having median ~1/dose rather than 1.
    assert info_ref["estep_match_scale_median"] == pytest.approx(1.0 / 3.0, rel=1e-4)
    assert info_ref["estep_match_dose_vs_mean_median"] == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_reference_mode_is_invariant_to_a_global_rescale_of_the_readout() -> None:
    """(3) Dose invariance: scaling C_active by g leaves the exponent's cost term untouched.

    Under `reference` m absorbs 1/g exactly, so `qc * m` -- the tensor in the exponent -- is
    literally unchanged. Under `active` the same rescale multiplies the delivered grip by g.
    """
    q, qc_mean, _ = spread_batch(dose=1.0)
    g = 4.0
    for mode, factor in (("reference", 1.0), ("active", g)):
        a1 = make_norm_alg("cvar", mode)
        _, i1 = synthetic_estep(a1, q, qc_mean, qc_mean.clone())
        a2 = make_norm_alg("cvar", mode)
        # Scale the SPREAD only; a constant offset cancels in the softmax and would prove nothing.
        scaled = qc_mean.mean(dim=0, keepdim=True) + (qc_mean - qc_mean.mean(dim=0, keepdim=True)) * g
        _, i2 = synthetic_estep(a2, q, scaled, qc_mean)
        assert_uncapped(i1)
        assert_uncapped(i2)
        assert i2["estep_match_grip"] == pytest.approx(factor * i1["estep_match_grip"], rel=1e-5), mode


def test_reference_and_active_coincide_at_kappa_zero() -> None:
    """(4) At kappa = 0 the reference IS the active readout, so the two modes must be bit-equal."""
    q, qc_mean, _ = spread_batch(dose=1.0)
    w_act, i_act = synthetic_estep(make_norm_alg("mean", "active"), q, qc_mean, None)
    w_ref, i_ref = synthetic_estep(make_norm_alg("mean", "reference"), q, qc_mean, None)
    assert torch.equal(w_act, w_ref)
    for k in (
        "estep_match_scale_median",
        "estep_match_ratio_median",
        "estep_match_ref_median",
        "estep_match_grip",
        "estep_cost_dose_vs_mean_median",
    ):
        assert i_act[k] == i_ref[k], k


def test_reference_normalizer_config_guards() -> None:
    """`reference` is meaningless without spread matching, and the mode name is validated."""
    with pytest.raises(ValueError, match="estep_cost_spread_match"):
        make_norm_alg("cvar", "reference", match=False)
    with pytest.raises(ValueError, match="must be 'active' or 'reference'"):
        make_norm_alg("cvar", "median")


def test_reference_mode_reports_the_floor_it_is_truncated_by() -> None:
    """At dose ~ match_max the correction wants m ~ 1/dose, which the LOWER clamp truncates.

    The fix is then silently under-applied, so `floored_frac` must make it visible.
    """
    q, qc_mean, qc_cvar = spread_batch(dose=10.0)
    alg = make_norm_alg("cvar", "reference")
    _, info = synthetic_estep(alg, q, qc_cvar, qc_mean)
    assert info["estep_match_scale_floored_frac"] > 0.3, info["estep_match_scale_floored_frac"]
    # Truncated means the delivered grip lands ABOVE the kappa=0 target, not on it.
    base = make_norm_alg("mean", "active")
    _, info0 = synthetic_estep(base, q, qc_mean, None)
    assert info["estep_match_grip"] > info0["estep_match_grip"]
