"""Tests for CVPOPerState: CVPO with a global eta and a per-state cost multiplier.

The solver's own gates live in `tests/test_per_state_dual.py`. This file covers the wiring: that
the per-state target is what it claims to be, that the inherited scalar-multiplier machinery
cannot be switched on alongside it, that lambda is computed rather than learned, and -- the one
that locks the design in place -- that the static target degenerates while the reachable one does
not.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safe_rl.common.per_state_dual import solve_per_state_dual  # noqa: E402

NUM_OBS = 6
NUM_ACT = 2

# The cost critic this repo actually trains, per codex/cvpo-cost-critic-investigation.md:
# a high, nearly state-independent LEVEL with almost no spread across candidate actions. Only the
# spread survives the per-state softmax, so these two numbers are what decide whether a per-state
# multiplier has anything to work with.
MEASURED_QC = dict(level=3.119, std_across_actions=0.0379, std_across_states=0.5)


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


def _make_alg(**overrides):
    from safe_rl.algorithms import CVPOPerState

    kwargs = dict(
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_horizon=1000,
        lambda_mode="per_state",
        lambda_max=3.0,
        device="cpu",
    )
    kwargs.update(overrides)
    return CVPOPerState(_make_policy(), **kwargs)


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


def _measured_problem(seed: int = 0, k: int = 64, b: int = 128):
    """Synthetic (Q_r, Q_c) with this repo's measured cost-critic statistics."""
    g = torch.Generator().manual_seed(seed)
    q_r = torch.randn(k, b, generator=g, dtype=torch.float64) * 0.021 + 18.0
    level = MEASURED_QC["level"] + MEASURED_QC["std_across_states"] * torch.randn(b, generator=g, dtype=torch.float64)
    q_c = level.unsqueeze(0) + MEASURED_QC["std_across_actions"] * torch.randn(k, b, generator=g, dtype=torch.float64)
    return q_r, q_c.clamp_min(0.0)


# --------------------------------------------------------------------------------------------
# The regression lock on the design decision
# --------------------------------------------------------------------------------------------


def test_static_target_degenerates_and_reachable_target_does_not() -> None:
    """The reason CVPOPerState uses a per-state reachable target rather than the static qc_thres.

    Held to the static threshold, the solve is *correct* -- g decreases monotonically, and for a
    state whose candidates cannot reach the target lambda_b -> lambda_max really is optimal -- but
    the answer is degenerate: the reward term is erased at nearly every state and the constraint
    is still violated. Because eta is shared, those states also drag it up and take the feasible
    states with them.

    If someone later "simplifies" the target back to the static one, this fires. Do not relax it
    without reading codex/cvpo-cost-critic-investigation.md first.
    """
    q_r, q_c = _measured_problem()
    eps, lam_max = 0.1, 3.0
    qc_thres = 25.0 * 0.0764  # the measured-scale threshold this repo runs with

    static_d = torch.full((q_c.shape[1],), qc_thres, dtype=q_c.dtype)
    # A sigma-sized ask, deliberately: this test isolates static-vs-reachable *target*, so it uses
    # the simplest reachable formula rather than the shipped default (a fraction of the KL-reachable
    # drop -- see test_reachable_ask_is_always_inside_the_trust_region for that one).
    reachable_d = torch.clamp_min(q_c.mean(dim=0) - 0.25 * q_c.std(dim=0), qc_thres)

    static = [solve_per_state_dual(q_r, q_c, static_d, eps, lam_max, sweeps=n) for n in (1, 4)]
    reach = [solve_per_state_dual(q_r, q_c, reachable_d, eps, lam_max, sweeps=n) for n in (1, 4)]

    # 1. Static: essentially every state pins at the cap, where the weights no longer see Q_r.
    assert float(static[-1].lam_at_cap.double().mean()) > 0.9
    # 2. ... and the constraint is violated anyway -- saturation buys nothing.
    assert float((static[-1].eqc - static_d).quantile(0.9)) > 0.5
    # 3. ... while eta is dragged well above where the reachable target puts it. This is the
    #    coupling: a shared eta averages the KL, so saturated states raise every state's temperature.
    assert float(static[-1].eta) > 2.5 * float(reach[-1].eta)

    # 4. Reachable: no saturation, and the target met exactly.
    assert float(reach[-1].lam_at_cap.double().mean()) == 0.0
    assert float((reach[-1].eqc - reachable_d).quantile(0.9)) < 1e-2

    # 5. One sweep is enough *from a warm eta* -- which is what `dual_sweeps: 1` relies on, since
    #    eta is warm-started from the previous update. From cold it is not: in this regime the
    #    coordinate descent needs ~16 sweeps to reach its fixed point, so the opening updates of a
    #    run settle over a handful of *updates* rather than in one. At the fixed point a single
    #    warm sweep reproduces it exactly, which is the property `dual_sweeps: 1` actually needs.
    converged = solve_per_state_dual(q_r, q_c, reachable_d, eps, lam_max, sweeps=64)
    warm = solve_per_state_dual(q_r, q_c, reachable_d, eps, lam_max, sweeps=1, eta_init=converged.eta)
    assert abs(float(warm.eta) - float(converged.eta)) / float(converged.eta) < 1e-6
    assert float((warm.lam - converged.lam).abs().max()) < 1e-6

    # 5. The multiplier actually varies across states -- otherwise this is just CVPO with extra steps.
    assert float(reach[-1].lam.std()) > 1e-3


# --------------------------------------------------------------------------------------------
# Configuration: the per-state solve supersedes CVPO's scalar machinery, loudly
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(lambda_mode="grad"), "lambda_mode='per_state'"),
        (dict(lambda_mode="dual"), "lambda_mode='per_state'"),
        (dict(lambda_source="episodic"), "lambda_source='qspace'"),
        (dict(lambda_lr=0.03), "computes lambda, it does not learn it"),
        (dict(lambda_update="pid"), "computes lambda, it does not learn it"),
        (dict(lambda_kp=0.1), "computes lambda, it does not learn it"),
        (dict(lambda_kd=0.1), "computes lambda, it does not learn it"),
        (dict(lambda_init=1.0), "lambda_init is meaningless"),
        (dict(rescale_by_lambda=True), "breaks the monotonicity"),
        (dict(qc_thres_homotopy=True), "replaces it"),
        (dict(dstate_mode="nope"), "dstate_mode must be"),
        (dict(dstate_beta_mode="nope"), "dstate_beta_mode must be"),
        (dict(dstate_beta=1.5), "dstate_beta must be"),
        (dict(lambda_max_mode="nope"), "lambda_max_mode must be"),
        (dict(solver_dtype="float16"), "solver_dtype must be"),
        (dict(dual_sweeps=0), "dual_sweeps must be"),
    ],
)
def test_init_rejects_conflicting_options(overrides, match) -> None:
    """Conflicting inherited settings raise; they are never silently ignored.

    A config that reads `lambda_mode: grad` and quietly does something else would be the worst
    available failure mode -- it looks configured and is not.
    """
    with pytest.raises(ValueError, match=match):
        _make_alg(**overrides)


def test_scalar_multiplier_paths_are_unreachable() -> None:
    """CVPO's SLSQP dual and its LambdaController are not part of this algorithm."""
    alg = _make_alg()
    with pytest.raises(RuntimeError, match="scalar SLSQP dual"):
        alg._solve_dual(None, None)
    with pytest.raises(RuntimeError, match="CLAUDE.md rule 1"):
        alg._update_lambda(1.0)
    with pytest.raises(RuntimeError, match="lambda is a vector"):
        alg._track_lambda_saturation()


def test_lambda_is_computed_not_learned() -> None:
    """CLAUDE.md rule 1, as a test: no optimizer, no nn.Parameter, no state carried across updates."""
    alg = _make_alg()
    _fill_buffer(alg)
    named = dict(alg.policy.named_parameters())
    assert not [n for n in named if "lam" in n.lower()]
    for opt in (alg.actor_optimizer, alg.critic_optimizer):
        for group in opt.param_groups:
            for p in group["params"]:
                assert not any(p is q for n, q in named.items() if "lam" in n.lower())
    alg.update()
    # The inherited controller must not have been advanced by anything.
    assert alg._lambda_ctrl.lam == 0.0 and alg._lambda_ctrl.integral == 0.0


# --------------------------------------------------------------------------------------------
# The E-step
# --------------------------------------------------------------------------------------------


def test_lambda_is_per_state_and_weights_are_a_distribution() -> None:
    alg = _make_alg()
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert info["lambda_min"] <= info["lambda_median"] <= info["lambda_max"]
    assert 0.0 <= info["lambda_min"] and info["lambda_max"] <= alg._lambda_max_used + 1e-9

    obs = torch.randn(8, NUM_OBS)
    with torch.no_grad():
        actor_obs = alg.policy.actor_obs_normalizer(obs)
        _, actions, q, _, _ = alg._estep_sample(actor_obs, obs)
        w = alg._estep_weights(q, actions, obs)
    assert w.shape == q.shape
    assert torch.allclose(w.sum(dim=0), torch.ones(8), atol=1e-5)


def test_per_state_target_is_floored_at_q_target_and_leaves_scalar_homotopy_inert() -> None:
    """d_b never dips below the real budget, and the inherited scalar ratchet stays untouched.

    `_update_homotopy_threshold` is reused purely for its q_target side of the computation; with
    `qc_thres_homotopy` off it returns before touching `_qc_thres_eff`, and that must stay true.
    """
    alg = _make_alg(qc_target_ema=True)
    _fill_buffer(alg)
    for _ in range(10):
        alg.update()
    assert alg._qc_thres_eff == float("inf"), "the scalar homotopy ratchet was advanced"

    q_c = torch.rand(16, 24) * 3.0
    floor = alg._clamped_q_target()
    # float32 clamp, so compare at float32 resolution rather than exactly.
    assert float(alg._per_state_target(q_c, floor).min()) >= floor * (1.0 - 1e-6)
    # Below the floor everywhere -> the floor binds at every state.
    assert torch.allclose(alg._per_state_target(q_c * 1e-6, floor), torch.full((24,), floor), rtol=1e-6)


def test_static_mode_uses_the_scalar_threshold() -> None:
    alg = _make_alg(dstate_mode="static")
    q_c = torch.rand(16, 24) * 3.0
    d = alg._per_state_target(q_c, alg._clamped_q_target())
    assert torch.allclose(d, torch.full((24,), alg.qc_thres))


def test_spread_ask_mode_scales_with_action_spread() -> None:
    """`dstate_beta_mode="spread"` sets the ask in units of std_k(Q_c), so it ignores the level."""
    alg = _make_alg(dstate_beta_mode="spread", dstate_spread_kappa=0.5)
    q_c = torch.rand(16, 24) * 3.0
    d_low = alg._per_state_target(q_c, 0.0)
    d_high = alg._per_state_target(q_c + 100.0, 0.0)  # same spread, far higher level
    assert torch.allclose(d_high - d_low, torch.full((24,), 100.0), atol=1e-4)


# --------------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------------


def test_penalty_info_has_every_required_key() -> None:
    """The diagnostics CLAUDE.md requires, plus the per-state distributions a shared eta needs."""
    import math as _math

    alg = _make_alg()
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    required = [
        "eta_star",
        "eta",
        "lambda_median",
        "lambda_p10",
        "lambda_p90",
        "lambda_frac_zero",
        "lambda_frac_at_cap",
        "lambda_max_used",
        "lambda_over_balanced",
        "dual_residual_lambda",
        "dual_residual_lambda_interior_absmax",
        "kkt_residual_max",
        "dual_residual_eta",
        "dual_residual_eta_solver",
        "viol_vs_dstate_p90",
        "viol_vs_qtarget_p90",
        "frac_states_violating",
        "frac_states_at_floor",
        "frac_infeasible_lambda_cap",
        "frac_infeasible_support",
        "ess",
        "ess_min",
        "ess_p10",
        "ess_frac_below_4",
        "kl_q_p90",
        "kl_dispersion_ratio",
        "kl_q_frac_over_eps",
        "dstate_p50",
        "dstate_ask_over_spread",
        "qc_thres_target",
        "solver_iters",
        "solver_status",
        "solver_sweeps",
        "estep_host_syncs",
    ]
    missing = [k for k in required if k not in info]
    assert not missing, missing
    # frac_infeasible_support is deliberately NaN on updates where the probe does not run.
    bad = [k for k in required if not _math.isfinite(info[k]) and k != "frac_infeasible_support"]
    assert not bad, bad


def test_solver_kl_matches_mpo_measured_kl() -> None:
    """The weights returned must be the weights that were optimised.

    MPO computes `dual_residual_eta` independently, from the returned weights; the solver reports
    its own from the same quantity internally. A mismatch is the most likely integration bug --
    it means the E-step handed back something other than what it solved for.
    """
    alg = _make_alg()
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert abs(info["dual_residual_eta"] - info["dual_residual_eta_solver"]) < 1e-5


def test_host_syncs_do_not_scale_with_batch() -> None:
    """The E-step's device->host traffic must be a fixed handful of scalars, not O(B) or O(K).

    One scalar sync before the solve is unavoidable (the inherited q_target EMA is Python-side
    state), and the diagnostics reduce to a single batched transfer. What must never happen is a
    per-state `.item()`, which is how a vectorised solver quietly becomes slower than the SciPy
    call it replaced.
    """
    counts = {}
    for b in (16, 128):
        alg = _make_alg(batch_size=b)
        _fill_buffer(alg, n=max(256, b * 2))
        real_cpu, real_item = torch.Tensor.cpu, torch.Tensor.item
        n = [0]

        def counted_cpu(self, *a, _f=real_cpu, **k):
            n[0] += 1
            return _f(self, *a, **k)

        def counted_item(self, *a, _f=real_item, **k):
            n[0] += 1
            return _f(self, *a, **k)

        obs = torch.randn(b, NUM_OBS)
        with torch.no_grad():
            actor_obs = alg.policy.actor_obs_normalizer(obs)
            _, actions, q, _, _ = alg._estep_sample(actor_obs, obs)
            torch.Tensor.cpu, torch.Tensor.item = counted_cpu, counted_item
            try:
                alg._estep_weights(q, actions, obs)
            finally:
                torch.Tensor.cpu, torch.Tensor.item = real_cpu, real_item
        counts[b] = n[0]
    assert counts[16] == counts[128], f"host syncs scale with batch size: {counts}"


def test_update_runs_and_moves_actor() -> None:
    before = None
    alg = _make_alg()
    _fill_buffer(alg)
    before = [p.detach().clone() for p in alg.policy.actor.parameters()]
    losses = alg.update()
    assert all(torch.isfinite(torch.as_tensor(v)) for v in losses.values() if isinstance(v, (int, float)))
    after = list(alg.policy.actor.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_lambda_max_balanced_measures_then_freezes() -> None:
    """`lambda_max_mode="balanced"` must stop moving, so saturation stays comparable over time."""
    alg = _make_alg(lambda_max_mode="balanced", lambda_max_warmup_updates=2, lambda_max_balanced_mult=1.0)
    _fill_buffer(alg)
    assert not alg._lambda_max_frozen
    for _ in range(4):
        alg.update()
    assert alg._lambda_max_frozen
    frozen = alg._lambda_max_used
    for _ in range(3):
        alg.update()
    assert alg._lambda_max_used == frozen
    lo, hi = alg.lambda_max_clamp
    assert lo <= frozen <= hi


def test_cost_critic_passive_disables_the_constraint() -> None:
    """Passive mode pins lambda at zero, giving a second independent route to 'recovers MPO'."""
    alg = _make_alg(cost_critic_passive=True, lambda_max=0.0)
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert info["lambda_max"] == 0.0
    assert info["lambda_frac_zero"] == 1.0
    assert info["lambda_frac_at_cap"] == 0.0


# --------------------------------------------------------------------------------------------
# Phase 2: the dispersion gate
# --------------------------------------------------------------------------------------------


def _make_dist_policy():
    """SafeActorCritic with a distributional cost critic, for the dispersion gate."""
    from safe_rl.modules import SafeActorCritic

    return SafeActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        num_costs=1,
        cost_critic_type="distributional",
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"num_atoms": 51, "v_min": 0.0, "v_max": 10.0, "network_kwargs": {"hidden_dims": [32, 32]}},
    )


def test_shared_lambda_solves_the_batch_constraint() -> None:
    """The gate's fallback is CVPO's own constraint, solved by the same bisection."""
    from safe_rl.common.per_state_dual import expected_qc, solve_shared_lambda

    q_r, q_c = _measured_problem(seed=3, b=64)
    eta = torch.tensor(0.06, dtype=q_c.dtype)
    d = q_c.mean(dim=0) - 0.25 * q_c.std(dim=0)
    lam = solve_shared_lambda(q_r, q_c, eta, d, lam_max=3.0, iters=40)
    assert lam.dim() == 0 and 0.0 < float(lam) < 3.0
    achieved = expected_qc(q_r, q_c, eta, lam.expand(q_c.shape[1])).mean()
    assert abs(float(achieved - d.mean())) < 1e-8


def test_gate_routes_flat_states_to_the_shared_lambda() -> None:
    """States with no across-action spread take the batch lambda; the rest keep their own root."""
    from safe_rl.common.per_state_dual import solve_per_state_dual as solve

    q_r, q_c = _measured_problem(seed=4, b=64)
    q_c[:, :32] = q_c[:, :32].mean(dim=0, keepdim=True)  # exactly flat -> lambda_b unidentifiable
    d = torch.clamp_min(q_c.mean(dim=0) - 0.25 * q_c.std(dim=0), 0.0)
    gate = q_c.std(dim=0) >= torch.quantile(q_c.std(dim=0), 0.5)

    sol = solve(q_r, q_c, d, 0.1, 3.0, sweeps=4, gate=gate)
    assert sol.lam_shared is not None and sol.lam_shared.dim() == 0
    # Every ungated state carries exactly the shared value, and it is one value.
    assert torch.allclose(sol.lam[~gate], sol.lam_shared.expand((~gate).sum()))
    assert float(sol.lam[gate].std()) > 0.0  # the gated ones still vary
    # Ungated here means flat: their own root would have been arbitrary.
    assert float(q_c[:, ~gate].std(dim=0).max()) < float(q_c[:, gate].std(dim=0).min())


def test_gate_off_is_identical_to_no_gate() -> None:
    """`lambda_gate_mode: none` must leave the Phase-1 path bit-identical."""
    from safe_rl.common.per_state_dual import solve_per_state_dual as solve

    q_r, q_c = _measured_problem(seed=5, b=48)
    d = torch.clamp_min(q_c.mean(dim=0) - 0.25 * q_c.std(dim=0), 0.0)
    a = solve(q_r, q_c, d, 0.1, 3.0, sweeps=3)
    b = solve(q_r, q_c, d, 0.1, 3.0, sweeps=3, gate=torch.ones(48, dtype=torch.bool))
    assert torch.equal(a.lam, b.lam) and torch.equal(a.eta, b.eta)


@pytest.mark.parametrize("mode", ["spread", "dispersion", "both"])
def test_gate_modes_run_and_report(mode) -> None:
    policy = _make_dist_policy() if mode in ("dispersion", "both") else _make_policy()
    from safe_rl.algorithms import CVPOPerState

    alg = CVPOPerState(
        policy,
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        lambda_mode="per_state",
        lambda_max=3.0,
        lambda_gate_mode=mode,
        gate_spread_quantile=0.5,
        device="cpu",
    )
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert 0.0 <= info["frac_states_gated"] <= 1.0
    for k in ("lambda_shared", "gate_threshold", "ess_gated", "ess_ungated"):
        assert k in info


def test_dispersion_gate_requires_a_distributional_cost_critic() -> None:
    with pytest.raises(ValueError, match="distributional cost critic"):
        _make_alg(lambda_gate_mode="dispersion")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(lambda_gate_mode="nope"), "lambda_gate_mode must be"),
        (dict(gate_spread_quantile=1.0), "gate_spread_quantile must be"),
        (dict(lambda_head_mode="nope"), "lambda_head_mode must be"),
    ],
)
def test_phase2_phase3_config_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _make_alg(**kwargs)


def test_gate_off_reports_all_states_gated_and_no_shared_lambda() -> None:
    """With the gate off, `frac_states_gated` is 1.0 and the shared lambda is NaN, not 0.

    A shared lambda silently reported as 0.0 when none was solved would read as "the batch
    constraint is slack" on a dashboard.
    """
    import math

    alg = _make_alg()
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert info["frac_states_gated"] == 1.0
    assert math.isnan(info["lambda_shared"])
    assert math.isnan(info["ess_ungated"])  # there are no ungated states


# --------------------------------------------------------------------------------------------
# Phase 3: the amortized lambda head
# --------------------------------------------------------------------------------------------


def test_lambda_head_is_a_regressor_not_a_dual() -> None:
    """CLAUDE.md rule 1's carve-out, held to its terms.

    The head may have an optimizer, because its loss is a supervised regression onto targets the
    convex solve already produced. What it must never do is put a gradient into the cost critic or
    the actor -- that would be the Lagrangian-backprop design this method exists to replace.
    """
    alg = _make_alg(lambda_head_mode="observer")
    _fill_buffer(alg)
    for p in alg.policy.parameters():
        p.grad = None
    alg.update()

    head_params = {id(p) for p in alg.lambda_head.parameters()}
    for group in alg.lambda_head_optimizer.param_groups:
        for p in group["params"]:
            assert id(p) in head_params, "the head's optimizer reaches beyond the head"
    # No policy parameter is in the head's optimizer, and vice versa.
    for group in alg.actor_optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in head_params


def test_observer_mode_does_not_change_the_policy_update() -> None:
    """ "observer" trains the head while the exact solve still drives the M-step.

    That ordering is the point: the head's regression error becomes measurable before it is ever
    trusted with the policy.
    """
    torch.manual_seed(0)
    off = _make_alg()
    torch.manual_seed(0)
    obs = _make_alg(lambda_head_mode="observer")
    obs.policy.load_state_dict(off.policy.state_dict())

    for alg in (off, obs):
        torch.manual_seed(7)
        _fill_buffer(alg)
        torch.manual_seed(11)
        alg.update()
    for a, b in zip(off.policy.actor.parameters(), obs.policy.actor.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_lambda_head_learns_the_kkt_solution() -> None:
    """The head must beat the constant predictor; otherwise it is not worth amortizing."""
    from safe_rl.modules.lambda_head import LambdaHead, lambda_head_loss

    torch.manual_seed(0)
    n, lam_max = 512, 3.0
    obs = torch.randn(n, NUM_OBS)
    # A learnable target with mass on both corners, as the real KKT solution has.
    raw = 1.5 + 2.0 * obs[:, 0]
    lam = raw.clamp(0.0, lam_max)
    inactive, at_cap = lam <= 0.0, lam >= lam_max

    head = LambdaHead(NUM_OBS, lam_max, hidden_dims=(64, 64))
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(400):
        loss, diag = lambda_head_loss(head, obs, lam, inactive, at_cap)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert diag["lambda_head_r2"] > 0.9, diag
    assert diag["lambda_head_class_acc"] > 0.95, diag


def test_amortized_mode_uses_the_head_but_still_solves_for_targets() -> None:
    """ "amortized" swaps the head in; the bisection still runs, so KKT diagnostics stay comparable."""
    alg = _make_alg(lambda_head_mode="amortized")
    _fill_buffer(alg)
    alg.update()
    info = alg.get_penalty_info()
    assert "lambda_head_mae" in info and "lambda_head_r2" in info
    # The exact solve's own KKT residual is still reported -- it is what the head is scored against.
    assert "dual_residual_lambda_interior_absmax" in info


def test_lambda_head_predict_snaps_the_corners() -> None:
    from safe_rl.modules.lambda_head import INACTIVE, INFEASIBLE, INTERIOR, LambdaHead

    head = LambdaHead(NUM_OBS, lambda_max=3.0, hidden_dims=(8,))
    obs = torch.randn(6, NUM_OBS)
    with torch.no_grad():
        logits, _ = head(obs)
        for cls, expect in ((INACTIVE, 0.0), (INFEASIBLE, 3.0)):
            logits.zero_()
            logits[:, cls] = 10.0
            head.net = _ConstantNet(logits, head)
            assert torch.allclose(head.predict(obs), torch.full((6,), expect))
        head.net = _ConstantNet(torch.nn.functional.one_hot(torch.full((6,), INTERIOR), 3).float() * 10.0, head)
        assert torch.all((head.predict(obs) > 0.0) & (head.predict(obs) < 3.0))


class _ConstantNet(torch.nn.Module):
    """Returns fixed class logits with a mid-range interior value, to test `predict`'s branches."""

    def __init__(self, logits, head):
        super().__init__()
        self.out = torch.cat([logits, torch.zeros(logits.shape[0], 1)], dim=1)

    def forward(self, _obs):
        return self.out


def test_reachable_ask_is_always_inside_the_trust_region() -> None:
    """The default target asks for a fraction of a *provably attainable* reduction.

    `reachable_qc_min` is the smallest `E_{q_b}[Q_c]` the KL budget permits, so an ask of
    `rho * (C_now_b - floor_b)` with rho < 1 can never be infeasible on the KL side -- which is
    what lets this mode work without knowing the cost critic's level or spread. A sigma-sized ask
    cannot make that promise: the reachable drop measures ~0.44 sigma, so kappa = 0.25 is silently
    asking for well over half of it, and saturates once lambda_max binds too.
    """
    from safe_rl.common.per_state_dual import reachable_qc_min

    for seed, scale in ((0, 1.0), (1, 0.05), (2, 30.0)):
        q_r, q_c = _measured_problem(seed=seed, b=64)
        q_c = q_c * scale
        floor = reachable_qc_min(q_c, 0.1)
        c_now = q_c.mean(dim=0)
        assert torch.all(floor <= c_now + 1e-9), "the reachable floor must sit below the current cost"
        d = c_now - 0.25 * (c_now - floor)
        sol = solve_per_state_dual(q_r * scale, q_c, d, 0.1, 3.0, sweeps=8)
        assert float(sol.lam_at_cap.double().mean()) == 0.0, f"saturated at scale {scale}"
        # KKT, not equality: interior states sit exactly on the target, but a state can also come
        # in already under it. `C_now_b` is the UNWEIGHTED mean over candidates, whereas at
        # lambda_b = 0 the E-step is already at the reward-weighted mean -- so where reward and
        # cost are anti-correlated across actions, part of the ask is delivered by the reward term
        # for free and the constraint is simply slack.
        interior = ~(sol.lam_inactive | sol.lam_at_cap)
        residual = sol.eqc - d
        # Tolerance relative to the cost scale: bisection resolves lambda to lam_max/2^iters, and
        # that maps to an absolute cost residual proportional to the magnitude of Q_c.
        tol = 1e-9 * max(float(q_c.abs().max()), 1.0)
        assert float(residual[interior].abs().max()) < tol
        slack = residual[sol.lam_inactive]
        assert slack.numel() == 0 or float(slack.max()) <= tol

        # The drop is a fixed fraction of sigma, set by eps rather than by the critic's scale.
        assert 0.3 < float(((c_now - floor) / q_c.std(dim=0)).median()) < 0.6


def test_reachable_mode_is_the_default_and_rho_is_bounded() -> None:
    alg = _make_alg()
    assert alg.dstate_beta_mode == "reachable" and alg.dstate_reachable_rho == 0.25
    with pytest.raises(ValueError, match="dstate_reachable_rho must be"):
        _make_alg(dstate_reachable_rho=1.0)
    with pytest.raises(ValueError, match="dstate_beta_mode must be"):
        _make_alg(dstate_beta_mode="nope")


def test_reachable_target_computes_its_floor_once() -> None:
    """The target and the feasibility probe share one bisection rather than running two."""
    import safe_rl.algorithms.cvpo_per_state as mod

    alg = _make_alg(feasibility_probe_interval=1)
    _fill_buffer(alg)
    calls = [0]
    real = mod.reachable_qc_min

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    mod.reachable_qc_min = counted
    try:
        alg.update()
    finally:
        mod.reachable_qc_min = real
    assert calls[0] == alg.num_updates_per_step, f"{calls[0]} bisections for {alg.num_updates_per_step} updates"


def test_kkt_residual_excludes_ungated_states() -> None:
    """The KKT gate must score only states that were solved per-state.

    An ungated state carries the batch-level lambda by design; it satisfies the *batch* constraint
    and has no reason to satisfy its own stationarity. Counting it made
    `dual_residual_lambda_interior_absmax` jump to ~1e-1 the moment the gate was switched on, which
    reads as a broken solver rather than as the gate working.
    """
    alg_gated = _make_alg(lambda_gate_mode="spread", gate_spread_quantile=0.5)
    alg_plain = _make_alg()
    for alg in (alg_gated, alg_plain):
        torch.manual_seed(3)
        _fill_buffer(alg)
        torch.manual_seed(5)
        alg.update()
    for alg in (alg_gated, alg_plain):
        assert alg.get_penalty_info()["dual_residual_lambda_interior_absmax"] < 1e-4
    assert alg_gated.get_penalty_info()["frac_states_gated"] < 1.0
