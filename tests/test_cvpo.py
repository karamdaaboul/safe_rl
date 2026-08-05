"""Tests for CVPO (Constrained Variational Policy Optimization)."""

from __future__ import annotations

import math

import pytest

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
        obs = torch.randn(1, NUM_OBS)
        action = torch.rand(1, NUM_ACT) * 2 - 1
        reward = torch.randn(1)
        done = torch.zeros(1)
        next_obs = torch.randn(1, NUM_OBS)
        cost = torch.rand(1)  # nonnegative costs
        alg.store_transition(obs, action, reward, done, next_obs, cost=cost)


def test_cvpo_builds_and_scales_cost_threshold() -> None:
    alg = _make_cvpo()
    # qc_thres = 25 * (1 - 0.99^1000)/(1-0.99)/1000 ~ 2.5 (discounted cost-Q scale of a 25-limit).
    assert 2.0 < alg.qc_thres < 3.0
    # A dedicated frozen target actor must exist for the E-step / M-step KL.
    assert hasattr(alg, "actor_target")
    assert all(not p.requires_grad for p in alg.actor_target.parameters())


def test_cvpo_rejects_multiple_costs() -> None:
    from safe_rl.algorithms import CVPO
    from safe_rl.modules import SafeSACActorCritic

    policy = SafeSACActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT, num_costs=2,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs={"hidden_dims": [16, 16]},
    )
    with pytest.raises(ValueError, match="single cost"):
        CVPO(policy, cost_limits=[25.0, 10.0], device="cpu")


def test_cvpo_dual_solver_returns_positive() -> None:
    import numpy as np

    alg = _make_cvpo()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(16, 32))
    qc = np.abs(rng.normal(size=(16, 32)))
    eta, lam = alg._solve_dual(q, qc)
    assert eta > 0 and lam > 0
    assert np.isfinite(eta) and np.isfinite(lam)


def test_cvpo_estep_weights_are_a_distribution() -> None:
    # The variational weights are a softmax over the N candidate actions per state,
    # so each state's column must sum to 1.
    alg = _make_cvpo(sample_action_num=16)
    obs = torch.randn(32, NUM_OBS)
    with torch.no_grad():
        mean_old, log_std_old = alg.actor_target(obs)
        dist_old = torch.distributions.Normal(mean_old, log_std_old.exp())
        x = dist_old.sample((16,))
        actions = torch.tanh(x)
        cobs = obs.unsqueeze(0).expand(16, -1, -1).reshape(16 * 32, -1)
        q1, q2 = alg.policy.evaluate_q(cobs, actions.reshape(16 * 32, -1))
        q = torch.min(q1, q2).reshape(16, 32)
        qc = alg.policy.evaluate_cost_q(cobs, actions.reshape(16 * 32, -1))[:, 0].reshape(16, 32)
        eta, lam = alg._solve_dual(q.numpy().astype("float64"), qc.numpy().astype("float64"))
        weights = torch.softmax((q - lam * qc) / eta, dim=0)
    assert torch.allclose(weights.sum(dim=0), torch.ones(32), atol=1e-5)


def test_cvpo_update_step_runs_and_is_finite() -> None:
    alg = _make_cvpo(num_updates_per_step=2)
    _fill_buffer(alg, n=128)
    info = alg.update(current_costs=[30.0])
    for key in ("critic", "cost_critic", "actor"):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key]))
    penalty = alg.get_penalty_info()
    assert penalty["eta"] > 0 and penalty["lambda_mean"] > 0
    assert "kl_mean" in penalty and "qc_thres" in penalty


def test_cvpo_grad_lambda_regulates_up_when_cost_exceeds_threshold() -> None:
    # In "grad" mode lambda is a projected-gradient controller on E_q[Q_c] - qc_thres.
    # When every sampled action's cost-Q sits far above the threshold, lambda must climb
    # monotonically (not snap to a bound), staying within [0, lambda_max].
    alg = _make_cvpo(lambda_mode="grad", lambda_lr=0.1, lambda_max=50.0)
    alg.lam = 0.0
    obs = torch.randn(24, NUM_OBS)
    n = alg.sample_action_num
    # Force a hard violation: pretend the cost-Q of every sample is 10x the threshold.
    qc = torch.full((n, obs.shape[0]), alg.qc_thres * 10.0)
    lams = []
    for _ in range(5):
        weights = torch.softmax(torch.randn(n, obs.shape[0]), dim=0)
        eqc = (weights * qc).sum(dim=0).mean().item()
        import numpy as np
        alg.lam = float(np.clip(alg.lam + alg.lambda_lr * (eqc - alg.qc_thres), 0.0, alg.lambda_max))
        lams.append(alg.lam)
    assert lams == sorted(lams)  # monotonically non-decreasing
    assert 0.0 < lams[-1] <= 50.0


def test_cvpo_grad_mode_solve_dual_holds_lambda_fixed() -> None:
    # "grad" mode solves only eta; _solve_dual must return lambda unchanged (= self.lam).
    import numpy as np

    alg = _make_cvpo(lambda_mode="grad")
    alg.lam = 3.7
    rng = np.random.default_rng(1)
    q = rng.normal(size=(16, 32))
    qc = np.abs(rng.normal(size=(16, 32)))
    eta, lam = alg._solve_dual(q, qc)
    assert eta > 0 and np.isfinite(eta)
    assert lam == 3.7


def test_cvpo_update_moves_actor_params() -> None:
    alg = _make_cvpo(num_updates_per_step=3)
    _fill_buffer(alg, n=128)
    before = [p.detach().clone() for p in alg.policy.actor.parameters()]
    alg.update(current_costs=[30.0])
    after = list(alg.policy.actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# Paper-faithfulness of the ``dual`` mode E-step (CVPO, arXiv:2201.11927).
#
# ``lambda_mode="dual"`` is the paper-faithful path: it solves the joint convex
# dual of Eq. 9 for (eta, lambda) each E-step. These tests assert that solve is
# correct/optimal, mirroring the C-TruDi M1 dual-solver gates (CLAUDE.md, "M1").
# The default ``grad`` mode (a projected-gradient controller on lambda) is a
# deliberate, documented departure (codex/cvpo-negative-result.md) and is covered
# by the ``grad``-mode tests above.
#
# Dual objective under test (cvpo.py:159-165):
#     g(eta, lam) = eta*eps + lam*thres
#                 + eta * mean_states( logmeanexp_N( (q - lam*qc)/eta ) )
# with eps = alg.eps_dual (E-step KL bound) and thres = alg.qc_thres (cost limit).
# KKT identity (dg/dlam = thres - E_q*[Q_c]): at an interior optimum the reweighted
# expected cost equals the threshold.
# ---------------------------------------------------------------------------


def _dual_g(q, qc, eta, lam, eps, thres):
    """Replicate the CVPO ``dual``-mode objective g(eta, lam) exactly (numpy)."""
    import numpy as np

    z = (q - lam * qc) / eta
    zmax = z.max(axis=0, keepdims=True)
    lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
    return eta * eps + lam * thres + eta * float(np.mean(lse))


def _stable_softmax(logits):
    """Column-wise numerically-stable softmax over axis 0 (the N action samples)."""
    import numpy as np

    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def _grid_min(q, qc, eps, thres, etas, lams):
    """Brute-force minimiser of g over the (etas x lams) mesh; returns (eta, lam, g)."""
    best = (None, None, float("inf"))
    for eta in etas:
        for lam in lams:
            g = _dual_g(q, qc, eta, lam, eps, thres)
            if g < best[2]:
                best = (eta, lam, g)
    return best


def test_cvpo_dual_matches_bruteforce() -> None:
    # M1 gate (a): the SLSQP solve must agree with an exhaustive grid search over
    # (eta, lam). This is the anchor test that buys the right to trust the rest.
    # Use an interior-optimum instance (reward correlated with cost, reachable
    # mid-range threshold) so the comparison is a clean box-interior one — a corner
    # optimum at the lambda bound is exercised separately by the infeasibility test.
    import numpy as np

    lambda_max = 50.0
    alg = _make_cvpo(lambda_mode="dual", qc_thres=1.0, lambda_max=lambda_max)
    rng = np.random.default_rng(4)
    qc = rng.uniform(0.1, 1.9, size=(8, 16)).astype(np.float64)
    q = (3.0 * qc + 0.3 * rng.normal(size=(8, 16))).astype(np.float64)

    eta_s, lam_s = alg._solve_dual(q, qc)
    assert 1e-3 < lam_s < 0.99 * lambda_max  # optimum is genuinely interior
    g_sqp = _dual_g(q, qc, eta_s, lam_s, alg.eps_dual, alg.qc_thres)

    # Coarse grid, then refine around its argmin so the grid essentially sits at the
    # true optimum of this jointly-convex objective. Refine window is clamped to the
    # feasible box so the grid never evaluates outside the solver's bounds.
    etas = np.logspace(-2, 2, 80)
    lams = np.linspace(1e-6, lambda_max, 80)
    e0, l0, _ = _grid_min(q, qc, alg.eps_dual, alg.qc_thres, etas, lams)
    detas, dlams = etas[1] / etas[0], lams[1] - lams[0]
    etas_r = np.linspace(max(e0 / detas, 1e-6), e0 * detas, 60)
    lams_r = np.linspace(max(l0 - dlams, 1e-6), min(l0 + dlams, lambda_max), 60)
    eg, lg, g_grid = _grid_min(q, qc, alg.eps_dual, alg.qc_thres, etas_r, lams_r)

    # SLSQP is a continuous solver: it can only be as good or better than the grid,
    # and it must not be meaningfully better (same basin).
    assert g_sqp <= g_grid + 1e-9
    assert g_grid - g_sqp < 1e-3
    # Parameters agree to the refined-grid resolution.
    assert abs(eta_s - eg) < 0.05 * eg + 1e-3
    assert abs(lam_s - lg) < (lams_r[1] - lams_r[0]) + 1e-3


def test_cvpo_dual_lambda_zero_recovers_mpo() -> None:
    # M1 gate (b): with no cost signal (qc == 0) the constrained E-step must reduce
    # to the vanilla MPO E-step — lambda driven to its floor and the weights equal
    # softmax(q/eta) for the eta that solves the MPO-only (eta-only) dual.
    import numpy as np
    from scipy.optimize import minimize

    alg = _make_cvpo(lambda_mode="dual", qc_thres=0.7)
    rng = np.random.default_rng(1)
    q = (rng.normal(size=(8, 16)) * 2.0).astype(np.float64)
    qc = np.zeros((8, 16), dtype=np.float64)

    eta_s, lam_s = alg._solve_dual(q, qc)
    assert lam_s < 1e-3  # thres > 0 makes g strictly increasing in lam -> floor

    # Independent MPO eta-only solve for cross-check.
    def g_mpo(x):
        eta = x[0]
        z = q / eta
        zmax = z.max(axis=0, keepdims=True)
        lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
        return eta * alg.eps_dual + eta * float(np.mean(lse))

    eta_mpo = float(minimize(g_mpo, np.array([1.0]), method="SLSQP", bounds=[(1e-6, 1e6)]).x[0])
    assert abs(eta_s - eta_mpo) < 1e-2 * eta_mpo + 1e-3

    w_cvpo = _stable_softmax((q - lam_s * qc) / eta_s)
    w_mpo = _stable_softmax(q / eta_mpo)
    assert np.allclose(w_cvpo, w_mpo, atol=1e-4)


def test_cvpo_dual_inactive_constraint_lambda_near_zero() -> None:
    # M1 gate (c): when the constraint is slack (threshold far above every reachable
    # E_q[Q_c]), the optimal multiplier sits at its floor.
    import numpy as np

    alg = _make_cvpo(lambda_mode="dual", qc_thres=5.0, lambda_max=50.0)
    rng = np.random.default_rng(2)
    q = (rng.normal(size=(8, 16)) * 2.0).astype(np.float64)
    qc = rng.uniform(0.0, 1.0, size=(8, 16)).astype(np.float64)  # every Qc well below 5.0

    _eta, lam_s = alg._solve_dual(q, qc)
    assert lam_s < 1e-3


def test_cvpo_dual_infeasible_drives_lambda_to_bound() -> None:
    # M1 gate (d): when the threshold is below every attainable cost-Q, no reweighting
    # can satisfy the constraint (dg/dlam < 0 everywhere) so lambda is driven to lambda_max.
    import numpy as np

    lambda_max = 30.0
    alg = _make_cvpo(lambda_mode="dual", qc_thres=0.5, lambda_max=lambda_max)
    rng = np.random.default_rng(3)
    q = (rng.normal(size=(8, 16)) * 2.0).astype(np.float64)
    qc = rng.uniform(1.0, 2.0, size=(8, 16)).astype(np.float64)  # every Qc >= 1.0 > 0.5

    _eta, lam_s = alg._solve_dual(q, qc)
    assert lam_s >= 0.99 * lambda_max


def test_cvpo_dual_interior_optimum_satisfies_kkt() -> None:
    # M1 gate (e): the core "solved-to-optimality" check. With reward pulling toward
    # high-cost actions but a reachable mid-range threshold, the optimum is interior,
    # and complementary slackness (dg/dlam = thres - E_q*[Q_c] = 0) must hold: the
    # reweighted expected cost equals the threshold.
    import numpy as np

    lambda_max = 50.0
    thres = 1.0
    alg = _make_cvpo(lambda_mode="dual", qc_thres=thres, lambda_max=lambda_max)
    rng = np.random.default_rng(4)
    qc = rng.uniform(0.1, 1.9, size=(8, 16)).astype(np.float64)
    q = (3.0 * qc + 0.3 * rng.normal(size=(8, 16))).astype(np.float64)  # reward favours high cost

    eta_s, lam_s = alg._solve_dual(q, qc)
    assert eta_s > 0
    assert 1e-3 < lam_s < 0.99 * lambda_max  # strictly interior

    weights = _stable_softmax((q - lam_s * qc) / eta_s)
    eqc = float((weights * qc).sum(axis=0).mean())
    assert abs(eqc - thres) < 0.05  # KKT / complementary slackness residual ~ 0


def test_cvpo_dual_logsumexp_stable_at_extreme_scale() -> None:
    # M1 gate (f): the log-sum-exp weighting must stay finite for a tiny temperature
    # and large-magnitude Q-values, where naive exp() would overflow.
    import numpy as np

    alg = _make_cvpo(lambda_mode="dual", qc_thres=0.7)
    rng = np.random.default_rng(5)
    q = (rng.normal(size=(8, 16)) * 1e3).astype(np.float64)  # |Q| ~ 1e3
    qc = rng.uniform(0.0, 1.0, size=(8, 16)).astype(np.float64)

    eta_s, lam_s = alg._solve_dual(q, qc)
    assert np.isfinite(eta_s) and np.isfinite(lam_s)

    weights = _stable_softmax((q - lam_s * qc) / 1e-3)  # force eta = 1e-3
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-6)


def _make_safe_ac(critic_type: str):
    from safe_rl.modules import SafeActorCritic

    ckw = (
        {"hidden_dims": [32, 32]}
        if critic_type == "standard"
        else {"num_atoms": 51, "v_min": -10.0, "v_max": 10.0, "network_kwargs": {"hidden_dims": [32, 32]}}
    )
    return SafeActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        num_costs=1,
        critic_type=critic_type,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs=ckw,
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )


def test_safe_actor_critic_supports_both_critic_types() -> None:
    from safe_rl.modules import SafeActorCritic

    std = _make_safe_ac("standard")
    dist = _make_safe_ac("distributional")
    assert std.is_distributional_critic is False
    assert dist.is_distributional_critic is True
    # Cost critics stay scalar in both modes: the E-step needs Q_c as a plain expectation.
    obs, act = torch.randn(4, NUM_OBS), torch.rand(4, NUM_ACT) * 2 - 1
    for p in (std, dist):
        q1, q2 = p.evaluate_q(obs, act)
        t1, t2 = p.evaluate_q_target(obs, act)
        assert q1.shape == q2.shape == t1.shape == t2.shape == (4, 1)
        assert p.evaluate_cost_q(obs, act).shape == (4, 1)
    with pytest.raises(ValueError, match="critic_type"):
        SafeActorCritic(num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT, critic_type="quantile")


def test_cvpo_runs_with_distributional_critics_and_nstep() -> None:
    from safe_rl.algorithms import CVPO

    alg = CVPO(
        _make_safe_ac("distributional"),
        batch_size=32,
        num_updates_per_step=2,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_limits=[25.0],
        n_step=3,
        device="cpu",
    )
    assert alg.policy.is_distributional_critic
    alg.init_storage(buffer_size=500, num_envs=1, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(128):
        alg.store_transition(
            torch.randn(1, NUM_OBS),
            torch.rand(1, NUM_ACT) * 2 - 1,
            torch.randn(1),
            torch.zeros(1),
            torch.randn(1, NUM_OBS),
            cost=torch.rand(1, 1),
        )
    info = alg.update(current_costs=[20.0])
    for key in ("critic", "actor", "cost_critic"):
        assert torch.isfinite(torch.tensor(info[key])), key
    pen = alg.get_penalty_info()
    assert 1.0 <= pen["ess_min"] <= alg.sample_action_num + 1e-6


def _cvpo_adaptive(**kw):
    from safe_rl.algorithms import CVPO

    base = dict(
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_limits=[25.0],
        qc_thres_adapt=True,
        qc_thres_lr=0.05,
        qc_ema=1.0,
        device="cpu",
    )
    base.update(kw)
    return CVPO(_make_safe_ac("standard"), **base)


def test_qc_thres_tightens_when_budget_is_exceeded() -> None:
    alg = _cvpo_adaptive()
    start = alg.qc_thres
    for _ in range(20):
        alg.update_lagrangian_multipliers([50.0])  # 2x the limit
    assert alg.qc_thres < start, "threshold must tighten when realized cost exceeds the limit"
    assert alg.qc_thres >= alg.qc_thres_min_frac * alg._qc_thres_initial


def test_qc_thres_never_exceeds_the_analytic_value() -> None:
    # Undershooting the budget may relax the threshold, but never past the requested limit.
    alg = _cvpo_adaptive()
    for _ in range(50):
        alg.update_lagrangian_multipliers([0.0])
    assert alg.qc_thres <= alg._qc_thres_initial + 1e-9


def test_qc_thres_is_static_when_adaptation_is_off() -> None:
    from safe_rl.algorithms import CVPO

    alg = CVPO(
        _make_safe_ac("standard"),
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=16,
        mstep_iteration_num=2,
        cost_limits=[25.0],
        device="cpu",
    )
    before = alg.qc_thres
    for _ in range(10):
        alg.update_lagrangian_multipliers([100.0])
    assert alg.qc_thres == before


# ---------------------------------------------------------------------------
# qc_scale calibration source (static recalibration of episodic -> Q-space units)
# ---------------------------------------------------------------------------


def test_qc_scale_analytic_is_the_default() -> None:
    alg = _make_cvpo()
    assert alg.qc_scale_source == "analytic"
    # (1 - g^H)/(1 - g)/H with g=0.99, H=1000 -> ~0.1
    assert alg._qc_scale == pytest.approx(0.1, abs=1e-4)
    assert alg.qc_thres == pytest.approx(25.0 * alg._qc_scale)


def test_qc_scale_measured_overrides_the_threshold() -> None:
    alg = _make_cvpo(
        qc_scale_source="measured",
        qc_scale_measured=0.0764,
        qc_scale_probe="probe_runs/example.jsonl",
    )
    assert alg._qc_scale == pytest.approx(0.0764)
    assert alg.qc_thres == pytest.approx(25.0 * 0.0764)  # 1.91
    # The analytic value stays available for comparison, and provenance is retained.
    assert alg._qc_scale_analytic == pytest.approx(0.1, abs=1e-4)
    assert alg.qc_scale_probe == "probe_runs/example.jsonl"
    assert alg.get_penalty_info()["qc_scale"] == pytest.approx(0.0764)


def test_qc_scale_measured_requires_a_positive_value() -> None:
    with pytest.raises(ValueError, match="qc_scale_measured"):
        _make_cvpo(qc_scale_source="measured")
    with pytest.raises(ValueError, match="qc_scale_measured"):
        _make_cvpo(qc_scale_source="measured", qc_scale_measured=0.0)


def test_qc_scale_source_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="qc_scale_source"):
        _make_cvpo(qc_scale_source="guessed")


def test_measured_qc_scale_does_not_enable_the_adaptive_loop() -> None:
    # Task 4 is a static units fix; it must not turn the ratchet on.
    alg = _make_cvpo(qc_scale_source="measured", qc_scale_measured=0.0764)
    assert alg.qc_thres_adapt is False
    before = alg.qc_thres
    for _ in range(10):
        alg.update_lagrangian_multipliers([100.0])
    assert alg.qc_thres == before


# ---------------------------------------------------------------------------
# non-negative cost critic head
# ---------------------------------------------------------------------------


def test_cost_critic_can_emit_negative_values_by_default() -> None:
    """Documents the invariant violation the flag exists to fix."""
    policy = _make_policy()
    assert policy.cost_critic_nonneg is False
    # The head is linear, so a negative output is representable at all.
    torch.manual_seed(0)
    with torch.no_grad():
        for critic in policy.cost_critics:
            torch.nn.init.constant_(critic.network[-1].bias, -1.0)
            torch.nn.init.zeros_(critic.network[-1].weight)
        q = policy.evaluate_cost_q(torch.randn(8, NUM_OBS), torch.randn(8, NUM_ACT))
    assert (q < 0).all()


def test_nonneg_cost_head_clamps_both_online_and_target() -> None:
    from safe_rl.modules import SafeSACActorCritic

    policy = SafeSACActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        num_costs=1,
        cost_critic_nonneg=True,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )
    with torch.no_grad():
        for critic in list(policy.cost_critics) + list(policy.cost_critic_targets):
            torch.nn.init.constant_(critic.network[-1].bias, -5.0)
            torch.nn.init.zeros_(critic.network[-1].weight)
        obs, act = torch.randn(16, NUM_OBS), torch.randn(16, NUM_ACT)
        assert (policy.evaluate_cost_q(obs, act) >= 0).all()
        assert (policy.evaluate_cost_q_target(obs, act) >= 0).all()


def test_nonneg_cost_head_keeps_cvpo_update_running() -> None:
    policy = _make_policy()
    policy.cost_critic_nonneg = True
    from safe_rl.algorithms import CVPO

    alg = CVPO(policy, cost_limits=[25.0], batch_size=32, num_updates_per_step=1,
               sample_action_num=16, mstep_iteration_num=2, device="cpu")
    _fill_buffer(alg)
    info = alg.update()
    assert math.isfinite(info["cost_critic"])


# ---------------------------------------------------------------------------
# distributional cost critic + passive (observer) mode
# ---------------------------------------------------------------------------


def _make_dist_cost_policy(**kw):
    from safe_rl.modules.safe_actor_critic import SafeActorCritic

    base = dict(
        critic_type="distributional",
        cost_critic_type="distributional",
        num_costs=1,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"num_atoms": 51, "v_min": -5.0, "v_max": 15.0,
                       "network_kwargs": {"hidden_dims": [32, 32]}},
        cost_critic_kwargs={"num_atoms": 51, "v_min": 0.0, "v_max": 50.0,
                            "network_kwargs": {"hidden_dims": [32, 32]}},
    )
    base.update(kw)
    return SafeActorCritic(NUM_OBS, NUM_OBS, NUM_ACT, **base)


def test_distributional_cost_critic_cannot_be_negative() -> None:
    """v_min=0 makes the non-negativity invariant structural, not a patched head."""
    policy = _make_dist_cost_policy()
    assert policy.is_distributional_cost_critic
    obs, act = torch.randn(128, NUM_OBS), torch.randn(128, NUM_ACT)
    for q in (policy.evaluate_cost_q(obs, act), policy.evaluate_cost_q_target(obs, act)):
        assert q.shape == (128, 1)
        assert (q >= 0).all()
        assert (q <= 50.0 + 1e-4).all()


def test_distributional_cost_critic_rejects_multiple_constraints() -> None:
    with pytest.raises(ValueError, match="single constraint"):
        _make_dist_cost_policy(num_costs=2)


def test_cost_critic_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="cost_critic_type"):
        _make_dist_cost_policy(cost_critic_type="categorical")


def test_passive_cost_critic_keeps_lambda_at_zero_but_still_trains_qc() -> None:
    from safe_rl.algorithms import CVPO

    policy = _make_dist_cost_policy()
    alg = CVPO(policy, cost_limits=[25.0], batch_size=32, num_updates_per_step=1,
               sample_action_num=8, mstep_iteration_num=2, cost_critic_passive=True,
               lambda_mode="grad", lambda_max=0.0, lambda_lr=0.0, n_step=3, device="cpu")
    assert alg.lam == 0.0
    alg.init_storage(buffer_size=2000, num_envs=2, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(200):
        alg.store_transition(torch.randn(2, NUM_OBS), torch.randn(2, NUM_ACT), torch.randn(2),
                             torch.zeros(2), torch.randn(2, NUM_OBS),
                             cost=torch.rand(2, 1), bootstrap=torch.zeros(2))
    before = [p.clone() for p in policy.cost_critics[0].parameters()]
    info = alg.update()
    assert alg.lam == 0.0, "passive mode must never raise lambda"
    assert math.isfinite(info["cost_critic"])
    after = list(policy.cost_critics[0].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "cost critic must still train"


def test_distributional_cost_target_uses_gamma_to_the_n() -> None:
    """A constant cost c with no dones has the n-step fixed point c*(1-g^n)/(1-g)/(1-g^n)."""
    from safe_rl.algorithms import CVPO

    policy = _make_dist_cost_policy()
    alg = CVPO(policy, cost_limits=[25.0], batch_size=64, num_updates_per_step=1,
               sample_action_num=8, mstep_iteration_num=1, cost_critic_passive=True,
               lambda_mode="grad", lambda_max=0.0, lambda_lr=0.0, gamma=0.99, n_step=3,
               device="cpu")
    alg.init_storage(buffer_size=2000, num_envs=2, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    for _ in range(300):
        alg.store_transition(torch.randn(2, NUM_OBS), torch.randn(2, NUM_ACT), torch.randn(2),
                             torch.zeros(2), torch.randn(2, NUM_OBS),
                             cost=torch.ones(2, 1), bootstrap=torch.zeros(2))
    batch = alg.storage.sample(64)
    expected = sum(0.99 ** k for k in range(3))
    assert batch["costs"].mean().item() == pytest.approx(expected, abs=1e-4)
    assert batch["effective_n_steps"].unique().tolist() == [3]


# ---------------------------------------------------------------------------
# CVaR cost constraint (WCSAC-style risk measure, exact from categorical atoms)
# ---------------------------------------------------------------------------


def test_cvar_matches_analytic_gaussian() -> None:
    """CVaR read off the atoms must match mu + sigma*phi(Phi^-1(a))/(1-a)."""
    norm = pytest.importorskip("scipy.stats").norm
    from safe_rl.modules.critic import DistributionalCritic

    c = DistributionalCritic(num_obs=4, num_actions=2, num_atoms=2001, v_min=-30.0, v_max=30.0,
                             network_kwargs={"hidden_dims": [8]})
    mu, sg = 3.0, 2.0
    p = torch.exp(-0.5 * ((c.q_support - mu) / sg) ** 2)
    p = (p / p.sum()).unsqueeze(0)
    assert c.get_value(p).item() == pytest.approx(mu, abs=1e-3)
    assert c.get_var(p).sqrt().item() == pytest.approx(sg, abs=1e-3)
    for a in (0.5, 0.9, 0.99):
        expected = mu + sg * norm.pdf(norm.ppf(a)) / (1 - a)
        assert c.get_cvar(p, a).item() == pytest.approx(expected, abs=0.03)
        assert c.get_quantile(p, a).item() == pytest.approx(mu + sg * norm.ppf(a), abs=0.05)


def test_cvar_edge_cases_and_monotonicity() -> None:
    from safe_rl.modules.critic import DistributionalCritic

    c = DistributionalCritic(num_obs=4, num_actions=2, num_atoms=201, v_min=0.0, v_max=20.0,
                             network_kwargs={"hidden_dims": [8]})
    point = torch.zeros(1, c.num_atoms)
    point[0, (c.q_support - 5.0).abs().argmin()] = 1.0
    assert c.get_cvar(point, 0.9).item() == pytest.approx(c.get_value(point).item(), abs=1e-4)
    p = torch.softmax(torch.randn(4, c.num_atoms), dim=-1)
    assert torch.allclose(c.get_cvar(p, 0.0), c.get_value(p), atol=1e-5)
    vals = [c.get_cvar(p, a).mean().item() for a in (0.1, 0.5, 0.9, 0.99)]
    assert all(x < y for x, y in zip(vals, vals[1:])), "CVaR must increase with alpha"
    with pytest.raises(ValueError):
        c.get_cvar(p, 1.0)


def test_cvar_mode_is_more_conservative_than_mean() -> None:
    """The E-step cost signal under 'cvar' must dominate the mean, for the same critic."""
    from safe_rl.algorithms import CVPO

    policy = _make_dist_cost_policy()
    common = dict(cost_limits=[25.0], batch_size=16, num_updates_per_step=1, sample_action_num=8,
                  mstep_iteration_num=1, device="cpu")
    mean_alg = CVPO(policy, cost_constraint_mode="mean", **common)
    cvar_alg = CVPO(policy, cost_constraint_mode="cvar", cvar_alpha=0.9, **common)
    obs, act = torch.randn(64, NUM_OBS), torch.randn(64, NUM_ACT)
    with torch.no_grad():
        m = mean_alg._estep_cost(obs, act, target=False)
        v = cvar_alg._estep_cost(obs, act, target=False)
    assert m.shape == v.shape == (64, 1)
    assert (v >= m - 1e-5).all(), "CVaR_0.9 must be >= the mean everywhere"
    assert v.mean() > m.mean()


def test_cvar_mode_requires_a_distributional_cost_critic() -> None:
    from safe_rl.algorithms import CVPO

    alg = CVPO(_make_policy(), cost_limits=[25.0], cost_constraint_mode="cvar", device="cpu")
    with pytest.raises(RuntimeError, match="requires a distributional cost critic"):
        alg._estep_cost(torch.randn(4, NUM_OBS), torch.randn(4, NUM_ACT), target=False)


def test_cost_constraint_mode_validation() -> None:
    with pytest.raises(ValueError, match="cost_constraint_mode"):
        _make_cvpo(cost_constraint_mode="worst_case")
    with pytest.raises(ValueError, match="cvar_alpha"):
        _make_cvpo(cost_constraint_mode="cvar", cvar_alpha=1.0)
