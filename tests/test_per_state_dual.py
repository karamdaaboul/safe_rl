"""M1-style gates for the batched per-state E-step dual (global eta, per-state lambda).

The scalar CVPO dual earned trust from `tests/test_cvpo.py`'s brute-force agreement test. This
file does the same job for the per-state solver, and adds the two gates that only a *batched*
solver needs: agreement with a naive per-state Python loop (which catches every broadcasting bug
a vectorised implementation can have), and the absence of any host synchronisation inside the
solve (which is the entire reason this solver exists).

Everything here runs in float64 -- bisection roots agree with a brute-force grid to ~1e-3 in
float32 but to ~1e-12 in float64, and a test that cannot distinguish a real error from rounding
is not a gate.
"""

from __future__ import annotations

import math
import numpy as np

import pytest

torch = pytest.importorskip("torch")

from safe_rl.common.per_state_dual import (  # noqa: E402
    dual_objective,
    effective_sample_size,
    estep_weights,
    expected_qc,
    per_state_kl,
    reachable_qc_min,
    solve_eta,
    solve_lambda,
    solve_per_state_dual,
)

EPS = 0.1
LAM_MAX = 8.0


def _problem(seed: int = 0, k: int = 32, b: int = 24, hard: int = 0, flat: int = 0):
    """Random [K, B] (Q_r, Q_c) with optional infeasible and zero-spread states."""
    g = torch.Generator().manual_seed(seed)
    q_r = torch.randn(k, b, generator=g, dtype=torch.float64) * 2.0
    q_c = torch.rand(k, b, generator=g, dtype=torch.float64) * 3.0
    if hard:
        q_c[:, :hard] += 20.0  # every candidate action expensive -> unreachable target
    if flat:
        q_c[:, hard : hard + flat] = 0.4  # no across-action spread -> lambda has no signal
    return q_r, q_c


def _target(b: int, value: float = 1.2) -> torch.Tensor:
    return torch.full((b,), value, dtype=torch.float64)


# --------------------------------------------------------------------------------------------
# 1. The batching itself
# --------------------------------------------------------------------------------------------


def test_batched_equals_per_state_python_loop() -> None:
    """The vectorised solve must equal solving each state on its own, to round-off.

    This is the highest value-per-line test in the file: a per-state solver that is *only* ever
    run batched has no other way to catch a broadcasting mistake, and a broadcast bug here would
    silently mix one state's Q values into another's multiplier.
    """
    q_r, q_c = _problem(seed=1, hard=2, flat=2)
    d = _target(q_r.shape[1])
    eta = torch.tensor(0.9, dtype=torch.float64)

    batched = solve_lambda(q_r, q_c, eta, d, LAM_MAX, iters=40)
    for b in range(q_r.shape[1]):
        single = solve_lambda(q_r[:, b : b + 1], q_c[:, b : b + 1], eta, d[b : b + 1], LAM_MAX, iters=40)
        assert torch.allclose(single[0], batched[b], atol=1e-12), f"state {b}"

    # ... and the same for the full coordinate descent's weights.
    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3)
    recomputed = estep_weights(q_r, q_c, sol.eta, sol.lam)
    assert torch.allclose(sol.weights, recomputed, atol=1e-14)


# --------------------------------------------------------------------------------------------
# 2-4. Agreement with brute force and with SciPy
# --------------------------------------------------------------------------------------------


def _per_state_dual_term(q_r_b, q_c_b, d_b, eta, lam_grid):
    """g's lambda-dependent part at one state, evaluated on a grid of lambda. Shape [len(grid)]."""
    z = (q_r_b.unsqueeze(1) - lam_grid.unsqueeze(0) * q_c_b.unsqueeze(1)) / eta  # [K, G]
    return lam_grid * d_b + eta * (torch.logsumexp(z, dim=0) - math.log(q_r_b.shape[0]))


def test_lambda_matches_bruteforce_grid() -> None:
    """M1 gate (a), per state: bisection agrees with exhaustive minimisation of g over lambda."""
    q_r, q_c = _problem(seed=2, k=6, b=8)
    d = _target(8, value=1.0)
    eta = torch.tensor(0.7, dtype=torch.float64)
    grid = torch.linspace(0.0, LAM_MAX, 40001, dtype=torch.float64)
    step = float(grid[1] - grid[0])

    lam = solve_lambda(q_r, q_c, eta, d, LAM_MAX, iters=45)
    for b in range(q_r.shape[1]):
        g = _per_state_dual_term(q_r[:, b], q_c[:, b], d[b], eta, grid)
        ref = grid[int(g.argmin())]
        assert abs(float(ref - lam[b])) < 2 * step, f"state {b}: {float(lam[b])} vs {float(ref)}"


def test_eta_matches_bruteforce_grid() -> None:
    """The eta block agrees with exhaustive minimisation of g over a log-spaced eta grid."""
    q_r, q_c = _problem(seed=3, k=16, b=12)
    d = _target(12)
    lam = torch.linspace(0.0, 3.0, 12, dtype=torch.float64)
    grid = torch.logspace(-2, 2, 20001, dtype=torch.float64)

    eta = solve_eta(q_r, q_c, lam, EPS, iters=50)
    g = torch.stack([dual_objective(q_r, q_c, d, EPS, e, lam) for e in grid])
    ref = grid[int(g.argmin())]
    assert abs(float(eta - ref)) / float(ref) < 5e-3


def test_coordinate_descent_matches_scipy_joint_solve() -> None:
    """Coordinate descent reaches the same optimum as a joint SLSQP over (eta, lambda_1..lambda_B).

    SciPy is an *oracle here only*. It is deliberately not a runtime dependency of the solver --
    removing it from the E-step is the point of this module -- so do not "simplify" the solver to
    match this test.
    """
    scipy_opt = pytest.importorskip("scipy.optimize")
    q_r, q_c = _problem(seed=4, k=5, b=4)
    d = _target(4, value=1.1)
    qr_np, qc_np, d_np = q_r.numpy(), q_c.numpy(), d.numpy()

    def g(x):
        eta, lam = x[0], x[1:]
        z = (qr_np - lam[None, :] * qc_np) / eta
        zmax = z.max(axis=0, keepdims=True)
        lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
        return eta * EPS + float(np.mean(lam * d_np)) + eta * float(np.mean(lse))

    res = scipy_opt.minimize(
        g,
        np.concatenate([[1.0], np.full(4, 0.5)]),
        method="SLSQP",
        bounds=[(1e-3, 1e4)] + [(0.0, LAM_MAX)] * 4,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=25, lam_iters=45, eta_iters=45)
    ours = float(dual_objective(q_r, q_c, d, EPS, sol.eta, sol.lam))
    assert ours <= float(res.fun) + 1e-4, f"ours {ours} worse than SLSQP {res.fun}"
    assert np.allclose(sol.lam.numpy(), res.x[1:], atol=1e-2)


# --------------------------------------------------------------------------------------------
# 5-8. The KKT corners
# --------------------------------------------------------------------------------------------


def test_lambda_max_zero_recovers_mpo() -> None:
    """M1 gate (b): with the constraint disabled the weights are exactly MPO's exp(Q_r/eta).

    Doubles as a cross-solver check on eta: the torch bisection must agree with MPO's SLSQP.
    """
    from safe_rl.algorithms.mpo import MPO

    q_r, q_c = _problem(seed=5)
    d = _target(q_r.shape[1])
    sol = solve_per_state_dual(q_r, q_c, d, EPS, lam_max=0.0, sweeps=4, eta_iters=50)
    assert torch.all(sol.lam == 0.0)
    assert not bool(sol.lam_at_cap.any())

    eta_mpo = MPO._solve_eta(_EtaStub(), q_r.numpy())
    assert abs(float(sol.eta) - eta_mpo) / eta_mpo < 1e-3
    assert torch.allclose(sol.weights, torch.softmax(q_r / sol.eta, dim=0), atol=1e-12)


class _EtaStub:
    """Minimal stand-in exposing the two attributes ``MPO._solve_eta`` reads."""

    eps_dual = EPS
    eta = 1.0
    _solver_status = -1.0
    _solver_iters = 0.0


def test_inactive_constraint_gives_zero_lambda() -> None:
    """M1 gate (c): a target above what pi_old already spends leaves lambda at exactly zero."""
    q_r, q_c = _problem(seed=6)
    d = torch.full((q_r.shape[1],), float(q_c.max()) + 1.0, dtype=torch.float64)
    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3)
    assert torch.all(sol.lam == 0.0)
    assert bool(sol.lam_inactive.all())
    assert not bool(sol.lam_at_cap.any())


def test_infeasible_state_pins_at_cap_and_does_not_perturb_neighbours() -> None:
    """M1 gate (d), plus the separability claim the whole design rests on.

    At *fixed* eta the states are independent, so making a few of them infeasible must leave every
    other state's multiplier bit-identical. That is what localises the coupling to eta alone --
    and eta is precisely how an unreachable target contaminates the rest of the batch.
    """
    q_r, q_c = _problem(seed=7)
    d = _target(q_r.shape[1], value=1.0)
    eta = torch.tensor(0.8, dtype=torch.float64)

    lam_clean = solve_lambda(q_r, q_c, eta, d, LAM_MAX, iters=40)
    q_c_hard = q_c.clone()
    q_c_hard[:, :4] += 50.0
    lam_hard = solve_lambda(q_r, q_c_hard, eta, d, LAM_MAX, iters=40)

    assert torch.all(lam_hard[:4] == LAM_MAX)
    assert torch.equal(lam_hard[4:], lam_clean[4:]), "an infeasible state leaked into its neighbours"

    sol = solve_per_state_dual(q_r, q_c_hard, d, EPS, LAM_MAX, sweeps=2)
    assert bool(sol.lam_at_cap[:4].all())
    # The sampled-support test must agree with the lambda-cap certificate on these states.
    assert torch.all(reachable_qc_min(q_c_hard, EPS)[:4] > d[:4])


def test_kkt_residual_per_state() -> None:
    """M1 gate (e): dg/dlambda_b vanishes at interior states and has the right sign at the corners."""
    q_r, q_c = _problem(seed=8, hard=3)
    d = _target(q_r.shape[1], value=1.0)
    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=4, lam_iters=50)
    residual = d - sol.eqc  # dg/dlambda_b

    interior = ~(sol.lam_inactive | sol.lam_at_cap)
    assert bool(interior.any()), "degenerate fixture: no interior state to test"
    assert float(residual[interior].abs().max()) < 1e-9

    # lambda_b = 0 is optimal only if increasing it cannot help: dg/dlambda_b >= 0.
    at_zero = residual[sol.lam_inactive]
    assert at_zero.numel() == 0 or float(at_zero.min()) >= -1e-9
    # lambda_b = lam_max is optimal only if dg/dlambda_b <= 0 there.
    at_cap = residual[sol.lam_at_cap]
    assert at_cap.numel() == 0 or float(at_cap.max()) <= 1e-9


# --------------------------------------------------------------------------------------------
# 9-11. Numerical properties the method rests on
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logsumexp_stability(dtype) -> None:
    """M1 gate (f): extreme scales must not produce NaN or an unnormalised column."""
    g = torch.Generator().manual_seed(9)
    q_r = (torch.randn(24, 16, generator=g) * 1e3).to(dtype)
    q_c = (torch.rand(24, 16, generator=g) * 1e3).to(dtype)
    d = torch.full((16,), 1.0, dtype=dtype)

    for eta in (torch.tensor(1e-3, dtype=dtype), torch.tensor(1e4, dtype=dtype)):
        for lam in (torch.zeros(16, dtype=dtype), torch.full((16,), LAM_MAX, dtype=dtype)):
            w = estep_weights(q_r, q_c, eta, lam)
            assert torch.isfinite(w).all()
            assert torch.allclose(w.sum(dim=0), torch.ones(16, dtype=dtype), atol=1e-5)

    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=2, eta_min=1e-3, eta_max=1e4)
    assert torch.isfinite(sol.weights).all() and torch.isfinite(sol.eta).all()
    assert torch.isfinite(sol.lam).all()


@pytest.mark.parametrize("seed", range(8))
def test_expected_qc_is_nonincreasing_in_lambda(seed: int) -> None:
    """The monotonicity bisection on lambda depends on: d/dlambda E_q[Q_c] = -Var_q(Q_c)/eta <= 0."""
    q_r, q_c = _problem(seed=seed, k=16, b=10)
    eta = torch.tensor(0.6, dtype=torch.float64)
    vals = torch.stack(
        [
            expected_qc(q_r, q_c, eta, torch.full((10,), float(lam_v), dtype=torch.float64))
            for lam_v in torch.linspace(0, LAM_MAX, 120)
        ]
    )
    assert float(vals.diff(dim=0).max()) <= 1e-12


@pytest.mark.parametrize("seed", range(8))
def test_mean_kl_is_nonincreasing_in_eta(seed: int) -> None:
    """Its eta counterpart: dg/deta = eps - KL is monotone increasing, so eta is bisectable."""
    q_r, q_c = _problem(seed=seed, k=16, b=10)
    lam = torch.linspace(0.0, 2.0, 10, dtype=torch.float64)
    kls = torch.stack(
        [per_state_kl(estep_weights(q_r, q_c, e, lam)).mean() for e in torch.logspace(-2, 2, 200, dtype=torch.float64)]
    )
    assert float(kls.diff().max()) <= 1e-12


# --------------------------------------------------------------------------------------------
# 12-14. Convergence, warm start, precision
# --------------------------------------------------------------------------------------------


def test_coordinate_descent_decreases_g_monotonically() -> None:
    """Exact block solves on a jointly convex g can only decrease it. See the module docstring:
    when eta and lambda run to large values this is what proves the solver is not at fault."""
    q_r, q_c = _problem(seed=10, hard=4)
    d = _target(q_r.shape[1], value=1.0)
    gs = [
        float(dual_objective(q_r, q_c, d, EPS, s.eta, s.lam))
        for s in (solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=n) for n in range(1, 8))
    ]
    assert all(b <= a + 1e-9 for a, b in zip(gs, gs[1:])), gs


def test_one_warm_started_sweep_matches_converged() -> None:
    """From a warm eta, a single sweep lands where many sweeps do -- what makes sweeps=1 the default."""
    q_r, q_c = _problem(seed=11)
    d = torch.clamp_min(q_c.mean(dim=0) * 0.99, 0.5)  # a reachable target, as the algorithm supplies
    converged = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=12)
    warm = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=1, eta_init=converged.eta)
    assert abs(float(warm.eta - converged.eta)) / float(converged.eta) < 0.01
    assert float((warm.lam - converged.lam).abs().max()) < 0.05


def test_float32_agrees_with_float64() -> None:
    """float32 is what training uses; float64 is what the gates above use. They must agree."""
    q_r, q_c = _problem(seed=12)
    d = torch.clamp_min(q_c.mean(dim=0) * 0.98, 0.5)
    hi = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3)
    lo = solve_per_state_dual(q_r.float(), q_c.float(), d.float(), EPS, LAM_MAX, sweeps=3)
    assert abs(float(hi.eta) - float(lo.eta)) / float(hi.eta) < 1e-3
    assert float((hi.lam - lo.lam.double()).abs().max()) < 1e-3


# --------------------------------------------------------------------------------------------
# 15-16. The design goal itself
# --------------------------------------------------------------------------------------------


class _NoHostSync:
    """Make any device->host transfer raise, so 'never leaves the GPU' is enforced, not asserted."""

    _NAMES = ("item", "cpu", "numpy", "tolist")

    def __enter__(self):
        self._saved = {n: getattr(torch.Tensor, n) for n in self._NAMES}

        def _boom(name):
            def f(*_a, **_k):
                raise AssertionError(f"host synchronisation via Tensor.{name}() inside the solve")

            return f

        for n in self._NAMES:
            setattr(torch.Tensor, n, _boom(n))
        return self

    def __exit__(self, *exc):
        for n, fn in self._saved.items():
            setattr(torch.Tensor, n, fn)
        return False


def test_solver_does_no_host_sync() -> None:
    """The solve must not synchronise: that is the whole reason it replaces the SciPy path.

    Without this test the requirement is decorative -- a stray ``.item()`` in the bisection loop
    would cost more than the SLSQP call it replaced and nothing would notice.
    """
    q_r, q_c = _problem(seed=13, hard=2, flat=2)
    d = _target(q_r.shape[1])
    with _NoHostSync():
        sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3)
        _ = reachable_qc_min(q_c, EPS)
    assert sol.lam.shape == (q_r.shape[1],)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu() -> None:
    q_r, q_c = _problem(seed=14)
    d = _target(q_r.shape[1])
    cpu = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3)
    gpu = solve_per_state_dual(q_r.cuda(), q_c.cuda(), d.cuda(), EPS, LAM_MAX, sweeps=3)
    assert abs(float(cpu.eta) - float(gpu.eta.cpu())) < 1e-8
    assert float((cpu.lam - gpu.lam.cpu()).abs().max()) < 1e-8


# --------------------------------------------------------------------------------------------
# Shared definitions must not drift from MPO's
# --------------------------------------------------------------------------------------------


def test_kl_and_ess_match_mpo_definitions() -> None:
    """This module duplicates MPO's two weight statistics to stay import-free; keep them identical."""
    from safe_rl.algorithms.mpo import effective_sample_size as mpo_ess
    from safe_rl.algorithms.mpo import nonparametric_kl_from_weights as mpo_kl

    q_r, q_c = _problem(seed=15)
    w = estep_weights(q_r, q_c, torch.tensor(0.7, dtype=torch.float64), torch.ones(q_r.shape[1], dtype=torch.float64))
    assert torch.allclose(per_state_kl(w), mpo_kl(w), atol=1e-14)
    assert torch.allclose(effective_sample_size(w), mpo_ess(w), atol=1e-14)


def test_per_state_eta_broadcasts() -> None:
    """The [B] eta path must stay a one-line experiment rather than a refactor (module docstring)."""
    q_r, q_c = _problem(seed=16)
    d = _target(q_r.shape[1])
    sol = solve_per_state_dual(q_r, q_c, d, EPS, LAM_MAX, sweeps=3, per_state_eta=True)
    assert sol.eta.shape == (q_r.shape[1],)

    # What the [B] eta buys is KL_b = eps at *every* state rather than on average. Assert that
    # against the eta block itself: `solve_per_state_dual` deliberately ends on a lambda block
    # (see its docstring), so the KL of the weights it returns has moved on from the eta solve.
    lam = solve_lambda(q_r, q_c, sol.eta, d, LAM_MAX, iters=40)
    eta_b = solve_eta(q_r, q_c, lam, EPS, iters=50, per_state=True)
    assert float((per_state_kl(estep_weights(q_r, q_c, eta_b, lam)) - EPS).abs().max()) < 1e-6

    # A shared eta can only hit eps on average, so its per-state KL genuinely disperses.
    eta_shared = solve_eta(q_r, q_c, lam, EPS, iters=50)
    kl_shared = per_state_kl(estep_weights(q_r, q_c, eta_shared, lam))
    assert abs(float(kl_shared.mean()) - EPS) < 1e-6
    assert float((kl_shared - EPS).abs().max()) > 1e-3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(sweeps=0), "sweeps must be"),
        (dict(lam_max=-1.0), "lam_max must be"),
    ],
)
def test_solve_rejects_invalid_arguments(kwargs, match) -> None:
    q_r, q_c = _problem(seed=17, k=4, b=3)
    call = dict(q_r=q_r, q_c=q_c, d=_target(3), eps=EPS, lam_max=LAM_MAX)
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        solve_per_state_dual(**call)


def test_solve_rejects_mismatched_shapes() -> None:
    q_r, q_c = _problem(seed=18, k=4, b=3)
    with pytest.raises(ValueError, match="same shape"):
        solve_per_state_dual(q_r, q_c[:, :2], _target(3), EPS, LAM_MAX)
    with pytest.raises(ValueError, match=r"d must be"):
        solve_per_state_dual(q_r, q_c, _target(2), EPS, LAM_MAX)
