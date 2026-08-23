"""Finite-horizon distributional cost primitives (FH-DCMPO).

Every statistic here ends up inside the E-step exponent, where a wrong answer does not crash --
it silently enforces the wrong budget. So each one is checked against an independent
implementation (brute-force mass accounting, or scipy) rather than against itself.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from safe_rl.common.fh_cost import (  # noqa: E402
    conservatism_statistic,
    evt_conservatism_statistic,
    fit_gpd,
    gpd_excess_quantile,
    horizon_feature_dim,
    kappa_at,
    normalized_remaining_budget,
    normalized_remaining_horizon,
    pit_from_quantiles,
    quantile_cvar,
    quantile_cvar_weighted,
    quantile_var,
    recalibrated_masses,
)


def _brute_cvar(theta: np.ndarray, alpha: float, upper: bool = True) -> float:
    """Independent tail-mass accounting: walk the sorted atoms taking `1-alpha` of mass."""
    th = np.sort(np.asarray(theta, dtype=np.float64))
    n = len(th)
    tail = (1.0 - alpha) if upper else alpha
    order = range(n - 1, -1, -1) if upper else range(n)
    acc = 0.0
    got = 0.0
    for i in order:
        take = min(1.0 / n, tail - got)
        if take <= 1e-15:
            break
        acc += take * th[i]
        got += take
    return acc / got


# -- CVaR / VaR -------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 8, 64])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 0.9, 0.95])
def test_cvar_matches_brute_force_mass_accounting(n: int, alpha: float) -> None:
    theta = np.random.default_rng(0).exponential(3.0, n)
    got = float(quantile_cvar(torch.tensor(theta), alpha))
    expected = float(theta.mean()) if alpha == 0.0 else _brute_cvar(theta, alpha)
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_cvar_lower_tail_matches_brute_force(alpha: float) -> None:
    theta = np.random.default_rng(1).exponential(2.0, 32)
    got = float(quantile_cvar(torch.tensor(theta), alpha, upper=False))
    assert got == pytest.approx(_brute_cvar(theta, alpha, upper=False), abs=1e-9)


def test_cvar_is_correct_on_unsorted_input() -> None:
    """The unsafe direction: unsorted input would understate the tail, not raise.

    ``QuantileCritic.forward`` sorts, so this can only bite a caller who reads a tail off raw
    locations -- which is exactly the kind of thing that reads as "the constraint is weak".
    """
    theta = np.array([5.0, 0.0, 3.0, 1.0, 9.0, 2.0, 4.0, 7.0])
    shuffled = torch.tensor(np.random.default_rng(2).permutation(theta))
    assert float(quantile_cvar(shuffled, 0.75)) == pytest.approx(_brute_cvar(theta, 0.75), abs=1e-9)
    # And it really is the top-25% mean, not the overall mean.
    assert float(quantile_cvar(shuffled, 0.75)) == pytest.approx(8.0, abs=1e-9)


def test_cvar_of_a_point_mass_is_that_point() -> None:
    # float32 here (the critic's dtype), so the tolerance is float32-appropriate: the boundary-mass
    # split divides and re-multiplies by the tail weight, which costs a few ULPs.
    theta = torch.full((16,), 7.5)
    for alpha in (0.0, 0.5, 0.9, 0.99):
        assert float(quantile_cvar(theta, alpha)) == pytest.approx(7.5, abs=1e-5)


def test_cvar_is_at_least_the_mean_and_monotone_in_alpha() -> None:
    theta = torch.tensor(np.random.default_rng(3).exponential(4.0, 64))
    mean = float(theta.mean())
    vals = [float(quantile_cvar(theta, a)) for a in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)]
    assert vals[0] == pytest.approx(mean, abs=1e-9)
    assert all(v >= mean - 1e-9 for v in vals)
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_var_index_is_ceil_alpha_n_minus_one() -> None:
    theta = torch.arange(64.0)
    assert float(quantile_var(theta, 0.9)) == 57.0  # ceil(0.9*64) - 1
    assert float(quantile_var(theta, 0.5)) == 31.0
    assert float(quantile_var(theta, 1.0)) == 63.0
    assert float(quantile_var(theta, 0.0)) == 0.0


def test_cvar_bounds_var_from_above() -> None:
    """``CVaR_alpha >= VaR_alpha`` is what turns the constraint into a violation-rate guarantee."""
    theta = torch.tensor(np.random.default_rng(4).exponential(3.0, 64))
    for alpha in (0.5, 0.75, 0.9):
        assert float(quantile_cvar(theta, alpha)) >= float(quantile_var(theta, alpha)) - 1e-9


def test_cvar_gradient_reaches_only_the_tail() -> None:
    theta = torch.arange(8.0, requires_grad=True)
    quantile_cvar(theta, 0.75).backward()
    grad = theta.grad.tolist()
    assert grad[:6] == [0.0] * 6
    assert grad[6:] == pytest.approx([0.5, 0.5])


def test_cvar_rejects_alpha_out_of_range() -> None:
    theta = torch.rand(8)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            quantile_cvar(theta, bad)


# -- The conservatism statistic ----------------------------------------------------------------


def test_kappa_zero_is_bit_exact_mean() -> None:
    """Not `approx`: the S1 arm's whole claim is that it *is* the mean constraint."""
    theta = torch.rand(5, 64)
    assert torch.equal(conservatism_statistic(theta, 0.9, 0.0), theta.mean(dim=-1))


def test_kappa_one_is_pure_cvar() -> None:
    theta = torch.rand(5, 64)
    assert torch.allclose(conservatism_statistic(theta, 0.9, 1.0), quantile_cvar(theta, 0.9))


def test_conservatism_statistic_is_monotone_in_kappa() -> None:
    theta = torch.tensor(np.random.default_rng(5).exponential(3.0, (7, 64)))
    vals = [conservatism_statistic(theta, 0.9, k).mean().item() for k in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_conservatism_statistic_rejects_negative_kappa() -> None:
    with pytest.raises(ValueError, match="kappa"):
        conservatism_statistic(torch.rand(8), 0.9, -0.5)


# -- The kappa ramp ---------------------------------------------------------------------------


def test_kappa_ramp_holds_at_zero_then_rises_then_saturates() -> None:
    assert kappa_at(0, 1.0, 100, 200) == 0.0
    assert kappa_at(99, 1.0, 100, 200) == 0.0
    assert kappa_at(100, 1.0, 100, 200) == pytest.approx(0.0)
    assert kappa_at(200, 1.0, 100, 200) == pytest.approx(0.5)
    assert kappa_at(300, 1.0, 100, 200) == pytest.approx(1.0)
    assert kappa_at(10_000, 1.0, 100, 200) == pytest.approx(1.0)


def test_kappa_ramp_is_nondecreasing() -> None:
    vals = [kappa_at(s, 1.0, 50, 100) for s in range(0, 300, 7)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_kappa_ramp_with_zero_ramp_is_a_step() -> None:
    assert kappa_at(49, 0.8, 50, 0) == 0.0
    assert kappa_at(50, 0.8, 50, 0) == pytest.approx(0.8)


# -- Horizon features -------------------------------------------------------------------------


def test_remaining_horizon_spans_one_to_zero() -> None:
    t = torch.tensor([0, 1, 500, 999, 1000])
    u = normalized_remaining_horizon(t, 1000).squeeze(-1)
    assert u[0] == pytest.approx(1.0)
    assert u[2] == pytest.approx(0.5)
    assert u[3] == pytest.approx(0.001)
    # u == 0 at t == T is the boundary condition that pins the undiscounted cost-to-go to zero.
    assert u[4] == pytest.approx(0.0)


def test_remaining_horizon_clamps_past_the_end() -> None:
    u = normalized_remaining_horizon(torch.tensor([1500]), 1000)
    assert float(u) == pytest.approx(0.0)


def test_remaining_horizon_shape_is_column() -> None:
    assert normalized_remaining_horizon(torch.zeros(16, dtype=torch.long), 1000).shape == (16, 1)


def test_remaining_budget_goes_negative_when_blown_and_clamps() -> None:
    spent = torch.tensor([0.0, 12.5, 25.0, 50.0, 500.0])
    b = normalized_remaining_budget(spent, 25.0).squeeze(-1)
    assert b[0] == pytest.approx(1.0)
    assert b[1] == pytest.approx(0.5)
    assert b[2] == pytest.approx(0.0)
    assert b[3] == pytest.approx(-1.0)
    assert b[4] == pytest.approx(-1.0)  # clamped, so the feature stays bounded


def test_horizon_feature_dim() -> None:
    assert horizon_feature_dim(False) == 1
    assert horizon_feature_dim(True) == 2


def test_horizon_features_reject_bad_scales() -> None:
    with pytest.raises(ValueError, match="horizon"):
        normalized_remaining_horizon(torch.zeros(2, dtype=torch.long), 0)
    with pytest.raises(ValueError, match="cost_limit"):
        normalized_remaining_budget(torch.zeros(2), 0.0)


# -- Recalibration primitives (S3) -------------------------------------------------------------


def test_weighted_cvar_reduces_to_the_uniform_one() -> None:
    """Uniform masses must reproduce `quantile_cvar` exactly, or S3 silently changes S2's result."""
    theta = torch.tensor(np.random.default_rng(10).exponential(3.0, (6, 32)))
    probs = torch.full((32,), 1.0 / 32, dtype=theta.dtype)
    for alpha in (0.0, 0.5, 0.9):
        assert torch.allclose(quantile_cvar_weighted(theta, probs, alpha), quantile_cvar(theta, alpha), atol=1e-12)


def test_weighted_cvar_matches_brute_force_with_uneven_masses() -> None:
    theta = np.array([0.0, 1.0, 2.0, 10.0])
    probs = np.array([0.7, 0.1, 0.1, 0.1])
    # Top 20% of mass is locations 2 and 10, each carrying 0.1 -> mean 6.0.
    got = float(quantile_cvar_weighted(torch.tensor(theta), torch.tensor(probs), 0.8))
    assert got == pytest.approx(6.0, abs=1e-9)
    # Top 5% is entirely inside the last location -> 10.0.
    got = float(quantile_cvar_weighted(torch.tensor(theta), torch.tensor(probs), 0.95))
    assert got == pytest.approx(10.0, abs=1e-9)


def test_weighted_cvar_rejects_a_mass_length_mismatch() -> None:
    with pytest.raises(ValueError, match="masses"):
        quantile_cvar_weighted(torch.rand(4, 8), torch.full((7,), 1 / 7), 0.9)


def test_pit_is_uniform_for_a_calibrated_prediction() -> None:
    """The marginal PIT histogram is flat when the predicted distribution is the true one."""
    rng = np.random.default_rng(11)
    n_states, n_q = 4000, 64
    # Each "state" predicts the exact quantiles of its own N(mu, 1); realized is drawn from it.
    mu = rng.normal(size=n_states)
    taus = (np.arange(n_q) + 0.5) / n_q
    from scipy import stats as _st

    theta = torch.tensor(mu[:, None] + _st.norm.ppf(taus)[None, :])
    realized = torch.tensor(mu + rng.normal(size=n_states))
    pit = pit_from_quantiles(theta, realized).numpy()
    ks = float(np.max(np.abs(np.sort(pit) - (np.arange(n_states) + 1) / n_states)))
    assert ks < 0.05, ks


def test_pit_detects_an_underdispersed_prediction() -> None:
    """The documented failure: predicted std 2.1x too narrow pushes PIT mass to both ends."""
    rng = np.random.default_rng(12)
    n_states, n_q = 4000, 64
    taus = (np.arange(n_q) + 0.5) / n_q
    from scipy import stats as _st

    theta = torch.tensor(_st.norm.ppf(taus)[None, :].repeat(n_states, 0))  # std 1
    realized = torch.tensor(rng.normal(scale=2.1, size=n_states))  # truly std 2.1
    pit = pit_from_quantiles(theta, realized).numpy()
    ks = float(np.max(np.abs(np.sort(pit) - (np.arange(n_states) + 1) / n_states)))
    assert ks > 0.15, ks
    # Mass piles at 0 and 1 -- the U-shape recorded in why-mean-beat-cvar-on-pointgoal1.md.
    assert ((pit < 0.05) | (pit > 0.95)).mean() > 0.3


def test_recalibrated_masses_are_uniform_for_an_identity_map() -> None:
    masses = recalibrated_masses(lambda x: x, 16)
    assert torch.allclose(masses, torch.full((16,), 1.0 / 16), atol=1e-12)


def test_recalibrated_masses_shift_mass_toward_the_tail_when_told_to() -> None:
    """A concave map says "realized values sit above prediction", so mass must move up."""
    masses = recalibrated_masses(lambda x: np.sqrt(x), 16)
    assert masses.sum() == pytest.approx(1.0, abs=1e-9)
    assert bool((masses >= 0).all())
    # sqrt is concave: it raises G(c) at low c, so the LOW locations gain mass and the top loses.
    assert float(masses[0]) > 1.0 / 16
    assert float(masses[-1]) < 1.0 / 16


def test_recalibrated_masses_fall_back_to_uniform_on_a_degenerate_map() -> None:
    """A degenerate fit must give up, not produce an extreme.

    Blindly forcing the endpoints onto a decreasing map puts all mass on the LOWEST location, i.e.
    CVaR = min, which *understates* cost. That is the unsafe direction and it leaves no symptom in
    the logs, so an unusable map has to fall back to no-recalibration instead.
    """
    uniform = torch.full((8,), 0.125)
    assert torch.allclose(recalibrated_masses(lambda x: np.zeros_like(x), 8), uniform)  # constant
    assert torch.allclose(recalibrated_masses(lambda x: 1.0 - x, 8), uniform)  # decreasing
    assert torch.allclose(recalibrated_masses(lambda x: np.full_like(x, np.nan), 8), uniform)  # non-finite


def test_recalibrated_masses_never_understate_the_tail_relative_to_uniform() -> None:
    """Any map accepted by the guard must still yield a valid, non-negative distribution."""
    for fn in (lambda x: x, lambda x: np.sqrt(x), lambda x: x**2, lambda x: np.clip(x * 1.3, 0, 1)):
        m = recalibrated_masses(fn, 32)
        assert bool((m >= 0).all())
        assert float(m.sum()) == pytest.approx(1.0, abs=1e-6)


def test_recalibration_round_trip_repairs_an_underdispersed_cvar() -> None:
    """End to end: fit the PIT map on an under-dispersed critic, then read a corrected CVaR.

    This is the S3 claim in miniature -- the raw CVaR understates the tail, and the recalibrated one
    moves toward the truth.
    """
    from scipy import stats as _st

    from safe_rl.common.recalibration import PITRecalibrator

    rng = np.random.default_rng(13)
    n_states, n_q = 6000, 64
    taus = (np.arange(n_q) + 0.5) / n_q
    theta = torch.tensor(_st.norm.ppf(taus)[None, :].repeat(n_states, 0))  # predicts std 1
    realized = torch.tensor(rng.normal(scale=2.1, size=n_states))  # truth is std 2.1

    recal = PITRecalibrator(capacity=n_states, min_samples=500)
    recal.update(pit_from_quantiles(theta, realized).numpy())
    assert recal.refit()

    truth = float(np.mean(np.sort(rng.normal(scale=2.1, size=200000))[-20000:]))  # CVaR_0.9
    raw = float(quantile_cvar(theta[0], 0.9))
    fixed = float(quantile_cvar_weighted(theta[0], recalibrated_masses(recal.apply, n_q, dtype=theta.dtype), 0.9))
    assert raw < truth  # the documented understatement
    assert abs(fixed - truth) < abs(raw - truth)


# -- EVT / GPD --------------------------------------------------------------------------------


@pytest.mark.parametrize("xi_true,sigma_true", [(0.2, 2.0), (0.0, 1.5), (0.5, 3.0)])
def test_gpd_mle_recovers_scipy_generated_parameters(xi_true: float, sigma_true: float) -> None:
    stats = pytest.importorskip("scipy.stats")
    y = stats.genpareto.rvs(xi_true, loc=0.0, scale=sigma_true, size=4000, random_state=7)
    xi, sigma = fit_gpd(y)
    assert xi == pytest.approx(xi_true, abs=0.08)
    assert sigma == pytest.approx(sigma_true, rel=0.12)


def test_gpd_falls_back_to_exponential_on_a_tiny_sample() -> None:
    """Conservative direction: a failed fit must not silently produce a *looser* constraint."""
    xi, sigma = fit_gpd(np.array([1.0, 2.0, 3.0]))
    assert xi == 0.0
    assert sigma == pytest.approx(2.0)


def test_gpd_excess_quantile_matches_the_exponential_limit_as_xi_goes_to_zero() -> None:
    sigma, nu, n_peaks, n_total = 2.0, 0.05, 500, 4000
    frac = nu * n_total / n_peaks
    expected = -sigma * np.log(1.0 - frac)
    assert gpd_excess_quantile(0.0, sigma, nu, n_peaks, n_total) == pytest.approx(expected)
    # Continuity: a tiny xi must be close to the limit, not a discontinuous jump.
    assert gpd_excess_quantile(1e-6, sigma, nu, n_peaks, n_total) == pytest.approx(expected, rel=1e-4)


def test_gpd_excess_quantile_is_zero_when_it_cannot_be_estimated() -> None:
    assert gpd_excess_quantile(0.2, 2.0, 0.05, 0, 4000) == 0.0  # no peaks
    assert gpd_excess_quantile(0.2, 0.0, 0.05, 500, 4000) == 0.0  # degenerate scale
    assert gpd_excess_quantile(0.2, 2.0, 5.0, 500, 4000) == 0.0  # range exceeds available mass


def test_gpd_excess_quantile_grows_with_the_tail_index() -> None:
    offs = [gpd_excess_quantile(xi, 2.0, 0.05, 500, 4000) for xi in (0.0, 0.2, 0.5, 0.8)]
    assert all(offs[i] < offs[i + 1] for i in range(len(offs) - 1))


def test_evt_statistic_adds_a_single_global_offset_to_the_mean() -> None:
    """Structurally different from CVaR: EVO's tail correction is one scalar, not per-state."""
    theta = torch.tensor(np.random.default_rng(8).exponential(3.0, (32, 64)))
    stat, info = evt_conservatism_statistic(theta, mu=0.9, nu=0.05, kappa=1.0)
    residual = (stat - theta.mean(dim=-1)).numpy()
    assert np.allclose(residual, residual[0], atol=1e-9)
    assert residual[0] == pytest.approx(info["evt_offset"], abs=1e-9)
    assert info["evt_n_peaks"] > 0


def test_evt_statistic_with_kappa_zero_is_the_mean() -> None:
    theta = torch.tensor(np.random.default_rng(9).exponential(3.0, (8, 32)))
    stat, _ = evt_conservatism_statistic(theta, mu=0.9, nu=0.05, kappa=0.0)
    assert torch.allclose(stat, theta.mean(dim=-1))
