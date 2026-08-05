"""Item 2: quantile recalibration of the categorical cost distribution.

Pure synthetic data -- no RL. The property under test is distributional calibration: after
recalibration, the predicted CDF evaluated at realized outcomes (the PIT) should be uniform,
which is what makes a CVaR read off that distribution meaningful.

Motivation from the measured critic: predicted std 3.21 vs realized 5.55, PIT KS 0.381,
VaR_0.9 covering only 0.858 of realized returns instead of 0.90 -- so raw CVaR understates
tail risk by ~25-30%.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from safe_rl.common.recalibration import (  # noqa: E402
    PITRecalibrator,
    recalibrate_distribution,
)

ATOMS = np.linspace(0.0, 50.0, 101)


def _gaussian_probs(atoms: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    p = np.exp(-0.5 * ((atoms - mu) / sigma) ** 2)
    return p / p.sum()


def _cdf_at(atoms: np.ndarray, probs: np.ndarray, x: float) -> float:
    """Interpolated predicted CDF at x (the PIT transform for a single sample)."""
    cdf = np.cumsum(probs)
    return float(np.interp(x, atoms, cdf))


def _quantile(atoms: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, alpha))
    return float(atoms[min(idx, len(atoms) - 1)])


def _cvar(atoms: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    tail = 1.0 - alpha
    beyond = np.clip(1.0 - np.cumsum(probs), 0.0, None)
    w = np.minimum(probs, np.clip(tail - beyond, 0.0, None))
    return float((w * atoms).sum() / max(w.sum(), 1e-12))


# --- (a) an under-dispersed forecast gets its coverage repaired ------------------


def test_recalibration_repairs_coverage_of_under_dispersed_forecast() -> None:
    rng = np.random.default_rng(0)
    mu, true_sigma, pred_sigma = 20.0, 8.0, 4.0     # predicted 2x too narrow, as measured
    probs = _gaussian_probs(ATOMS, mu, pred_sigma)

    truth = rng.normal(mu, true_sigma, size=4000)
    pits = np.array([_cdf_at(ATOMS, probs, x) for x in truth])

    recal = PITRecalibrator(capacity=8000, min_samples=200)
    recal.update(pits)
    recal.refit()
    assert recal.is_fitted

    new_probs = recalibrate_distribution(ATOMS, probs, recal)
    held = rng.normal(mu, true_sigma, size=4000)
    coverage = float(np.mean(held <= _quantile(ATOMS, new_probs, 0.9)))
    assert 0.88 <= coverage <= 0.92, f"coverage after recalibration = {coverage}"

    raw_coverage = float(np.mean(held <= _quantile(ATOMS, probs, 0.9)))
    assert raw_coverage < 0.88, "the raw forecast should have been mis-covered to begin with"


def test_recalibration_raises_cvar_of_under_dispersed_forecast() -> None:
    """Under-dispersion understates tail risk; recalibration must push CVaR UP."""
    rng = np.random.default_rng(1)
    probs = _gaussian_probs(ATOMS, 20.0, 4.0)
    truth = rng.normal(20.0, 8.0, size=4000)
    recal = PITRecalibrator(capacity=8000, min_samples=200)
    recal.update(np.array([_cdf_at(ATOMS, probs, x) for x in truth]))
    recal.refit()
    assert _cvar(ATOMS, recalibrate_distribution(ATOMS, probs, recal), 0.9) > _cvar(ATOMS, probs, 0.9)


# --- (b) an already-calibrated forecast is left alone ---------------------------


def test_uniform_pit_gives_near_identity_map() -> None:
    rng = np.random.default_rng(2)
    recal = PITRecalibrator(capacity=8000, min_samples=200)
    recal.update(rng.uniform(0.0, 1.0, size=4000))
    recal.refit()
    grid = np.linspace(0.02, 0.98, 25)
    assert np.max(np.abs(recal.apply(grid) - grid)) < 0.05


def test_calibrated_forecast_cvar_within_one_percent() -> None:
    rng = np.random.default_rng(3)
    mu, sigma = 20.0, 8.0
    probs = _gaussian_probs(ATOMS, mu, sigma)          # correctly dispersed
    truth = rng.normal(mu, sigma, size=6000)
    recal = PITRecalibrator(capacity=8000, min_samples=200)
    recal.update(np.array([_cdf_at(ATOMS, probs, x) for x in truth]))
    recal.refit()
    raw = _cvar(ATOMS, probs, 0.9)
    new = _cvar(ATOMS, recalibrate_distribution(ATOMS, probs, recal), 0.9)
    assert abs(new - raw) / raw < 0.01, f"raw {raw} vs recalibrated {new}"


# --- (c) output is always a valid distribution ----------------------------------


def test_output_is_a_valid_distribution_for_random_inputs() -> None:
    rng = np.random.default_rng(4)
    recal = PITRecalibrator(capacity=4000, min_samples=100)
    recal.update(rng.beta(2.0, 5.0, size=2000))         # deliberately skewed map
    recal.refit()
    for _ in range(25):
        p = rng.dirichlet(np.ones(len(ATOMS)) * rng.uniform(0.05, 5.0))
        out = recalibrate_distribution(ATOMS, p, recal)
        assert np.all(out >= 0.0)
        assert out.sum() == pytest.approx(1.0, abs=1e-9)
        assert out.shape == p.shape


def test_point_mass_edge_case() -> None:
    rng = np.random.default_rng(5)
    recal = PITRecalibrator(capacity=2000, min_samples=100)
    recal.update(rng.uniform(size=1000))
    recal.refit()
    for idx in (0, 50, len(ATOMS) - 1):
        p = np.zeros(len(ATOMS))
        p[idx] = 1.0
        out = recalibrate_distribution(ATOMS, p, recal)
        assert np.all(out >= 0.0)
        assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_unfitted_recalibrator_is_the_identity() -> None:
    """Before enough samples arrive, recalibration must be a no-op, not a crash."""
    recal = PITRecalibrator(capacity=1000, min_samples=500)
    recal.update(np.array([0.3, 0.7]))
    assert not recal.is_fitted
    p = _gaussian_probs(ATOMS, 20.0, 5.0)
    assert np.allclose(recalibrate_distribution(ATOMS, p, recal), p)
    assert np.allclose(recal.apply(np.linspace(0, 1, 11)), np.linspace(0, 1, 11))


# --- buffer behaviour ------------------------------------------------------------


def test_buffer_is_fifo_and_capped() -> None:
    recal = PITRecalibrator(capacity=100, min_samples=10)
    recal.update(np.zeros(80))
    recal.update(np.ones(80))
    # 160 pushed into a 100-slot FIFO -> the last 100 survive: 20 zeros + 80 ones.
    assert len(recal) == 100
    assert float(np.mean(recal.values)) == pytest.approx(0.8, abs=1e-9)
    recal.update(np.ones(100))          # now evict everything older
    assert len(recal) == 100
    assert float(np.mean(recal.values)) == pytest.approx(1.0, abs=1e-9)


def test_refit_is_deterministic() -> None:
    rng = np.random.default_rng(6)
    pits = rng.uniform(size=1000)
    a, b = PITRecalibrator(min_samples=10), PITRecalibrator(min_samples=10)
    a.update(pits); a.refit()
    b.update(pits); b.refit()
    grid = np.linspace(0, 1, 51)
    assert np.array_equal(a.apply(grid), b.apply(grid))


def test_apply_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(7)
    recal = PITRecalibrator(min_samples=10)
    recal.update(rng.beta(0.5, 3.0, size=2000))
    recal.refit()
    grid = np.linspace(0.0, 1.0, 201)
    out = recal.apply(grid)
    assert np.all(np.diff(out) >= -1e-12), "recalibration map must be non-decreasing"
    assert out.min() >= 0.0 and out.max() <= 1.0
