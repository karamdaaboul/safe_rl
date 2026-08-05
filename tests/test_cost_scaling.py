"""Item 1: episodic-to-discounted cost scaling, and threshold construction.

These tests are written before the module. The bug class they exist to prevent: a threshold
built with the analytic uniform-cost scale (0.1) while the true ratio is 0.0764, which silently
grants a 32.7 episodic budget when 25 was requested.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from safe_rl.common.cost_scaling import (  # noqa: E402
    analytic_qc_scale,
    make_thresholds,
    measured_qc_scale,
)

GAMMA = 0.99
HORIZON = 1000
MEASURED = 0.0764          # measured G_c(s0)/J_c on SafetyPointGoal1
COST_LIM = 25.0
# CVaR_alpha / mean ratios measured on a reference policy (512 states x 64 actions):
# mean 4.45, CVaR_0.5 6.89, CVaR_0.9 11.19
RATIOS = {0.5: 6.89 / 4.45, 0.9: 11.19 / 4.45}


# --- analytic / measured scale -------------------------------------------------


def test_analytic_scale_matches_closed_form() -> None:
    expected = (1 - GAMMA**HORIZON) / (1 - GAMMA) / HORIZON
    assert analytic_qc_scale(GAMMA, HORIZON) == pytest.approx(expected, rel=1e-12)
    assert analytic_qc_scale(GAMMA, HORIZON) == pytest.approx(0.1, abs=1e-4)


def test_measured_scale_recovers_analytic_on_uniform_cost() -> None:
    """A rollout with cost 1 at every step must reproduce the uniform-cost formula."""
    episodes = [np.ones(HORIZON) for _ in range(5)]
    got = measured_qc_scale(episodes, GAMMA)
    assert got == pytest.approx(analytic_qc_scale(GAMMA, HORIZON), abs=1e-6)


def test_measured_scale_detects_late_weighted_cost() -> None:
    """Cost concentrated late is discounted away -> measured scale BELOW analytic."""
    late = np.zeros(HORIZON)
    late[HORIZON // 2:] = 1.0
    got = measured_qc_scale([late] * 3, GAMMA)
    assert got < analytic_qc_scale(GAMMA, HORIZON)


def test_measured_scale_detects_early_weighted_cost() -> None:
    early = np.zeros(HORIZON)
    early[: HORIZON // 2] = 1.0
    got = measured_qc_scale([early] * 3, GAMMA)
    assert got > analytic_qc_scale(GAMMA, HORIZON)


def test_measured_scale_tolerates_zero_cost_episodes() -> None:
    """~10% of real episodes cost nothing; a mean-of-per-episode-ratios would divide by zero."""
    episodes = [np.ones(HORIZON), np.zeros(HORIZON), np.ones(HORIZON)]
    got = measured_qc_scale(episodes, GAMMA)
    assert np.isfinite(got)
    assert got == pytest.approx(analytic_qc_scale(GAMMA, HORIZON), abs=1e-6)


def test_measured_scale_rejects_empty_or_all_zero() -> None:
    with pytest.raises(ValueError):
        measured_qc_scale([], GAMMA)
    with pytest.raises(ValueError):
        measured_qc_scale([np.zeros(10)], GAMMA)


def test_measured_scale_accepts_2d_array() -> None:
    """[T, num_envs] layout, the shape rollouts naturally come in."""
    arr = np.ones((HORIZON, 4))
    assert measured_qc_scale(arr, GAMMA) == pytest.approx(analytic_qc_scale(GAMMA, HORIZON), abs=1e-6)


# --- threshold construction ----------------------------------------------------


def test_mean_threshold_uses_the_given_scale() -> None:
    t = make_thresholds(COST_LIM, MEASURED, mode="matched", ratio_stats=RATIOS)
    assert t["mean"] == pytest.approx(1.91, abs=1e-3)


def test_matched_mode_scales_cvar_by_measured_ratio() -> None:
    t = make_thresholds(COST_LIM, MEASURED, mode="matched", ratio_stats=RATIOS)
    assert t["cvar_0.5"] == pytest.approx(2.96, abs=0.02)
    assert t["cvar_0.9"] == pytest.approx(4.80, abs=0.02)


def test_wcsac_mode_gives_every_alpha_the_mean_threshold() -> None:
    """WCSAC semantics: same budget applied to the tail statistic -> intentionally stricter."""
    t = make_thresholds(COST_LIM, MEASURED, mode="wcsac", ratio_stats=RATIOS)
    assert t["mean"] == pytest.approx(1.91, abs=1e-3)
    assert t["cvar_0.5"] == pytest.approx(1.91, abs=1e-3)
    assert t["cvar_0.9"] == pytest.approx(1.91, abs=1e-3)


def test_effective_budget_round_trips() -> None:
    """The hard-fail invariant: threshold / (scale * ratio) must return the episodic budget.

    This is the check that would have caught the confound.
    """
    for mode in ("wcsac", "matched"):
        t = make_thresholds(COST_LIM, MEASURED, mode=mode, ratio_stats=RATIOS)
        assert t["effective_budget_mean"] == pytest.approx(COST_LIM, rel=1e-9)
        for a, r in RATIOS.items():
            eff = t[f"effective_budget_cvar_{a}"]
            if mode == "matched":
                assert eff == pytest.approx(COST_LIM, rel=1e-9)
            else:
                # wcsac deliberately targets a tighter episodic budget
                assert eff == pytest.approx(COST_LIM / r, rel=1e-9)
                assert eff < COST_LIM


def test_analytic_scale_would_loosen_the_budget() -> None:
    """Documents the original bug numerically: 0.1 instead of 0.0764 -> budget 32.7, not 25."""
    wrong = make_thresholds(COST_LIM, analytic_qc_scale(GAMMA, HORIZON), mode="matched",
                            ratio_stats=RATIOS)
    # A threshold built on the analytic scale, interpreted with the TRUE scale, over-grants:
    implied = wrong["mean"] / MEASURED
    assert implied == pytest.approx(32.7, abs=0.2)


def test_make_thresholds_validates_inputs() -> None:
    with pytest.raises(ValueError, match="mode"):
        make_thresholds(COST_LIM, MEASURED, mode="tail", ratio_stats=RATIOS)
    with pytest.raises(ValueError, match="qc_scale"):
        make_thresholds(COST_LIM, 0.0, mode="matched", ratio_stats=RATIOS)
    with pytest.raises(ValueError, match="ratio"):
        make_thresholds(COST_LIM, MEASURED, mode="matched", ratio_stats={0.9: 0.0})
    with pytest.raises(ValueError, match="ratio_stats"):
        make_thresholds(COST_LIM, MEASURED, mode="matched", ratio_stats=None)


def test_wcsac_mode_allows_missing_ratio_stats() -> None:
    """wcsac needs no ratios -- the threshold is the same for every alpha."""
    t = make_thresholds(COST_LIM, MEASURED, mode="wcsac", ratio_stats=None)
    assert t["mean"] == pytest.approx(1.91, abs=1e-3)
