"""Item 4: tail-aware evaluation statistics.

Reporting only the mean episodic cost cannot detect a CVaR win -- two policies with the same
mean can differ completely in their tail, and the tail is what a risk constraint targets.
Measured on the real arms: mean cost 21.4 while 23-31% of individual episodes still exceeded
the limit, so the mean alone was hiding the thing being optimised.

Statistics are checked against hand-computed values on a known list, not against another
implementation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from safe_rl.common.tail_eval import (  # noqa: E402
    EPISODE_CSV_FIELDS,
    SUMMARY_FIELDS,
    episode_cost_stats,
    write_episode_csv,
)


def test_statistics_match_hand_computed_values() -> None:
    # 10 costs, deliberately skewed: mean 25.0, median 12.5, and a heavy upper tail.
    costs = [0.0, 5.0, 10.0, 10.0, 12.0, 13.0, 20.0, 30.0, 50.0, 100.0]
    s = episode_cost_stats(costs, cost_limit=25.0)

    assert s["cost_mean"] == pytest.approx(25.0)
    assert s["cost_median"] == pytest.approx(12.5)
    assert s["cost_min"] == pytest.approx(0.0)
    assert s["cost_max"] == pytest.approx(100.0)
    assert s["n_episodes"] == 10
    # 30, 50 and 100 strictly exceed 25 -> 3 of 10
    assert s["budget_exceedance_rate"] == pytest.approx(0.3)
    # CVaR_0.5 = mean of the worst 50% = mean(13, 20, 30, 50, 100) = 42.6
    assert s["cost_cvar_0.5"] == pytest.approx(42.6)
    # CVaR_0.9 = mean of the worst 10% = 100.0
    assert s["cost_cvar_0.9"] == pytest.approx(100.0)


def test_percentiles_use_linear_interpolation_consistently() -> None:
    costs = list(range(101))          # 0..100
    s = episode_cost_stats(costs, cost_limit=25.0)
    assert s["cost_p90"] == pytest.approx(90.0)
    assert s["cost_p95"] == pytest.approx(95.0)
    assert s["cost_median"] == pytest.approx(50.0)


def test_cvar_is_monotone_in_alpha_and_at_least_the_mean() -> None:
    rng = np.random.default_rng(0)
    costs = rng.gamma(2.0, 15.0, size=500).tolist()
    s = episode_cost_stats(costs, cost_limit=25.0)
    assert s["cost_mean"] <= s["cost_cvar_0.5"] <= s["cost_cvar_0.9"]


def test_reward_statistics_included() -> None:
    s = episode_cost_stats([1.0, 2.0, 3.0], cost_limit=25.0, rewards=[10.0, 20.0, 30.0])
    assert s["reward_mean"] == pytest.approx(20.0)
    assert s["reward_std"] == pytest.approx(np.std([10.0, 20.0, 30.0], ddof=1))
    assert s["reward_sem"] == pytest.approx(s["reward_std"] / np.sqrt(3))


def test_all_zero_and_single_episode_edge_cases() -> None:
    s = episode_cost_stats([0.0] * 5, cost_limit=25.0)
    assert s["cost_mean"] == 0.0 and s["budget_exceedance_rate"] == 0.0
    assert s["cost_cvar_0.9"] == 0.0
    one = episode_cost_stats([7.0], cost_limit=25.0, rewards=[3.0])
    assert one["n_episodes"] == 1
    assert one["cost_std"] == 0.0 and one["reward_std"] == 0.0


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError):
        episode_cost_stats([], cost_limit=25.0)


def test_summary_has_the_declared_schema() -> None:
    s = episode_cost_stats([1.0, 2.0], cost_limit=25.0, rewards=[1.0, 2.0])
    assert set(SUMMARY_FIELDS).issubset(s.keys())


def test_csv_schema_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    write_episode_csv(path, costs=[1.0, 30.0], rewards=[5.0, 6.0], lengths=[1000, 1000])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == list(EPISODE_CSV_FIELDS)
    assert len(rows) == 2
    assert float(rows[1]["cost"]) == pytest.approx(30.0)
    assert int(rows[0]["episode"]) == 0


def test_csv_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_episode_csv(tmp_path / "x.csv", costs=[1.0, 2.0], rewards=[1.0], lengths=[1, 2])


def test_stats_do_not_mutate_input() -> None:
    costs = [5.0, 1.0, 3.0]
    before = list(costs)
    episode_cost_stats(costs, cost_limit=25.0)
    assert costs == before, "sorting must not happen in place"


def test_distinct_episode_count_is_reported() -> None:
    """Guards against counting duplicated parallel-env episodes as independent samples."""
    s = episode_cost_stats([22.0] * 8 + [0.0] * 8, cost_limit=25.0)
    assert s["n_episodes"] == 16
    assert s["n_distinct"] == 2


def test_summary_warns_when_episodes_are_duplicated() -> None:
    from safe_rl.common.tail_eval import format_summary

    dup = episode_cost_stats([22.0] * 12 + [0.0] * 12, cost_limit=25.0, rewards=[1.0] * 24)
    assert "WARNING" in format_summary(dup)
    varied = episode_cost_stats(list(range(1, 25)), cost_limit=25.0, rewards=[1.0] * 24)
    assert "WARNING" not in format_summary(varied)
