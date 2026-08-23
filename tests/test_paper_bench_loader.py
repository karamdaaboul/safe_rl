"""Pins the paper-results loader against known values.

If someone repoints `PAPER_RESULTS` at a different tree, or the CSV schema shifts, these
fail loudly rather than producing a comparison against the wrong numbers.

The expected values were read independently from the CSVs before the loader existed.
"""

from __future__ import annotations

import pytest

from safe_rl.analysis import paper_bench as pb

pytestmark = pytest.mark.skipif(
    not pb.PAPER_RESULTS.exists(), reason="reference results tree not present"
)

# (suite, task, mean, sd, n) — final row only.
KNOWN = [
    ("maniskill", "PickSingleYCB-v1", 0.802, 0.016, 14),
    ("maniskill", "PullCube-v1", 1.000, 0.000, 12),
    ("maniskill", "PokeCube-v1", 0.733, 0.404, 13),
    ("dmc", "CheetahRun", 924.10, 51.70, 40),
    ("dmc", "HopperHop", 178.66, 154.81, 40),
    ("dmc", "AcrobotSwingupSparse", 11.46, 13.00, 40),
]


@pytest.mark.parametrize("suite,task,mean,sd,n", KNOWN)
def test_paper_final_matches_known_values(suite, task, mean, sd, n) -> None:
    got = pb.paper_final(suite, task)
    assert got["n"] == n
    assert got["mean"] == pytest.approx(mean, abs=0.01)
    assert got["sd"] == pytest.approx(sd, abs=0.01)
    assert got["steps"] == pytest.approx(5e7)


def test_curve_has_the_expected_grid() -> None:
    steps, trials = pb.load_paper_curve("dmc", "CheetahRun")
    assert len(steps) == 20, "their curves are resampled onto a 20-point grid"
    assert all(len(t) == 20 for t in trials)
    assert steps[0] == 0.0 and steps[-1] == pytest.approx(5e7)


def test_row_zero_is_an_artifact_not_a_starting_score() -> None:
    """Documents WHY only the last row is quotable.

    CheetahRun reads ~627 at step 0, which no untrained policy achieves — the curves were
    resampled onto a fixed grid and extrapolate backwards. Anything that starts quoting
    intermediate rows as scores should trip over this test first.
    """
    _, trials = pb.load_paper_curve("dmc", "CheetahRun")
    row_zero_mean = sum(t[0] for t in trials) / len(trials)
    assert row_zero_mean > 400, "if this ever drops, re-examine the quote-last-row-only rule"


def test_missing_task_raises_rather_than_returning_empty() -> None:
    with pytest.raises(FileNotFoundError):
        pb.paper_final("maniskill", "PickCube-v1")  # deliberately absent from the paper


class TestVerdictRule:
    def test_within_one_sd_is_a_match(self) -> None:
        assert pb.classify(920.0, 924.1, 51.7) == "match"

    def test_between_one_and_two_sd_is_weak(self) -> None:
        assert pb.classify(1000.0, 924.1, 51.7) == "weak"

    def test_beyond_two_sd_is_a_miss(self) -> None:
        assert pb.classify(500.0, 924.1, 51.7) == "miss"

    def test_saturated_task_uses_a_ratio_rule(self) -> None:
        """Their sd is exactly 0 on PullCube/LiftPegUpright; a z-score is undefined."""
        assert pb.classify(1.0, 1.0, 0.0) == "match"
        assert pb.classify(0.97, 1.0, 0.0) == "weak"
        assert pb.classify(0.5, 1.0, 0.0) == "miss"


def test_incomplete_runs_are_flagged_not_scored(tmp_path) -> None:
    """A crashed run leaves a readable event file; scoring it against a 50M number
    would look like a legitimate (terrible) result."""
    rec = pb.our_final(tmp_path, "dmc")
    assert rec["status"] == "no_eval_data"
    assert rec["value"] is None


def test_retried_cell_reads_the_furthest_attempt_not_the_first(tmp_path) -> None:
    """A cell can hold several event dirs when the driver retries an OOM-killed run.

    Reading the first one found reported a *finished* PullCube-v1 cell as incomplete,
    because the killed attempt (stopping at 39.98M of 49.94M steps) sorted first and is
    still perfectly readable. That silently drops real results from the comparison.
    """
    from torch.utils.tensorboard import SummaryWriter

    tag = pb.OUR_TAG["maniskill"]
    # Killed attempt: stops short of the budget.
    with SummaryWriter(str(tmp_path / "aaa_killed")) as w:
        for step in range(0, 40_000_000, 2_000_000):
            w.add_scalar(tag, 0.5, step)
    # Successful retry: reaches the full budget. Sorts AFTER the killed one by name.
    with SummaryWriter(str(tmp_path / "zzz_finished")) as w:
        for step in range(0, pb.BUDGET_ENV_STEPS + 1, 2_496_921):
            w.add_scalar(tag, 0.97, step)

    steps, values = pb.load_our_curve(tmp_path, "maniskill")
    assert steps[-1] >= pb.BUDGET_ENV_STEPS * 0.98, "must pick the attempt that got furthest"
    assert values[-1] == pytest.approx(0.97)
    assert pb.our_final(tmp_path, "maniskill")["status"] == "ok"


def test_markdown_table_reports_missing_cells_explicitly() -> None:
    rows = [
        {
            "task": "PullCube-v1", "suite": "maniskill", "theirs_mean": 1.0, "theirs_sd": 0.0,
            "theirs_n": 12, "ours": [], "ours_n": 0, "skipped": [("/x/s1", "no_eval_data")],
            "verdict": "no_data",
        }
    ]
    table = pb.markdown_table(rows)
    assert "no_data" in table
    assert "Excluded runs" in table and "/x/s1" in table
