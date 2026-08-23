"""Compare our REPPO runs against the REPPO paper's published per-task results.

The comparison target is the authors' own raw result curves, committed in their GitHub
clone at ``reppo_original/results/<suite>/<Task>.csv`` (schema ``steps,trial_0..trial_N``).
No numbers are transcribed from the paper.

Three properties of that data are load-bearing and are enforced here rather than left to
the caller's memory:

1. **Only the last row is quotable.** The curves were resampled onto a fixed 20-point
   step grid, and the resampling extrapolates backwards — ``CheetahRun`` row 0 already
   reads 626.9, which is impossible for an untrained policy. :func:`paper_final` reads
   the last row; :func:`load_paper_curve` returns the rest flagged as plot-only.
2. **Their ``n`` is not a seed count.** Trial counts run 12-50 and appear to pool the
   ``small_data``/``large_data`` experiment variants. It is reported alongside every
   number so a reader can see the asymmetry against our 3 seeds.
3. **The metric must be the deterministic eval**, not ``Train/episode_reward``. Their
   ManiSkill number is a deterministic policy over 1024 one-episode envs (every final
   value in those CSVs is an exact multiple of 1/1024); their DMC number is a
   deterministic 1000-step scan. ``Train/episode_reward`` is a stochastic-policy,
   100-episode rolling window, and our ``reward_scale`` is applied inside the algorithm
   where theirs is applied in the env wrapper — two independent reasons it is not
   commensurable.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

PAPER_RESULTS = Path("/home/human/workspaces/reppo_original/results")

#: TensorBoard tag carrying the quantity comparable to the paper's, per suite.
OUR_TAG = {
    "maniskill": "eval/success_ode_100",
    "dmc": "eval/episode_return_ode_100",
}

#: Where each suite's reference CSVs live under ``PAPER_RESULTS``.
PAPER_SUITE_DIR = {"maniskill": "maniskill", "dmc": "mujoco_playground"}

#: 1024 envs x 128 steps x 381 iters. Quote this, not "50M".
BUDGET_ENV_STEPS = 49_938_432


# ---------------------------------------------------------------------------
# The paper's numbers
# ---------------------------------------------------------------------------


def load_paper_curve(suite: str, task: str) -> tuple[list[float], list[list[float]]]:
    """Return ``(steps, trials)`` where ``trials[i]`` is one trial's value per step.

    Rows before the last are resampling artifacts — usable as a plotted band, never as
    a quoted number.
    """
    path = PAPER_RESULTS / PAPER_SUITE_DIR[suite] / f"{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no published result for {suite}/{task}: {path}")
    with path.open() as fh:
        rows = [r for r in csv.reader(fh) if r]
    steps = [float(r[0]) for r in rows[1:]]
    n_trials = len(rows[1]) - 1
    trials = [[float(r[i + 1]) for r in rows[1:]] for i in range(n_trials)]
    return steps, trials


def paper_final(suite: str, task: str) -> dict:
    """Final-row summary: ``{steps, mean, sd, n, trials}``."""
    steps, trials = load_paper_curve(suite, task)
    finals = [t[-1] for t in trials]
    mean = sum(finals) / len(finals)
    var = sum((v - mean) ** 2 for v in finals) / len(finals)  # population sd, as reported
    return {
        "steps": steps[-1],
        "mean": mean,
        "sd": math.sqrt(var),
        "n": len(finals),
        "trials": finals,
    }


# ---------------------------------------------------------------------------
# Our numbers
# ---------------------------------------------------------------------------


def _event_dirs(run_root: Path) -> list[Path]:
    """Every directory under ``run_root`` holding TensorBoard events."""
    return sorted({p.parent for p in Path(run_root).rglob("events.out.tfevents.*")})


def load_our_curve(run_root: Path, suite: str) -> tuple[list[int], list[float]]:
    """Our comparable metric as ``(env_steps, values)`` for one cell.

    Configs set ``log_env_steps: true``, so the x-axis is already cumulative env steps
    and overlays directly on the paper's ``steps`` column.

    A cell can hold SEVERAL event directories: the driver retries OOM-killed runs, and
    each attempt writes its own timestamped tree. Take the attempt that got **furthest**,
    not the first one found — a killed attempt still leaves a perfectly readable event
    file, and reading it would silently report the cell as incomplete while a finished
    retry sat next to it on disk.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tag = OUR_TAG[suite]
    best: tuple[list[int], list[float]] = ([], [])
    for event_dir in _event_dirs(run_root):
        acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in set(acc.Tags().get("scalars", [])):
            continue
        events = acc.Scalars(tag)
        if not events:
            continue
        if not best[0] or events[-1].step > best[0][-1]:
            best = ([e.step for e in events], [e.value for e in events])
    return best


def our_final(run_root: Path, suite: str, budget_tol: float = 0.02) -> dict:
    """Final value of our comparable metric, with a completeness check.

    A crashed run still leaves a readable event file. Comparing a 12M-step run against
    the paper's 50M number would look like a legitimate (and terrible) result, so the
    budget is verified and short runs are marked incomplete rather than reported.
    """
    steps, values = load_our_curve(Path(run_root), suite)
    if not values:
        return {"status": "no_eval_data", "value": None, "steps": None}
    reached = steps[-1]
    complete = reached >= BUDGET_ENV_STEPS * (1.0 - budget_tol)
    return {
        "status": "ok" if complete else "incomplete",
        "value": values[-1],
        "steps": reached,
        "n_points": len(values),
    }


# ---------------------------------------------------------------------------
# Aggregation and verdicts
# ---------------------------------------------------------------------------


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic (fixed-seed), as in eval_matrix."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(12345)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def classify(ours_mean: float, theirs_mean: float, theirs_sd: float) -> str:
    """Pre-registered verdict rule (reports/PAPER_BENCH_PROTOCOL.md).

    ``match``  : within their +/-1 sd
    ``weak``   : within +/-2 sd, or >= 95% of theirs when their sd is ~0 (saturated task)
    ``miss``   : outside +/-2 sd
    """
    if theirs_sd < 1e-9:
        return "match" if ours_mean >= theirs_mean * 0.999 else ("weak" if ours_mean >= theirs_mean * 0.95 else "miss")
    z = abs(ours_mean - theirs_mean) / theirs_sd
    if z <= 1.0:
        return "match"
    if z <= 2.0:
        return "weak"
    return "miss"


def compare_task(suite: str, task: str, our_run_roots: list[Path]) -> dict:
    """One row of the comparison: their final vs ours across seeds."""
    theirs = paper_final(suite, task)
    per_seed, skipped = [], []
    for root in our_run_roots:
        rec = our_final(root, suite)
        if rec["status"] == "ok":
            per_seed.append(rec["value"])
        else:
            skipped.append((str(root), rec["status"]))

    row = {
        "task": task,
        "suite": suite,
        "theirs_mean": theirs["mean"],
        "theirs_sd": theirs["sd"],
        "theirs_n": theirs["n"],
        "ours": per_seed,
        "ours_n": len(per_seed),
        "skipped": skipped,
    }
    if per_seed:
        mean = sum(per_seed) / len(per_seed)
        row["ours_mean"] = mean
        row["ours_sd"] = math.sqrt(sum((v - mean) ** 2 for v in per_seed) / len(per_seed))
        row["ci"] = bootstrap_ci(per_seed)
        row["delta"] = mean - theirs["mean"]
        row["rel"] = (mean - theirs["mean"]) / theirs["mean"] if theirs["mean"] else float("nan")
        row["verdict"] = classify(mean, theirs["mean"], theirs["sd"])
    else:
        row["verdict"] = "no_data"
    return row


def markdown_table(rows: list[dict]) -> str:
    """Comparison table. Cells that have no data say so; nothing is silently omitted."""
    head = (
        "| task | ours (mean ± sd, n) | paper (mean ± sd, n) | Δ | Δ/paper | verdict |\n"
        "|---|---|---|---|---|---|\n"
    )
    out = [head]
    for r in sorted(rows, key=lambda x: x["task"]):
        theirs = f"{r['theirs_mean']:.3f} ± {r['theirs_sd']:.3f} (n={r['theirs_n']})"
        if r["verdict"] == "no_data":
            out.append(f"| {r['task']} | — (no completed run) | {theirs} | — | — | **no_data** |\n")
            continue
        ours = f"{r['ours_mean']:.3f} ± {r['ours_sd']:.3f} (n={r['ours_n']})"
        out.append(
            f"| {r['task']} | {ours} | {theirs} | {r['delta']:+.3f} | {r['rel']:+.1%} | {r['verdict']} |\n"
        )

    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("match", "weak", "miss", "no_data")}
    out.append(
        f"\n**Verdicts:** {counts['match']} match · {counts['weak']} weak · "
        f"{counts['miss']} miss · {counts['no_data']} no data\n"
    )
    incomplete = [(r["task"], s) for r in rows for s in r["skipped"]]
    if incomplete:
        out.append("\n**Excluded runs** (never silently dropped):\n\n")
        out.extend(f"- `{task}`: {root} — {status}\n" for task, (root, status) in incomplete)
    return "".join(out)
