#!/usr/bin/env python3
"""Build the REPPO paper-comparison table and per-task learning-curve overlays.

Reads the sweep output written by ``scripts/bench/run_paper_bench.py`` and the authors'
own result CSVs in ``reppo_original/results/``, and emits a markdown table plus one PNG
per task (our seeds against their trial band).

    python scripts/analysis/compare_to_paper.py --out reports/paper_bench.md

Metric choice is not free — see ``safe_rl/analysis/paper_bench.OUR_TAG`` and
``reports/PAPER_BENCH_PROTOCOL.md``. We compare deterministic-eval numbers, never
``Train/episode_reward``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from safe_rl.analysis import paper_bench as pb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = REPO / "experiments" / "paper_bench"

# Colour-blind-safe, matching scripts/analysis/plot_v00.py
OURS_COLOR = "#4C72B0"
THEIRS_COLOR = "#DD8452"

SUITE_TITLE = {
    "maniskill": ("ManiSkill", "success rate", "PARITY claim: their torch trainer"),
    "dmc": ("MuJoCo Playground DMC", "episode return", "BREADTH claim: their JAX trainer"),
}


def seed_dirs(sweep_root: Path, suite: str, task: str) -> list[Path]:
    base = sweep_root / suite / task
    return sorted(p for p in base.glob("s*") if p.is_dir()) if base.exists() else []


def plot_task(suite: str, task: str, sweep_root: Path, out_dir: Path) -> Path | None:
    try:
        steps, trials = pb.load_paper_curve(suite, task)
    except FileNotFoundError:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Their trials as a band. Rows before the last are resampling artifacts (row 0 of
    # CheetahRun already reads 627), so the band is dashed and captioned as such; only
    # the final point is a quotable number.
    lo = [min(t[i] for t in trials) for i in range(len(steps))]
    hi = [max(t[i] for t in trials) for i in range(len(steps))]
    mean = [sum(t[i] for t in trials) / len(trials) for i in range(len(steps))]
    ax.fill_between(steps, lo, hi, color=THEIRS_COLOR, alpha=0.20, linewidth=0)
    ax.plot(steps, mean, color=THEIRS_COLOR, linestyle="--", linewidth=1.6,
            label=f"paper (n={len(trials)})")
    ax.plot([steps[-1]], [mean[-1]], marker="o", color=THEIRS_COLOR, markersize=6)
    ax.annotate(f"{mean[-1]:.3g}", (steps[-1], mean[-1]), textcoords="offset points",
                xytext=(-6, 8), ha="right", color=THEIRS_COLOR, fontsize=9)

    n_ours = 0
    for i, sdir in enumerate(seed_dirs(sweep_root, suite, task)):
        xs, ys = pb.load_our_curve(sdir, suite)
        if not ys:
            continue
        n_ours += 1
        ax.plot(xs, ys, color=OURS_COLOR, alpha=0.75, linewidth=1.4,
                label="ours (per seed)" if n_ours == 1 else None)

    _, metric, claim = SUITE_TITLE[suite]
    ax.set_title(f"{task}  —  {claim}", fontsize=10)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01,
             "paper curve dashed: intermediate points are resampling artifacts; only the final point is quotable",
             fontsize=6, alpha=0.7)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"paper_bench_{suite}_{task}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "paper_bench.md")
    ap.add_argument("--figs", type=Path, default=REPO / "reports" / "figs")
    ap.add_argument("--suites", default="maniskill,dmc")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    sections = []
    for suite in args.suites.split(","):
        suite_dir = args.sweep / suite
        if not suite_dir.exists():
            print(f"[skip] no runs for {suite} under {suite_dir}")
            continue
        tasks = sorted(p.name for p in suite_dir.iterdir() if p.is_dir())
        rows = []
        for task in tasks:
            try:
                rows.append(pb.compare_task(suite, task, seed_dirs(args.sweep, suite, task)))
            except FileNotFoundError as exc:
                print(f"[skip] {suite}/{task}: {exc}")
            if not args.no_plots:
                if (path := plot_task(suite, task, args.sweep, args.figs)) is not None:
                    print(f"[fig] {path}")
        if not rows:
            continue
        title, metric, claim = SUITE_TITLE[suite]
        sections.append(
            f"## {title}\n\n"
            f"Metric: **{metric}** at the final checkpoint, deterministic policy "
            f"({pb.OUR_TAG[suite]}), budget {pb.BUDGET_ENV_STEPS:,} env steps.\n\n"
            f"**{claim}.**\n\n" + pb.markdown_table(rows)
        )

    if not sections:
        print("no completed runs found; nothing to report")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# REPPO vs. the published REPPO results\n\n"
        "Generated by `scripts/analysis/compare_to_paper.py`. Protocol and confounders: "
        "`reports/PAPER_BENCH_PROTOCOL.md`.\n\n" + "\n\n".join(sections) + "\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
