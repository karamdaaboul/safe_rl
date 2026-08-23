#!/usr/bin/env python
"""Aggregate per-episode eval CSVs into the comparison table the FH-DCMPO gates are judged on.

Reads `logs/deteval_*.csv` (written by `eval_safety_gymnasium.py --eval_csv`), recomputes the tail
statistics with the same code the evaluator uses, and prints them beside the recorded baselines.

Why a script rather than eyeballing the evaluator's own summary: the gates are comparisons, and the
seed spread on cost is +-3.05 (codex/qr-dmpo-math.md), so a single arm's numbers mean little without
the reference rows and the seed count next to them. Printing them together makes it hard to quietly
compare a 1-seed arm against a 3-seed baseline.

Usage:
    python scripts/eval/compare_fhdcmpo.py                    # every logs/deteval_*.csv
    python scripts/eval/compare_fhdcmpo.py 'logs/deteval_fhdcmpo_*.csv'
    python scripts/eval/compare_fhdcmpo.py --limit 25 --group  # average seeds sharing a prefix
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safe_rl.common.tail_eval import episode_cost_stats  # noqa: E402

# Recorded reference points, 50-episode deterministic eval at budget 25 (codex/qr-dmpo-math.md 6.1).
# p90 and CVaR_0.5 are printed as "-" because that note records only CVaR_0.9; where the raw
# per-episode CSV still exists locally the MEASURED block below recomputes all of them, and it
# reproduces these rows exactly (e.g. dmpo_c1_s2 -> 33.24 / 24.84 / 79.4 / 46.0).
BASELINES = [
    # (name, task, seeds, reward, cost, cvar_0.9, violations%)
    ("QR-DMPO", "PointGoal1", 3, 23.84, 18.97, 56.9, 32.0),
    ("DMPO (C51)", "PointGoal1", 3, 25.70, 24.68, 71.5, 43.3),
    ("CVPO (scalar)", "PointGoal1", 3, 23.31, 26.73, 73.1, 43.3),
    ("PPOL-PID", "PointGoal1", 3, 21.65, 26.50, 76.7, 40.7),
    ("QR-DMPO", "CarGoal1", 1, 32.30, 28.68, 121.2, 34.0),
    ("DMPO (C51)", "CarGoal1", 1, 33.24, 24.84, 79.4, 46.0),
    ("CVPO (scalar)", "CarGoal1", 1, 33.25, 24.74, 85.8, 38.0),
]

HEADER = (
    f"{'arm':<34} {'task':<11} {'n':>3} {'ep':>4} {'reward':>8} {'cost':>7} {'p90':>7} {'CVaR.5':>7} {'CVaR.9':>7} {'viol%':>7}"
)


def read_csv(path: str) -> tuple[list[float], list[float]]:
    costs, rewards = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            costs.append(float(row["cost"]))
            rewards.append(float(row["reward"]))
    return costs, rewards


def task_of(name: str) -> str:
    low = name.lower()
    if "cargoal1" in low or "_c1" in low:
        return "CarGoal1"
    # Everything else is PointGoal1: this repo's deteval naming marks CarGoal1 explicitly
    # (`_c1_`) and leaves PointGoal1 implicit (`deteval_qrdmpo_s2`), so "unlabelled" means
    # PointGoal1 rather than unknown.
    return "PointGoal1"


def row(label: str, task: str, seeds: int, n_ep: int, s: dict) -> str:
    return (
        f"{label:<34} {task:<11} {seeds:>3} {n_ep:>4} {s['reward_mean']:>8.2f} {s['cost_mean']:>7.2f} "
        f"{s['cost_p90']:>7.1f} {s['cost_cvar_0.5']:>7.1f} {s['cost_cvar_0.9']:>7.1f} "
        f"{100 * s['budget_exceedance_rate']:>7.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=["logs/deteval_*.csv"])
    ap.add_argument("--limit", type=float, default=25.0, help="cost budget d")
    ap.add_argument(
        "--group",
        action="store_true",
        help="pool episodes across seeds sharing a name prefix (strips a trailing _s<N>)",
    )
    args = ap.parse_args()

    paths = sorted({p for pat in (args.patterns or ["logs/deteval_*.csv"]) for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no CSVs matched {args.patterns}")

    print("\nRECORDED BASELINES (50-ep deterministic eval, budget 25)")
    print(HEADER)
    for name, task, seeds, rew, cost, cvar, viol in BASELINES:
        print(
            f"{name:<34} {task:<11} {seeds:>3} {50:>4} {rew:>8.2f} {cost:>7.2f} "
            f"{'-':>7} {'-':>7} {cvar:>7.1f} {viol:>7.1f}"
        )

    groups: dict[str, list[str]] = {}
    for p in paths:
        stem = Path(p).stem.replace("deteval_", "")
        key = re.sub(r"_s\d+$", "", stem) if args.group else stem
        groups.setdefault(key, []).append(p)

    print("\nMEASURED")
    print(HEADER)
    for key, members in sorted(groups.items()):
        costs: list[float] = []
        rewards: list[float] = []
        for p in members:
            c, r = read_csv(p)
            costs += c
            rewards += r
        if not costs:
            continue
        stats = episode_cost_stats(costs, cost_limit=args.limit, rewards=rewards)
        print(row(key, task_of(key), len(members), len(costs), stats))
        # Seed spread on cost is the number that decides whether a difference is interpretable.
        if args.group and len(members) > 1:
            per_seed = [statistics.mean(read_csv(p)[0]) for p in members]
            print(
                f"{'':<34} {'':<11} {'':>3} {'':>4} {'':>8} "
                f"{'+-%.2f' % (statistics.stdev(per_seed)):>7}  <- per-seed cost sd"
            )

    print("\nGates: S1 cost within +-3.05 of 18.97 (PointGoal1). S2 violations < 20% at reward >= 22.")
    print("S3/S4 target: violations <= 10% (the alpha=0.9 bound's prediction) at reward >= 23.8.")
    print("Seed spread on cost is +-3.05, so single-seed differences under ~3 are not interpretable.\n")


if __name__ == "__main__":
    main()
