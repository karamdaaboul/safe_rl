#!/usr/bin/env python3
"""Aggregate P3O HL-Gauss vs MSE comparison from SLURM .out logs.

Parses 'Mean reward:' and 'Constraint 0 cost:' per iteration, then reports the
mean/std over the final WINDOW iterations (converged estimate) per run and group.
"""
import glob
import os
import re
import statistics as st

LOGS = "/p/scratch/hai_1075/safe_rl/logs"
WINDOW = 100  # final iterations used as the "converged" window
LIMIT = 25.0

rew_re = re.compile(r"Mean reward:\s*([-\d.]+)")
cost_re = re.compile(r"Constraint 0 cost:\s*([-\d.]+)")


def series(path):
    rews, costs = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            m = rew_re.search(line)
            if m:
                rews.append(float(m.group(1)))
            m = cost_re.search(line)
            if m:
                costs.append(float(m.group(1)))
    return rews, costs


def tail_stats(xs):
    w = xs[-WINDOW:] if len(xs) >= WINDOW else xs
    if not w:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = st.mean(w)
    sd = st.pstdev(w) if len(w) > 1 else 0.0
    return mean, sd, min(w), max(w)


TAGS = ("g2c40hlg",)
groups = {t: [] for t in TAGS}
print(f"{'run':<10} {'n_iter':>6} {'rew_mean':>9} {'rew_sd':>7} "
      f"{'cost_mean':>9} {'cost_sd':>7} {'cost_min':>8} {'cost_max':>8} {'cost_ratio':>10}")
for tag in TAGS:
    for f in sorted(glob.glob(os.path.join(LOGS, f"{tag}_s*.out"))):
        r, c = series(f)
        rm, rsd, _, _ = tail_stats(r)
        cm, csd, cmn, cmx = tail_stats(c)
        groups[tag].append((rm, cm, csd))
        name = os.path.basename(f).split("-")[0]
        print(f"{name:<10} {len(r):>6} {rm:>9.2f} {rsd:>7.2f} "
              f"{cm:>9.2f} {csd:>7.2f} {cmn:>8.2f} {cmx:>8.2f} {cm/LIMIT:>10.2f}")

print("\n=== group means over final window (across seeds) ===")
print(f"{'group':<6} {'rew_mean':>9} {'cost_mean':>9} {'cost_ratio':>10} {'within_seed_cost_sd':>20}")
for tag in TAGS:
    g = groups[tag]
    if not g:
        continue
    rew = st.mean(x[0] for x in g)
    cost = st.mean(x[1] for x in g)
    wsd = st.mean(x[2] for x in g)  # avg within-seed cost volatility in the window
    print(f"{tag:<6} {rew:>9.2f} {cost:>9.2f} {cost/LIMIT:>10.2f} {wsd:>20.2f}")
print(f"\nWINDOW={WINDOW} iters, cost_limit={LIMIT}")
