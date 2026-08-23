"""Tail-aware evaluation statistics for episodic cost.

A mean cannot detect a risk-constraint win: two policies with the same mean episodic cost can
have entirely different tails, and the tail is exactly what a CVaR constraint targets. Measured
on the constrained arms, mean cost was 21.4 (under a limit of 25) while 23-31% of individual
episodes still exceeded that limit -- the mean was hiding the quantity under study.

Reports mean / median / P90 / P95 / min / max, empirical CVaR at 0.5 and 0.9, and the
budget-exceedance rate, plus reward mean/std/sem.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

CVAR_LEVELS = (0.5, 0.9)

EPISODE_CSV_FIELDS = ("episode", "cost", "reward", "length")

SUMMARY_FIELDS = (
    "n_episodes", "n_distinct",
    "cost_mean", "cost_std", "cost_sem", "cost_min", "cost_median",
    "cost_p90", "cost_p95", "cost_max",
    "cost_cvar_0.5", "cost_cvar_0.9",
    "budget_exceedance_rate", "cost_limit",
    "reward_mean", "reward_std", "reward_sem",
)


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    """Mean of the worst ``1 - alpha`` fraction (upper tail; the cost convention).

    Uses at least one sample, so ``CVaR_0.99`` of a short run degrades to the maximum rather
    than dividing by zero.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if values.size == 0:
        raise ValueError("cannot compute CVaR of an empty sample")
    k = max(1, int(round((1.0 - alpha) * values.size)))
    return float(np.sort(values)[-k:].mean())


def episode_cost_stats(
    costs: Sequence[float],
    cost_limit: float,
    rewards: Sequence[float] | None = None,
) -> dict:
    """Summary statistics over completed episodes. Does not mutate its inputs."""
    c = np.asarray(list(costs), dtype=np.float64)
    if c.size == 0:
        raise ValueError("no episodes to summarise")

    # Distinct-episode count. safety_gymnasium_vec_env tiles ONE seed across all sub-envs
    # (deliberate, for hidden-goal/MAML per-task adaptation), so a deterministic policy makes
    # every parallel env run the SAME trajectory. Counting those as independent episodes
    # inflates n by num_envs and silently produces falsely tight SEMs and degenerate tails --
    # observed: 24 "episodes" that were 3 distinct ones repeated 8 times.
    #
    # Count distinctness on the (cost, reward) PAIR, not the cost alone. Episodic cost is a
    # sum of binary contact penalties, so it is integer-valued and repeats legitimately: 50
    # genuinely different episodes routinely yield only ~24-32 distinct costs, which tripped
    # the n/2 threshold and reported a seed-sharing failure that had not happened. Reward is
    # continuous, so a repeated trajectory collides in both coordinates while merely-equal
    # costs do not.
    _r = np.asarray(list(rewards), dtype=np.float64) if rewards is not None else None
    if _r is not None and _r.size == c.size:
        pairs = np.stack([np.round(c, 9), np.round(_r, 9)], axis=1)
        n_distinct = int(np.unique(pairs, axis=0).shape[0])
    else:
        n_distinct = int(np.unique(np.round(c, 9)).size)
    out: dict = {
        "n_episodes": int(c.size),
        "n_distinct": n_distinct,
        "cost_limit": float(cost_limit),
        "cost_mean": float(c.mean()),
        "cost_std": float(c.std(ddof=1)) if c.size > 1 else 0.0,
        "cost_min": float(c.min()),
        "cost_median": float(np.median(c)),
        "cost_p90": float(np.percentile(c, 90)),
        "cost_p95": float(np.percentile(c, 95)),
        "cost_max": float(c.max()),
        "budget_exceedance_rate": float(np.mean(c > cost_limit)),
    }
    out["cost_sem"] = out["cost_std"] / np.sqrt(c.size) if c.size > 1 else 0.0
    for a in CVAR_LEVELS:
        out[f"cost_cvar_{a}"] = empirical_cvar(c, a)

    if rewards is not None:
        r = np.asarray(list(rewards), dtype=np.float64)
        if r.size != c.size:
            raise ValueError(f"rewards has {r.size} entries but costs has {c.size}")
        out["reward_mean"] = float(r.mean())
        out["reward_std"] = float(r.std(ddof=1)) if r.size > 1 else 0.0
        out["reward_sem"] = out["reward_std"] / np.sqrt(r.size) if r.size > 1 else 0.0
    else:
        out["reward_mean"] = out["reward_std"] = out["reward_sem"] = float("nan")
    return out


def write_episode_csv(
    path: str | Path,
    costs: Sequence[float],
    rewards: Sequence[float],
    lengths: Sequence[int],
) -> Path:
    """Write one row per episode. Per-episode data is kept so the tail can be re-analysed."""
    if not (len(costs) == len(rewards) == len(lengths)):
        raise ValueError(
            f"length mismatch: costs {len(costs)}, rewards {len(rewards)}, lengths {len(lengths)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EPISODE_CSV_FIELDS)
        for i, (c, r, ln) in enumerate(zip(costs, rewards, lengths)):
            writer.writerow([i, float(c), float(r), int(ln)])
    return path


def format_summary(stats: dict) -> str:
    """One-line human-readable summary for logs."""
    dup = ""
    if stats.get("n_distinct", stats["n_episodes"]) < max(2, stats["n_episodes"] // 2):
        dup = (f"  [WARNING: only {stats['n_distinct']} distinct episodes among "
               f"{stats['n_episodes']} -- parallel envs may share a seed (the vec env tiles "
               f"one seed across sub-envs), so the effective sample size is far below n]")
    return (
        f"n={stats['n_episodes']} "
        f"reward {stats['reward_mean']:.2f}+-{stats['reward_sem']:.2f} | "
        f"cost mean {stats['cost_mean']:.2f} med {stats['cost_median']:.2f} "
        f"p90 {stats['cost_p90']:.2f} p95 {stats['cost_p95']:.2f} "
        f"CVaR0.5 {stats['cost_cvar_0.5']:.2f} CVaR0.9 {stats['cost_cvar_0.9']:.2f} | "
        f"over-budget {100 * stats['budget_exceedance_rate']:.1f}%" + dup
    )
