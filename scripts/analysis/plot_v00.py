#!/usr/bin/env python3
"""Plots for the V0 baseline report.

Reads the aggregated summaries written by ``scripts/eval/eval_matrix.py`` plus the
TensorBoard event files of the training runs, and writes the figure set the program
requires into ``reports/figs/``.

    python scripts/analysis/plot_v00.py --summary experiments/v00/summary_E1.json \
        --logs logs/safe_rl/go2_v00 --out reports/figs

Figures
    v00_tracking.png       primary metric per arm/mode, with 95% bootstrap CIs
    v00_deploy_gap.png     deterministic vs stochastic per arm -- the hypothesis plot
    v00_learning.png       learning curves (reward, tracking) vs env steps
    v00_entropy_std.png    policy entropy and action sigma over training
    v00_duals.png          KL, alpha_temp, alpha_kl over training
    v00_command_response.png  commanded vs achieved forward velocity from a traj CSV
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind-safe qualitative palette; one colour per arm family.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def load_scalars(event_dir: Path, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    """Pull scalar series out of a TensorBoard run directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WARN] tensorboard not installed; skipping training-curve figures")
        return {}
    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = acc.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def fig_tracking(summary: list[dict], out: Path) -> None:
    rows = [r for r in summary if r["action_mode"] == "deterministic"]
    rows.sort(key=lambda r: r["tracking_error_xy"].get("mean", 9e9))
    if not rows:
        return
    labels = [f"{r['arm']}\n{(r['env_steps'] or 0)/1e6:.1f}M" for r in rows]
    means = [r["tracking_error_xy"]["mean"] for r in rows]
    lo = [m - r["tracking_error_xy"].get("ci_lo", m) for m, r in zip(means, rows)]
    hi = [r["tracking_error_xy"].get("ci_hi", m) - m for m, r in zip(means, rows)]

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(rows)), 4.5))
    ax.bar(labels, means, yerr=[lo, hi], capsize=4,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))])
    ax.set_ylabel(r"mean per-step $\|v_{cmd} - v_{xy}\|_2$  (lower is better)")
    ax.set_title("Go2 velocity tracking, deterministic action (protocol E1)\n"
                 "budget shown under each arm — bars are NOT budget-matched unless equal")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v00_tracking.png", dpi=150)
    plt.close(fig)


def fig_deploy_gap(summary: list[dict], out: Path) -> None:
    """The hypothesis figure: is the deterministic mode worse than the sampled policy?"""
    by_arm: dict[str, dict[str, float]] = {}
    for r in summary:
        by_arm.setdefault(r["arm"], {})[r["action_mode"]] = r["tracking_error_xy"].get("mean", float("nan"))
    arms = [a for a, v in by_arm.items() if "deterministic" in v and "stochastic" in v]
    if not arms:
        return
    arms.sort(key=lambda a: by_arm[a]["deterministic"])
    x = range(len(arms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(arms)), 4.5))
    ax.bar([i - w / 2 for i in x], [by_arm[a]["deterministic"] for a in arms], w,
           label="deterministic  tanh(mu)", color=PALETTE[0])
    ax.bar([i + w / 2 for i in x], [by_arm[a]["stochastic"] for a in arms], w,
           label="stochastic  a ~ pi(.|s)", color=PALETTE[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel(r"mean per-step $\|v_{cmd} - v_{xy}\|_2$")
    ax.set_title("Exploration/deployment gap: deployed mode vs the trained distribution\n"
                 "hypothesis predicts deterministic WORSE (taller) than stochastic for REPPO")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v00_deploy_gap.png", dpi=150)
    plt.close(fig)


def fig_training(log_root: Path, out: Path) -> None:
    runs = sorted(p for p in log_root.glob("*") if p.is_dir())
    if not runs:
        return
    # Tag names verified against a real event file; a missing tag is skipped, which is
    # how PPO runs (no duals, no categorical critic) share these panels with REPPO.
    panels = [
        ("v00_learning.png", [("Train/episode_reward", "episode return"),
                              ("Train/episode_length", "episode length"),
                              ("Metrics/twist/error_vel_xy", "mjlab error_vel_xy (cumulative — see protocol)")]),
        ("v00_entropy_std.png", [("Loss/entropy", "policy entropy (nats, summed over dims)"),
                                 ("Policy/mean_noise_std", "action sigma")]),
        ("v00_duals.png", [("Loss/kl", "KL(pi_old || pi_new)"),
                           ("Loss/alpha_temp", "alpha_temp"),
                           ("Loss/alpha_kl", "alpha_kl")]),
        ("v00_critic.png", [("Loss/q_value", "Q estimate"),
                            ("Loss/q_bias", "Q bias  E[Q - target]"),
                            ("Loss/frac_targets_clipped", "fraction of categorical targets clipped")]),
    ]
    for fname, tags in panels:
        fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4))
        if len(tags) == 1:
            axes = [axes]
        drew = False
        for run_i, run in enumerate(runs):
            series = load_scalars(run, [t for t, _ in tags])
            for ax, (tag, title) in zip(axes, tags):
                if tag not in series:
                    continue
                steps, vals = series[tag]
                ax.plot(steps, vals, label=run.name, color=PALETTE[run_i % len(PALETTE)], lw=1.2)
                ax.set_title(title)
                ax.set_xlabel("iteration")
                ax.grid(alpha=0.3)
                drew = True
        if drew:
            axes[-1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def fig_command_response(traj_csvs: list[Path], out: Path) -> None:
    """Commanded vs achieved forward velocity — exposes gain, bias and lag."""
    traj_csvs = [p for p in traj_csvs if p.exists()]
    if not traj_csvs:
        return
    fig, axes = plt.subplots(len(traj_csvs), 1, figsize=(11, 3.1 * len(traj_csvs)), sharex=True)
    if len(traj_csvs) == 1:
        axes = [axes]
    for ax, path in zip(axes, traj_csvs):
        rows = list(csv.DictReader(path.open()))
        step = [float(r["step"]) for r in rows]
        ax.plot(step, [float(r["cmd_vx"]) for r in rows], label="commanded $v_x$",
                color="black", lw=1.4, ls="--")
        ax.plot(step, [float(r["vx"]) for r in rows], label="achieved $v_x$",
                color=PALETTE[0], lw=1.2)
        ax.set_title(path.stem)
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("control step")
    fig.tight_layout()
    fig.savefig(out / "v00_command_response.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="experiments/v00/summary_E1.json")
    ap.add_argument("--logs", default="logs/safe_rl/go2_v00")
    ap.add_argument("--traj", nargs="*", default=[], help="Trajectory CSVs for the command-response figure.")
    ap.add_argument("--out", default="reports/figs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        fig_tracking(summary, out)
        fig_deploy_gap(summary, out)
    else:
        print(f"[WARN] no summary at {summary_path}; skipping evaluation figures")

    log_root = Path(args.logs)
    if log_root.exists():
        fig_training(log_root, out)
    else:
        print(f"[WARN] no log root at {log_root}; skipping training figures")

    fig_command_response([Path(p) for p in args.traj], out)
    print(f"[INFO] figures written to {out}")


if __name__ == "__main__":
    main()
