#!/usr/bin/env python3
"""Drive the frozen evaluation protocol over a manifest of arms.

Runs ``scripts/eval/unitree_mjlab.py`` once per (arm x action mode x eval seed), each
in its own subprocess so every evaluation builds a fresh mjlab env, then aggregates
per-arm statistics and appends rows to ``experiments/registry.csv``.

The protocol itself is frozen in ``reports/EVAL_PROTOCOL.md``; this script only
executes it. It is resumable: an evaluation whose JSON already exists is skipped, so
an interrupted sweep can be re-run without repeating work.

Usage
-----
    python scripts/eval/eval_matrix.py --manifest experiments/v00/arms.json \
        --out experiments/v00 --protocol E1 --device cuda:0 --gpu 0

Manifest: a JSON list of objects with keys
    label            unique arm name, used for output filenames
    checkpoint       path to model_*.pt
    config           path to the YAML the checkpoint was trained with
    algorithm        PPO | REPPO   (q_argmax is only offered for REPPO)
    env_steps        training budget, for the honest budget column
    train_seed       the training seed (or null for external checkpoints)
    modes            optional override of the action modes to run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO / "scripts" / "eval" / "unitree_mjlab.py"
REGISTRY = REPO / "experiments" / "registry.csv"

# Frozen protocols — see reports/EVAL_PROTOCOL.md. Changing these invalidates
# cross-version comparison.
PROTOCOLS: dict[str, dict[str, Any]] = {
    "E1": {"num_envs": 50, "episodes": 50, "seeds": [42, 43, 44], "one_per_env": True},
    "E2": {"num_envs": 1, "episodes": 1, "seeds": [3, 7, 11, 21, 33], "one_per_env": True},
}


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic: fixed-seed RNG."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    import random

    rng = random.Random(12345)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def interquartile_mean(values: list[float]) -> float:
    """IQM: mean of the middle 50%. Robust to the catastrophic runs safe-RL produces."""
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    lo, hi = n // 4, n - n // 4
    core = s[lo:hi] or s
    return sum(core) / len(core)


def run_one(
    arm: dict[str, Any],
    mode: str,
    seed: int,
    proto: dict[str, Any],
    out_dir: Path,
    device: str,
    gpu: str,
    dry_run: bool,
) -> Path | None:
    """Run a single evaluation; return the JSON path (or None if it failed)."""
    tag = f"{arm['label']}__{mode}__s{seed}"
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"[skip] {tag} (already evaluated)")
        return json_path

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--env_id", arm.get("env_id", "Unitree-Go2-Flat"),
        "--config", arm["config"],
        "--checkpoint", arm["checkpoint"],
        "--num_envs", str(proto["num_envs"]),
        "--episodes", str(proto["episodes"]),
        "--seed", str(seed),
        "--headless",
        "--device", device,
        "--json_out", str(json_path),
    ]
    if proto["one_per_env"]:
        cmd.append("--one_episode_per_env")
    if mode.startswith("q_argmax"):
        cmd += ["--q_argmax", mode.split(":", 1)[1]]
    else:
        cmd += ["--action_mode", mode]

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return None

    env = dict(os.environ)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu,
        "MUJOCO_GL": "egl",
        "WANDB_MODE": "disabled",
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
    })
    print(f"[run ] {tag}")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0 or not json_path.exists():
        # Record the failure rather than dropping it — the program forbids silently
        # excluding failed runs from the results.
        (out_dir / f"{tag}.FAILED.log").write_text(proc.stdout[-8000:] + "\n---STDERR---\n" + proc.stderr[-8000:])
        print(f"[FAIL] {tag} rc={proc.returncode}; log written")
        return None
    return json_path


def aggregate(results: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Aggregate one metric across evaluation seeds."""
    vals = [r[key] for r in results if r.get(key) is not None]
    if not vals:
        return {}
    lo, hi = bootstrap_ci(vals)
    return {
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "iqm": interquartile_mean(vals),
        "ci_lo": lo,
        "ci_hi": hi,
        "worst": max(vals) if "error" in key else min(vals),
        "n": len(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="JSON list of arms to evaluate.")
    ap.add_argument("--out", required=True, help="Directory for per-evaluation JSON summaries.")
    ap.add_argument("--protocol", default="E1", choices=sorted(PROTOCOLS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--version", default="v00", help="Version label written to the registry.")
    ap.add_argument("--git_commit", default=None, help="Commit hash for the registry (defaults to HEAD).")
    ap.add_argument("--only", default=None, help="Comma-separated arm labels to restrict to.")
    ap.add_argument(
        "--modes", default=None,
        help="Comma-separated action modes overriding the per-arm default "
             "(e.g. 'deterministic' for the E2 survival protocol, or 'q_argmax:64').",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_registry", action="store_true", help="Skip appending to experiments/registry.csv.")
    args = ap.parse_args()

    proto = PROTOCOLS[args.protocol]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = json.loads(Path(args.manifest).read_text())
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        arms = [a for a in arms if a["label"] in wanted]

    commit = args.git_commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), capture_output=True, text=True
        ).stdout.strip() or "unknown"

    summary_rows: list[dict[str, Any]] = []
    registry_rows: list[list[str]] = []

    for arm in arms:
        if not Path(arm["checkpoint"]).exists():
            print(f"[MISS] {arm['label']}: checkpoint not found -> {arm['checkpoint']}")
            continue
        if args.modes:
            modes = [m.strip() for m in args.modes.split(",")]
        else:
            modes = arm.get("modes") or ["deterministic", "stochastic"]
        # q_argmax needs an explicit Q; PPO has none.
        modes = [m for m in modes if not (m.startswith("q_argmax") and arm["algorithm"] != "REPPO")]
        for mode in modes:
            per_seed: list[dict[str, Any]] = []
            for seed in proto["seeds"]:
                jp = run_one(arm, mode, seed, proto, out_dir, args.device, args.gpu, args.dry_run)
                if jp is None:
                    continue
                per_seed.append(json.loads(jp.read_text()))
            if not per_seed:
                continue

            agg = {
                "arm": arm["label"],
                "algorithm": arm["algorithm"],
                "env_steps": arm.get("env_steps"),
                "train_seed": arm.get("train_seed"),
                "action_mode": mode,
                "protocol": args.protocol,
                "eval_seeds": proto["seeds"][: len(per_seed)],
                "reward": aggregate(per_seed, "mean_reward"),
                "length": aggregate(per_seed, "mean_length"),
                "survival_rate": aggregate(per_seed, "survival_rate"),
                "tracking_error_xy": aggregate(per_seed, "tracking_error_xy"),
                "yaw_error": aggregate(per_seed, "yaw_error"),
                "per_seed": per_seed,
            }
            summary_rows.append(agg)

            def g(k: str, stat: str = "mean") -> str:
                v = agg[k].get(stat)
                return "" if v is None else f"{v:.4f}"

            registry_rows.append([
                args.version, f"{arm['label']}__{mode}__{args.protocol}", commit,
                arm["config"], arm.get("env_id", "Unitree-Go2-Flat"), arm["algorithm"],
                str(arm.get("train_seed", "")), str(arm.get("env_steps", "")), "",
                arm["checkpoint"], "evaluated",
                g("reward") if mode == "deterministic" else "",
                g("reward") if mode == "stochastic" else "",
                g("tracking_error_xy"), g("yaw_error"), g("survival_rate"),
                "", "", "", "", "", "", "", "", "",
                f"protocol={args.protocol}; mode={mode}; eval_seeds={proto['seeds']}",
            ])

    summary_path = out_dir / f"summary_{args.protocol}.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    keep = {(r["arm"], r["action_mode"]) for r in summary_rows}
    existing = [r for r in existing if (r["arm"], r["action_mode"]) not in keep]
    summary_path.write_text(json.dumps(existing + summary_rows, indent=2))
    print(f"\n[INFO] wrote {len(summary_rows)} aggregated rows to {summary_path}")

    if registry_rows and not args.no_registry and not args.dry_run:
        with REGISTRY.open("a", newline="") as fh:
            csv.writer(fh).writerows(registry_rows)
        print(f"[INFO] appended {len(registry_rows)} rows to {REGISTRY}")

    # Console table, sorted by the primary metric.
    print(f"\n{'arm':<28} {'mode':<14} {'steps':>10} {'track_xy':>9} {'reward':>8} {'surv':>6}")
    for r in sorted(summary_rows, key=lambda x: x["tracking_error_xy"].get("mean", 9e9)):
        print(
            f"{r['arm']:<28} {r['action_mode']:<14} {str(r['env_steps'] or '?'):>10} "
            f"{r['tracking_error_xy'].get('mean', float('nan')):>9.4f} "
            f"{r['reward'].get('mean', float('nan')):>8.2f} "
            f"{r['survival_rate'].get('mean', float('nan')):>6.2f}"
        )


if __name__ == "__main__":
    main()
