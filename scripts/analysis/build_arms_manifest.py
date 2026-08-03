#!/usr/bin/env python3
"""Build the arms manifest consumed by scripts/eval/eval_matrix.py.

Run directories carry a timestamp in their name, so the manifest cannot be written by
hand ahead of training. This discovers the runs by ``run_name`` suffix, resolves the
final checkpoint (the pre-registered selection rule — see reports/EVAL_PROTOCOL.md),
and emits a JSON list. Checkpoints that do not exist yet are skipped with a warning so
the manifest can be regenerated as seeds finish.

    python scripts/analysis/build_arms_manifest.py --out experiments/v00/arms.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Arms whose checkpoints are produced by this program's own training runs.
LOCAL_RUNS = [
    # (run_name glob, label prefix, algorithm, config, env_steps, final iter)
    ("reppo_v00_s*", "reppo_v00", "REPPO", "config/mjlab_go2_reppo_v00_baseline.yaml", 39_321_600, 299),
    ("ppo_39M_s*", "ppo_39M", "PPO", "config/mjlab_go2_ppo.yaml", 39_321_600, 399),
]

# Pre-existing checkpoints: the convergence-budget PPO reference and the single-lever
# reference-parity REPPO ablations. Kept so V0 re-measures them under the fixed
# evaluator rather than trusting the superseded pre-V0 numbers.
EXTERNAL = [
    {
        "label": "ppo_196M",
        "checkpoint": "logs/juwels_pull/2026-07-31_12-58-54_go2_ppo_4096/model_1999.pt",
        "config": "logs/juwels_pull/2026-07-31_12-58-54_go2_ppo_4096/params/agent.yaml",
        "algorithm": "PPO", "env_steps": 196_608_000, "train_seed": 1,
    },
    {
        "label": "reppo_v28_refparity",
        "checkpoint": "logs/safe_rl/go2_velocity/2026-08-03_09-11-35_go2_v28_refparity_s1/model_299.pt",
        "config": "config/mjlab_go2_reppo_v28_refparity.yaml",
        "algorithm": "REPPO", "env_steps": 39_321_600, "train_seed": 1,
    },
    {
        "label": "reppo_v29_dualclip",
        "checkpoint": "logs/safe_rl/go2_velocity/2026-08-03_10-10-40_go2_v29_dualclip_s1/model_299.pt",
        "config": "config/mjlab_go2_reppo_v29_dualclip.yaml",
        "algorithm": "REPPO", "env_steps": 39_321_600, "train_seed": 1,
    },
    {
        "label": "reppo_v30_atoms151",
        "checkpoint": "logs/safe_rl/go2_velocity/2026-08-03_11-02-06_go2_v30_atoms151_s1/model_299.pt",
        "config": "config/mjlab_go2_reppo_v30_atoms151.yaml",
        "algorithm": "REPPO", "env_steps": 39_321_600, "train_seed": 1,
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", default="logs/safe_rl/go2_v00")
    ap.add_argument("--out", default="experiments/v00/arms.json")
    args = ap.parse_args()

    arms: list[dict] = []
    log_root = REPO / args.log_root

    for glob, prefix, algo, config, steps, final_iter in LOCAL_RUNS:
        for run_dir in sorted(log_root.glob(f"*_{glob}")):
            m = re.search(r"_s(\d+)$", run_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            ckpt = run_dir / f"model_{final_iter}.pt"
            if not ckpt.exists():
                print(f"[skip] {run_dir.name}: no {ckpt.name} yet (still training?)")
                continue
            arms.append({
                "label": f"{prefix}_s{seed}",
                "checkpoint": str(ckpt.relative_to(REPO)),
                "config": config,
                "algorithm": algo,
                "env_steps": steps,
                "train_seed": seed,
                "run_dir": str(run_dir.relative_to(REPO)),
            })

    for arm in EXTERNAL:
        if (REPO / arm["checkpoint"]).exists():
            arms.append(arm)
        else:
            print(f"[skip] {arm['label']}: checkpoint missing")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arms, indent=2))
    print(f"[INFO] wrote {len(arms)} arms to {out}")
    for a in arms:
        print(f"  {a['label']:<24} {a['algorithm']:<6} {a['env_steps']/1e6:>6.1f}M  {a['checkpoint']}")


if __name__ == "__main__":
    main()
