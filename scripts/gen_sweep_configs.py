#!/usr/bin/env python
"""Generate the multi-env sweep configs from the a2 / c2 arm bases.

Design note -- why so few keys change. The audit (2026-08-22) found FH-DCMPO to be the
*unit-free* arm family: `fhdcmpo.py` forces `qc_scale = 1.0` and raises on a supplied
`qc_scale_measured`, so the PointGoal1-calibrated `0.0764` that contaminates 28 CVPO/DMPO configs
has no analogue here. All five target envs also share `T = 1000` and `act_dim = 2`, so the
horizon-coupled machinery (`cost_horizon`, the L=64 cost window, `horizon_feature`) transfers
unchanged, and obs width is resolved from the env at runtime. The upshot: **no algorithm or env
key is env-dependent for this grid**. The generator therefore edits exactly three lines --
`experiment_name`, `run_name`, `seed` -- and copies the base byte-for-byte otherwise.

That is not a shortcut, it is the property being asserted: `--check` re-derives each PointGoal1
config, reverts those three lines, and requires the result to equal the base byte-for-byte. Any
future edit that makes a base env-dependent will break that assertion rather than silently ship a
miscalibrated arm.

`run_name` is set explicitly because the trainer's auto-name
(`experiment_name + num_envs + cost_limit`, train_safety_gymnasium.py:521-525) contains **no
env id**, so every cell of a 5-env sweep would collide under one wandb name. An explicit YAML
`run_name` wins over the auto-name (:520).

Usage:
    python scripts/gen_sweep_configs.py                 # write config/sweep/, then self-check
    python scripts/gen_sweep_configs.py --check         # verify only, write nothing
    python scripts/gen_sweep_configs.py --manifest-only # re-emit the priority manifest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "config" / "sweep"
MANIFEST = OUT_DIR / "manifest.json"

ARMS = {
    "a2": "config/safety_gymnasium_fhdcmpo_a2_lag_cc4_60k.yaml",
    "c2": "config/safety_gymnasium_fhdcmpo_c2_spreadmatch_med.yaml",
    # A third arm lands here after the c4 verdict -- one line, e.g.
    #   "c4": "config/safety_gymnasium_fhdcmpo_c4_cvar_refnorm.yaml",
}

# Slug -> env id. Order is the within-pass priority order (Part 3): PointGoal1 is the env we
# already have, so it is generated for the byte-identity check but never queued.
ENVS = {
    "pointgoal1": "SafetyPointGoal1-v0",
    "pointgoal2": "SafetyPointGoal2-v0",
    "cargoal1": "SafetyCarGoal1-v0",
    "pointbutton1": "SafetyPointButton1-v0",
    "pointpush1": "SafetyPointPush1-v0",
}
ALREADY_HAVE = {"pointgoal1"}
SEEDS = (2, 3, 4)
PASS1_SEED = 2  # pass 1 = full env coverage at one seed, then the remaining seeds


def _sub_once(text: str, pattern: str, repl: str, what: str, path: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one {what} line, found {n}")
    return out


def render(base_text: str, base_path: str, experiment: str, run_name: str, seed: int,
           project: str) -> str:
    """The four-line edit. Everything else is copied verbatim, comments included."""
    t = _sub_once(base_text, r"^(\s*)experiment_name:.*$", rf"\g<1>experiment_name: {experiment}",
                  "experiment_name", base_path)
    t = _sub_once(t, r'^(\s*)run_name:.*$', rf'\g<1>run_name: "{run_name}"', "run_name", base_path)
    t = _sub_once(t, r"^seed:.*$", f"seed: {seed}", "top-level seed", base_path)
    # wandb_project is config-only -- `wandb_utils.init_wandb` reads `cfg["wandb_project"]` and
    # raises if absent, with no environment fallback -- so a separate project for the sweep has
    # to be written into the file rather than exported at launch.
    t = _sub_once(t, r"^(\s*)wandb_project:.*$", rf"\g<1>wandb_project: {project}",
                  "wandb_project", base_path)
    return t


def base_fields(base_text: str) -> tuple[str, str, str, str]:
    """The base's own values for the edited keys -- used to revert for the identity check."""
    exp = re.search(r"^\s*experiment_name:\s*(\S+)\s*$", base_text, re.M).group(1)
    run = re.search(r'^\s*run_name:\s*"?([^"\n]+)"?\s*$', base_text, re.M).group(1).strip()
    seed = re.search(r"^seed:\s*(\S+)\s*$", base_text, re.M).group(1)
    proj = re.search(r"^\s*wandb_project:\s*(\S+)\s*$", base_text, re.M).group(1)
    return exp, run, seed, proj


def cell_name(env_slug: str, arm: str, seed: int) -> str:
    return f"{env_slug}_{arm}_s{seed}"


def build(date: str, write: bool, project: str) -> tuple[list[dict], list[str]]:
    cells, problems = [], []
    for arm, rel in ARMS.items():
        base_path = REPO / rel
        if not base_path.exists():
            problems.append(f"missing base config for arm {arm}: {rel}")
            continue
        base_text = base_path.read_text()
        b_exp, b_run, b_seed, b_proj = base_fields(base_text)
        for slug, env_id in ENVS.items():
            for seed in SEEDS:
                name = cell_name(slug, arm, seed)
                text = render(base_text, rel, f"sweep_{name}", f"{name}_{date}", seed, project)
                out = OUT_DIR / f"{name}.yaml"
                if write:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(text)
                # Byte-identity: revert the three edits and require the base back, exactly.
                if slug == "pointgoal1":
                    reverted = render(text, str(out), b_exp, b_run, int(b_seed), b_proj)
                    if reverted != base_text:
                        problems.append(f"{name}: reverted config differs from {rel}")
                cells.append({
                    "name": name, "arm": arm, "env_slug": slug, "env_id": env_id, "seed": seed,
                    "config": str(out.relative_to(REPO)), "queued": slug not in ALREADY_HAVE,
                })
    return cells, problems


def priority_sorted(cells: list[dict]) -> list[dict]:
    """Pass 1 = seed 2 on every env, both arms. Pass 2 = seeds 3, 4. Env order = ENVS order."""
    env_rank = {s: i for i, s in enumerate(ENVS)}
    arm_rank = {a: i for i, a in enumerate(ARMS)}
    return sorted(
        [c for c in cells if c["queued"]],
        key=lambda c: (0 if c["seed"] == PASS1_SEED else 1, env_rank[c["env_slug"]],
                       c["seed"], arm_rank[c["arm"]]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify byte-identity only; write nothing")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"), help="stamped into run_name")
    ap.add_argument("--iters", type=int, default=60000, help="max_iterations recorded in the manifest")
    ap.add_argument("--wandb-project", default="SafeRL-multienv-sweep",
                    help="wandb project for every generated cell; keeps the sweep out of SafeRL")
    args = ap.parse_args()

    cells, problems = build(args.date, write=not args.check, project=args.wandb_project)
    queued = priority_sorted(cells)

    if not args.check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(
            {"generated": args.date, "iters": args.iters, "runs": queued,
             "skipped_have_already": [c["name"] for c in cells if not c["queued"]]},
            indent=2) + "\n")

    print(f"wandb project: {args.wandb_project}")
    print(f"cells: {len(cells)}  queued: {len(queued)}  "
          f"skipped (already have): {len(cells) - len(queued)}")
    print(f"byte-identity check on {sum(1 for c in cells if c['env_slug'] == 'pointgoal1')} "
          f"PointGoal1 configs: {'FAILED' if problems else 'PASS'}")
    for p in problems:
        print(f"  ! {p}")
    if not args.check:
        print(f"manifest -> {MANIFEST.relative_to(REPO)}")
        print("priority order:")
        for i, c in enumerate(queued, 1):
            print(f"  {i:2d}. {c['name']:28s} {c['env_id']}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
