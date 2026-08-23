#!/usr/bin/env python
"""Cross-env sweep report: per (env, arm) across seeds, from the sweep's checkpoints.

Metric definitions are NOT redefined here -- `evaluate_checkpoint` is imported from
`boot_ema_report.py`, so pred/real CVaR_0.9, tail error, VaR coverage, level ratio, PIT-KS,
mean cost, violation rate and mean reward are byte-for-byte the same quantities as the
`c2_vs_baseline.json` tables. Reward is the "last-3 mean": the mean over the last three
checkpoints of a run, which is what the c2/c3 comparison used to damp end-of-run seed noise.

Built to run against a PARTIALLY finished sweep: cells with no run dir, no checkpoints, or fewer
than three checkpoints are reported as such and excluded from the aggregate rather than crashing
or being silently imputed. A cell's absence is itself a result -- `coverage` in the output says
how much of the grid each number rests on.

One env is evaluated per process invocation (the env twin is built once), so this loops envs
internally and rebuilds the runner per env -- obs width differs across envs (60/72/76 + 1 horizon
column), so checkpoints are NOT portable between them.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/analysis/sweep_report.py \
        --manifest config/sweep/manifest.json --episodes 20
    python scripts/analysis/sweep_report.py --dry-run     # inventory only, no GPU, no rollouts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

METRICS = ("mean_reward", "mean_cost", "violation_rate", "pred_cvar", "real_cvar",
           "cvar_tail_error", "var_coverage", "level_ratio", "pit_ks")
LAST_N = 3  # "last-3 mean", as in the c2 vs baseline comparison


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/sweep/manifest.json")
    ap.add_argument("--out_json", default="logs/sweep/sweep_results.json")
    ap.add_argument("--out_dir", default="logs/sweep/report")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cost_limits", default="25.0")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--last_n", type=int, default=LAST_N)
    ap.add_argument("--dry-run", action="store_true", help="inventory the grid; no rollouts")
    return ap.parse_args()


def run_dir_for(env_id: str, cell: str) -> Path | None:
    root = REPO / "logs" / "safety_gymnasium" / env_id / "FHDCMPO"
    if not root.is_dir():
        return None
    c = [d for d in root.iterdir() if (d / ".sweep_cell").is_file()
         and (d / ".sweep_cell").read_text().strip() == cell]
    return max(c, key=lambda d: d.stat().st_mtime) if c else None


def checkpoints(d: Path) -> list[tuple[int, Path]]:
    out = []
    for p in d.glob("model_*.pt"):
        m = re.match(r"model_(\d+)\.pt$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def agg(vals: list[float]) -> dict | None:
    """Median + min/max across seeds. None when no seed produced the metric."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v)), "n_seeds": len(v)}


def fmt(a: dict | None, prec: int = 2) -> str:
    if a is None:
        return "--"
    return f"{a['median']:.{prec}f} [{a['min']:.{prec}f}, {a['max']:.{prec}f}]"


def main() -> None:
    args = parse_args()
    man = json.loads((REPO / args.manifest).read_text())
    cells = man["runs"]

    # --- inventory: what actually exists on disk right now -------------------------------
    inventory = []
    for c in cells:
        d = run_dir_for(c["env_id"], c["name"])
        cks = checkpoints(d) if d else []
        inventory.append({**c, "run_dir": str(d.relative_to(REPO)) if d else None,
                          "n_ckpts": len(cks), "last_iter": cks[-1][0] if cks else -1,
                          "usable": len(cks) >= 1})
    n_usable = sum(1 for i in inventory if i["usable"])
    print(f"grid: {len(cells)} cells, {n_usable} with >=1 checkpoint "
          f"({len(cells) - n_usable} not started or no checkpoint yet)")

    if args.dry_run:
        for i in inventory:
            print(f"  {'OK ' if i['usable'] else '-- '} {i['name']:26s} "
                  f"ckpts={i['n_ckpts']:2d} last_iter={i['last_iter']}")
        return

    # Imported late so --dry-run needs no torch/GPU.
    import torch
    import yaml
    from boot_ema_report import evaluate_checkpoint
    from eval_safety_gymnasium import load_train_cfg, obs_shaping_env_kwargs, parse_cost_limits

    from safe_rl.envs import make_env
    from safe_rl.runners import OffPolicyRunner

    per_cell: dict[str, dict] = {}
    by_env: dict[str, list] = {}
    for i in inventory:
        by_env.setdefault(i["env_id"], []).append(i)

    for env_id, items in by_env.items():
        live = [i for i in items if i["usable"]]
        if not live:
            print(f"[{env_id}] nothing to evaluate yet; skipping")
            continue
        cfg_path = REPO / live[0]["config"]
        torch.manual_seed(args.seed)
        train_cfg = load_train_cfg(str(cfg_path))
        env_cfg = (yaml.safe_load(cfg_path.read_text()) or {}).get("env", {}) or {}
        env_kwargs = {"device": args.device, "cost_limits": parse_cost_limits(args.cost_limits),
                      "seed": args.seed}
        env_kwargs.update(obs_shaping_env_kwargs(env_cfg, str(cfg_path)))
        env = make_env(env_id=env_id, num_envs=args.num_envs, **env_kwargs)
        train_cfg.setdefault("runner", {})["max_size"] = 1024
        runner = OffPolicyRunner(env, train_cfg, log_dir=None, device=args.device)

        for i in live:
            cks = checkpoints(REPO / i["run_dir"])[-args.last_n:]
            rows = []
            for it, ck in cks:
                try:
                    rows.append(evaluate_checkpoint(env, runner, ck, args))
                except Exception as e:  # a truncated checkpoint must not sink the whole report
                    print(f"  ! {i['name']} iter {it}: {type(e).__name__}: {e}")
            if not rows:
                continue
            # last-N mean per metric
            cell = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
            cell.update({"n_ckpts_used": len(rows), "iters_used": [it for it, _ in cks],
                         "last_iter": i["last_iter"], "partial": i["last_iter"] < man.get("iters", 0) - 1})
            per_cell[i["name"]] = cell
            print(f"  {i['name']:26s} rew {cell['mean_reward']:6.1f}  cost {cell['mean_cost']:5.1f}  "
                  f"viol {cell['violation_rate']:.2f}  CVaR {cell['pred_cvar']:5.1f}/"
                  f"{cell['real_cvar']:5.1f} (err {cell['cvar_tail_error']:+5.1f})")
        env.close()

    # --- aggregate across seeds -----------------------------------------------------------
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        if c["name"] in per_cell:
            groups.setdefault((c["env_id"], c["arm"]), []).append(per_cell[c["name"]])

    summary = {}
    for (env_id, arm), rows in sorted(groups.items()):
        summary[f"{env_id}|{arm}"] = {
            "n_seeds": len(rows),
            "any_partial": any(r["partial"] for r in rows),
            **{m: agg([r[m] for r in rows]) for m in METRICS},
        }

    out = {"manifest": args.manifest, "episodes": args.episodes, "last_n": args.last_n,
           "grid_cells": len(cells), "cells_evaluated": len(per_cell),
           "coverage": round(len(per_cell) / max(len(cells), 1), 3),
           "per_cell": per_cell, "per_env_arm": summary,
           "inventory": [{k: i[k] for k in ("name", "run_dir", "n_ckpts", "last_iter", "usable")}
                         for i in inventory]}
    op = REPO / args.out_json
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, default=float) + "\n")

    write_markdown(REPO / args.out_dir, summary, out, args)

    print(f"\n-> {op.relative_to(REPO)}")
    print(f"-> {(REPO / args.out_dir).relative_to(REPO)}/SUMMARY.md")
    print(f"coverage {len(per_cell)}/{len(cells)} cells")


def write_markdown(od: Path, summary: dict, out: dict, args) -> None:
    """One table per env plus a cross-env summary. Median [min, max] across seeds."""
    od.mkdir(parents=True, exist_ok=True)
    hdr = ("| arm | seeds | reward | cost | violation | CVaR pred | CVaR real | tail err | "
           "coverage | level |\n|---|---|---|---|---|---|---|---|---|---|\n")

    def row(arm: str, s: dict) -> str:
        star = " *" if s["any_partial"] else ""
        return (f"| {arm}{star} | {s['n_seeds']} | {fmt(s['mean_reward'], 1)} | "
                f"{fmt(s['mean_cost'], 1)} | {fmt(s['violation_rate'], 3)} | "
                f"{fmt(s['pred_cvar'], 1)} | {fmt(s['real_cvar'], 1)} | "
                f"{fmt(s['cvar_tail_error'], 1)} | {fmt(s['var_coverage'], 3)} | "
                f"{fmt(s['level_ratio'], 2)} |\n")

    envs = sorted({k.split("|")[0] for k in summary})
    for env_id in envs:
        md = [f"# {env_id}\n", f"\nMedian [min, max] across seeds; reward is the last-{args.last_n} "
              f"checkpoint mean, {args.episodes} stochastic episodes each. `*` = at least one seed "
              f"is still short of the full iteration budget. `level` is pred/realized mean cost "
              f"and blows up when a policy incurs ~no cost, so read it only once a run is trained."
              f"\n\n", hdr]
        for arm in sorted(a for e, a in ((k.split("|")[0], k.split("|")[1]) for k in summary) if e == env_id):
            md.append(row(arm, summary[f"{env_id}|{arm}"]))
        (od / f"{env_id}.md").write_text("".join(md))

    lines = ["# Cross-env sweep summary\n",
             f"\nCoverage: {out['cells_evaluated']}/{out['grid_cells']} cells evaluated "
             f"({out['coverage'] * 100:.0f}%). `*` = a seed is still short of budget.\n\n",
             "| env " + hdr.split("\n")[0] + "\n|---" + hdr.split("\n")[1] + "\n"]
    for env_id in envs:
        for arm in sorted(a for e, a in ((k.split("|")[0], k.split("|")[1]) for k in summary) if e == env_id):
            lines.append(f"| {env_id} " + row(arm, summary[f"{env_id}|{arm}"]))
    (od / "SUMMARY.md").write_text("".join(lines))
    print(f"-> wrote {len(envs)} per-env tables + SUMMARY.md")


if __name__ == "__main__":
    main()
