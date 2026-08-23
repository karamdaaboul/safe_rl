#!/usr/bin/env python
"""Checkpoint calibration report for the EMA-bootstrap experiment (and its baseline).

For every ``model_<iter>.pt`` in each given run dir: roll N stochastic episodes (the critic's
training distribution), then report the full tail-calibration panel against realized cost-to-go:

    pred/real CVaR_0.9, CVaR tail error, VaR_0.9 coverage (target 0.90),
    level pred/real (target 1.0), PIT-KS, mean episodic cost, violation rate (J_c > limit),
    mean episodic reward.

``boot_kl_mean`` (KL of pi_current from pi_boot, the lag-bias check) is logged during training
by FHDCMPO itself; read it off wandb/tensorboard rather than here (checkpoints do not carry
pi_boot).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/analysis/boot_ema_report.py \
        --config config/safety_gymnasium_fhdcmpo_b1_bootema.yaml \
        --run_dirs logs/safety_gymnasium/SafetyPointGoal1-v0/FHDCMPO/<baseline> \
                   logs/safety_gymnasium/SafetyPointGoal1-v0/FHDCMPO/<b1> ... \
        --labels baseline b1_tau0.005 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from eval_safety_gymnasium import load_train_cfg, obs_shaping_env_kwargs, parse_cost_limits  # noqa: E402

from safe_rl.common.fh_cost import pit_from_quantiles, quantile_cvar, quantile_var  # noqa: E402
from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OffPolicyRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="any arm config (nets/env are identical across arms)")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--env_id", default="SafetyPointGoal1-v0")
    ap.add_argument("--cost_limits", default="25.0")
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--checkpoints", default="", help="comma-separated iters; empty = all")
    ap.add_argument("--out", default="logs/opd_probe/boot_ema_report.json")
    return ap.parse_args()


def ks_uniform(u: np.ndarray) -> float:
    s = np.sort(u)
    return float(np.max(np.abs(s - (np.arange(1, s.size + 1) / s.size)))) if s.size else float("nan")


@torch.inference_mode()
def evaluate_checkpoint(env, runner, ckpt: Path, args) -> dict:
    """Roll episodes with the checkpoint's stochastic policy; measure the full panel."""
    state = torch.load(ckpt, weights_only=False, map_location=runner.device)
    pol = runner.actor_critic
    pol.load_state_dict(state["model_state_dict"], strict=False)
    pol.eval()
    critic = pol.cost_critics[0]

    n = env.num_envs
    obs, _ = env.reset()
    obs = obs.to(args.device)
    open_theta = [[] for _ in range(n)]
    open_prior = [[] for _ in range(n)]
    run_cost = np.zeros(n)
    run_rew = np.zeros(n)
    step_in_ep = np.zeros(n, dtype=int)
    thetas, realized, ep_costs, ep_rews = [], [], [], []

    while len(ep_costs) < args.episodes:
        acts, _ = pol.sample_with_log_prob(obs)
        record = (step_in_ep % args.stride) == 0
        if record.any():
            theta = critic(pol.critic_obs_normalizer(obs), acts)
        nxt, rews, dones, infos = env.step(acts)
        costs = infos.get("costs", torch.zeros(n, device=args.device)).reshape(n, -1).sum(-1).cpu().numpy()
        for e in range(n):
            if record[e]:
                open_theta[e].append(theta[e].float().cpu().clone())
                open_prior[e].append(float(run_cost[e]))
        run_cost += costs
        run_rew += rews.reshape(-1).cpu().numpy()
        step_in_ep += 1
        for e in (dones > 0).nonzero(as_tuple=False).reshape(-1).cpu().tolist():
            for th, prior in zip(open_theta[e], open_prior[e]):
                thetas.append(th)
                realized.append(run_cost[e] - prior)
            ep_costs.append(float(run_cost[e]))
            ep_rews.append(float(run_rew[e]))
            open_theta[e], open_prior[e] = [], []
            run_cost[e] = run_rew[e] = 0.0
            step_in_ep[e] = 0
        obs = nxt.to(args.device)

    th = torch.stack(thetas).sort(dim=-1).values
    rz = torch.tensor(np.asarray(realized, dtype=np.float32))
    a = args.alpha
    limit = parse_cost_limits(args.cost_limits)[0]
    pred_cvar = float(quantile_cvar(th.reshape(-1).sort().values, a))
    real_cvar = float(quantile_cvar(rz.sort().values, a))
    return {
        "pred_cvar": pred_cvar,
        "real_cvar": real_cvar,
        "cvar_tail_error": pred_cvar - real_cvar,
        "var_coverage": float((rz <= quantile_var(th, a)).float().mean()),
        "level_ratio": float(th.mean()) / max(float(rz.mean()), 1e-9),
        "pit_ks": ks_uniform(pit_from_quantiles(th, rz).numpy()),
        "mean_cost": float(np.mean(ep_costs)),
        "violation_rate": float(np.mean(np.asarray(ep_costs) > limit)),
        "mean_reward": float(np.mean(ep_rews)),
        "n_states": int(th.shape[0]),
        "n_episodes": len(ep_costs),
    }


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) != len(args.labels):
        raise SystemExit("--run_dirs and --labels must pair up")
    torch.manual_seed(args.seed)

    train_cfg = load_train_cfg(args.config)
    with open(args.config, encoding="utf-8") as fh:
        env_cfg = (yaml.safe_load(fh) or {}).get("env", {}) or {}
    env_kwargs = {"device": args.device, "cost_limits": parse_cost_limits(args.cost_limits), "seed": args.seed}
    env_kwargs.update(obs_shaping_env_kwargs(env_cfg, args.config))
    env = make_env(env_id=args.env_id, num_envs=args.num_envs, **env_kwargs)
    train_cfg.setdefault("runner", {})["max_size"] = 1024
    runner = OffPolicyRunner(env, train_cfg, log_dir=None, device=args.device)

    wanted = {int(x) for x in args.checkpoints.split(",") if x.strip()}
    results: dict = {}
    for run_dir, label in zip(args.run_dirs, args.labels):
        cks = sorted((int(p.stem.split("_")[1]), p) for p in Path(run_dir).glob("model_*.pt"))
        if wanted:
            cks = [(i, p) for i, p in cks if i in wanted]
        results[label] = {}
        for it, ck in cks:
            row = evaluate_checkpoint(env, runner, ck, args)
            results[label][it] = row
            print(
                f"{label:16s} iter {it:6d}: CVaR {row['pred_cvar']:6.1f}/{row['real_cvar']:6.1f} "
                f"(err {row['cvar_tail_error']:+6.1f})  cov {row['var_coverage']:.3f}  "
                f"level {row['level_ratio']:.2f}  KS {row['pit_ks']:.3f}  "
                f"cost {row['mean_cost']:5.1f}  viol {row['violation_rate']:.2f}  rew {row['mean_reward']:5.1f}"
            )
    env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
