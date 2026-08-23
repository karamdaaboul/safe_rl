"""Sampled Q-greedy evaluation for REPPO — does the critic know better actions than the mode?

Training is untouched. At every control step we build a candidate set

    {tanh(mu)} u {N samples from pi(.|s)}

score every candidate with the learned critic, act on argmax_a Q(s,a), and record
both what the critic *predicted* the improvement would be and what actually
happened. The decisive plot is predicted-vs-realized: a critic that predicts large
gains and delivers none is miscalibrated, and no amount of actor-side work fixes
that.

Reports per N: predicted Q improvement, realized return, tracking error, fall rate,
action discontinuity, per-step latency, fraction of steps where the argmax differed
from the mode, and tracking error bucketed by commanded speed.

Usage:
  python scripts/eval/q_greedy_probe.py --checkpoint <ckpt> --config <yaml> \
      --env_id Unitree-Go2-Flat --n_list 0,1,4,16,64,256 --seeds 3,7,11,21,33 \
      --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _add_unitree_repo_to_path() -> None:
    explicit = os.environ.get("UNITREE_RL_MJLAB_PATH")
    for candidate in ([explicit] if explicit else []) + [
        "/opt/unitree_rl_mjlab",
        str(Path.home() / "workspaces" / "unitree_rl_mjlab"),
        str(Path(__file__).resolve().parents[3] / "unitree_rl_mjlab"),
    ]:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
            break


_add_unitree_repo_to_path()

import torch  # noqa: E402
import yaml  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
import mjlab.tasks  # noqa: E402,F401

try:
    import src.tasks  # noqa: E402,F401
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] unitree task registration failed: {type(exc).__name__}: {exc}", file=sys.stderr)

from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OnPolicyRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", default="Unitree-Go2-Flat")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--n_list",
        default="0,1,4,16,64,256",
        help=(
            "Candidate counts. 0 = deterministic mode (the deployment baseline). "
            "-1 = ONE stochastic sample with NO Q selection -- the control that "
            "separates 'the critic picks a better action' from 'anything but the "
            "mode is better'. Without it, N=1 is ambiguous: {mode, 1 sample} -> "
            "argmax Q looks like a critic win even if the critic is uninformative."
        ),
    )
    p.add_argument("--seeds", default="3,7,11,21,33")
    p.add_argument("--cmd_script", default=None, help="Same format as the evaluator's --cmd_script.")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--selection", default="single", choices=["single", "conservative"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def build(args: argparse.Namespace, seed: int):
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("algorithm", {}).setdefault("class_name", "PPO")
    cfg.setdefault("policy", {}).setdefault("class_name", "ActorCritic")
    if "runner" in cfg and "num_steps_per_env" not in cfg:
        cfg.update(cfg.pop("runner"))

    env_cfg = load_env_cfg(args.env_id)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    from src.envs import build_env

    env = build_env(env_cfg, args.device, render_mode=None)
    agent_cfg = load_rl_cfg(args.env_id)
    vec_env = make_env(env_id=args.env_id, env=env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(vec_env, cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    return env, vec_env, runner


def parse_script(spec: str) -> torch.Tensor:
    table = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        steps, vec = chunk.split(":")
        v = [float(x) for x in vec.split(",")]
        table.extend([v] * int(steps))
    return torch.tensor(table, dtype=torch.float32)


def rollout(args, seed: int, n_samples: int, script: torch.Tensor | None) -> dict[str, Any]:
    env, vec_env, runner = build(args, seed)
    policy = runner.alg.policy
    dev = runner.device
    scale = float(getattr(policy, "action_scale", 1.0))
    squashed = getattr(policy, "squash", "none") == "tanh"

    base_env = getattr(vec_env, "env", vec_env)
    base_env = getattr(base_env, "unwrapped", base_env)
    cmd_mgr = getattr(base_env, "command_manager", None)
    term = next((t for t in ("twist", "base_velocity") if cmd_mgr and t in cmd_mgr.active_terms), None)
    robot = base_env.scene["robot"]
    cmd_obj = cmd_mgr._terms[term] if (term and script is not None) else None

    # REPPO is single-critic (as the reference is), so there is no conservative
    # min(Q1,Q2) variant to select.
    twin = False
    conservative = False

    obs, extras = vec_env.get_observations()
    obs = obs.to(dev)
    critic_obs = extras.get("observations", {}).get("critic", obs).to(dev)

    ret = 0.0
    steps = 0
    fell = False
    prev_action = None
    dq_pred, disc, lat, differ = [], [], [], []
    err_by_speed: list[tuple[float, float]] = []
    track_err = 0.0
    track_yaw = 0.0

    with torch.inference_mode():
        while steps < args.max_steps:
            if cmd_obj is not None:
                row = script[min(steps, len(script) - 1)].to(cmd_obj.vel_command_b.device)
                cmd_obj.vel_command_b[:] = row
                obs_d = base_env.observation_manager.compute()
                obs = obs_d.get("actor", obs_d.get("policy")).to(dev)
                critic_obs = obs_d.get("critic", obs).to(dev)

            t0 = time.perf_counter()
            norm_obs = policy.actor_obs_normalizer(obs)
            dist = policy._build_distribution(norm_obs)
            mode = torch.tanh(dist.mean) * scale if squashed else dist.mean
            if n_samples < 0:
                # Control: a single stochastic sample, critic never consulted.
                raw = dist.sample()
                action = torch.tanh(raw) * scale if squashed else raw
                dq_pred.append(0.0)
                differ.append(1.0)
            elif n_samples > 0:
                raw = dist.sample((n_samples,))
                cand = torch.tanh(raw) * scale if squashed else raw
                cand = torch.cat([mode.unsqueeze(0), cand], dim=0)  # index 0 == mode
                k = cand.shape[0]
                flat_obs = critic_obs.expand(k, -1)
                q = policy.evaluate_q(flat_obs, cand.reshape(k, -1)).reshape(k)
                best = int(q.argmax().item())
                action = cand[best]
                dq_pred.append(float(q[best].item() - q[0].item()))
                differ.append(1.0 if best != 0 else 0.0)
            else:
                action = mode
                dq_pred.append(0.0)
                differ.append(0.0)
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)

            if prev_action is not None:
                disc.append(float((action - prev_action).abs().mean().item()))
            prev_action = action.clone()

            obs, rew, dones, infos = vec_env.step(action)
            obs = obs.to(dev)
            critic_obs = infos.get("observations", {}).get("critic", obs).to(dev)
            ret += float(rew.sum().item())
            steps += 1

            if term is not None:
                c = cmd_mgr.get_command(term)[0]
                lin = robot.data.root_link_lin_vel_b[0]
                ang = robot.data.root_link_ang_vel_b[0]
                e = float(torch.norm(c[:2] - lin[:2]).item())
                track_err += e
                track_yaw += float(torch.abs(c[2] - ang[2]).item())
                err_by_speed.append((float(torch.norm(c[:2]).item()), e))

            if bool((dones > 0).any().item()):
                fell = steps < args.max_steps
                break

    vec_env.close()
    buckets: dict[str, float] = {}
    if err_by_speed:
        arr = np.array(err_by_speed)
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9.9)):
            sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if sel.sum() >= 5:
                buckets[f"{lo}-{hi}"] = round(float(arr[sel, 1].mean()), 4)
    return {
        "seed": seed,
        "n": n_samples,
        "steps": steps,
        "fell": fell,
        "return": round(ret, 3),
        "track_err": round(track_err / max(steps, 1), 4),
        "track_yaw": round(track_yaw / max(steps, 1), 4),
        "dq_pred": round(float(np.mean(dq_pred)), 4),
        "discontinuity": round(float(np.mean(disc)) if disc else 0.0, 4),
        "latency_ms": round(float(np.mean(lat)), 3),
        "frac_differ": round(float(np.mean(differ)), 4),
        "err_by_speed": buckets,
        "twin_critics": twin,
        "conservative": conservative,
    }


def main() -> None:
    args = parse_args()
    script = parse_script(args.cmd_script) if args.cmd_script else None
    n_list = [int(x) for x in args.n_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for n in n_list:
        for seed in seeds:
            r = rollout(args, seed, n, script)
            rows.append(r)
            print(f"N={n:<4} seed={seed:<3} steps={r['steps']:<5} fell={int(r['fell'])} "
                  f"ret={r['return']:<9.2f} track={r['track_err']:.4f} dQ={r['dq_pred']:+.4f} "
                  f"differ={r['frac_differ']:.3f} disc={r['discontinuity']:.4f} lat={r['latency_ms']:.2f}ms",
                  flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
