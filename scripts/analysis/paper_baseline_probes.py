#!/usr/bin/env python
"""Cost-critic quality + deterministic evaluation for the paper's baseline comparison.

Two modes, one metric panel, so FSRL's reference CVPO (off-policy, discounted scalar Q_c),
our PPOL_PID (on-policy, discounted scalar V_C) and FH-DCMPO (off-policy, undiscounted
quantile Z_c; measured separately by boot_ema_report.py) are directly comparable:

* DETERMINISTIC EVAL (deployment): N greedy episodes -> reward, episodic cost, cost p90,
  violation rate (episodic cost > limit).
* CRITIC CALIBRATION (training distribution): N stochastic episodes -> critic prediction at
  strided (s, a) vs realized cost-to-go IN THE CRITIC'S OWN UNITS (discounted by the run's
  own gamma for FSRL/PPOL) -> level ratio pred/real, plus p90 pred/real as the scalar-critic
  tail proxy.

Modes:
  --mode fsrl --run_dir <fsrl run dir with config.yaml + checkpoint/model.pt>
      (run under /home/human/venvs/fsrl_m0/bin/python, cwd anywhere; imports fsrl_m0_src)
  --mode ppol --checkpoint <model_*.pt> --config config/safety_gymnasium_ppol_pid.yaml
      (run under the agx_plain venv)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def discounted_ctg(costs: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(costs))
    acc = 0.0
    for t in range(len(costs) - 1, -1, -1):
        acc = costs[t] + gamma * acc
        out[t] = acc
    return out


def panel(preds: np.ndarray, reals: np.ndarray, det_rews, det_costs, limit: float) -> dict:
    det_costs = np.asarray(det_costs)
    return {
        "det_reward": float(np.mean(det_rews)),
        "det_cost": float(det_costs.mean()),
        "det_cost_p90": float(np.quantile(det_costs, 0.9)),
        "det_violation_rate": float((det_costs > limit).mean()),
        "critic_pred_mean": float(preds.mean()),
        "critic_real_mean": float(reals.mean()),
        "critic_level": float(preds.mean() / max(reals.mean(), 1e-9)),
        "critic_p90_pred": float(np.quantile(preds, 0.9)),
        "critic_p90_real": float(np.quantile(reals, 0.9)),
        "critic_p90_ratio": float(np.quantile(preds, 0.9) / max(np.quantile(reals, 0.9), 1e-9)),
        "n_states": int(len(preds)),
        "n_det_episodes": len(det_costs),
    }


# ---------------------------------------------------------------------------------- FSRL mode
def probe_fsrl(run_dir: str, episodes: int, stride: int) -> dict:
    sys.path.insert(0, "/home/human/workspaces/fsrl_m0_src")
    import gymnasium as gym
    import torch

    try:
        import safety_gymnasium  # noqa: F401
    except ImportError:
        pass
    import bullet_safety_gym  # noqa: F401
    from fsrl.agent import CVPOAgent
    from fsrl.utils import BaseLogger
    from fsrl.utils.exp_util import load_config_and_model
    from tianshou.data import Batch

    cfg, model = load_config_and_model(run_dir)
    gamma, limit = float(cfg["gamma"]), float(cfg["cost_limit"])
    env = gym.make(cfg["task"])
    agent = CVPOAgent(
        env=env, logger=BaseLogger(), device="cpu", thread=cfg["thread"], seed=cfg["seed"],
        hidden_sizes=cfg["hidden_sizes"], unbounded=cfg["unbounded"],
        last_layer_scale=cfg.get("last_layer_scale", False),
    )
    agent.policy.load_state_dict(model["model"])
    policy = agent.policy
    policy.eval()

    def rollout(n_eps: int, deterministic: bool):
        obs_l, act_l, cost_l, bounds, rews = [], [], [], [], []
        for ep in range(n_eps):
            obs, _ = env.reset(seed=5000 + ep)
            done = trunc = False
            r_sum = 0.0
            start = len(cost_l)
            while not (done or trunc):
                with torch.no_grad():
                    # tianshou ActorProb: eval-mode forward returns the mode when
                    # deterministic_eval; force explicitly via the dist's mode/sample.
                    out = policy(Batch(obs=np.asarray(obs)[None], info={}))
                    raw = out.dist.mode if deterministic else out.act
                    raw = raw.cpu().numpy()[0]
                    act = policy.map_action(raw)
                nobs, rew, done, trunc, info = env.step(act)
                obs_l.append(np.asarray(obs, np.float32))
                act_l.append(np.asarray(raw, np.float32))
                cost_l.append(float(info.get("cost", 0.0)))
                r_sum += float(rew)
                obs = nobs
            bounds.append((start, len(cost_l)))
            rews.append(r_sum)
        return obs_l, act_l, cost_l, bounds, rews

    # Deterministic panel
    _, _, dcost, dbounds, drews = rollout(episodes, deterministic=True)
    det_costs = [sum(dcost[s:e]) for s, e in dbounds]

    # Calibration on the stochastic (training) distribution
    o, a, c, bounds, _ = rollout(episodes, deterministic=False)
    preds, reals = [], []
    for s, e in bounds:
        ctg = discounted_ctg(c[s:e], gamma)
        idx = list(range(0, e - s, stride))
        with torch.no_grad():
            qc, _ = policy.critics[1].predict(
                torch.as_tensor(np.stack([o[s + i] for i in idx])),
                torch.as_tensor(np.stack([a[s + i] for i in idx])),
            )
        preds.extend(qc.reshape(-1).tolist())
        reals.extend(ctg[idx].tolist())
    out = panel(np.asarray(preds), np.asarray(reals), drews, det_costs, limit)
    out.update({"mode": "fsrl_cvpo", "gamma": gamma, "run_dir": run_dir})
    return out


# ---------------------------------------------------------------------------------- PPOL mode
def probe_ppol(checkpoint: str, config: str, episodes: int, stride: int, device: str) -> dict:
    import torch
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
    from eval_safety_gymnasium import load_train_cfg, parse_cost_limits  # noqa: E402

    from safe_rl.envs import make_env
    from safe_rl.runners import OnPolicyRunner

    train_cfg = load_train_cfg(config)
    with open(config, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    gamma = float(raw["algorithm"].get("gamma", 0.99))
    limit = 25.0
    env = make_env(env_id="SafetyPointGoal1-v0", num_envs=1, device=device,
                   cost_limits=parse_cost_limits("25.0"), seed=6000)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(checkpoint, load_optimizer=False)
    pol = runner.alg.policy
    pol.eval()
    obs_norm = runner.obs_normalizer if runner.empirical_normalization else torch.nn.Identity()

    def rollout(n_eps: int, deterministic: bool):
        obs_l, cost_l, bounds, rews = [], [], [], []
        obs, _ = env.reset()
        done_eps, r_sum, start = 0, 0.0, 0
        while done_eps < n_eps:
            with torch.no_grad():
                o_n = obs_norm(obs.to(device))
                act = pol.act_inference(o_n) if deterministic else pol.act(o_n)
            nobs, rew, dones, infos = env.step(act)
            obs_l.append(obs[0].cpu().numpy())
            cost_l.append(float(infos["costs"].reshape(-1)[0]))
            r_sum += float(rew.reshape(-1)[0])
            obs = nobs
            if float(dones.reshape(-1)[0]) > 0:
                bounds.append((start, len(cost_l)))
                rews.append(r_sum)
                start, r_sum = len(cost_l), 0.0
                done_eps += 1
        return obs_l, cost_l, bounds, rews

    _, dcost, dbounds, drews = rollout(episodes, deterministic=True)
    det_costs = [sum(dcost[s:e]) for s, e in dbounds]

    o, c, bounds, _ = rollout(episodes, deterministic=False)
    preds, reals = [], []
    for s, e in bounds:
        ctg = discounted_ctg(c[s:e], gamma)
        idx = list(range(0, e - s, stride))
        with torch.no_grad():
            obs_t = torch.as_tensor(np.stack([o[s + i] for i in idx])).to(device)
            v_c = pol.evaluate_cost(obs_norm(obs_t))  # state-only V_C(s), [n, num_costs]
        preds.extend(v_c[:, 0].cpu().reshape(-1).tolist())
        reals.extend(ctg[idx].tolist())
    env.close()
    out = panel(np.asarray(preds), np.asarray(reals), drews, det_costs, limit)
    out.update({"mode": "ppol_pid", "gamma": gamma, "checkpoint": checkpoint})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fsrl", "ppol"], required=True)
    ap.add_argument("--run_dir", default=None, help="fsrl mode: run dir with config.yaml")
    ap.add_argument("--checkpoint", default=None, help="ppol mode: model_*.pt")
    ap.add_argument("--config", default="config/safety_gymnasium_ppol_pid.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "fsrl":
        res = probe_fsrl(args.run_dir, args.episodes, args.stride)
    else:
        res = probe_ppol(args.checkpoint, args.config, args.episodes, args.stride, args.device)
    for k, v in res.items():
        print(f"{k:22s} {v}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
