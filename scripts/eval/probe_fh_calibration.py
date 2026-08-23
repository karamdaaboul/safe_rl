#!/usr/bin/env python
"""Measure Delta_critic for an FH-DCMPO checkpoint: is the finite-horizon cost critic calibrated?

This is the instrument the S1 gate is judged on, and it exists because the finite-horizon
formulation makes the ground truth *directly measurable*. The critic predicts the distribution of
the undiscounted cost still to come,

    Z_c(s_t, u_t, a_t) ~ sum_{t'=t}^{T-1} c_t' ,

and since Safety-Gymnasium episodes run a fixed T steps, the realized value of that quantity for
every visited state is known once the episode ends:

    realized(t) = (episode total cost) - (cost incurred before t).

No discounting, no bootstrap, no proxy. Under the old discounted scheme this check was not available
at all: the target was `sum gamma^t c_t`, whose realization needs the whole discounted tail and whose
relation to the stated budget ran through a measured `qc_scale`.

What it reports, and why each number is a term in the bound of codex/fh-dcmpo-math.md section 6:

* **level**   predicted mean vs realized mean. The S1 gate. A critic whose level is wrong makes every
  downstream tail statistic wrong in the same direction, which is exactly what
  codex/cvpo-cost-critic-investigation.md found (level under-read by 2.2x, so lambda never engaged).
* **marginal PIT KS**  the calibration test. Judged on the MARGINAL -- the pooled histogram over all
  visited states -- never per state: codex/qr-dmpo-math.md 6.1 shows per-state coverage collapses
  toward 0.5 regardless of correctness when the conditional is near-deterministic, so a per-state
  test would reject a provably correct critic.
* **tail**    predicted vs realized CVaR_alpha, and the realized coverage of the predicted VaR_alpha.
  `CVaR_alpha` is 1/(1-alpha)-Lipschitz in the quantile function, so an error here is amplified by
  10x at alpha=0.9 -- this is the term that decides whether S3 recalibration is needed.

Usage:
    python scripts/eval/probe_fh_calibration.py \
        --env_id SafetyPointGoal1-v0 --config config/safety_gymnasium_fhdcmpo_goal1.yaml \
        --checkpoint logs/.../model_60000.pt --episodes 20
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
import torch
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safe_rl.common.fh_cost import pit_from_quantiles, quantile_cvar, quantile_var  # noqa: E402
from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OffPolicyRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_safety_gymnasium import load_train_cfg, obs_shaping_env_kwargs, parse_cost_limits  # noqa: E402


def ks_uniform(u: np.ndarray) -> float:
    """KS distance of PIT values from uniform. 0 = perfectly calibrated."""
    if u.size == 0:
        return float("nan")
    s = np.sort(u)
    return float(np.max(np.abs(s - (np.arange(1, s.size + 1) / s.size))))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cost_limits", default="25.0")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument(
        "--stride",
        type=int,
        default=25,
        help=(
            "record every Nth timestep; 1000-step episodes are highly autocorrelated, "
            "so every step would inflate the sample count without adding information"
        ),
    )
    ap.add_argument(
        "--policy",
        choices=["stochastic", "deterministic"],
        default="stochastic",
        help=(
            "WHICH POLICY THE ROLLOUTS USE, and it changes what the numbers mean. "
            "`stochastic` (default) matches the behaviour distribution the critic was "
            "trained on and that the E-step queries, so it measures the Delta_critic "
            "term of the bound. `deterministic` measures accuracy on the deployment "
            "distribution instead. Mixing them up manufactures a huge apparent "
            "miscalibration: on an early checkpoint the deterministic policy is often "
            "near-idle and realizes far less cost than the exploring policy the critic "
            "learned from, so pred/real reads ~6x when the critic may be fine."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_train_cfg(args.config)
    # Read the `env` block from the RAW yaml, not from train_cfg: load_train_cfg restructures the
    # config (it flattens/normalizes the runner section) and does not preserve `env`. Taking it from
    # train_cfg silently yields {}, the env is built without the horizon column, and the checkpoint
    # then fails to load with a 60-vs-61 size mismatch. eval_safety_gymnasium.py reads the raw yaml
    # for the same reason.
    with open(args.config, encoding="utf-8") as fh:
        env_cfg = (yaml.safe_load(fh) or {}).get("env", {}) or {}
    cost_limits = parse_cost_limits(args.cost_limits)

    env_kwargs = {"device": args.device, "cost_limits": cost_limits, "seed": args.seed}
    env_kwargs.update(obs_shaping_env_kwargs(env_cfg, args.config))
    env = make_env(env_id=args.env_id, num_envs=args.num_envs, **env_kwargs)

    train_cfg.setdefault("runner", {})["max_size"] = 1000  # probe never fills a buffer
    runner = OffPolicyRunner(env, train_cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.alg.policy
    if args.policy == "deterministic":
        policy_fn = runner.get_inference_policy(device=args.device)
    else:
        # Match the behaviour distribution the critic was trained on. `act(deterministic=False)`
        # samples from the actor, which is what fills the replay buffer and what the E-step draws
        # its candidate actions from.
        norm = runner.obs_normalizer if runner.empirical_normalization else torch.nn.Identity()
        runner.eval_mode()

        def policy_fn(x):
            return policy.act(norm(x), deterministic=False)

    critic = policy.cost_critics[0]
    if not getattr(policy, "is_quantile_cost_critic", False):
        raise SystemExit("probe expects a quantile cost critic (FH-DCMPO)")

    n_envs = env.num_envs
    obs, _ = env.reset()
    obs = obs.to(runner.device)

    # Per-env open trace: (theta prediction, cost accumulated before that step).
    open_theta: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
    open_prior: list[list[float]] = [[] for _ in range(n_envs)]
    running_cost = np.zeros(n_envs)
    step_in_ep = np.zeros(n_envs, dtype=int)

    thetas: list[np.ndarray] = []
    realized: list[float] = []
    episodes_done = 0

    while episodes_done < args.episodes:
        with torch.inference_mode():
            actions = policy_fn(obs)
            record = (step_in_ep % args.stride) == 0
            if record.any():
                theta = critic(runner.critic_obs_normalizer(obs), actions)  # [n_envs, N]
        for e in range(n_envs):
            if record[e]:
                open_theta[e].append(theta[e].detach().float().cpu().clone())
                open_prior[e].append(float(running_cost[e]))

        obs, _, dones, infos = env.step(actions)
        obs = obs.to(runner.device)
        costs = infos.get("costs", torch.zeros(n_envs, device=runner.device))
        costs = costs.to(runner.device).reshape(n_envs, -1).sum(-1).cpu().numpy()
        running_cost += costs
        step_in_ep += 1

        for e in (dones > 0).nonzero(as_tuple=False).reshape(-1).cpu().tolist():
            total = float(running_cost[e])
            for th, prior in zip(open_theta[e], open_prior[e]):
                thetas.append(th.numpy())
                realized.append(total - prior)  # undiscounted cost still to come at that step
            episodes_done += 1
            open_theta[e], open_prior[e] = [], []
            running_cost[e] = 0.0
            step_in_ep[e] = 0

    env.close()

    th = torch.tensor(np.stack(thetas))  # [M, N]
    rz = torch.tensor(np.asarray(realized, dtype=np.float32))  # [M]
    a = args.alpha

    pred_mean = float(th.mean(dim=-1).mean())
    real_mean = float(rz.mean())
    pit = pit_from_quantiles(th, rz).numpy()
    ks = ks_uniform(pit)
    # Marginal tail: pool the predicted distributions (law of total probability) and compare with the
    # realized marginal. This is the valid comparison; a per-state one is not (see the module docstring).
    pred_pool = th.reshape(-1)
    pred_cvar = float(quantile_cvar(pred_pool.sort().values, a))
    real_cvar = float(quantile_cvar(rz.sort().values, a))
    pred_var = float(quantile_var(pred_pool.sort().values, a))
    coverage = float((rz <= quantile_var(th, a)).float().mean())
    n_states = th.shape[0]
    crit = 1.36 / np.sqrt(n_states)  # KS 95% critical value

    print(f"\nFH cost-critic calibration probe -- {args.env_id}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  samples        {n_states} states from {episodes_done} episodes (stride {args.stride})")
    print(
        f"  rollout policy {args.policy}"
        f"{'  <- matches the critic training distribution' if args.policy == 'stochastic' else '  <- deployment distribution, NOT the training one'}"
    )
    print("\n  LEVEL   (the S1 gate: is the undiscounted critic on the right scale?)")
    print(f"    predicted mean cost-to-go   {pred_mean:8.2f}")
    print(f"    realized  mean cost-to-go   {real_mean:8.2f}")
    print(f"    ratio pred/real             {pred_mean / max(real_mean, 1e-9):8.3f}   (want ~1.0)")
    print("\n  MARGINAL CALIBRATION")
    print(f"    PIT KS vs uniform           {ks:8.3f}   (95% critical {crit:.3f}; lower is better)")
    print(f"    PIT deciles                 {np.round(np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit), 3)}")
    print(f"\n  TAIL    (Delta_critic is amplified by 1/(1-alpha) = {1 / (1 - a):.0f}x at alpha={a})")
    print(f"    predicted CVaR_{a}          {pred_cvar:8.2f}")
    print(f"    realized  CVaR_{a}          {real_cvar:8.2f}")
    print(f"    predicted VaR_{a}           {pred_var:8.2f}")
    print(f"    realized coverage of VaR    {coverage:8.3f}   (want {a})")
    verdict = (
        "UNDER-disperses the tail (understates cost -- unsafe direction)"
        if pred_cvar < real_cvar
        else "over-disperses the tail (overstates cost -- conservative)"
    )
    print(f"\n  verdict: critic {verdict}")
    print(
        f"           tail error {abs(pred_cvar - real_cvar):.2f}, i.e. up to "
        f"{abs(pred_cvar - real_cvar) / (1 - a):.1f} of slack in the bound's Delta_critic term\n"
    )


if __name__ == "__main__":
    main()
