#!/usr/bin/env python
"""Across-action spread of the cost signal inside the FH-DCMPO E-step.

Next suspect after the bootstrap-EMA arms (codex/offpolicy-mismatch-td-lambda.md): the E-step
compares candidate actions at one state, so the cost constraint can only bite through the
ACROSS-ACTION spread of ``rho_c(s, a)`` -- and through how it co-varies with ``Q_r(s, a)``. If
``rho_c`` is nearly flat across the sampled actions (the cost-to-go is mostly a function of the
state, not the action), then no lambda within reach can re-rank actions, the constraint is inert
at the E-step even when the critic's LEVEL and TAIL are perfectly calibrated, and further work on
the target (TD(lambda), replay age, bootstrap smoothing) cannot help.

Measured per checkpoint, on that checkpoint's own cached rollout states, with the E-step's exact
action proposal (actor Gaussian, ``sample_action_num`` draws) and exact readouts
(``min`` twin reward Q; quantile-mean rho at kappa=0, CVaR_0.9 at kappa=1):

* ``std_a(Q_r)``, ``std_a(rho_mean)``, ``std_a(rho_cvar)`` -- median per-state spreads;
* ``dose_cvar_vs_mean`` = median_s[std_a(rho_cvar)/std_a(rho_mean)] -- the factor by which
  ramping kappa to 1 multiplies the E-step's cost pressure at fixed lambda;
* ``balance`` = std_a(Q_r) / std_a(rho) -- the lambda needed for the cost term to match the
  reward term's influence; compare with the run's lambda_max (1.8);
* ``corr(Q_r, rho)`` across actions -- aligned signals cannot be re-ranked by a penalty;
* the direct test: solve the E-step temperature eta at lambda=0 (KL = dual_constraint), then
  measure the total-variation distance between the weights at lambda=0 and lambda=lambda_max.
  ``tv`` near 0 = the constraint cannot move the E-step at all at that state.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/analysis/qc_spread_probe.py \
        --run_dir logs/safety_gymnasium/SafetyPointGoal1-v0/FHDCMPO/20260819_080249 \
        --rollouts logs/opd_probe/rollouts_20260819_080249_ep20.pt \
        --config config/safety_gymnasium_fhdcmpo_a2_lag_cc4_60k.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from eval_safety_gymnasium import load_train_cfg  # noqa: E402

from safe_rl.common.fh_cost import conservatism_statistic  # noqa: E402
from safe_rl.common.per_state_dual import estep_weights, solve_eta  # noqa: E402
from safe_rl.modules import SafeActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--rollouts", required=True, help="cached rollouts .pt from offpolicy_mismatch_probe")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--states", type=int, default=1024, help="states per checkpoint")
    ap.add_argument("--actions", type=int, default=64, help="candidate actions per state (= sample_action_num)")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="logs/opd_probe/qc_spread.json")
    return ap.parse_args()


def build_policy(cfg: dict, num_obs: int, num_actions: int, device: str) -> SafeActorCritic:
    pol_cfg = {k: v for k, v in cfg["policy"].items() if k != "class_name"}
    pol = SafeActorCritic(num_obs, num_obs, num_actions, **pol_cfg).to(device)
    pol.eval()
    return pol


@torch.inference_mode()
def probe_checkpoint(pol, obs: torch.Tensor, n_act: int, alpha: float, eps_dual: float, lam_max: float) -> dict:
    """Spread panel for one checkpoint on [B, obs] states."""
    B = obs.shape[0]
    mean, log_std = pol.actor(pol.actor_obs_normalizer(obs))
    x = torch.distributions.Normal(mean, log_std.exp()).sample((n_act,))  # [N, B, A]
    actions = pol.actor.action_b + pol.actor.action_c * torch.tanh(x)

    obs_exp = obs.unsqueeze(0).expand(n_act, -1, -1).reshape(n_act * B, -1)
    act_flat = actions.reshape(n_act * B, -1)
    q1, q2 = pol.evaluate_q(obs_exp, act_flat)
    q_r = torch.min(q1, q2).reshape(n_act, B)  # [N, B] -- exactly the E-step's reward signal

    obs_n = pol.critic_obs_normalizer(obs_exp)
    theta = torch.stack([c(obs_n, act_flat) for c in pol.cost_critics]).mean(0)  # [N*B, Nq]
    rho_mean = conservatism_statistic(theta, alpha, 0.0).reshape(n_act, B)  # kappa = 0 (a2 arm)
    rho_cvar = conservatism_statistic(theta, alpha, 1.0).reshape(n_act, B)  # kappa = 1 reference

    def std_a(v: torch.Tensor) -> torch.Tensor:
        return v.std(dim=0)  # spread over actions, [B]

    s_qr, s_rm, s_rc = std_a(q_r), std_a(rho_mean), std_a(rho_cvar)

    # Per-state Pearson corr(Q_r, rho_mean) across the N candidate actions.
    qc_ = (q_r - q_r.mean(0)) / q_r.std(0).clamp_min(1e-8)
    rc_ = (rho_mean - rho_mean.mean(0)) / rho_mean.std(0).clamp_min(1e-8)
    corr = (qc_ * rc_).mean(0)  # [B]

    # Can lambda_max move the E-step? Solve eta at lambda = 0 (KL = eps_dual), then compare
    # weights at lambda = 0 vs lambda = lambda_max: per-state total variation distance.
    zeros = torch.zeros(B, device=q_r.device)
    eta = solve_eta(q_r, rho_mean, zeros, eps_dual)
    w0 = estep_weights(q_r, rho_mean, eta, zeros)
    w1 = estep_weights(q_r, rho_mean, eta, torch.full_like(zeros, lam_max))
    tv = 0.5 * (w0 - w1).abs().sum(0)  # [B]

    balance = s_qr / s_rm.clamp_min(1e-8)
    return {
        "std_a_qr_median": float(s_qr.median()),
        "std_a_rho_mean_median": float(s_rm.median()),
        "std_a_rho_cvar_median": float(s_rc.median()),
        # median of the PER-STATE ratio -- the constraint "dose" the E-step exponent picks up when
        # kappa ramps to 1. The two medians above give only a ratio-of-medians, which is not it.
        "dose_cvar_vs_mean_median": float((s_rc / s_rm.clamp_min(1e-8)).median()),
        "balance_median": float(balance.median()),
        "balance_p90": float(balance.quantile(0.9)),
        "frac_balance_above_lam_max": float((balance > lam_max).float().mean()),
        "corr_qr_rho_median": float(corr.median()),
        "tv_at_lam_max_median": float(tv.median()),
        "frac_tv_below_005": float((tv < 0.05).float().mean()),
        "eta_star": float(eta),
        "rho_level_median": float(rho_mean.median()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_train_cfg(args.config)
    eps_dual = float(cfg["algorithm"].get("dual_constraint", 0.1))
    lam_max = float(cfg["algorithm"].get("lambda_max", 1.8))

    rolls = torch.load(args.rollouts, weights_only=False, map_location=args.device)
    cks = sorted((int(p.stem.split("_")[1]), p) for p in Path(args.run_dir).glob("model_*.pt"))
    first_obs = next(v for k, v in rolls.items() if k != "heldout")["o"]
    num_obs = first_obs.shape[-1]
    num_actions = next(v for k, v in rolls.items() if k != "heldout")["a"].shape[-1]
    pol = build_policy(cfg, num_obs, num_actions, args.device)

    results = {}
    for it, ck in cks:
        if it not in rolls:
            continue
        state = torch.load(ck, weights_only=False, map_location=args.device)
        pol.load_state_dict(state["model_state_dict"], strict=False)
        o = rolls[it]["o"]
        flat = torch.tensor(o.reshape(-1, o.shape[-1]), device=args.device)
        idx = torch.randperm(flat.shape[0], device=args.device)[: args.states]
        row = probe_checkpoint(pol, flat[idx], args.actions, args.alpha, eps_dual, lam_max)
        results[it] = row
        print(
            f"iter {it:6d}: std_a Qr {row['std_a_qr_median']:.3f}  rho {row['std_a_rho_mean_median']:.3f} "
            f"(cvar {row['std_a_rho_cvar_median']:.3f}, dose {row['dose_cvar_vs_mean_median']:.2f}x)  "
            f"balance {row['balance_median']:.2f} "
            f"(p90 {row['balance_p90']:.2f}, >lam_max {row['frac_balance_above_lam_max']:.2f})  "
            f"corr {row['corr_qr_rho_median']:+.2f}  TV@lam_max {row['tv_at_lam_max_median']:.3f} "
            f"(inert frac {row['frac_tv_below_005']:.2f})"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
