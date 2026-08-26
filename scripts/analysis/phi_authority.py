#!/usr/bin/env python
"""Normalisation-free decomposition of the E-step exponent: Phi.

A = lambda*sigma_a(Qc)/(eta*sqrt(2*eps)) divides by a batch-MEAN epsilon over a per-state KL
distribution spanning ~210x (median 0.025, mean 0.190 on pointgoal2_a2_s2). That mis-scales every
state by sqrt(eps_batch/KL_s). Phi removes the normaliser entirely by taking ratios to the total
exponent spread, so eta and eps both cancel:

    z(s,a)      = (Q_r - lambda*Q_c)/eta          the E-step exponent
    Phi_cost(s) = lambda*sigma_a(Q_c) / sigma_a(Q_r - lambda*Q_c)
    Phi_rew(s)  =        sigma_a(Q_r) / sigma_a(Q_r - lambda*Q_c)

with the exact identity  1 = Phi_rew^2 + Phi_cost^2 - 2*rho*Phi_rew*Phi_cost,
rho = corr_a(Q_r, Q_c) across the sampled actions. Verified numerically, not assumed.

Headline: the MOVEMENT-WEIGHTED share, weighting each state by how far the E-step actually moves
it, so states the E-step barely touches do not dilute the average:

    Phi_bar_cost = sum_s KL_s * Phi_cost(s) / sum_s KL_s,   KL_s = sum_i w_i log(N w_i)

STATE DISTRIBUTION. The logged `estep_std_qc` is computed on replay-buffer states; every probe
here uses rollouts. The buffer is never serialised (`OffPolicyRunner.save` stores model/optimizer/
normalizer only), so it cannot be replayed. Both distributions are therefore reported:

  onpolicy : rollouts of the FINAL checkpoint's policy, t-stratified
  bufproxy : rollouts pooled over every saved checkpoint (5k..60k), t-stratified -- a proxy for
             the buffer's mixture over the training history, NOT the buffer itself

Critics are always the final checkpoint's, in both cases.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "_aud", os.path.join(REPO, "scripts/analysis/rank_consistency_audit.py"))
_aud = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aud)
import safety_gymnasium  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

DEC = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def logged(run_dir: str) -> dict:
    ea = EventAccumulator(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))[0])
    ea.Reload()
    g = lambda k: float(np.mean([e.value for e in ea.Scalars(k)][-3:]))  # noqa: E731
    return {"lam": g("SafeRL/lambda_mean"), "eta": g("Train/eta"), "eps": g("Train/kl_q"),
            "std_qc_logged": g("Train/estep_std_qc"), "std_qr_logged": g("Train/estep_std_qr")}


def collect_states(env, pol, n, seed, accept=None, skip=50, horizon=1000):
    """t-stratified: acceptance is set so `n` states span the whole episode, not just its start.

    A fixed acceptance rate silently breaks stratification when `n` is small: at 0.28 the loop
    collects 21 states within ~75 steps and never leaves the episode's opening. Scaling the rate
    to n/(horizon-skip) keeps the t-marginal spread whatever `n` is.
    """
    if accept is None:
        accept = min(0.9, max(1.5 * n / float(horizon - skip), 0.01))
    rng = np.random.RandomState(seed)
    obs, _ = env.reset(seed=seed)
    t, out = 0, []
    while len(out) < n:
        ot = _aud._obs_with_u(obs, t, horizon, "cpu")
        if t >= skip and rng.rand() < accept:
            out.append((np.asarray(obs, copy=True), t))
        obs, _r, _c, te, tr, _i = env.step(_aud.act_stochastic(pol, ot).numpy().reshape(-1))
        t += 1
        if te or tr:
            obs, _ = env.reset()
            t = 0
    return out


def score_states(pol, states, n_act, gen, horizon=1000):
    QR, QC, T = [], [], []
    for obs, t in states:
        ot = _aud._obs_with_u(obs, t, horizon, "cpu")
        a = _aud.sample_actions(pol, ot, n_act, gen, "policy")
        with torch.no_grad():
            q1, q2 = pol.evaluate_q(ot.expand(n_act, -1), a)
            QR.append(torch.min(q1, q2).reshape(-1).numpy())
            QC.append(pol.cost_critics[0](ot.expand(n_act, -1), a).mean(dim=-1).reshape(-1).numpy())
        T.append(t)
    return np.array(QR), np.array(QC), np.array(T)


def phi(QR, QC, lam, eta, n_act):
    sz = (QR - lam * QC).std(axis=1, ddof=1)
    sr, sc = QR.std(axis=1, ddof=1), lam * QC.std(axis=1, ddof=1)
    rho = np.array([np.corrcoef(QR[i], QC[i])[0, 1] if QC[i].std() > 1e-12 else np.nan
                    for i in range(len(QR))])
    z = (QR - lam * QC) / eta
    w = np.exp(z - z.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    KL = (w * np.log(n_act * w + 1e-8)).sum(axis=1)
    szc = np.clip(sz, 1e-12, None)
    return {"Phi_cost": sc / szc, "Phi_rew": sr / szc, "rho": rho, "KL": KL, "sigma_z": sz}


def wmean_ci(v, wt, B=10000, seed=0):
    m = np.isfinite(v) & np.isfinite(wt)
    v, wt = v[m], wt[m]
    obs = float((wt * v).sum() / max(wt.sum(), 1e-12))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(v), len(v))
        bs.append((wt[i] * v[i]).sum() / max(wt[i].sum(), 1e-12))
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--actions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="outputs/paper_figs/phi_authority.json")
    args = ap.parse_args()

    CELLS = {
        "pointgoal2_a2_s2": ("SafetyPointGoal2-v0", "logs/safety_gymnasium/SafetyPointGoal2-v0/FHDCMPO/20260822_182732"),
        "cargoal1_a2_s2": ("SafetyCarGoal1-v0", "logs/safety_gymnasium/SafetyCarGoal1-v0/FHDCMPO/20260823_041146"),
        "pointbutton1_a2_s2": ("SafetyPointButton1-v0", "logs/safety_gymnasium/SafetyPointButton1-v0/FHDCMPO/20260823_134048"),
        "pointpush1_a2_s2": ("SafetyPointPush1-v0", "logs/safety_gymnasium/SafetyPointPush1-v0/FHDCMPO/20260823_231722"),
    }
    cfg = _aud.load_cfg(os.path.join(REPO, "config/safety_gymnasium_fhdcmpo_a2_lag_cc4_60k.yaml"))
    out = {}
    for cell, (env_id, rd) in CELLS.items():
        rd = os.path.join(REPO, rd)
        L = logged(rd)
        env = safety_gymnasium.make(env_id)
        env.reset(seed=args.seed)
        n_obs, n_act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        final = os.path.join(rd, "model_59999.pt")
        if not os.path.exists(final):
            final = sorted(glob.glob(os.path.join(rd, "model_*.pt")),
                           key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))[-1]
        pol_final, _b, _p = _aud.build_policy(cfg, n_obs + 1, n_act_dim, final, "cpu")

        out[cell] = {"logged": L, "env": env_id, "checkpoint": os.path.basename(final)}
        for dist in ("onpolicy", "bufproxy"):
            # Seeded for reproducibility: the rollout policy draws from the GLOBAL torch RNG,
            # so leaving it unseeded made the same statistic read 0.52 and 0.77 on two draws.
            torch.manual_seed(args.seed)
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            if dist == "onpolicy":
                states = collect_states(env, pol_final, args.states, args.seed)
            else:
                cks = sorted(glob.glob(os.path.join(rd, "model_*.pt")),
                             key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
                per = max(1, args.states // len(cks))
                states = []
                for j, ck in enumerate(cks):
                    torch.manual_seed(args.seed + j)
                    pol_j, _, _ = _aud.build_policy(cfg, n_obs + 1, n_act_dim, ck, "cpu")
                    states += collect_states(env, pol_j, per, args.seed + j)
                states = states[:args.states]
            torch.manual_seed(args.seed)
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            QR, QC, T = score_states(pol_final, states, args.actions, gen)
            P = phi(QR, QC, L["lam"], L["eta"], args.actions)
            ident = np.abs(np.sqrt(np.clip(P["Phi_rew"] ** 2 + P["Phi_cost"] ** 2
                                           - 2 * P["rho"] * P["Phi_rew"] * P["Phi_cost"], 0, None)) - 1.0)
            bar = wmean_ci(P["Phi_cost"], P["KL"], seed=args.seed)
            barr = wmean_ci(P["Phi_rew"], P["KL"], seed=args.seed)
            out[cell][dist] = {
                "n": len(T), "t_min": int(T.min()), "t_max": int(T.max()), "t_median": int(np.median(T)),
                "Phi_cost_dec": np.nanpercentile(P["Phi_cost"], DEC).tolist(),
                "Phi_rew_dec": np.nanpercentile(P["Phi_rew"], DEC).tolist(),
                "rho_dec": np.nanpercentile(P["rho"], DEC).tolist(),
                "rho_mean": float(np.nanmean(P["rho"])),
                "rho_median": float(np.nanmedian(P["rho"])),
                "rho_frac_pos": float(np.nanmean(P["rho"] > 0)),
                "rho_frac_strong_pos": float(np.nanmean(P["rho"] > 0.5)),
                "rho_frac_strong_neg": float(np.nanmean(P["rho"] < -0.5)),
                "rho_kl_weighted": float(np.nansum(P["KL"] * P["rho"]) / np.nansum(P["KL"])),
                "KL_dec": np.nanpercentile(P["KL"], DEC).tolist(),
                "Phi_bar_cost": bar, "Phi_bar_rew": barr,
                "Phi_cost_unweighted_median": float(np.nanmedian(P["Phi_cost"])),
                "std_qc_measured_median": float(np.median(QC.std(axis=1, ddof=1))),
                "identity_max_err": float(np.nanmax(ident)),
            }
            print(f"{cell:<20} {dist:<9} n={len(T)} t[{T.min()},{T.max()}] med {int(np.median(T))}  "
                  f"Phi_bar_cost={bar[0]:.3f} [{bar[1]:.3f},{bar[2]:.3f}]  "
                  f"Phi_bar_rew={barr[0]:.3f}  identity_err={out[cell][dist]['identity_max_err']:.1e}",
                  flush=True)
        env.close()
    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=1, default=float)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
