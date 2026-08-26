#!/usr/bin/env python
"""Validate the constraint-authority statistic A against a positive control.

    A = lambda * std_a(Q_c) / (eta * sqrt(2*eps)),    eps = the run's own logged kl_q

A ~ 1 means the cost term can move the E-step as far as the trust region allows; A << 1 means
it is decorative. Measured on the multi-env sweep A was 0.00-0.54 -- but every one of those runs
FAILED its budget, so a statistic that is low on all failures and never seen on a success predicts
nothing. PointGoal1 is where a2/c2 held the budget, so it supplies the positive control.

Design note: the headline is the WITHIN-RUN trajectory correlation, not the cross-run scatter.
Across arms, A is confounded with everything else that differs between arms (lambda_max, cost
window, spread matching, risk mode). Within a run, A(t) and realized cost move under one fixed
config, so the correlation is confounder-free.

x-axis is realized_cost_ema/limit, taken as 1 + lambda_delta: the controller logs
lambda_delta = (realized_ema - limit)/limit, which IS its realized-cost signal.
`Train/realized_cost_ema` is NaN throughout these runs (populated only on the qc-adapt paths).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PG1 = os.path.join(REPO, "logs/safety_gymnasium/SafetyPointGoal1-v0/FHDCMPO")
LIMIT = 25.0

# run dir -> (arm, seed, lambda_max). Identified in Phase 1; the six unidentified runs and the
# ambiguous 20260818_223403 (a1_lag_base / a3_lag_cc8 launched in the same second) are excluded.
PG1_RUNS = {
    "20260817_145216": ("s1", 2, 4.0),
    "20260817_165854": ("s2", 2, None),
    "20260818_082646": ("s1b", 2, 1.8),
    "20260818_083010": ("s1c", 2, 4.0),
    "20260818_125149": ("s2b", 2, 1.8),
    "20260818_190418": ("s2b_tdlam_L64", 2, 1.8),
    "20260818_234159": ("a2_lag_cc4", 2, 1.8),
    "20260819_051126": ("a4_lag_lrhalf", 2, 1.8),
    "20260819_065512": ("a1_base_60k", 2, 1.8),
    "20260819_080249": ("a2_cc4_60k", 2, 1.8),
    "20260819_185710": ("a5_cc4_lrhalf", 2, 1.8),
    "20260820_101946": ("c1", 2, 1.8),
    "20260820_115458": ("c2", 2, 1.8),
    "20260821_030844": ("c2_60k", 2, 1.8),
    "20260820_203502": ("c3", 2, 1.8),
    "20260820_224619": ("c3", 3, 1.8),
    "20260821_005517": ("c3", 4, 1.8),
}
KEYS = ("SafeRL/lambda_mean", "Train/eta", "Train/estep_std_qc", "Train/kl_q",
        "SafeRL/lambda_delta", "Episode/cost")


def load(d: str) -> dict | None:
    f = glob.glob(os.path.join(d, "events.out.tfevents.*"))
    if not f:
        return None
    ea = EventAccumulator(f[0])
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    if not set(KEYS) <= tags:
        return None
    # All six are written in the same loss_dict at log_interval, so steps align by construction;
    # intersect anyway rather than assume it.
    series = {k: {e.step: e.value for e in ea.Scalars(k)} for k in KEYS}
    steps = sorted(set.intersection(*(set(v) for v in series.values())))
    return {"step": np.array(steps, float),
            **{k.split("/")[-1]: np.array([series[k][s] for s in steps], float) for k in KEYS}}


def authority(r: dict, lam_max: float | None) -> dict:
    """A, A^2, lambda_req, shortfall -- elementwise over training."""
    eps = np.clip(r["kl_q"], 1e-12, None)
    denom = r["eta"] * np.sqrt(2.0 * eps)
    A = r["lambda_mean"] * r["estep_std_qc"] / np.clip(denom, 1e-12, None)
    lam_req = denom / np.clip(r["estep_std_qc"], 1e-12, None)
    cap = lam_max if lam_max else np.nanmax(r["lambda_mean"]) / 0.98
    return {"A": A, "A2": A ** 2, "lambda_req": lam_req, "shortfall": lam_req / cap,
            "cost_ratio": 1.0 + r["lambda_delta"], "cap": cap}


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return np.nan

    def rk(v):
        o = v.argsort(kind="mergesort")
        rr = np.empty(len(v))
        rr[o] = np.arange(len(v))
        sv = v[o]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                rr[o[i:j + 1]] = np.arange(i, j + 1).mean()
            i = j + 1
        return rr
    a, b = rk(x), rk(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def tail_mean(v, step, frac=0.05):
    cut = (1 - frac) * step.max()
    m = (step >= cut) & np.isfinite(v)
    return float(np.mean(v[m])) if m.any() else np.nan


def at_iter(v, step, target=19999, tol=1500):
    d = np.abs(step - target)
    i = int(np.argmin(d))
    return float(v[i]) if d[i] <= tol and np.isfinite(v[i]) else np.nan


def collect():
    out = []
    for run, (arm, seed, lm) in PG1_RUNS.items():
        r = load(os.path.join(PG1, run))
        if r is None:
            print(f"  ! {run}: missing scalars, skipped")
            continue
        a = authority(r, lm)
        out.append({"run": run, "arm": arm, "seed": seed, "env": "PointGoal1",
                    "cap": a["cap"], "series": {**r, **a},
                    "rho_within": spearman(a["A"], a["cost_ratio"]),
                    "A_final": tail_mean(a["A"], r["step"]),
                    "A_at20k": at_iter(a["A"], r["step"]),
                    "cost_ratio_final": tail_mean(a["cost_ratio"], r["step"]),
                    "cost5_ratio": tail_mean(r["cost"], r["step"]) / LIMIT,
                    "cost_at20k": at_iter(a["cost_ratio"], r["step"]),
                    "shortfall_final": tail_mean(a["shortfall"], r["step"]),
                    "lam_req_final": tail_mean(a["lambda_req"], r["step"]),
                    "iters": int(r["step"].max())})
    # the 14 multi-env sweep cells
    for d in sorted(glob.glob(os.path.join(REPO, "logs/safety_gymnasium/*/FHDCMPO/*/"))):
        mk = os.path.join(d, ".sweep_cell")
        if not os.path.exists(mk):
            continue
        cell = open(mk).read().strip()
        r = load(d)
        if r is None:
            continue
        a = authority(r, 1.8)
        env = d.split("safety_gymnasium/")[1].split("/")[0]
        out.append({"run": os.path.basename(d.rstrip("/")), "arm": cell.rsplit("_s", 1)[0],
                    "seed": int(cell.rsplit("_s", 1)[1]), "env": env, "cap": 1.8,
                    "series": {**r, **a},
                    "rho_within": spearman(a["A"], a["cost_ratio"]),
                    "A_final": tail_mean(a["A"], r["step"]),
                    "A_at20k": at_iter(a["A"], r["step"]),
                    "cost_ratio_final": tail_mean(a["cost_ratio"], r["step"]),
                    "cost5_ratio": tail_mean(r["cost"], r["step"]) / LIMIT,
                    "cost_at20k": at_iter(a["cost_ratio"], r["step"]),
                    "shortfall_final": tail_mean(a["shortfall"], r["step"]),
                    "lam_req_final": tail_mean(a["lambda_req"], r["step"]),
                    "iters": int(r["step"].max())})
    return out


if __name__ == "__main__":
    runs = collect()
    json.dump([{k: v for k, v in r.items() if k != "series"} for r in runs],
              open(os.path.join(REPO, "outputs/paper_figs/authority_table.json"), "w"),
              indent=1, default=float)
    print(f"collected {len(runs)} runs "
          f"({sum(r['env'] == 'PointGoal1' for r in runs)} PointGoal1, "
          f"{sum(r['env'] != 'PointGoal1' for r in runs)} sweep cells)")
