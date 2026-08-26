#!/usr/bin/env python
"""Rank-consistency audit of the FH-DCMPO cost critic.

THE QUESTION. The E-step weight is ``softmax_a((Q_r - lambda*Q_c)/eta)``, a per-state softmax
over candidate actions. Any per-state constant added to Q_c cancels in the normalisation, so
only the *ordering and relative spacing* of Q_c across the candidate actions at one state ever
reaches the policy. A filter that keeps the beta*N lowest-Q_c actions depends on the ordering
alone. This script measures whether that ordering is right, against a Monte-Carlo ground truth.

It is a go/no-go gate, so the failure mode to guard against is a number that looks plausible and
is meaningless. Two guards, both mandatory:

  * **Restore verification.** Every branch depends on rewinding the simulator exactly. A silent
    restore failure produces beautifully plausible garbage. `assert_restore_roundtrip` runs
    before anything else and aborts on mismatch.
  * **A noise ceiling.** Single-rollout MC is a noisy estimate of an expectation, which
    attenuates the observed correlation. Without measuring that attenuation a low rho cannot be
    told apart from a noisy measurement, so we roll each action `--reps` times independently and
    report `rho_ceiling` next to `rho_critic`.

WHAT THE CRITIC ACTUALLY PREDICTS (read from the code, 2026-08-25 -- this is the thing the MC
must match, and it is *not* what the L=64 window name suggests):

    Q_c(s_t, a) = E[ sum_{k=t}^{T-1} c_k ]      -- UNDISCOUNTED cost-to-go to EPISODE END

The target in `FHDCMPO._update_cost_critic_quantile` (fhdcmpo.py:360-378) is a TD(lambda)
mixture over j=1..L of ``G_j = sum_{k<j} c_k + m_j * theta_target(s_{t+j}, a')``. Three separate
things make its fixed point undiscounted-to-termination rather than a 64-step sum:
  * ``ReplayStorage(cost_gamma=1.0)``            -- the window partial sums are undiscounted
  * ``_cost_bootstrap_discount() -> 1.0``        -- the bootstrap is undiscounted
  * ``_cost_bootstrap_mask() -> 1 - done``       -- the bootstrap is dropped ONLY at episode end
The L=64 window is the estimator's horizon; the bootstrap carries the value past it. Corroborated
by ``qc_thres == cost_limit == 25`` with ``qc_scale == 1``: comparing a 64-step sum against an
episodic budget would be a unit error. So the MC below sums costs to termination, NOT for 64
steps. Truncating at 64 would measure a different quantity and invalidate the audit.

REUSED FROM ``scripts/eval/cost_critic_rank_probe.py``:
  * ``_SimState`` -- MuJoCo snapshot/restore, imported verbatim by file path. It restores
    qpos/qvel/act/ctrl/qacc_warmstart/time plus the episode bookkeeping (terminated, truncated,
    steps_taken, TimeLimit._elapsed_steps) and the task RNG. Its comments record two bugs already
    paid for: omitting ctrl/qacc_warmstart made identical branches diverge from step 0, and
    without RNG restore two branches of the SAME action diverged with noise std 2.85.
  * The single-env approach. ``SafetyGymnasiumVecEnv`` is always an *async* vector env, so the
    simulator lives in another process and cannot be snapshotted.
REPLACED:
  * The MC definition: that probe computes a *discounted* return over a fixed ``--horizon``.
    Here it is undiscounted to episode end, to match FH-DCMPO's target.
  * The noise ceiling: that probe has none; it reports a single-rollout rho.
  * The policy stack: that probe builds CVPO; here it is FH-DCMPO with a quantile cost critic.
  * The horizon feature: that probe predates it. Here ``u = (T-t)/T`` is appended to every
    observation and advanced every step (see `_obs_with_u`).

RNG AND THE CEILING. ``_SimState`` restores the task RNG so branches share a common random
stream -- which is what makes actions comparable, but would make two reps of the same action
*identical* and force ``rho_ceiling`` to a meaningless 1.0. So each rep restores the snapshot and
then RESEEDS the task RNG with a per-(state, rep) value: common random numbers across the N
actions *within* a rep, independent noise *between* reps. ``rho_ceiling`` then measures how much
the ranking depends on the noise draw.

``rho_normalized = rho_critic / rho_ceiling`` is a HEURISTIC disattenuation, not an estimator
with a derivation. It is the classical correction-for-attenuation applied to Spearman
coefficients, where it holds only approximately: rank correlations are not linear correlations,
the ceiling is itself a noisy statistic from few reps, and it is undefined as the denominator
approaches zero (guarded by ``--ceiling-floor``, which yields NaN rather than a large number).
Read it as "roughly how much of the achievable ordering the critic recovers", not as a
calibrated quantity.

Usage:
    python scripts/analysis/rank_consistency_audit.py \
        --config config/safety_gymnasium_fhdcmpo_a2_lag_cc4_60k.yaml \
        --checkpoint logs/.../model_59999.pt \
        --env_id SafetyPointGoal2-v0 --cell pointgoal2_a2_s2 \
        --states 40 --actions 64 --reps 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Single-threaded on purpose. The nets are 256x256 MLPs evaluated one state at a time, so torch's
# intra-op threads buy nothing, and MuJoCo -- the actual bottleneck -- is single-threaded anyway.
# Measured: with the default thread pool, five concurrent cells each consumed ~3.2 cores on a
# 16-core box and made no progress in 74 minutes; the time went to thread contention, not work.
torch.set_num_threads(1)

import safety_gymnasium  # noqa: E402

import safe_rl.modules as sr_modules  # noqa: E402

from safe_rl.common.fh_cost import normalized_remaining_horizon  # noqa: E402

# `_SimState` by file path: the probe is a script, not an importable module.
_spec = importlib.util.spec_from_file_location(
    "_rank_probe", REPO / "scripts" / "eval" / "cost_critic_rank_probe.py"
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)


class _SimState(_probe._SimState):
    """`_SimState` plus the builder's own step counter.

    BUG FOUND 2026-08-25. The inherited class restores ``env.unwrapped.steps_taken``, guarded by
    ``hasattr``. This safety_gymnasium version names that counter ``Builder.steps``, so the guard
    silently skips it and the counter is never rewound. The original probe never noticed because
    it rolled a FIXED short horizon per branch and never approached ``task.num_steps``. Rolling to
    episode end saturates it: the first branch leaves ``steps == 1000``, and every subsequent
    branch is truncated after a single step (measured: 929, then 1, 1, 1...).

    Left as a subclass rather than a fix in the probe, which is existing, in-use code.
    """

    _EXTRA = ("steps",)

    def save(self) -> dict:
        snap = super().save()
        u = self.env.unwrapped
        snap["_extra"] = {k: getattr(u, k) for k in self._EXTRA if hasattr(u, k)}
        return snap

    def restore(self, snap: dict) -> None:
        super().restore(snap)
        u = self.env.unwrapped
        for k, v in (snap.get("_extra") or {}).items():
            setattr(u, k, v)


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties. NaN if either side is constant.

    Ties matter here: a degenerate cost critic emits many identical predictions (softplus floor),
    and ordinal ranking would invent an arbitrary order among them. Average ranks make tied
    predictions contribute no ordering information, which is the honest treatment.
    """
    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ranks within tied groups
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.arange(i, j + 1).mean()
            i = j + 1
        return r

    rx, ry = rank(np.asarray(x, dtype=np.float64)), rank(np.asarray(y, dtype=np.float64))
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def mean_pairwise_spearman(reps: np.ndarray) -> float:
    """Mean Spearman over all rep pairs -- the noise ceiling. ``reps`` is [R, N]."""
    R = reps.shape[0]
    vals = [spearman(reps[i], reps[j]) for i in range(R) for j in range(i + 1, R)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def iqr(v: list[float]) -> tuple[float, float, float]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


# ---------------------------------------------------------------------------------------------
# Policy / env plumbing
# ---------------------------------------------------------------------------------------------


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_policy(cfg: dict, num_obs: int, num_act: int, checkpoint: str, device: str):
    """Rebuild SafeActorCritic from the checkpoint's own policy_cfg and load the weights.

    The checkpoint's `policy_cfg` is authoritative over the YAML: `OffPolicyRunner.load` uses
    strict=False, so a config that disagrees on e.g. n_quantiles would load partially and leave
    randomly-initialised tensors behind while looking fine.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pol_cfg = dict(blob.get("policy_cfg") or {})
    if not pol_cfg:
        raise RuntimeError(f"{checkpoint} carries no policy_cfg; refusing to guess the architecture")
    cls_name = dict(cfg.get("policy", {})).get("class_name", "SafeActorCritic")
    policy = getattr(sr_modules, cls_name)(
        num_actor_obs=num_obs, num_critic_obs=num_obs, num_actions=num_act, **pol_cfg
    )
    missing, unexpected = policy.load_state_dict(blob["model_state_dict"], strict=False)
    hard = [k for k in missing if not k.startswith(("actor_target", "critic_obs_normalizer", "actor_obs_normalizer"))]
    if hard:
        raise RuntimeError(f"checkpoint missing policy tensors: {sorted(hard)[:6]}")
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, blob, pol_cfg


def preflight(cfg: dict, pol_cfg: dict, policy, env, num_obs_raw: int) -> dict:
    """Hard assertions that the audit measures what it claims to. Aborts on any mismatch."""
    alg = dict(cfg.get("algorithm", {}))
    env_kwargs = dict((cfg.get("env", {}) or {}).get("kwargs", {}) or {})
    facts = {}

    # (a) undiscounted-to-episode-end cost target
    if alg.get("class_name") != "FHDCMPO":
        raise SystemExit(f"expected FHDCMPO, got {alg.get('class_name')}")
    if float(alg.get("qc_scale_measured", 1.0)) != 1.0:
        raise SystemExit("qc_scale != 1: the episodic limit is no longer the threshold")
    facts["cost_target"] = "undiscounted cost-to-go to episode end (gamma_c=1, mask=1-done)"
    facts["cost_n_step_window"] = alg.get("cost_n_step")
    facts["cost_td_lambda"] = alg.get("cost_td_lambda")

    # (b) horizon feature appended, and we know its width
    if not env_kwargs.get("horizon_feature"):
        raise SystemExit("horizon_feature is off in this config; this audit assumes it is on")
    if env_kwargs.get("budget_feature"):
        raise SystemExit("budget_feature is on; _obs_with_u only appends the horizon column")
    facts["horizon_feature"] = True

    # (c) quantile-mean readout, non-negative head
    if pol_cfg.get("cost_critic_type") != "quantile":
        raise SystemExit(f"cost critic is {pol_cfg.get('cost_critic_type')}, expected quantile")
    if not (pol_cfg.get("cost_critic_kwargs") or {}).get("nonneg"):
        raise SystemExit("cost critic is not nonneg; the softplus-floor diagnostic assumes it")
    if len(policy.cost_critics) != 1:
        raise SystemExit(f"{len(policy.cost_critics)} cost critics; readout aggregation undefined")
    facts["readout"] = "mean over quantiles of the single cost critic (kappa=0 => rho == mean)"

    # (d) E-step proposal is the unwidened target-actor distribution
    scale = float(alg.get("estep_sample_std_scale", 1.0))
    if scale != 1.0:
        raise SystemExit(f"estep_sample_std_scale={scale} != 1.0; sampling would not match the E-step")
    facts["estep_sample_std_scale"] = scale
    facts["sample_action_num_cfg"] = alg.get("sample_action_num")

    # (e) actor_target is NOT persisted -- see the module docstring note in the report.
    facts["actor_used"] = "policy.actor (online); actor_target is an algorithm-side deepcopy and is not in the checkpoint"

    # (f) observation width: raw env + 1 horizon column must equal what the nets expect
    facts["num_obs_raw"] = num_obs_raw
    facts["num_obs_model"] = num_obs_raw + 1
    return facts


def _obs_with_u(obs_raw: np.ndarray, t: int, horizon: int, device: str) -> torch.Tensor:
    """Append u = (T-t)/T, exactly as HorizonAugmentedVecEnv does, using the true step counter.

    `t` is the env's own elapsed-step count, never inferred from the observation -- the wrapper
    reads it from `episode_length_buf` for the same reason.
    """
    u = normalized_remaining_horizon(torch.tensor([t], dtype=torch.float32), horizon)
    o = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).reshape(1, -1)
    return torch.cat([o, u.to(device)], dim=-1)


def elapsed_steps(env) -> int:
    e = env
    while e is not None:
        if hasattr(e, "_elapsed_steps"):
            return int(e._elapsed_steps)
        e = getattr(e, "env", None)
    return int(getattr(env.unwrapped, "steps_taken", 0))


@torch.no_grad()
def act_stochastic(policy, obs_t: torch.Tensor) -> torch.Tensor:
    return policy.act(obs_t)


@torch.no_grad()
def sample_actions(policy, obs_t: torch.Tensor, n: int, gen: torch.Generator,
                   proposal: str = "policy") -> torch.Tensor:
    """N candidate actions. ``policy`` reproduces the E-step's own proposal.

    Mirrors `MPO._estep_sample` (mpo.py:239-260) with estep_sample_std_scale == 1.0, asserted in
    preflight. Uses `policy.actor` rather than the algorithm's `actor_target`, which no
    checkpoint persists; the two differ only by the Polyak/hard target lag.

    The wider proposals exist to separate "actions do not matter" from "this policy only ever
    proposes near-identical actions". ``uniform`` ignores the policy entirely and covers the
    action box, which is the strongest available test for a real action effect.
    """
    a_b, a_c = policy.actor.action_b, policy.actor.action_c
    if proposal == "uniform":
        # Uniform over the action box: a_b +- a_c is exactly the reachable range of the squash.
        u = torch.rand((n, a_c.numel() if a_c.dim() else obs_t.shape[-1]), generator=gen,
                       device=obs_t.device)
        return a_b + a_c * (2.0 * u - 1.0)
    mean, log_std = policy.actor(obs_t)
    std = log_std.exp()
    if proposal == "policy_std_x3":
        std = std * 3.0
    elif proposal != "policy":
        raise SystemExit(f"unknown --proposal {proposal}")
    x = mean + std * torch.randn((n, mean.shape[-1]), generator=gen, device=mean.device)
    return a_b + a_c * torch.tanh(x)


@torch.no_grad()
def cost_readout(policy, obs_t: torch.Tensor, actions: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(mean over quantiles, softplus-floor fraction) per action. kappa=0 => rho IS the mean."""
    obs_rep = obs_t.expand(actions.shape[0], -1)
    critic = policy.cost_critics[0]
    theta = critic(obs_rep, actions)          # [N, Nq], sorted, softplus'd
    return theta.mean(dim=-1).cpu().numpy(), critic.zero_frac(theta).cpu().numpy()


@torch.no_grad()
def reward_readout(policy, obs_t: torch.Tensor, actions: torch.Tensor) -> np.ndarray:
    """Twin-critic min, as the E-step scores reward (mpo.py:258-259)."""
    q1, q2 = policy.evaluate_q(obs_t.expand(actions.shape[0], -1), actions)
    return torch.min(q1, q2).cpu().numpy()


# ---------------------------------------------------------------------------------------------
# Restore verification
# ---------------------------------------------------------------------------------------------


def assert_restore_roundtrip(env, sim, steps: int, tol: float, rng: np.random.RandomState) -> float:
    """Save, replay a fixed action sequence, restore, replay it again -- observations must match.

    Hard assertion: a silent restore failure makes every downstream number garbage while looking
    entirely plausible, so this runs before any measurement and aborts the script on mismatch.
    """
    seq = [env.action_space.sample() for _ in range(steps)]
    snap = sim.save()
    first = []
    for a in seq:
        o, _r, c, _te, _tr, _i = env.step(a)
        first.append(np.concatenate([np.asarray(o, dtype=np.float64).ravel(), [float(c)]]))
    sim.restore(snap)
    second = []
    for a in seq:
        o, _r, c, _te, _tr, _i = env.step(a)
        second.append(np.concatenate([np.asarray(o, dtype=np.float64).ravel(), [float(c)]]))
    sim.restore(snap)
    d = float(np.abs(np.asarray(first) - np.asarray(second)).max())
    if not (d <= tol):
        raise SystemExit(
            f"RESTORE ROUND-TRIP FAILED: max |obs,cost| diff {d:.3e} > {tol:.1e} over {steps} steps. "
            "Every branch would be built on a broken rewind; aborting."
        )

    # Second, stronger check: a SHORT round-trip cannot see a counter that only saturates at the
    # episode limit. This is what caught the unrestored `Builder.steps` -- the first branch ran
    # 929 steps and every later one was truncated after 1. Run two full branches to termination
    # and require the same length.
    lens = []
    for _ in range(2):
        sim.restore(snap)
        n = 0
        while True:
            _o, _r, _c, te, tr, _i = env.step(env.action_space.sample())
            n += 1
            if te or tr:
                break
            if n > 2 * (getattr(env.unwrapped.task, "num_steps", 1000) or 1000):
                raise SystemExit("branch did not terminate; episode bookkeeping is not being restored")
        lens.append(n)
    sim.restore(snap)
    if lens[0] != lens[1]:
        raise SystemExit(
            f"RESTORE IS NOT REPEATABLE: consecutive full branches ran {lens[0]} then {lens[1]} steps. "
            "Some episode counter survives the rewind; every Monte-Carlo return would be truncated "
            "differently. Aborting."
        )
    if lens[0] < 2:
        raise SystemExit(f"branch length {lens[0]} at this state: episode already at its limit")
    return d, lens[0]


# ---------------------------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------------------------


def collect_states(env, policy, sim, n_states: int, skip: int, horizon: int,
                   device: str, rng: np.random.RandomState) -> list[dict]:
    """Roll the checkpoint's own policy and snapshot `n_states` states, skipping episode starts.

    Skipping the first `skip` steps keeps the sample off the initial-state manifold, where every
    episode looks alike and the remaining horizon is always ~T.
    """
    states, t = [], elapsed_steps(env)
    obs_raw, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
    t = 0
    # Spread snapshots uniformly over the reachable band rather than taking the first n.
    while len(states) < n_states:
        obs_t = _obs_with_u(obs_raw, t, horizon, device)
        a = act_stochastic(policy, obs_t).cpu().numpy().reshape(-1)
        obs_raw, _r, _c, term, trunc, _i = env.step(a)
        t += 1
        if term or trunc:
            obs_raw, _ = env.reset()
            t = 0
            continue
        if t >= skip and rng.rand() < 0.02:
            states.append({"t": t, "obs_raw": np.asarray(obs_raw, copy=True), "snap": sim.save()})
    return states


def rollout_branch(env, policy, sim, snap: dict, action: np.ndarray, t0: int,
                   horizon: int, device: str, seed: int,
                   max_steps: int | None = None) -> tuple[float, int, bool]:
    """Restore, take `action`, then follow pi to episode end. Returns (undiscounted cost, len, ended).

    The reseed is what makes reps independent: `_SimState.restore` puts the task RNG back to the
    snapshot value (common random numbers across actions, which is what we want *within* a rep),
    so without reseeding two reps of the same action would be bit-identical and the noise ceiling
    would be a meaningless 1.0.
    """
    sim.restore(snap)
    r = sim._rng()
    if r is not None:
        r.seed(seed)
    total, t, steps = 0.0, t0, 0
    obs_raw, _rw, c, term, trunc, _i = env.step(action)
    total += float(c)
    t += 1
    steps += 1
    while not (term or trunc):
        if max_steps is not None and steps >= max_steps:
            break
        obs_t = _obs_with_u(obs_raw, t, horizon, device)
        a = act_stochastic(policy, obs_t).cpu().numpy().reshape(-1)
        obs_raw, _rw, c, term, trunc, _i = env.step(a)
        total += float(c)
        t += 1
        steps += 1
    return total, steps, bool(term)


def audit_state(env, policy, sim, st: dict, n_actions: int, reps: int, horizon: int,
                device: str, gen: torch.Generator, base_seed: int,
                proposal: str = "policy", max_steps: int | None = None) -> dict:
    """Predicted vs Monte-Carlo cost for N actions at one state, with a noise ceiling.

    NOTE on `max_steps`: capping the rollout makes the MC a truncated H-step cost sum, which is
    NOT what Q_c predicts (cost-to-go to episode end). At H < full, `rho_critic` therefore is not
    a critic-quality number -- it compares two different quantities. `rho_ceiling` and the
    variance decomposition remain valid at any H, because both are MC-vs-MC.
    """
    obs_t = _obs_with_u(st["obs_raw"], st["t"], horizon, device)
    actions = sample_actions(policy, obs_t, n_actions, gen, proposal)
    predicted, zero_frac = cost_readout(policy, obs_t, actions)
    q_r = reward_readout(policy, obs_t, actions)

    a_np = actions.cpu().numpy()
    mc = np.zeros((reps, n_actions), dtype=np.float64)
    lens = np.zeros((reps, n_actions), dtype=np.int64)
    terminated = 0
    for rep in range(reps):
        # One seed per (state, rep): common random numbers across actions, independent across reps.
        rep_seed = (base_seed + 7919 * st["t"] + 104729 * rep) % (2**31 - 1)
        for i in range(n_actions):
            total, steps, term = rollout_branch(
                env, policy, sim, st["snap"], a_np[i], st["t"], horizon, device, rep_seed, max_steps
            )
            mc[rep, i], lens[rep, i] = total, steps
            terminated += int(term)
    sim.restore(st["snap"])

    mc_mean = mc.mean(axis=0)
    rho_c = spearman(predicted, mc_mean)
    rho_ceil = mean_pairwise_spearman(mc)
    # Unbiased split of the across-action variance. Under "the action has no effect",
    # var over action-means == sigma_noise^2 / R; the excess over that is the action effect.
    noise_var = float(mc.var(axis=0, ddof=1).mean()) if reps > 1 else float("nan")
    across_var = float(mc_mean.var(ddof=1))
    excess = across_var - noise_var / reps
    return {
        "t": int(st["t"]),
        "action_std_per_dim": a_np.std(axis=0).tolist(),
        "across_action_var": across_var,
        "noise_var": noise_var,
        "excess_var": excess,
        "excess_share": excess / across_var if across_var > 1e-12 else float("nan"),
        "rho_critic": rho_c,
        "rho_ceiling": rho_ceil,
        "std_a_qc": float(np.std(predicted)),
        "std_a_qr": float(np.std(q_r)),
        "zero_frac_mean": float(np.mean(zero_frac)),
        "mc_mean": float(np.mean(mc_mean)),
        "mc_std_a": float(np.std(mc_mean)),
        "mc_len_mean": float(lens.mean()),
        "terminated_frac": float(terminated / (reps * n_actions)),
        "predicted": predicted.tolist(),
        "mc": mc.tolist(),
    }


def bucket_rows(rows: list[dict], lo: int, hi: int, n: int = 4) -> list[dict]:
    """rho binned over t, so a horizon-dependent failure is visible rather than averaged away."""
    edges = np.linspace(lo, hi, n + 1)
    out = []
    for k in range(n):
        sel = [r for r in rows if edges[k] <= r["t"] < edges[k + 1] or (k == n - 1 and r["t"] == edges[-1])]
        m_c, _, _ = iqr([r["rho_critic"] for r in sel])
        m_ce, _, _ = iqr([r["rho_ceiling"] for r in sel])
        m_n, _, _ = iqr([r["rho_normalized"] for r in sel])
        out.append({"t_lo": int(edges[k]), "t_hi": int(edges[k + 1]), "n": len(sel),
                    "rho_critic_med": m_c, "rho_ceiling_med": m_ce, "rho_normalized_med": m_n})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--cell", required=True, help="label for the output file")
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--actions", type=int, default=64, help="must match sample_action_num")
    ap.add_argument("--reps", type=int, default=2, help=">=2; independent MC rollouts per action")
    ap.add_argument("--skip", type=int, default=50, help="min episode step for a sampled state")
    ap.add_argument("--t-hi", type=int, default=950, help="upper edge of the t bucketing")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--device", default="cpu", help="cpu: nets are tiny, MuJoCo is the bottleneck")
    ap.add_argument("--restore-steps", type=int, default=10)
    ap.add_argument("--restore-tol", type=float, default=1e-6)
    ap.add_argument("--ceiling-floor", type=float, default=0.05,
                    help="|rho_ceiling| below this -> rho_normalized is NaN, not a huge number")
    ap.add_argument("--horizon", default="full",
                    help="MC rollout cap in steps, or 'full' for episode end. Only 'full' matches "
                         "what Q_c predicts; shorter H makes rho_critic non-comparable (ceiling "
                         "and excess stay valid).")
    ap.add_argument("--proposal", default="policy",
                    choices=("policy", "policy_std_x3", "uniform"),
                    help="candidate-action distribution; wider ones test whether a narrow policy "
                         "proposal is what hides the action effect")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--out_dir", default="outputs/rank_audit")
    args = ap.parse_args()

    if args.reps < 2:
        raise SystemExit("--reps must be >= 2: with one rollout there is no noise ceiling")
    max_steps = None if str(args.horizon).lower() == "full" else int(args.horizon)
    if max_steps is not None and max_steps < 1:
        raise SystemExit("--horizon must be >= 1 or 'full'")

    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    cfg = load_cfg(args.config)
    env = safety_gymnasium.make(args.env_id)
    env.reset(seed=args.seed)
    horizon = int(env.spec.max_episode_steps)
    num_obs_raw = int(env.observation_space.shape[0])
    num_act = int(env.action_space.shape[0])

    policy, blob, pol_cfg = build_policy(cfg, num_obs_raw + 1, num_act, args.checkpoint, args.device)
    facts = preflight(cfg, pol_cfg, policy, env, num_obs_raw)
    if args.actions != int(facts["sample_action_num_cfg"] or args.actions):
        print(f"  ! --actions {args.actions} != sample_action_num {facts['sample_action_num_cfg']}")

    print(f"[{args.cell}] {args.env_id}  T={horizon}  obs={num_obs_raw}(+1 u)  act={num_act}  "
          f"ckpt iter={blob.get('iter')}  seed={args.seed}")
    for k, v in facts.items():
        print(f"    {k}: {v}")

    sim = _SimState(env)
    # Warm the sim off the initial state before the round-trip, so the check exercises a
    # mid-episode state with contacts, not the trivially-restorable reset state.
    for _ in range(args.skip):
        env.step(env.action_space.sample())
    d, blen = assert_restore_roundtrip(env, sim, args.restore_steps, args.restore_tol, rng)
    print(f"    restore round-trip: max diff {d:.3e} over {args.restore_steps} steps; "
          f"two full branches both {blen} steps -- PASS")

    states = collect_states(env, policy, sim, args.states, args.skip, horizon, args.device, rng)
    print(f"    collected {len(states)} states, t in [{min(s['t'] for s in states)}, "
          f"{max(s['t'] for s in states)}]")

    rows = []
    for k, st in enumerate(states, 1):
        row = audit_state(env, policy, sim, st, args.actions, args.reps, horizon,
                          args.device, gen, args.seed, args.proposal, max_steps)
        ceil = row["rho_ceiling"]
        row["rho_normalized"] = (row["rho_critic"] / ceil
                                 if np.isfinite(ceil) and abs(ceil) >= args.ceiling_floor
                                 else float("nan"))
        rows.append(row)
        print(f"    [{k:3d}/{len(states)}] t={row['t']:4d}  rho_c={row['rho_critic']:+.3f}  "
              f"ceil={ceil:+.3f}  norm={row['rho_normalized']:+.3f}  "
              f"std_a(Qc)={row['std_a_qc']:.4f}  zero={row['zero_frac_mean']:.2f}  "
              f"mc_len={row['mc_len_mean']:.0f}", flush=True)

    env.close()

    med_c, lo_c, hi_c = iqr([r["rho_critic"] for r in rows])
    med_ce, lo_ce, hi_ce = iqr([r["rho_ceiling"] for r in rows])
    med_n, lo_n, hi_n = iqr([r["rho_normalized"] for r in rows])
    summary = {
        "cell": args.cell, "env_id": args.env_id, "checkpoint": args.checkpoint,
        "iter": blob.get("iter"), "seed": args.seed, "states": len(rows),
        "actions": args.actions, "reps": args.reps, "horizon": horizon,
        "mc_horizon": args.horizon, "proposal": args.proposal,
        "rho_critic_comparable": max_steps is None,
        "action_std_per_dim": np.mean([r["action_std_per_dim"] for r in rows], axis=0).tolist(),
        "excess_var": {
            "mean": float(np.mean([r["excess_var"] for r in rows])),
            "median": float(np.median([r["excess_var"] for r in rows])),
            "deciles": np.percentile([r["excess_var"] for r in rows],
                                     [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist(),
            "frac_positive": float(np.mean([r["excess_var"] > 0 for r in rows])),
        },
        "excess_share_median": float(np.nanmedian([r["excess_share"] for r in rows])),
        "wall_clock_s": round(time.time() - t_start, 1),
        "restore_roundtrip_maxdiff": d,
        "rho_critic": {"median": med_c, "q25": lo_c, "q75": hi_c},
        "rho_ceiling": {"median": med_ce, "q25": lo_ce, "q75": hi_ce},
        "rho_normalized": {"median": med_n, "q25": lo_n, "q75": hi_n},
        "mean_std_a_qc": float(np.mean([r["std_a_qc"] for r in rows])),
        "mean_std_a_qr": float(np.mean([r["std_a_qr"] for r in rows])),
        "mean_zero_frac": float(np.mean([r["zero_frac_mean"] for r in rows])),
        "mean_mc_std_a": float(np.mean([r["mc_std_a"] for r in rows])),
        "mean_mc_len": float(np.mean([r["mc_len_mean"] for r in rows])),
        "terminated_frac": float(np.mean([r["terminated_frac"] for r in rows])),
        "buckets": bucket_rows(rows, args.skip, args.t_hi),
        "facts": facts,
    }
    name = args.cell if not args.tag else f"{args.cell}__{args.tag}"
    out = REPO / args.out_dir / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_state": rows}, indent=1, default=float) + "\n")

    print(f"\n[{args.cell}] rho_critic {med_c:+.3f} [{lo_c:+.3f}, {hi_c:+.3f}]   "
          f"rho_ceiling {med_ce:+.3f} [{lo_ce:+.3f}, {hi_ce:+.3f}]   "
          f"rho_normalized {med_n:+.3f} [{lo_n:+.3f}, {hi_n:+.3f}]")
    print(f"    std_a(Qc) {summary['mean_std_a_qc']:.4f}  zero_frac {summary['mean_zero_frac']:.3f}  "
          f"mc_len {summary['mean_mc_len']:.0f}  {summary['wall_clock_s']:.0f}s")
    ev = summary["excess_var"]
    print(f"    H={args.horizon} proposal={args.proposal}  excess mean={ev['mean']:+.2f} "
          f"median={ev['median']:+.2f} frac>0={ev['frac_positive']:.0%}  "
          f"action_std/dim={np.round(summary['action_std_per_dim'], 4).tolist()}")
    if max_steps is not None:
        print("    NOTE: H < full -> rho_critic compares Q_c (episode-end) against a truncated "
              "MC; read ceiling/excess, not rho_critic.")
    for b in summary["buckets"]:
        print(f"    t[{b['t_lo']:4d},{b['t_hi']:4d}) n={b['n']:3d}  rho_c={b['rho_critic_med']:+.3f}  "
              f"ceil={b['rho_ceiling_med']:+.3f}  norm={b['rho_normalized_med']:+.3f}")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
