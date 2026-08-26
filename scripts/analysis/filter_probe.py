#!/usr/bin/env python
"""Test-time action-filter probe: does the Q_c ORDERING buy anything behaviourally?

The rank audit asked whether Q_c orders actions correctly against a Monte-Carlo ground truth,
and ran into a measurement wall: the MC noise ceiling at the episode-end horizon is ~0, so the
correlation is unmeasurable at feasible rep counts. This probe sidesteps that entirely. Instead
of asking "is the ordering correct?", it asks "does acting on the ordering change realized cost?"
-- which is the question the filter proposal actually rests on, and it needs no ground truth.

Five rules, each in its own paired episode set. At every step, draw K actions from the policy
proposal and score all of them with Q_r (twin min, as the E-step does) and Q_c (quantile mean):

    base      one sample from the proposal            -- what the policy does today
    greedyR   argmax Q_r over all K
    randfilt  random M of K, then argmax Q_r among them
    filter25  the M with LOWEST Q_c, then argmax Q_r among them
    argminQc  argmin Q_c over all K

THE DECISIVE COMPARISON IS `filter25` vs `randfilt`. Both keep M of K and then take the greedy
Q_r action; they differ only in whether the surviving M were chosen by Q_c or at random. So the
difference isolates the contribution of the *cost ordering* from the contribution of doing greedy
Q_r selection at all -- which `greedyR` vs `base` would confound.

PAIRING. Episode e uses the same env seed under every condition, so the conditions start from
identical initial states; and the K candidate actions at step t are drawn from a generator seeded
by (episode, step), so they are identical across conditions until the trajectories diverge. That
is common random numbers at the episode level, which is what the paired CIs assume.

Note on the proposal: `actor_target` is an algorithm-side deepcopy (mpo.py:124) and is NOT saved
in checkpoints -- only `actor` is. This uses `policy.actor` with estep_sample_std_scale = 1.0,
asserted from the config; the two differ only by the Polyak/hard target lag.

Usage:
    python scripts/analysis/filter_probe.py --config <cfg> --checkpoint <ckpt> \
        --env_id SafetyPointGoal2-v0 --cell pointgoal2_a2_s2 --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import safety_gymnasium  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_rank_audit", REPO / "scripts" / "analysis" / "rank_consistency_audit.py"
)
_ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ra)

CONDITIONS = ("base", "greedyR", "randfilt", "filter25", "argminQc")


def parse_rules(spec: str) -> list[str]:
    """"filter:16,penalty:1.8" -> ["filter:16", "penalty:1.8"]. Empty -> the 5 legacy rules."""
    return [x.strip() for x in spec.split(",") if x.strip()] if spec else list(CONDITIONS)


@torch.no_grad()
def score(policy, obs_t: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(Q_r, Q_c) for K actions at one state. Q_r is the twin min, as `MPO._estep_sample` uses."""
    rep = obs_t.expand(actions.shape[0], -1)
    q1, q2 = policy.evaluate_q(rep, actions)
    q_r = torch.min(q1, q2).reshape(-1)
    q_c = policy.cost_critics[0](rep, actions).mean(dim=-1).reshape(-1)
    return q_r, q_c


REF_RULES = ("filter:1", "penalty:40", "argminQc")


def rule_distribution(rule: str, q_r: torch.Tensor, q_c: torch.Tensor, eta: float,
                      chosen: int) -> tuple[float, float]:
    """(KL of the rule's induced action distribution from q*_0, KL of the executed point mass).

    q*_0 = softmax(Q_r/eta) is the unconstrained E-step distribution over the K candidates. Every
    rule induces some distribution over those same candidates, so KL(p_rule || q*_0) puts both
    families on one axis in nats -- which is the honest common currency, since A is only a
    second-order proxy for exactly this displacement.

      penalty:LAM -> p = softmax((Q_r - LAM*Q_c)/eta), i.e. q*_LAM. KL is the quantity the
                     theory approximates as A^2/2.
      filter:M    -> p = q*_0 restricted to the M lowest-Q_c actions and renormalized. This is
                     the filter AS ORIGINALLY PROPOSED ("run the E-step over that support"); the
                     probe executes argmax Q_r within the support, whose displacement is the
                     point-mass figure instead.
      base        -> p = uniform over the K proposal draws.
    """
    logq0 = torch.log_softmax(q_r / max(eta, 1e-8), dim=0)
    kind, _, val = rule.partition(":")
    if kind == "penalty":
        logp = torch.log_softmax((q_r - float(val) * q_c) / max(eta, 1e-8), dim=0)
    elif kind == "filter":
        keep = torch.argsort(q_c)[:int(val)]
        mask = torch.full_like(q_r, float("-inf"))
        mask[keep] = 0.0
        logp = torch.log_softmax(logq0 + mask, dim=0)     # renormalise q*_0 on the survivors
    elif rule == "base":
        logp = torch.full_like(q_r, -float(np.log(q_r.numel())))
    else:
        logp = torch.full_like(q_r, float("-inf"))
        logp[chosen] = 0.0
    p = logp.exp()
    kl = float((p * (logp - logq0))[torch.isfinite(logp)].sum())
    kl_exec = float(-logq0[chosen])                        # KL of a point mass at the executed action
    return kl, kl_exec


def choose_swept(rule: str, q_r: torch.Tensor, q_c: torch.Tensor, eta: float,
                 gen: torch.Generator) -> int:
    """`filter:M` -> argmax Q_r among the M lowest Q_c. `penalty:LAM` -> sample the E-step softmax.

    The penalty family is the EXOGENOUS intervention on A: policy, critics and start states are
    all fixed, and lambda is set by hand, so A = lambda*std_a(Qc)/(eta*sqrt(2*eps)) is varied
    without the PID loop in the way. If cost falls as A rises, A is causally real; if cost is
    flat in A, it is inert and the theory is missing the trust-region rate limit.
    """
    kind, _, val = rule.partition(":")
    if kind == "filter":
        # SAMPLE from q*_0 restricted to the M lowest-Q_c actions -- "run the E-step over that
        # support", the filter as originally proposed. It previously took argmax Q_r among the
        # survivors, which is a POINT MASS, not the restricted distribution: at M = K that made
        # filter:K identical to greedyR (measured: both 46.4667 on cargoal1) while the KL axis
        # reported 0.000 for the un-restricted q*_0. Distribution and executed policy disagreed.
        # Sampling makes filter:K and penalty:0 the same policy, which `assert_family_endpoints`
        # now enforces, and keeps filter:1 = argmin Q_c (a single survivor).
        M = int(val)
        keep = torch.argsort(q_c)[:M]
        mask = torch.full_like(q_r, float("-inf"))
        mask[keep] = 0.0
        w = torch.softmax(q_r / max(eta, 1e-8) + mask, dim=0)
        return int(torch.multinomial(w, 1, generator=gen))
    if kind == "penalty":
        lam = float(val)
        w = torch.softmax((q_r - lam * q_c) / max(eta, 1e-8), dim=0)
        return int(torch.multinomial(w, 1, generator=gen))
    raise SystemExit(f"unknown rule {rule}")


def assert_family_endpoints(K: int, eta: float, seed: int = 0, trials: int = 200) -> None:
    """filter:K and penalty:0 are both "sample q*_0" and MUST pick the same action.

    Permanent harness check: they are the shared endpoint of the two families, so if they ever
    diverge the KL axis no longer describes the executed policy and the matched-KL comparison is
    meaningless. Runs on synthetic critic values, so it costs nothing and needs no env.
    """
    g = torch.Generator().manual_seed(seed)
    bad = 0
    for _ in range(trials):
        q_r = torch.randn(K, generator=g) * 2.0
        q_c = torch.randn(K, generator=g).abs() * 0.5
        s1 = torch.Generator().manual_seed(1234)
        s2 = torch.Generator().manual_seed(1234)
        if choose_swept(f"filter:{K}", q_r, q_c, eta, s1) != choose_swept("penalty:0", q_r, q_c, eta, s2):
            bad += 1
    if bad:
        raise SystemExit(
            f"HARNESS CHECK FAILED: filter:{K} and penalty:0 disagreed on {bad}/{trials} draws. "
            "They are the same distribution (q*_0) and must be the same policy; the KL axis is "
            "not describing what was executed."
        )


def choose(cond: str, q_r: torch.Tensor, q_c: torch.Tensor, keep: int,
           gen: torch.Generator) -> int:
    if cond == "base":
        return 0                                    # index 0 IS one draw from the proposal
    if cond == "greedyR":
        return int(torch.argmax(q_r))
    if cond == "argminQc":
        return int(torch.argmin(q_c))
    if cond == "randfilt":
        sub = torch.randperm(q_r.numel(), generator=gen, device=q_r.device)[:keep]
    elif cond == "filter25":
        sub = torch.argsort(q_c)[:keep]             # the `keep` LOWEST predicted cost
    else:
        raise SystemExit(f"unknown condition {cond}")
    return int(sub[torch.argmax(q_r[sub])])


def run_condition(env, policy, cond: str, episodes: int, K: int, keep: int,
                  horizon: int, device: str, base_seed: int, eta: float = 1.0) -> list[dict]:
    swept = ":" in cond
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)     # same seed per episode across conditions
        t, c_tot, r_tot, agree = 0, 0.0, 0.0, 0
        sd_qc, sd_qr, kls, kls_exec = [], [], [], []
        agree_f1 = 0
        ref_idx = {rr: [] for rr in REF_RULES}
        while True:
            obs_t = _ra._obs_with_u(obs, t, horizon, device)
            # Candidates depend only on (episode, step), so conditions see identical proposals
            # until their trajectories diverge.
            g = torch.Generator(device=device).manual_seed(base_seed * 1000003 + ep * 10007 + t)
            actions = _ra.sample_actions(policy, obs_t, K, g, "policy")
            q_r, q_c = score(policy, obs_t, actions)
            sd_qc.append(float(q_c.std()))
            sd_qr.append(float(q_r.std()))
            i = choose_swept(cond, q_r, q_c, eta, g) if swept else choose(cond, q_r, q_c, keep, g)
            kl_d, kl_e = rule_distribution(cond, q_r, q_c, eta, i)
            kls.append(kl_d)
            kls_exec.append(kl_e)
            # Counterfactual choices of the reference rules on THIS state stream: the three
            # diverge after their first differing action, so agreement can only be measured on a
            # shared stream, not by comparing separate episodes.
            for rr in REF_RULES:
                ref_idx[rr].append(choose_swept(rr, q_r, q_c, eta, g) if ":" in rr
                                   else choose(rr, q_r, q_c, keep, g))
            # Does THIS rule pick what filter:1 (== argminQc) would, at this same state?
            agree_f1 += int(i == ref_idx["filter:1"][-1])
            if cond in ("filter25", "randfilt"):
                agree += int(i == int(torch.argmax(q_r)))
            obs, rew, cost, term, trunc, _ = env.step(actions[i].cpu().numpy().reshape(-1))
            c_tot += float(cost)
            r_tot += float(rew)
            t += 1
            if term or trunc:
                break
        out.append({"ep": ep, "cost": c_tot, "reward": r_tot, "len": t,
                    "agree_with_greedyR": agree / max(t, 1),
                    "std_a_qc": float(np.mean(sd_qc)),
                    "std_a_qr": float(np.mean(sd_qr)),
                    "agree_with_filter1": agree_f1 / max(t, 1),
                    "kl_from_q0": float(np.mean(kls)),
                    "kl_exec_from_q0": float(np.mean(kls_exec)),
                    "ref_agree": {f"{a}|{b}": float(np.mean(np.array(ref_idx[a]) == np.array(ref_idx[b])))
                                  for i_, a in enumerate(REF_RULES) for b in REF_RULES[i_ + 1:]}})
    return out


def paired_ci(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 0) -> tuple:
    """Bootstrap CI for mean(a-b) over PAIRED episodes."""
    d = a - b
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--actions", type=int, default=64, help="K, must match sample_action_num")
    ap.add_argument("--keep", type=int, default=16, help="M kept by the filter rules")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rules", default="",
                    help='comma list, e.g. "filter:16,penalty:1.8"; empty = the 5 legacy rules')
    ap.add_argument("--eta", type=float, default=None,
                    help="E-step temperature for the penalty family; default = checkpoint's logged eta")
    ap.add_argument("--eps", type=float, default=None, help="logged kl_q, recorded for A")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default="outputs/filter_probe")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    cfg = _ra.load_cfg(args.config)
    env = safety_gymnasium.make(args.env_id)
    env.reset(seed=args.seed)
    horizon = int(env.spec.max_episode_steps)
    n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
    policy, blob, pol_cfg = _ra.build_policy(cfg, n_obs + 1, n_act, args.checkpoint, args.device)
    facts = _ra.preflight(cfg, pol_cfg, policy, env, n_obs)
    print(f"[{args.cell}] {args.env_id} K={args.actions} keep={args.keep} "
          f"episodes={args.episodes} iter={blob.get('iter')}")
    print(f"    proposal: {facts['actor_used']}; std_scale={facts['estep_sample_std_scale']}")

    rules = parse_rules(args.rules)
    eta = args.eta if args.eta is not None else 1.0
    assert_family_endpoints(args.actions, eta)
    print(f"    harness check: filter:{args.actions} == penalty:0 over 200 synthetic draws -- PASS")
    print(f"    rules: {rules}   eta={eta:.4f} eps={args.eps}")
    res = {}
    for cond in rules:
        res[cond] = run_condition(env, policy, cond, args.episodes, args.actions, args.keep,
                                  horizon, args.device, args.seed, eta)
        c = np.array([e["cost"] for e in res[cond]])
        r = np.array([e["reward"] for e in res[cond]])
        print(f"    {cond:<9} cost {c.mean():7.2f} +-{c.std()/np.sqrt(len(c)):5.2f}   "
              f"reward {r.mean():7.2f} +-{r.std()/np.sqrt(len(r)):5.2f}", flush=True)
    env.close()

    cost = {k: np.array([e["cost"] for e in v]) for k, v in res.items()}
    rew = {k: np.array([e["reward"] for e in v]) for k, v in res.items()}
    comps = [(c, "base") for c in rules if c != "base" and "base" in rules]
    if "filter25" in rules and "randfilt" in rules:
        comps.append(("filter25", "randfilt"))          # the legacy decisive comparison
    if "filter:1" in rules and "penalty:40" in rules:
        comps.append(("filter:1", "penalty:40"))        # do the two families meet at the extreme?
    summary = {"cell": args.cell, "env_id": args.env_id, "episodes": args.episodes,
               "K": args.actions, "keep": args.keep, "seed": args.seed,
               "iter": blob.get("iter"), "wall_clock_s": round(time.time() - t0, 1),
               "mean_cost": {k: float(v.mean()) for k, v in cost.items()},
               "mean_reward": {k: float(v.mean()) for k, v in rew.items()},
               "agree_with_greedyR": {k: float(np.mean([e["agree_with_greedyR"] for e in res[k]]))
                                      for k in ("randfilt", "filter25") if k in res},
               "std_a_qc": {k: float(np.mean([e["std_a_qc"] for e in v])) for k, v in res.items()},
               "std_a_qr": {k: float(np.mean([e["std_a_qr"] for e in v])) for k, v in res.items()},
               "agree_with_filter1": {k: float(np.mean([e["agree_with_filter1"] for e in v]))
                                      for k, v in res.items()},
               "kl_from_q0": {k: float(np.mean([e["kl_from_q0"] for e in v])) for k, v in res.items()},
               "kl_exec_from_q0": {k: float(np.mean([e["kl_exec_from_q0"] for e in v]))
                                   for k, v in res.items()},
               "ref_agree": {k: {kk: float(np.mean([e["ref_agree"][kk] for e in v]))
                                 for kk in v[0]["ref_agree"]} for k, v in res.items()},
               "eta": eta, "eps": args.eps,
               "paired": {}}
    print()
    for a, b in comps:
        dc, lc, hc = paired_ci(cost[a], cost[b], seed=args.seed)
        dr, lr, hr = paired_ci(rew[a], rew[b], seed=args.seed)
        summary["paired"][f"{a}-{b}"] = {"d_cost": dc, "cost_ci": [lc, hc],
                                         "d_reward": dr, "reward_ci": [lr, hr]}
        star = "  *" if (lc > 0) or (hc < 0) else ""
        print(f"    {a:>9} - {b:<9} dcost {dc:+7.2f} [{lc:+7.2f},{hc:+7.2f}]{star:<3} "
              f"dreward {dr:+7.2f} [{lr:+7.2f},{hr:+7.2f}]")
    out = REPO / args.out_dir / f"{args.cell if not args.tag else args.cell + '__' + args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": res}, indent=1, default=float) + "\n")
    print(f"-> {out.relative_to(REPO)}  ({summary['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
