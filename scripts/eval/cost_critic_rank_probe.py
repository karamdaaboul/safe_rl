"""Rank-quality probe for a CVPO cost critic: does Q_c *order* actions correctly?

Why this and not the level-calibration probe it replaces
--------------------------------------------------------
With ``qc_target_ema`` the E-step threshold is derived as
``q_target = cost_lim * EMA[C_now] / EMA[J_c]``, where ``C_now`` is the critic's own mean
reading. The critic's level therefore cancels *exactly* -- if Q_c reads 2x low everywhere,
q_target reads 2x low too and the constraint is unchanged. The dual has become a feedback
controller on realized episodic cost, and the critic contributes only per-state discrimination.

Consequences, and they invert the old priorities:

* A **level** bug can no longer hurt the constraint. The old calibration gate
  (``MC ~ a*Q_c + b`` at ``a=1, b=0``) no longer decides anything and should not be
  reported as a pass/fail criterion.
* A **ranking** bug is now the only critic failure that can. The E-step weight is
  ``softmax_a((Q_r - lambda*Q_c)/eta)``: a per-state softmax over candidate *actions*.
  Adding any per-state constant to Q_c cancels in the normalisation. Only the ordering and
  relative spacing of Q_c *across the candidate actions at one state* reaches the policy.

So the statistic that matters is the **within-state** Spearman correlation between
``Q_c(s, a_i)`` and the true cost-to-go of each branched action -- not a pooled regression
over visited states, which is dominated by between-state variation the softmax never sees.

Method
------
Roll the policy out. At ``--states`` sampled timesteps, save the MuJoCo state, branch
``--actions`` actions sampled from the policy, and for each branch roll forward ``--horizon``
steps under the policy to get a Monte-Carlo discounted cost-to-go. Correlate within the state.

The save/restore is self-checked: for each state the first branch replays the *same* action
that was actually taken and asserts the resulting observation matches to ``--restore-tol``.
If that check fails the probe aborts rather than reporting numbers built on a broken restore.

Usage
-----
    python scripts/eval/cost_critic_rank_probe.py \
        --config config/safety_gymnasium_cvpo_both_lam4.yaml \
        --checkpoint logs/.../model_29999.pt \
        --states 256 --actions 8 --horizon 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import safety_gymnasium  # noqa: E402

import safe_rl.modules as sr_modules  # noqa: E402

from safe_rl.algorithms import CVPO  # noqa: E402
from scripts.eval.eval_safety_gymnasium import load_train_cfg, obs_shaping_env_kwargs  # noqa: E402


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks (average ranks for ties)."""
    if x.size < 2:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(1, v.size + 1, dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _build_policy(train_cfg: dict, num_obs: int, num_act: int, checkpoint: str | None):
    """Rebuild the policy/critics from the config and load a runner checkpoint into them.

    Bypasses SafetyGymnasiumVecEnv on purpose: it is always an *async* vector env (its sync
    path raises, because safety_gymnasium's sync vector env drops the cost channel), so the
    simulator lives in another process and cannot be snapshot/restored. For the flat
    `Safety*Goal*-v0` observation spaces a raw single env yields byte-identical observations,
    which is asserted against the config's expected dimension by the caller.
    """
    pol_cfg = dict(train_cfg["policy"])
    # Resolve the policy class from the config: the distributional arms use SafeActorCritic
    # with a 101-atom categorical head, and loading one into the other fails on head shape.
    policy_cls = getattr(sr_modules, pol_cfg.pop("class_name", "SafeSACActorCritic"))
    # Two config shapes are in the tree: the SAC-style ones give flat actor_hidden_dims /
    # critic_hidden_dims / activation, the distributional ones give nested *_kwargs blocks
    # directly. Synthesise the nested form only when it is absent.
    activation = pol_cfg.pop("activation", "relu")
    ah = pol_cfg.pop("actor_hidden_dims", [256, 256])
    ch = pol_cfg.pop("critic_hidden_dims", [256, 256])
    pol_cfg.setdefault("actor_kwargs", {"hidden_dims": ah, "activation": activation})
    pol_cfg.setdefault("critic_kwargs", {"hidden_dims": ch, "activation": activation})
    pol_cfg.setdefault("cost_critic_kwargs", {"hidden_dims": ch, "activation": activation})
    policy = policy_cls(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_act,
        **pol_cfg,
    )
    alg_cfg = dict(train_cfg["algorithm"])
    alg_cfg.pop("class_name", None)
    alg = CVPO(policy, device="cpu", **alg_cfg)
    if checkpoint:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = policy.load_state_dict(blob["model_state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"checkpoint is missing policy tensors: {sorted(missing)[:5]}")
        if unexpected:
            print(f"  (ignored {len(unexpected)} unexpected tensors in checkpoint)")
        print(f"loaded {checkpoint} (iter {blob.get('iter')})")
    else:
        print("WARNING: no --checkpoint; probing a randomly-initialised critic")
    policy.eval()
    return alg, policy


class _SimState:
    """Snapshot/restore of the underlying MuJoCo state.

    Restores ``qpos``/``qvel``/``act``/``time`` and re-runs the forward dynamics. Anything the
    task holds *outside* mjData -- goal placement after a goal is reached, hazard layout -- is
    not restored, which is why the caller self-checks every snapshot.
    """

    def __init__(self, env):
        self.env = env

    # Resolved lazily on every access: safety_gymnasium rebuilds the world (and therefore
    # mjModel/mjData) on each reset, so caching them hands back stale pointers after an
    # episode boundary.
    @property
    def data(self):
        return self.env.unwrapped.task.data

    @property
    def model(self):
        return self.env.unwrapped.task.model

    def _time_limit(self):
        """The TimeLimit wrapper in the chain, if present."""
        e = self.env
        while e is not None:
            if hasattr(e, "_elapsed_steps"):
                return e
            e = getattr(e, "env", None)
        return None

    def save(self) -> dict:
        d = self.data
        u = self.env.unwrapped
        tl = self._time_limit()
        return {
            "qpos": np.array(d.qpos, copy=True),
            "qvel": np.array(d.qvel, copy=True),
            "act": np.array(d.act, copy=True) if d.act.size else None,
            # ctrl and qacc_warmstart feed the next solver call, so omitting them makes two
            # "identical" branches diverge from step 0 -- measured before this was added.
            "ctrl": np.array(d.ctrl, copy=True) if d.ctrl.size else None,
            "qacc_ws": np.array(d.qacc_warmstart, copy=True) if d.qacc_warmstart.size else None,
            "time": float(d.time),
            # Episode bookkeeping lives outside mjData. Without restoring it, a branch that
            # ends an episode leaves the builder refusing to step ("must be reset"), and the
            # TimeLimit counter keeps advancing across branches so later branches get
            # truncated early -- which would silently shorten their Monte-Carlo returns.
            "terminated": bool(u.terminated),
            "truncated": bool(u.truncated),
            "steps_taken": int(getattr(u, "steps_taken", 0)),
            "elapsed": None if tl is None else int(tl._elapsed_steps),
            # The env's own np.random.RandomState drives per-step action noise and goal
            # respawn. Restoring it is what makes branches share a common random stream --
            # without it, two branches of the SAME action still diverge (measured: rollout
            # noise std 2.85, larger than the action effect being tested for).
            "rng": self._rng_state(),
        }

    def _rng(self):
        rg = getattr(self.env.unwrapped.task, "random_generator", None)
        return getattr(rg, "random_generator", None) if rg is not None else None

    def _rng_state(self):
        r = self._rng()
        return None if r is None else r.get_state()

    def restore(self, snap: dict) -> None:
        import mujoco

        d = self.data
        d.qpos[:] = snap["qpos"]
        d.qvel[:] = snap["qvel"]
        if snap["act"] is not None and d.act.size:
            d.act[:] = snap["act"]
        if snap.get("ctrl") is not None and d.ctrl.size:
            d.ctrl[:] = snap["ctrl"]
        if snap.get("qacc_ws") is not None and d.qacc_warmstart.size:
            d.qacc_warmstart[:] = snap["qacc_ws"]
        d.time = snap["time"]
        mujoco.mj_forward(self.model, d)

        u = self.env.unwrapped
        u.terminated = snap["terminated"]
        u.truncated = snap["truncated"]
        if hasattr(u, "steps_taken"):
            u.steps_taken = snap["steps_taken"]
        tl = self._time_limit()
        if tl is not None and snap["elapsed"] is not None:
            tl._elapsed_steps = snap["elapsed"]
        r = self._rng()
        if r is not None and snap.get("rng") is not None:
            r.set_state(snap["rng"])


def _act(policy, obs_t: torch.Tensor, deterministic: bool) -> torch.Tensor:
    with torch.no_grad():
        if deterministic and hasattr(policy, "act_inference"):
            return policy.act_inference(obs_t)
        return policy.act(obs_t)


def _report_decomposition(per_state, valid, qc_sp, n_actions: int) -> None:
    """One-way variance decomposition of the MC targets, and the de-attenuated ranking.

    Without this the ranking number cannot be judged: the MC target is itself noisy, so even a
    noiseless critic's observed rho ceilings at sqrt(reliability). At M=1 that ceiling is only
    ~0.52 -- a raw threshold would flag a perfect critic as broken.
    """
    reps = [np.asarray(r["reps"], float) for r in per_state if r.get("reps") is not None]
    reps = [m for m in reps if m.ndim == 2 and m.shape[1] >= 2]
    if not reps:
        return
    m_rolls = reps[0].shape[1]
    within = np.array([m.var(axis=1, ddof=1).mean() for m in reps])
    betw = np.array([m.mean(axis=1).var(ddof=1) * m_rolls for m in reps])
    sig = (betw - within) / m_rolls
    s_med, n_med = float(np.median(sig)), float(np.median(within))
    relia = s_med / (s_med + n_med / m_rolls) if s_med > 0 else float("nan")

    print(f"\n  --- variance decomposition (M = {m_rolls} rollouts/action) ---")
    print(f"  rollout-noise var (within) : {n_med:8.4f}  -> sd {np.sqrt(n_med):.4f}")
    print(f"  TRUE action-effect var     : {s_med:+8.4f}  -> sd {np.sqrt(max(s_med, 0)):.4f}")
    print(f"  states with signal > 0     : {float((sig > 0).mean()):.2f}   (0.50 = pure noise)")
    print(f"  target reliability         : {relia:.3f}")
    if np.isfinite(relia) and relia > 0:
        se = (1.0 / np.sqrt(max(n_actions - 1, 1))) / np.sqrt(max(valid.size, 1))
        print(f"  ceiling on observable rho  : {np.sqrt(relia):.3f}  (a PERFECT critic scores this)")
        print(f"  DE-ATTENUATED rho          : {valid.mean() / np.sqrt(relia):+.4f}  <-- judge on this")
        print(f"  detectable at 95%          : |rho| >= {1.96 * se / np.sqrt(relia):.3f}")
    print(
        "  critic expresses           : "
        f"{np.median(qc_sp) / max(np.sqrt(max(s_med, 0)), 1e-12):.1%} of the true action effect"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="omit to probe a randomly-initialised critic")
    p.add_argument("--env_id", default="SafetyPointGoal1-v0")
    p.add_argument("--states", type=int, default=256, help="branch points")
    p.add_argument("--actions", type=int, default=8, help="actions branched per state")
    p.add_argument("--horizon", type=int, default=300, help="MC rollout length per branch")
    p.add_argument("--gamma", type=float, default=None, help="defaults to the config's gamma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restore-tol", type=float, default=1e-5)
    p.add_argument(
        "--repeats",
        type=int,
        default=8,
        help=(
            "Rollouts per action. DEFAULT 8: at M=1 the target reliability is only 0.27, so a "
            "PERFECT critic ceilings at rho_obs ~ 0.52 and any raw threshold misjudges it.  With R > 1 the probe reports a one-way variance "
            "decomposition: MS_between and MS_within separate the true action effect from "
            "rollout noise WITHOUT needing bit-reproducible branches. This is the estimator "
            "that answers the question -- at R=1 the noise floor (~2.4) swamps the effect."
        ),
    )
    p.add_argument(
        "--crn",
        action="store_true",
        help=(
            "COMMON RANDOM NUMBERS: reseed the policy RNG identically before every branch, so "
            "all branches from one state share the same stochastic stream. The rollout noise "
            "then cancels in the paired differences and std_a(MC) isolates the effect of the "
            "initial action. Without this the noise floor (~2.4) swamps the action effect and "
            "the ranking test is uninformative -- verified with --same-action."
        ),
    )
    p.add_argument(
        "--same-action",
        action="store_true",
        help=(
            "NOISE CONTROL: branch the SAME (deterministic) action every time instead of "
            "sampling. std_a(MC) is then pure rollout noise, which is the floor any "
            "single-sample ranking test has to beat. Without this control a rho ~ 0 cannot "
            "be attributed to the critic -- it may just mean the MC targets are noise."
        ),
    )
    p.add_argument("--out", default=None, help="write per-state records as JSON")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_cfg = load_train_cfg(args.config)
    gamma = args.gamma if args.gamma is not None else float(train_cfg["algorithm"].get("gamma", 0.99))

    env = safety_gymnasium.make(args.env_id)
    raw_obs_dim = int(np.prod(env.observation_space.shape))
    # FH-DCMPO arms append the normalized remaining horizon u_t = (T - t)/T to the observation
    # (HorizonAugmentedVecEnv). The raw single env this probe uses does not, so the column has to
    # be reproduced here or the checkpoint will not even load (61 vs 60 inputs).
    # load_train_cfg() keeps only the algorithm/policy/runner blocks, so the raw YAML is reread
    # for the env block the observation width depends on.
    with open(args.config, encoding="utf-8") as _f:
        _raw_cfg = yaml.safe_load(_f)
    obs_kwargs = obs_shaping_env_kwargs(_raw_cfg.get("env", {}) or {}, args.config)
    horizon_feature = bool(obs_kwargs.get("horizon_feature", False))
    if bool(obs_kwargs.get("budget_feature", False)):
        raise SystemExit("budget_feature is not reproduced by this probe; only horizon_feature is")
    max_ep = int(getattr(env.spec, "max_episode_steps", None) or 1000)
    num_obs = raw_obs_dim + (1 if horizon_feature else 0)
    if horizon_feature:
        print(f"  horizon column reproduced: u_t = (T - t)/T with T = {max_ep}")
    num_act = int(np.prod(env.action_space.shape))
    _, policy = _build_policy(train_cfg, num_obs, num_act, args.checkpoint)
    sim = _SimState(env)  # handles resolve lazily; the world does not exist until reset()

    # Episode step counter, in a holder so the closures below can mutate it. It is part of the
    # branch state: every sim.restore() must rewind it too, or the horizon column drifts.
    tick = [0]

    def _aug(o_np) -> torch.Tensor:
        o = np.asarray(o_np, dtype=np.float32).reshape(-1)
        if horizon_feature:
            u = (max_ep - min(tick[0], max_ep)) / float(max_ep)
            o = np.concatenate([o, np.asarray([u], dtype=np.float32)])
        return torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)

    def step(a_t: torch.Tensor):
        """One env step from a [1, A] action tensor -> (obs[1,O], cost, done)."""
        a = a_t.detach().cpu().numpy().reshape(-1)
        o, _r, c, term, trunc, _i = env.step(a)
        tick[0] += 1
        return (_aug(o), float(c), bool(term or trunc))

    obs_np, _ = env.reset(seed=args.seed)
    tick[0] = 0
    obs = _aug(obs_np)

    per_state = []
    restore_failures = 0
    disc = gamma ** np.arange(args.horizon, dtype=np.float64)

    for i in range(args.states):
        # Advance between branch points so consecutive states are not near-duplicates.
        for _ in range(8):
            obs, _c, done = step(_act(policy, obs, deterministic=False))
            if done:
                obs_np, _ = env.reset()
                tick[0] = 0
                obs = _aug(obs_np)

        snap = sim.save()
        tick_at_branch = tick[0]

        _sim_restore = sim.restore

        def sim_restore(sn, _r=_sim_restore, _t=tick_at_branch):
            _r(sn)
            tick[0] = _t

        sim.restore = sim_restore
        obs_at_branch = obs.clone()

        # Restore self-check: replay one fixed action twice from the snapshot and require the
        # resulting observations to agree. If the snapshot does not fully determine the
        # dynamics, every number below is meaningless -- so this aborts rather than warns.
        probe_a = _act(policy, obs_at_branch, deterministic=True)
        sim.restore(snap)
        o1, c1, _ = step(probe_a)
        sim.restore(snap)
        o2, c2, _ = step(probe_a)
        if float((o1 - o2).abs().max()) > args.restore_tol or abs(c1 - c2) > args.restore_tol:
            restore_failures += 1

        qc_vals, mc_vals, mc_reps = [], [], []
        for _j in range(args.actions):
            sim.restore(snap)
            # Draw the branch action BEFORE any CRN reseed. Reseeding first pins the actor's
            # sampling stream too, so every branch would draw the IDENTICAL action -- which
            # silently collapses std_a(Q_c) to 0, leaves rho undefined at every state and
            # reports "0% of the true action effect" no matter how good the critic is.
            a_j = _act(policy, obs_at_branch, deterministic=args.same_action)
            if args.crn:
                # Same ROLLOUT stream for every branch at this state (varied across states), so
                # the paired differences isolate the effect of the initial action.
                torch.manual_seed(args.seed * 1_000_003 + i)
            with torch.no_grad():
                qc = float(policy.evaluate_cost_q(obs_at_branch, a_j)[0, 0])

            costs = np.zeros(args.horizon)
            a_step = a_j
            for t in range(args.horizon):
                o, c, d = step(a_step)
                costs[t] = c
                if d:
                    break
                a_step = _act(policy, o, deterministic=False)
            reps = [float((costs * disc).sum())]
            for _r in range(args.repeats - 1):
                sim.restore(snap)
                if args.crn:
                    torch.manual_seed(args.seed * 1_000_003 + i)
                c2 = np.zeros(args.horizon)
                a2 = a_j
                for t in range(args.horizon):
                    o, c, d = step(a2)
                    c2[t] = c
                    if d:
                        break
                    a2 = _act(policy, o, deterministic=False)
                reps.append(float((c2 * disc).sum()))
            qc_vals.append(qc)
            mc_reps.append(reps)
            mc_vals.append(float(np.mean(reps)))

        # Leave the env on a well-defined state for the next outer iteration.
        sim.restore(snap)
        sim.restore = _sim_restore

        qc_a = np.asarray(qc_vals)
        mc_a = np.asarray(mc_vals)
        per_state.append(
            {
                "reps": mc_reps if args.repeats > 1 else None,
                "spearman": _spearman(qc_a, mc_a),
                "qc_spread": float(qc_a.std()),
                "mc_spread": float(mc_a.std()),
                "qc_mean": float(qc_a.mean()),
                "mc_mean": float(mc_a.mean()),
            }
        )
        if (i + 1) % 32 == 0:
            r = np.array([x["spearman"] for x in per_state], dtype=np.float64)
            print(f"  {i+1}/{args.states} states, running mean rho = {np.nanmean(r):+.3f}")

    if restore_failures:
        print(f"ABORT: MuJoCo restore self-check failed on {restore_failures} states.")
        return 2

    rho = np.array([r["spearman"] for r in per_state], dtype=np.float64)
    valid = rho[~np.isnan(rho)]
    qc_sp = np.array([r["qc_spread"] for r in per_state])
    mc_sp = np.array([r["mc_spread"] for r in per_state])

    dead = int((mc_sp <= 0.0).sum())
    print("\n=== within-state action ranking (the statistic the softmax actually sees) ===")
    print(f"  states probed             : {len(per_state)}")
    print(f"  no-signal states          : {dead}  ({dead / max(len(per_state), 1):.1%})")
    print("      (no branch from these states incurred any cost, so Q_c cannot be scored on")
    print("       them and the E-step's cost term is inert there whatever lambda does)")
    print(f"  states with a defined rho : {valid.size}/{len(per_state)}")
    print(f"  mean Spearman rho         : {valid.mean():+.4f}")
    print(f"  median                    : {np.median(valid):+.4f}")
    print(f"  frac rho > 0              : {float((valid > 0).mean()):.3f}   (0.5 = coin flip)")
    print(f"  median std_a(Q_c)         : {np.median(qc_sp):.5f}")
    print(f"  median std_a(MC)          : {np.median(mc_sp):.5f}")

    _report_decomposition(per_state, valid, qc_sp, args.actions)
    print("\n  Reference: a critic that cannot rank actions gives rho ~ 0 and the E-step's")
    print("  cost term is noise, whatever lambda does. The level is deliberately NOT gated")
    print("  here -- qc_target_ema cancels it. See codex/cvpo-cost-critic-investigation.md.")

    if args.out:
        Path(args.out).write_text(json.dumps(per_state, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
