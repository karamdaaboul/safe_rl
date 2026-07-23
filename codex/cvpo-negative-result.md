# CVPO on SafetyCarGoal2 — negative result + dual diagnosis

*2026-07-05, local overnight run (Claude Code). Question: does CVPO (off-policy EM safe RL,
Liu et al. ICML 2022, arXiv:2201.11927) move the reward/cost frontier on a level-2 Safety-Gymnasium
task where penalty methods (P3O, PPO-Lag) cannot?*

## TL;DR

**No. CVPO as implemented did not enforce the cost constraint on SafetyCarGoal2-v0.** Over a full
100k-iteration run (16 envs, ~1.5M steps, 8.5 h, GPU) the episodic cost stayed at **100–150** the
entire time — never within 4–6× of the limit of **25**. It landed in the same "engaged + unsafe"
corner P3O found (goals ~6–8/ep, reward ~19). The frontier did not move.

The per-batch cost multiplier λ went **bang-bang** — every batch it snapped to either its floor
(1e-6) or its ceiling (1e5), never a regulating interior value. This is *correct solver behaviour*
given the setup, not a coding bug: it is the signature of a constraint that is **infeasible w.r.t.
the sampled action support**.

Run: `logs/safety_gymnasium/SafetyCarGoal2-v0/CVPO/20260704_215959/` (checkpoints every 5k to
`model_99999.pt`); wandb offline `offline-run-20260704_220003-paisdgyp`.

## The trace (every ~6k iters)

| iter | reward | episode cost | goals/ep | λ (mean=max) |
|---|---|---|---|---|
| 7k  | 7.1  | 149 | 6.4 | 1e5 (ceiling) |
| 13k | 17.1 | 141 | 6.4 | 1e-6 (floor) |
| 25k | 19.6 | 134 | 8.8 | 1e-6 |
| 43k | 20.1 | 118 | 8.8 | 1.8e-5 |
| 61k | 16.3 | 109 | 6.6 | 1e5 |
| 79k | 19.7 | 125 | 8.6 | 1e-6 |
| 97k | 19.0 | 120 | 6.8 | 1e5 |
| 99k | 18.6 | 146 | 6.2 | 1e5 |

Cost never trends toward 25. λ never settles. No safe checkpoint exists, so the honest
≥50-episode goals-counted eval was not run (every checkpoint is unsafe by 4–6×).

## Why the dual goes bang-bang (diagnosis)

The E-step dual (`cvpo.py:_solve_dual`) minimises
`g(η,λ) = η·ε + λ·qc_thres + η·mean_s log mean_a exp((Q_r − λ Q_c)/η)` over η,λ ≥ 0.

Its gradient in λ is **`dg/dλ = qc_thres − E_q[Q_c]`** (the η cancels). So:

- If `E_q[Q_c] > qc_thres` for *all* λ in bounds → `dg/dλ < 0` everywhere → minimiser drives **λ → upper bound (1e5)**.
- If `E_q[Q_c] < qc_thres` already near λ=0 → `dg/dλ > 0` → minimiser drives **λ → lower bound (1e-6)**.

An *interior* λ (i.e. real regulation) exists only when the non-parametric reweighting of the N
sampled actions can hit `E_q[Q_c] = qc_thres` exactly. Here it essentially never can:

- `qc_thres ≈ 2.5` (episodic 25 → discounted cost-Q scale, γ=0.99, H=1000).
- The policy's own samples have `Q_c ≈ 12` (episodic cost ~120 ⇒ ~0.12/step ⇒ 0.12·(1−γ^H)/(1−γ) ≈ 12).
- **All 64 candidate actions come from the current, already-unsafe policy.** No reweighting of
  a set of uniformly-unsafe actions reaches a 5×-safer target. So most batches are *infeasible*
  (λ→1e5); a few borderline batches read as slack (λ→1e-6). The flip-flop cancels: λ=1e5 for one
  batch makes the M-step chase the least-bad of 64 still-unsafe actions (marginal, non-cumulative),
  then λ=1e-6 the next batch chases pure reward and undoes it. Net constraint pressure ≈ 0.

**Root cause is task-level, not just numerical:** on CarGoal2 the goals sit *inside* the hazard
field, so goals-reached and cost are the same axis. A method that can only reweight actions drawn
from its own unsafe policy cannot escape that frontier any better than a penalty method — it just
needs the policy to have *been* safe first. Our policy went unsafe during warmup (random steps +
uninformed early cost critic) and the on-policy candidate set can't recover.

## What would actually help CVPO (untested hypotheses)

Ranked by expected payoff. The code is correct and stable (ran 8.5 h, no crash, no std-collapse —
var-KL trust region held std ~0.96 throughout; `Alpha 1e-8`, `Noise std` field is the unused
SAC slot and can be ignored).

1. **Graded λ instead of per-batch bang-bang** — solve η via the dual for fixed λ, and update λ by
   slow *projected* gradient ascent `λ ← clip(λ + η_λ·(E_q[Q_c] − qc_thres), 0, λ_max)` with a
   moderate `λ_max` (~100) and warm-start continuity. This integrates the violation so a single
   slack batch can't collapse λ to the floor, keeping sustained pressure. It reintroduces a mild
   "multiplier catch-up" but trades bang-bang for a controller that actually regulates. (Implemented
   as `lambda_mode: "grad"` in `cvpo.py`; unit-tested. Whether it beats the frontier on this task is
   still open — the infeasible-sampling problem is fundamental.)
2. **Keep the policy from ever going deep-unsafe** — start cost pressure *before* the reward critic
   dominates (higher initial λ / lower `qc_thres` ramp / shorter warmup), so the candidate set stays
   near-feasible. Once feasible, the interior dual works as designed.
3. **Widen the candidate support** — more `sample_action_num`, guard policy std from collapsing, so
   safer actions appear in the sample set.

## Strategic conclusion

Do **not** rely on CVPO (or any own-policy-reweighting / penalty / Lagrangian method) to *find* safe
behaviour from an unsafe start on a level-2/3 task. The reliable path is **runtime shielding** that
decouples skill from safety — the CBF projection (`safe_rl/cbf/`, config
`safety_gymnasium_p3o_cbf.yaml`) or the learned reachability filter (`safe_rl/filters/`). Shielded
P3O run launched on CarGoal2 alongside this note; see `codex/cbf-shielded-run.md` for results.

Cross-refs: [[distributional-cost-critic.md]] (frontier confirmed under P3O + HL-Gauss cost critic).
