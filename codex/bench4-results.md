# bench4 — SAC / MPO / SafeSAC / CVPO on SafetyPointGoal1-v0

*2026-07-23/24 (Claude Code), local box. One seed (1). 8 envs, UTD 0.5 (4 updates/iter),
batch 256, cost limit 25 for the safe algos. Runs stopped at plateau, not at the full
5M-step budget. Configs: `config/bench4/*.yaml`; logs/checkpoints: `logs/bench4/`;
wandb offline runs in `wandb_logs/` (SAC=4lbyd0q0, SafeSAC=vr1pjlpu, MPO=csxb69eb
synced; CVPO=hrs45mlc + final run unsynced).*

## Results

| algorithm | num_envs | reward | episode cost / limit | env steps | iters | wall clock |
|---|---|---|---|---|---|---|
| SAC | 8 | 28.1 | not logged¹ / — | 2.1M | 261k | 5h10m |
| MPO | 8 | 28.6 ± 0.1 | not logged¹ / — | 2.1M | 264k | 9h26m |
| SafeSAC | 8 | 0.7 ± 0.1 | 67 ± 19 / 25 | 2.2M | 273k | 5h29m |
| CVPO | 8 | 26.1 ± 0.3 | 27.7 ± 2.5 / 25 | 3.4M | 429k | 19h51m |

¹ training logs don't record cost for unconstrained algos (measurable via checkpoint eval).
± = temporal sd over the final ~40k iterations of a *single* run (operating-point
stability / oscillation amplitude), **not** across-seed variance (n=1 seed).
Wall clock on a shared box (2 concurrent foreign trainings) — indicative only.

## Key findings

1. **CVPO: 93% of unconstrained reward at near-limit cost.** R 26.1 vs the MPO/SAC ceiling
   of ~28.4, C 27.7 vs limit 25 (still drifting down when stopped). Beats the FSRL
   reference on the same task (reference tail: R≈20.6, C≈27.1) — plausibly our 64 vs 16
   E-step candidate actions.
2. **SafeSAC vs CVPO isolates multiplier dynamics as the failure cause.** Identical
   critics/replay/reward path (shared code post-refactor); only the constraint mechanism
   differs. PID multiplier: λ pinned 14–16, reward collapsed 18→0.7, cost never below ~50
   — lost the task AND the constraint. Same failure signature as the *reference* FSRL
   SAC-Lag in M0 (two independent implementations) → SAC-Lag-style outer-loop multipliers
   fail on PointGoal1, period. Per-batch dual: works.
3. **MPO ≥ SAC unconstrained** (28.6 vs 28.1, and faster per-iteration learning early) —
   the EM machinery costs nothing in final performance; ~2× wall clock per iteration.
4. **CVPO's residual pathology: λ bang-bang.** In dual mode λ flips between the bounds
   (1e-6 ↔ 20) rather than settling interior; cost oscillates ±2.5 at the operating point.
   This is the violation-oscillation amplitude the C-TruDi exact-solve dual + improvement
   budgets (§6.2) target. Baseline number to beat: **±2.5 at C≈27.7**.

## CVPO config archaeology (3 attempts, all logged in logs/bench4/)

| attempt | E-step ε | λ mechanism | outcome |
|---|---|---|---|
| `cvpo_eps0.1_weak.out` | 0.1 | grad (lr 0.03) | λ stalled ≈0.4, cost stuck ~52: loose ε lets the reweighting satisfy the constraint on paper while the policy stays unsafe |
| `cvpo_eps0.02_gradlam.out` | 0.02 | grad (lr 0.1) | safer early (−20 cost at matched iters); superseded at 9k |
| **final (`cvpo.out`)** | **0.02** | **dual (SLSQP, λmax 20)** | R 26.1 / C 27.7 — table row |

Lesson: **E-step ε is the safety-critical hyperparameter** (paper's 0.02, not 0.1), and
γ=0.995 + n_step=3 (FSRL-matched; exercises the refactor's n-step cost aggregation).

## Caveats / next steps

- n=1 seed — spec requires ≥5 for reported numbers. CVPO + MPO reruns are the priority.
- SAC/MPO true cost (and all-algo goals-reached) needs the ≥50-episode fixed-seed
  checkpoint eval ([[safety-goal-eval-gotchas]]).
- CVPO was still improving slowly at stop (cost 35→28 over prior 80k iters); the full
  625k budget might reach C≤25.

Cross-refs: [[safe-sac-inheritance-refactor]] (the shared-lineage refactor this validates),
[[m0-baselines]] (FSRL reference numbers), [[cvpo-negative-result.md]] (CarGoal2 bang-bang).
