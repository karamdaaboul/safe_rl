# FSRL reference CVPO: why it looks good on Bullet and bad on Safety-Gymnasium Circle

Date: 2026-08-08. Scope: the **reference** (FSRL) CVPO, not our own `safe_rl/algorithms/cvpo.py`.
Complements [cvpo-cost-critic-investigation.md](cvpo-cost-critic-investigation.md) (which
diagnoses *our* implementation on PointGoal1) and [m0-baselines.md](m0-baselines.md).

## What triggered this

FSRL CVPO on `SafetyCarCircle1Gymnasium-v0` (its auto-assigned `Mujoco2MCfg`, 2M steps,
3 seeds) reported test cost **66–97 against `cost_limit=25`** — read as a 3–4x constraint
failure that would fail the M0 gate. Two follow-ups were run and both were wrong-headed:
`estep_iter_num=5` (worse, see below) and extending 2M→3M (completely inert, cost flat at
~166→170). Both were stopped.

## Finding 1 — FSRL never validated this task

FSRL's published benchmark (`fsrl.readthedocs.io/en/latest/tutorials/benchmark.html`) is
18 envs: 7 Bullet-Safety-Gym + 5 SG-Velocity + 6 SG-Navigation. **The SG-Navigation six
are `Point` only** (Button1/2, Goal1/2, Push1/2). There is not one Safety-Gymnasium
*Circle* task and not one Safety-Gymnasium *Car* task in it; every Circle result they
publish is Bullet. `SafetyCarCircle1Gymnasium-v0 -> Mujoco2MCfg` is a lookup-table
default (`examples/mlp/train_cvpo_agent.py:43`), never tuned. The README's promised
"practical guide for tuning the key hyper-parameters" was never written (`docs/tutorials/`
has only `get_started.rst` and `benchmark.rst`). Budget is 2M where OmniSafe states 3M is
needed *for this env specifically* and where FSRL used 5M for its own published
PointGoal1 result.

**So there is no published number to reproduce here.** The M0 gate should be judged on
PointGoal1, which FSRL did publish.

## Finding 2 — the "violation" is mostly the deterministic-eval protocol

`train/cost` is the *stochastic* collection policy; `test/cost` uses
`deterministic_eval=True` (the mean action, `cvpo.py:239`). Tail-20 means:

| task | limit | train cost (stochastic) | test cost (deterministic) | ratio |
|---|---|---|---|---|
| Bullet CarCircle (γ=.97, H=300) | 10 | 10.2 / 11.3 / 9.3 | **0.00 / 0.90 / 0.07** | ~0.0 |
| SG CarCircle1 (γ=.995, H=500) | 25 | 24.1 / 28.1 / 25.5 | **73.3 / 96.9 / 66.1** | ~3.0 |
| SG PointGoal1 (γ=.995, H=1000) | 25 | 27.2 / 27.5 / 28.4 | 25.9 / 29.6 / 26.0 | ~1.0 |

In **all three** the constrained stochastic policy sits at its budget and
`val_q1 ≈ thres_q1` within 1%. CVPO is satisfying the constraint it optimizes. What
differs is only how the deterministic policy relates to it: free safety margin on Bullet,
neutral on Goal, ~3x penalty on SG Circle. FSRL's Bullet headline numbers inherit a margin
that SG Circle does not provide.

Confirmed by direct rollout of the final checkpoints (50 fixed-seed episodes/mode,
scratchpad `cvpo_det_vs_stoch.py`):

| seed | det `J_c` | stoch `J_c` | ratio | det disc | stoch disc | 1st/2nd half | clip% |
|---|---|---|---|---|---|---|---|
| 0 | 16.7 ± 18.1 | 4.5 ± 10.3 | 3.7 | 3.69 | 0.86 | 3.8 / 12.9 | 82 |
| 1 | 87.5 ± 45.7 | 27.6 ± 31.5 | 3.2 | 24.97 | 9.11 | 38.9 / 48.7 | 96 |
| 2 | 76.4 ± 30.4 | 38.6 ± 35.9 | 2.0 | 19.06 | 11.76 | 22.7 / 53.7 | 89 |

## Finding 3 — `qc_thres` is NOT the culprit here

`qc_thres = (cost_limit/H)(1−γ^H)/(1−γ)` (`fsrl/policy/cvpo.py:130-133`, `H =
env.spec.max_episode_steps`). Implied episodic budgets: **9.999** (Bullet) and **22.96**
(SG Circle1) — the SG threshold is 8% *stricter*, not lax. Logged `thres_q1` = 9.1843
matches the formula exactly. On PointGoal1 the same formula lands the reference at cost
~26 vs limit 25. This is a different mechanism from the qc_scale problem measured for our
own implementation in [cvpo-cost-critic-investigation.md](cvpo-cost-critic-investigation.md).

Two second-order inconsistencies do exist, both harmless on Bullet and not on SG Circle:
- Cost on Circle1 is **back-loaded** (1st/2nd half ≈ 1:1.3 to 1:3.4), so the uniform-cost
  premise behind the conversion is *lenient* here. (Contrast PointGoal1, measured
  near-uniform 0.483/0.517 — no contradiction, different task.)
- The truncation mask is commented out (`fsrl/policy/base_policy.py:499-501`,
  `value_mask = ~terminated`), so TimeLimit truncations still bootstrap and the cost
  critic estimates *infinite-horizon* cost while the threshold uses the H-truncated sum.
  Consistent only when γ^H ≈ 0 — true for Bullet (1.1e-4), false for SG Circle1 (0.0816).
  Circle tasks never terminate, only truncate.

## Finding 4 — the constraint is enforced against a stale buffer mixture

`val_q1` is pinned at 9.09 in **every** seed, but each seed's *current* policy has a very
different true discounted cost from `s₀`: 0.86 / 9.11 / 11.76. Seed 0's critic over-reads
by ~10x. `val_q1` averages `Q_c` over **replay-buffer** states, and `buffer_size=200000`
holds ~10 epochs of a policy whose cost oscillates 16→138. So the quantity being driven
to the threshold is decoupled from the policy being evaluated. The authors' own Car-Circle
script used `buffer_size=8000` with `episode_rerun_num=28` — near-on-policy, 25x smaller.

## Finding 5 — the policy is saturated (`unbounded=True`)

The mean action lies outside `[-1,1]` **82–96%** of steps. `Mujoco*Cfg` sets
`unbounded=True` (no tanh squash) with `action_bound_method="clip"`; **every Bullet config
uses `unbounded=False`**. The result is effectively bang-bang control — a strong candidate
for both the deterministic/stochastic divergence and the low reward.

**This is unreachable from the FSRL CLI.** `train_cvpo_agent.py:70-74` diffs CLI args
against `TrainCfg` and overlays the difference on the task config, so `--unbounded False`
(equal to the `TrainCfg` default) is dropped and then overwritten to `True` by
`Mujoco2MCfg`. Same trap for `--cost_limit 10` and `--gamma 0.97` on SG tasks. Use an
explicit-config runner.

## Finding 6 — we are also simply worse than the field

OmniSafe's published off-policy benchmark on this exact env, `cost_limit=25`, 3e6 steps:

| | reward | cost (limit 25) |
|---|---|---|
| TD3Lag | **34.38 ± 1.55** | **2.25 ± 3.90** |
| DDPGLag | 33.29 ± 6.55 | 20.67 ± 28.48 |
| SACLag | 31.42 ± 11.67 | 22.33 ± 26.16 |
| unconstrained DDPG/TD3/SAC | 43–45 | 372–407 |
| **FSRL CVPO (2M, deterministic)** | **~20** | **66–97** |

Worse on **both** axes, so the eval protocol is not the whole story. TD3Lag at cost 2.25
proves a *deterministic* policy can be safe here — the 3x gap is a deficiency, not an
inevitability. Note OmniSafe's own ± are ±28 to ±40 on this env: violent oscillation is
the known behaviour of this task, so 3-seed claims mean little.

Task geometry, for why Circle is the hard case (both suites use the same reward
`0.1·(pos×vel)/(1+|‖pos‖−R|)` and the same per-step out-of-bounds indicator):

| | Bullet CarCircle | SG CarCircle1 |
|---|---|---|
| radius R / wall x_lim | 7.0 / 6.0 | 1.5 / 1.125 |
| out-of-bounds arc on optimal circle | 34.4% | **46.0%** |
| H | 300 | 500 |
| cost if tracking optimal circle | ~103 | **~230** |
| DSRL max episode reward | 534.31 | **24.94** |

Reward *requires* violation, and the reward scale differs 21x while cost stays 0/1 — so
the λ that balances `Q_r − λQ_c` is on a different scale between suites, and
`estep_dual_max=20` may be mis-set. Ours ran at 0.8–1.7, not saturated.

## `estep_iter_num` is not a convergence knob — do not raise it

Measured: `estep_iter_num=5` drove λ to **0.037** (from 1.687), left train cost at the
limit (26.1) but widened the deterministic gap to **6.5x**, and cost reward (19.0 vs
20.0). It counts Adam steps at `estep_dual_lr=0.02` (`cvpo.py:346-352`) with the clamp
applied once *after* the loop — the reference solves its dual by gradient ascent, which is
the learned-dual scheme CLAUDE.md rule 1 rejects. More steps overshoot and drive λ to the
floor. Consistent with `m0-baselines.md:117-121`: the reference's realized cost is
insensitive to its threshold because the dual never tracks the target.

## Upstream bug: cost term subtracted twice

`fsrl/policy/cvpo.py:282-286` — `combined_q = q_values[0].detach()` shares storage with
`q_values[0]` (built under `no_grad`), and line 284 mutates it in place. By line 360 the
M-step target is computed from an already-modified `q_values[0]`, so the weights are
`softmax((Q_r − 2λQ_c)/η)` while the dual solved for λ under `Q_r − λQ_c`. Verified
independently by two investigations against upstream `main`; reproduced numerically
(10.0 → 4.0). Direction is over-conservatism for a given λ, which pushes λ down, which
then under-penalizes.

Consequence: the E-step and M-step are inconsistent, so the paper's Proposition 3
feasibility argument does not hold as derived. **Matters for the M2 gate** — if our
implementation is correct and theirs double-counts, the numbers will not align and the
difference must be attributed, not tuned away. Not filed upstream yet (repo has 7
issues+PRs total; PR #3, the one prior CVPO correctness fix, is already in our tree).

## Operational gotchas found the hard way

- **`progress.txt` has 37 header names but 38 data columns** from row 3 on (a constant
  `cost_limit` inserted at index 2). Reading by header name silently returns the wrong
  column and produced a plausible-looking but entirely wrong first result ("cost 19.98",
  which was actually the reward column). Parse positionally; cross-check the `Final eval`
  lines in the `.out` files.
- **No resume exists.** `TrainCfg.resume` is `False  # TODO`, `base_trainer.py:161`'s
  resume branch is commented out, and checkpoints hold only `{"model": state_dict}`.
  `estep_dual` is a plain tensor (`cvpo.py:144`), not an `nn.Parameter`, so λ/η are absent,
  as are the replay buffer and optimizer moments. Any "resume" is a warm start; λ/η can be
  re-injected from `progress.txt`.
- Bullet-Safety-Gym must come from **liuzuxin's gymnasium fork** (1.4.0). PyPI
  `bullet-safety-gym` 1.1.0 is SvenGronauer's original and registers into legacy `gym`, so
  `gymnasium.make("SafetyCarCircle-v0")` raises `NameNotFound`. The fork differs in every
  source file and shortens horizons (CarCircle 500→300).
- One MuJoCo CVPO run peaks ~25 GB of 31 GB; **two concurrent runs OOM-kill both**.
  GPU (`--device cuda`) gives only ~1.6x here — the nets are (128,128), so kernel-launch
  overhead dominates.

## Tuning arms

### Arm 1 — "revert to validated values" (combined): FAILED

`fsrl_runs/run_carcircle1_validated.sh`. K 16→32 (paper Table 1), γ 0.995→0.99 (paper),
buffer 200k→40k, `estep_kl` 0.02→0.1 (paper), `unbounded` True→False. 2M steps, GPU.

| arm | reward | cost (limit 25) | λ | entropy | `val_q1` vs `thres_q1` |
|---|---|---|---|---|---|
| baseline s0/s1 | 19.98 / 22.64 | 73.3 ± 34 / 96.9 ± 20 | 1.7 / 0.8 | 4.3 / 5.4 | 9.09 vs 9.18 — feasible |
| arm 1 s0/s1 | **6.96 / 4.58** | 56.9 ± 44 / 62.2 ± 69 | **14.0 / 15.5** | 10.2 / 11.4 | **6.87 vs 4.97 — INFEASIBLE** |

Reward collapsed 3–5x, cost barely improved, cost variance got *worse*. Classic reward
annihilation from a saturated multiplier (cap is `estep_dual_max=20`), the same bimodal
failure `cvpo-cost-critic-investigation.md` records for large `lambda_max`.

**Cause is γ, and the mechanism is the back-loading from Finding 3.** Dropping γ
0.995→0.99 cut `thres_q1` by 0.54x (9.18→4.97) but cut achieved `Q_c` by only 0.76x
(9.09→6.87), so the Q-space constraint became infeasible; λ ran to its cap, the E-step
weighted almost purely by cost, reward died, and the constraint *still* was not met. The
`qc_thres` formula assumes uniform cost; when cost is concentrated late, lowering γ
shrinks the uniform-derived threshold faster than it shrinks the real `Q_c`.

**The CVPO paper's γ=0.99 is not transferable to a task with back-loaded cost.** Keep
γ=0.995 on SG Circle. The other four changes remain untested — arm 1 confounded them.

Seed 2 of this arm **crashed at epoch 23** with `EOFError` on a worker `recv`
(`fast_collector.py:134` → `subproc.py:204`) — the worker-kill signature that
`m0-baselines.md` attributes to OOM. Partial data only; do not quote it.

### Arm 2 — `unbounded=False` alone (in flight)

`fsrl_runs/run_carcircle1_unbounded.sh`. Everything exactly `Mujoco2MCfg` except
`unbounded` True→False; CPU, to match the baseline's conditions. 2 seeds, ~2.4 h each.
Isolates the Finding-5 suspect. Target: OmniSafe's operating point, reward ≥31 at
deterministic cost ≤25.
