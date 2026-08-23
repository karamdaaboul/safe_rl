# Our MPO vs. DeepMind Acme's reference MPO loss

*2026-07-30 (Claude Code). User question: compare `safe_rl/algorithms/mpo.py` against
[acme/jax/losses/mpo.py](https://github.com/google-deepmind/acme/blob/master/acme/jax/losses/mpo.py)
on four axes — (1) E-step KL, (2) M-step KL, (3) action clipping, (4) tanh handling — then run an
A/B on mjlab Ant-Flat.*

## TL;DR

On the four axes asked about, our implementation is **mathematically equivalent or stronger**.
The real deviations are elsewhere: three in the M-step and two in what the E-step is fed.

| Axis | Verdict |
|---|---|
| 1. E-step KL / temperature | **Same dual objective**, solved exactly (SLSQP) instead of one gradient step. Ours is stronger. |
| 2. M-step KL | Same decomposition and direction, but we use a **scalar** multiplier where Acme uses **per-dimension**, we **cap** the multipliers where Acme only floors them, and we use a **coupled** weighted-MLE where Acme decouples it. |
| 3. Action clipping | Different by design and self-consistent. Acme needs MO-MPO out-of-bound penalization because it samples a raw Gaussian; we tanh-squash, so out-of-bound actions cannot occur. Ours is cleaner. |
| 4. tanh log-prob correction | **Correct as written — do not "fix" it.** Omitting the Jacobian is exactly right (see below). |

## 1. E-step KL — equivalent objective, better solver

Acme's `compute_weights_and_temperature_loss`:
`loss_temperature = temperature * (epsilon + mean_s logsumexp_a(Q/temp) - log N)`, with
`temperature = softplus(log_temperature) + 1e-8` a **trainable parameter** taking one optimizer
step per update, floored at `log_temperature >= -18`.

Ours (`mpo.py:_solve_eta`): `g(eta) = eta*eps + eta*mean_s[log mean_a exp(Q/eta)]`, minimized by
SLSQP to optimality every update, warm-started, bounded `[1e-6, 1e6]`.

`logsumexp - log N == log mean exp`, so these are the **same function**. (We avoid the common
`sum`-instead-of-`mean` bug.) Weights are `softmax(Q/eta, dim=0)` in both; both share one scalar
`eta` across the batch and sample candidates from the target actor. The exact solve is the
C-TruDi rule-1 behaviour ("duals are computed, never learned").

Confirmed empirically: `dg/deta = eps - KL(q*||pi_old)` exactly (derivation in
`tests/test_mpo.py::test_mpo_dual_optimum_satisfies_kkt_on_eta`), and in a live Ant-Flat run the
logged `dual_residual_eta` sits at ~7.6e-4 against `eps_dual = 0.1`, with SLSQP converging in
2 iterations from the warm start.

## 2. M-step KL — same decomposition, three real deviations

Shared: `KL(old || new)`, split into `kl_mean = KL(N(mu_old,s_old) || N(mu_new,s_old))` and
`kl_std = KL(N(mu_old,s_old) || N(mu_old,s_new))`; multiplier sign `alpha += lr*(kl - eps)`;
trust region measured against the target actor.

**(a) Scalar vs. per-dimension multiplier.** Acme defaults to `per_dim_constraining=True`: KL keeps
shape `[B, D]` and `alpha_mean`/`alpha_stddev` are length-`D` vectors, one budget and one
multiplier per action dimension. Its docstring says to keep it on for best control-suite results.
We did `.sum(dim=-1).mean()`, i.e. Acme's non-default `per_dim_constraining=False`. Consequences:
one saturated dimension can absorb the whole budget, and `eps_kl_mean=0.01` bounds the **sum over
D dims**, so it is D× tighter than a per-dim epsilon of the same number.

**(b) Hard caps on the multipliers.** Acme only floors its duals (`log_alpha >= -18`), never caps
them, so a persistently violated KL keeps tightening. We clip to `[0, alpha_mean_max=0.1]` and
`[0, alpha_var_max=10.0]`. If `kl_mean` persistently exceeds `eps_kl_mean`, our mean trust region
silently stops tightening. `kl_mean_rel` is now logged precisely to detect this.

**(c) Coupled vs. decoupled weighted MLE.** Acme computes the weighted log-prob **twice** and sums
— once under `Normal(online_mean, target_std)`, once under `Normal(target_mean, online_std)` — so
the mean gradient is scaled by the *old* `1/sigma^2` and the std gradient is taken at the *old*
mean. We used a single `Normal(mean, std)`, coupling them: as `std` shrinks the mean gradient is
amplified by `1/std^2`. Acme decouples the *loss*, not just the KL; we had only decoupled the KL.
Note the decoupled form is exactly 2× the coupled one at the identity point (unit-tested) — the
same effective-scale doubling Acme gets from summing its two weight sets.

Minor: Acme takes 1 gradient step per E-step, we take `mstep_iteration_num=5` with weights frozen.

## 3. Action clipping — different, and ours is cleaner

Acme's policy is a raw diagonal Gaussian with no squashing, so sampled actions can leave `[-1, 1]`.
It handles that with MO-MPO action penalization (on by default): cost `-||a - clip(a,-1,1)||`
pushed through a **second, independent temperature dual** with `epsilon_penalty=0.001`, whose
weights are then **added** to the main weights (`normalized_weights += penalty_normalized_weights`,
so the combined weights sum to 2 per state and the policy-loss scale doubles when it is on).

We sample pre-tanh `x` and squash `a = b + c*tanh(x)`, so actions are in bounds by construction and
no penalty is needed — as `mpo.py`'s docstring already claimed. There is no action `clamp` anywhere
in `mpo.py`/`cvpo.py` or the runner, and none is required.

The residual risk our parameterization introduces instead is **tanh saturation**: `Q` is flat in `x`
once `tanh(x)` saturates, so nothing in the objective penalizes the pre-tanh mean drifting outward,
and only the (capped, see 2b) mean-KL limits per-step drift. Acme's action penalization is what
suppresses the equivalent drift in raw space. We now log `pretanh_mean_absmax` and
`frac_saturated` as the substitute diagnostic.

## 4. tanh log-prob correction — correct as written

The M-step evaluates `dist.log_prob(x)` on **pre-tanh** samples with no tanh Jacobian, while
`StochasticActor.sample` does carry the full correction. This looks like an omission and is not:

- `log pi_theta(a) = log N_theta(x) - log|da/dx|`, and `log|da/dx|` depends only on `x`, not on
  `theta`. In the weighted MLE it is a `theta`-independent additive constant, so **the gradients
  are identical** with or without it.
- KL is invariant under a bijection applied to both arguments, so `KL(N_old || N_theta)` in
  pre-tanh space **equals** the KL between the squashed action distributions. Both `eps_dual` and
  `eps_kl_mean`/`eps_kl_var` therefore bound exactly what we intend.
- The non-parametric `q` is a discrete reweighting of N samples, so no density enters it.

Adding the Jacobian would change nothing and only add noise. Acme has no tanh at all, so there is
nothing to compare against — this is our design, and it is sound.

## 5. Two E-step input choices not present in Acme

- **Online critic.** `policy.evaluate_q` uses `critic_1`/`critic_2`, not their targets; Acme scores
  the target-policy samples with the **target** critic.
- **`min(Q1, Q2)` pessimism.** Acme takes the critic mean. Ours biases the E-step weights against
  high-variance (exploratory) actions, which compounds with the saturation risk in §3.

## What changed in the code

All new behaviour is **default-off**, so existing runs and results are unaffected.

- `mpo.py` / `cvpo.py`: four new options — `per_dim_constraining`, `decoupled_mstep`,
  `estep_use_target_critic`, `estep_q_reduction` (`"min"`/`"mean"`). The M-step multipliers became
  arrays of shape `[1]` or `[num_actions]` so both trust-region modes share one code path.
- New diagnostics (CLAUDE.md mandates most of these; their absence is why
  [[cvpo-negative-result]] had to be diagnosed post-hoc): `kl_q`, `kl_q_rel`,
  `dual_residual_eta`, `ess`, `ess_min`, `kl_mean_rel`, `kl_var_rel`, `pi_std_min/max/cond`,
  `pretanh_mean_absmax`, `frac_saturated`, `solver_status`, `solver_iters`. Helpers
  `nonparametric_kl_from_weights` / `effective_sample_size` live in `mpo.py` and are imported by
  `cvpo.py`.
- **Bug fix:** the E/M steps called `self.policy.actor(obs)` directly, bypassing
  `policy.actor_obs_normalizer`, while `policy.act()` applies it — so MPO/CVPO trained on a
  different input scale than they acted on whenever `actor_obs_normalization: true`. Now normalized
  once per update. No-op (Identity) for every previously shipped config, which all left it off.
- **Runner:** `off_policy_runner.py` only merged `get_penalty_info()` for safe-RL algorithms, so
  unconstrained MPO's diagnostics were silently dropped. Scalar keys are now merged for any
  algorithm that exposes the hook.

## The A/B on mjlab Ant-Flat

Two arms, identical except the four flags, 256 envs, 100k iterations (~25.6M env steps), one per
GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, arm A on the Blackwell, arm B on the Ada):

| | `config/mjlab_ant_mpo.yaml` (A) | `config/mjlab_ant_mpo_acme.yaml` (B) |
|---|---|---|
| `per_dim_constraining` | false | true |
| `kl_mean_constraint` / `kl_var_constraint` | 0.01 / 1e-4 (summed over 8 dims) | 0.00125 / 1.25e-5 (per dim — same total budget) |
| `alpha_mean_max` / `alpha_var_max` | 0.1 / 10 | 10 / 1000 (100×) |
| `decoupled_mstep` | false | true |
| `estep_use_target_critic` / `estep_q_reduction` | false / min | true / mean |

The per-dim epsilons are the baseline's summed budget divided by the 8 Ant action dims, so both
arms get the same **total** trust region and the comparison isolates the mechanism, not the budget.
Arm B bundles four changes deliberately as a first cut: if it wins, ablate; if it doesn't, the
baseline stands.

A third arm was added once the first telemetry came in, because the baseline's failure had an
obvious single-variable explanation worth isolating:

- **arm C** = `config/mjlab_ant_mpo_caps.yaml`, arm A with **only** `alpha_mean_max` 0.1 -> 10 and
  `alpha_var_max` 10 -> 1000. Nothing else changes.

### Results — mechanism (iteration 200, ~51k env steps, decisive and stable)

| arm | KL_mean / budget | KL_var / budget | alpha_mean | alpha_var | E-step KL / eps | ESS (of 64) |
|---|---|---|---|---|---|---|
| A baseline | **3.43** | **5.27** | **0.100 (PINNED at cap)** | **10.0 (PINNED at cap)** | 1.00 | 53.1 |
| B acme-parity | 1.05 | 2.73 | 1.32 (free) | 19.1 (free) | 1.00 | 53.1 |
| C caps-only | **0.87** | **1.23** | 0.99 (free) | 51.1 (free) | 1.00 | 53.1 |

**The multiplier caps were the whole problem.** In the baseline both M-step multipliers saturate at
their ceilings within 200 iterations and stay there, and the mean/covariance KLs run 3.4x and 5.3x
over budget — the M-step trust region is simply not enforced. Raising only the caps (arm C) frees
the multipliers, and both KLs drop back to ~1x budget: the trust region starts working. This is
deviation 2(b), confirmed by measurement rather than argued from the source.

Two secondary readings:

- The E-step is healthy and identical in all three arms: `KL(q*||pi_old) / eps_dual = 1.00`,
  `dual_residual_eta ~ 1e-3`, SLSQP converging in 1-2 iterations from the warm start, ESS 53/64.
  The exact-dual E-step needs no fixing — §1's conclusion holds in practice.
- Arm B controls KL slightly *worse* than arm C (1.05/2.73 vs 0.87/1.23) despite the same cap
  headroom. Most likely `decoupled_mstep`: it doubles the MLE term's scale against the KL penalty
  (unit-tested), so it needs proportionally more multiplier to hold the same budget. If the reward
  curves end up tied, prefer C — it is one config line, not four.

### Results — FINAL (stopped by user at iteration ~36k of the planned 100k)

Tail-30-block means (last 3,000 iterations), single seed per arm:

| arm | reward | eplen | KL_mu/budget | KL_Sigma/budget | alpha_mu | **pi_std_max** | sat |
|---|---|---|---|---|---|---|---|
| A baseline | **9.99** | 952 | 3.31 | 4.65 | 0.100 pinned | **7.08** | 86% |
| B acme-parity | **13.88** | 950 | 0.79 | 0.62 | 1.00 free | 2.18 | 98% |
| C caps-only | 12.60 | 948 | 0.90 | 0.74 | 1.78 free | 2.77 | 82% |

Reward trajectory (every 2,500 iterations):

| iter | A | B | C |
|---|---|---|---|
| 2,500 | 7.85 | 6.73 | 7.20 |
| 7,500 | 9.15 | 10.88 | 10.65 |
| 12,500 | **11.39** (peak) | 13.91 | 13.13 |
| 20,000 | 10.64 | 13.31 | 12.93 |
| 25,000 | 11.28 | 13.63 | 13.35 |
| 30,000 | 10.52 | 13.75 | 12.05 |
| 35,000 | 9.67 | **13.85** | 11.59 |

**Verdict: B > C > A**, +39% for B over the baseline and +26% for C. The baseline peaks at ~11.4
around iteration 12.5k and then decays for the rest of the run; both corrected arms keep climbing
and hold their level.

**Revision to an earlier read.** Through the first five checkpoints I reported B and C as
indistinguishable, because their ordering flipped at every sample. Over the last six samples that
stops being true: B is 12.88–13.85 (mean ~13.6) while C is 11.59–13.35 (mean ~12.6), so B is both
higher *and* markedly more stable. The caps fix alone (arm C) recovers most of the gap, but not all
of it — the per-dimension KL and/or decoupled M-step appear to add real stability on top. With one
seed per arm this is directional, not conclusive.

**The mechanism that survived all scrutiny is the covariance one.** The baseline's `pi_std_max`
grew monotonically across every checkpoint — 1.29 -> 2.09 -> 2.49 -> 3.04 -> **7.08** — while both
corrected arms stayed at ~2.2–2.8. That is exactly what an unenforced `KL_Sigma` predicts:
`alpha_var` pinned at its cap of 10 with the covariance KL running 3.6–4.9x over budget means
nothing limits sigma inflation. A policy whose sigma blows up to 7 in pre-tanh space is almost
entirely in the tanh-flat region, which is why the baseline decays rather than merely plateauing.

**A hypothesis that did NOT survive**, recorded so it isn't re-proposed: early on the baseline's
`frac_saturated` was climbing fastest and I read that as the cause of its weakness. It inverted —
final saturation is B 98% > A 86% > C 82%, i.e. the *best* arm is the most saturated. Saturation
rises in all three arms and does not explain the ordering. `pi_std_max` does.

**Caveats.** One seed per arm (CLAUDE.md requires >=5 for a reported number); stopped at 36% of the
planned budget; and for the first ~1h two orphaned processes from an aborted first launch were also
on the box, so absolute throughput figures from that window are understated. The *relative*
comparison is unaffected — all three arms shared the same machine load throughout.

### Results — mid-run trace (iteration 20k of 100k)

Rate is ~5.3 iter/s with all three arms sharing the box, so the full budget lands at ~5.2 h.
Numbers are means over the last 5 logged blocks (500 iterations).

| iter | A baseline | B acme-parity | C caps-only |
|---|---|---|---|
| 2,200 | 7.20 | 6.97 | 6.54 |
| 10,300 | 9.84 | **12.05** | 10.76 |
| 16,400 | 9.78 | 11.85 | **12.73** |
| 20,300 | 10.55 | **13.59** | 13.11 |

**Both corrected arms are consistently ahead of the baseline** — the ordering has held for three
consecutive checkpoints, currently ~13.1–13.6 vs 10.55 (a 24–29% gap). Episode lengths have
converged to the 960-step time-out cap for A and B (960 / 959) with C at 935.

**B and C are not separable.** Their ordering flipped at every checkpoint (B, then C, then B). On
this evidence the win is attributable to *fixing the multiplier caps alone* (arm C, two config
lines) — the per-dimension KL, decoupled M-step and target-critic changes in B buy nothing
measurable on top. Do not read a B-vs-C winner out of any single checkpoint; I did that twice and
was wrong both times.

**A saturation hypothesis that did NOT survive.** Early on, the baseline's `frac_saturated` was
climbing fastest (70.6% vs 45.3% / 36.6% at iteration 10k) and I read that as the mechanism behind
its weakness — pre-tanh mean drifting into the flat region, dead gradients, per
[[mpo-estep-mstep-math]] §3.3. It does not hold: at iteration 20k the ordering is B 88.0% >
A 78.0% > C 71.1%, i.e. **the best-performing arm has the most saturation**. Saturation is rising
in all three arms and does not explain the reward ordering. What does still hold is the baseline's
inflated `pi_std_max` (2.49 vs 1.76 / 1.30) — a widening Gaussian under a trust region that isn't
being enforced.

The one measurement that has been stable since iteration 200 is the mechanism itself: baseline
multipliers pinned at their caps with KL 3.8–4.7x over budget, both corrected arms at ~0.8x with
multipliers free and integrating; E-step healthy and identical in all three
(`KL/eps ~ 1.0`, ESS 56–57 of 64, `dual_residual_eta ~ 1e-3`).

### v2 (improved) results — the fixes compound

`config/mjlab_ant_mpo_v2.yaml` = arm B's four parity flags + `target_actor_update: hard`
(period 100). Committed code changes (raised cap defaults, hard-update option, normalizer fix,
diagnostics): `feature/em-algorithms` @ 14858c7.

**mjlab Ant-Flat, 36k iterations, 256 envs (tail-30):**

| version | reward |
|---|---|
| original MPO (arm A) | 9.99 |
| caps-only (C) | 12.60 |
| parity flags (B) | 13.88 |
| **v2 = B + hard target update** | **28.88** |
| PPO baseline (train reward, 1024 envs) | 12.75 |

2.9x the original, 2.1x the best flag-only arm. KL at 0.51x/0.43x budget to the end,
alpha_var settled ~126 (the old cap of 10 would have pinned 12x over). Single seed; box shared
with a concurrent 1024-env REPPO run.

**mjlab Humanoid-Flat (D=21), 36k iterations, 256 envs (tail-30):**
`config/mjlab_humanoid_mpo_v2.yaml` (per-dim eps = 0.01/21, 1e-4/21 — note the mjlab Humanoid
has 21 action dims, not classic MuJoCo's 17).

- **reward 58.7, eplen 862/1000**, still climbing at budget exhaustion (no plateau):
  trajectory -0.04 -> 4.5 (8k) -> 44 (16k) -> 52 (24k) -> 56.5 (30k) -> 58.7 tail.
- Trust region held for the whole run at D=21: KL_mu 0.82x / KL_Sigma 1.22x budget,
  alpha_var integrated freely to ~138, E-step at 0.992 of eps, ESS 57/64, sigma_max 1.76.
- Checkpoints: `logs/safe_rl/humanoid/2026-07-31_13-29-07_humanoid_mpo_v2/`.
- Comparison vs REPPO v24 on the same env (user's concurrent run, 1024 envs) pending that
  run's completion; env-count differs 4x, so compare at equal env steps, not iterations.

**Humanoid target-update ablation (2026-07-31): hard copy vs Polyak is a TIE at the ceiling.**
`config/mjlab_humanoid_mpo_acme.yaml` = the v2 config minus the two target_actor lines
(Polyak, the original behaviour). Same env, same budget, back-to-back on the same GPU:

| | v2 (hard copy) | acme (Polyak) |
|---|---|---|
| tail-30 reward | 58.74 | 58.43 |
| tail-30 eplen | 862 | 830 |
| reward @ 16k | 44.2 | 22.1 |

Hard copy doubles the takeoff speed and smooths the tail (Polyak dipped to 38.9 @ 32k before
recovering); final performance is identical. **Consequence: the Ant v2-vs-B gap (28.88 vs
13.88) is NOT attributable to the target update** — this clean single-variable test rules that
out. The Ant gap was most likely box-load asymmetry (arm B ran three-concurrent alongside two
orphaned runs; v2 ran nearly alone) compounding early-speed differences within a fixed budget.
Do not quote 28.88 without a clean matched re-run; the load-robust claims are (a) the caps fix,
reproduced on four runs / three envs, and (b) improved-MPO reaching ~58.6 on the 21-DoF
Humanoid with the trust region enforced throughout — twice, under two target-update modes.
target_actor_update is a speed/smoothness knob, not a ceiling knob.

### FINAL: the DMPO stack (2026-08-02/03) — goal achieved, then parity with PPO/REPPO

Adding Acme's remaining agent components — C51 distributional twin critics + in-storage
3-step returns (`critic_type: distributional`, `n_step: 3`) — required one real fix: the
categorical projection bootstrapped with scalar gamma even for n-step samples;
`project()` now takes per-sample `gamma**n` (sac.py/critic.py, unit-tested).

**Humanoid-Flat, 3 seeds, 50-ep deterministic evals (9.2M steps each):**

| config | det eval |
|---|---|
| acme-parity (goal target) | 67.0 ± 8.5 |
| **DMPO stack** | **83.9 ± 3.8** (86.5 / 79.5 / 85.5) |

Goal ("similar or better than Acme on Humanoid") achieved: +25%, worst challenger seed >
best target seed, seeds 2+3 never fall (all evals at the 960 cap). Walk videos in the run
dirs under `videos/eval/`.

**Humanoid-Flat at REPPO-matched budget (39.3M steps), deterministic, 1 seed each:**

| algorithm | det eval | eplen |
|---|---|---|
| PPO @300 iters (4096 envs) | 109.7 | 960 |
| **DMPO budget run (153.5k iters, 256 envs)** | **108.4** | 952 |
| REPPO v24 @300 iters | 106.4 | 925 |

A statistical three-way tie at matched budget — and at 9.2M steps DMPO already had 83.9
while PPO sat at 3.7 training reward (hadn't taken off). MPO's story: matches the strong
on-policy baselines at full budget, ~4x more sample-efficient on the way there.

**Other envs (DMPO vs plain best):** Ant-Flat 19.9±3.3 vs 19.7±3.4 (neutral, 3 seeds);
SafetyPointGoal1 ~27 = plain (task ceiling); Unitree-Go2-Flat 38.8 det / eplen 1000-cap
(1 seed; the go2 v24 config header's reference arm reached 32.8/32.7 det on the identical
task). Pattern: the distributional stack pays where the task is hard (Humanoid D=21, Go2),
is neutral where the plain critic suffices.

Configs: `mjlab_humanoid_mpo_dist.yaml`, `mjlab_ant_mpo_dist.yaml`,
`mjlab_go2_mpo_dist.yaml`, `safety_gymnasium_mpo_dist.yaml` (all UNCOMMITTED, as is the
gamma**n projection fix — commit to feature/em-algorithms pending user go-ahead).

Cross-refs: [[cvpo-negative-result]] (why diagnostics matter here), [[m0-baselines]] (FSRL CVPO
reference, whose dual is gradient-ascent — the weakness our exact solve targets),
[[fixA_vs_your_impl]] (FSRL vs our CVPO knobs), [[mpo-estep-mstep-math]] (derivations).
