# CVPO on SafetyPointGoal1: the cost critic is near-constant, and fixing it gets cost to budget

*2026-08-04 (Claude Code). Supersedes the mechanism section of [[cvpo-qc-threshold-calibration]],
which was wrong. Question: why does CVPO exceed its cost budget ~2x while its own constraint
reads satisfied? Answer: the cost critic barely varies with state, so `lambda` never engages.
Fixing that gets realized cost from ~47 to ~26 at unchanged reward.*

## TL;DR

| arm | lambda_max | reward | cost (limit 25) | lambda peak |
|---|---|---|---|---|
| baseline s1 (reference) | 100 | 20.71 | 50.25 | 14.5 |
| baseline s2 | 100 | 24.76 | 46.73 | 5.0 |
| criticfix s1 | 100 | 21.15 | **24.91** | 97.0 |
| criticfix s2 | 100 | 25.90 | 57.66 | 100 |
| both s1 | 100 | 9.51 | 9.15 | 100 |
| both s2 | 100 | 19.12 | 27.77 | 100 |
| lam1 s1 * | 1 | 23.01 | 33.05 | 1.00 |
| lam4 s2 * | 4 | 22.04 | 34.05 | 4.00 |
| **lam4 s1** | 4 | 19.15 | **27.40** | 4.00 |
| **both_lam4 s1** | 4 | 19.16 | **25.55** | 4.00 |

`criticfix` = non-negative cost head + n-step cost targets. `both` = that plus the measured
`qc_scale`. All 30k iterations, 8 envs, `SafetyPointGoal1-v0`, cost limit 25.
\* checkpoints unusable (run-dir collision, below); curves are clean.

**The critic-side fix works.** Cost drops from ~47-50 to ~26-27 with reward essentially
unchanged. **Capping `lambda_max` at 4 is what makes it reproducible** — at 100 the outcome is
bimodal (cost 9.2 to 57.7), at 4 all three arms land in 25.6-34.1.

**What did not get fixed by the scalar critic: the level.** The cost reduction came from `Q_c`
reading high enough for `lambda` to engage at all.

> **METRIC CORRECTION (later the same day).** Several statements below judge the critic by the
> slope of `Q_c` regressed on the realized MC return, against a target of 0.85. **That gate is
> unreachable by any critic.** Regressing a predictor on a noisy realization has ceiling
> `Var(E[G|s])/Var(G)` = the ideal predictor's R². Measured on this env that ceiling is
> **~0.36**, not 1.0 (see "How predictable is the cost, really?"). The correct, unattenuated
> test is the *reverse* regression `MC ~ a*Q_c + b`, calibrated at `a=1, b=0`. Read the
> forward-slope numbers below as descriptive, not as a pass/fail criterion.

## Correcting the previous note

[[cvpo-qc-threshold-calibration]] claimed hazard contacts "concentrate late" so that `gamma^t`
discounts them away. **Measured, that is false.** Cost mass over 10 equal-width time bins,
200 episodes:

```
0.0-0.1 0.073   0.5-0.6 0.101      pooled first / last half: 0.483 / 0.517
0.1-0.2 0.111   0.6-0.7 0.108      (uniform = 0.500)
0.2-0.3 0.096   0.7-0.8 0.096
0.3-0.4 0.099   0.8-0.9 0.077
0.4-0.5 0.104   0.9-1.0 0.136
```

Cost is near-uniform in time. Every bin is within ~0.03 of uniform except a mild excess in the
final bin. The `qc_thres` uniform-cost assumption is approximately valid in shape.

That note also reported the cost critic as "accurate to 2%". That was protocol-dependent — it
averaged over all visited states with truncated returns and compared aggregate-to-aggregate on
a different arm. Under per-episode comparison at initial states it does not hold.

## What the measurement actually shows

Probe: `scratchpad/cost_critic_probe.py`, 200 complete episodes, per-episode records to
`scratchpad/probe_runs/*.jsonl`.

**Initial states** (baseline reference checkpoint, n=200):

```
J_c  (undiscounted)   46.475 +- 34.150     66% of episodes over the limit, 21/200 cost nothing
G_c(s0) (MC)           3.550 +-  4.317
Q_c(s0,a0)             2.123 +-  1.075     aggregate ratio 0.598
```

**Calibration curve over 100k visited states** (>= 500 steps remaining, so MC truncation error
`gamma^500 = 0.007`) — this is the finding:

```
 bin       n      MC range      MC mean  Qc mean    Qc-MC   ratio   neg%
   0   20000 [ 0.000, 0.028]      0.003    1.957    1.954  694.65   3.1%
   3   10000 [ 0.876, 2.046]      1.408    2.208    0.800    1.57   2.7%
   5   10000 [ 3.913, 6.256]      5.040    2.274   -2.767    0.45   2.1%
   8   10000 [13.493,38.518]     17.962    3.841  -14.121    0.21   0.2%

 per-state OLS   Q_c = 0.0943 * MC + 1.9311   (Pearson r = 0.381)
```

**Slope 0.094 against a true cost-to-go range of 0 to 38.** The critic is essentially a
constant ~2.0-2.4. It over-predicts by +1.95 in the lowest bin and under-predicts by -14.1 in
the top. The "1.7x under-read at s0" is just where that flat line crosses the truth. Negative
predictions (2.16% overall) concentrate in low-cost states — 3.1% in bin 0 versus 0.2% in bin 8.

## Root cause, and the four suspects that were not it

Suspects checked before fixing:

- **min-over-twin cost critics** — does not exist. `num_cost_critics` defaults to 1 and no
  config sets it, so `evaluate_cost_q` returns a single critic; the multi-critic branch reduces
  by **mean**, not min (`safe_sac_actor_critic.py`). `min` is on the reward path only.
- **n-step / bootstrap** — clean. `_bootstrap_discount` returns scalar `gamma = 0.99`, identical
  to the `gamma` used for `qc_scale`; `SAC.uses_bootstrap_channel = True` so the runner stores
  truthful dones plus a `time_outs` flag and `_bootstrap_mask` returns 1 at truncation.
  Decisively: **all 200 probe episodes ran exactly 1000 steps** — the env never terminates
  early, so a truncated-return target never occurs.
- **relative under-training** — not structural. `cost_critic_lr` = `critic_lr` = 3e-4, one cost
  update per reward-critic update, same `tau`.
- **buffer staleness** — dead. Episodic cost is 44-89 from iteration 1000 on, so the buffer
  average is *higher* than the final policy's; staleness would bias the critic high, not low.
- **negative outputs** — confirmed real. The head is linear with nothing enforcing `Q_c >= 0`.

**The mechanism.** 1-step TD at `gamma = 0.99` on a sparse binary cost gives a per-sample
learning signal far below the per-sample noise. Measured on fresh on-policy transitions:

```
Bellman residual  c + g*Q_target(s',a') - Q(s,a)   mean +0.0426  std 0.368
                  with ONLINE next-Q               mean +0.0100  std 0.356
mean per-step cost c_bar 0.034  ->  constant-critic fixed point c_bar/(1-gamma) = 3.395
actual mean Q_c(s,a) 2.267
```

Signal 0.010 against noise 0.356 — **36:1 against**. Amplified by `1/(1-gamma) = 100`, that
residual is a level deficit of ~1.0, and it is not closing: `Q_c` on a fixed state set is flat
across every checkpoint from 5k to 30k (3.69, 2.97, 2.33, 2.59, 2.57, 2.76) with slope stuck at
0.06-0.08 throughout. The critic converged to this solution; it is not still climbing.

## The fixes

**n-step cost targets** (`n_step: 10`, buffer-wide). Verified correct two ways before trusting
the arm — `scratchpad/nstep_check.py`:

- *Aggregation*: constant cost 1.0, `gamma=0.99`, n=10 -> expected `sum gamma^k = 9.561792`,
  sampled `9.561791`; `effective_n_steps = [10]`; truncation at a done exact for every
  `n_eff` 1..10; implied fixed point `sum/(1-gamma^n) = 100.0000` = `c/(1-gamma)`.
- *End to end*: constant cost, true `Q_c = 100`. Same gradient steps through the real
  `_update_cost_critic`: after 3000 updates `n_step=1` reaches 13.58, `n_step=10` reaches
  **73.20**. 5.4x faster level propagation.

**Non-negative cost head** (`cost_critic_nonneg`, softplus). Removes the invariant violation:
negatives go from 0.9-2.2% to exactly 0.00% in every arm that uses it.

**Static `qc_scale` recalibration** (`qc_scale_source: measured`). The analytic scale assumes
uniform cost and gives 0.1; the measured `G_c(s0)/J_c = 3.550/46.475 = 0.0764`, so `qc_thres`
becomes 1.91 rather than 2.50 at limit 25. This is a units fix, not the adaptive ratchet from
[[cvpo-qc-threshold-calibration]] — `qc_thres_adapt` stays off.

**`lambda_max` sized from the E-step scale.** Only the *spread across candidate actions* matters
in `(Q_r - lambda*Q_c)/eta`; the level cancels in the softmax. Over 512 states with K=64:

```
Q_r  level 18.011   per-state std across actions  0.0211 (median)
Q_c  level  3.119   per-state std across actions  0.0379 (median)
median std_a(Q_r)/std_a(Q_c) = 0.78
```

So `lambda ~ 0.8` balances the two terms and **`lambda_max = 100` made the cost term ~128x the
reward term** — that is the reward annihilation as a number. Tested `lambda_max` in {1, 4}.

## What each fix contributed

- **n-step alone did the work of engaging the constraint.** `Eqc` reaches 3.61 by iteration
  1000 where the baseline is at 0.38 — 9.5x higher, consistent with the measured 5.4x speedup.
- **That immediately exposed the next problem**: `lambda` saturated at 100 by iteration 2000,
  the reward term was erased, and the policy degenerated (reward -1.17 at 5k). The 5k
  calibration probe of that arm measured a collapsed policy, not a broken critic.
- **Capping `lambda_max` at 4** removes the saturation (settles at 2.3-4.0) and collapses the
  variance: cost 25.6-34.1 across three arms versus 9.2-57.7 at `lambda_max = 100`.
- **`qc_scale` alone does nothing.** Lowering the threshold to 1.91 left `lambda` at 0, because
  `Eqc` fell to 1.47 alongside it — a near-constant critic just drifts with the threshold.

## Calibration after the fixes (200-episode probes, 30k checkpoints)

```
arm             J_c     G_c(s0)  Q_c(s0)  slope   Qc/MC   neg%
baseline s1    46.48    3.550    2.123    0.094   0.505   2.16
baseline s2    38.54    4.072    2.428    0.100   0.506   0.88
qcscale s1     45.15    3.536    2.148    0.059   0.411   9.09
criticfix s1   23.12    0.933    1.982    0.165   1.515   0.00
criticfix s2   34.39    3.761    1.514    0.001   0.005   0.00
both s1        20.68    0.762    1.462    0.087   1.222   0.00
both s2        22.99    2.242    2.045    0.143   0.721   0.00
```

**The Task 3 gate was not met as written** — but see the metric correction in the TL;DR: the
slope leg of that gate was unreachable in principle. Only the zero-negatives criterion passes.
Slope improves at most 0.094 -> 0.165, against a measured ceiling of ~0.36.
`criticfix s2` is the instructive failure: its critic collapsed to a constant (slope 0.0009),
`lambda` fell back to 0, and it reverted to baseline behaviour (cost 57.7).

## DMPO reference point (unconstrained ceiling)

MPO is unconstrained and logs no episodic cost, so its cost was never measured. Rolled out
(`scratchpad/dmpo_eval.py`, 100 complete episodes per checkpoint, 300 total, stochastic
actions, seed 1):

| run | reward | cost | cost p10 / p50 / p90 | over limit |
|---|---|---|---|---|
| `20260802_193548` | 26.92 +- 0.12 | 51.60 +- 4.39 | 0.0 / 45.5 / 113.4 | 69% |
| `20260802_202312` | 27.16 +- 0.11 | 50.38 +- 3.68 | 2.7 / 48.5 / 104.2 | 66% |
| `20260802_211129` | 27.13 +- 0.12 | 52.61 +- 3.75 | 7.0 / 50.0 / 106.1 | 74% |
| **pooled** | **27.07 +- 0.07** | **51.53 +- 2.27** | | 70% |

So the unconstrained optimum earns **27.07** and spends **51.5** — 2.1x the budget. Placing the
constrained arms against it:

| | reward | cost | reward vs ceiling |
|---|---|---|---|
| DMPO (unconstrained) | 27.07 | 51.5 | — |
| baseline CVPO | 20.7 / 24.8 | 50.3 / 46.7 | -2.3 to -6.4 |
| criticfix lam_max=4 | 19.15 | 27.40 | -7.9 |
| both lam_max=4 | 19.16 | 25.55 | -7.9 |

**The baseline CVPO was spending essentially what the unconstrained policy spends** (50.3/46.7
against 51.5) — it was not constrained in any meaningful sense. The fixed arms buy a 2x cost
reduction for ~8 reward points off the unconstrained ceiling.

**Caveat on the DMPO runs**: they requested `n_step: 3` but the whitelist bug (below) dropped
it, so they are 1-step distributional.

### DMPO 1-step vs 3-step (n-step actually applied)

Re-ran with the whitelist fixed, 3 seeds, 8 envs, same config, stopped at 15k:

| iter | 1-step (n=3) | 3-step (n=3) | delta |
|---|---|---|---|
| 4000 | 6.39 +- 1.16 | **13.99 +- 2.42** | +7.60 |
| 6000 | 14.88 +- 1.54 | **19.92 +- 3.07** | +5.04 |
| 8000 | 21.56 +- 0.32 | 21.51 +- 1.69 | -0.05 |
| 10000 | 23.51 +- 0.58 | 23.01 +- 0.28 | -0.50 |
| 15000 | 25.07 +- 0.69 | 24.77 +- 0.63 | -0.30 |

**n-step buys sample efficiency on DMPO, not a better policy.** More than 2x the reward at
iteration 4000, the gap fully spent by 8000, and dead level from there. It is also ~65% slower
per iteration (0.158 vs 0.095 s/iter, CPU-side replay sampling — not GPU-bound, since a run
with the box to itself is no faster), so 3-step reaches a given reward in *more* wall-clock
despite fewer iterations.

Neither variant is converged at 15k: 1-step still gains +1.86 going 15k -> 30k (25.07 -> 26.93).
Quote **26.9** as DMPO's performance on this env, not the 15k figure.

This matters for the CVPO reading: n-step was adopted there because it fixed the *cost critic's*
level, and DMPO has no cost critic. This experiment shows the reward-side effect is only
convergence speed — so the CVPO cost improvement should not be attributed to n-step making
policies generally better.

## How predictable is the cost, really? (and a distributional cost critic)

### The nature of the cost signal

Measured over 200 recorded episodes / 200k steps on `SafetyPointGoal1-v0`:

| property | value |
|---|---|
| per-step cost values | **exactly {0, 1}** (sum == count of nonzero steps in 200/200 episodes) |
| sparsity | 9295 / 200000 steps = **4.65%** |
| contiguous hazard visits | 685 total, **3.4 per episode** |
| visit duration | mean **13.6** steps, median 14, p90 19, max 42 |
| gap between visits | median **104** steps, mean 154 |
| episodic cost | mean 46.5, std 34.1 (CV 0.73), 21/200 episodes cost nothing |

Cost is binary but **bursty**, not isolated: you enter a hazard and pay 1 for ~14 consecutive
steps, ~3 times an episode. The gap between visits (~104 steps) is almost exactly the discount
horizon `1/(1-gamma) = 100`, so `G_c(s)` is dominated by *when the next entry happens*, roughly
one horizon ahead.

### The information ceiling

Fitting a supervised model directly on `(obs, action) -> MC cost-to-go` (an oracle: it sees the
targets, unlike a bootstrapped critic), 10k held-out-split rows per seed:

```
ridge  R^2 = 0.106 / 0.105
MLP    R^2 = 0.355 / 0.372     <- best achievable from these inputs
```

**~64% of the variance in the discounted cost return is not predictable from the observation.**
This is why the forward-slope gate was unreachable: a *perfect* critic scores ~0.36 there.

### Distributional 3-step cost critic, trained passively

`cost_critic_passive`: CVPO with `lambda` pinned at 0, so the E-step weight is `exp(Q_r/eta)` —
literally unconstrained MPO — while `Q_c` trains every update but never touches the actor, the
M-step or the dual. Verified against DMPO: reward curve matches within seed noise at every
iteration (deltas +0.28 to +2.37 against a DMPO seed std of 2.4-3.2), `lambda` identically 0.

| | scalar 1-step | dist. 3-step s1 | dist. 3-step s2 |
|---|---|---|---|
| implied vs realized episodic cost | 21.2 vs 46.5 (**0.46x**) | 42.6 vs 45.5 (**0.94x**) | 57.2 vs 54.7 (**1.05x**) |
| reverse slope (calibrated = 1.0) | 1.54 | 1.70 | 1.63 |
| Pearson r | 0.381 | 0.496 | 0.437 |
| forward slope (ceiling ~0.36) | 0.094 | 0.144 | 0.117 |
| negative predictions | 2.16% | **0.00%** | **0.00%** |

**The level is fixed.** Implied episodic cost is now within 6% of realized, against 2.2x low
before — and the level is exactly what broke CVPO (`Eqc` under-read `qc_thres`, `lambda` stayed
0, budget blown). Negatives are structurally impossible on a `[0, 50]` support.

**Dispersion is not fixed, and is marginally worse.** Reverse slope 1.63-1.70 vs the scalar's
1.54: all three under-disperse, compressing predictions toward their own mean. Crossover at
`Q_c ~ 5.5` — below it the critic over-predicts, above it under-predicts. The critic reaches
r^2 0.19-0.25 against an achievable 0.36, so roughly half to two-thirds of the available signal
is captured; the rest of the gap to "perfect" is noise, not critic failure.

This retires the framing at the top of this note that a *near-constant* critic was the core
defect. The defect was the **level**; the flatness is substantially irreducible.

Caveat: the distributional arm changes three things at once (categorical head, 3-step targets,
one-sided support). Which of the three fixed the level is not isolated here.

## CORRECTION: the information ceiling, and whether history helps

**The earlier ceiling figure (R^2 ~ 0.36) was leaked.** It used a random train/test split over
rows, but adjacent timesteps are near-duplicate states sharing almost all of their discounted
future, so held-out rows sat next to training rows. At stride 1 the same protocol reported
R^2 = 0.946 for a single frame — the signature of leakage, not signal.

Corrected, holding out **whole episodes** (40 episodes, 30k rows, supervised MLP oracle on
`(obs, action) -> MC cost-to-go`):

| history k | features | held-out R^2 |
|---|---|---|
| 1 | 62 | **0.268** |
| 4 | 242 | 0.290 |
| 16 | 962 | **0.309** |
| 32 | 1922 | 0.231 (overfits) |

**Stacking lidar frames does not solve it**: +0.04 R^2 at best, reversing by k=32. Partial
observability is real but small; the residual is aleatoric. Cost is binary {0,1}, 4.65% of
steps, in bursts of ~13.6 steps, ~3.4 per episode, separated by ~104 steps — almost exactly
the discount horizon `1/(1-gamma)=100`. So `G_c` is dominated by *when* the next hazard entry
happens, which is close to unknowable from the current observation.

Against the corrected ceiling, the distributional 3-step critic (r^2 0.246) sits at **~92% of
what is achievable from one observation**. The critic is near the information limit; the
flatness is the task, not the critic. This supersedes the earlier "roughly half to two-thirds,
with real headroom" statement.

## CVaR from the categorical cost critic (WCSAC-style, no Gaussian)

WCSAC (arXiv:2011.11814) responds to exactly this situation — an unpredictable cost return —
by constraining a *risk measure* instead of the mean. Their safety critic is two heads (mean +
variance, softplus, clipped) with a second-moment Bellman backup
`c^2 + 2*g*c*qc' + g^2*(qc_var' + qc'^2) - qc^2`, a Wasserstein-style variance loss
`0.5*mean(v + v' - 2*sqrt(v*v'))`, and CVaR read off a fitted normal via
`pdf_cdf = cl^-1 * phi(Phi^-1(cl))`. Note their `cost_constraint` uses the **same uniform-cost
scaling** shown mis-specified above.

Since our cost critic is already categorical, CVaR comes out exactly — no Gaussian assumption,
no second head. `DistributionalCritic.get_cvar/get_quantile/get_cdf/get_var` match analytic
Gaussian CVaR and VaR to 4 decimals across alpha in {0.5, 0.9, 0.95, 0.99}.

### But the raw distribution is under-dispersed

Evaluated on a pretrained agent (no retraining), 30k states:

```
predicted std 3.21   realized std 5.55      (2.1x too narrow)
PIT KS 0.381 (95% crit 0.008)   deciles 48.0 7.9 5.1 3.9 3.1 2.9 3.3 4.5 6.5 14.9  (U-shaped)
coverage: VaR_0.9 covers 0.858 (want 0.90), VaR_0.99 covers 0.961 (want 0.99)
```

So **raw CVaR understates tail risk by ~25-30%**. The ranking is monotone though (predicted
CVaR_0.9 8.6/10.2/11.4/12.9/16.6 vs realized top-10% means 12.9/14.5/16.8/20.7/29.2), so the
shape is informative and only the calibration is wrong.

### Post-hoc quantile recalibration fixes it, without retraining

Fit a monotone map on CDF values from calibration episodes (Kuleshov et al. 2018), apply to
held-out episodes. Whole episodes are split; a per-row split leaks.

| method | PIT KS | mean PIT | cover 0.5 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| raw | 0.381 | 0.340 | 0.688 | 0.858 | 0.903 | 0.961 |
| affine widen (lam=1.9) | 0.183 | 0.455 | 0.669 | 0.947 | 0.976 | 0.996 |
| **quantile recalibration** | **0.064** | **0.484** | **0.533** | **0.919** | **0.961** | **0.994** |

Reproduced on the independent seed-2 agent: KS 0.284 -> 0.058, coverage at 0.9 0.786 -> 0.889.
Affine widening fails because the true law is a spike at 0 plus a heavy right tail, which a
symmetric mean-preserving widening cannot represent.

### What CVaR changes for the E-step (measured, pretrained agent)

512 states x 64 candidate actions, `qc_thres = 2.50`:

| signal | value | vs qc_thres | per-state action spread |
|---|---|---|---|
| `E[Z_c]` | 4.45 | 1.78x | 0.0170 |
| `CVaR_0.5` | 6.89 | 2.75x | 0.0230 (1.36x) |
| `CVaR_0.9` | 11.19 | 4.48x | 0.0273 (1.60x) |

Two consequences. Constraining CVaR against the mean-derived `qc_thres` is far stricter than
CVPO's constraint. And **CVaR carries 1.4-1.6x more per-state discriminative signal than the
mean** — the E-step has more to choose between, which is the opposite of the mean's problem.

## Code changes (this session)

| file | change |
|---|---|
| `safe_rl/modules/safe_sac_actor_critic.py` | `cost_critic_nonneg` arg + `_cost_head()` softplus, applied in `evaluate_cost_q` and `evaluate_cost_q_target`. Default off. |
| `safe_rl/algorithms/cvpo.py` | `qc_scale_source` {"analytic","measured"}, `qc_scale_measured`, `qc_scale_probe` provenance; `_qc_scale_analytic` kept for comparison; `qc_scale` added to `get_penalty_info`; renamed `_qc_thres_analytic` -> `_qc_thres_initial` (it is no longer necessarily analytic). |
| `scripts/train/train_safety_gymnasium.py` | one line: `n_step` added to the off-policy runner whitelist. It was being **silently dropped** — `n_step` under `algorithm:` is ignored, and the runner block was filtered. **This affected past work**: `safety_gymnasium_mpo_dist.yaml` has asked for `n_step: 3` since it was written, and the 2026-08-02 DMPO runs on this env all trained 1-step. Any Safety-Gymnasium off-policy run predating this fix used `n_step = 1` regardless of its config. |
| `safe_rl/modules/safe_actor_critic.py` | `cost_critic_type` {"standard","distributional"} — C51 cost critic on a one-sided `[0, 50]` support (negative `Q_c` structurally impossible). Single-constraint only. |
| `safe_rl/algorithms/safe_sac.py` | `_update_cost_critic_distributional`: categorical cost backup. No entropy term, and no min-over-twins (understating cost is the *unsafe* direction). |
| `safe_rl/algorithms/cvpo.py` | `cost_critic_passive`: pins `lambda` at 0 so `Q_c` trains as a pure observer and the E-step reduces to unconstrained MPO. `cost_constraint_mode` {"mean","cvar"} + `cvar_alpha`: constrain `CVaR_alpha(Z_c)` instead of `E[Z_c]` (needs a distributional cost critic). |
| `safe_rl/modules/critic.py` | `DistributionalCritic.get_cdf / get_var / get_quantile / get_cvar` — exact CVaR and VaR from the categorical atoms, boundary atom split correctly. |
| `config/safety_gymnasium_dmpo_costprobe.yaml` | DMPO + passive distributional cost critic. |
| `tests/test_cvpo.py` | 8 tests: qc_scale analytic/measured/validation/does-not-enable-ratchet; the negative-output invariant it fixes; softplus clamps online and target; CVPO update still runs. 483 pass. |

New configs: `safety_gymnasium_cvpo_{criticfix,qcscale,both,criticfix_lam1,criticfix_lam4,both_lam4}.yaml`.

Scratchpad tooling (not for commit): `cost_critic_probe.py` (N complete episodes, per-episode
stats, time-bin histogram, calibration curve, JSONL output), `qc_convergence.py`,
`qc_bellman_residual.py`, `nstep_check.py`, `overnight.sh`, `overnight2.sh`.

## Caveats and known defects

- **One seed per configuration.** The repo convention is >= 5. `criticfix` gives 24.91 on s1 and
  57.66 on s2 — do not quote the good number alone.
- **A small reward cost is likely.** `lambda_max=4` arms sit at 19.2-22.0 versus the baseline's
  20.7-24.8. One seed each cannot size it.
- **`n_step` is buffer-wide**, so it changes the reward critic's horizon too. `baseline_s2` was
  run as a control but is not an isolation of the cost-side effect.
- **Run-dir collision**: run directories are timestamped to the second, and `lam1_s1` / `lam4_s2`
  launched in the same second, so both wrote `model_*.pt` into `20260804_081309`. Their
  checkpoints are unattributable. Tensorboard survived (PID is in the filename). Fix would be a
  PID suffix on the run dir, or staggered launches.
- **Broken checkpoint guard** in `scratchpad/overnight2.sh`: it compares checkpoint mtime against
  the run's log, which is written until process exit and so is always newer. Every phase-4 arm
  reported `NO FRESH CHECKPOINT` and skipped its auto-probe. Training was unaffected;
  `lam4_s1` and `both_lam4_s1` checkpoints are on disk and unprobed.

## Where this leaves the method

The binding limitation identified in [[cvpo-negative-result]] — E-step candidates drawn only
from the current policy — is **not** what was stopping cost from reaching budget here. A critic
that reads high enough for `lambda` to engage, plus a `lambda` cap sized to the E-step scale, is
sufficient to hit the budget on this task. What remains unsolved is the critic's inability to
discriminate states (slope 0.17 against 1.0); the constraint is currently enforced by a roughly
state-independent penalty. Whether per-state discrimination would buy anything is the natural
next question, and n-step at larger n, or a distributional cost critic, are the obvious levers.

Cross-refs: [[cvpo-qc-threshold-calibration]] (mechanism corrected here),
[[cvpo-negative-result]], [[m0-baselines]] (FSRL reference: reward 20.6 at cost 27.1),
[[mpo-vs-acme-reference]].
