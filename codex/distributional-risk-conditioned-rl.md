# Risk-conditioned distributional safe RL: build report

**Date:** 2026-08-12 · **Task:** `SafetyPointGoal1-v0` / `SafetyPointGoal2-v0` · **Base algorithm:** CVPO
**Configs:** `safety_gymnasium_dmpo_pid_goal1.yaml`, `safety_gymnasium_dmpo_risk_goal1.yaml`

One policy network that carries a **safety dial**: a scalar input selects how pessimistically the
agent reads its own cost distribution, so a single checkpoint yields several operating points on
the reward/cost frontier without retraining.

---

## 1. What was built, in three layers

### 1.1 Distributional critics (DMPO)

CVPO with both critics categorical (C51) instead of scalar:

| | reward critic | cost critic |
|---|---|---|
| atoms | 101 | 101 |
| support | `[-5, 15]` | `[0, 50]` — one-sided, so `Q_c < 0` is structurally impossible |
| loss | cross-entropy to the projected Bellman target | same |

`Q` is reduced for the E-step by `get_value(dist) = Σ pᵢ·zᵢ`, the **expectation**. Everything
else (E-step sampling, M-step weighted MLE, PID λ) is unchanged from the scalar CVPO.

**Result vs the scalar baseline** (`SafetyPointGoal1`, seed 4, 60k, budget 25):

| | reward | cost | goals | λ |
|---|---|---|---|---|
| CVPO (scalar) | 23.32 | 26.06 ✗ over | 11.44 | 2.23 |
| DMPO (distributional) | 23.34 | **22.98** ✓ | 11.34 | **1.07** |
| DMPO (rerun, same seed) | 23.67 | **21.48** ✓ | 11.63 | **0.57** |

Same reward, but *inside* the budget at half the multiplier. The scalar run never satisfied its
constraint; the distributional one did, twice.

### 1.2 Risk conditioning

A **scalar** risk level in `[0,1]` is appended to the observation by the vec env
(`SafetyGymnasiumVecEnv`, `risk_modes` config key):

```
actor  : 60 obs + 1 risk            = 61
critic : 60 obs + 1 risk + 2 action = 63
```

Because it rides inside the observation, the actor, both critics and the replay buffer carry it
with **no storage-schema change**.

Scalar rather than one-hot so the dial is *ordered* — a trained policy can be queried at levels
never sampled. Drawn **once per episode, not per step**: `Q_c` is the cost-to-go of the policy
actually being executed, and switching mid-episode makes the bootstrap target a mixture over
modes, so it matches no policy.

### 1.3 CVaR distortion

Following DPPO (Schneider, Frey, Miki & Hutter, [arXiv:2309.14246]), each mode reads a
**tail mean**, not a single quantile. `DistributionalCritic.risk_value(dist, level)` takes a
**signed tail fraction**:

| level | statistic | role |
|---|---|---|
| `-0.5` | mean of the best 50% | risk-seeking |
| `1.0` | whole distribution = plain mean | neutral |
| `0.1` | mean of the worst 10% | risk-averse |

The sign encoding removes the `"mean"` special case entirely: **CVaR over the whole
distribution *is* the mean**, so `|level| = 1` needs no branch.

**Verified** against the paper's distortion formula
`V_β = Σₖ (g(τₖ) − g(τₖ₋₁))·θₖ`, `g_β(τ) = min(τ/β, 1)`, over 200 random zero-inflated
distributions per level: agreement to **≤ 1e-5** (float32), and `β = 1` reproduces `get_value`
exactly.

---

## 2. The measurement that shaped every design decision

Before writing the conditioning, we probed the trained cost critic over 3200 visited states.

```
P(Z_c ≤ mean) = 0.787       → the mean is the q0.79, NOT the median
mean(Z_c)     = 1.0213
median(Z_c)   = 0.0283      → 36x smaller than the mean
mass on the zero atom = 0.573   (>50% at zero on 97% of states)
```

The cost distribution is **zero-inflated and heavily right-skewed**: usually nothing, occasionally
a lot. Physically sensible — on most states the agent is nowhere near a hazard.

Consequences, all of which forced design changes:

| quantile | value | vs mean |
|---|---|---|
| q0.10 | **0.0000** | degenerate |
| q0.20 | **0.0000** | degenerate |
| q0.50 | 0.0237 | 0.02× |
| q0.60 | 0.4250 | 0.4× |
| **mean** | **1.0315** | **1× (= q0.79)** |
| q0.80 | 1.7088 | 1.7× |
| q0.90 | 3.3725 | 3.3× |

**Stability across training** (10k → 40k): absolute values drift ±25%, but the *ratios* are
near-constant — `q0.6/mean` = 0.42 ± 0.04, `q0.9/mean` = 3.24 ± 0.03, rank of the mean
0.771 → 0.789. This is why levels are defined as **quantiles/tail fractions, not absolute
values**: they self-normalise as the policy changes.

---

## 3. Challenges

### 3.1 The mean is not neutral

Choosing "the mean" looks like the risk-neutral default. On this distribution it is the **q0.79** —
already fairly pessimistic. Every CVPO run in this project has therefore been operating at a ~79th
percentile risk level without anyone choosing that. It also means the dial is asymmetric: there is
a lot of room to make the agent riskier and comparatively little to make it safer.

### 3.2 Low quantiles silently disable the constraint

`q0.2` was **identically 0.0000 on all 3200 states**. A mode using it would not be "risky" — it
would be **completely unconstrained**, since `Q_c = 0` removes the cost term from
`exp((Q_r − λQ_c)/η)` exactly as if λ were 0. The originally proposed `{q0.2, mean, q0.6}` was
discarded for this reason.

Mitigation: `risk_floor_frac = 0.3` floors any level at 0.3 × mean. **CVaR does not remove this
need** — with 57% of mass at zero, the mean of *any* lower-tail fraction below ~0.57 is exactly
zero too.

### 3.3 A quantile is one atom; the tail is many

`q0.6 − q0.2` had a *median* of exactly one atom (0.5), and the two landed on the **same atom for
20.5% of states**. On a discrete support a VaR reading is coarse and jumps. CVaR averages every
atom beyond the cut. This was the main reason for the switch.

### 3.4 Direction inverts between reward and cost

For a *reward*, risk-aversion takes the **lower** tail. For a *cost*, it takes the **upper** tail.
The initial spec (`0.6 = risky`, `0.2 = safe`) had it backwards, and applying "the same idea" to
cost naively would have broken it: the **median cost is 0**, so a median-based cost gate is
satisfied by an agent that idles half the time — the same trap in mirror image.

### 3.5 Truncation bootstrap vs the extra observation column

`final_observation` (used to repair the Q-bootstrap on truncation) is produced by the env at raw
width, while `_last_obs` carries the risk column. Sizing the buffer from `_last_obs` crashes; and
the terminal observation needs the risk level of the episode that **ended**, not the one drawn for
the next. Both handled explicitly; `_build_extras` runs before the resample so `_risk_idx` is
still the old value there.

### 3.6 Silent config plumbing

`train_safety_gymnasium.py` forwards env options **one by one** rather than splatting the `env:`
block. The first risk run built a 60-input actor with `risk_modes: 3` apparently set — the key was
dropped without warning. Same class of bug as the known `reward_normalization` whitelist issue.

### 3.7 Logging that cannot distinguish success from failure

- `Episode/risk_mode` (mean of the mode index) is **useless** — a uniform draw pinned at its own
  mean forever. It was also off by one episode.
- `Episode/cost` **averages the three agents together**. If the dial works perfectly (40/25/10) the
  aggregate reads 25; if it fails completely (25/25/25) it also reads 25. The aggregate is blind.

Fix: per-mode `cost_{seeking,neutral,averse}` and `reward_{…}`, split by the mode each finished
episode actually ran under.

### 3.8 Evaluation needs the level pinned

Without `--risk_level`, evaluation samples a mode per episode and averages a random mixture. The
eval path also never passed `risk_modes`, so the checkpoint would not even load. Added
`risk_fixed_level` on the env and `--risk_level` on the evaluator, which now **refuses to run**
on a risk-conditioned config without one.

### 3.9 "Same seed" is not the same environment

The eval loop seeds only at the initial reset and relies on auto-reset thereafter. In Goal tasks
**every goal reached respawns a goal, consuming env RNG** — so modes that score differently drift
onto different layouts from episode 2 onward. Only **episode 1** is genuinely comparable. Fair
multi-episode comparison requires one eval process per seed.

### 3.10 The distributional critic is nondeterministic — and it is fixable

The same config and the **same seed** run twice diverged badly mid-training (reward 13.83 vs 7.49
at 12k) before converging to 23.34 and 23.67. Three DMPO runs on the **same GPU** shared **0 of 8**
log blocks and were 9.8 reward apart by 7k.

**Cause, isolated by a control experiment:** scalar CVPO with the same seed and GPU was
**13/13 log blocks bit-identical**, so nothing in the shared pipeline is at fault. The categorical
projection accumulates mass with `index_add_`, which on CUDA uses `atomicAdd` — several source
elements target the same atom, arrive in hardware-scheduling order, and float addition is not
associative. Measured in isolation: two identical `index_add_` calls differed by **7.2e-05**
without deterministic algorithms and by **exactly 0** with them. The off-policy loop
(policy → visited states → replay → critic → policy) amplifies that into whole-policy divergence.

**Fix:** `--deterministic` (already wired to `seed_everything`; it was simply never passed).
Verified: **5 runs bit-identical over 8000 iterations**, no fallback warnings. Costs ~20%
wall-clock (113 → 136 ms/iter).

Consequences:
- A run that looks dead may be recovering. A seed-3 run stopped at 13k (reward 3.19) was plausibly
  on the same slow trajectory as the rerun, which was at 8.71 at 13k and finished at 23.67.
- **Single-run claims about sample efficiency do not survive** for runs made without the flag.
  "DMPO matches CVPO by 35k" held for one run; the rerun needed 55k.
- What *is* robust is the endpoint: ~23.5 reward inside budget, three times.
- **Every result in this report predates the flag**, so mid-training numbers are one draw from a
  ~10-reward band. Re-run with `--deterministic` before quoting any of them.
- Determinism pins *which* trajectory you get, not a better one — multiple **seeds** are still
  required for a performance claim.

### 3.12 A/B arms must sit on the same GPU

Two "identical" scalar runs diverged completely — because I put them on different devices. This
box has two different GPU models (Blackwell / Ada), which select different kernels. Same seed +
same GPU reproduces exactly; same seed + different GPU does not. Any comparison whose arms sat on
different devices is confounded.

### 3.11 Shared λ regulates the mixture, not each mode

One multiplier across all modes is deliberate — it is what produces the spread (per-mode
controllers would force every mode to the same realized cost, erasing the dial). The cost is that
"the constraint is satisfied" becomes a statement about the **average across modes**: one mode sits
above budget and one below.

---

## 4. Results

### 4.1 Training, end of run (60k, per-mode averages)

| mode | reward | cost |
|---|---|---|
| seeking | 24.72 | 27.40 |
| neutral | 22.68 | 24.92 |
| averse | **19.98** | **9.56** |

The averse mode runs **2.9× cheaper** than seeking while keeping **81%** of its reward. Neutral
lands at 24.92 against a budget of 25 — the shared λ regulating the mixture onto target.

### 4.2 Held-out evaluation, `SafetyPointGoal1`, one episode per seed

Seeds **0,1,2,3,4**, identical layouts across modes, deterministic action:

| mode | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| seeking | 26.33 / 0 | 25.90 / 2 | 25.76 / 0 | 26.52 / **34** | 25.45 / 0 | 25.99 / 7.2 |
| neutral | 26.47 / 0 | 27.64 / 0 | 26.47 / 0 | 24.05 / **53** | 27.52 / 0 | 26.43 / 10.6 |
| averse | 6.87 / 0 | 23.34 / 0 | 21.02 / 0 | 9.26 / **16** | 24.21 / 0 | 16.94 / 3.2 |

**Only seed 3 carries any cost signal** — 4 of 5 layouts cost zero for every mode. On that one
informative episode the dial works: averse cuts cost **70%** vs neutral (53 → 16).

The averse mode is **bimodal, not uniformly timid**: 23.34 / 21.02 / 24.21 on three layouts,
collapsing to 6.87 / 9.26 on two. The mean of 16.94 describes no actual episode.

### 4.3 Zero-shot transfer to `SafetyPointGoal2` (never trained on)

Single episode, identical layout, seed 0:

| mode | reward | cost | reward per unit cost |
|---|---|---|---|
| seeking | 23.39 | 259 | 0.090 |
| neutral | 21.94 | 193 | **0.114** |
| averse | 9.19 | 185 | 0.050 |

Ordering survives transfer. But seeking → neutral is a good trade (**−25% cost for −6% reward**)
while neutral → averse is a bad one (**−4% cost for −58% reward**).

---

## 5. Open problems

1. **Is `risk_levels: 0.1` too aggressive?** Level-2 transfer and 2 of 5 level-1 seeds say yes;
   3 of 5 level-1 seeds say no. Worth testing `0.25` (worst quarter, ≈1.7× mean).
2. **Why is the averse mode bimodal?** It performs normally on most layouts and shuts down on
   some. Not explained.
3. **Deterministic vs stochastic action.** Training reports averse at 19.98; deterministic eval
   gives 16.94 (and 6.87 on one layout). The averse mode may rely on action noise.
4. **PointGoal1 is nearly blind for this question** — 4 of 5 layouts cost zero. Use PointGoal2,
   where every episode costs 185–259.
5. **Eval seed sharing.** The evaluator warns "only N distinct cost values among n episodes" on
   every run; effective sample size is below `n`. Unfixed, and it weakens every eval number here.
6. **Seed-to-seed variance not quantified.** Run-to-run variance is now solved (`--deterministic`),
   but seed-to-seed spread is still unmeasured, so no performance claim here has an error bar.
7. **Not implemented from the paper:** quantile-regression critic (we use fixed-support C51, which
   can saturate), the Wang distortion, and continuous β sampling (we sample 3 discrete modes).

---

## 6. Reproduce

```bash
# risk-conditioned training -- --deterministic is REQUIRED for reproducibility (sec. 3.10)
python scripts/train/train_safety_gymnasium.py --env_id SafetyPointGoal1-v0 --num_envs 8 \
  --config config/safety_gymnasium_dmpo_risk_goal1.yaml --max_iterations 60000 \
  --seed 4 --cost_limits 25.0 --deterministic

# evaluate one mode (level: 0 = seeking, 0.5 = neutral, 1 = averse)
python scripts/eval/eval_safety_gymnasium.py --env_id SafetyPointGoal1-v0 --num_envs 1 \
  --config config/safety_gymnasium_dmpo_risk_goal1.yaml --checkpoint <model.pt> \
  --episodes 1 --seed 0 --risk_level 1.0 --cost_limits 25.0
```

> **Naming collision to be aware of:** `--risk_level` is the *policy input* (dial position:
> 0 = seeking, 1 = averse), while `risk_levels` in the config are *CVaR tail fractions*
> (`1.0` = neutral). The value `1.0` means opposite ends in the two places.

**Tests:** `tests/test_risk_conditioned_cost.py` (7) — statistic selection per mode, monotone
ordering, the floor preventing a degenerate lower tail, tail-mean-vs-quantile, interpolated levels
snapping to the nearest trained mode, per-mode metrics keyed by the finished episode's mode.

**See also:** [`pointgoal2-curriculum.md`](pointgoal2-curriculum.md) for the level-2 cost-limit
curriculum and why its floor of 20 is infeasible.
