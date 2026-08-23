# Reading REPPO's diagnostics: what each axis means and how it fails

Date: 2026-08-04. Written against the v33 locomotion run (Go2, 4096 envs, gamma 0.97,
`action_scale 3`, `init_alpha_temp 0.01`) at iteration 70/600, which is healthy on every
axis. Each section says what the number measures, why the observed value is the *right*
one, and what the corresponding failure looks like — most of the failure modes below were
actually hit by this project at some point, and are cited.

Observed at iteration 70:

| q_value | clipped | alpha_temp | returns | reward | ep len | sigma | KL |
|---|---|---|---|---|---|---|---|
| 2.25 | 0.0000 | 0.0021 | 2.25 | +40.7 | 1000 | 0.23 | 0.056 |

---

## 1. `frac_targets_clipped` = 0.0000 — the critic can represent its own targets

**What it is.** The critic is categorical over a fixed support `[v_min, v_max]`. Before
building the HL-Gauss target, `_hlgauss_embed` does `targets.clamp(v_min, v_max)`. This
metric is the fraction of targets that hit that clamp.

**Why zero matters.** A clamped target is a *lie* the critic is trained to believe: the
cross-entropy pushes probability mass into the edge bin, Q saturates at the ceiling, and
`dQ/da` — the pathwise actor's entire learning signal — goes flat. REPPO has no likelihood
-ratio fallback; if `dQ/da` carries no information the actor follows noise.

**The failure, measured.** The `init_alpha_temp: 1.0` attempt of this same config sat at
**0.97 clipped through iteration 133**, `q_value` pinned at 9.84 against `v_max` 10, and
reward *degraded* from -10.5 to -48.8. Earlier: v3-v5 put the whole task inside ~3 bins;
v23 clipped 14% and `q_bias` drifted +0.13 -> +0.64.

**Rule of thumb.** Anything persistently above ~0.01 means the support is wrong for the
current reward scale and discount. Fix the support (or the term inflating the target),
never tune around it.

## 2. `q_value` ~= `returns_mean` (2.2480 vs 2.2474) — the critic is calibrated

**What it is.** `q_value` is `E[Q(s,a)]` over the sampled state-action pairs; `returns_mean`
is the mean lambda-target over those *same* pairs. `q_bias` is their difference stated
directly.

**Why agreement matters.** It says the critic has actually fit its regression target rather
than drifting away from it. Because REPPO bootstraps off its own Q, an error here is
self-confirming.

**The failure, measured.** v17: Q stabilized at 22.7 while realized returns implied ~8 —
a 3x overestimation spiral, where HL-Gauss clamps the overshoot at the edge, CE pushes mass
further up, and the optimism feeds itself. Reward decayed 12.9 -> 4-7.

**Caveat.** Agreement does not prove the critic is *right* — both sides can drift together
if the bootstrap is biased. It proves there is no *regression* error. For the absolute
check, compare Q against `r_step / (1 - gamma)`.

## 3. `q_value` falling (3.69 -> 2.25) while reward rises — this is the entropy term leaving

**What it is.** The critic's target is the *soft* return: `r' = r - gamma * alpha * log pi(a'|s')`.
So Q is not the reward return; it is reward plus an entropy bonus weighted by `alpha`.

**Why falling Q is good here.** As `alpha` decays toward zero the entropy contribution is
withdrawn from the target, so Q must fall toward the pure-reward return
`r_step / (1 - gamma)` ~= 0.05 / 0.03 ~= **1.7**. Observed 2.25 and still descending is
exactly that trajectory.

**How to tell it apart from collapse.** Falling Q *with rising reward* = the entropy term
leaving. Falling Q *with falling reward* = the policy is actually getting worse. Always
read these two together; neither is interpretable alone.

## 4. `alpha_temp` 0.0066 -> 0.0021 — the entropy dual reached equilibrium

**What it is.** The multiplier on the entropy constraint, updated by dual descent on
`alpha * (H - target)`. It falls while entropy exceeds target and rises when it drops below.

**Why this value.** The reference's converged Go2 temperature is **0.0022**. Landing on the
same number from a different starting point says the constraint found its equilibrium
rather than the dual running away.

**The failures, measured.** v11: `alpha_kl` collapsed 0.5 -> 0.0018 within minutes, leaving
the KL gate toothless (a dual *rate* problem — 512 Adam steps per iteration). The opposite
failure is a dual that ratchets up and pins the policy against its constraint forever.

## 5. `KL` = 0.056 against a 0.1 bound — the trust region is not binding

**What it is.** `KL(pi_old || pi_new)`, 16-sample MC estimate, against `desired_kl`.

**Why slack is good.** Under `kl_clip_mode: clipped` the actor loss is a *hard per-sample
gate*: when a sample's KL exceeds the bound, its reward term is **replaced** by
`alpha_kl * KL`. Slack means most samples still receive the reward gradient. KL pinned at
the bound means nearly every sample is gated and the policy is being dragged by the KL
penalty instead of the return.

**The failure, measured.** v10-v12: KL pinned at the bound for entire runs, with sigma
frozen as a direct consequence.

## 6. `sigma` 0.50 -> 0.23 and still annealing — sharpening, not saturating

**What it is.** Mean action noise std of the base (pre-tanh) Normal.

**Why annealing matters, twice over.**
1. *Train/deploy gap.* Deployment uses `action_scale * tanh(mu)` — the mode. The wider the
   training distribution, the further that mode is from what the critic was trained to
   value. Small sigma closes that gap. PPO reaches ~0.1 and tracks well; REPPO stuck at
   ~0.5 was the long-standing symptom.
2. *It proves the entropy target is being met the right way.* A tanh-squashed policy can
   satisfy a low entropy target either by concentrating sigma (good) or by pushing mu into
   tanh saturation, where `d(tanh)/dx -> 0` kills the pathwise gradient (bad).

**The failure, measured.** v13: entropy hit its target exactly while sigma *rose* to 1.47 —
target met purely by saturation, gradients starved. v31: entropy reached -17 with sigma
unchanged at 0.49, same trick. Sigma falling is what distinguishes real sharpening.

## 7. `reward` -37 -> +40.7 with `episode length` = 1000 — improving, not idling

**Why both are needed.** Episode length alone is satisfiable by standing still: the Go2
reward pays for survival and smoothness as well as tracking, so a policy that never falls
and never moves scores long episodes at low reward. High length *and* rising reward means
it is moving and staying up.

**The failure, measured.** `as3_full` at 1200 iterations reached 5/5 survival and a perfect
1000-step mean by trading away **58% of the linear-velocity tracking reward** — total
reward rose because eliminating falls pays more than tracking does. Length + total reward
cannot detect that; only the per-term breakdown (`Episode_Reward/track_linear_velocity`)
and the deterministic tracking error can.

---

## The joint reading

No single axis is sufficient, and several are only interpretable in pairs:

* Q falling **and** reward rising -> entropy leaving the target (good).
  Q falling **and** reward falling -> collapse.
* Entropy on target **and** sigma falling -> genuine sharpening.
  Entropy on target **and** sigma flat or rising -> tanh saturation.
* Episode length high **and** reward rising -> walking.
  Episode length high **and** reward flat -> standing still.
* `frac_targets_clipped` ~ 0 **and** `q_value ~= returns_mean` -> the critic is both
  representable and fitted. Either alone is not enough.

## And the standing caveat

**Training reward is a poor proxy for deployed performance in REPPO.** v23's stochastic
training reward sat at 4-8 while its deterministic policy evaluated at 53.5 — the sigma
~0.6 exploration noise burns the energy/action-rate penalties during training. Every
verdict in this project comes from the deterministic evaluations (scripted head-to-head,
5-seed single-env, seeded 50-episode), never from the training curve. The diagnostics above
tell you the *optimization* is healthy; they do not tell you the *policy* is good.
