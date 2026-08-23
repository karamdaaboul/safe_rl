# V37 — Sampled Q-greedy on strong checkpoints: the critic adds nothing

Date: 2026-08-04/05. Branch `reppo_test`. 270 rollouts, ~1.4 h.

**Research question.** Does REPPO's critic know better actions than the deterministic
policy mean? Training untouched; selection changed only at evaluation.

**Answer: no — not on a strong checkpoint.** Q-greedy is within noise of the mode on
tracking, strictly worse on return, and the critic's own predicted improvement is
~0.01% of Q. The earlier V2 result (0.467 -> 0.283) was a property of a *weak, saturated*
checkpoint and **is retracted** as a general finding.

---

## 1. Design

Checkpoints: **v34 @200 iterations, seeds 1 / 2 / 3** (`config/mjlab_go2_reppo_v34_lasttrunc.yaml`,
4096 envs x 64 steps x 200 iters = 52.4M). These are the strongest arms produced —
scripted tracking 0.2335 / 0.2404, 5/5 survival each.

| arm | selection rule |
|---|---|
| **N = -1** | one stochastic sample, **critic never consulted** — the control |
| N = 0 | deterministic mode `action_scale * tanh(mu)` — the deployment baseline |
| N = 1, 4, 16, 64 | `argmax_a Q(s,a)` over {mode} u N samples |

**The N = -1 control is what makes this readable.** Without it, N=1 beating the mode is
ambiguous: `{mode, 1 sample} -> argmax Q` looks like a critic win even when the critic is
uninformative and the real cause is "anything but the mode". The control separates them.

**Stage A uses `--cmd_script`** so every arm faces an identical command sequence. This was
not optional: with random commands the *command draw alone* produced a 6x spread inside a
single arm (track 0.156 / 0.556 / 0.971 across seeds), which swamps the effect being
measured. Stage B repeats a reduced set on random commands for external validity.

Stage A: 6 arms x 10 seeds x 3 checkpoints = 180 rollouts. Stage B: 3 x 10 x 3 = 90.

## 2. Stage A — scripted, identical task (180 rollouts)

| arm | tracking | sd | sem | return | falls | a*≠mode | discont. | predicted ΔQ |
|---|---|---|---|---|---|---|---|---|
| sample (no Q) | 0.2481 | 0.038 | 0.0069 | 39.83 | 0/30 | 100% | 0.551 | — |
| **mode** | **0.2384** | 0.019 | 0.0035 | **48.05** | 0/30 | — | 0.205 | — |
| Q-greedy N=1 | 0.2226 | 0.040 | 0.0073 | 47.66 | 0/30 | 2.2% | 0.206 | +0.0001 |
| Q-greedy N=4 | 0.2241 | 0.041 | 0.0074 | 47.45 | 0/30 | 7.1% | 0.219 | +0.0002 |
| Q-greedy N=16 | 0.2231 | 0.042 | 0.0076 | 47.18 | 0/30 | 16.4% | 0.238 | +0.0004 |
| Q-greedy N=64 | 0.2240 | 0.041 | 0.0074 | 46.89 | 0/30 | 27.9% | 0.254 | +0.0008 |

Consistent across all three checkpoints (scripted tracking means):

| ckpt | N=-1 | N=0 | N=1 | N=16 | N=64 |
|---|---|---|---|---|---|
| seed 1 | 0.2466 | 0.2328 | 0.2155 | 0.2170 | 0.2188 |
| seed 2 | 0.2536 | 0.2447 | 0.2336 | 0.2357 | 0.2377 |
| seed 3 | 0.2442 | 0.2377 | 0.2187 | 0.2167 | 0.2155 |

## 3. Stage B — random commands (90 rollouts)

| arm | tracking | sd | return | falls |
|---|---|---|---|---|
| sample (no Q) | 0.4295 | 0.244 | 39.85 | 2/30 |
| **mode** | **0.3529** | 0.086 | 48.83 | 1/30 |
| Q-greedy N=16 | 0.4223 | 0.263 | 49.67 | 0/30 |

Q-greedy is **actively worse** here (0.422 vs 0.353), with 3x the standard deviation.

## 4. Reading

**The critic cannot distinguish the candidates.** Predicted ΔQ is +0.0001 to +0.0008 on
Q ≈ 2.0 — between **0.005% and 0.04%**. At N=1 the argmax picks the mode 97.8% of the time.
This is not miscalibration (predicting a gain and failing to deliver); it is the critic
being *uninformative between nearby good actions*, which is a different and more benign
condition.

**The apparent tracking gain is not resolvable.** 0.2384 -> 0.2226 is ~1.5 sem, and the sem
is itself suspect: Q-greedy at N=1 acts identically to the mode 97.8% of the time yet has
**2x the standard deviation** (0.040 vs 0.019). A policy that differs on 2% of steps cannot
legitimately be twice as variable — that spread is trajectory chaos from a handful of
perturbed actions, not a systematic effect.

**Return degrades monotonically** 48.05 -> 46.89 as N rises, tracking the rise in action
discontinuity 0.205 -> 0.254. Sampling costs smoothness, and the Go2 reward pays for
smoothness through the action-rate term.

**The control kills the alternative hypothesis too.** A plain stochastic sample gives 0.2481
— *worse* than the mode — with return collapsing 48.05 -> 39.83 and discontinuity 2.7x. So
"the mode is bad, anything else is better" is false as well. **The mode is already the right
action.**

**Latency** (ms/step): mode 0.87, N=-1 1.08, N>=1 ~1.9 regardless of N — candidates are
scored in one batch, so N is nearly free. Irrelevant given the result.

## 5. Retraction

[[reppo-implementation-and-ppo-comparison]] and the earlier V2 sweep reported Q-greedy
taking tracking from 0.467 -> 0.283 on `as3_full/model_1199` with no falls, and I described
it as a free win. **That does not generalize.** `as3_full` was the gamma-0.99 1200-iteration
run whose critic was saturated (`frac_targets_clipped` ~0.97 in the failing variant, and a
policy that had traded away 58% of its tracking reward). On a healthy checkpoint — zero
clipping, `q_value` matching `returns_mean` to 4 decimals — there is nothing left for
eval-time selection to recover.

Consistent with the training-time diagnostic: **`deployment_gap`** =
`E_s[E_a[Q(s,a)] - Q(s, tanh(mu))]` measured **negative** on every healthy run (-0.037,
-0.008, -0.066). The critic already rates the mode *above* typical samples, so no gap
existed to exploit.

## 6. Practical upshot

**Deploy the mode.** No eval-time sampling: it does not improve tracking beyond noise, it
costs return, it doubles action discontinuity, and it adds latency. `--q_argmax` remains in
the evaluator as a diagnostic, not a recommended deployment path.

## 7. Reproduction

```bash
SC="150:1.0,0,0;150:0,0,1.0;150:-0.6,0,0;150:0,0.6,0;150:1.5,0,0;150:1.0,0,-1.0;100:0,0,0"
python scripts/eval/q_greedy_probe.py \
  --checkpoint logs/safe_rl/go2_velocity/<v34x200_s1>/model_199.pt \
  --config config/mjlab_go2_reppo_v34_lasttrunc.yaml \
  --n_list=-1,0,1,4,16,64 --seeds 3,7,11,21,33,5,13,29,41,57 \
  --cmd_script "$SC" --out results.json
```

Note `--n_list=-1,...` needs the `=`; argparse reads a leading `-1` as a flag otherwise.
Raw data: `v37A_s{1,2,3}.json` (scripted), `v37B_s{1,2,3}.json` (random) in the session
scratchpad.

## 8. What this closes and what it opens

**Closed:** eval-time action selection as a route to better tracking. Both branches of the
original gate are answered — Q-greedy neither improves deterministic performance nor
exhibits the predict-high/deliver-nothing miscalibration signature.

**Still open** (from [[reppo-go2-results-and-videos]]):
1. v34 wins the scripted comparison (0.237) but loses the random-command average
   (0.337 vs PPO's 0.296) — unexplained.
2. PPO has no error bars; every PPO number is a single seed.
3. `dual_optim_mode: actor` is the last untested parity switch (low expected value).
