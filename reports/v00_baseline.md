# V0 — Freeze and reproduce the Go2 REPPO baseline

**Date:** 2026-08-03 · **Branch:** `reppo_test` · **Freeze commit:** `7a54733`
· **Protocol correction:** `e55a861` · **Environment:** `Unitree-Go2-Flat` (mjlab 1.2.0)

**Hardware:** 1× NVIDIA RTX PRO 4500 Blackwell (32 GB, "GPU 0"), 1× NVIDIA RTX 4000 Ada
(20 GB, "GPU 1"), 16 CPU cores, 31 GB RAM.
**Software:** Python 3.10.20, torch 2.10.0+cu128 (CUDA 12.8), mujoco 3.5.0,
mujoco-warp 3.5.0, numpy 2.2.6, gymnasium 1.3.0. Interpreter
`/home/human/venvs/agx_plain/bin/python`.

---

## Hypothesis

The program's working hypothesis:

> REPPO trains a high-entropy stochastic policy, but deployment uses the deterministic
> mean action. The deterministic action may be poorly optimized because the actor is
> trained mainly on noisy sampled actions. This exploration–deployment mismatch may
> explain REPPO's low forward-velocity gain and large tracking error versus PPO.

V0 tests the **falsifiable core** of that claim that can be measured without changing
any algorithm:

> **H0-a.** REPPO's deterministic action `tanh(mu)` performs *worse* than a sample from
> the stochastic policy `pi(.|s)` it is actually trained on.
> **H0-b.** The PPO-vs-REPPO gap survives equalizing the environment-step budget.

V0 changes no algorithm code.

---

## Code changes

No change to `safe_rl/algorithms/reppo.py`, the entropy target, the actor objective,
the critic, KL settings, the reward function, or any learning rate.

**Evaluator correctness** (`scripts/eval/unitree_mjlab.py`, and
`safe_rl/utils/eval_utils.py` which now holds the testable pieces):

| id | defect | fix |
|---|---|---|
| **R1** | `--q_argmax` built candidates from the **base Normal** and never squashed them. Training samples via `TransformedDistribution(Normal, TanhTransform).rsample()` and `evaluate_q` does not squash internally, so the critic had only ever seen actions in `[-1,1]`. | `tanh` candidates before scoring *and* before returning; the mode candidate becomes `tanh(mu)` |
| **R7** | Episode harvest kept whichever episodes finished first, then truncated — retaining the *earliest finishers*, over-representing falls. | `--one_episode_per_env`: each env contributes exactly its first completed episode |
| **R10** | No stochastic evaluation mode existed. | `--action_mode {deterministic,stochastic}` |
| — | no machine-readable output | `--json_out`, plus survival/fall rate |

**Measured severity of R1.** On the v28 checkpoint, the pre-fix path executed **40% of
actions outside `[-1,1]`, max |a| = 3.79**. Any Q-greedy-vs-mode comparison built on it
would have been meaningless. This is V2's core measurement, so the fix was mandatory
before V2, not cosmetic.

**New files:** `config/mjlab_go2_reppo_v00_baseline.yaml` (byte-identical to
`v28_refparity` outside the header), `config/mjlab_go2_ppo.yaml` (transcribed from the
JUWELS run's resolved `agent.yaml`, which had no YAML in-repo),
`scripts/eval/eval_matrix.py`, `scripts/analysis/{plot_v00,build_arms_manifest}.py`,
`reports/EVAL_PROTOCOL.md`, `experiments/registry.csv`.

---

## Correctness tests

`tests/test_eval_protocol.py` — 9 tests, all passing. Full suite: **574 passed**.

- Q-argmax returns actions within `[-1,1]`, including with σ forced wide (the regime the
  bug appeared in — REPPO holds σ≈0.5 all run).
- `q_argmax(N=0)` reduces exactly to `act_inference`, pinning the mode candidate to
  `tanh(mu)` so any measured improvement is attributable to the sampled candidates.
- Selected action's Q is never below the mode's Q.
- `filter_recordable` takes only the first episode per env; a fast-failing env cannot
  leak a second episode; legacy mode is unchanged.

**The R1 test was verified to fail against the pre-fix implementation** — a
reconstruction of the old code produced max |a| = 3.79 and 40% out-of-bounds actions.

**Metric hand-verification.** The primary metric recomputed from a `--dump_traj` CSV
matches the evaluator's printed value to **1e-6** (0.375801 vs 0.3758). Independently,
mjlab's own `Metrics/twist/error_vel_xy` came out at **2.495×** the per-step value on
1000-step episodes, confirming the documented "cumulative ÷ 400" caveat.

---

## Experiment matrix

| arm | config | budget | seeds | checkpoint |
|---|---|---|---|---|
| REPPO v00 baseline | `mjlab_go2_reppo_v00_baseline.yaml` | 39.3 M (1024×128×300) | 1–5 | `model_299.pt` |
| PPO budget-matched | `mjlab_go2_ppo.yaml` | 39.3 M (4096×24×400) | 1–5 | `model_399.pt` |
| PPO at convergence | JUWELS `agent.yaml` | 196.6 M (4096×24×2000) | 1 | `model_1999.pt` |
| REPPO v28 / v29 / v30 | single-lever parity ablations | 39.3 M | 1 | `model_299.pt` |

Protocol per `reports/EVAL_PROTOCOL.md`: **E1** = 128 envs, one episode per env, eval
seeds {42,43,44} → 384 episodes per arm per mode. **E2** = 1 env, seeds {3,7,11,21,33},
survival only. Checkpoint rule pre-registered as *final checkpoint*. Training seeds
(1–5) are disjoint from eval seeds. Full commands are in the registry and the
scratchpad launcher scripts; representative form:

```bash
python scripts/train/unitree_mjlab.py --env_id Unitree-Go2-Flat --num_envs 1024 \
  --config config/mjlab_go2_reppo_v00_baseline.yaml --seed $S --experiment_name go2_v00 \
  --logger tensorboard --run_name reppo_v00_s$S
python scripts/eval/eval_matrix.py --manifest experiments/v00/arms.json --out experiments/v00 --protocol E1
```

**The reference REPPO arm was NOT evaluated.** It requires `trudi_ref/eval_their_ckpt.py`,
a separate harness that does not implement `--one_episode_per_env`, so its number would
not be comparable under the corrected protocol. Listed as a limitation, not reported.

---

## Results

### Primary — E1, deterministic action, 384 episodes/seed, 5 training seeds

| arm | budget | tracking err ↓ | 95% CI | reward ↑ | yaw err ↓ | survival |
|---|---|---|---|---|---|---|
| PPO @196.6 M | 196.6 M | **0.2348** | – (1 seed) | 54.88 | 0.084 | 0.97 |
| **PPO @39.3 M** | 39.3 M | **0.2714** | [0.2558, 0.2909] | 53.53 | 0.088 | 0.97 |
| **REPPO v00** | 39.3 M | **0.4521** | [0.4278, 0.4811] | 47.73 | 0.160 | 0.91 |
| REPPO v29 (1 seed) | 39.3 M | 0.4215 | – | 49.47 | – | 0.93 |
| REPPO v30 (1 seed) | 39.3 M | 0.4432 | – | 48.62 | – | 0.94 |
| REPPO v28 (1 seed) | 39.3 M | 0.5178 | – | 46.47 | – | 0.95 |

Per-seed tracking — REPPO v00: 0.446 / 0.505 / 0.418 / 0.429 / 0.464 · PPO: 0.250 /
0.303 / 0.260 / 0.284 / 0.260.

**PPO beats REPPO at equal budget by 1.67×**, a gap of 0.181 = **16 SEM**, with
disjoint CIs and **P(PPO better) = 1.00** over all seed pairs.

### H0-a — deterministic vs stochastic

| arm | deterministic | stochastic | Δ (det − stoch) | P(det better) | seedwise |
|---|---|---|---|---|---|
| REPPO v00 | **0.4521** | 0.4996 | **−0.0474** | 0.80 | 4/5 |
| PPO @39.3 M | **0.2714** | 0.2768 | −0.0053 | 0.60 | 3/5 |

Reward: REPPO det 47.73 vs stoch 43.38 (**+4.35, 5/5 seeds**); PPO det 53.53 vs stoch
48.16 (+5.37, 5/5 seeds).

**The deterministic action is better, not worse, on both metrics.** H0-a is refuted.

### E2 — survival (1 env, 5 seeds, noisy by construction)

| group | survival | worst seed | catastrophic (<0.5) |
|---|---|---|---|
| PPO @39.3 M | 0.88 | 0.80 | 0/5 |
| REPPO v00 | 0.68 | 0.40 | 1/5 |

### Command-response decomposition (identical scripted command, `--cmd_script`)

| | forward gain | bias | lag | err @ cmd 0–0.5 | err @ cmd >1.5 | achieved @ >1.5 |
|---|---|---|---|---|---|---|
| REPPO v00 | **0.299** | +0.018 | 6 steps | **0.102** | 1.248 | 0.301 m/s |
| PPO @39.3 M | **0.701** | +0.064 | 10 steps | 0.141 | 0.429 | 1.086 m/s |

**This is the most informative result in V0.** At low commanded speed REPPO is *better*
than PPO (0.102 vs 0.141). The entire deficit is at high commanded speed: asked for
≥1.5 m/s, REPPO delivers 0.30 m/s while PPO delivers 1.09. Lag is comparable (6 vs 10
steps) and bias is negligible for both — so this is **not** a lag or bias problem, and
not a general tracking-quality problem. It is a **forward-gain saturation** problem.

### Training-side diagnostics (mean of last 20 iterations)

| run | reward | ep len | σ | entropy | KL | q_bias | clipped |
|---|---|---|---|---|---|---|---|
| REPPO v00 s1–s5 | 49.92 ± 0.28 | 969–979 | 0.49–0.55 | −5.98…−6.02 | 0.0996–0.0999 | 0.045–0.080 | ~0.13% |
| PPO 39.3 M s1–s5 | 52.59 ± 0.58 | 990–996 | 0.36–0.39 | (+4.8…+5.7)¹ | – | – | – |

¹ PPO's entropy is analytic unsquashed Gaussian; REPPO's is MC `−log π` of the
tanh-squashed density. **Not comparable.** σ is comparable.

Every REPPO seed converges to entropy **−6.0 = the target** and KL **0.0996–0.0999
against the 0.1 bound** — the dual machinery is exact and the trust region is saturated
for the entire run. The critic is healthy (small q_bias, 0.13% of targets clipped).

### Compute

| | wall-clock / seed | ratio |
|---|---|---|
| PPO @39.3 M | 446 s | 1× |
| REPPO v00 | 3400 s | **7.6×** |

At equal environment steps REPPO costs 7.6× the wall-clock (4 epochs × 128 minibatches
= 512 gradient steps per iteration, versus PPO's 20).

---

## Plots

`reports/figs/` — `v00_tracking.png` (primary metric with CIs), **`v00_deploy_gap.png`**
(deterministic vs stochastic per arm — the H0-a figure), `v00_command_response.png`
(commanded vs achieved `v_x` under the scripted command),
`v00_learning.png`, `v00_entropy_std.png`, `v00_duals.png`, `v00_critic.png`.

---

## Interpretation

**H0-b is refuted: the budget gap was never the explanation.** PPO at 39.3 M scores
0.2714 and PPO at 196.6 M scores 0.2348 — PPO is close to converged at REPPO's own
budget, and already beats REPPO by 1.67× there. The standing "REPPO is ~2.3× worse than
PPO" claim was inflated by the 5× budget mismatch, but equalizing it leaves a large,
statistically unambiguous gap (16 SEM, P = 1.00).

**H0-a is refuted, and in the opposite direction.** REPPO's deterministic mode is
*better* than its own stochastic policy — by 0.047 tracking (4/5 seeds) and 4.35 reward
(5/5 seeds). There is no deployment penalty to recover. PPO shows the same sign, so what
we are seeing is the ordinary benefit of dropping exploration noise at deployment, not
anything REPPO-specific. REPPO's effect is larger (−0.047 vs −0.005), consistent with
its larger σ (0.50 vs 0.37) — i.e. REPPO carries *more* noise to drop, and dropping it
helps more. That is the opposite of "the mode is under-optimized".

**What the deficit actually is.** The command-response decomposition localizes it
precisely: forward gain 0.299 vs 0.701, with REPPO *outperforming* PPO at low commanded
speed and collapsing above 1.5 m/s (0.30 m/s achieved vs 1.09). Lag and bias are not
implicated. A policy that tracks well slowly and cannot go fast is saturating, not
mis-deployed.

**The leading explanation is now the action range, not the entropy.** A concurrent
measurement in this repo found PPO's unbounded Gaussian reaches −2.86…+4.46 on Go2 and
spends **61–67% of steps beyond |a| > 1** on the calf joints, while a tanh-squashed
policy is pinned at 1.00. Knee extension produces forward thrust. REPPO squashes; PPO
does not. That mechanism predicts exactly the signature measured here — parity at low
speed, saturation at high speed — and it is confounded with the entropy story in every
comparison in this report.

**What V0 does not test.** H0-a compares the mode against *samples from the same
policy*. It says nothing about whether the mode is worse than the action the **critic**
would pick (`argmax_a Q(s,a)`). That is a different claim and remains open; it is V2's
measurement, now possible because R1 is fixed.

---

## Confounders

1. **Action-range confound (major).** REPPO's tanh caps `|a| ≤ 1`; PPO is unbounded and
   uses that range heavily. Every PPO-vs-REPPO number here is confounded by it. This is
   an *algorithm-family* difference, not a tuning difference, and it may account for the
   whole gap.
2. **Command sequences are not identical across arms in E1.** `--seed` sets only
   `env_cfg.seed`; a policy that falls resets and draws different commands. REPPO's
   survival (0.91) is below PPO's (0.97), so the arms did not face identical tasks. The
   scripted-command run mitigates this and reproduces the same ordering, but only for
   n=1 episode per arm.
3. **GPU physics is not bitwise deterministic.** Same-seed evaluations diverge within a
   few steps. Reproducibility comes from sample size, not seeding — which is why E1 was
   corrected from 50 to 128 envs mid-report (see below).
4. **Single-seed arms.** v28/v29/v30 and PPO@196.6 M have one seed each. Their ordering
   (v29 < v30 < v28) is within the observed 5-seed spread of the v00 arm (0.418–0.505)
   and **should not be read as a lever ranking**.
5. **Reference REPPO not evaluated** under this protocol (separate harness). The
   "ours vs the authors' implementation" question is untouched by V0.
6. **Source drift during the sweep.** A concurrent session modified
   `reppo_actor_critic.py` and `reppo.py` between the starts of seeds 4 and 3. Seeds
   1/2/4 ran the frozen code, seeds 3/5 the modified code. Proven bitwise-equivalent at
   `action_scale=1.0` (`experiments/v00/check_equivalence.py`, `PROVENANCE.md`); seeds
   are pooled on that evidence and would have been discarded otherwise.
7. **Wall-clock is contended.** Other sessions ran jobs on both GPUs; the 7.6× ratio is
   from uncontended PPO runs and mostly-uncontended REPPO runs, so treat it as
   approximate.
8. **PPO@196.6 M was trained on different hardware** (JUWELS, A100) than the local arms.

### The protocol failed its own gate once, and was corrected

E1 was first frozen at 50 envs (150 episodes). The repeat-discrepancy gate then **failed
at 5.4%** (0.5152 vs 0.4873 on the same checkpoint). Cause: episode-level sd ≈ 0.244 on
a mean of ≈ 0.52 — a 47% coefficient of variation. 150 episodes give SEM ≈ 4%, so
5% swings are expected. E1 was raised to 128 envs (384 episodes, SEM ≈ 2.4%); the gate
now **passes at 1.94%**. The 50-env results are archived under
`experiments/v00/e1_50env_superseded/` as the evidence for the change.

Consequence worth stating plainly: the H0-a tracking difference (−0.047) is above the
corrected noise floor but was **below** the original one. It could not have been
resolved at the setting V0 originally froze.

---

## Gate check

| gate | result |
|---|---|
| metrics reproducible | **PASS** — 1.94% on repeated E1 (after correction) |
| all methods use the same evaluator | **PASS** for all local arms; reference REPPO excluded |
| no unexplained discrepancy > 5% | **PASS** — the 5.4% failure was explained and fixed |
| per-step metric hand-verified | **PASS** — 1e-6 |
| every launched run in the registry | **PASS** — `experiments/registry.csv`, no failed runs |

---

## Decision

# KEEP

The V0 baseline, the corrected evaluator, and the (revised) protocol are kept as the
frozen reference for V1–V9. The three evaluator fixes are correctness fixes and stay
regardless of their effect on any number.

**The program's working hypothesis is NOT supported by V0** in its
deterministic-vs-stochastic form, and the evidence points at a different mechanism. This
does not invalidate the program — V2's critic-based test is untouched — but it should
change what gets built next.

---

## Next version recommendation

**Do not proceed to V1 as originally scoped.** V1 (`final_observation` forwarding) is a
genuine correctness bug, but it is a *small* effect on 1/128 of samples per rollout, and
it is present in the reference implementation too. It is not on the critical path to
explaining a 1.67× gap.

**Recommended next isolated experiment — V1′: action-range saturation.**

> Does REPPO's tanh action bound, rather than its entropy or its deployment mode,
> cause the forward-gain deficit?

One lever: `action_scale` ∈ {1.0 (control), 2.0, 3.0, 4.0}, 5 seeds, everything else at
v00 values. The mechanism is already instrumented (`AffineTransform` with the exact
`−n_act·log(scale)` entropy correction), a concurrent session has runs in flight, and
the prediction is sharp and falsifiable: **if saturation is the cause, forward gain and
high-speed achieved velocity should rise while low-speed error stays flat.** If gain
does not move, the hypothesis dies cleanly and attention returns to the entropy/critic
story.

**Then V2 (sampled Q-greedy), unchanged and still worth running.** It answers the
distinct question V0 could not: whether the critic knows better actions than the mode.
R1 is fixed, so it will now produce a valid answer. Note V0 predicts it will *not* find
a large deployment gap.

**Deprioritize V3 (entropy targets).** A concurrent session is already running
`target_entropy: -1.5`, and V0's finding that removing noise *helps* weakens the premise
that the high entropy target is hurting the deployed action.

**Process recommendation.** Three sessions are editing this tree concurrently; V0
absorbed one source-drift incident that required a bitwise-equivalence proof to resolve,
and one duplicated experiment. V1 onward should run in an isolated git worktree.
