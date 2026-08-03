# Frozen Evaluation Protocol — Go2 mode-alignment program (V0–V9)

**Status:** frozen as of the V0 report. Every version V0…V9 is evaluated with this
protocol and no other. Changing anything below invalidates cross-version comparison
and requires re-running every prior version.

**Scope:** `Unitree-Go2-Flat` (mjlab). Applies to PPO, reference REPPO, local REPPO,
and every new algorithm version.

---

## 1. Primary metric

```
tracking_error_xy = mean_t || v_command_xy(t) − v_actual_xy(t) ||_2
```

- Averaged over the **real episode length**, per episode, then averaged over episodes.
- Implementation: `scripts/eval/unitree_mjlab.py`, accumulated per step into
  `track_err_sum` and divided by `length_buf` at each episode end.
- Command source: mjlab's `command_manager.get_command("twist")` (falls back to
  `"base_velocity"`); achieved velocity: `robot.data.root_link_lin_vel_b[:, :2]`.
  Both are in the **base frame**.

Secondary: `yaw_error = mean_t |w_command_z(t) − w_actual_z(t)|`, same normalization.

### Do NOT use mjlab's `Metrics/twist/error_vel_xy` as a primary number

mjlab's own metric is a **cumulative sum divided by a fixed constant**
(`resampling_time_range[1] / step_dt` = 400 steps for Go2), not a per-step mean — see
`mjlab/tasks/velocity/mdp/velocity_command.py::_update_metrics`. Consequences:

- For a full 1000-step episode it reads **2.5×** the true mean.
- It **scales with episode length**, so a policy that falls at step 290 scores ~3.4×
  "better" than an identical policy that survives 1000 steps.

It is therefore not comparable across policies with different survival times. It is
logged as a secondary diagnostic only, and any number quoted from it must carry this
caveat. Conversion for historical numbers: `per_step = reported × 400 / episode_length`.

---

## 2. Protocols

### E1 — primary (throughput, tracking, return)

| setting | value |
|---|---|
| `--num_envs` | 128 |
| `--episodes` | 128 |
| harvest rule | **exactly one episode per env** |
| eval seeds | `{42, 43, 44}` |
| episodes per arm per action mode | 384 |

**Why 128 and not 50.** V0 first froze this at 50 envs (150 pooled episodes) and the
protocol then failed its own reproducibility gate (§9): re-running one checkpoint gave
0.5152 vs 0.4873, a 5.4% discrepancy. The cause is the metric's episode-level spread,
measured at sd ≈ 0.244 against a mean of ≈ 0.52 — a 47% coefficient of variation,
driven mostly by which velocity commands a given episode happens to draw. 150 episodes
give a SEM of ~4%; 384 give ~2.4%. Reaching <2.5% would need ~350 episodes, which is
what this setting delivers. It is nearly free: with one-episode-per-env the wall-clock
is set by the 1000-step episode cap, and the extra envs are parallel GPU work.

Note that `--seed` does **not** pin the episode set. GPU physics is not bitwise
deterministic, so trajectories diverge within the first few steps and the drawn command
sequences differ between runs of the same seed. The seed controls initialization, not
the realized episodes; reproducibility therefore comes from sample size, not seeding.

The one-episode-per-env rule matters. The pre-V0 evaluator ran
`while len(ep_rewards) < episodes`, harvesting *every* env that finished on a given
step and then truncating with `[:episodes]`. With 64 envs started simultaneously and
50 requested episodes, the retained set was the **earliest finishers** — i.e. falls
were systematically over-represented and long clean episodes were discarded. All
pre-V0 50-episode numbers carry that bias and are superseded.

### E2 — robustness / survival

| setting | value |
|---|---|
| `--num_envs` | 1 |
| `--episodes` | 1 |
| eval seeds | `{3, 7, 11, 21, 33}` |

E2 exists to detect falls on specific hard seeds. **Its tracking error is never
quoted as a headline number** — 5 single episodes is far too noisy (observed spread
0.34–1.28 on a single checkpoint).

---

## 3. Action modes

Every arm is evaluated in every mode it supports:

| mode | definition | supported by |
|---|---|---|
| `deterministic` | `tanh(mu(s))` — `REPPOActorCritic.act_inference` | all |
| `stochastic` | one sample from `π(·\|s)`, tanh-squashed | all |
| `q_argmax N` | `argmax_a Q(s,a)` over `{tanh(mu)} ∪ {N tanh-squashed samples}` | REPPO only (has an explicit Q) |

**Squashing is mandatory in all three.** Training samples actions through
`TransformedDistribution(Normal, TanhTransform).rsample()`, and `evaluate_q` does *not*
squash internally — so any candidate handed to `evaluate_q` or to the environment must
already be `tanh`-ed. Scoring or executing pre-tanh actions puts Q off its training
support and sends `|a| > 1` to the env. (This was a real defect in the pre-V0
evaluator; `tests/test_eval_protocol.py` guards against its return.)

---

## 4. Checkpoint selection

**Pre-registered rule: the final checkpoint.** For REPPO arms that is `model_299.pt`
(300 iterations); for the budget-matched PPO arm, `model_399.pt`.

No checkpoint is ever selected using the evaluation seeds. If a best-validation rule
is introduced later it must use a validation-seed set disjoint from `{42,43,44}` and
`{3,7,11,21,33}`, and both numbers must be reported.

---

## 5. Seeds

| purpose | seeds |
|---|---|
| training | 1, 2, 3, 4, 5 |
| evaluation E1 | 42, 43, 44 |
| evaluation E2 | 3, 7, 11, 21, 33 |

Training and evaluation seed sets are disjoint by construction.

---

## 6. Budget accounting

Environment steps are reported for every arm, always:

| arm | envs × steps × iters | env steps |
|---|---|---|
| REPPO (all versions) | 1024 × 128 × 300 | 39.3 M |
| PPO budget-matched | 4096 × 24 × 400 | 39.3 M |
| PPO at convergence | 4096 × 24 × 2000 | 196.6 M |

**The primary anchor is 39.3 M.** Any table comparing PPO and REPPO must state the
budget of each row. The 196.6 M PPO row is reported as a separate "PPO at convergence"
reference and never used for an equal-budget claim.

---

## 7. Metrics recorded for every run

Reported per the program spec: true mean per-step XY velocity error, yaw tracking
error, episode return, episode length, survival rate, fall rate, forward-command gain,
steady-state forward-velocity bias, command-response lag, action smoothness, policy
entropy, mean and distribution of action standard deviation, KL divergence, temperature
multiplier, KL multiplier, critic Q estimate, Q bias, fraction of categorical targets
clipped, actor gradient norm, critic gradient norm, training environment steps,
wall-clock training time, inference time.

Gain / steady-state bias / lag are computed from `--dump_traj` CSVs by
`/home/human/workspaces/trudi_ref/analyze_traj.py` (least-squares slope, offset, and
cross-correlation peak). Training-time quantities (entropy, σ, KL, duals, Q bias,
target clip fraction, grad norms) come from the TensorBoard event files.

---

## 8. Known limitations — state these in every report

1. **Command sequences are only conditionally identical across policies.** `--seed`
   sets `env_cfg.seed` alone. mjlab resamples the twist command on a timer, so two
   policies with the same seed see the same schedule *until one of them falls and
   resets*. Policies with different fall rates therefore experience different command
   sequences. This is inherent to the mjlab command manager and is not corrected by
   this protocol; it is a confounder for any arm whose survival rate differs from the
   baseline's, and must be reported alongside the fall rate.

2. **`--seed` does not seed torch or numpy.** Stochastic-mode evaluation therefore
   varies run to run beyond the env RNG. The <5% repeat-discrepancy gate is checked in
   deterministic mode.

3. **Truncation bootstrap is known-wrong on mjlab** (`final_observation` is never
   forwarded; mjlab computes observations only after `_reset_idx`). This affects
   *training*, not this evaluation protocol, and is V1's target. The reference
   implementation has the same defect, so V0 comparisons remain like-for-like.

4. **Episode length is capped at 1000 steps**, so "survival" means "reached the cap",
   not indefinite stability.

---

## 9. Reproducibility gate

Before any version's results are accepted:

- Re-running E1 on the same checkpoint twice agrees to **< 5 %** on the primary metric.
- The primary metric recomputed by hand from a `--dump_traj` CSV matches the
  evaluator's printed value.
- Every launched run appears in `experiments/registry.csv`, including failures and
  killed runs.
