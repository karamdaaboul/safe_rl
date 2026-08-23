# Benchmarking our Gaussian REPPO against the published REPPO results

Date: 2026-08-10 / 2026-08-11. Branch `reppo_test`.
Protocol of record: `reports/PAPER_BENCH_PROTOCOL.md` (frozen before any run).
wandb comparison: https://wandb.ai/uqerh-kit/reppo_cartpole_compare

---

# SUMMARY

## What was done

Built a benchmark comparing our torch REPPO (`safe_rl/algorithms/reppo.py`) against the
authors' **own committed per-task result curves** (39 CSVs in `reppo_original/results/`) — not
against PPO, and not against their checkpoint, which is all prior `codex/reppo-*.md` had done.

* New MuJoCo Playground adapter (`Mjx*` prefix), 16 generated per-task configs, a
  (suite x task x seed) sweep driver, a comparison layer, and a frozen protocol.
* 13 benchmark cells completed at 49,938,432 env steps each, plus a **control run of the
  authors' own torch trainer** on the same task/seed/settings.
* Verified our implementation against theirs **numerically**, component by component, then
  per-minibatch. Two harnesses promoted to tests.
* Found and fixed two real defects; retracted six of my own wrong claims (below).

## Results

Our REPPO reaches parity where the critic is stable and fails where it is not.

| suite | task | ours | paper | Δ |
|---|---|---|---|---|
| DMC | `WalkerRun` | 906.7 | 898.0 ± 21.3 | **+1.0%** match |
| DMC | `WalkerWalk` (n=2) | 974.6 | 978.6 ± 1.3 | **−0.4%** |
| DMC | `CheetahRun` | 854.7 | 924.1 ± 51.7 | −7.5% |
| DMC | `FingerTurnHard` | 811.1 | 925.3 ± 40.2 | −12.3% |
| DMC | `AcrobotSwingupSparse` | 4.6 | 11.5 ± 13.0 | within 1 sd |
| DMC | `HumanoidRun` | 122.0 | 693.0 ± 89.4 | −82% |
| DMC | `CartpoleSwingupSparse` (n=2) | 51.7 | 748.7 ± 163.0 | **−93%** |
| DMC | `HopperHop` | 1.3 | 178.7 ± 154.8 | −99% |
| ManiSkill | `PullCube-v1` | **1.000** | 1.000 ± 0.000 | **exact match** |
| ManiSkill | `RollBall-v1` | 0.310 | 0.614 ± 0.037 | −50% |
| ManiSkill | `UnitreeG1PlaceAppleInBowl-v1` | 0.002 | 0.583 ± 0.493 | −99.7% |

All DMC cells above use one config generation (`predict_reward` head present, verified from the
checkpoints). Single seed unless noted; the paper's n is 12-50. Five ManiSkill cells never
completed (OOM-killed — see Operational).

## THE CURRENT PROBLEM

**Our critic takes occasional huge gradient steps that the reference does not.**

Measured per-minibatch at *identical* settings (same task, seed, 1024 envs, 128 minibatches x
4 epochs, 10,752 updates each, iterations 10-30):

| | median | p99 | p99.9 | max | **frac > clip (0.5)** |
|---|---|---|---|---|---|
| ours | 0.1028 | 0.9377 | 1.8701 | 3.483 | **4.80%** |
| theirs | 0.1037 | 0.3283 | 0.5089 | 0.742 | **0.11%** |

The typical update is identical; the tail is 3-5x fatter. **~5% of our critic updates hit the
clip and become direction-only steps (~25 per iteration) versus ~0.5 for the reference — a 43x
difference.** Downstream: Q volatility 40% vs 5.5%, and on sparse tasks where the bootstrap is
the only learning signal the policy is destroyed (`CartpoleSwingupSparse` reaches 717 then falls
to 0; the reference's own torch code reaches **849.0** and holds).

**There is no fix yet.** Five behavioural hypotheses were tested and refuted. Everything
comparable is verified identical. The symptom is localised to a specific measurable object — the
tail of the per-minibatch critic gradient — but its cause is unknown.

**Next step, bounded:** on a spiking minibatch, dump per-sample CE loss and target value and
check whether a few extreme lambda-returns dominate. Per-sample question, finite answer, no
training run needed. Do NOT add seeds to a collapsing config first.

---

# DETAIL

## The comparison target was already on disk

`reppo_original/results/<suite>/<Task>.csv` — 39 tasks, `steps,trial_0..trial_N`, 21 rows.

| suite | tasks | budget | metric | reachable |
|---|---|---|---|---|
| `mujoco_playground` | 27 (23 DMC + G1/T1) | 50M | episode return | yes, via the new adapter |
| `maniskill` | 8 | 50M | success rate | yes, wrapper already existed |
| `isaac` | 4 | 250M | episode return | no (no IsaacLab, 5x cost) |

Only the **last row** is quotable — the curves are resampled onto a fixed 20-point grid and
extrapolate backwards (`CheetahRun` row 0 reads 626.9, impossible untrained). The paper uses
**10 seeds/env** with bootstrapped CIs; committed trial counts are 12-50, apparently pooled
across `small_data`/`large_data` variants.

**`PickCube-v1` is not a paper task** — our only pre-existing ManiSkill config targets a task
with no published number. Smoke vehicle only; excluded from every table.

## The two suites support different claims

| suite | produced by | claim |
|---|---|---|
| ManiSkill | their **torch** trainer — the family this port mirrors | **parity**; a gap is our bug |
| DMC | their **JAX** trainer | **breadth**; a gap is not directly attributable |

Never merge them into one "reproduction" number.

## The three places the authors' own two implementations disagree

Durable and useful to anyone reproducing either. Test-pinned where possible.

| axis | JAX (`src/jaxrl/`) | torch (`src/torchrl/`) | ours matches |
|---|---|---|---|
| lambda-trace truncation gate | `truncated[t+1]` (reverse-scan carry init = ones) | `truncated[t]` | **torch** |
| optimizer | `optax.adam`, wd 0 | `optim.AdamW`, wd **0.01** (torch default; call omits it) | JAX |
| prediction head | `hidden_dim + 1`, has `pred_rew` | `hidden_dim`, **no reward head** | JAX |
| `prior_scale` | 40.0 | 40.9 | JAX |
| normalizer eps | — | clone: `sqrt(var)+eps`; zip: `sqrt(var+eps)` | the **zip** |
| `experiment_overrides` | merged (`jaxrl/reppo.py:932`) | **never merged** | n/a |

That last row is a live trap: passing `experiment_overrides=mjx_dmc_large_data` to the **torch**
trainer is silently ignored, so it runs `reppo.yaml` defaults (128 mb x 4 epochs, not 64 x 8).
It cost me a control run I first reported as "fair".

## Parity: what is verified identical (by measurement, not assertion)

| component | ours vs torch ref |
|---|---|
| HL-Gauss target embedding | 8e-07 |
| lambda-return / GVE (synthetic) | **0.000e+00** |
| lambda-return / GVE (real rollout, real truncations) | **0.000e+00** |
| critic cross-entropy + `denominator: batch` | 2e-06 |
| actor loss (clipped) | **0.000e+00** |
| aux loss (embed + reward, `(1-done)`, mean over D+1) | **0.000e+00** |
| action clamp before `log_prob` | `1-1e-6`, identical |
| env stream (obs/reward/done/truncation) | **0.00e+00 every step** |
| encoder gradient routing (CE-only / aux-only / both) | matches within init noise |
| update loop: epochs, per-epoch reshuffle, critic-then-actor on the same minibatch, clip scope, pi_old freeze | identical |
| critic prior | learnable `nn.Parameter` on both |

Tests: `tests/test_reference_numerical_parity.py` (12), `tests/test_playground_env_parity.py` (2).
Harnesses in the session scratchpad: `mb_capture.py` (per-minibatch), `numerical_parity.py`,
`grad_flow.py`, `rollout_diff.py`, `sync_check.py`.

## Retracted claims (mine, wrong)

1. **`reduce_kl` is a feature our port lacks.** No — it is a multiplier defaulting to 1 inside
   their `clipped` branch, which is exactly our
   `torch.where(kl < desired_kl, primary, alpha_kl*kl)`.
2. **"The control was fair."** It differed on four axes (optimizer/wd, batch shape, reward head,
   `prior_scale`) because the torch trainer ignores `experiment_overrides`.
3. **"Their critic is driven ~9x harder" (7.01 vs 0.77).** Aggregation artifact: they log the
   **last minibatch**, we logged **means over 512**. Fixed — `update()` now also emits
   `*_last` (`value_function_last`, `q_value_last`, `entropy_last`, `kl_last`,
   `critic_grad_norm_last`, `actor_grad_norm_last`). The artifact worked *against* the
   Q-volatility finding, so that gap is real and if anything understated.
4. **`train/rewards_batch` compared to our `Train/episode_reward`.** Per-step SOFT reward vs
   episode RETURN, ~1000x apart; theirs early on is almost entirely the entropy bonus
   (0.0044 at it=1 ~= gamma*alpha*H). Use `train/reward_per_step` for a like-for-like chart.
5. **"Episode synchronisation explains the task split."** Refuted: all 12 runs have episode
   length exactly 1000, so synchronisation is identical, yet Walker is at parity and Cartpole
   collapses. (Our envs *are* synchronised and theirs are staggered — it just is not the cause.)
6. **"No port bug exists; we faithfully reproduce a torch impl that is weak on DMC."** Refuted
   by the control: their torch code reaches 849.0 on the task where we collapse.

Hypotheses tested and **not** the cause: gradient explosion (healthy runs also exceed the clip —
`WalkerRun` at parity has 94% of iterations above it), the aux reward term (collapse in both
configs), episode synchronisation, the four config divergences (Q volatility 48.4%, no better),
the terminal-observation bootstrap (29.4%, inconclusive), normalizer eps mode (<=0.5% at these
variances).

## Defects found and fixed

* **Comparison layer read the wrong attempt.** A retried cell holds several event dirs; the
  loader took the first — the OOM-killed one — so a *finished* `PullCube-v1` was reported as
  incomplete while its successful retry sat next to it on disk. Now takes the
  furthest-progressed attempt. Regression test:
  `test_retried_cell_reads_the_furthest_attempt_not_the_first`.
* **`forward_final_observation` was dead code** on the playground path. `RSLRLBraxWrapper`
  forwards the true terminal obs as `observations["raw"]` — a *dict*, so our tensor-only filter
  discarded it, while REPPO looks for a top-level `infos["final_observation"]`. Now wired
  (verified |terminal − post_reset| = 0.0147; REPPO's own wrong-bootstrap warning no longer
  fires). Did **not** fix the collapse.
* **`load_train_cfg` is a silent whitelist** — an unlisted runner key is dropped with no
  warning. `eval_modes: [ode]` had no effect until whitelisted; covered by
  `test_bench_runner_keys_survive_the_train_cfg_whitelist`. Add a test when adding a runner key.

## Other findings worth keeping

* **Their ManiSkill headline is DETERMINISTIC** despite `stochastic_eval: true` — the stochastic
  call is overwritten at `torchrl/reppo.py:1305-1317`. Confirmed from the data: every final-row
  ManiSkill value is an exact multiple of 1/1024, pinning their eval at 1024 envs x one episode.
* **Evaluating on the training env would have manufactured a win.** Their eval env uses
  `reconfiguration_freq=1`; the training env does not reconfigure at all, so
  `PickSingleYCB-v1`/`PokeCube-v1` would be scored on the one scene they trained against. Fixed
  with a lazily-built eval twin; guarded by `tests/test_bench_eval_parity.py`.
* **Per-task ManiSkill gamma is derived and not constant**: `1 - 10/max_episode_steps` gives
  0.80 / 0.875 (`RollBall`) / 0.90 (`PegInsertionSide`, `UnitreeG1*`). Configs are generated from
  the installed registry because a wrong gamma here once flatlined a full 50M-step run.
* **MuJoCo Playground needed almost no new code** — `wrapper_torch.RSLRLBraxWrapper` already
  returns our contract. Two traps: its `get_observations()` **resets**, and it sets
  `observations["critic"] = None` for symmetric-obs tasks. Both pinned by tests.
* **`CheetahRun` and `CartpoleSwingupSparse` peak then decay** (848 -> 722; 717 -> 0), while
  `HopperHop` never learns at all (flat ~1) and `HumanoidRun` is merely slow (still climbing at
  the budget end). Three distinct failure shapes, not one.

## Operational: this box is shared

* A 1024-env ManiSkill cell is **task-dependent: 7.5 GB (`PullCube`) to 10.4 GB
  (`UnitreeG1TransportBox`)** — size RAM floors for 10.4 GB. DMC cells are ~3.2 GB.
* Five ManiSkill cells were OOM-killed while another session held ~19.9 GB of 32 GB. **No
  leak** — RSS is flat across evals. Driver mitigations: `MemAvailable` floor checked before each
  launch, one retry restricted to `rc=137`.
* Retry is **from scratch**, not `--resume_checkpoint`: resume does not restore the iteration
  counter, and `paper_bench.our_final` rejects any run more than 2% short of 49,938,432 steps.
* **`WANDB_MODE=offline`, never `disabled`**, when running their torch trainer: it sends
  `eval/avg_return` only to wandb, so `disabled` produces a finished run with unreadable numbers.
* Beware `pgrep`/`pkill -f` self-match. It killed two of my own shells and idled a GPU ~18
  minutes via a wait-loop matching its own command line. Use the bracket trick (`[e]nv=...`) or a
  captured PID. Also `nohup ... &` inside a tool call dies with the tool's shell — use
  `setsid ... & disown`.

## How to resume

```bash
# sweep (skips cells that already have model_380.pt)
nohup python -u scripts/bench/run_paper_bench.py --lanes maniskill,dmc --seeds 1,2,3 \
  >> logs/paper_bench_driver.log 2>&1 &

# report at any time; incomplete cells are listed, never scored
python scripts/analysis/compare_to_paper.py --out reports/paper_bench.md

# the authors' torch control (their code, their venv)
cd /home/human/workspaces/reppo_original && WANDB_MODE=offline .venv/bin/python -m src.torchrl.reppo \
  env=mjx_dmc env.name=CartpoleSwingupSparse hyperparameters.total_time_steps=49938432 \
  hyperparameters.num_mini_batches=64 hyperparameters.num_epochs=8 seed=1 num_trials=1
#   ^ pass the batch shape EXPLICITLY; experiment_overrides is ignored by the torch trainer
```

Budget: **49,938,432** env steps per cell (1024 x 128 x 381) — quote that, not "50M".
Old-config cells are preserved (not deleted) under `experiments/paper_bench_oldauxcfg/`.
