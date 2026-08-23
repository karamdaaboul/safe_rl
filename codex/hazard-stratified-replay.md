# Hazard-stratified replay for the CVPO cost critic

*2026-08-04 (Claude Code). Implementation record. Motivated by
[[cvpo-cost-critic-investigation]], whose open problem is the cost critic's inability to
discriminate states (calibration slope 0.165 against a target of 1.0). No training result
yet — this note records the mechanism, the design decisions, and what to measure.*

## The lever

The cost critic sees a cost signal on ~3.4% of stored transitions (mean per-step cost
0.034 on `SafetyPointGoal1-v0`). Its Bellman residual measures mean +0.010 against std
0.356 — **36:1 against** — and it converges to a near-constant `Q_c`.

`hazard_fraction` holds the cost critic's batch at a chosen fraction of cost-bearing
transitions (0.25, a ~7x upsample) and undoes the distribution shift with per-transition
importance weights. **The objective being minimized is unchanged**; only the estimator's
variance moves. That is the whole claim, and it is what makes this different from a
reward-shaping-style hack.

## Mechanism

Two *index views* over the existing `ReplayStorage`. Transitions are stored exactly once.

```
pools[SAFE]   : long[max_size]   flat indices, valid prefix [0, counts[SAFE])
pools[HAZARD] : long[max_size]
slot_pos      : long[max_size]   inverse map, shared (a slot is in exactly one pool)
slot_class    : int8[max_size]   -1 never written, 0 SAFE, 1 HAZARD
```

Classification is `(costs > 0).any(dim=-1)` on the stored value, done in `add()`.

**Pool maintenance is O(flips), not O(batch).** `add()` overwrites a contiguous run of
`num_envs` slots and re-inserts each one, so a slot whose class does not change needs no
pool operation at all. In steady state hazards are rare, so the flip set is usually empty.
Removal is a batched swap-delete in O(k log k): sort the positions to delete, treat those
below `M-k` as holes, refill them from the last `k` entries excluding any that are
themselves deleted. `holes < M-k <= survivors` guarantees the read and write index sets are
disjoint, so the advanced-index assignment is well defined.

**Importance weights.** Under uniform replay an element is drawn with expected multiplicity
`B/N`; under stratified sampling an element of stratum `c` is drawn with `B_c/N_c`:

```
w_c = (B/N) / (B_c/N_c) = (N_c/N) / (B_c/B)
```

Computed from the **actual** batch composition, never the configured target, and
deliberately *not* normalized to `max(w) = 1` — the point is to preserve the absolute
expectation of the uniform objective. `E[w] = 1` exactly, which is the cheapest available
check on the whole pipeline and is asserted under `SAFE_RL_REPLAY_DEBUG=1`.

For binary costs the correction is exact per draw, not merely unbiased:
`(cost_is_weights * costs).mean()` equals the buffer's hazard fraction to float precision.

## The decision that is easy to get wrong

CVPO's E-step, `eta` dual, `lambda` controller and M-step all take **plain means over the
sampled states**:

| site | code |
|---|---|
| `cvpo.py:330` | `eqc = (weights * qc).sum(dim=0).mean()`, then `lambda` integrates `eqc - qc_thres` |
| `cvpo.py:241`/`:263` | dual objective averages `lse` uniformly over states to solve `eta` |
| `cvpo.py:360` | M-step MLE averages over states |

Handing those a hazard-stratified batch biases `eqc` upward and drives `lambda` up
spuriously. [[cvpo-cost-critic-investigation]] found `lambda` calibration to be the fragile
part (bimodal 9.2–57.7 at `lambda_max=100`, only stable when capped at 4), so this would
confound exactly the thing the change is meant to improve.

**So the cost critic draws its own batch.** `SafeSAC._update_extra_critics` samples a second
`storage.sample(batch_size, stratified=True)`; everything else keeps the uniform batch.
Verified: the reward-critic loss is bit-identical with `hazard_fraction` 0 vs 0.25 under a
fixed seed. Cost is one extra `sample()` per update, negligible beside the E-step's
`sample_action_num=64` critic forward passes.

## Files

| file | change |
|---|---|
| `safe_rl/storage/replay_storage.py` | `hazard_fraction` ctor kwarg; pool metadata + `_update_pools`/`_pool_remove`/`_pool_insert`; `_draw_positions_wor`; `_stratified_counts`/`_stratified_weights`/`_sample_stratified`; n-step validity repair; `sample(stratified=)`; `check_pool_invariants`; pools rebuilt in `load_state_dict`. Behavior-preserving split of `_sample_n_step` into `_valid_start_t` + `_gather_n_step` so the stratified path reuses the aggregation verbatim. |
| `safe_rl/algorithms/sac.py` | `hazard_fraction` kwarg -> `ReplayStorage`; `_normalize_obs_tensors` extracted from `update()`; normalizers passed to `_update_extra_critics`. |
| `safe_rl/algorithms/safe_sac.py` | separate stratified cost batch; per-transition cost loss `(w * (Q_c - target)^2).mean()` in both the scalar and the (dormant) distributional branch; `_last_replay_info`. |
| `safe_rl/algorithms/cvpo.py` | `get_penalty_info()` merges `_last_replay_info`. |
| `safe_rl/utils/logger.py` | `replay_*` -> `Replay/*` routing rule. |
| `config/safety_gymnasium_cvpo_hazard.yaml` | `criticfix_lam4` + `hazard_fraction: 0.25`. |
| `tests/test_hazard_replay.py` | 24 tests. |

`hazard_fraction: 0.0` is the default and short-circuits before any RNG use, so every
existing run is unaffected; `tests/test_hazard_replay.py` pins that the disabled path
reproduces the old `torch.randint` draw index-for-index.

## Deviations from the brief

1. **"Classify on the stored n-step cost."** There is no such value. `add()` stores the raw
   1-step cost; `_sample_n_step` aggregates at *sample* time from the window. Classification
   is therefore on the stored 1-step cost. IS-weight unbiasedness does not depend on which
   partition is chosen — any disjoint cover works — so this changes only which transitions
   get upsampled.
2. **"Default 0.25."** The code default is `0.0` and `0.25` lives in the config. A `0.25`
   code default would silently change every existing SafeSAC/CVPO run, which the brief's own
   backward-compatibility requirement forbids.
3. **"Weight only in CVPO."** The cost-critic loss lives in `SafeSAC._update_cost_critic`;
   `cvpo.py` defines no critic loss at all. It is implemented there once, with weights
   defaulting to `None` (arithmetically identical to the old `F.mse_loss`), rather than
   forked into CVPO — the repo explicitly guards against a divergent copy of a critic loss
   (`tests/test_safe_sac_nstep.py:48`). Only CVPO configs enable it and only CVPO logs it.

## Known limitations — report these, do not tune them away

- **Under `n_step > 1` the stratum is "this window *starts* on a hazard."** A window whose
  cost fires 2 steps in is classed SAFE despite a positive aggregated cost, so
  hazard-containing windows are under-selected by roughly a factor of `n`. This matters:
  the recommended config runs `n_step: 10`. The exact fix is a "hazard shadow" (also mark
  the `n-1` preceding slots of the same env), but that makes a slot's class depend on
  *future* writes and breaks the remove-and-reinsert symmetry that keeps `add()` O(flips).
  Deferred; revisit if the `n_step=10` arm is ambiguous.
- **The IS denominator is approximate under `n_step > 1`.** The uniform n-step sampler is
  uniform over the *valid grid* (`|valid_t| * num_envs`), not over `size`, while the weights
  use pool counts. Error is `O((n-1)*num_envs / size)`.
- **RNG streams diverge** once `hazard_fraction > 0`, so a stratified run does not share
  trajectories with its `hazard_fraction=0` baseline even at the same seed. Multi-seed only
  (repo convention >= 5; the one-seed-per-arm in [[cvpo-cost-critic-investigation]] is
  flagged as a defect in its own caveats).
- **Do not set `hazard_fraction: 1.0`** — weights are not normalized to max 1, so an
  all-hazard batch scales every cost gradient by `N_h/N ~ 0.03`.

## Two traps worth remembering

- **`torch.unique` returns sorted values.** Both without-replacement draws truncate a
  deduped candidate set; truncating without shuffling first takes the *k smallest pool
  positions*, and pool position correlates with insertion recency — a silent bias toward
  stale transitions. Fixed in `_draw_positions_wor` and `_draw_valid_flat`.
- **Batch composition must come from the sampler, not from the sampled costs.** Deriving
  "which rows were hazards" from `costs > 0` is wrong under `n_step > 1` for the reason
  above; it initially reported `hazard_fraction_batch = 0.44` for a batch that was drawn
  0.25. `ReplayStorage.last_stratified_info` reports it exactly.

## What to measure

Log keys: `Replay/hazard_fraction_buffer`, `Replay/hazard_fraction_batch`,
`Replay/cost_is_weight_hazard`, `Replay/cost_is_weight_safe`, `Replay/hazard_pool_size`,
`Replay/safe_pool_size`.

Expected at steady state on `SafetyPointGoal1-v0`: buffer fraction ~0.034,
`cost_is_weight_hazard` ~0.034/0.25 = 0.14, `cost_is_weight_safe` ~0.966/0.75 = 1.29.

**Plumbing smoke run** (400 iterations, 8 envs, `n_step: 10`, CPU — plumbing only, far too
short to say anything about the method):

```
it 300   hazard_pool 77    / safe 9555    buffer 0.0080  batch 0.2500  w_h 0.0320  w_s 1.3227
it 399   hazard_pool 661   / safe 12139   buffer 0.0516  batch 0.2500  w_h 0.2066  w_s 1.2645
```

`batch` is pinned at the requested 0.25 throughout; `w_h = buffer/0.25` and
`w_s = (1-buffer)/0.75` hold to 4 decimals, and `0.25*w_h + 0.75*w_s = 1.0000` exactly at
both points. The buffer fraction is well under the trained-policy 0.034 early on and rises
as the policy starts entering hazards, so the realized upsample is 31x at iteration 300 and
5x at 399 — it self-adjusts, which is the intended behaviour. `Learning time` dominates
`Collection time` 168s to 5s per 100 iterations, confirming the extra `sample()` per update
is not measurable.
`Train/eqc` and `SafeRL/lambda_mean` must track the `criticfix_lam4` baseline early on —
they see the unchanged uniform batch, so a divergence there means the isolation leaked.

The actual readout is the calibration slope from `scratchpad/cost_critic_probe.py`
(baseline 0.165, gate 0.85–1.15) alongside realized cost and reward, A/B against
`config/safety_gymnasium_cvpo_criticfix_lam4.yaml` on >= 5 seeds.

Cross-refs: [[cvpo-cost-critic-investigation]] (the measurement this addresses),
[[cvpo-negative-result]], [[safe-sac-inheritance-refactor]] (why the cost path lives in
SafeSAC), [[m0-baselines]].
