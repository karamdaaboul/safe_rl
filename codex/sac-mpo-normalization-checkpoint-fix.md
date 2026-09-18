# SAC/MPO normalization and SAC checkpoint corrections

2026-09-17. User authorized fixes after the reference review.

## Plain-language explanation

Normalization was already present and enabled in the LEAP configs. The change
controls when its running statistics learn: collected observations update them;
replay, target-Q queries, and MPO's repeated candidate states only read them.
SAC checkpoints now remember its learned exploration coefficient alpha and the
optimizer state used to update it.

## Implementation

- Added `ExplicitEmpiricalNormalization`, a read-only-forward subclass preserving
  the existing buffer names/checkpoint layout. Shared `EmpiricalNormalization`
  behavior, including PPO, remains unchanged.
- `SACActorCritic` uses the explicit class and opts into runner-managed statistic
  updates. This is the policy used by both LEAP SAC and MPO; other policy classes
  do not silently opt into the new runner hook.
- Runner updates internal actor/critic statistics on the initial observation and
  once per collected transition successor. For timeouts this is the true final
  actor/critic observation; otherwise it is the returned observation. Ordinary
  termination returns an autoreset observation, which is used for statistics as
  in the reference. Replay and candidate count have no influence on statistics.
- `learn` enters training mode before the initial update; evaluation queries do
  not alter statistics. If external runner normalization is also configured,
  the internal hook receives the same externally normalized units as the model.
- SAC algorithm checkpoint hooks store log-alpha, alpha-optimizer state,
  entropy target/mode, and update counter. Evaluation-only loads restore alpha
  without loading the optimizer. Legacy checkpoints retain configured alpha
  with a warning, since the old missing value cannot be reconstructed.
- MPO and MPO-DIME retain their separate target/dual checkpoint contents; their
  load signatures accept the runner's optimizer flag.
- Replay and environment/RNG state are still not saved, so this does not promise
  bitwise continuation of an entire training trajectory.

## Verification

95 tests passed across the new SAC normalization/checkpoint suite, MPO correctness,
critic-action sampling, MPO-DIME, Playground reset, mjlab terminal observations,
and safe-SAC n-step handling. Two tuning-selection tests also passed.

New tests cover 1/16/64 E-step candidates without statistic mutation, SAC update
purity, evaluation immutability, legacy PPO-style forward behavior, true timeout
successor statistics, fixed/autotuned alpha round trips, optimizer restoration
via an identical next actor/alpha update, evaluation-only loads, and legacy loads.
An earlier critic-sampling test was updated to explicitly ingest the collected
batch before querying target values, reflecting the corrected contract.

## Controlled experiment

Campaign `logs/leap_normalization_fix_20260917/`: fresh SAC, original five-M-step
MPO, and one-M-step MPO, seed1,250 envs,8M transitions each (24M total cap).
Same pilot configs and evaluation seed10001,50 first episodes per action mode.
Only the six corrected runtime source files are overlaid onto the frozen pilot
source. No hyperparameter changes and no automatic longer-budget extension.
This tests the normalization correction; checkpoint changes do not change fresh
training objectives. Existing tuning workers keep their frozen original code.

Controller: `scripts/queue/leap_tuning.py --normalization_control`, frozen copy
inside campaign. Two runs at a time across GPUs0/1; reports and final evaluations
automatic. Read `status.json`, `comparison.json`, and `comparison.md` for status.
GPU smoke checks stored in `smoke/status.json`; launch requires passed smoke.

GPU smoke exposed inference-tensor `_std` replacement during collection, which
prevented reloading a checkpoint into that same policy object. The explicit
normalizer now updates under `inference_mode(False)` and `no_grad` so persistent
buffers stay ordinary tensors. Added regression for collection-mode update,
state-dict reload and subsequent update. Expanded focused suite:96 passed.

## Launch result

All three real LEAP GPU smoke tests passed at 2026-09-17 12:16:24 UTC:
training, checkpoint reload, deterministic and stochastic evaluations. Saved
actor/critic counts are12 for each algorithm:4 initial observations +8 new
transitions, independent of MPO's64 candidate actions.

Launched controlled campaign coordinator PID2244792. Two jobs run concurrently;
the third is queued. Final evaluations and before/after report are automatic.
