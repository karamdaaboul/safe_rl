# LEAP one-step MPO normalization: additional seeds

Launched 2026-09-18 at `logs/leap_mpo_norm_seeds_20260918`.

- Four fresh runs, each with 250 environments and 8M transitions.
- Execution order: corrected normalization seeds 2 and 3, followed by original
  normalization seeds 2 and 3.
- Same one-step MPO configuration as the completed seed-1 comparison.
- Evaluation: deterministic and stochastic actions, 50 first episodes per mode,
  common evaluation seed 10001.
- Clear wandb run names: `mpo1_corrected_seed2`, `mpo1_corrected_seed3`,
  `mpo1_original_seed2`, and `mpo1_original_seed3`.
- Both frozen source versions have hashes in the manifest. Corrected source is
  copied exactly from `leap_normalization_fix_20260917`; original source is copied
  exactly from `leap_zaxis_pilot_20260916`.
- Corrected and original GPU smoke tests passed training, checkpoint reload, and
  both evaluation modes before launch.
- GPU 0 was occupied by an unrelated long training job. The campaign uses GPU 1
  sequentially to avoid interference. Coordinator PID 3037535.
- Persistent state and results: `status.json`, per-run `status.json`,
  `comparison.json`, and `comparison.md`. Corrected seed 2 entered training at
  2026-09-18 10:08 UTC. No stopped older queue was resumed.

Controller: `scripts/queue/leap_mpo_normalization_seeds.py`; frozen executable
copy: campaign `controller.py`.

## Predeclared rollback rule

On 2026-09-18, before seeing seed 3 or either original-normalization result, the
user requested reverting the normalization correction if it shows no improvement.
The primary metric is the matched three-seed mean deterministic return at 8M
transitions (seeds 1, 2, and 3). If the corrected mean is less than or equal to
the original-normalization mean, revert only the normalization behavior. Keep the
separate SAC alpha/checkpoint serialization fix.

Prepared rollback artifacts in the campaign directory: `checkpoint_keep.patch`
and `normalization_test_cleanup.patch`, used together with the reverse of
`logs/leap_normalization_fix_20260917/corrections.patch`. The complete rollback
sequence was applied to an isolated copy and its focused tests passed: 13 passed.
The live workspace is unchanged while experiments run. Before rollback, verify
the involved workspace files have not changed since the rollback was prepared;
if they have, adapt the patch rather than overwriting later work.

## Final result

Campaign completed 2026-09-18 14:25 UTC. Three-seed results at 8M transitions:

- Corrected: deterministic 8.685 (seed SD 6.505), stochastic 8.573 (seed SD 6.382).
- Original normalization: deterministic 1.698 (seed SD 1.225), stochastic 2.628
  (seed SD 1.344).

The corrected version wins the predeclared primary metric, so the rollback
condition was not met and the live normalization implementation was retained.
Results show substantial seed variation; this is evidence for a higher mean in
this exact three-seed LEAP experiment, not a universal performance guarantee.
