# Ant PPO / SAC / MPO comparison — 2026-09-15

## Protocol

- Environment: mjlab `Ant-Flat`, 256 parallel environments, unchanged 16-second episode cap.
- Training seeds: 1, 2, 3 for each algorithm (nine runs).
- Equal budget: **8,192,000 transitions per run**, including SAC/MPO random warmup.
- PPO: 32 steps/env/iteration, 1,000 iterations; registered Ant PPO optimizer settings,
  with [256, 256] ReLU actor/critic networks and observation normalization to match
  the backbone sizes used by SAC/MPO. Standard Gaussian policy, gamma 0.99.
- SAC/MPO: 1 step/env/iteration, 32,000 iterations, four replay updates per iteration,
  batch size 256, replay size 1,000,000, 5,000-transition warmup, raw rewards,
  [256, 256] ReLU actor/critic networks with observation normalization. Tanh-squashed policies.
- SAC uses the standard SAC algorithm with automatic entropy tuning; this is not FastSAC.
- MPO uses the corrected E/M-step implementation and uncapped multipliers.
- All three use actual final observations at time limits. PPO enables the new
  `runner.bootstrap_final_observations: true` option to evaluate V(final_state).
  Its normalizer statistics are held fixed during this extra value evaluation.
- Evaluate the **final checkpoint**, without selecting on evaluation performance:
  50 environments, exactly one first episode from each, seed 10001 shared across all runs.
  Both deterministic and stochastic action modes are evaluated, with deterministic
  return as the primary comparison. Report mean return, episode length, and survival rate.
- Aggregate seed means with sample standard deviation across training seeds. Three seeds
  provide an initial variability check, not a definitive ranking.
- PPO/MPO use GPU 0; SAC uses GPU 1. Algorithms start concurrently, with seeds queued
  serially within each algorithm. Hardware and GPU sharing differ, so wall-clock durations
  are recorded but are not a controlled algorithm speed comparison.

## Artifacts and automation

- Campaign: `logs/ant_ppo_sac_mpo_20260915/`.
- Live report: `comparison.md`, machine-readable results: `comparison.json`.
- Plots: `learning_curves.png`, `final_comparison.png`.
- Per-run progress, training/evaluation logs, and episode-level results:
  `{ppo,sac,mpo}_seed{1,2,3}/`.
- Checkpoints and saved environment/agent configs: `runs/`.
- Working configs: `config/ant_comparison/{ppo,sac,mpo}.yaml`.
- The campaign stores a copy of the safe_rl Python sources and train/eval entry points
  in `source/`. All seeds use this copy. `manifest.json` records source/config hashes,
  dependency versions, GPU assignment, and the Unitree fork revision; git diffs are saved.
- Scheduler/report generator: `scripts/queue/ant_comparison.py`. Detached workers train,
  evaluate, and refresh the report every minute and at stage completion. They continue
  after the assistant turn ends. Failed runs remain explicit in the report.

Initial workers: PPO 250294, MPO 250296; SAC was restarted as worker 279622 after a GPU
index mapping error before initialization. Original SAC startup logs are preserved in
each SAC run's `failed_gpu_selection/`. The corrected launcher selects index 0 within
the single physical device exposed by `CUDA_VISIBLE_DEVICES`.

The previous standalone MPO run is excluded because it predates the timeout correction.

## Validation before full launch

- 73 focused CPU tests passed, including terminal observation handling and PPO final-state
  value bootstrapping without updating normalizer statistics.
- All three campaign configurations parsed and matched the exact transition budget.
- The frozen campaign source passed GPU smoke tests with four Ant environments and
  forced two-step timeouts. PPO, SAC, and MPO each trained, saved/reloaded a checkpoint,
  and completed deterministic plus stochastic evaluations (four episodes each).
- Smoke artifacts: `logs/ant_ppo_sac_mpo_20260915/smoke/` and `smoke.log`.

To refresh the report manually:

```bash
MPLCONFIGDIR=/tmp/safe_rl_mpl python scripts/queue/ant_comparison.py report \
  --campaign logs/ant_ppo_sac_mpo_20260915
```

The full comparison is in progress; smoke scores are never included in its results.
