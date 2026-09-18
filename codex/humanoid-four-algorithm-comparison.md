# Humanoid-Flat comparison, 2026-09-16

- User requested 256 parallel environments, matching the Ant comparison.
- Algorithms: PPO, SAC, original MPO (5 M-step iterations), MPO with 1 M-step iteration.
- Three seeds each (1, 2, 3): 12 runs, 8,192,000 transitions per run.
- Original MPO and one-M-step MPO have identical settings except M-step iteration count;
  both use 64 E-step actions, 1 critic-target action, and zero mean L2 regularization.
- Ant optimizer/backbone settings retained, including PPO value-loss coefficient 1.0;
  this is a transfer comparison, not a Humanoid-specific hyperparameter search.
- PPO: 1,000 iterations x 32 steps x 256 environments. Off-policy: 32,000 x 1 x 256.
- Evaluate final checkpoints with 50 environments, one episode each, seed 10001,
  deterministic and stochastic modes. Report seed-mean return SD, survival, episode length.
- GPU assignment: PPO/original MPO on GPU 0; SAC/one-M-step MPO on GPU 1.
  Seeds run sequentially per algorithm; algorithms run concurrently. GPU sharing means
  elapsed time is not a controlled algorithm-speed comparison.
- Campaign: `logs/humanoid_four_algorithms_20260916/`; live report `comparison.md`.
- Configs: `config/humanoid_comparison/`; frozen code/configs in campaign `source/`, `configs/`.
- Launcher: `scripts/queue/ant_comparison.py`, generalized with environment, config-dir,
  and algorithm-list arguments while preserving the Ant defaults.
- GPU smoke: campaign `smoke.py` and `smoke.log`; short forced-timeout training,
  finite parameters, checkpoint save, and both evaluation modes for every algorithm.
- Workers refresh the report automatically. Check meaningful phase changes, not frequent
  approval-gated process polls. A failed run must be inspected before restarting it.
- All four GPU smoke tests passed, including both evaluation modes; launch completed.
- Initial workers: PPO 1398931, SAC 1398932, original MPO 1398933, one-M-step MPO 1398934.
