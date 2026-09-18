# Parallel MPO experiments — 2026-09-16

The user requested starting suggestions 1 and 2 concurrently:

1. `single_mstep_regularized`: one M-step update plus `mean_l2_coef: 0.001`.
2. `critic_actions16`: original MPO configuration with `critic_action_samples: 16`.

Each uses training seeds 1, 2, 3, 256 Ant-Flat environments, and 8,192,000 transitions
per seed. Both retain 64 E-step candidates per state. Experiment 2 retains five
M-step updates and zero mean regularization to isolate its critic-target change.
Experiment 1 uses GPU 0; experiment 2 uses GPU 1. Training seeds are serial within
each experiment, and the two experiment workers run concurrently. The earlier
automatic tuning worker is independent and continues its existing queue.

## Implementation

- `MPO.critic_action_samples` defaults to one, preserving the existing target path.
- With K>1, the scalar critic target uses the mean over K independent target-policy
  actions of `min(Q1_target, Q2_target)` at the next state. The minimum is taken
  **before** averaging over actions. No entropy bonus is added.
- Actor and critic observations are normalized once before expansion, so increasing
  action samples does not multiply observation-normalizer counts.
- SAC exposes a scalar next-value hook; its default retains the original online
  policy sampling, clipped twin value, and entropy term. Timeout masks, n-step
  discounts, reward scaling, and critic losses remain in the common update path.
- K>1 explicitly rejects categorical/quantile critics; this experiment supports
  the standard scalar critics used by Ant. Distributional critics retain K=1.
- Sources are frozen from the completed baseline, with only the new MPO/SAC changes
  overlaid. The exact diff and source hashes are saved with the new experiment.

## Evaluation and files

- Final checkpoints only; 50 first episodes per training seed in each of deterministic
  and stochastic modes. Evaluation seed 10001 matches the existing development runs.
- These are development comparisons. A configuration selected using these scores
  still needs evaluation on an unused seed before claiming validated parity with SAC.
- Launcher: `scripts/queue/ant_mpo_parallel.py`.
- Campaign: `logs/ant_mpo_parallel_20260916/`.
- Shared configs: `configs/`; frozen sources: `source/`; metadata: `manifest.json`;
  code delta: `critic_action_samples.patch`.
- Live report: `report.md`; machine-readable results: `results.json`.
- Each named experiment directory contains `status.json`, `worker.log`, `worker.pid`,
  `trials/` for training/evaluation logs and `runs/` for checkpoints.
- Original tuning may later test a matching adaptive configuration; do not mistake
  a repeated training seed for independent replication when combining reports.

```bash
/home/human/venvs/agx_plain/bin/python scripts/queue/ant_mpo_parallel.py launch
/home/human/venvs/agx_plain/bin/python scripts/queue/ant_mpo_parallel.py report
```

Validation: 78 focused tests passed, including numerical action averaging, clipping
order, target-policy selection, preservation of timeout/n-step targets and SAC entropy,
normalizer counts, and one-action parity. GPU smoke and worker details follow after launch.

Both full configurations passed CPU checks with batch size 256. Both then passed GPU
Ant smoke training with four environments and forced two-step timeouts; their saved
checkpoints reloaded and completed deterministic and stochastic evaluation. Smoke
scores are excluded from the experiment results. Artifacts: `smoke_cpu.log`,
`smoke_gpu.log`, and `smoke/` inside the campaign.

Launched concurrently on 2026-09-16 around 07:01 UTC:

- Combined setting: worker **662587**, initial training process **662589**, GPU 0;
  seed-1 wandb: https://wandb.ai/uqerh-kit/mjlab/runs/8vsx9a8x.
- Sixteen-action critic target: worker **662588**, initial training process **662590**,
  GPU 1; seed-1 wandb: https://wandb.ai/uqerh-kit/mjlab/runs/oy3c5nc1.

Both workers held their exclusive locks and entered training successfully. Seeds 2
and 3 and final-checkpoint evaluations start automatically within each worker. GPU 0
is shared with the original tuning queue, so training time is not a controlled comparison.
