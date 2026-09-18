# Corrected MPO on Ant-Flat, 2026-09-15

- Started at 17:24 UTC, seed 1, GPU 0 (RTX PRO 4500 Blackwell).
- Environment: mjlab `Ant-Flat`, 256 parallel environments, 48 observations and 8 actions.
- Budget: 30,000 runner iterations = 7,680,000 environment transitions; 4 replay updates per iteration after warmup.
- Config: `config/mjlab_ant_mpo_corrected.yaml`, based on `mjlab_ant_mpo_best.yaml` with both KL multiplier caps removed. Uses the corrected E-step solver, target-policy critic backup, and MPO checkpoint state.
- Python: `/home/human/venvs/agx_plain/bin/python`.
- PID: `135964`; recorded in `logs/mpo_ant_corrected_20260915/train.pid`.
- Console log: `logs/mpo_ant_corrected_20260915/train.log`.
- Training artifacts: `logs/safe_rl/mjlab_ant_mpo_corrected/2026-09-15_17-24-52_corrected_seed1/`.
- wandb: https://wandb.ai/uqerh-kit/mjlab/runs/cvwhiumc
- Checkpoints every 2,500 iterations and at completion.
- Detached launcher and exact command: `logs/mpo_ant_corrected_20260915/launch.py` and `launch.json`.

Status: completed at approximately 18:49 UTC. All 30,000 iterations (7,680,000 transitions) ran in 1h24m17s. Final checkpoint: `model_29999.pt`. The final logged mean training episode reward was 17.69; this is not a deterministic evaluation result.

This run predates the timeout-observation port described in [mpo-ant-timeout-fix.md](mpo-ant-timeout-fix.md). Its replay targets were affected by missing final observations; later code changes do not repair the completed training run.

## Initial diagnostics and limitations

- E-step KL: 0.1000 for target 0.1; temperature residual about -1.49e-9; solver status 0 (success).
- Critic loss: 1.8591e-4; actor loss: 21.4349; mean training episode reward: 0.18. These are early training measurements, not an evaluation result.
- Maximum per-dimension mean/variance KL budget usage: 1.2693x / 1.5914x. The finite primal/dual M-step remains approximate and does not strictly enforce each budget at every update.
- Existing mjlab integration warning: time-limit truncations do not supply `final_observation`, so the runner bootstraps from the post-reset observation. The installed mjlab environment resets before computing observations, and `MjlabVecEnv` does not capture pre-reset observations. This can bias critic targets and limits interpretation of this run. No environment reset behavior was changed for this experiment.
- First shell-background launch did not survive; verified that its PID was absent before retrying with a detached process session. Only PID 135964 is the active run launched here.

## Reproduce

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/human/venvs/agx_plain/bin/python -u scripts/train/unitree_mjlab.py \
  --env_id Ant-Flat --config config/mjlab_ant_mpo_corrected.yaml \
  --num_envs 256 --seed 1 --gpu_ids 0 --device cuda:0 \
  --max_iterations 30000 --experiment_name mjlab_ant_mpo_corrected \
  --run_name corrected_seed1 --log_dir logs/safe_rl
```
