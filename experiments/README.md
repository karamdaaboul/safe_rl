# Experiments

Training entry points for JUWELS Booster. All jobs write SLURM logs to `logs/` relative to
the directory where `sbatch` is invoked.

## JUWELS Quick Commands

Useful SLURM commands on a JUWELS login node:

```bash
# Show my running + pending jobs
squeue -u $USER

# Find one specific job
squeue -j <JOB_ID>

# Show detailed info for one job
scontrol show job <JOB_ID>

# Follow the stdout log of a running job
tail -f logs/<job-name>-<JOB_ID>.out

# Follow the stderr log
tail -f logs/<job-name>-<JOB_ID>.err

# Cancel a job
scancel <JOB_ID>

# List my recent jobs (running, finished, failed, cancelled)
sacct -u $USER --format=JobID,JobName,Partition,State,Elapsed,Start,End

# List jobs from today only
sacct -u $USER --starttime today --format=JobID,JobName,State,Elapsed,Start

# Inspect the history of one job
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode

sacct -u $USER --starttime 2026-04-21 --format=JobID,JobName,State,Elapsed,Start,End,ExitCode
```

Tips:
- `squeue` shows the current queue state only.
- `sacct` is the command to use for old jobs / finished jobs.
- The log files are usually `logs/<job-name>-<jobid>.out` and `logs/<job-name>-<jobid>.err` because the sbatch scripts in this repo use `#SBATCH --output=logs/%x-%j.out` and `#SBATCH --error=logs/%x-%j.err`.

---

## Unitree MJLab (mjlab PPO / FastSAC / FastTD3)

**Setup** — sync the repo and build the container once:
```bash
bash experiments/juwels_unitree_rl_mjlab/sync_to_juwels.sh
bash experiments/juwels_unitree_rl_mjlab/build_unitree_rl_mjlab_image.sh
```

### PPO (mjlab default)
```bash
sbatch experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
```
Override task and number of envs:
```bash
sbatch --export=ALL,TASK_ID=Unitree-Go2-Flat,NUM_ENVS=2048 experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
```

### FastSAC
```bash
sbatch --export=ALL,CONFIG=config/unitree_g1_flat_fast_sac.yaml experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
```

### FastTD3
```bash
sbatch --export=ALL,CONFIG=config/safety_gymnasium_fast_td3.yaml,NUM_ENVS=256 experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
```

### REPPO (G1 Flat)
```bash
sbatch --export=ALL,TASK_ID=Unitree-G1-Flat,CONFIG=config/unitree_g1_flat_reppo.yaml,NUM_ENVS=4096 experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
```

Evaluate a trained REPPO checkpoint on the G1 flat task:
```bash
python scripts/eval/unitree_mjlab.py Unitree-G1-Flat \
  --checkpoint <RUN_DIR>/model_<ITER>.pt \
  --train_cfg <RUN_DIR>/params/agent.yaml \
  --num_envs 1 \
  --device cuda:0 \
  --episodes 5 \
  --video
```

Use `--headless` instead of `--video` for metrics-only evaluation. If `--video_dir`
is omitted, videos are saved to `<checkpoint_dir>/videos/eval`.

All jobs run **wandb in offline mode**. After the job finishes, sync the run:
```bash
bash experiments/juwels_unitree_rl_mjlab/post_train.sh <JOB_ID>
```

---

## Safety-Gymnasium

The current JUWELS launchers run inside the `unitree_rl_mjlab` Apptainer image,
not a bare Python venv. Build/sync that image first:
```bash
bash experiments/juwels_unitree_rl_mjlab/sync_to_juwels.sh
bash experiments/juwels_unitree_rl_mjlab/build_unitree_rl_mjlab_image.sh
```

Common environments: `SafetyCarGoal1-v0`, `SafetyPointGoal1-v0`, `SafetyAntVelocity-v1`

### Local / interactive command

```bash
python scripts/train/train_safety_gymnasium.py \
  --env_id <ENV_ID> \
  --num_envs <N> \
  --config config/<CONFIG_FILE>
```

Add `--cost_limits <value>` for constrained runs when needed.

### Example local commands

**PPO:**
```bash
python scripts/train/train_safety_gymnasium.py --env_id SafetyCarGoal1-v0 --num_envs 36 --config config/safety_gymnasium_ppo.yaml
```

**FPPO with a cost limit:**
```bash
python scripts/train/train_safety_gymnasium.py --env_id SafetyCarGoal1-v0 --num_envs 36 --config config/safety_gymnasium_fppo.yaml --cost_limits 25.0
```

**FastSAC:**
```bash
python scripts/train/train_safety_gymnasium.py --env_id SafetyCarGoal1-v0 --num_envs 8 --config config/safety_gymnasium_fast_sac.yaml
```

### Start Safety-Gymnasium training with `sbatch`

**Generic JUWELS launcher (single GPU, defaults to `PPOL-PID`):**
```bash
sbatch experiments/juwels_safety_gym/run_safety_gym.sh
```

**Run a different config through the generic launcher (example: P3O):**
```bash
sbatch --export=ALL,CONFIG=config/safety_gymnasium_p3o.yaml,COST_LIMIT=20.0 experiments/juwels_safety_gym/run_safety_gym.sh
```

**Dedicated FPPO launcher:**
```bash
sbatch experiments/juwels_safety_gym/run_fppo.sh
```

**Override environment or number of envs at submit time:**
```bash
sbatch --export=ALL,ENV_ID=SafetyPointGoal1-v0,NUM_ENVS=32 experiments/juwels_safety_gym/run_fppo.sh
```

**Override cost limit or max iterations:**
```bash
sbatch --export=ALL,COST_LIMIT=5.0,MAX_ITER=2000 experiments/juwels_safety_gym/run_fppo.sh
```

**Dedicated P3O launcher (uses `safety_gymnasium_p3o.yaml`):**
```bash
sbatch --export=ALL,COST_LIMIT=5.0 experiments/juwels_safety_gym/run_p3o.sh
```

Other available Safety-Gymnasium submit scripts:
- `experiments/juwels_safety_gym/run_cpo.sh`
- `experiments/juwels_safety_gym/run_cup.sh`
- `experiments/juwels_safety_gym/run_fppo.sh`
- `experiments/juwels_safety_gym/run_p3o.sh`
- `experiments/juwels_safety_gym/run_pcpo.sh`
- `experiments/juwels_safety_gym/run_ppo.sh`
- `experiments/juwels_safety_gym/run_ppol_pid.sh`
- `experiments/juwels_safety_gym/run_sac.sh`
- `experiments/juwels_safety_gym/run_safe_sac.sh`
- `experiments/juwels_safety_gym/run_safe_sac_2.sh`
- `experiments/juwels_safety_gym/run_sac_simba.sh`

After submitting:
```bash
squeue -u $USER
tail -f logs/<job-name>-<JOB_ID>.out
```

### Evaluate a downloaded checkpoint

**Interactive evaluation (local render):**
```bash
python scripts/eval/eval_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 1 \
  --config config/safety_gymnasium_cpo.yaml \
  --checkpoint /Users/daaboul/workspaces/safe_rl/logs/safe_rl/13738388/model_499.pt \
  --cost_limits 25.0 \
  --episodes 5 \
  --render_mode human
```

**Save the first evaluation rollout as an mp4:**
```bash
python scripts/eval/eval_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 1 \
  --config config/safety_gymnasium_cpo.yaml \
  --checkpoint /Users/daaboul/workspaces/safe_rl/logs/safe_rl/13738388/model_499.pt \
  --cost_limits 25.0 \
  --episodes 5 \
  --video \
  --video_dir /Users/daaboul/workspaces/safe_rl/logs/safe_rl/13738388/videos/eval
```

When `--video` is enabled, the evaluator records the first rollout to `mp4`. If
`--video_dir` is omitted, it defaults to `<checkpoint_dir>/videos/eval`. Use
`--video_length` to control how many rollout steps are recorded.

### W&B sweeps
```bash
wandb sweep sweeps/safe_ppo_sweep.yaml
wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID>
```
