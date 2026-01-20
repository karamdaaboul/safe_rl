#!/bin/bash -l
#SBATCH --job-name=safetycar_p3o
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch1/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch1/hai_1075/safe_rl/logs/%x-%j.err

set -euo pipefail

# ---- modules (must match your working interactive setup) ----
module --force purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3

# ---- env ----
source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

# Headless MuJoCo
export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable wandb interactive prompt + internet
export WANDB_MODE=disabled
export WANDB_SILENT=true

# (Optional) if your code reads this:
export GIT_PYTHON_REFRESH=quiet

# ---- dirs ----
RUNS_DIR=/p/scratch1/hai_1075/safe_rl/runs
LOGS_DIR=/p/scratch1/hai_1075/safe_rl/logs
mkdir -p "$RUNS_DIR" "$LOGS_DIR"

cd /p/project1/hai_1075/workspaces/safe_rl

# ---- train ----
python -u scripts/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_p3o.yaml \
  --cost_limits 25.0 \
  --device cuda:0
