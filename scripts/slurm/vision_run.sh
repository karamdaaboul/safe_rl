#!/bin/bash -l
#SBATCH --job-name=vision_ppol_pid
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Phase-1 vision safe-RL run on the JUWELS Booster: PPOL-PID on a *Vision-v0
# Safety-Gymnasium env with a frozen pretrained encoder (ResNet18). The actor
# consumes [encoder features, proprio]; reward+cost critics stay asymmetric on
# the full ground-truth state. PPO-PID was the best constrained method on the
# vision-only HASARD benchmark (ICLR 2025).
#
# Usage: sbatch vision_run.sh [NUM_ENVS] [MAX_ITERS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
NUM_ENVS=${1:-16}
MAX_ITERS=${2:-500}
COST_LIMIT=${3:-25.0}
ENV_ID=${4:-SafetyCarGoal1Vision-v0}
CONFIG=${5:-config/safety_gymnasium_ppol_pid_vision.yaml}

module --force purge
module load Stages/2024 GCCcore/.12.3.0 Python/3.11.3

source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

# Compute nodes are offline: read pretrained encoder weights from the shared
# project cache pre-populated on a login node (home is over quota).
export TORCH_HOME=/p/project1/hai_1075/torch_cache
export MUJOCO_GL=egl
export OMP_NUM_THREADS=16
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true

SCR=/p/scratch/hai_1075
mkdir -p $SCR/safe_rl/logs $SCR/safe_rl/runs
cd /p/project1/hai_1075/workspaces/safe_rl

echo "=== VISION config=$CONFIG env=$ENV_ID num_envs=$NUM_ENVS max_iters=$MAX_ITERS cost_limit=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --cost_limits "$COST_LIMIT" \
    --max_iterations "$MAX_ITERS" \
    --device cuda \
    --log_dir $SCR/safe_rl/runs
echo "=== DONE vision rc=$? ==="

# NOTE: do NOT run `wandb sync` in-job — on the offline Booster nodes it blocks
# until the wall-time limit. Sync the offline runs from a login node afterward:
#   cd /p/project1/hai_1075/workspaces/safe_rl && for d in wandb/offline-run-*; do wandb sync "$d" --sync-all; done
