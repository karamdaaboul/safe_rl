#!/bin/bash -l
#SBATCH --job-name=rcppo_run
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Full RCPPO (reachability-constrained PPO) run on the JUWELS Booster.
#
# COST_LIMIT is the feasibility threshold epsilon on the state-wise reachability level
# E[V_h] (typically ~0.1), NOT an episodic budget. Pass CONFIG=..._rcppo_filter.yaml to
# also enable the learned reachability safety filter during rollouts.
#
# Usage: sbatch rcppo_run.sh [NUM_ENVS] [MAX_ITERS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
NUM_ENVS=${1:-36}
MAX_ITERS=${2:-500}
COST_LIMIT=${3:-0.1}
ENV_ID=${4:-SafetyPointGoal1-v0}
CONFIG=${5:-config/safety_gymnasium_rcppo.yaml}

module --force purge
module load Stages/2024 GCCcore/.12.3.0 Python/3.11.3

source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

export MUJOCO_GL=egl
export OMP_NUM_THREADS=16
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true

SCR=/p/scratch/hai_1075
mkdir -p $SCR/safe_rl/logs $SCR/safe_rl/runs
cd /p/project1/hai_1075/workspaces/safe_rl

echo "=== RCPPO config=$CONFIG env=$ENV_ID num_envs=$NUM_ENVS max_iters=$MAX_ITERS eps=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --cost_limits "$COST_LIMIT" \
    --max_iterations "$MAX_ITERS" \
    --device cuda \
    --log_dir $SCR/safe_rl/runs
echo "=== DONE rcppo rc=$? ==="

# NOTE: do NOT `wandb sync` here. Booster compute nodes have no internet, so the sync
# hangs until the job hits its wall-clock limit (wasting the whole remaining allocation).
# Sync offline runs afterward from a login node instead: `bash wandb_sync.sh`.
