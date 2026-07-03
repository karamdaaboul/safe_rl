#!/bin/bash -l
#SBATCH --job-name=p3o_smoke
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Smoke test: verify P3O training starts and runs a few iterations end-to-end.
# Tiny run (few envs, few short iterations) — NOT for results.
#
# Usage: sbatch p3o_smoke.sh [NUM_ENVS] [MAX_ITERS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
NUM_ENVS=${1:-8}
MAX_ITERS=${2:-5}
COST_LIMIT=${3:-25.0}
ENV_ID=${4:-SafetyCarGoal1-v0}
CONFIG=${5:-config/safety_gymnasium_p3o.yaml}

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

echo "=== SMOKE p3o config=$CONFIG env=$ENV_ID num_envs=$NUM_ENVS max_iters=$MAX_ITERS cost_limit=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --cost_limits "$COST_LIMIT" \
    --max_iterations "$MAX_ITERS" \
    --num_steps_per_env 256 \
    --device cuda \
    --log_dir $SCR/safe_rl/runs
echo "=== DONE p3o-smoke rc=$? ==="
