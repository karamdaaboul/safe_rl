#!/bin/bash -l
#SBATCH --job-name=p3o_cmp
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Usage: sbatch --job-name=<name> p3o_hlgauss_compare.sh <CONFIG> <SEED> [NUM_ENVS] [COST_V_MAX] [ENV_ID] [COST_LIMIT]
set -euo pipefail
CONFIG=$1
SEED=$2
NUM_ENVS=${3:-8}
COST_V_MAX=${4:-}
ENV_ID=${5:-SafetyCarGoal1-v0}
COST_LIMIT=${6:-25.0}

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

VMAX_ARG=""
if [ -n "$COST_V_MAX" ]; then VMAX_ARG="--cost_v_max $COST_V_MAX"; fi

echo "=== RUN config=$CONFIG seed=$SEED env=$ENV_ID num_envs=$NUM_ENVS cost_v_max=${COST_V_MAX:-config} cost_limit=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py --env_id "$ENV_ID" --num_envs "$NUM_ENVS" --config "$CONFIG" --cost_limits "$COST_LIMIT" --seed "$SEED" --device cuda --log_dir $SCR/safe_rl/runs $VMAX_ARG
echo "=== DONE config=$CONFIG seed=$SEED ==="
