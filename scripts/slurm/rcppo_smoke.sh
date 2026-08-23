#!/bin/bash -l
#SBATCH --job-name=rcppo_smoke
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Smoke test: verify RCPPO (reachability-constrained PPO) training starts and runs a
# few iterations end-to-end. Tiny run (few envs, few short iterations) — NOT for results.
#
# NOTE: for RCPPO, COST_LIMIT is the feasibility threshold epsilon on the state-wise
# reachability level E[V_h] (typically ~0.1), NOT an episodic cost budget like P3O's 25.
#
# Usage: sbatch rcppo_smoke.sh [NUM_ENVS] [MAX_ITERS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
NUM_ENVS=${1:-8}
MAX_ITERS=${2:-5}
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

echo "=== SMOKE rcppo config=$CONFIG env=$ENV_ID num_envs=$NUM_ENVS max_iters=$MAX_ITERS eps=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --cost_limits "$COST_LIMIT" \
    --max_iterations "$MAX_ITERS" \
    --num_steps_per_env 256 \
    --device cuda \
    --log_dir $SCR/safe_rl/runs
echo "=== DONE rcppo-smoke rc=$? ==="
