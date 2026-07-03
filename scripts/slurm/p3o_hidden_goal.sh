#!/bin/bash -l
#SBATCH --job-name=p3o_hg
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Plain P3O baseline on the HIDDEN-GOAL env, trained jointly across a fixed set
# of goals ("5 seeds as tasks"). One non-adaptive policy, N goals spread across
# the parallel envs. This is the baseline to compare against cMAML adaptation.
#
# Usage: sbatch p3o_hidden_goal.sh [TASK_SEEDS] [NUM_ENVS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
TASK_SEEDS=${1:-0,1,2,3,4}
NUM_ENVS=${2:-40}                 # multiple of #seeds -> even split (8 envs/goal)
COST_LIMIT=${3:-15.0}
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

echo "=== RUN p3o-hidden-goal config=$CONFIG task_seeds=$TASK_SEEDS env=$ENV_ID num_envs=$NUM_ENVS cost_limit=$COST_LIMIT host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py --env_id "$ENV_ID" --num_envs "$NUM_ENVS" --config "$CONFIG" --hidden_goal --task_seeds "$TASK_SEEDS" --cost_limits "$COST_LIMIT" --device cuda --log_dir $SCR/safe_rl/runs
echo "=== DONE p3o-hidden-goal task_seeds=$TASK_SEEDS ==="
