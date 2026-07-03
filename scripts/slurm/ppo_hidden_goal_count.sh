#!/bin/bash -l
#SBATCH --job-name=ppo_hg_count
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Reward-only (plain PPO) on a SINGLE fixed HIDDEN-GOAL layout, with the goal
# respawning on reach (continue_goal=True). Measures how many hidden goals a
# reward-only policy can chain in one episode (Episode/goals_reached in wandb).
# No cost constraint -- this is the "only the reward function" ablation.
#
# Usage: sbatch ppo_hidden_goal_count.sh [SEED] [NUM_ENVS] [ENV_ID] [CONFIG]
set -euo pipefail
SEED=${1:-0}
NUM_ENVS=${2:-40}
ENV_ID=${3:-SafetyCarGoal1-v0}
CONFIG=${4:-config/safety_gymnasium_ppo_hidden_goal.yaml}

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

echo "=== RUN ppo-hidden-goal-count config=$CONFIG seed=$SEED env=$ENV_ID num_envs=$NUM_ENVS host=$(hostname) ==="
python scripts/train/train_safety_gymnasium.py --env_id "$ENV_ID" --num_envs "$NUM_ENVS" --config "$CONFIG" --hidden_goal --hidden_goal_continue --seed "$SEED" --device cuda --log_dir $SCR/safe_rl/runs
echo "=== DONE ppo-hidden-goal-count seed=$SEED ==="
