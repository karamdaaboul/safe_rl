#!/bin/bash -l
#SBATCH --job-name=cup_parallel
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

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
export WANDB_MODE=offline
export WANDB_SILENT=true

# (Optional) if your code reads this:
export GIT_PYTHON_REFRESH=quiet

# log dirs
SCR=/p/scratch/hai_1075
mkdir -p $SCR/safe_rl/logs $SCR/safe_rl/runs

cd /p/project1/hai_1075/workspaces/safe_rl

# ---- Cost thresholds to sweep ----
COST_LIMIT_1=10.0
COST_LIMIT_2=25.0
COST_LIMIT_3=50.0

# ---- train in parallel ----
# Training 1: CUP with cost_limit=10.0 on GPU 0
python -u scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_cup.yaml \
  --cost_limits $COST_LIMIT_1 \
  --device cuda:0 \
  2>&1 | sed "s/^/[CUP-${COST_LIMIT_1}] /" &

PID_CUP1=$!

# Training 2: CUP with cost_limit=25.0 on GPU 1
python -u scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_cup.yaml \
  --cost_limits $COST_LIMIT_2 \
  --device cuda:1 \
  2>&1 | sed "s/^/[CUP-${COST_LIMIT_2}] /" &

PID_CUP2=$!

# Training 3: CUP with cost_limit=50.0 on GPU 2
python -u scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_cup.yaml \
  --cost_limits $COST_LIMIT_3 \
  --device cuda:2 \
  2>&1 | sed "s/^/[CUP-${COST_LIMIT_3}] /" &

PID_CUP3=$!

echo "Started CUP training with cost_limit=$COST_LIMIT_1 (PID: $PID_CUP1) on cuda:0"
echo "Started CUP training with cost_limit=$COST_LIMIT_2 (PID: $PID_CUP2) on cuda:1"
echo "Started CUP training with cost_limit=$COST_LIMIT_3 (PID: $PID_CUP3) on cuda:2"

# Wait for all to finish
wait $PID_CUP1 $PID_CUP2 $PID_CUP3
echo "All three CUP training runs completed."
