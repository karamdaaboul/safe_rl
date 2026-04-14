#!/bin/bash -l
#SBATCH --job-name=sac_parallel
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

# ---- Seeds to sweep ----
SEED_1=1
SEED_2=42
SEED_3=123

# ---- Environment ----
ENV_ID="SafetyCarGoal1-v0"
NUM_ENVS=64
MAX_ITERATIONS=100000

# ---- train in parallel ----
# Training 1: SAC with seed=1 on GPU 0
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safety_gymnasium_sac.yaml \
  --device cuda:0 \
  --seed $SEED_1 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[SAC-seed${SEED_1}] /" &

PID_SAC1=$!

# Training 2: SAC with seed=42 on GPU 1
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safety_gymnasium_sac.yaml \
  --device cuda:1 \
  --seed $SEED_2 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[SAC-seed${SEED_2}] /" &

PID_SAC2=$!

# Training 3: SAC with seed=123 on GPU 2
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safety_gymnasium_sac.yaml \
  --device cuda:2 \
  --seed $SEED_3 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[SAC-seed${SEED_3}] /" &

PID_SAC3=$!

echo "Started SAC training with seed=$SEED_1 (PID: $PID_SAC1) on cuda:0"
echo "Started SAC training with seed=$SEED_2 (PID: $PID_SAC2) on cuda:1"
echo "Started SAC training with seed=$SEED_3 (PID: $PID_SAC3) on cuda:2"

# Wait for all to finish
wait $PID_SAC1 $PID_SAC2 $PID_SAC3
echo "All three SAC training runs completed."
