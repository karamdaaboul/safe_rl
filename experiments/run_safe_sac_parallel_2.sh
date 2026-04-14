#!/bin/bash -l
#SBATCH --job-name=safe_sac_pid_sweep_2
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

set -euo pipefail

# ---- modules ----
module --force purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3

# ---- env ----
source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

# Headless MuJoCo
export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_SILENT=true
export GIT_PYTHON_REFRESH=quiet

# log dirs
SCR=/p/scratch/hai_1075
mkdir -p $SCR/safe_rl/logs $SCR/safe_rl/runs

cd /p/project1/hai_1075/workspaces/safe_rl

# ---- Environment settings ----
ENV_ID="SafetyCarGoal1-v0"
NUM_ENVS=64
MAX_ITERATIONS=100000

echo "=========================================="
echo "Safe SAC PID Parameter Sweep - Batch 2"
echo "Cost Limit: 50.0 (all configs)"
echo "Testing 3 configurations in parallel"
echo "=========================================="

# Config 4: High Smoothing (slower, more stable lambda updates)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_high_smooth.yaml \
  --device cuda:0 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[High-Smooth] /" &
PID1=$!

# Config 5: Low Kp (gentler proportional control)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_low_kp.yaml \
  --device cuda:1 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[Low-Kp] /" &
PID2=$!

# Config 6: No sum_norm (no cost normalization, lower lambda_max)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_no_sumnorm.yaml \
  --device cuda:2 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[No-SumNorm] /" &
PID3=$!

echo "Started 3 configs in parallel:"
echo "  cuda:0 - High-Smooth  (PID: $PID1)"
echo "  cuda:1 - Low-Kp       (PID: $PID2)"
echo "  cuda:2 - No-SumNorm   (PID: $PID3)"

# Wait for all to finish
wait $PID1 $PID2 $PID3

echo ""
echo "=========================================="
echo "Batch 2 completed!"
echo "=========================================="
