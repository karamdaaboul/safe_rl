#!/bin/bash -l
#SBATCH --job-name=safe_sac_pid_sweep
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
NUM_ENVS=96
MAX_ITERATIONS=100000

echo "=========================================="
echo "Safe SAC PID Parameter Sweep - Batch 1"
echo "Cost Limit: 50.0 (all configs)"
echo "Testing 3 configurations in parallel"
echo "=========================================="

# Config 1: PI Controller Only (no derivative - most stable)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_pi_only.yaml \
  --device cuda:0 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[PI-Only] /" &
PID1=$!

# Config 2: Low Kd (reduced derivative oscillations)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_low_kd.yaml \
  --device cuda:1 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[Low-Kd] /" &
PID2=$!

# Config 3: Baseline (current settings for comparison)
python -u scripts/train/safety_gymnasium.py \
  --env_id $ENV_ID \
  --num_envs $NUM_ENVS \
  --config config/safesac/safety_gymnasium_safe_sac_baseline.yaml \
  --device cuda:2 \
  --max_iterations $MAX_ITERATIONS \
  2>&1 | sed "s/^/[Baseline] /" &
PID3=$!

echo "Started 3 configs in parallel:"
echo "  cuda:0 - PI-Only      (PID: $PID1)"
echo "  cuda:1 - Low-Kd       (PID: $PID2)"
echo "  cuda:2 - Baseline     (PID: $PID3)"

# Wait for all to finish
wait $PID1 $PID2 $PID3

echo ""
echo "=========================================="
echo "Batch 1 completed!"
echo "=========================================="
