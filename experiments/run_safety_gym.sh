#!/bin/bash -l
#SBATCH --job-name=safetycar_parallel
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:2
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

# ---- train in parallel ----
# Training 1: PPO-Lagrangian on GPU 0
python -u scripts/train/safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_ppol_pid.yaml \
  --cost_limits 20.0 \
  --device cuda:0 \
  2>&1 | sed 's/^/[PPOL_PID] /' &

PID_PPOL=$!

# Training 2: P3O on GPU 1
python -u scripts/train/safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 40 \
  --config config/safety_gymnasium_p3o.yaml \
  --cost_limits 20.0 \
  --device cuda:1 \
  2>&1 | sed 's/^/[P3O] /' &

PID_P3O=$!

echo "Started PPOL_PID training (PID: $PID_PPOL) on cuda:0"
echo "Started P3O training (PID: $PID_P3O) on cuda:1"

# Wait for both to finish
wait $PID_PPOL $PID_P3O
echo "Both training runs completed."
