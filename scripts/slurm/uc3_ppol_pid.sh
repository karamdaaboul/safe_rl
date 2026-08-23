#!/bin/bash
# PPOL-PID on Safety-Gymnasium, bwUniCluster 3.0 (uc3 / KIT SCC).
#
#   sbatch scripts/slurm/uc3_ppol_pid.sh [ENV_ID] [COST_LIMIT] [NUM_ENVS] [MAX_ITER] [STEPS_PER_ENV]
#
# Defaults give the standard config run (500 iterations). For a quick smoke:
#   sbatch -p dev_gpu_h100 -t 00:25:00 scripts/slurm/uc3_ppol_pid.sh SafetyPointGoal1-v0 25.0 16 5
#
# Keep NUM_ENVS * STEPS_PER_ENV constant when changing NUM_ENVS, so the on-policy
# batch (and therefore the update) is unchanged and the comparison is fair. The
# config baseline is 16 x 2048 = 32768, so e.g. 64 envs pairs with 512 steps.
# Match --cpus-per-task to NUM_ENVS: the vec env is one process per env.
#
# Partitions (MaxTime): dev_gpu_h100 00:30, gpu_h100_short 00:30, gpu_h100 3-00:00.
# H100 nodes are 4 GPUs / 64 cores, so 16 cores per GPU is the proportionate ask.
#SBATCH --job-name=ppol_pid
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl/logs/%x-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl/logs/%x-%j.err

set -euo pipefail

ENV_ID="${1:-SafetyPointGoal1-v0}"
COST_LIMIT="${2:-25.0}"
NUM_ENVS="${3:-16}"
MAX_ITER="${4:-}"
STEPS_PER_ENV="${5:-}"

WS=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl

# ---- modules ----
module purge
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8

# ---- env ----
source "$WS/venvs/safe_rl311/bin/activate"

# osmesa is not available on these nodes; egl is.
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export GIT_PYTHON_REFRESH=quiet

# Compute nodes have no route to wandb.ai (the KIT VPN only covers the login
# nodes from outside), so log offline and sync from a login node afterwards:
#   cd $WS/safe_rl && wandb sync wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR="$WS/safe_rl"

mkdir -p "$WS/logs"
cd "$WS/safe_rl"

echo "=== job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION"
echo "=== env=$ENV_ID cost_limit=$COST_LIMIT num_envs=$NUM_ENVS max_iter=${MAX_ITER:-config} steps_per_env=${STEPS_PER_ENV:-config}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"

ARGS=(
  --env_id "$ENV_ID"
  --num_envs "$NUM_ENVS"
  --config config/safety_gymnasium_ppol_pid.yaml
  --cost_limits "$COST_LIMIT"
  --device cuda
  --log_dir "$WS/logs/safety_gymnasium"
)
[ -n "$MAX_ITER" ] && ARGS+=(--max_iterations "$MAX_ITER")
[ -n "$STEPS_PER_ENV" ] && ARGS+=(--num_steps_per_env "$STEPS_PER_ENV")

srun python scripts/train/train_safety_gymnasium.py "${ARGS[@]}"

echo "=== done: $(date)"
