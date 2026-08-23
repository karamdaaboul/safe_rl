#!/bin/bash -l
#SBATCH --job-name=fzi_train
#SBATCH --partition=H100-Full
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Safety-Gymnasium training on the FZI GPU cluster (fzi-gpu-mgmt-01 head node).
#
# Partitions:  H100-Full (fzi-gpu-01, 4x H100 80GB) | H200-Full (fzi-gpu-04, 4x H200 140GB, 24h cap)
# Override at submit time:  sbatch --partition=H200-Full --time=24:00:00 scripts/slurm/fzi_train.sh
#
# Etiquette: at most 2 GPUs for jobs >= 3h, or you block other users. For many
# short tasks, run them sequentially inside ONE job (see fzi_sweep.sh) rather
# than queueing one job per task.
#
# Storage: home is shared across nodes but on a slow network mount. Training
# writes to node-local /tmp (5TB NVMe) and the run dir is copied back to home on
# exit -- /tmp is NOT visible from the login node and does not survive the node.
#
# Usage: sbatch scripts/slurm/fzi_train.sh [NUM_ENVS] [MAX_ITERS] [COST_LIMIT] [ENV_ID] [CONFIG]
set -euo pipefail
NUM_ENVS=${1:-16}
MAX_ITERS=${2:-500}
COST_LIMIT=${3:-25.0}
ENV_ID=${4:-SafetyPointGoal1-v0}
CONFIG=${5:-config/safety_gymnasium_p3o.yaml}

# No module system on this cluster: plain python3-venv + system CUDA 12.9.
source "${FZI_VENV:-$HOME/venvs/safe_rl}/bin/activate"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true

CODE_DIR=${FZI_CODE_DIR:-$HOME/workspaces/safe_rl}
SCR=/tmp/$USER/safe_rl/$SLURM_JOB_ID          # node-local NVMe, io-sensitive work
HOME_RUNS=$HOME/safe_rl_runs/$SLURM_JOB_ID    # shared home, survives the job
mkdir -p "$SCR/runs" "$HOME_RUNS"
cd "$CODE_DIR"

# Copy results back even if training crashes or hits the wall clock.
copy_back() {
    echo "=== copying $SCR/runs -> $HOME_RUNS ==="
    cp -r "$SCR/runs/." "$HOME_RUNS/" 2>/dev/null || echo "(nothing to copy)"
    rm -rf "$SCR"
}
trap copy_back EXIT

echo "=== FZI config=$CONFIG env=$ENV_ID num_envs=$NUM_ENVS max_iters=$MAX_ITERS cost_limit=$COST_LIMIT ==="
echo "=== host=$(hostname) partition=$SLURM_JOB_PARTITION gpus=$SLURM_GPUS scratch=$SCR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --cost_limits "$COST_LIMIT" \
    --max_iterations "$MAX_ITERS" \
    --device cuda \
    --log_dir "$SCR/runs"
echo "=== DONE rc=$? ==="

# W&B runs offline here; sync afterward from the head node (which has network):
#   ssh fzi 'cd workspaces/safe_rl && wandb sync --sync-all wandb'
