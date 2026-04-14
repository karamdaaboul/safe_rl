#!/bin/bash
# =============================================================================
# SLURM job — unitree_rl_mjlab training on JUWELS Booster (A100 40GB, CUDA 12)
#
# Node specs: 4× A100 40GB | 48 CPU cores | 512 GB RAM
#
# Submit:
#   sbatch experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
#
# Override defaults:
#   sbatch --export=ALL,TASK_ID=Unitree-Go2-Flat,NUM_ENVS=2048 \
#       experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch
#
# Monitor: squeue -u $USER
# Logs:    tail -f logs/unitree_train-<jobid>.out
# =============================================================================
#SBATCH --job-name=unitree_train
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
PROJECT="/p/project1/hai_1075"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SAFE_RL_DIR="$PROJECT/daaboul1/safe_rl"
SCRATCH="/p/scratch/hai_1075/daaboul1"

TASK_ID="${TASK_ID:-Unitree-G1-Flat}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-1500}"
RUN_NAME="${RUN_NAME:-juwels_train_${SLURM_JOB_ID}}"
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p "$SCRATCH/unitree_rl_mjlab"/{logs,tmp,apptainer_tmp,.cache,wandb}

# Required per JSC docs: point apptainer cache/tmp to writable scratch
export APPTAINER_CACHEDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")
export APPTAINER_TMPDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")

echo "========================================"
echo " unitree_rl_mjlab Training"
echo " Job:      $SLURM_JOB_ID on $SLURMD_NODENAME"
echo " Task:     $TASK_ID"
echo " Envs:     $NUM_ENVS"
echo " Max iter: $MAX_ITER"
echo " GPUs:     ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "========================================"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Container not found: $IMAGE"
    echo "Build it first: bash experiments/juwels_unitree_rl_mjlab/build_unitree_rl_mjlab_image.sh"
    exit 1
fi

if [[ ! -f "$SAFE_RL_DIR/scripts/train/unitree_mjlab.py" ]]; then
    echo "ERROR: safe_rl training script not found at $SAFE_RL_DIR"
    echo "Sync this repo to JUWELS first:"
    echo "  bash experiments/juwels_unitree_rl_mjlab/sync_to_juwels.sh"
    exit 1
fi

# JUWELS binds one GPU per task by default. For one-task multi-GPU jobs we
# explicitly expose all four devices and let the upstream launcher spawn workers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH/unitree_rl_mjlab:/scratch" \
    --bind "$SAFE_RL_DIR:/workspace/safe_rl" \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    --env MASTER_ADDR=127.0.0.1 \
    --env MASTER_PORT=29500 \
    --env GLOO_SOCKET_IFNAME=lo \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    --pwd /scratch \
    "$IMAGE" \
    bash -lc "
        set -euo pipefail
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        export PYTHONUNBUFFERED=1
        cd /workspace/safe_rl
        python scripts/train/unitree_mjlab.py $TASK_ID \
            --num_envs $NUM_ENVS \
            --max_iterations $MAX_ITER \
            --run_name $RUN_NAME \
            --log_dir /scratch/logs/safe_rl
    "

echo "========================================"
echo " Training finished."
echo " Run wandb sync from a login node (has internet) to upload results."
echo "========================================"
