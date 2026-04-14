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
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
PROJECT="/p/project1/hai_1075"
REPO="$PROJECT/daaboul1/unitree_rl_mjlab"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SCRATCH="/p/scratch/hai_1075/daaboul1"
SAFE_RL="$PROJECT/daaboul1/safe_rl"

TASK_ID="${TASK_ID:-Unitree-G1-Flat}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-1500}"
RUN_NAME="${RUN_NAME:-juwels_train_${SLURM_JOB_ID}}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
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
echo " GPUs:     $GPU_IDS"
echo "========================================"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Container not found: $IMAGE"
    echo "Build it first: bash experiments/juwels_unitree_rl_mjlab/build_unitree_rl_mjlab_image.sh"
    exit 1
fi

if [[ ! -d "$REPO" ]]; then
    echo "ERROR: Source checkout not found: $REPO"
    echo "Clone it once on JUWELS:"
    echo "  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git $REPO"
    exit 1
fi

if [[ ! -d "$SAFE_RL" ]]; then
    echo "ERROR: safe_rl not found at $SAFE_RL"
    echo "Clone it once on JUWELS:"
    echo "  git clone https://github.com/karamdaaboul/safe_rl.git $SAFE_RL"
    exit 1
fi

LOGDIR="$SCRATCH/unitree_rl_mjlab/logs"

# JUWELS binds one GPU per task by default. For one-task multi-GPU jobs we
# explicitly expose all four devices and let the upstream launcher spawn workers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$REPO:/workspace/unitree_rl_mjlab" \
    --bind "$LOGDIR:/workspace/logs" \
    --bind "$SAFE_RL:/opt/safe_rl" \
    --bind "$SCRATCH/unitree_rl_mjlab:/scratch" \
    --env PYTHONPATH=/opt/safe_rl:/workspace/unitree_rl_mjlab \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    --pwd /workspace \
    "$IMAGE" \
    bash -lc "
        set -euo pipefail
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        export PYTHONUNBUFFERED=1
        read -r -a GPU_IDS_ARR <<< \"$GPU_IDS\"
        python /workspace/unitree_rl_mjlab/scripts/train.py $TASK_ID \
            --gpu-ids \"\${GPU_IDS_ARR[@]}\" \
            --env.scene.num-envs=$NUM_ENVS \
            --agent.max_iterations=$MAX_ITER \
            --agent.run_name=$RUN_NAME
    "

echo "========================================"
echo " Training finished."
echo " Run wandb sync from a login node (has internet) to upload results."
echo "========================================"
