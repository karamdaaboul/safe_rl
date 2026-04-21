#!/bin/bash
# =============================================================================
# SLURM job — CPO parallel sweep on JUWELS Booster (A100 40GB, CUDA 12)
# Runs inside the unitree_rl_mjlab Apptainer image (has safety-gymnasium +
# mjlab installed) — NOT the bare venv.
#
# Node specs: 4× A100 40GB | 48 CPU cores | 512 GB RAM
# Runs 2 CPO configs side-by-side, one per GPU.
#
# Submit:
#   sbatch experiments/run_cpo_parallel.sh
#
# Override defaults:
#   sbatch --export=ALL,ENV_ID=SafetyPointGoal1-v0,NUM_ENVS=32 \
#       experiments/run_cpo_parallel.sh
#   sbatch --export=ALL,COST_LIMIT_1=5.0,COST_LIMIT_2=25.0,MAX_ITER=2000 \
#       experiments/run_cpo_parallel.sh
#
# Monitor: squeue -u $USER
# Logs:    tail -f logs/cpo_parallel-<jobid>.out
# =============================================================================
#SBATCH --job-name=cpo_parallel
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.k.daaboul@gmail.com

set -euo pipefail

# ---------------------------------------------------------------------------
PROJECT="/p/project1/hai_1075"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SAFE_RL_DIR="$PROJECT/daaboul1/safe_rl"
SCRATCH="/p/scratch/hai_1075/daaboul1/safe_rl"

ENV_ID="${ENV_ID:-SafetyCarGoal1-v0}"
NUM_ENVS="${NUM_ENVS:-48}"
MAX_ITER="${MAX_ITER:-}"
CONFIG="${CONFIG:-config/safety_gymnasium_cpo.yaml}"
COST_LIMIT_1="${COST_LIMIT_1:-10.0}"
COST_LIMIT_2="${COST_LIMIT_2:-25.0}"
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p "$SCRATCH"/{logs,tmp,apptainer_tmp,.cache,wandb,runs}

# Required per JSC docs: point apptainer cache/tmp to writable scratch
export APPTAINER_CACHEDIR=$(mktemp -d -p "$SCRATCH/apptainer_tmp")
export APPTAINER_TMPDIR=$(mktemp -d -p "$SCRATCH/apptainer_tmp")

echo "========================================"
echo " CPO Parallel Sweep (Apptainer)"
echo " Job:       $SLURM_JOB_ID on $SLURMD_NODENAME"
echo " Image:     $IMAGE"
echo " Env:       $ENV_ID"
echo " Num envs:  $NUM_ENVS per run"
echo " Max iter:  ${MAX_ITER:-<from config>}"
echo " Config:    $CONFIG"
echo " Limits:    [$COST_LIMIT_1, $COST_LIMIT_2]"
echo " GPUs:      ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "========================================"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Container not found: $IMAGE"
    exit 1
fi

if [[ ! -d "$SAFE_RL_DIR" ]]; then
    echo "ERROR: safe_rl repo not found at $SAFE_RL_DIR"
    echo "Sync first: bash experiments/juwels_unitree_rl_mjlab/sync_to_juwels.sh"
    exit 1
fi

if [[ ! -f "$SAFE_RL_DIR/$CONFIG" ]]; then
    echo "ERROR: config not found at $SAFE_RL_DIR/$CONFIG"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH:/scratch" \
    --bind "$SAFE_RL_DIR:/workspace/safe_rl" \
    --env PYTHONPATH=/workspace/safe_rl \
    --env PYTHONUNBUFFERED=1 \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env TRITON_CACHE_DIR=/scratch/.cache/triton \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env GIT_PYTHON_REFRESH=quiet \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    --env ENV_ID="$ENV_ID" \
    --env NUM_ENVS="$NUM_ENVS" \
    --env MAX_ITER="$MAX_ITER" \
    --env CONFIG="$CONFIG" \
    --env COST_LIMIT_1="$COST_LIMIT_1" \
    --env COST_LIMIT_2="$COST_LIMIT_2" \
    --pwd /workspace/safe_rl \
    "$IMAGE" \
    bash -lc '
        set -euo pipefail
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb /scratch/logs

        echo "[container] python: $(which python) | $(python --version)"
        echo "[container] safety_gymnasium: $(python -c "import safety_gymnasium; print(safety_gymnasium.__version__, safety_gymnasium.__file__)")"

        MAX_ITER_FLAG=""
        if [[ -n "$MAX_ITER" ]]; then
            MAX_ITER_FLAG="--max_iterations $MAX_ITER"
        fi

        CUDA_VISIBLE_DEVICES=0 python -u scripts/train/train_safety_gymnasium.py \
            --env_id "$ENV_ID" \
            --num_envs "$NUM_ENVS" \
            --config "$CONFIG" \
            --cost_limits "$COST_LIMIT_1" \
            --device cuda:0 \
            $MAX_ITER_FLAG \
            2>&1 | sed "s/^/[CPO-${COST_LIMIT_1}] /" &
        PID1=$!

        CUDA_VISIBLE_DEVICES=1 python -u scripts/train/train_safety_gymnasium.py \
            --env_id "$ENV_ID" \
            --num_envs "$NUM_ENVS" \
            --config "$CONFIG" \
            --cost_limits "$COST_LIMIT_2" \
            --device cuda:0 \
            $MAX_ITER_FLAG \
            2>&1 | sed "s/^/[CPO-${COST_LIMIT_2}] /" &
        PID2=$!

        echo "Started CPO cost_limit=$COST_LIMIT_1 (PID: $PID1) on GPU 0"
        echo "Started CPO cost_limit=$COST_LIMIT_2 (PID: $PID2) on GPU 1"

        wait $PID1 $PID2
    '

echo "========================================"
echo " Both CPO runs finished."
echo " Run wandb sync from a login node to upload results:"
echo "   wandb sync $SCRATCH/wandb/offline-run-*"
echo "========================================"
