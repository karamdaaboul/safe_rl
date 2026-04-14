#!/bin/bash
# =============================================================================
# SLURM job — unitree_rl_mjlab smoke test on JUWELS Booster
#
# Runs a minimal training (few iterations) to verify the container + repo work.
#
# Submit:  sbatch experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_smoke.sbatch
# Logs:    tail -f logs/unitree_smoke-<jobid>.out
# =============================================================================
#SBATCH --job-name=unitree_smoke
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
PROJECT="/p/project1/hai_1075"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SCRATCH="/p/scratch/hai_1075/daaboul1"

TASK_ID="${TASK_ID:-Unitree-Go2-Flat}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITER="${MAX_ITER:-2}"
RUN_NAME="${RUN_NAME:-juwels_smoke_${SLURM_JOB_ID}}"
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p "$SCRATCH/unitree_rl_mjlab"/{logs,tmp,apptainer_tmp,.cache,wandb}

export APPTAINER_CACHEDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")
export APPTAINER_TMPDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")

echo "========================================"
echo " unitree_rl_mjlab Smoke Test"
echo " Job:     $SLURM_JOB_ID on $SLURMD_NODENAME"
echo " Task:    $TASK_ID | Envs: $NUM_ENVS | Iter: $MAX_ITER"
echo "========================================"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Container not found: $IMAGE"
    exit 1
fi

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH/unitree_rl_mjlab:/scratch" \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --pwd /scratch \
    "$IMAGE" \
    bash -lc "
        set -euo pipefail
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        export PYTHONUNBUFFERED=1
        python -c 'import torch; print(\"cuda_available:\", torch.cuda.is_available()); print(\"cuda_device_count:\", torch.cuda.device_count())'
        python /opt/unitree_rl_mjlab/scripts/list_envs.py --keyword Unitree
        python /opt/unitree_rl_mjlab/scripts/train.py $TASK_ID \
            --env.scene.num-envs=$NUM_ENVS \
            --agent.max_iterations=$MAX_ITER \
            --agent.run_name=$RUN_NAME
    "

echo "========================================"
echo " Smoke test finished."
echo "========================================"
