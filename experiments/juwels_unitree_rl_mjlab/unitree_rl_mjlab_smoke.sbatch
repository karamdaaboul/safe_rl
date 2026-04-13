#!/bin/bash -l
#SBATCH --job-name=unitree_smoke
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/hai_1075/unitree_rl_mjlab/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/unitree_rl_mjlab/logs/%x-%j.err

set -euo pipefail

ACCOUNT="${ACCOUNT:-hai_1075}"
PROJECT_ROOT="${PROJECT_ROOT:-/p/project1/${ACCOUNT}/unitree_rl_mjlab}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/p/scratch/${ACCOUNT}/unitree_rl_mjlab}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src/unitree_rl_mjlab}"
IMAGE_PATH="${IMAGE_PATH:-${PROJECT_ROOT}/images/unitree_rl_mjlab_cuda12.4.1_py311.sif}"
APPTAINER="${APPTAINER:-/usr/bin/apptainer}"
TASK_ID="${TASK_ID:-Unitree-Go2-Flat}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITER="${MAX_ITER:-2}"
RUN_NAME="${RUN_NAME:-juwels_smoke_${SLURM_JOB_ID}}"

mkdir -p \
    "${SCRATCH_ROOT}/logs" \
    "${SCRATCH_ROOT}/tmp" \
    "${SCRATCH_ROOT}/apptainer_tmp" \
    "${SCRATCH_ROOT}/.cache" \
    "${SCRATCH_ROOT}/wandb"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Image not found: ${IMAGE_PATH}" >&2
    exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
    echo "Source checkout not found: ${SRC_DIR}" >&2
    exit 1
fi

export TASK_ID NUM_ENVS MAX_ITER RUN_NAME
export APPTAINER_CACHEDIR
APPTAINER_CACHEDIR="$(mktemp -d -p "${SCRATCH_ROOT}/apptainer_tmp")"
export APPTAINER_TMPDIR
APPTAINER_TMPDIR="$(mktemp -d -p "${SCRATCH_ROOT}/apptainer_tmp")"

srun "${APPTAINER}" exec --nv \
    --writable-tmpfs \
    --bind "${SRC_DIR}:/workspace/unitree_rl_mjlab" \
    --bind "${SCRATCH_ROOT}:/scratch" \
    --env PYTHONPATH=/workspace/unitree_rl_mjlab \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --pwd /scratch \
    "${IMAGE_PATH}" \
    bash -lc '
        set -euo pipefail
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        export PYTHONUNBUFFERED=1
        python - <<'"'"'PY'"'"'
import torch
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
        python /workspace/unitree_rl_mjlab/scripts/list_envs.py --keyword Unitree
        python /workspace/unitree_rl_mjlab/scripts/train.py "${TASK_ID}" \
            --env.scene.num-envs="${NUM_ENVS}" \
            --agent.max_iterations="${MAX_ITER}" \
            --agent.run_name="${RUN_NAME}"
    '
