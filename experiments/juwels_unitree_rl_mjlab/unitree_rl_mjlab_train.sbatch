#!/bin/bash -l
#SBATCH --job-name=unitree_train
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch/hai_1075/unitree_rl_mjlab/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/unitree_rl_mjlab/logs/%x-%j.err

set -euo pipefail

ACCOUNT="${ACCOUNT:-hai_1075}"
PROJECT_ROOT="${PROJECT_ROOT:-/p/project1/${ACCOUNT}/unitree_rl_mjlab}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/p/scratch/${ACCOUNT}/unitree_rl_mjlab}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src/unitree_rl_mjlab}"
IMAGE_PATH="${IMAGE_PATH:-${PROJECT_ROOT}/images/unitree_rl_mjlab_cuda12.4.1_py311.sif}"
TASK_ID="${TASK_ID:-Unitree-G1-Flat}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-1500}"
RUN_NAME="${RUN_NAME:-juwels_train_${SLURM_JOB_ID}}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3}"

mkdir -p "${SCRATCH_ROOT}/logs" "${SCRATCH_ROOT}/tmp"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Image not found: ${IMAGE_PATH}" >&2
    exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
    echo "Source checkout not found: ${SRC_DIR}" >&2
    exit 1
fi

export TASK_ID NUM_ENVS MAX_ITER RUN_NAME GPU_IDS_STRING

# JUWELS binds one GPU per task by default. For one-task multi-GPU jobs we
# explicitly expose all four devices and let the upstream launcher spawn workers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

srun apptainer exec --nv \
    --bind "${SRC_DIR}:/workspace/unitree_rl_mjlab" \
    --bind "${SCRATCH_ROOT}:/scratch" \
    "${IMAGE_PATH}" \
    bash -lc '
        set -euo pipefail
        cd /scratch
        export MUJOCO_GL=egl
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=/workspace/unitree_rl_mjlab:${PYTHONPATH:-}
        read -r -a GPU_IDS_ARR <<< "${GPU_IDS_STRING}"
        python /workspace/unitree_rl_mjlab/scripts/train.py "${TASK_ID}" \
            --gpu-ids "${GPU_IDS_ARR[@]}" \
            --env.scene.num-envs="${NUM_ENVS}" \
            --agent.max_iterations="${MAX_ITER}" \
            --agent.run_name="${RUN_NAME}"
    '
