#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-hai_1075}"
PARTITION="${PARTITION:-booster}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PROJECT_ROOT="${PROJECT_ROOT:-/p/project1/${ACCOUNT}/unitree_rl_mjlab}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src/unitree_rl_mjlab}"
IMAGE_PATH="${IMAGE_PATH:-${PROJECT_ROOT}/images/unitree_rl_mjlab_cuda12.4.1_py311.sif}"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Image not found: ${IMAGE_PATH}" >&2
    echo "Run build_unitree_rl_mjlab_image.sh first." >&2
    exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
    echo "Source checkout not found: ${SRC_DIR}" >&2
    exit 1
fi

srun \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres=gpu:1 \
    --time="${TIME_LIMIT}" \
    --pty \
    apptainer exec --nv \
    --bind "${SRC_DIR}:/workspace/unitree_rl_mjlab" \
    "${IMAGE_PATH}" \
    bash -lc '
        set -euo pipefail
        cd /workspace/unitree_rl_mjlab
        export MUJOCO_GL=egl
        export PYTHONPATH=/workspace/unitree_rl_mjlab:${PYTHONPATH:-}
        python - <<'"'"'PY'"'"'
import os
import torch
import mjlab
import src.tasks

print("cwd:", os.getcwd())
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("mjlab_import:", mjlab.__name__)
print("src_tasks_import:", src.tasks.__name__)
PY
        python scripts/list_envs.py Unitree
        exec bash
    '
