#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-hai_1075}"
PROJECT_ROOT="${PROJECT_ROOT:-/p/project1/${ACCOUNT}/unitree_rl_mjlab}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/p/scratch/${ACCOUNT}/unitree_rl_mjlab}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src/unitree_rl_mjlab}"
IMAGE_PATH="${IMAGE_PATH:-${PROJECT_ROOT}/images/unitree_rl_mjlab_cuda12.4.1_py311.sif}"
APPTAINER="${APPTAINER:-/usr/bin/apptainer}"
CMD="${1:-bash}"

mkdir -p \
    "${SCRATCH_ROOT}/logs" \
    "${SCRATCH_ROOT}/tmp" \
    "${SCRATCH_ROOT}/apptainer_tmp" \
    "${SCRATCH_ROOT}/.cache" \
    "${SCRATCH_ROOT}/wandb"

export APPTAINER_CACHEDIR
APPTAINER_CACHEDIR="$(mktemp -d -p "${SCRATCH_ROOT}/apptainer_tmp")"
export APPTAINER_TMPDIR
APPTAINER_TMPDIR="$(mktemp -d -p "${SCRATCH_ROOT}/apptainer_tmp")"

"${APPTAINER}" exec \
    --nv \
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
    --pwd /workspace/unitree_rl_mjlab \
    "${IMAGE_PATH}" \
    bash -lc "
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        ${CMD}
    "
