#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACCOUNT="${ACCOUNT:-hai_1075}"
PROJECT_ROOT="${PROJECT_ROOT:-/p/project1/${ACCOUNT}/unitree_rl_mjlab}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/p/scratch/${ACCOUNT}/unitree_rl_mjlab}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src/unitree_rl_mjlab}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_ROOT}/images}"
CACHE_DIR="${CACHE_DIR:-${SCRATCH_ROOT}/apptainer-cache}"
TMP_DIR="${TMP_DIR:-${SCRATCH_ROOT}/apptainer-tmp}"
IMAGE_NAME="${IMAGE_NAME:-unitree_rl_mjlab_cuda12.4.1_py311.sif}"
IMAGE_PATH="${IMAGE_DIR}/${IMAGE_NAME}"
DEF_TEMPLATE="${SCRIPT_DIR}/unitree_rl_mjlab.def.template"
GENERATED_DEF="${IMAGE_DIR}/unitree_rl_mjlab.generated.def"
REPO_URL="${REPO_URL:-https://github.com/unitreerobotics/unitree_rl_mjlab.git}"

mkdir -p "${PROJECT_ROOT}" "${SCRATCH_ROOT}" "${IMAGE_DIR}" "${CACHE_DIR}" "${TMP_DIR}"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available in PATH." >&2
    echo "Make sure your JUWELS account has container runtime access." >&2
    exit 1
fi

if [ ! -d "${SRC_DIR}/.git" ]; then
    mkdir -p "$(dirname "${SRC_DIR}")"
    git clone "${REPO_URL}" "${SRC_DIR}"
else
    git -C "${SRC_DIR}" pull --ff-only
fi

export APPTAINER_CACHEDIR="${CACHE_DIR}"
export APPTAINER_TMPDIR="${TMP_DIR}"

python3 - <<PY
from pathlib import Path

template = Path("${DEF_TEMPLATE}").read_text()
output = template.replace("__SRC_DIR__", "${SRC_DIR}")
Path("${GENERATED_DEF}").write_text(output)
PY

echo "Source checkout : ${SRC_DIR}"
echo "Image output    : ${IMAGE_PATH}"
echo "Cache dir       : ${APPTAINER_CACHEDIR}"
echo "Tmp dir         : ${APPTAINER_TMPDIR}"
echo "Definition file : ${GENERATED_DEF}"

apptainer build --fakeroot "${IMAGE_PATH}" "${GENERATED_DEF}"

echo
echo "Build completed."
echo "Next step:"
echo "  bash ${SCRIPT_DIR}/unitree_rl_mjlab_interactive_check.sh"
