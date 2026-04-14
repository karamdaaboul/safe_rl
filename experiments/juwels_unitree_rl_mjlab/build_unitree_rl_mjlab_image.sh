#!/usr/bin/env bash
# =============================================================================
# Build the unitree_rl_mjlab Apptainer image on the JUWELS login node.
#
# Prerequisites — run once before building:
#   1. Clone unitree_rl_mjlab on JUWELS:
#        git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \
#            /p/project1/hai_1075/daaboul1/unitree_rl_mjlab
#
# Usage (on JUWELS login node):
#   cd /p/project1/hai_1075/daaboul1/safe_rl
#   bash experiments/juwels_unitree_rl_mjlab/build_unitree_rl_mjlab_image.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="/p/project1/hai_1075"
USER_DIR="$PROJECT/daaboul1"
SCRATCH="/p/scratch/hai_1075/daaboul1"
REPO="$USER_DIR/unitree_rl_mjlab"
IMAGE="$USER_DIR/unitree_rl_mjlab.sif"
DEF_FILE="$SCRIPT_DIR/unitree_rl_mjlab.def.template"
CACHE_DIR="$SCRATCH/apptainer-cache"
TMP_DIR="$SCRATCH/apptainer-tmp"

mkdir -p "$USER_DIR" "$SCRATCH" "$CACHE_DIR" "$TMP_DIR"

APPTAINER="/usr/bin/apptainer"
if [[ ! -x "$APPTAINER" ]]; then
    echo "ERROR: apptainer not found at $APPTAINER"
    echo "Try: module load Apptainer"
    exit 1
fi

if [[ ! -d "$REPO/.git" ]]; then
    echo "ERROR: unitree_rl_mjlab repo not found at $REPO"
    echo "Clone it first:"
    echo "  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git $REPO"
    exit 1
fi

export APPTAINER_CACHEDIR="$CACHE_DIR"
export APPTAINER_TMPDIR="$TMP_DIR"

echo "Building $IMAGE from $DEF_FILE ..."
echo "Source: $REPO"
echo ""

$APPTAINER build \
    --fakeroot \
    --build-arg SRC_DIR="$REPO" \
    "$IMAGE" \
    "$DEF_FILE"

echo ""
echo "Build complete: $IMAGE"
echo "Size: $(du -sh "$IMAGE" | cut -f1)"
echo ""
echo "Next steps:"
echo "  sbatch $SCRIPT_DIR/unitree_rl_mjlab_smoke.sbatch   # verify"
echo "  sbatch $SCRIPT_DIR/unitree_rl_mjlab_train.sbatch   # train"
