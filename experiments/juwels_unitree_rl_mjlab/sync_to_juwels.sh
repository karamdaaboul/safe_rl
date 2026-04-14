#!/usr/bin/env bash
# =============================================================================
# Sync safe_rl (this repo) to JUWELS from your Mac.
#
# Usage:
#   bash experiments/juwels_unitree_rl_mjlab/sync_to_juwels.sh
# =============================================================================
set -euo pipefail

JUWELS_USER="${JUWELS_USER:-daaboul1}"
JUWELS_HOST="${JUWELS_HOST:-juwels}"
PROJECT_PATH="/p/project1/hai_1075"
REMOTE_SAFE_RL="$PROJECT_PATH/$JUWELS_USER/safe_rl"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_DIR"

echo "=== Syncing safe_rl to JUWELS ==="
echo "Remote: ${JUWELS_USER}@${JUWELS_HOST}:${REMOTE_SAFE_RL}"
echo ""

ssh "${JUWELS_USER}@${JUWELS_HOST}" "mkdir -p ${REMOTE_SAFE_RL}"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.eggs' \
    --exclude='logs/' \
    --exclude='wandb/' \
    --exclude='outputs/' \
    --exclude='*.sif' \
    --exclude='.venv' \
    --exclude='venv' \
    ./ \
    "${JUWELS_USER}@${JUWELS_HOST}:${REMOTE_SAFE_RL}/"

echo ""
echo "=== Done. ==="
echo "On JUWELS, submit a job with:"
echo "  cd $REMOTE_SAFE_RL"
echo "  sbatch experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_smoke.sbatch"
echo "  sbatch experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch"
