#!/usr/bin/env bash
# =============================================================================
# One-command training submission from your Mac directly to JUWELS.
#
# Usage:
#   bash experiments/juwels_unitree_rl_mjlab/train_from_mac.sh \
#       --task Unitree-G1-Flat --num-envs 4096 --max-iter 1500
#
# Prerequisites:
#   1. Register your Mac's IP in JuDoor: https://judoor.fz-juelich.de
#   2. Add SSH ControlMaster to ~/.ssh/config, authenticate once via OTP.
# =============================================================================
set -euo pipefail

JUWELS_USER="daaboul1"
JUWELS_HOST="juwels.fz-juelich.de"
JUWELS_PROJECT="/p/project1/hai_1075"
JUWELS_SAFE_RL="${JUWELS_PROJECT}/daaboul1/safe_rl"

TASK_ID="Unitree-G1-Flat"
NUM_ENVS="4096"
MAX_ITER="1500"
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)      TASK_ID="$2"; shift 2 ;;
        --num-envs)  NUM_ENVS="$2"; shift 2 ;;
        --max-iter)  MAX_ITER="$2"; shift 2 ;;
        --run-name)  RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================"
echo " unitree_rl_mjlab — Submit Training from Mac"
echo " Task:    $TASK_ID"
echo " Envs:    $NUM_ENVS  |  Max iter: $MAX_ITER"
echo "============================================"
echo ""

SSH_SOCK="/tmp/juwels-train-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_SOCK -o ControlPersist=60"
REMOTE="${JUWELS_USER}@${JUWELS_HOST}"

echo "Opening SSH connection (authenticate once if needed)..."
ssh $SSH_OPTS -fNM "$REMOTE"
trap "ssh -o ControlPath=$SSH_SOCK -O exit $REMOTE 2>/dev/null" EXIT

RSYNC_SSH="ssh -o ControlPath=$SSH_SOCK"

echo "[1/2] Syncing safe_rl to JUWELS..."
rsync -az --progress -e "$RSYNC_SSH" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='logs/' \
    --exclude='wandb/' \
    --exclude='outputs/' \
    --exclude='*.sif' \
    --exclude='.venv' \
    --exclude='venv' \
    "$REPO_DIR/" \
    "${REMOTE}:${JUWELS_SAFE_RL}/"
echo "   Done."
echo ""

echo "[2/2] Submitting training job on JUWELS..."
EXPORTS="ALL,TASK_ID=${TASK_ID},NUM_ENVS=${NUM_ENVS},MAX_ITER=${MAX_ITER}"
if [[ -n "$RUN_NAME" ]]; then
    EXPORTS="${EXPORTS},RUN_NAME=${RUN_NAME}"
fi

OUT=$(ssh -o ControlPath="$SSH_SOCK" "$REMOTE" \
    "cd ${JUWELS_SAFE_RL} && mkdir -p logs && sbatch --export=${EXPORTS} experiments/juwels_unitree_rl_mjlab/unitree_rl_mjlab_train.sbatch")
JOB_ID=$(echo "$OUT" | grep -oE '[0-9]+$')
echo "   $OUT"

echo ""
echo "============================================"
echo " Job submitted: $JOB_ID"
echo ""
echo " Monitor:"
echo "   ssh ${REMOTE} 'squeue -u ${JUWELS_USER}'"
echo "   ssh ${REMOTE} 'tail -f ${JUWELS_SAFE_RL}/logs/unitree_train-${JOB_ID}.out'"
echo "============================================"
