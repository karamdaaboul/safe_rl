#!/usr/bin/env bash
# Push the local working tree to the FZI cluster before submitting jobs.
#
# The FZI home is shared across all nodes but is on a slow network mount, so it
# holds code only — job outputs go to node-local /tmp (see fzi_train.sh).
#
# Usage: bash scripts/slurm/fzi_sync.sh [--delete]
set -euo pipefail

REMOTE=${FZI_HOST:-fzi}
REMOTE_DIR=${FZI_CODE_DIR:-workspaces/safe_rl}
LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

ssh "$REMOTE" "mkdir -p ~/$REMOTE_DIR"

# --delete is opt-in: it removes remote files that no longer exist locally,
# which would also wipe anything you created directly on the cluster.
rsync -avz --human-readable "$@" \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'wandb_logs/' \
    --exclude '*.egg-info/' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "=== synced $LOCAL_DIR -> $REMOTE:~/$REMOTE_DIR ==="
echo "Submit with:  ssh $REMOTE 'cd $REMOTE_DIR && sbatch scripts/slurm/fzi_train.sh'"
