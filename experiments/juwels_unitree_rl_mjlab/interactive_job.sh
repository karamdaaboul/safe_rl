#!/usr/bin/env bash
# =============================================================================
# Start an interactive session on JUWELS Booster for unitree_rl_mjlab work.
#
# Usage (from JUWELS login node):
#   bash experiments/juwels_unitree_rl_mjlab/interactive_job.sh
#
# Inside the session, drop into the container:
#   bash experiments/juwels_unitree_rl_mjlab/run_in_container.sh
# =============================================================================
set -euo pipefail

TIME="${TIME:-02:00:00}"
NUM_CPUS="${NUM_CPUS:-24}"
MEM="${MEM:-256G}"

echo "=== Requesting interactive node on JUWELS Booster ==="
echo "    Time:  $TIME  |  CPUs: $NUM_CPUS  |  Mem: $MEM"
echo
echo "After allocation, you are ON the compute node."
echo "Run your container commands directly, e.g.:"
echo "  bash experiments/juwels_unitree_rl_mjlab/run_in_container.sh"
echo

salloc \
    --account=hai_1075 \
    --partition=booster \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task="$NUM_CPUS" \
    --gres=gpu:1 \
    --mem="$MEM" \
    --time="$TIME"
