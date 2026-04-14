#!/usr/bin/env bash
# =============================================================================
# Drop into a bash shell (or run a command) inside the unitree_rl_mjlab
# container on a compute node.
#
# Run this AFTER salloc (interactive_job.sh) has given you a node, or inside a
# SLURM job via: srun bash experiments/juwels_unitree_rl_mjlab/run_in_container.sh
#
# Usage:
#   bash experiments/juwels_unitree_rl_mjlab/run_in_container.sh           # shell
#   bash experiments/juwels_unitree_rl_mjlab/run_in_container.sh "python scripts/list_envs.py"
# =============================================================================
set -euo pipefail

PROJECT="/p/project1/hai_1075"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SAFE_RL_DIR="$PROJECT/daaboul1/safe_rl"
SCRATCH="/p/scratch/hai_1075/daaboul1"

CMD="${1:-bash}"

mkdir -p "$SCRATCH/unitree_rl_mjlab"/{logs,tmp,apptainer_tmp,.cache,wandb}

export APPTAINER_CACHEDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")
export APPTAINER_TMPDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH/unitree_rl_mjlab:/scratch" \
    --bind "$SAFE_RL_DIR:/workspace/safe_rl" \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --env PYTHONPATH=/workspace/safe_rl \
    --pwd /opt/unitree_rl_mjlab \
    "$IMAGE" \
    bash -lc "
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        ${CMD}
    "
