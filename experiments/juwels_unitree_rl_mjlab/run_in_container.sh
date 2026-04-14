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
REPO="$PROJECT/daaboul1/unitree_rl_mjlab"
IMAGE="$PROJECT/daaboul1/unitree_rl_mjlab.sif"
SCRATCH="/p/scratch/hai_1075/daaboul1"
SAFE_RL="$PROJECT/daaboul1/safe_rl"

CMD="${1:-bash}"

mkdir -p "$SCRATCH/unitree_rl_mjlab"/{logs,tmp,apptainer_tmp,.cache,wandb}

export APPTAINER_CACHEDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")
export APPTAINER_TMPDIR=$(mktemp -d -p "$SCRATCH/unitree_rl_mjlab/apptainer_tmp")

/usr/bin/apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$REPO:/workspace/unitree_rl_mjlab" \
    --bind "$SAFE_RL:/opt/safe_rl" \
    --bind "$SCRATCH/unitree_rl_mjlab:/scratch" \
    --env PYTHONPATH=/opt/safe_rl:/workspace/unitree_rl_mjlab \
    --env MUJOCO_GL=egl \
    --env XDG_CACHE_HOME=/scratch/.cache \
    --env TMPDIR=/scratch/tmp \
    --env WANDB_MODE=offline \
    --env WANDB_SILENT=true \
    --env WANDB_DIR=/scratch/wandb \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMBA_NUM_THREADS=1 \
    --pwd /workspace/unitree_rl_mjlab \
    "$IMAGE" \
    bash -lc "
        mkdir -p /scratch/.cache /scratch/tmp /scratch/wandb
        ${CMD}
    "
