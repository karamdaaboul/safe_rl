#!/bin/bash -l
#SBATCH --job-name=fzi_sweep
#SBATCH --partition=H100-Full
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Multi-seed sweep on the FZI cluster, run SEQUENTIALLY inside a single job.
#
# This is the pattern the cluster admins ask for: 10 x 1h training runs should be
# 10 subsequent `srun` calls inside ONE sbatch job, not 10 queued jobs. Keep the
# GPU count at 1-2 for anything >= 3h.
#
# Usage: sbatch scripts/slurm/fzi_sweep.sh [ENV_ID] [CONFIG] [COST_LIMIT] [MAX_ITERS] [SEEDS...]
set -uo pipefail
ENV_ID=${1:-SafetyPointGoal1-v0}
CONFIG=${2:-config/safety_gymnasium_p3o.yaml}
COST_LIMIT=${3:-25.0}
MAX_ITERS=${4:-500}
shift 4 2>/dev/null || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4 5)   # >= 5 seeds: safe-RL variance is large

source "${FZI_VENV:-$HOME/venvs/safe_rl}/bin/activate"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true

CODE_DIR=${FZI_CODE_DIR:-$HOME/workspaces/safe_rl}
SCR=/tmp/$USER/safe_rl/$SLURM_JOB_ID
HOME_RUNS=$HOME/safe_rl_runs/$SLURM_JOB_ID
mkdir -p "$SCR/runs" "$HOME_RUNS"
cd "$CODE_DIR"

copy_back() {
    echo "=== copying $SCR/runs -> $HOME_RUNS ==="
    cp -r "$SCR/runs/." "$HOME_RUNS/" 2>/dev/null || echo "(nothing to copy)"
    rm -rf "$SCR"
}
trap copy_back EXIT

echo "=== SWEEP env=$ENV_ID config=$CONFIG seeds=${SEEDS[*]} host=$(hostname) ==="
for seed in "${SEEDS[@]}"; do
    echo "----- seed $seed -----"
    srun --ntasks=1 --gpus=1 python scripts/train/train_safety_gymnasium.py \
        --env_id "$ENV_ID" \
        --num_envs 16 \
        --config "$CONFIG" \
        --cost_limits "$COST_LIMIT" \
        --max_iterations "$MAX_ITERS" \
        --seed "$seed" \
        --device cuda \
        --log_dir "$SCR/runs/seed_$seed"
    echo "----- seed $seed rc=$? -----"
done
echo "=== SWEEP DONE ==="
