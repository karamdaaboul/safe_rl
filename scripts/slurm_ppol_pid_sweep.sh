#!/bin/bash -l
#SBATCH --job-name=ppol_pid_sweep
#SBATCH --account=hai_1075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
#SBATCH --time=08:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

set -euo pipefail

# ---- modules ----
module --force purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3

# ---- env ----
source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

export MUJOCO_GL=egl
export OMP_NUM_THREADS=16
export GIT_PYTHON_REFRESH=quiet

# log dirs
SCR=/p/scratch/hai_1075
mkdir -p $SCR/safe_rl/logs $SCR/safe_rl/runs

cd /p/project1/hai_1075/workspaces/safe_rl

# ========== SWEEP CONFIGURATION ==========
# Set your sweep ID here (from: wandb sweep sweeps/ppol_pid_sweep.yaml)
SWEEP_ID="YOUR_USERNAME/YOUR_PROJECT/SWEEP_ID"

# Number of runs per agent (total = 3 agents * RUNS_PER_AGENT)
RUNS_PER_AGENT=10

# ========== OPTION A: ONLINE MODE ==========
# Uncomment if compute nodes have internet access
# export WANDB_MODE=online

# ========== OPTION B: OFFLINE MODE ==========
# Results sync after job completion
export WANDB_MODE=offline
export WANDB_SILENT=true

# ---- Run 3 agents in parallel, one per GPU ----
echo "Starting 3 wandb agents for sweep: $SWEEP_ID"

CUDA_VISIBLE_DEVICES=0 wandb agent --count $RUNS_PER_AGENT $SWEEP_ID \
  2>&1 | sed "s/^/[GPU0] /" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 wandb agent --count $RUNS_PER_AGENT $SWEEP_ID \
  2>&1 | sed "s/^/[GPU1] /" &
PID1=$!

CUDA_VISIBLE_DEVICES=2 wandb agent --count $RUNS_PER_AGENT $SWEEP_ID \
  2>&1 | sed "s/^/[GPU2] /" &
PID2=$!

echo "Agent PIDs: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2"

wait $PID0 $PID1 $PID2
echo "All agents completed."

# ---- Sync offline runs (if using offline mode) ----
if [ "${WANDB_MODE:-online}" = "offline" ]; then
    echo "Syncing offline runs..."
    for dir in wandb/offline-run-*; do
        if [ -d "$dir" ]; then
            wandb sync "$dir" --sync-all || echo "Failed to sync $dir"
        fi
    done
    echo "Sync complete."
fi
