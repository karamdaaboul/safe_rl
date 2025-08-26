#!/bin/bash
#SBATCH --job-name=go2_velocity_sweep
#SBATCH --partition=H100-MIG
#SBATCH --gpus=3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=1-3  # Run 3 agents in parallel

# Usage: sbatch run_sweep_array.sh SWEEP_ID
# Example: sbatch run_sweep_array.sh sfpp6mok

# Check if sweep ID is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide sweep ID as argument"
    echo "Usage: sbatch run_sweep_array.sh SWEEP_ID"
    echo "Example: sbatch run_sweep_array.sh sfpp6mok"
    exit 1
fi

SWEEP_ID=$1
AGENT_ID=$SLURM_ARRAY_TASK_ID

# Optional: Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Change to the project directory
cd /home/daaboul@fzi.de/workspaces/safe_rl

echo "Starting wandb agent $AGENT_ID with sweep ID: $SWEEP_ID"
wandb agent uqerh-kit/safe_rl/$SWEEP_ID 