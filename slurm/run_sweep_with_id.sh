#!/bin/bash
#SBATCH --job-name=go2_velocity_train
#SBATCH --partition=H100-MIG
#SBATCH --gpus=3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch run_sweep_with_id.sh SWEEP_ID
# Example: sbatch run_sweep_with_id.sh abc123def

# Check if sweep ID is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide sweep ID as argument"
    echo "Usage: sbatch run_sweep_with_id.sh SWEEP_ID"
    exit 1
fi

SWEEP_ID=$1

# Optional: Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Change to the project directory
cd /home/dh1659/workspace/test2/safe_rl

echo "Starting wandb agent with sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID