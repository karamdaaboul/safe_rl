#!/bin/bash -l
#SBATCH --job-name=reppo_mjlab
#SBATCH --account=hai_1074
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# REPPO on mjlab Ant-Flat (branch reppo_test): train one config/seed, then run
# the 50-episode deterministic eval in the same job.
#
# Requires the mjlab venv (/p/project1/hai_1075/venvs/mjlab311) and the
# unitree_rl_mjlab workspace next to safe_rl (the trainer resolves it by path).
#
# Usage: sbatch reppo_mjlab.sh CONFIG SEED [NUM_ENVS] [RUN_NAME]
#   e.g. sbatch reppo_mjlab.sh config/mjlab_ant_reppo_v24_support.yaml 2
set -uo pipefail
CONFIG=${1:?config yaml required}
SEED=${2:?seed required}
NUM_ENVS=${3:-1024}
RUN_NAME=${4:-$(basename "$CONFIG" .yaml)_s${SEED}}

module --force purge
module load Stages/2024 GCCcore/.12.3.0 Python/3.11.3

# Booster-native venv — the Cluster-built mjlab311 venv's python symlinks into
# /p/software/juwels, which is not mounted on Booster nodes (jobs died with
# ModuleNotFoundError). Built offline from /p/project1/hai_1075/wheelhouse by
# scripts/slurm/reppo_mjlab_setup.sh.
source /p/project1/hai_1075/venvs/mjlab311_booster/bin/activate

export MUJOCO_GL=egl
export OMP_NUM_THREADS=16
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true
# $HOME is over quota — keep every cache off it.
export MPLCONFIGDIR=/p/scratch/hai_1075/cache
export XDG_CACHE_HOME=/p/scratch/hai_1075/cache
export WANDB_DIR=/p/scratch/hai_1075/safe_rl/wandb

SCR=/p/scratch/hai_1075
LOGROOT=$SCR/safe_rl/reppo_test
mkdir -p $SCR/safe_rl/logs $SCR/cache $SCR/safe_rl/wandb "$LOGROOT"
cd /p/project1/hai_1075/workspaces/safe_rl

echo "=== REPPO config=$CONFIG seed=$SEED num_envs=$NUM_ENVS run=$RUN_NAME host=$(hostname) ==="
python -u scripts/train/unitree_mjlab.py \
    --env_id Ant-Flat \
    --num_envs "$NUM_ENVS" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --logger wandb --wandb_project mjlab \
    --run_name "$RUN_NAME" \
    --log_dir "$LOGROOT"
rc=$?
echo "=== TRAIN DONE rc=$rc ==="
[ $rc -ne 0 ] && exit $rc

CKPT=$(find "$LOGROOT" -name "model_380.pt" -path "*${RUN_NAME}*" | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "=== NO CHECKPOINT for $RUN_NAME under $LOGROOT ==="
    exit 3
fi
echo "=== EVAL ckpt=$CKPT ==="
python -u scripts/eval/unitree_mjlab.py \
    --env_id Ant-Flat \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --num_envs 64 --episodes 50 --headless --device cuda:0
echo "=== EVAL DONE rc=$? ==="

# NOTE: no `wandb sync` here — Booster compute nodes have no internet; sync
# offline runs from a login node afterward (see wandb_sync.sh).
