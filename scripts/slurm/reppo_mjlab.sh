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
# Usage: sbatch reppo_mjlab.sh CONFIG SEED [NUM_ENVS] [RUN_NAME] [ENV_ID] [MAX_ITERS]
#   CONFIG=none runs the task's registered PPO config (safe_rl PPO class) instead
#   of a YAML — the baseline arm of algorithm comparisons.
#   e.g. sbatch reppo_mjlab.sh config/mjlab_go2_reppo_v24.yaml 1 4096 go2_reppo Unitree-Go2-Flat 381
set -uo pipefail
CONFIG=${1:?config yaml or 'none' required}
SEED=${2:?seed required}
NUM_ENVS=${3:-1024}
RUN_NAME=${4:-$(basename "$CONFIG" .yaml)_s${SEED}}
ENV_ID=${5:-Ant-Flat}
MAX_ITERS=${6:-}

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

echo "=== config=$CONFIG env=$ENV_ID seed=$SEED num_envs=$NUM_ENVS iters=${MAX_ITERS:-cfg} run=$RUN_NAME host=$(hostname) ==="
EXTRA=()
[ "$CONFIG" != "none" ] && EXTRA+=(--config "$CONFIG")
[ -n "$MAX_ITERS" ] && EXTRA+=(--max_iterations "$MAX_ITERS")
python -u scripts/train/unitree_mjlab.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --logger wandb --wandb_project mjlab \
    --run_name "$RUN_NAME" \
    --log_dir "$LOGROOT" \
    "${EXTRA[@]}"
rc=$?
echo "=== TRAIN DONE rc=$rc ==="
[ $rc -ne 0 ] && exit $rc

CKPT=$(find "$LOGROOT" -path "*${RUN_NAME}*" -name "model_*.pt" ! -name "model_0.pt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2)
if [ -z "$CKPT" ]; then
    echo "=== NO CHECKPOINT for $RUN_NAME under $LOGROOT ==="
    exit 3
fi
echo "=== EVAL ckpt=$CKPT ==="
EVAL_EXTRA=()
[ "$CONFIG" != "none" ] && EVAL_EXTRA+=(--config "$CONFIG")
python -u scripts/eval/unitree_mjlab.py \
    --env_id "$ENV_ID" \
    --checkpoint "$CKPT" \
    --num_envs 64 --episodes 50 --headless --device cuda:0 \
    "${EVAL_EXTRA[@]}" 
echo "=== EVAL DONE rc=$? ==="

# NOTE: no `wandb sync` here — Booster compute nodes have no internet; sync
# offline runs from a login node afterward (see wandb_sync.sh).
