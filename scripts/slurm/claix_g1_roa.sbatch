#!/bin/bash
# G1 velocity training (PPO or PPO+ROA) through safe_rl on one CLAIX H100, one seed per job.
#   sbatch --export=ALL,CODE=<safe_rl snapshot>,UNITREE=<unitree_rl_mjlab snapshot>,\
#     CONFIG=config/unitree_g1_flat_ppo_roa.yaml,TASK=Unitree-G1-Flat-ROA,SEED=1,NAME=g1_flat_ppo_roa \
#     claix_g1_roa.sbatch
# Both snapshots are `git archive` exports with a COMMIT file. SMOKE=1 runs 30 iterations at 512
# envs, so a short job can gate the real ones with --dependency=afterok.
#SBATCH --job-name=g1_roa
#SBATCH --partition=c23g
#SBATCH --account=rwth2182
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpcwork/qzi10910/roa/slurm/%x-%j.out
#SBATCH --error=/hpcwork/qzi10910/roa/slurm/%x-%j.out
set -euo pipefail
module load Python/3.12.3 >/dev/null 2>&1
: "${CODE:?CODE=<safe_rl dir>}" "${UNITREE:?UNITREE=<unitree_rl_mjlab dir>}" "${CONFIG:?}" "${TASK:?}" "${SEED:?}" "${NAME:?}"
R=/hpcwork/qzi10910/roa
PY=$R/venv/bin/python
# mjlab must import from the snapshot, and W&B has no guaranteed route off the compute nodes:
# runs are synced from a login node afterwards (wandb sync $R/wandb/wandb/offline-run-*).
export PYTHONPATH=$CODE UNITREE_RL_MJLAB_PATH=$UNITREE PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$R/wandb WARP_CACHE_PATH=$R/warp_cache
ITERS=${ITERS:-2000}
NUM_ENVS=${NUM_ENVS:-4096}
if [[ "${SMOKE:-0}" == 1 ]]; then ITERS=30; NUM_ENVS=512; fi
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME safe_rl=$(cat "$CODE/COMMIT") unitree=$(cat "$UNITREE/COMMIT")"
echo "task=$TASK config=$CONFIG seed=$SEED iters=$ITERS num_envs=$NUM_ENVS name=$NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$R/logs" "$R/wandb" "$R/warp_cache"
cd "$CODE"
exec $PY -u scripts/train/unitree_mjlab.py \
  "$TASK" --config "$CONFIG" --seed "$SEED" \
  --max_iterations "$ITERS" --num_envs "$NUM_ENVS" \
  --log_dir "$R/logs" --experiment_name "$NAME" --run_name "seed$SEED"
