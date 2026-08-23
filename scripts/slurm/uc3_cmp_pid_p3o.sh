#!/bin/bash
# PPOL-PID vs P3O, 3 seeds each, one node, one job.
# All runs: SafetyPointGoal1-v0, cost_limit 25, 64 envs x 512 steps (= 32768
# batch), matching run 6345948 which serves as PPOL-PID seed 1. 500 iters both.
#
# Submit with:  sbatch -p gpu_h100,gpu_h100_il scripts/slurm/uc3_cmp_pid_p3o.sh
# -c 128 keeps the job eligible for both node types (128 logical = 64 physical
# on il). 5 runs x 64 envs = 320 env processes: oversubscribed, expect ~2.5-3h.
#SBATCH --job-name=cmp_pid_p3o
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl/logs/%x-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl/logs/%x-%j.err

set -uo pipefail

WS=/pfs/work9/workspace/scratch/ka_jg5338-safe_rl
CMP=$WS/logs/cmp3
mkdir -p "$CMP"

module purge
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source "$WS/venvs/safe_rl311/bin/activate"

export MUJOCO_GL=egl
export GIT_PYTHON_REFRESH=quiet
# 6 concurrent runs: without this every process grabs all cores for BLAS.
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR="$WS/safe_rl"
export WANDB_RUN_GROUP=cmp_pid_p3o_3seed

cd "$WS/safe_rl"

echo "=== job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

declare -a PIDS NAMES
i=0
# ppol_pid seed 1 already ran as job 6345948 (64x512) and joins the comparison.
for spec in "ppol_pid 2" "ppol_pid 3" "p3o 1" "p3o 2" "p3o 3"; do
    alg=${spec% *}; seed=${spec#* }
    name="${alg}_s${seed}"
    CUDA_VISIBLE_DEVICES=$((i % 2)) \
    python scripts/train/train_safety_gymnasium.py \
      --env_id SafetyPointGoal1-v0 \
      --num_envs 64 \
      --num_steps_per_env 512 \
      --config "config/safety_gymnasium_${alg}.yaml" \
      --cost_limits 25.0 \
      --max_iterations 500 \
      --seed "$seed" \
      --device cuda \
      --log_dir "$CMP/$name" \
      > "$CMP/$name.log" 2>&1 &
    PIDS+=($!); NAMES+=("$name")
    echo "launched $name pid=${PIDS[-1]} gpu=$((i % 2))"
    i=$((i + 1))
    sleep 5   # keep the %Y%m%d_%H%M%S run dirs distinct
done

FAIL=0
for j in "${!PIDS[@]}"; do
  if wait "${PIDS[$j]}"; then
    echo "OK   ${NAMES[$j]}"
  else
    echo "FAIL ${NAMES[$j]} (see $CMP/${NAMES[$j]}.log)"
    FAIL=1
  fi
done

echo "=== all runs finished: $(date)"
echo "=== final numbers (last logged iteration per run):"
for n in "${NAMES[@]}"; do
  echo "--- $n"
  grep -E "Mean episode reward|Mean episode cost|Constraint 0 λ" "$CMP/$n.log" | tail -3
done
exit $FAIL
