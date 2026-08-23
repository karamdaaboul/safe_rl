#!/usr/bin/env bash
# Run one FH-DCMPO arm: train N seeds sequentially, then a 50-episode deterministic eval each.
#
# Sequential on purpose. This box has 31GB of RAM and two concurrent Safety-Gymnasium trainings
# have OOM'd both before, so arms chain rather than fan out. One 60k-iteration run is ~2.3h
# (measured: ~26k iterations/hour), so a 3-seed arm is ~7h and a 3-seed x 2-task arm ~14h.
#
# Usage:
#   scripts/run_fhdcmpo_arm.sh <config> <env_id> <tag> [iters] [seeds...]
#
# Examples:
#   # S0 gate: does the whole path run at all?
#   scripts/run_fhdcmpo_arm.sh config/safety_gymnasium_fhdcmpo_smoke.yaml SafetyPointGoal1-v0 smoke 2000 1
#   # S1 gate: finite horizon alone, 1 seed screen
#   scripts/run_fhdcmpo_arm.sh config/safety_gymnasium_fhdcmpo_s1_mean_goal1.yaml SafetyPointGoal1-v0 s1 20000 2
#   # S4 confirmation: 3 seeds
#   scripts/run_fhdcmpo_arm.sh config/safety_gymnasium_fhdcmpo_goal1.yaml SafetyPointGoal1-v0 s4p1 60000 2 3 4
set -euo pipefail

CONFIG=${1:?usage: run_fhdcmpo_arm.sh <config> <env_id> <tag> [iters] [seeds...]}
ENV_ID=${2:?missing env_id}
TAG=${3:?missing tag}
ITERS=${4:-60000}
shift 4 2>/dev/null || shift $#
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(2)

PY=/home/human/venvs/agx_plain/bin/python
# 8 envs matches the qrdmpo arms these results are compared against. It also keeps the CPU
# footprint modest, which matters because another session's training may be sharing the box's
# 16 cores (measured contention: ~219 steps/s vs ~1080 when alone).
NUM_ENVS=8
EPISODES=50
# The Blackwell 32GB card. nvidia-smi and CUDA enumerate these in opposite orders, so this is
# pinned by index under CUDA's ordering and the run log prints the resolved device name to confirm.
# Respect an outer setting so a second concurrent arm can take the other card; the box
# holds two trainings, and putting both on one GPU makes them contend for no reason.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p logs/fhdcmpo

for SEED in "${SEEDS[@]}"; do
  STAMP=$(date +%Y%m%d_%H%M%S)
  RUN="${TAG}_$(basename "$ENV_ID" | sed 's/-v0//')_s${SEED}"
  LOG="logs/fhdcmpo/${RUN}_${STAMP}.log"
  echo "=== [$(date +%H:%M:%S)] TRAIN $RUN  ($ITERS iters, config $CONFIG) -> $LOG"

  $PY -c "import torch; print('[device]', torch.cuda.get_device_name(0))" | tee "$LOG"

  $PY scripts/train/train_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs "$NUM_ENVS" \
    --device cuda:0 \
    --config "$CONFIG" \
    --cost_limits 25.0 \
    --max_iterations "$ITERS" \
    --seed "$SEED" \
    --deterministic >>"$LOG" 2>&1

  # Newest run dir for this env + algorithm.
  RUN_DIR=$(ls -1dt logs/safety_gymnasium/"$ENV_ID"/FHDCMPO/*/ 2>/dev/null | head -1)
  if [ -z "$RUN_DIR" ]; then echo "!! no run dir for $RUN; see $LOG" >&2; continue; fi
  CKPT=$(ls -1v "$RUN_DIR"/model_*.pt 2>/dev/null | tail -1)
  if [ -z "$CKPT" ]; then echo "!! no checkpoint in $RUN_DIR; see $LOG" >&2; continue; fi

  CSV="logs/deteval_fhdcmpo_${RUN}.csv"
  echo "=== [$(date +%H:%M:%S)] EVAL  $RUN  ($EPISODES episodes) ckpt=$CKPT"
  $PY scripts/eval/eval_safety_gymnasium.py \
    --env_id "$ENV_ID" \
    --num_envs 1 \
    --episodes "$EPISODES" \
    --device cuda:0 \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --cost_limits 25.0 \
    --seed 12345 \
    --eval_csv "$CSV" 2>&1 | tee -a "$LOG"
  echo "=== [$(date +%H:%M:%S)] DONE  $RUN -> $CSV"
done

echo "=== arm '$TAG' complete: ${#SEEDS[@]} seed(s)"
