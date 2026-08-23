#!/usr/bin/env bash
# Fixed-lambda front-tracing sweep for CVPO on SafetyPointGoal1-v0.
#
# Runs SEQUENTIALLY on purpose: 31GB box, no swap, and another session's trainings are
# resident. codex records CVPO OOMing when two run concurrently on an *empty* box.
#
# lambda is pinned (lambda_lr = 0, lambda_init = L), so the E-step exponent is exactly
# (Q_r - L*Q_c)/eta with no controller, no threshold and no homotopy in the loop. This traces
# the reward/cost front directly and bounds what any controller on top could achieve.
# Reference: median_s std_a(Q_r)/std_a(Q_c) = 0.78, so L = 0.78 balances the two terms.
#
# RUN ON GPU. Measured on this box: 12.86 it/s on the RTX 4000 Ada vs 0.281 it/s on CPU --
# a 45.8x speedup, i.e. 0.6 h/arm instead of 29.7 h/arm. The CPU default silently turns this
# sweep into a 6-day job. CUDA_VISIBLE_DEVICES uses torch's enumeration, NOT nvidia-smi's:
# torch cuda:0 = RTX 4000 Ada, torch cuda:1 = RTX PRO 4500 Blackwell (the indices are
# inverted relative to nvidia-smi). Override GPU= to move arms off a busy card.
set -u
# PID BOOKKEEPING -- never stop runs by matching the script name.
# This box is shared and other sessions launch the SAME entrypoint: e.g. ManiSkill runs are
# `train_safety_gymnasium.py --env_id ManiSkillPickCube-v1`. A `pkill -f train_safety_gymnasium`
# therefore kills THEIR training too. Record our own child PIDs and stop only those:
#     kill $(cat logs/.own_pids/<driver>.pids)
OWN_PIDS="logs/.own_pids/${0##*/}.pids"
mkdir -p "$(dirname "$OWN_PIDS")"; : > "$OWN_PIDS"
echo $$ >> "$OWN_PIDS"
PY=/home/human/venvs/agx_plain/bin/python
GPU=${GPU:-0}
ITERS=${ITERS:-30000}
ENVS=${ENVS:-8}
OUT=${OUT:-logs/fixlam_sweep}
mkdir -p "$OUT"
for TAG in 0p0 0p4 0p78 1p5 3p0; do
  CFG="config/safety_gymnasium_cvpo_fixlam_${TAG}.yaml"
  LOG="$OUT/fixlam_${TAG}.log"
  echo "=== $(date -Is) starting lambda=$TAG -> $LOG"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES="$GPU" $PY -u scripts/train/train_safety_gymnasium.py \
      --env_id SafetyPointGoal1-v0 --num_envs "$ENVS" --config "$CFG" \
      --cost_limits 25.0 --max_iterations "$ITERS" --seed 1 --device cuda \
      --log_dir "$OUT" > "$LOG" 2>&1 &
  CHILD=$!; echo "$CHILD" >> "$OWN_PIDS"; wait "$CHILD"
  echo "=== $(date -Is) finished lambda=$TAG rc=$?"
done
echo "=== $(date -Is) SWEEP COMPLETE"
