#!/usr/bin/env bash
# Phase 2, queued behind the running fixed-lambda sweep.
#
#   (a) lambda = 2.0 and 4.0 -- extend the front so it BRACKETS the cost limit of 25.
#       The front so far: lambda 0 -> cost 48.5, 0.4 -> 46.7, 0.78 -> 33.8. Nothing reaches
#       25 yet, and both_lam4 (controller, lambda -> 4) reached 25.55, so the crossing is
#       between 0.78 and 4. Measured lambda_balanced in these runs is 1.86-2.35, NOT the
#       0.78 taken from codex -- that value came from a different arm and does not transfer.
#
#   (b) homotopy vs both_lam4, 3 seeds each. The head-to-head that has never been run.
#       The homotopy is default-off everywhere else, so this is its first real test.
#
# GPU, sequential -- see run_fixlam_sweep.sh for why (31GB box, no swap, shared machine).
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

# Wait for the phase-1 sweep to finish so we never run two trainings at once.
while pgrep -f "[r]un_fixlam_sweep.sh" > /dev/null; do sleep 60; done
echo "=== $(date -Is) phase 1 complete, starting phase 2"

run () {  # run <tag> <config> <seed> <outdir>
  local TAG=$1 CFG=$2 SEED=$3 OUT=$4
  mkdir -p "$OUT"
  local LOG="$OUT/${TAG}.log"
  echo "=== $(date -Is) start $TAG (seed $SEED) -> $LOG"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES="$GPU" $PY -u scripts/train/train_safety_gymnasium.py \
      --env_id SafetyPointGoal1-v0 --num_envs "$ENVS" --config "$CFG" \
      --cost_limits 25.0 --max_iterations "$ITERS" --seed "$SEED" --device cuda \
      --log_dir "$OUT" > "$LOG" 2>&1 &
  CHILD=$!; echo "$CHILD" >> "$OWN_PIDS"; wait "$CHILD"
  echo "=== $(date -Is) done  $TAG rc=$?"
}

for TAG in 2p0 4p0; do
  run "fixlam_${TAG}" "config/safety_gymnasium_cvpo_fixlam_${TAG}.yaml" 1 logs/fixlam_sweep
done

for SEED in 1 2 3; do
  run "homotopy_s${SEED}"  config/safety_gymnasium_cvpo_homotopy.yaml   "$SEED" logs/homotopy_ab
  run "both_lam4_s${SEED}" config/safety_gymnasium_cvpo_both_lam4.yaml  "$SEED" logs/homotopy_ab
done
echo "=== $(date -Is) PHASE 2 COMPLETE"
