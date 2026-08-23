#!/usr/bin/env bash
# ManiSkill pilot for the paper-comparison benchmark (gate G2).
#
# Two full 49,938,432-step cells at seed 1, run SERIALLY on the Ada card:
#   PullCube-v1      paper 1.000 (saturated) -> unambiguous pass/fail on the pipeline
#   PickSingleYCB-v1 paper 0.802 +/- 0.016   -> discriminative, and the task most
#                                              sensitive to the reconfiguring eval env
#
# Writes into the same layout the sweep driver uses, so these cells count as done and
# are not re-run by the full sweep.
set -uo pipefail

# MUST be exported before anything touches CUDA: without it CUDA's device order is
# INVERTED relative to nvidia-smi (nvidia-smi 0 = Blackwell = CUDA 1), and on 2026-08-07
# that inversion killed another session's training on this shared box.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ManiSkill must have the Ada (nvidia-smi index 1): SAPIEN PhysX predates Blackwell
# sm_120 and falls back to CPU physics SILENTLY -- the run would finish and produce
# numbers that are not comparable to anything.
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

REPO=/home/human/workspaces/safe_rl
PY=/home/human/venvs/maniskill/bin/python
OUT="$REPO/experiments/paper_bench/maniskill"
PIDS="$REPO/logs/.own_pids/pilot_maniskill.pids"

cd "$REPO" || exit 1
mkdir -p "$(dirname "$PIDS")" "$OUT"
echo $$ >> "$PIDS"

run_cell() {
  local task="$1" seed="$2"
  local dir="$OUT/$task/s$seed"
  if compgen -G "$dir/*/REPPO/*/model_380.pt" > /dev/null; then
    echo "[pilot] SKIP $task s$seed (already complete)"
    return 0
  fi
  mkdir -p "$dir"
  echo "[pilot] START $task s$seed at $(date -Is)"
  local t0=$SECONDS
  "$PY" -u scripts/train/train_safety_gymnasium.py \
    --env_id "ManiSkill$task" \
    --num_envs 1024 \
    --config "config/bench/paper/maniskill/$task.yaml" \
    --device cuda:0 \
    --seed "$seed" \
    --log_dir "$dir" \
    >> "$dir/train.log" 2>&1 &
  local child=$!
  echo "$child" >> "$PIDS"
  wait "$child"
  local rc=$?
  local dt=$((SECONDS - t0))
  echo "[pilot] END   $task s$seed rc=$rc wall=${dt}s ($((dt / 60))min) at $(date -Is)"
  if [ $rc -ne 0 ]; then
    echo "[pilot] FAILED $task s$seed -- last 40 lines:"
    tail -40 "$dir/train.log"
  fi
  return $rc
}

run_cell PullCube-v1 1
run_cell PickSingleYCB-v1 1
echo "[pilot] all cells finished at $(date -Is)"
