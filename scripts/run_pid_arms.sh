#!/usr/bin/env bash
# CVPO + PID-Lagrangian on realized episodic cost, 3 seeds.
# Falsification target (codex/cvpo-feasible-threshold-homotopy.md): lambda should integrate to
# ~2 and cost -> 25 WITHOUT ever reading Q_c's level. If lambda still random-walks near 0, the
# leak is elsewhere -- most likely the realized-cost estimate itself (the runner reports
# mean(costbuffer)=0.0 before any episode completes; those reports are discarded, and
# lambda_episodic_reports is logged so a starved controller is distinguishable from a stuck one).
set -u
# PID BOOKKEEPING -- never stop runs by matching the script name.
# This box is shared and other sessions launch the SAME entrypoint: e.g. ManiSkill runs are
# `train_safety_gymnasium.py --env_id ManiSkillPickCube-v1`. A `pkill -f train_safety_gymnasium`
# therefore kills THEIR training too. Record our own child PIDs and stop only those:
#     kill $(cat logs/.own_pids/<driver>.pids)
OWN_PIDS="logs/.own_pids/${0##*/}.pids"
mkdir -p "$(dirname "$OWN_PIDS")"; : > "$OWN_PIDS"
echo $$ >> "$OWN_PIDS"
# GPU SELECTION -- read this before changing it.
# Without CUDA_DEVICE_ORDER there are TWO index spaces for the same cards:
#     nvidia-smi 0 = Blackwell        CUDA_VISIBLE_DEVICES=0 = Ada
#     nvidia-smi 1 = Ada              CUDA_VISIBLE_DEVICES=1 = Blackwell
# On 2026-08-07 that inversion put a run onto a card another session was already using and
# killed its training: nvidia-smi reported index 0 free (the Blackwell) and the launcher
# passed CUDA_VISIBLE_DEVICES=0, which is the Ada. PCI_BUS_ID forces CUDA's order to match
# nvidia-smi's, so GPU=<n> below means exactly what nvidia-smi calls <n>.
# This box is shared. ManiSkill can only run on the Ada (SAPIEN PhysX predates Blackwell
# sm_120), so the Ada is the card most likely to be occupied -- default to the Blackwell.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/human/venvs/agx_plain/bin/python
GPU=${GPU:-0}; ITERS=${ITERS:-30000}; ENVS=${ENVS:-8}; OUT=logs/pid_arms

# Refuse to start if the target card is not idle (checked immediately before launch).
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c        "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")" || true)
if [ "$BUSY" -ne 0 ]; then
  echo "REFUSING: GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)) is in use." >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi
echo "GPU $GPU = $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU") -- verified idle" 
mkdir -p "$OUT"
for SEED in 1 2 3; do
  echo "=== $(date -Is) start pid_s${SEED}"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES="$GPU" $PY -u scripts/train/train_safety_gymnasium.py \
      --env_id SafetyPointGoal1-v0 --num_envs "$ENVS" --config config/safety_gymnasium_cvpo_pid.yaml \
      --cost_limits 25.0 --max_iterations "$ITERS" --seed "$SEED" --device cuda \
      --log_dir "$OUT" > "$OUT/pid_s${SEED}.log" 2>&1 &
  CHILD=$!; echo "$CHILD" >> "$OWN_PIDS"; wait "$CHILD"
  echo "=== $(date -Is) done  pid_s${SEED} rc=$?"
done
echo "=== $(date -Is) PID ARMS COMPLETE"
