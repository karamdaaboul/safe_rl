#!/bin/bash
# Queued after the PickCube 20M run: REPPO-Gaussian on Mjlab-Lift-Cube-Yam-Grasp,
# then a video from the final checkpoint.
#
# Same config/env/reward/env-count as the 2026-07-31 grasp run that recorded
# Episode_Reward/grasp == 0.0000 across all 300 iterations, so the numbers stay
# comparable. Only the log dir and run name differ.
set -u
cd /home/human/workspaces/safe_rl

PICKCUBE_PID=${1:?usage: queued_liftcube_grasp.sh <pid-to-wait-for>}
echo "[queue] waiting for PickCube PID $PICKCUBE_PID to finish..."
while kill -0 "$PICKCUBE_PID" 2>/dev/null; do sleep 30; done
echo "[queue] PickCube done at $(date). Starting Lift-Cube grasp run."

PY=/home/human/venvs/agx_plain/bin/python
RUN=logs/mjlab_grasp_queued
# mjlab initializes CUDA at import, so --device/--gpu_ids are inert; the GPU must
# be chosen via CUDA_VISIBLE_DEVICES. 1 -> the 32GB Blackwell in torch's ordering.
export CUDA_VISIBLE_DEVICES=1

$PY scripts/train/unitree_mjlab.py \
  --env_id Mjlab-Lift-Cube-Yam-Grasp \
  --num_envs 1024 \
  --config config/mjlab_liftcube_reppo_v25.yaml \
  --experiment_name mjlab_liftcube_grasp_reppo_queued \
  --max_iterations 300 \
  --seed 1 \
  --log_dir "$RUN" \
  --logger wandb --wandb_project mjlab \
  > logs/mjlab_grasp_queued.log 2>&1
echo "[queue] training exited rc=$? at $(date)"

CKPT=$(ls -t "$RUN"/*/*/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[queue] no checkpoint found under $RUN — skipping video"
  exit 1
fi
echo "[queue] recording video from $CKPT"
mkdir -p videos/liftcube_grasp
$PY scripts/eval/unitree_mjlab.py \
  --env_id Mjlab-Lift-Cube-Yam-Grasp \
  --checkpoint "$CKPT" \
  --num_envs 1 --episodes 8 --device cuda:0 \
  --video --video_length 800 --video_dir videos/liftcube_grasp \
  >> logs/mjlab_grasp_queued.log 2>&1
echo "[queue] video step exited rc=$? at $(date)"
