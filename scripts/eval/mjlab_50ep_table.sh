#!/bin/bash
# 50-episode deterministic mjlab evaluation, ours vs the TruDi reference.
#
# OUR arms run in agx_plain on the Blackwell (CUDA_VISIBLE_DEVICES=1).
# THEIR arms run in trudi_ref, whose pinned torch 2.7.1+cu126 cannot drive the
# Blackwell (sm_120 needs cu128+), so they take the Ada (=0) and must wait for the
# PickCube reference run to release it.
#
# eval_their_ckpt.py exists precisely for this: it loads the authors' Actor +
# EmpiricalNormalization from their checkpoint and applies OUR eval protocol, so
# both sides are measured the same way.
set -u
cd /home/human/workspaces/safe_rl
PY=/home/human/venvs/agx_plain/bin/python
export MUJOCO_GL=egl

run_ours () {  # task ckpt config label
  echo "----- OURS  $4  ($1) -----"
  CUDA_VISIBLE_DEVICES=1 timeout 1800 $PY scripts/eval/unitree_mjlab.py \
    --env_id "$1" --checkpoint "$2" --config "$3" \
    --num_envs 64 --episodes 50 --device cuda:0 --headless 2>&1 \
    | grep -E "Mean reward|Mean length|Survival rate|Fall rate|Mean Metrics" || echo "  FAILED"
}

echo "########## Humanoid-Flat ##########"
run_ours Humanoid-Flat \
  logs/dime_humanoid/mjlab_humanoid_reppodime/2026-08-06_14-52-09_dime_humanoid_v0/model_380.pt \
  config/mjlab_humanoid_reppodime.yaml "REPPO-DIME"
run_ours Humanoid-Flat \
  logs/safe_rl/humanoid/2026-07-31_12-51-07_humanoid_support_refit/model_380.pt \
  config/mjlab_humanoid_reppo_v24.yaml "REPPO-Gaussian"

echo "########## Unitree-Go2-Flat ##########"
run_ours Unitree-Go2-Flat \
  logs/safe_rl/go2_v00/2026-08-03_12-31-06_reppo_v00_s1/model_299.pt \
  config/mjlab_go2_reppo_v00_baseline.yaml "REPPO-Gaussian s1"

echo "########## waiting for the Ada (PickCube reference) ##########"
while pgrep -f "[t]orchrl.reppo_dime" >/dev/null; do sleep 60; done
echo "Ada free at $(date)"

echo "########## THEIR checkpoints (reference code, our protocol) ##########"
cd /home/human/workspaces/trudi_ref
for spec in \
  "Humanoid-Flat trudi_humanoid_ckpt/reppo_zip_humanoid_torch_Humanoid-Flat_latest.pt" \
  "Unitree-Go2-Flat trudi_go2_ckpt/reppo_zip_go2_torch_Unitree-Go2-Flat_latest.pt" ; do
  set -- $spec
  echo "----- REFERENCE  $1 -----"
  CUDA_VISIBLE_DEVICES=0 timeout 1800 /home/human/venvs/trudi_ref/bin/python eval_their_ckpt.py \
    --task "$1" --ckpt "$2" --num-envs 64 --episodes 50 --device cuda:0 2>&1 \
    | grep -E "Mean reward|mean reward|Mean length|episodes|Survival|Fall|error|Error|Traceback" | tail -8 \
    || echo "  FAILED"
done
echo "########## DONE $(date) ##########"
