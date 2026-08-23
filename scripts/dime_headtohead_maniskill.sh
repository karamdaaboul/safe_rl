#!/bin/bash
# Head-to-head on ManiSkill PickCube-v1: the TruDi authors' REPPO-DIME vs ours.
#
# SEQUENTIAL, not concurrent. Another session is running 7 CVPO jobs holding
# ~19GB of the box's 31GB, and the reference trainer has already been
# OOM-killed twice (rc=137) when made to share RAM with another ManiSkill run.
#
# Both arms log to wandb project `dime_benchmark_rerun` with identical metric
# keys (ours via trudi_wandb_schema: true) so the curves overlay. They are told
# apart by tag: 'reference' vs 'ours'.
#
# GPU: the reference venv pins torch 2.7.1+cu126, which cannot drive the
# Blackwell (sm_120 needs cu128+), so it MUST take the Ada. In torch's device
# ordering on this box that is CUDA_VISIBLE_DEVICES=0.
set -u

REF_DIR=/home/human/workspaces/trudi_ref/trudi
REF_PY=/home/human/venvs/trudi_ref/bin/python
OUR_DIR=/home/human/workspaces/safe_rl
OUR_PY=/home/human/venvs/maniskill/bin/python

echo "[h2h] === ARM 1/2: reference TruDi REPPO-DIME ==="
cd "$REF_DIR" || exit 1
CUDA_VISIBLE_DEVICES=0 $REF_PY -m src.torchrl.reppo_dime \
  --config-name=reppo_dime_maniskill \
  env.name=PickCube-v1 \
  tags="[experimental,reference]" \
  > "$OUR_DIR/logs/h2h_reference_dime.log" 2>&1
echo "[h2h] reference exited rc=$? at $(date)"

echo "[h2h] === ARM 2/2: our REPPO-DIME ==="
cd "$OUR_DIR" || exit 1
CUDA_VISIBLE_DEVICES=0 $OUR_PY scripts/train/train_safety_gymnasium.py \
  --env_id ManiSkillPickCube-v1 \
  --num_envs 1024 \
  --device cuda:0 \
  --config config/maniskill_pickcube_reppodime.yaml \
  --log_dir logs/h2h_our_dime \
  --seed 1 \
  > "$OUR_DIR/logs/h2h_our_dime.log" 2>&1
echo "[h2h] ours exited rc=$? at $(date)"
