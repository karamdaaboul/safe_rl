#!/bin/bash
# Arm 1 of the PickCube head-to-head, retried. Waits for our arm to finish first —
# the reference venv pins torch 2.7.1+cu126 which cannot drive the Blackwell
# (sm_120 needs cu128+), so both arms must share the Ada, and the box only has
# ~11GB RAM free (another session holds ~19GB with 7 CVPO jobs). The reference
# has already been OOM-killed twice when made to share RAM with a ManiSkill run.
#
# BUDGET: 10M env steps per arm, not the config's 50M. The authors' own published
# PickCube result is eval/success_ode_100 = 0.498 at 7.4M, so 10M is past the
# point where the reference has already demonstrably solved the task — enough to
# decide whether the two implementations agree, at a fifth of the wall-clock.
# Our arm is matched at 77 iterations (77 * 1024 * 128 = 10,092,544 steps).
#
# THE FIX: config/reppo_dime_maniskill.yaml has no `reward_surface` key, and
# reppo_dime.py:956 does `if not hasattr(cfg,'reward_surface'): cfg.reward_surface = ...`
# which omegaconf's struct mode rejects (you cannot add keys to a struct config).
# Supplying it as a hydra `+` override makes hasattr() true so the in-code
# assignment is skipped. This changes NO reference source or config file, which
# matters: the whole point of this arm is that it is the authors' code unmodified.
set -u

# Wait on a PROCESS PATTERN, not a PID. Under `setsid ... & $!` the captured pid
# is setsid's, and setsid exits the moment it forks the real child — so a
# `kill -0 $!` wait returns instantly, launches this arm alongside the other one,
# and the reference gets OOM-killed (rc=137). That is exactly what happened on the
# first attempt. The bracket in "[h]2h_our_dime" stops pgrep matching itself.
echo "[ref] waiting for our arm to finish (matching h2h_our_dime)..."
sleep 20  # let the other arm appear in the process table before we test for it
while pgrep -f "[h]2h_our_dime" >/dev/null; do sleep 60; done
echo "[ref] our arm done at $(date). Starting reference."

cd /home/human/workspaces/trudi_ref/trudi || exit 1
CUDA_VISIBLE_DEVICES=0 /home/human/venvs/trudi_ref/bin/python -m src.torchrl.reppo_dime \
  --config-name=reppo_dime_maniskill \
  env.name=PickCube-v1 \
  tags="[experimental,reference]" \
  hyperparameters.total_time_steps=10000000 \
  "+reward_surface={enabled:false,interval:10,grid_size:11,save_dir:./reward_surfaces,log_wandb:true,plot_type:matplotlib}" \
  > /home/human/workspaces/safe_rl/logs/h2h_reference_dime.log 2>&1
echo "[ref] reference exited rc=$? at $(date)"
