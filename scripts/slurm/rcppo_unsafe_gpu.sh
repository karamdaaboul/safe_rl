#!/usr/bin/env bash
# Finish the unconstrained (lambda=0) velocity baselines for Hopper + Swimmer on GPU.
# GPU offloads the 40-epoch learning phase (~20s CPU -> ~2s), which on this 16-core
# box was the main slowdown; collection stays on CPU. Sequential (GPU-parallel gives
# no gain since collection is CPU-bound). Same rcppo_diag_G_scaledAdv.yaml + --lambda_max 0.
set -uo pipefail
cd /home/human/workspaces/safe_rl
PY=/home/human/venvs/agx_plain/bin/python
CFG=config/rcppo_diag_G_scaledAdv.yaml

run() {
  local tag=$1 env_id=$2
  echo "===== UNSAFE-GPU baseline: $tag ($env_id) ====="
  "$PY" scripts/train/train_safety_gymnasium.py \
    --env_id "$env_id" --num_envs 16 --config "$CFG" \
    --max_iterations 400 --lambda_max 0 --device cuda --seed 1 \
    --log_dir "logs/rcppo_vel/UNSAFE_${tag}" 2>&1 | tee "logs/rcppo_vel/UNSAFE_${tag}.log"
}

run hopper  SafetyHopperVelocity-v1
run swimmer SafetySwimmerVelocity-v1
echo "===== HOPPER+SWIMMER UNSAFE-GPU DONE ====="
