#!/usr/bin/env bash
# Unconstrained (lambda frozen at 0) baselines for the G-suite velocity envs, to
# quantify the RCPPO safety tax per task (Walker2d/Ant/Hopper/Swimmer). Same
# rcppo_diag_G_scaledAdv.yaml verbatim as the constrained G runs; only --lambda_max 0
# differs (PID output is clamped to 0 every iteration -> pure-reward surrogate, but
# the reach/cost critics still train, matching the report's "unconstrained + trained Q_h").
set -euo pipefail
cd /home/human/workspaces/safe_rl

PY=/home/human/venvs/agx_plain/bin/python
CFG=config/rcppo_diag_G_scaledAdv.yaml

declare -A ENVS=(
  [walker]=SafetyWalker2dVelocity-v1
  [ant]=SafetyAntVelocity-v1
  [hopper]=SafetyHopperVelocity-v1
  [swimmer]=SafetySwimmerVelocity-v1
)

for tag in walker ant hopper swimmer; do
  env_id=${ENVS[$tag]}
  echo "===== UNSAFE baseline: $tag ($env_id) ====="
  "$PY" scripts/train/train_safety_gymnasium.py \
    --env_id "$env_id" \
    --num_envs 16 \
    --config "$CFG" \
    --max_iterations 400 \
    --lambda_max 0 \
    --device cpu \
    --seed 1 \
    --log_dir "logs/rcppo_vel/UNSAFE_${tag}" \
    2>&1 | tee "logs/rcppo_vel/UNSAFE_${tag}.log"
done
echo "===== ALL UNSAFE BASELINES DONE ====="
