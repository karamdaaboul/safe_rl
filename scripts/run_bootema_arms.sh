#!/usr/bin/env bash
# Bootstrap-stability experiment: EMA bootstrap policy for the FH-DCMPO cost target.
# Runs the three tau_b arms SEQUENTIALLY (this box cannot hold two trainings), after the
# baseline a2 run (PID passed as $1, if still alive) has finished. Baseline for comparison:
# safety_gymnasium_fhdcmpo_a2_lag_cc4_60k, run 20260819_080249, seed 2.
#
# Usage: nohup bash scripts/run_bootema_arms.sh [WAIT_PID] > logs/bootema_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/home/human/venvs/agx_plain/bin/python

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "$(date +%T) waiting for baseline run (pid $WAIT_PID) to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "$(date +%T) baseline finished."
fi

for tag in b1 b2 b3; do
    cfg="config/safety_gymnasium_fhdcmpo_${tag}_bootema.yaml"
    echo "$(date +%T) === starting ${tag} (${cfg}) ==="
    $PY scripts/train/train_safety_gymnasium.py \
        --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
        --config "$cfg" --cost_limits 25.0 --max_iterations 60000 --seed 2 --deterministic
    echo "$(date +%T) === ${tag} exited with code $? ==="
done
echo "$(date +%T) all bootema arms done."
