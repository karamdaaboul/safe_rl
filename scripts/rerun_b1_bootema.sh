#!/usr/bin/env bash
# b1 was OOM-killed at ~iter 2.2k before its first checkpoint (kernel OOM 2026-08-19 22:33,
# a concurrent analysis process pushed the box over). Re-run it fresh AFTER the main queue
# (b2, b3) finishes, so at most one training ever runs on this box.
set -u
cd "$(dirname "$0")/.."
QUEUE_PID="${1:?pass the run_bootema_arms.sh pid}"
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 120; done
echo "$(date +%T) main queue done; rerunning b1"
/home/human/venvs/agx_plain/bin/python scripts/train/train_safety_gymnasium.py \
    --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
    --config config/safety_gymnasium_fhdcmpo_b1_bootema.yaml \
    --cost_limits 25.0 --max_iterations 60000 --seed 2 --deterministic
echo "$(date +%T) b1 rerun exited with code $?"
