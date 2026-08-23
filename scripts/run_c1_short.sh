#!/usr/bin/env bash
# c1 spread-match, capped at 20k iterations. Queued behind whatever training currently
# holds the box (another session's vtmpo run); one training at a time, never preempt.
set -u
cd "$(dirname "$0")/.."
WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi
echo "$(date +%T) box free; starting c1 (20k iters)"
/home/human/venvs/agx_plain/bin/python scripts/train/train_safety_gymnasium.py \
    --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
    --config config/safety_gymnasium_fhdcmpo_c1_spreadmatch.yaml \
    --cost_limits 25.0 --max_iterations 20000 --seed 2 --deterministic
echo "$(date +%T) c1 exited with code $?"
