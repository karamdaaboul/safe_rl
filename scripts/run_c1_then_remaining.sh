#!/usr/bin/env bash
# Re-ordered queue (2026-08-20): after b2 finishes -> c1 (spread match, the prioritized arm)
# -> b3 (strong EMA) -> b1 rerun (mild EMA). One training at a time, always.
set -u
cd "$(dirname "$0")/.."
PY=/home/human/venvs/agx_plain/bin/python
B2_PID="${1:?pass the b2 python pid}"
while kill -0 "$B2_PID" 2>/dev/null; do sleep 120; done
echo "$(date +%T) b2 finished."
for cfg in safety_gymnasium_fhdcmpo_c1_spreadmatch safety_gymnasium_fhdcmpo_b3_bootema safety_gymnasium_fhdcmpo_b1_bootema; do
    echo "$(date +%T) === starting ${cfg} ==="
    $PY scripts/train/train_safety_gymnasium.py \
        --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
        --config "config/${cfg}.yaml" --cost_limits 25.0 --max_iterations 60000 --seed 2 --deterministic
    echo "$(date +%T) === ${cfg} exited with code $? ==="
done
echo "$(date +%T) all queued arms done."
