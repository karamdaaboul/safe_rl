#!/usr/bin/env bash
# Overnight queue 2026-08-20: c3 (spread-match + CVaR constraint) x 3 seeds, 20k iters each,
# sequential (~3.5h per run). Seed-level evidence for the tail-constraint claim in one night.
set -u
cd "$(dirname "$0")/.."
PY=/home/human/venvs/agx_plain/bin/python
for seed in 2 3 4; do
    echo "$(date +%T) === starting c3 seed ${seed} ==="
    CUDA_VISIBLE_DEVICES=1 $PY scripts/train/train_safety_gymnasium.py \
        --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
        --config config/safety_gymnasium_fhdcmpo_c3_spreadmatch_cvar.yaml \
        --cost_limits 25.0 --max_iterations 20000 --seed ${seed} --deterministic
    echo "$(date +%T) === c3 seed ${seed} exited with code $? ==="
done
echo "$(date +%T) overnight queue done."
