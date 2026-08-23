#!/usr/bin/env bash
# c4 (spread match + CVaR ramp + reference-median normalizer) x 3 seeds, 20k iters each,
# SEQUENTIAL -- this box cannot hold two trainings (see the M0 OOM note). ~3.5h per run.
#
# c4 is the dose-corrected retry of c3: same kappa ramp to 1.0, but the per-state match factor is
# normalized by the MEAN-readout median, so the tail readout's 2-10x spread inflation no longer
# multiplies the constraint pressure. Matched to c3's protocol (same seeds, same 20k cap, same
# eval cadence) so the two are directly comparable; judged against c2 by the pre-registered
# winner rule in the config header.
#
# Usage: nohup bash scripts/run_c4_arms.sh [WAIT_PID] > logs/c4_queue.log 2>&1 &
#   WAIT_PID: optional pid of a training already running; c4 starts only once it exits.
set -u
cd "$(dirname "$0")/.."
PY=/home/human/venvs/agx_plain/bin/python
CFG=config/safety_gymnasium_fhdcmpo_c4_cvar_refnorm.yaml

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "$(date +%T) waiting for pid $WAIT_PID to finish before starting c4..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "$(date +%T) predecessor finished."
fi

for seed in 2 3 4; do
    echo "$(date +%T) === starting c4 seed ${seed} ==="
    CUDA_VISIBLE_DEVICES=1 $PY scripts/train/train_safety_gymnasium.py \
        --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
        --config "$CFG" \
        --cost_limits 25.0 --max_iterations 20000 --seed ${seed} --deterministic
    echo "$(date +%T) === c4 seed ${seed} exited with code $? ==="
done
echo "$(date +%T) c4 queue done."
echo "Next: scripts/analysis/pick_c_winner.py --c2_dir <c2 run> --c3_dirs <the three c4 run dirs>"
