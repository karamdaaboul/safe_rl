#!/usr/bin/env bash
# Paper baseline queue (2026-08-21): after the c2-60k confirmation run (pid $1) finishes,
# 1) FSRL reference CVPO on SafetyPointGoal1Gymnasium-v0, seed 2, epoch=100 (2M env steps --
#    matched to our 60k-iteration budget of 1.92M). Benchmark MujocoBaseCfg hyperparams made
#    explicit (their defaults: unbounded, gamma 0.995, n_step 3), testing_num 8 for a dense
#    deterministic test curve in progress.txt.
# 2) Our on-policy PPOL_PID on SafetyPointGoal1-v0, seed 2, repo-standard config.
# One training at a time. Report + critic-quality probes run afterwards (see codex note).
set -u
C2_PID="${1:?pass the c2-60k pid}"
while kill -0 "$C2_PID" 2>/dev/null; do sleep 300; done
echo "$(date +%T) c2-60k finished; starting FSRL CVPO PointGoal1"

cd /home/human/workspaces/fsrl_m0_src
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online MUJOCO_GL=egl OMP_NUM_THREADS=4 \
/home/human/venvs/fsrl_m0/bin/python examples/customized/train_cvpo.py \
    --task SafetyPointGoal1Gymnasium-v0 --seed 2 --device cuda \
    --epoch 100 --cost_limit 25.0 --unbounded True --gamma 0.995 --n_step 3 \
    --step_per_epoch 20000 --buffer_size 200000 --testing_num 8 \
    --logdir /home/human/workspaces/fsrl_runs/pointgoal1_paper --suffix paper
echo "$(date +%T) FSRL CVPO exited with code $?; starting PPOL_PID"

cd /home/human/workspaces/safe_rl
CUDA_VISIBLE_DEVICES=1 /home/human/venvs/agx_plain/bin/python scripts/train/train_safety_gymnasium.py \
    --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
    --config config/safety_gymnasium_ppol_pid.yaml --cost_limits 25.0 --seed 2 --deterministic
echo "$(date +%T) PPOL_PID exited with code $?"
echo "$(date +%T) paper baseline queue done."
