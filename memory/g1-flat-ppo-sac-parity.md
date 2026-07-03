---
name: g1-flat-ppo-sac-parity
description: SAC matches/exceeds PPO on Unitree-G1-Flat once the n-step buffer-threshold bug is fixed
metadata:
  type: project
---

On `Unitree-G1-Flat` (mjlab, num_envs=4096), 2026-06-12 comparison of `Train/episode_reward`:

- **PPO** (mjlab registered cfg): converged ~**32.7** (max 33.2) in 1500 iters (~147M env steps).
- **SAC** (`config/unitree_g1_flat_sac_tune.yaml`, original hyperparams): ~**31** at 6k iters
  (~24.6M env steps), and ~**40** at 12k iters — i.e. it *exceeds* PPO with ~6x fewer env steps.

The only change vs the shipped `unitree_g1_flat_sac.yaml` was scaling
`start_random_steps`/`update_after` to 40960 for num_envs=4096 (see
[[sac-nstep-threshold-scaling]]). No SAC hyperparameter tuning was required.

**Eval/play gotcha:** training strips `class_name` from the dumped `params/agent.yaml`
(runner pops it as a kwarg before dump), so `scripts/eval/unitree_mjlab.py` auto-detects
PPO and fails on SAC checkpoints with `Unknown actor_type: stochastic`. Workaround: pass
`--config config/unitree_g1_flat_sac_tune.yaml` explicitly when evaluating SAC. Headless
video also needs `MUJOCO_GL=egl` and `--device cuda:0` (the eval script sets neither).
