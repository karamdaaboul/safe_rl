# Memory Index

- [SAC n-step threshold scaling](sac-nstep-threshold-scaling.md) — off-policy n-step replay crashes at high num_envs unless start_random_steps/update_after >= num_envs*n_step; this alone explained SAC underperformance on G1-Flat.
- [G1-Flat PPO/SAC parity](g1-flat-ppo-sac-parity.md) — SAC matches (6k iters) then exceeds (12k iters) PPO on Unitree-G1-Flat after the threshold fix; includes eval/play gotchas (class_name stripped from saved cfg, MUJOCO_GL=egl).
