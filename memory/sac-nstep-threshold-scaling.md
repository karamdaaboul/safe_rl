---
name: sac-nstep-threshold-scaling
description: Off-policy SAC/TD3 n-step replay sampling crashes at high num_envs unless start_random_steps/update_after are scaled
metadata:
  type: project
---

Off-policy runs (SAC, and any algo using `ReplayStorage` with `n_step > 1`) require
`start_random_steps` and `update_after` (env-step units in `OffPolicyRunner`) to be
**>= num_envs * n_step**. Otherwise `ReplayStorage._sample_n_step` raises
`RuntimeError: Not enough contiguous transitions for n-step sampling` on the first
update, because the buffer's per-env time dimension (`max_size // num_envs`) has fewer
than `n_step` filled columns.

The shipped `config/unitree_g1_flat_sac.yaml` sets both to 5000, which is fine at
`num_envs=1` but crashes at `num_envs=4096` (needs >= 4096*3 = 12288). Fixed copy used
for tuning: `config/unitree_g1_flat_sac_tune.yaml` with both at 40960.

**Key result (2026-06-12):** This threshold scaling was the *entire* reason SAC
underperformed on `Unitree-G1-Flat`. With it fixed and num_envs=4096, the original SAC
hyperparameters reached `Train/episode_reward` ~31, matching PPO (~30-31). The old
num_envs=1 SAC runs plateaued at -4.6 purely because of the tiny step budget. No SAC
hyperparameter changes were needed to match PPO. See [[g1-flat-ppo-sac-parity]].
