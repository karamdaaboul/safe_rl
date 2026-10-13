# Regularized Online Adaptation (ROA) from Deep Whole-Body Control

*2026-10-13. `safe_rl` main, `unitree_rl_mjlab` main (G1 ROA tasks).
Status: implemented, unit-tested, one full 2000-iteration seed against the same-task PPO baseline.*

Source: Fu, Cheng, Pathak, *Deep Whole-Body Control*, CoRL 2022
(openreview zldI4UpuG7v, code github.com/MarkFzp/Deep-Whole-Body-Control, `rsl_rl/`).
Only ROA is ported; DWBC's advantage mixing, torque supervision and leg/arm heads are not.

## What the reference does (read from their rsl_rl, not the paper)

- Actor obs = `[prop | priv | history]`; `priv_encoder` MLP `[64, 20]` (ELU after every layer)
  gives `z`; `StateHistoryEncoder` (Linear->30 ch, Conv1d layouts for T=10/20/50, Linear->20)
  gives `z_hat`. Actor MLP sees `[prop, latent]`. Critic sees `prop + priv`.
- Every `dagger_update_freq = 20` iterations (`it % 20 == 0`, so iteration 0 too): rollout
  with `z_hat`, then `update_dagger` only (no PPO update): `||sg(z) - z_hat||_2`, separate Adam
  on the history encoder.
- Other iterations: PPO with `+ coef * ||z - sg(z_hat)||_2`; coef ramps `[0 -> 0.1]` from update
  3000 over 7000 updates (`priv_reg_coef_schedual`). Both update types advance the counter.
- L2 *norm* (not squared), averaged over the batch.

## How it maps here

| Piece | Where |
|---|---|
| encoders + actor | `safe_rl/modules/actor_critic_roa.py` (`ActorCriticROA`, `StateHistoryEncoder`) |
| `[obs, priv, history]` composition | `safe_rl/envs/roa_obs_wrapper.py`, applied by `OnPolicyRunner` only for ROA policies |
| reg loss, adaptation step, separate optimizer, checkpoint state | `PPO(roa_cfg=...)` in `safe_rl/algorithms/ppo.py` |
| iteration switch | `OnPolicyRunner.learn` (`roa_adaptation`) |
| env groups `priv`, `actor_history` | `unitree_rl_mjlab: src/tasks/velocity/config/g1/roa_env_cfgs.py` |
| privileged terms | `unitree_rl_mjlab: src/tasks/velocity/mdp/observations.py` (`priv_*`) |

Decisions:
- `act_inference` always uses `z_hat`: periodic eval and export measure the deployable policy.
  `act_inference_privileged` gives the teacher upper bound.
- The history group excludes the height scan; the current actor obs (with scan on rough) still
  goes to the actor directly. History includes the current step (mjlab buffer), the reference's
  excludes it; immaterial.
- The baseline must run on the same task (`--env_id Unitree-G1-Flat-ROA` with plain
  `ActorCritic`): the wrapper is not applied, so it never sees `priv`, but its critic does.
- PPO only for now. P3O / PPOL_PID have their own `_update_policy` and P3O rebuilds the optimizer
  over all policy params, so extending needs the same two hooks there.

## Measured

- G1-Flat-ROA groups: actor 98, priv 92 (friction 1, CoM 3, mass 1, Kp 29, Kd 29, encoder
  bias 29), history 10x98, critic 113+92.
- Env speed at 1024 envs, zero actions (reset-heavy worst case): stock 27.9k env-steps/s;
  everything re-sampled per episode 7.6k; mass+CoM at startup 23.1k.
  Mass/CoM changes trigger `set_const` recompute on every step in which ANY env resets, so with
  thousands of envs it is effectively every step. Default is therefore mass/CoM at startup,
  friction / PD gains / encoder bias per episode (`resample_mass_com_per_episode` flag).

## First full run (CLAIX c23g, one H100 each, seed 1, 4096 envs, 2000 iterations; pre-port branch)

| | PPO + ROA | PPO baseline |
|---|---|---|
| config | `unitree_g1_flat_ppo_roa.yaml` | `unitree_g1_flat_ppo.yaml` |
| wall time | 48 min | 44 min |
| final train reward / episode length | ~68 / ~1000 | ~67.5 / ~1000 |
| final `error_vel_xy` / `error_vel_yaw` | 0.45 / 0.60 | 0.45 / 0.61 |

`roa_priv_reg` fell from ~1.0 to 0.025 and `roa_adaptation` ended at 0.027, so the student
latent tracks the teacher closely. The ROA training reward uses the teacher latent; one
20 s video episode with the student (`scripts/eval/unitree_mjlab.py`) scored 80.9 vs 80.7 for the
baseline. One seed and one episode show both walk, not which is better.

## Open

- Multi-seed eval of the student vs the baseline under the randomized physics.
- Deployment: the ONNX/deploy path must feed the 10-step history in time-major order and drop
  `priv`; not done.
- `priv_reg_coef_schedule` was scaled to [0, 0.1, 500, 1000] for 2000 iterations; not tuned.
