# Ant timeout-observation fix

## Source and scope

The user's SAC correction is in **agxcave**, commit
[`9a4a35c8640945bc094aa9450cd30f9cf8f03211`](https://github.com/karamdaaboul/agxcave/commit/9a4a35c),
"Forward terminal observations for SAC timeout bootstrapping". It modifies
`agxcave/agxrl/saferl/vecenv_wrapper.py` to forward SB3's `terminal_observation`
as `final_observation`, validate shapes, and reject timeouts without a valid final state.
The commit was inspected in the local agxcave checkout. It was adapted into safe_rl;
no unrelated agxcave changes were merged.

## safe_rl adaptation

- `MjlabVecEnv` forwards native terminal/final actor observations and separate final
  critic observations, with independent copies and shape validation.
- `OffPolicyRunner` enables final-observation capture on supporting wrappers and
  repairs both replay inputs. The next policy action still uses the reset observation.
- Older mjlab releases, including the one used by the Ant run, reset before computing
  observations. For compatible stateless tasks such as `Ant-Flat`, the adapter temporarily
  hooks `_reset_idx` during `step`, refreshes derived simulation quantities, and captures
  fresh actor/critic groups before reset. It bypasses the observation manager's stale
  cache and restores the original reset method even if capture fails.
- True termination takes precedence over simultaneous timeout, so a terminal transition
  does not bootstrap.
- The snapshot fallback is restricted to uncorrupted, stateless observations with no
  history, delay, commands, or step/interval events. Other mjlab off-policy tasks must
  publish native final actor/critic observations; if they do not, training raises at a
  timeout. This avoids advancing stateful observation processes twice or using reset states.
- Default on-policy construction does not enable snapshot capture. The subsequent
  PPO/SAC/MPO Ant comparison explicitly enables it for feed-forward PPO through
  `bootstrap_final_observations: true`; see `ant-ppo-sac-mpo-comparison.md`.

## Validation

106 CPU checks passed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mjlab_terminal_observations.py tests/test_truncation_bootstrap.py \
  tests/test_mpo_correctness.py tests/test_mpo.py tests/test_mpo_dime.py \
  tests/test_safe_sac_nstep.py tests/test_core_components.py
```

GPU integration used four `Ant-Flat` environments with two-step episodes. Captured
actor and critic observations for timeout rows matched a same-seed reference Ant with
resets disabled (`atol=rtol=1e-4`); running rows also matched. Replay retained the terminal
observations while the policy retained reset observations. Both SAC and MPO completed
12 training iterations with finite parameters and without missing-final-observation warnings.

- Reproduction script: `logs/mpo_ant_corrected_20260915/timeout_smoke.py`.
- Output: `logs/mpo_ant_corrected_20260915/timeout_smoke.log`.
- GPU command (requires GPU access):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safe_rl_mpl \
  /home/human/venvs/agx_plain/bin/python -u \
  logs/mpo_ant_corrected_20260915/timeout_smoke.py
```

The earlier 7.68M-step MPO run finished before this port and retains the timeout-target
bias. No full replacement training run was launched as part of this change.
