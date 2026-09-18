# LEAP cube Z-axis pilot

Campaign: `logs/leap_zaxis_pilot_20260916/`.
User authorized PPO, SAC, original MPO (5 M steps), one-step MPO (1 M step),
250 parallel environments per algorithm, one seed, 8,000,000 transitions each.
SAC/PPO and MPO optimizer settings retained from Humanoid comparison.
Default Playground task settings (500-step horizon, noisy actor observations,
privileged critic observations); no extra domain randomization.

Runtime: `/home/human/venvs/agx_plain/bin/python`, with campaign-local PYTHONPATH
for frozen safe_rl source, FastTD3 Playground fork, and missing Python packages.
Playground pinned to `1699a6d3d9996f2e8391791d55c2d91eccbe33ba`, matching the
existing wrapper's raw terminal observation API. Upstream latest downloaded
previously is unused. Existing virtualenv packages are not modified.

Fixed `MujocoPlaygroundVecEnv.reset` to preserve privileged observations at
initialization; evaluation twins preserve the final-observation flag.
Regression: `tests/test_playground_asymmetric_reset.py` passed.
GPU smoke exercises forced timeouts, all four updates, reload, and both eval modes.

Controller: `scripts/queue/leap_comparison.py` (frozen copy in campaign).
Workers refresh `comparison.json` and `comparison.md` every 60 seconds and on
phase changes. Read these for status instead of frequent GPU/process polling.
Per-algorithm `status.json`, `train.log`, and `eval.log`; checkpoints in each
algorithm's `checkpoints/`. Final evaluation automatically follows training:
50 first episodes, seed 10001, deterministic/stochastic action modes,
return, episode length, drop rate, and integrated cube Z angular velocity.
One seed is a pilot, not a conclusive algorithm ranking.

Launch only after `smoke/status.json` says passed:

```bash
/home/human/venvs/agx_plain/bin/python logs/leap_zaxis_pilot_20260916/leap_comparison.py launch --campaign logs/leap_zaxis_pilot_20260916
```

GPU access requires escalation. State at note creation: dependency setup/smoke;
consult campaign status files for current state.

Additional compatibility fixes verified during smoke: tuple-valued flat observation
sizes are accepted; terminal actor/critic observations are separate tensors,
matching current PPO and off-policy runner APIs. The previous dict payload was
incompatible. Two focused wrapper regression tests pass; 24 existing PPO/mjlab
timeout tests also pass. First MJX GPU compilation takes several minutes.

## Launch

All four GPU smoke tests passed at 2026-09-17 06:24:29 UTC. Launched all four
full-budget workers at 06:24:49 UTC. Worker PIDs: PPO 1804250, SAC 1804251,
MPO 1804252, one-step MPO 1804253. Training child PIDs: PPO 1804256, SAC
1804255, MPO 1804257, one-step MPO 1804254. GPU 0: PPO and original MPO;
GPU 1: SAC and one-step MPO. Initial status confirmed `training` for all four;
full-size MJX initialization/compilation precedes the first logged transitions.

## Completed results

All four runs and both final evaluations completed on 2026-09-17, last finish
08:19:48 UTC. Each final checkpoint exists and each run logged 8,000,000 transitions.
Evaluation means checked against 50 stored episode returns per mode.
Deterministic / stochastic returns: PPO 11.279 / 11.612, SAC 1.826 / 2.216,
original MPO 1.146 / 1.018, one-step MPO 2.116 / 3.780.
Corrected the live reporting script's step parser to read Total timesteps/steps;
PPO's heading already used transitions because log_env_steps was enabled.
Regenerated comparison reports; frozen training controller remains unchanged.
