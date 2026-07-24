# SafeSAC now inherits SAC — reward-path drift eliminated, n-step for safe algos

*2026-07-23 (Claude Code). Motivated by the MPO-vs-CVPO comparison: the actor updates were
line-identical, but `SafeSAC` was a standalone reimplementation of SAC's scaffolding, so the
reward path of CVPO/SafeSAC could silently drift from SAC/MPO.*

## What changed

1. **`SafeSAC(SAC)`** (`safe_rl/algorithms/safe_sac.py`): all non-safety mechanics are now
   inherited — replay buffer (incl. in-buffer n-step), twin reward-critic update, bootstrap
   channel, entropy machinery, optimizer factory (`adam`/`adamw` + `weight_decay`/`betas`).
   SafeSAC keeps only the safety layer: cost critics, PID Lagrangian, safety actor objective.
2. **Extra-critics hook in `SAC.update()`** (`sac.py`): `_update_extra_critics(batch, ...)`
   is called once per gradient step after the reward-critic update; `_extra_critic_keys`
   declares the reported loss keys. SafeSAC trains its cost critics through it.
3. **n-step cost aggregation** (`storage/replay_storage.py::_sample_n_step`): `costs` are
   discounted-summed over the same done-masked window as `rewards`. Previously costs fell to
   the "start step" branch, which would have paired a 1-step cost with an n-step-discounted
   bootstrap. The FSRL reference CVPO applies n-step uniformly to reward and cost critics
   (verified against `fsrl/policy/cvpo.py`: one `compute_nstep_returns` for all critics).
4. **Safe off-policy algos can now use `n_step > 1`**: `SafeSAC.supports_storage_n_step` is
   inherited `True`, so the runner's in-buffer path activates and its old
   "n_step > 1 is not supported with safe RL algorithms" guard no longer triggers (it stays
   as a backstop for non-native algos). Closes a fidelity gap vs FSRL (MujocoBaseCfg n_step=3).
5. **Restored `safe_rl/env/__init__.py`** legacy alias (documented in CLAUDE.md, tested in
   `test_imports.py`, but missing from the tree — pre-existing failure, unrelated).

## Deliberate behavior changes (small, documented)

- SafeSAC's reward-critic loss is now SAC's `0.5*mse(q1)+0.5*mse(q2)` (was unscaled
  `mse+mse`, i.e. ×2). Near-neutral under Adam; buys exact MPO/CVPO reward-path consistency.
  (FSRL itself uses unscaled TD²; there is no single canonical choice — consistency is the point.)
- SafeSAC gains `optimizer`/`weight_decay`/`betas`/`target_entropy_scale` passthrough
  (defaults unchanged: Adam, wd=0).

## Consequence for the algorithm comparison

`CVPO(λ=0)` and `MPO` now share the *same executable code* for everything except the cost
terms: same critic backup, same n-step, same optimizers, byte-identical E/M-step scaffolding
(`mpo.py` / `cvpo.py` were already line-identical modulo safety). Drift is now structurally
impossible rather than accidentally absent.

## Validation

- Full suite: **396 passed, 0 failed** (was 390 passed + 1 pre-existing `safe_rl.env` failure).
- New regression tests in `tests/test_safe_sac_nstep.py`: inheritance contract, n-step
  cost==reward aggregation equality, done-truncation, SafeSAC and CVPO end-to-end with
  n_step>1.

Cross-refs: [[fixA_vs_your_impl]] (the comparison that surfaced this), [[m0-baselines]].
