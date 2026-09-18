# MPO correctness fixes — 2026-09-15

Implemented the follow-up to `mpo-correctness-review-2026-09-15.md`.

## Changes

- E-step temperature: replace finite-difference SLSQP with a bracketed solve of
  `mean(KL(weights || uniform)) - epsilon`. Center Q per state, normalize its
  scale, solve in inverse temperature, and validate the residual. Handle flat Q,
  single samples, ties and inactive constraints explicitly. Nonfinite Q fails
  clearly instead of reusing an invalid temperature. Compute detached sample
  weights in float64 before converting back to the critic dtype.
- Standard proposals: `estep_sample_std_scale` must equal 1. Other values are
  rejected because the existing dual/weights have no importance correction.
- E-step critic reduction: honor and validate `estep_q_reduction: min|mean`.
- Reward policy evaluation: scalar, categorical and quantile backups use the
  frozen reference actor, with no entropy contribution. A SAC sampling hook
  preserves SAC's online policy/entropy behavior. Diffusion MPO overrides that
  hook with its target SDE sampler.
- M-step: keep the weighted likelihood and Acme-style mean/covariance constraints.
  Default multipliers have no upper cap (`alpha_*_max: null`); explicit legacy
  caps remain supported and are reported by `alpha_*_at_cap`. Remove the old
  low caps from Brax MPO, mjlab Ant MPO and bench4 MPO presets.
- Diagnostics: recompute mean/covariance KL and policy statistics after the final
  optimizer step, before syncing the reference actor. `kl_mean_max_rel` and
  `kl_var_max_rel` expose the largest budget ratio across constrained dimensions.
  Diffusion MPO also recomputes final path KL/likelihood.
- Checkpointing: runner hooks save/restore the target actor, temperature,
  mean/variance multipliers and update counters. Diffusion MPO additionally saves
  its path multiplier. Legacy checkpoints initialize the reference from the loaded
  online actor and reset missing MPO dual state, with a warning.
- Correct the old split-MLE test's interpretation: at equality the split loss
  value doubles, but its actor gradients equal the coupled gradients.

## Validation

Two overlapping CPU batches passed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mpo_correctness.py tests/test_mpo.py tests/test_mpo_dime.py \
  tests/test_cvpo.py tests/test_algorithm_mro.py tests/test_mean_l2_penalty.py \
  tests/test_fhdcmpo_dime.py tests/test_dueling_cost_critic.py tests/test_survival_horizons.py
# 191 passed before adding the two additional diffusion regression checks below.

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mpo_dime.py tests/test_core_components.py tests/test_safe_sac_nstep.py \
  tests/test_quantile_critic.py tests/test_truncation_bootstrap.py tests/test_offpolicy_diag.py \
  tests/test_cvpo_per_state.py tests/test_fhdcmpo.py tests/test_fhdcmpo_per_state.py \
  tests/test_fhdcmpo_td_lambda_fixed_point.py
# 180 passed, including the additional diffusion backup/checkpoint checks.
```

`python scripts/train/train_safety_gymnasium.py --help` succeeded. The main
Safety-Gymnasium MPO, Brax MPO, mjlab Ant MPO and bench4 MPO YAMLs initialized a
small CPU runner and completed an actor update; mean/min reduction and uncapped
multipliers matched their configs. `git diff --check` passed.

The M-step is still a finite primal/dual optimization, not an exact projection:
KL budgets can be exceeded temporarily. Monitor the final budget ratios instead
of treating the iteration count as a feasibility guarantee. Checkpoints restore
MPO-specific EM state; replay, RNG, and additional safe-variant cost-controller
state are not added by this change. No long training or performance comparison
was run. Shared MPO corrections affect inheriting variants as well.
