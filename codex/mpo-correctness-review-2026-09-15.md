# MPO correctness review — 2026-09-15

Follow-up: the reviewed defects have been addressed; see
[implementation changes and validation](mpo-correctness-fixes.md). This report
preserves the evidence from the implementation before those changes.

Reviewed the working tree at HEAD `b1edc58`, including the existing uncommitted
`mean_l2_coef` change. Scope: Gaussian `MPO` in `safe_rl/algorithms/mpo.py`, inherited
SAC critic updates, runner checkpointing, MPO configs and focused tests. This is
a review; algorithm, config and test files were not changed.

## Verdict

The standard-proposal E-step equations and M-step optimization signs are correct.
The implementation uses Acme-style decoupled Gaussian constraints, but it has
numerical, optional-proposal and integration defects. Passing smoke tests does
not establish accurate constrained optimization or literal paper parity.

## Findings

### P1 — Checkpoint resume loses the E/M reference policy

`safe_rl/runners/off_policy_runner.py:524–613` saves/loads the policy and optimizers,
but not `MPO.actor_target`, `eta`, `alpha_mean`, `alpha_var`, or `_actor_update_count`.
The target actor is a separate object created at initialization (`mpo.py:129`).
On resume, the online actor is restored while the E-step sampler and M-step KL
anchor retain the new run's random initialization. Multipliers reset to zero.

An in-memory probe of the actual runner save/load methods restored online mean
bias `[2, 2]`, but left the destination target bias `[0.135331, -0.020684]` instead
of the source target `[1, 1]`. Saved dual state `(eta=7, alpha_mean=3, alpha_var=4,
counter=17)` became `(1, 0, 0, 0)`. No checkpoint file was written by this probe.

Recommendation: serialize algorithm state and restore it before training. For
legacy checkpoints without a target actor, explicitly initialize it from the
loaded online actor. Ordinary deterministic inference uses the online actor and
does not depend on this missing target.

### P2 — E-step solver success does not establish the requested KL

`mpo.py:223–242` uses finite-difference SLSQP with default tolerances, does not
check `res.success` or constraint residuals, and accepts every finite temperature.
Absolute Q offsets affect numerical derivatives; small action-value spreads make
absolute objective tolerances too loose. The warm start is also floored at 0.001.

With NumPy `default_rng(0).normal(size=(64, 128))`, initial eta 1 and epsilon 0.1:

| Q input | Returned eta | Independently solved eta | Realized mean KL | Status |
|---|---:|---:|---:|---:|
| unit normal | 2.204237 | 2.205464 | 0.100110 | success |
| normal × 0.0001 | 0.000353214 | 0.000220546 | 0.039276 | success |
| normal + 100,000,000 | 1.0 | 2.205464 | 0.462544 | success |

The large-offset case tests the float64 solver directly; it is not evidence that
float32 critics preserve such small differences at that offset. The small-scale
failure also reproduces with float32 Q values converted to float64 as in training.
Warm starting improves it, but does not make every solve exact.

Recommendation: center Q per state, handle scale explicitly, and validate the
residual `epsilon - mean(KL)` with boundary-aware conditions. A bracketed solve of
that monotone residual is suitable for a nondegenerate, active constraint. Handle
flat Q, ties and inactive constraints separately; equality is not always required.

### P2 — Widened proposals change the E-step reference distribution

`mpo.py:255–273` samples from `rho = Normal(mu_old, scale * sigma_old)` but retains
the uncorrected dual and `softmax(Q / eta)`. If `scale != 1`, the population target
is proportional to `rho * exp(Q / eta)`, not `pi_old * exp(Q / eta)`. The logged
`sum(w * log(N*w))` measures reweighting relative to the proposal samples.

Reproduction: two action dimensions, scale 2, constant Q. Weights are uniform and
reported `kl_q` is zero, but `KL(rho || pi_old) = 3 - 2*log(2) = 1.613706`, far
above epsilon 0.1. This is the analytic population KL; finite weighted samples
approximate that population. Default scale 1 avoids this defect.

Recommendation: require scale 1 for standard MPO, or include `pi_old/rho`
importance ratios consistently in both the dual integral and fitted weights,
and adapt the diagnostics to the corrected estimator.

### P2 — Bellman evaluation and the E-step use different policies

The E-step samples `actor_target` (`mpo.py:252`), but the critic backup calls
`policy.sample_with_log_prob`, which samples the online actor:

- scalar: `safe_rl/algorithms/sac.py:544`;
- categorical: `safe_rl/algorithms/mpo.py:184`;
- quantile: `safe_rl/algorithms/sac.py:693`.

Instrumentation of the scalar backup recorded one online sampling call and zero
target-actor calls. The critic therefore tracks a different policy from the
E-step reference once the two actors diverge. Setting `estep_use_target_critic`
only changes which Q network is queried; it does not repair the backup policy.
This is a paper-fidelity issue, not proof that this hybrid cannot learn.
The original E-step uses Q of its reference policy; Acme also samples its target
actor for the critic bootstrap. [Paper, §3.2](https://arxiv.org/pdf/1806.06920),
[Acme learner](https://raw.githubusercontent.com/google-deepmind/acme/master/acme/agents/tf/mpo/learning.py).

Recommendation: use one explicitly chosen reference policy consistently for
candidate sampling and policy evaluation when implementing the paper algorithm.

### P2 — Configured mean critic reduction is ignored

`config/safety_gymnasium_mpo.yaml:27` and several parity configs request
`estep_q_reduction: mean`. MPO has no such argument; SAC prints an ignored-argument
warning and discards it (`sac.py:115–116`). `mpo.py:264` always takes the minimum.
A probe with Q1=0 and Q2=2 returned zero despite requesting the mean (one).

Recommendation: implement and validate the configured reduction, or update the
configs to explicitly describe the intended fixed minimum. A pessimistic minimum
is a deliberate algorithm variant; the config/behavior mismatch is the defect.

### P2 — M-step diagnostics omit the final optimizer update

`mpo.py:340–364` updates parameters and then reports KL, std and mean tensors
computed before that update. With identical online/target initialization and
one M-step iteration, the probe reported both KLs as zero even though the returned
actor had mean KL `7.32e-7` and variance KL `1.19e-6` against the original anchor.
This is a diagnostic defect, not a reversed policy gradient.

Recommendation: recompute diagnostics after the final optimizer step and before
updating the target. With per-dimension constraints, also report the largest
individual budget ratio; the average ratio can hide one violated dimension.

## E-step mathematics that is correct

For N samples from the reference policy, the implemented objective is

`g(eta) = eta*epsilon + eta*mean_s(log(mean_i(exp(Q_i(s)/eta))))`.

`_solve_eta` correctly uses log-mean-exp, and `_estep_weights` correctly normalizes
over the action-sample axis (`dim=0`). The missing explicit policy-density factor
in the weights is intentional because samples already come from the reference:
`q*(a|s) ∝ pi_old(a|s) exp(Q(s,a)/eta)`. Values, samples, weights and old policy
statistics are detached during the E-step. These match equations 8–9 of the
[original paper](https://arxiv.org/pdf/1806.06920).

For an interior active solution, `g'(eta) = epsilon - mean_s(sum_i(w_i*log(N*w_i)))`.
This supplies a direct correctness check beyond optimizer status.

## M-step mathematics and limitations

The actor minimizes negative weighted log likelihood plus positive KL penalties.
The multiplier update `alpha += step_size * (KL - epsilon)` correctly strengthens
the penalty when a constraint is exceeded. Old statistics stay fixed throughout
the inner loop; the target is updated after the loop.

For each action dimension, the code computes

- `C_mean = (mu - mu_old)^2 / (2*sigma_old^2)`;
- `C_var = log(sigma/sigma_old) + sigma_old^2/(2*sigma^2) - 1/2`.

With `decoupled_mstep=True`, its two weighted likelihoods also match the
[Acme loss](https://raw.githubusercontent.com/google-deepmind/acme/master/acme/jax/losses/mpo.py).
The false setting is a coupled weighted-MLE variant. Joint and per-dimension
budgets both have valid meanings, but require different epsilon units.

These Acme-style KLs are **not an exact additive decomposition of the full joint
old-to-new Gaussian KL**: the latter's mean term uses the new variance. Appendix
D.3 of the [2018 paper](https://arxiv.org/pdf/1806.06920) presents that latter
decomposition (and swaps the mean/covariance term labels). This distinction is a
variant choice, not a reason to reverse the implemented KL arguments.

Two qualifications matter:

1. Five primal/dual iterations with upper-capped multipliers do not guarantee a
   feasible final actor. `config/bench4/mpo.yaml:32–33` and `config/brax_mpo.yaml:40–41`
   still use the old low caps 0.1/10; defaults are 10/1000. Larger finite caps also
   require monitoring. For a fixed unit-variance Gaussian with weighted sample
   mean m, penalized optimum `mu=m/(1+alpha)` gives KL `m^2/(2*(1+alpha)^2)`.
   At m=0.4 and cap 0.1 this is 0.0661, so a 0.01 constraint cannot be enforced even
   after convergence. A finite penalty is not a hard projection.
2. Omitting the tanh/affine Jacobian in the M-step is correct: samples are fixed,
   so that Jacobian is independent of actor parameters. Gaussian KLs also equal
   the KLs of the corresponding transformed distributions under the shared
   invertible transform. `mean_l2_coef > 0` deliberately adds an extra prior; its
   sign and gradient are correct, but it changes the unregularized MPO objective.

## Correction to earlier project explanations

`codex/mpo-estep-mstep-math.md` overstates the solver as exact/unconditionally
better than Acme. The numerical probes above show why that is not justified.
Its split-MLE scaling explanation and the comment in `tests/test_mpo.py:181`
also confuse loss value with gradient: at online=old, the split loss value is
doubled, but its mean and log-std gradients equal the coupled gradients because
the old halves are detached. Probe gradients were `[0.969686, 1.677825]` for both.
Do not divide the split loss by two on the basis of that existing value-only test.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mpo.py tests/test_mean_l2_penalty.py tests/test_offpolicy_diag.py
```

Result: **47 passed in 6.06 seconds** with PyTorch 2.10.0 and SciPy 1.15.3.
Additional CPU probes covered temperature scaling/offsets, proposal KL,
configured critic reduction, backup actor selection, final-step diagnostics,
runner checkpoint round-trip and split-MLE gradients. The first checkpoint probe
needed its dummy environment completed with `cfg` and `reset`; the completed
probe then ran successfully. No training-performance claims are inferred.

Minimal reproduction of the small-scale solver issue:

```python
import numpy as np
from scipy.special import softmax, xlogy
from safe_rl.algorithms.mpo import MPO

alg = object.__new__(MPO)  # The solver only needs eta and eps_dual.
alg.eta, alg.eps_dual = 1.0, 0.1
q = np.random.default_rng(0).normal(size=(64, 128)) * 1e-4
eta = alg._solve_eta(q)
w = softmax(q / eta, axis=0)
print(eta, xlogy(w, 64 * w).sum(axis=0).mean(), alg._solver_status)
# 0.000353214..., 0.03927578..., 0.0 (success)
```
