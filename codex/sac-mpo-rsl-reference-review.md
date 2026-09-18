# SAC/MPO correctness review against RSL-RL-SAC

Reviewed 2026-09-17. Scope: scalar twin-critic, one-step SAC/MPO used by LEAP;
actor/critic observation normalization, squashed Gaussian actions, timeout replay,
target networks and checkpoint state. Not an exhaustive audit of all safe-RL,
distributional, recurrent, or distributed variants.

## Conclusion

Core SAC loss/update math matches the supplied implementation in a controlled
one-update comparison. MPO E/M formulas are consistent with the paper/Acme
objectives, with deliberate algorithmic differences. Two implementation issues
remain: normalization statistics change during model queries and overcount MPO
candidate states; SAC checkpoint restoration omits entropy state. Poor LEAP
performance cannot currently be attributed solely to hyperparameters, but neither
finding proves the cause of the measured performance gap.

## Findings

### 1. Observation statistics depend on replay queries and MPO action count

Priority: training correctness/stability investigation; affects the LEAP config.

`safe_rl/modules/normalizer.py:63` updates statistics in every training-mode
forward. `safe_rl/modules/sac_actor_critic.py` calls these normalizers inside
sampling and Q evaluation. Thus target queries and repeated replay sampling
change the statistics, rather than only new observations doing so.

`safe_rl/algorithms/mpo.py:365` expands each critic state across N action samples
before calling `evaluate_q_target`, which normalizes this expanded batch. A CPU
reproduction on the exact frozen LEAP source shows **8 distinct states ×64 actions
incrementing the critic statistics count by512**. Increasing E-step action count
therefore changes observation weighting, not just the action integral estimate.
This also changes the input transform used by a parameter-frozen target network.

Actual final pilot checkpoint counts after8,000,000 transitions:

| Algorithm | Actor observation count | Critic observation count |
|---|---:|---:|
| SAC | 73,492,088 | 98,245,632 |
| MPO5 | 73,492,088 | 2,161,403,904 |
| MPO1 | 73,492,088 | 2,161,403,904 |

These are statistics updates, not extra environment transitions. Large counts
alone do not prove numerical instability; the demonstrable issue is sample/query
count-dependent statistics. Repeated replay-state updates might be an intentional
alternative estimator, but are not equivalent to the supplied reference workflow.

The reference's normalization forward is read-only; its environment transition
handler explicitly updates actor and critic statistics. See
[reference normalizer](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/modules/normalization.py#L43)
and [transition handling](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py#L191).

Suggested correction: make SAC/MPO model queries read-only with respect to
normalization; update their statistics explicitly on collected actor/critic
observations, with a documented timeout/reset policy. Avoid changing the shared
normalizer globally without checking PPO and other consumers. Normalize distinct
states before candidate expansion. Verify that changing N leaves counts/statistics
unchanged, and that a replay update does not mutate observation statistics.
Then compare a fresh corrected seed1 run against existing results.

### 2. SAC checkpoint omits learned entropy state

Priority: genuine resumed-training bug, not an explanation for uninterrupted pilot
training or policy evaluation.

`safe_rl/runners/off_policy_runner.py:528` saves actor/critic weights and optimizers,
and conditionally calls algorithm checkpoint hooks. MPO has these hooks; SAC does
not. SAC's `log_alpha` is a standalone algorithm tensor, absent from policy state.
Its alpha optimizer and update counter are also not restored.

CPU save/load reproduction: alpha **0.001 before save ->0.2 after reload**, even
with `load_optimizer=True`. The reference saves alpha/log-alpha and alpha optimizer.
See [reference checkpoint code](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py#L453).

Suggested correction: SAC checkpoint hooks for log_alpha, alpha optimizer,
update counter and versioning; preserve MPO's separate hook semantics and legacy
checkpoint behavior. Replay/RNG restoration is additionally needed for exact
resumption. The current tuning campaign starts validation runs fresh, so this
checkpoint defect does not alter their training or action-only evaluations.

## Core checks

### SAC

Compared to pinned [reference SAC](https://github.com/leggedrobotics/rsl_rl_sac/blob/e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521/rsl_rl/algorithms/sac.py):

- Scalar critic target: reward + discount ×bootstrap mask ×(minimum target Q −alpha logπ).
- Actor objective: mean(alpha logπ −minimum online Q).
- Log-alpha gradient sign and detached entropy target match.
- Critic loss uses half the sum of the two mean-squared errors.
- Target networks interpolate toward online critics with tau.
- Replay contains raw actor and privileged critic observations; timeout successors
  are replaced by true final observations, and true terminations do not bootstrap.
- Local action sampling uses a reparameterized Normal followed by tanh, with the
  stable tanh log-Jacobian and action-scale determinant.

A CPU test executes the actual reference `SAC.update` method, extracted from its
AST without altering the method, using adapters around identical local networks.
This isolates update equations from architecture and environment interfaces.
Fixed-alpha case: maximum parameter difference0.0, actor loss difference0.0.
Autotuning case with alpha LR0: same results and identical log-alpha gradient.
The zero alpha LR deliberately removes reference/local alpha-before/after-actor
ordering; this is not a claim of bitwise full-run parity with ordinary alpha LR.
Normalizer updates are disabled in this algebra test; the issue above is tested
separately. Frozen-critic gradients in reference save computation; their absence
locally did not change the tested actor/critic parameter update.

### MPO

The supplied repository has no MPO implementation. Checked independently against
[the MPO paper](https://arxiv.org/html/1806.06920v1) and
[DeepMind Acme MPO loss](https://github.com/google-deepmind/acme/blob/master/acme/tf/losses/mpo.py).

The E-step samples the frozen target policy, weights actions along the sample
axis using exp(Q/eta), and solves the dual against the state-averaged KL budget.
The M-step uses detached sample weights, weighted log likelihood, old-to-new
Gaussian KL, and separate mean/covariance constraints. Multiplier updates increase
penalties on violations. Actor target synchronization follows the M-step.
The target-policy critic backup excludes SAC's entropy term.

Using pre-tanh likelihoods/KLs is valid for a shared invertible tanh transform:
the Jacobian is parameter-independent for fixed samples, and KL is invariant.
Finite numerical tanh saturation can still impair exploration/performance.
A finite number of primal/dual steps does not guarantee every realized KL meets
its budget on every minibatch.

The experiment label “original MPO” means our five-M-step baseline, **not a
faithful reproduction of the original paper**. Differences include tanh-bounded
actions, twin scalar TD critics with one-step backups instead of paper Retrace,
Polyak actor targets, and a bracketed E-step solve. Acme uses a Gaussian with an
optional out-of-bounds penalty and different dual optimization. These are variant
choices, not by themselves sign or derivation errors.

## Other performance-relevant differences

Reference actor output heads use near-zero initial means and controlled log-std;
LEAP configs leave local `head_init=default`. Reference action scaling derives
joint bounds from IsaacLab; LEAP uses the Playground [-1,1] action contract.
These are meaningful configuration/architecture differences; copying the
IsaacLab scaling directly into LEAP would require checking its action semantics.

Our four updates per250 new transitions correspond to0.016 gradient updates per
new transition, with a256-sample minibatch (~4.096 replay sample draws per new
transition). “Four updates per step” here means vector step, not single transition.
The 5000-transition warmup is only20 vector steps. Such settings warrant tuning,
but are not automatically coding errors.

## Evidence and reproducibility

Reference checkout: `/tmp/rsl_rl_sac_review_20260917`, commit
`e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`.
Relevant current workspace files byte-match frozen LEAP source (SAC, MPO, actor,
SAC policy, normalizer, runner, replay, Playground wrapper).

Artifacts under `logs/sac_mpo_review_20260917/`:

- `reproduce.py`, `reproduction.json`: normalization and alpha save/load findings;
  includes actual pilot checkpoint statistics.
- `reference_parity.py`, `reference_parity.json`: direct adapted reference update comparison.

Commands:

```bash
/home/human/venvs/agx_plain/bin/python logs/sac_mpo_review_20260917/reproduce.py
/home/human/venvs/agx_plain/bin/python logs/sac_mpo_review_20260917/reference_parity.py
/home/human/venvs/agx_plain/bin/python -m pytest tests/test_mpo_correctness.py tests/test_mpo_critic_action_samples.py tests/test_playground_asymmetric_reset.py tests/test_mjlab_terminal_observations.py tests/test_safe_sac_nstep.py -q
```

Existing focused suite:70 passed. Passing these tests does not cover the newly
reproduced issues. Review artifacts only: no algorithm edits, no changes to
running jobs, and no new training launched during this review.
