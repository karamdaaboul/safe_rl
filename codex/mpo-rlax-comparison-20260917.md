# MPO comparison against pinned RLax

Reference: https://github.com/google-deepmind/rlax/blob/036c859d6be16659db1dce7e4381431149a8eed3/rlax/_src/mpo_ops.py

Reviewed current `safe_rl/algorithms/mpo.py` and the actual frozen LEAP YAMLs.
No training code changed and no training jobs launched. This is an actor-loss
comparison, not proof of full learner correctness or performance superiority.

## Findings

- **E-step:** same action-axis softmax(Q/eta), with weights summing to one per
  state. Local `_solve_eta` minimizes the batch-mean dual by solving its stationary
  condition, epsilon minus mean KL(weights || uniform) = 0. RLax returns the same
  dual objective for an external optimizer. Direct solving versus incremental
  optimization changes dynamics and cost, not the underlying objective. Flat-Q
  and inactive-constraint cases are handled separately locally.
- **M-step:** local negative weighted log likelihood matches RLax's policy loss.
  Local KL penalties have the correct positive sign in the minimized actor loss.
  Local alpha ascent uses KL - epsilon, agreeing with gradient descent on RLax's
  alpha loss alpha*(epsilon - stop_gradient(KL)). Both support a joint constraint
  or separate action-dimension constraints.
- **Gradient separation:** local sampling, Q scoring, and E-step weights are
  under no_grad; old Gaussian parameters are detached. Multipliers are updated
  outside autograd. This matches the intended actor/dual separation of the RLax
  loss. RLax's helper does not itself freeze all caller-supplied Q/KL inputs.
- **KL definition:** local constraints use KL(old || mean-updated Gaussian) and
  KL(old || variance-updated Gaussian). RLax accepts caller-computed KL arrays;
  it does not select direction, Gaussian decomposition, or policy architecture.
- **Split fitting:** actual LEAP runs set `decoupled_mstep: true` and
  `per_dim_constraining: true`. Class defaults are false, so inspecting only the
  constructor would misdescribe these experiments. RLax can represent split
  fitting when supplied the sum of the two appropriate log probabilities.
- **Optimizer differences:** local alphas start at zero, use manually projected
  ascent with separate step sizes, and are updated before the actor optimizer
  step. RLax provides differentiable losses, not an optimizer/schedule. Its default
  projection clips at 1e-10; a raw alpha below that floor has zero gradient through
  the clip. Local projected ascent can leave zero. Numerical gradient parity was
  therefore tested at positive interior multipliers, not at this distinct boundary.
  Finite actor/dual steps do not guarantee exact satisfaction of each M-step bound.
- **Loss scalar versus gradient:** local actor loss includes constants
  -alpha*epsilon. RLax exposes temperature and alpha losses as well. Their logged
  total loss values need not match even when actor gradients match.
- **Tanh:** local candidates are stored pre-tanh, then squashed for Q evaluation.
  Omitting the tanh Jacobian from weighted fitting does not alter actor gradients
  for fixed sampled latent actions. KL is invariant under the common invertible
  action transform in exact arithmetic. Saturation remains a numerical/exploration
  concern; this argument does not promise good performance near action bounds.
  RLax takes log probabilities as input and imposes no tanh convention.
- **Selected LEAP arm:** `mpo1_mean_l2` adds coefficient 0.01 times the batch mean
  of squared pre-tanh means, summed over actions. This is an extra actor penalty
  beyond the referenced loss. Corrected one-step baseline uses coefficient zero.
- **Outside reference scope:** critic backup, replay, target networks, number of
  action samples, number of optimizer steps, and observation normalization are
  learner choices absent from this file. LEAP uses 64 E-step candidates, one or
  five M-steps, target-critic mean for candidate scoring, twin-min scalar TD
  backups with one action sample, and Polyak target actor updates. The file
  cannot validate the recent normalization change or rank training performance.

## Numerical verification

Artifacts: `logs/mpo_rlax_review_20260917/`.
Downloaded exact reference SHA256:
`781850c8b784cc68bb2b3def4580179e9d28fabcb9f34e2bdc4f4b45df43f632`.

`compare.py` executes the pinned RLax mathematical function bodies through JAX
on CPU. Because chex is unavailable, only its assertion statements are removed
via AST; mathematical statements are unchanged. This is not a full RLax package
integration test. The installed CUDA plugin emits a discovery warning, but the
CPU comparison finishes successfully.

Compared four cases: joint/per-dimension KL crossed with joint/split fitting,
64 action samples, seven states, two action dimensions, float64. Held actor
optimizer step inactive and captured actual local `_mstep` gradients with respect
to actor mean/log-std outputs; compared these against RLax autodiff using the same
updated positive multipliers and fixed Q/samples. Mean L2 was disabled to isolate
the common objective. Maximum errors across cases:

- E-step weights: 3.47e-17.
- Actor mean/log-std gradients: 1.32e-16.
- Interior dual gradients: 6.42e-17.
- Absolute RLax temperature derivative at local solved eta: 9.97e-15.

Reproduce:

```bash
PYTHONPATH=. /home/human/venvs/agx_plain/bin/python logs/mpo_rlax_review_20260917/compare.py
/home/human/venvs/agx_plain/bin/python -m pytest -q tests/test_mpo.py tests/test_mpo_correctness.py tests/test_mpo_critic_action_samples.py
```

Existing focused tests: **62 passed**, 14 PyTorch deprecation warnings.

Conclusion: no E/M-step weighting, sign, or gradient-separation error found in
this comparison. Local MPO is compatible with the reference loss mathematics,
with different optimization mechanics and additional configurable choices.
Neither this check nor the prior single-seed normalization experiment establishes
which configuration learns best. All cancelled training remains cancelled.
