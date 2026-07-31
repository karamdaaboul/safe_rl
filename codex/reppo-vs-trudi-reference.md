# REPPO: our implementation vs. the TruDi reference source

Date: 2026-07-30. Reference: `safe_rl/trudi (1).zip` — a snapshot of the TruDi repo
(branch `cleanup_reppo_dime`), which ships the authors' own REPPO code.

Until now our `safe_rl/algorithms/reppo.py` was written against the paper
(arXiv:2507.11019) plus second-hand notes about the torchrl port. With the actual
source on disk, this note records which of our v2–v20 deviations are deliberate and
which were unintended.

## Where the reference code lives (inside the zip)

| Path in zip | Role |
|---|---|
| `trudi/src/jaxrl/reppo.py` | canonical JAX/nnx REPPO (the paper's own implementation) |
| `trudi/src/torchrl/reppo.py` | PyTorch port — the closest analogue to ours |
| `trudi/src/networks/torch_models.py` | reference `Actor` / `Critic` modules |
| `trudi/config/reppo.yaml` | reference hyperparameters |
| `trudi/src/torchrl/reppo_dime.py`, `trudi/src/jaxrl/reppo_dime.py` | REPPO + DIME diffusion policy — **this is the denoiser source C-TruDi M3 is required to read** |
| also | `reppo_analyse.py`, `reppo_eval.py`, `reppo_util.py` (`hl_gauss`, `EmpiricalNormalization`), `reppo_dime_rev_kl.py`, `crossq_reppo.py` |

Line references below are into the *torch* reference unless marked JAX.

## The math, and where each file implements it

Notation: `π_old` = policy at rollout collection, `π_θ` = policy being updated,
`Q` = the (distributional) action-value critic, `n` = action dimension, `α` = entropy
temperature (`temperature` / `alpha_temp`), `β` = KL multiplier (`lagrangian` /
`alpha_kl`), `ε` = `kl_bound` / `desired_kl`, `d_t` = terminated, `u_t` = truncated.

### 1. The optimization problem

REPPO is max-entropy policy improvement under a per-state trust region:

```
max_θ  E_{s~d}[ E_{a~π_θ(·|s)}[Q(s,a)] + α·H(π_θ(·|s)) ]
s.t.   E_s[ KL(π_old(·|s) ‖ π_θ(·|s)) ] ≤ ε
```

Its Lagrangian, with both multipliers held fixed inside the policy gradient
(stop-gradient, exactly as SAC treats its temperature):

```
L(θ) = E_s,ξ[ α·log π_θ(a_θ|s) − Q(s, a_θ) + β·KL_s(θ) ],    a_θ = a(s, ξ; θ)
```

The reward signal reaches θ **pathwise**: with `a_θ = tanh(μ_θ(s) + σ_θ(s)⊙ξ)`,
`ξ ~ N(0, I)`,

```
∇_θ E[Q(s, a_θ)] = E_ξ[ ∇_a Q(s,a)|_{a=a_θ} · ∇_θ a(s, ξ; θ) ]
```

so the critic is differentiated *through the action*, not scored by a likelihood ratio.
(Contrast C-TruDi, where the analogous gradient is deliberately avoided — see
[CLAUDE.md](../CLAUDE.md) rule 3. REPPO can afford it because its policy is a
one-step Gaussian, not a `T`-step denoising chain.)

**Actor loss, `clipped` mode** (the reference default) is a per-state switch, not a
soft penalty:

```
L_actor = E_s[ 1{KL_s < ε}·(α·log π_θ(a_θ|s) − Q(s,a_θ)) + 1{KL_s ≥ ε}·β·KL_s ]
```

so a state whose policy has already moved too far contributes *only* a restoring
gradient; its reward term is switched off until it re-enters the trust region.
`full` mode keeps both terms: `L = α·log π − Q + β·KL`.
— reference `torchrl/reppo.py:374-387`, ours `safe_rl/algorithms/reppo.py::_update_actor`.

### 2. The two duals (computed by gradient, unlike C-TruDi's)

Both multipliers are parameterized in log space (`α = e^ρ > 0`, `β = e^ω > 0`) and
updated by descending

```
L_α = α·( H̄ − H* ),        H* = target_entropy·n     (ours)
L_α = temperature·( c·n + H̄ ),   c = ent_target_mult  (reference)
L_β = −β·( K̄L − ε )
```

with `H̄`, `K̄L` detached. `∂L_α/∂α = H̄ − H*`, so α grows exactly when the policy is
less stochastic than the target; the two forms coincide under `H* = −c·n`, i.e. our
`target_entropy: -0.5` ≡ their `ent_target_mult: 0.5`. Likewise `∂L_β/∂β = −(K̄L − ε)`,
so β grows while the trust region is violated and decays while it is slack.

Note this is *not* the C-TruDi discipline — here the duals are learned by gradient
ascent with all the lag that implies, which is why our `alpha_kl_min` floor exists
(without it β decays to ~0 during slack stretches and the gate has no restoring force
left when the bound is next reached).

### 3. The λ-return target (`compute_gve` / `compute_returns`)

The soft-Bellman operator for a max-entropy Q is

```
Q^π(s,a) = E[ r + γ·( Q^π(s',a') − α·log π(a'|s') ) ],   a' ~ π(·|s')
```

Both codebases implement this by folding the entropy term into the **reward**, at full
weight, using a single next-state sample:

```
r̃_t = r_t − γ·α·log π(a'_t | s'_t)
```

and then running a Q(λ) recursion backwards over the rollout:

```
reference:  G_t = r̃_t + γ·[ u_t ? V'_t : (1−d_t)·( λ·G_{t+1} + (1−λ)·V'_t ) ]
ours:       G_t = r̃_t + γ·m_t·[ u_t ? V'_t : (1−λ)·V'_t + λ·G_{t+1} ],  m_t = max(1−d_t, u_t)
```

where `V'_t = Q(s'_t, a'_t)`. These are equal: for `u_t = 1`, `m_t = 1` and both give
`γ·V'_t`; for `u_t = 0`, `m_t = 1−d_t` and both give `γ(1−d_t)(…)`. Unrolled on a
non-terminating segment this is the familiar

```
G_t = Σ_{k≥0} (γλ)^k [ r̃_{t+k} + γ(1−λ)·Q(s_{t+k+1}, a_{t+k+1}) ]
```

Two things the recursion structure buys, both worth not breaking:

- **Entropy is counted once, at weight 1.** Putting `−α·log π(a')` inside the
  `(1−λ)`-blended bootstrap instead would discount the entire future entropy chain by
  `(1−λ)` (~5 % at λ = 0.95); adding `−α·log π(a_t|s_t)` on top of `r̃_t` double-counts it.
- **Truncation cuts the λ-trace** (`u_t ⇒ blend = V'_t`, not the blend). A timeout ends
  the episode, so `G_{t+1}` belongs to a *different* episode and must not leak backwards.

### 4. Distributional critic (HL-Gauss)

The scalar target `G_t` is embedded as a categorical distribution over `num_atoms` bin
centers `c_i ∈ [v_min, v_max]` by integrating a Gaussian of width `σ = 0.75·Δz`:

```
p_i(y) = [ Φ((b_{i+1} − y)/σ) − Φ((b_i − y)/σ) ] / Z,     Z = Φ((b_last − y)/σ) − Φ((b_0 − y)/σ)
L_critic = − Σ_i p_i(G_t) · log softmax(z(s,a))_i ,   masked by (1 − u_t)
Q(s,a)   = Σ_i softmax(z(s,a))_i · c_i
```

(Farebrother et al. 2024. Bin edges extend half a bin-width past the endpoints so that
extreme targets are smoothed symmetrically rather than truncated.) Ours: `_hlgauss_embed`
in `safe_rl/algorithms/reppo.py`; reference: `reppo_util.hl_gauss`.

### 5. Self-predictive auxiliary loss

With `f` = critic trunk features and `g` = the prediction head:

```
L_aux = E[ ‖ g(f(s,a)) − sg[ f(s', a') ] ‖² ],   masked by (1 − u_t)
L_critic_total = L_critic + aux_loss_mult · L_aux
```

The head `g` is what makes this *prediction*. Dropping it (what we did) collapses the
objective to `‖f(s,a) − sg[f(s',a')]‖²`, which pulls the representation toward its own
next-state value — a smoothness penalty on `f`, and hence on `dQ/da`, which is precisely
the quantity the pathwise actor gradient consumes.

### 6. Two math-level observations from the comparison

**(a) The 16-sample KL estimator is estimating a closed-form quantity.** KL is invariant
under a bijective reparameterization: if `T = tanh` and `π = T#p`, then
`log π(a) = log p(T⁻¹a) − log|det J_T(T⁻¹a)|`, and the Jacobian terms cancel in the
difference, so

```
KL( T#p_old ‖ T#p_new ) = KL( p_old ‖ p_new )
                        = Σ_i [ log(σ_n,i/σ_o,i) + (σ_o,i² + (μ_o,i − μ_n,i)²)/(2σ_n,i²) − ½ ]
```

The reference nevertheless estimates it with 16 samples from `π_old`
(`torchrl/reppo.py:367-372`), i.e. it pays `O(1/√16)` noise — injected directly into the
gate `1{KL_s < ε}` and into the dual's `K̄L` — for a quantity available exactly. Our
`squash: none` branch already computes the closed form; it could be used for
`squash: tanh` too. (Caveat: the reference clips samples to `±(1−1e-6)` before
`log_prob`, which makes its estimator very slightly not-the-KL; the closed form is not
bit-compatible with it, so this is an A/B, not a drop-in.)

**(b) The torch port has a terminal-step entropy bias that the JAX original does not.**
JAX multiplies the entropy bonus by the bootstrap factor
(`jaxrl/reppo.py:392-394`): `r̃_t = r_t − (1−d_t)·γ·α·log π(a'|s'_t)`. The torch port has
that masking commented out (`torchrl/reppo.py:184-187`), and we inherited the unmasked
form. On a *terminated* transition the target should be exactly `r_t` — no bootstrap, no
future entropy — but both the torch reference and we produce

```
G_t = r_t − γ·α·log π(a' | s'_t)          (d_t = 1, m_t = 0)
```

where `s'_t` is the post-reset observation and `a'` a sample from it. The bias is
`−γ·α·log π(a'|s'_t)` per terminal step: small when α is small, but it scales with the
termination rate, so it matters most on early-terminating tasks (Ant-Flat, Humanoid) and
least on fixed-horizon ones. The JAX form is the correct one.

## Verified equivalent — do not re-litigate

- **Soft λ-return.** Our `compute_returns` is algebraically identical to reference
  `compute_gve` (`torchrl/reppo.py:234-248`): λ-blend of the plain next-state Q,
  done-mask applied only to the blended branch, and the λ-trace cut on truncation.
- **Entropy in the reward.** `r' = r − γ·α·log π(a'|s')` at full weight, from a single
  next-state sample (`torchrl/reppo.py:186-187`). We match the torch reference — but see
  §6(b) above: matching it here means inheriting its terminal-step bias, which the JAX
  original does not have.
- **Critic loss.** HL-Gauss cross-entropy against the λ-target, masked by `(1−truncated)`.
- **Actor loss.** Pathwise `−Q(s, a_π) + α·log π`, with `clipped` / `full` matching the
  reference's `torch.where(kl < kl_bound, primary, β·kl)` (`torchrl/reppo.py:374-387`).
- **KL estimator.** Forward `KL(π_old‖π_new)`, 16 samples drawn from the old policy,
  summed over action dims, averaged over samples (`torchrl/reppo.py:367-372`). Matches
  ours whenever `squash: tanh`.
- **Both duals.** Reference `entropy_loss = (target_entropy + entropy)·temperature` and
  `lagrangian_loss = −β·(kl − bound)` have the same stationary points as our
  `α_temp·(H − H_target)` and `α_kl·(bound − KL)`. Our `target_entropy: -0.5` is the
  reference's `ent_target_mult: 0.5`.
- **Old-policy semantics.** Reference hard-copies `actor → old_actor` at the end of each
  iteration (`torchrl/reppo.py:1253`); our stored rollout `(mu, sigma)` is that same
  distribution.
- **Update loop.** 4 epochs × 128 minibatches, critic then actor on the *same* minibatch.
- **No target networks.** Reference bootstraps from the online actor and critic; v20 sets
  `use_target_networks: false`.
- **`zero_init_prior`** plays the role of the reference's `logits + 40.9·hl_gauss(0)`
  (`networks/torch_models.py:235`); our additive `min_std` matches `std = exp(log_std) + min_std`.

## Gaps that are real bugs

Fixes for 1–3 are applied in the working tree (uncommitted, 2026-07-30); 4 is a finding
only.

1. **Observation-normalizer drift corrupted the KL.** The reference stores *normalized*
   obs in the rollout buffer (`torchrl/reppo.py:204-206`), so `π_old` and `π_new` are
   compared on identical inputs. We stored raw obs and re-normalized at update time,
   while the stored `old_mu/old_sigma` had been produced under per-step-evolving
   statistics (`EmpiricalNormalization.forward` updates on every training-mode call).
   Freezing the normalizer in `compute_returns` did not close this — the mismatch is
   between collection time and update time, so the trust region was being charged for
   input drift. Worst early in training, when the statistics move fastest; a plausible
   contributor to the "KL pinned at the bound" pathology in the v10/v11 config notes.
   *Fix:* REPPO now normalizes once in `act()` / `process_env_step` and stores the
   normalized tensors; the whole update path runs with `normalized=True`.
2. **Aux loss had no predictor head.** The reference critic carries a `pred_module` and
   regresses `pred_module(features(s,a))` onto `sg[features(s',a')]`
   (`networks/torch_models.py:231-238`, `torchrl/reppo.py:315-322`). We regressed the
   trunk features straight onto the next-state features — a collapse-prone objective
   that smooths the critic's own representation instead of predicting, which is what the
   v20 comment suspected when it ablated aux. *Fix:* `DistributionalCritic` gained an
   optional `aux_predictor` head; the online side goes through it, targets stay raw
   features. The JAX version additionally predicts the reward (`jaxrl/reppo.py:518`) —
   not ported.
3. **Minibatch partition was not reshuffled per epoch.** Our `randperm` sat outside the
   epoch loop, so all 4 epochs reused the same 128 partitions; the reference re-permutes
   every epoch (`torchrl/reppo.py:1239-1243`). *Fix:* moved inside the loop.
4. **Terminal-step entropy bias**, inherited from the torch port — the entropy bonus is
   not masked by `(1−d_t)`, so terminated transitions get a target of
   `r_t − γ·α·log π(a'|s'_t)` instead of `r_t`. Derivation and magnitude in §6(b) above.
   Not changed: it would make us differ from the torch reference we are calibrating
   against. Worth an ablation on Ant-Flat, where termination is frequent.

## Deliberate deviations — keep, don't "restore"

- SimBa/SimbaV2 trunks vs. the reference's FCNN + RMSNorm + swish (actor 3 layers,
  critic 2/2/2 encoder/head/pred, hidden 512).
- Twin critics with `min` reduction (reference: a single critic). Note our
  `_update_critic` *sums* the two CE terms, i.e. ~2× the reference's effective critic
  gradient; `num_critics: 1` recovers the reference behaviour exactly.
- Bounded `log_std` (`log_std_squash: sigmoid`) vs. the reference's unbounded
  `exp(log_std)`. Adopted after raw-Gaussian sigma ratcheting (v9).
- `alpha_kl_min` floor on the KL dual — the reference has none; we added it after the
  exponential parameterization decayed `alpha_kl` to ~0 during low-KL stretches (v10/v11).
- Optimizers: ours is `AdamW(wd=1e-3, betas=(0.9, 0.95))` with a separate
  `alpha_optimizer` at `alpha_lr`; the reference uses plain `Adam(3e-4)` with `log_temp` /
  `log_lagrange` living *inside* the actor and sharing the actor's optimizer and lr
  (`torchrl/reppo.py:970-977`, `networks/torch_models.py:266-271`).
- `reward_scale` / `RewardNormalization` — the reference scales rewards at the env level.
- Hyperparameter drift in v20 vs. `config/reppo.yaml`: `desired_kl 0.2` (ref `kl_bound
  0.1`), `init_alpha_temp 0.001` (ref `ent_start 1.0`), `learning_rate 1e-3` (ref `3e-4`),
  `aux_loss_mult 0` (ref `1.0`).
- Ours only: optional target networks + polyak, closed-form Gaussian KL for
  `squash: none`, `q_bias` / `frac_targets_clipped` diagnostics.
- Reference only, not ported (both off by default in `config/reppo.yaml`): `reverse_kl`,
  and the JAX per-env exploration-noise scaling with importance-weighted λ-returns
  (`exploration_noise_max/min`, `lmbda_min`, `jaxrl/reppo.py:348-383,443-460`).
- Last-step handling: the reference forces `truncated[-1] = 1.0`, which both bootstraps
  the final step *and* (because the same tensor feeds the critic mask) drops it from the
  critic loss. We keep the real flag and train on that step.

## Follow-ups

- `config/mjlab_ant_reppo_v21.yaml` = v20 + the three fixes, with `aux_loss_mult: 1.0`
  re-enabled now that the predictor head exists. A/B against v20 before concluding.
- For C-TruDi M3, read `trudi/src/torchrl/reppo_dime.py` (and its JAX twin) rather than
  inventing a denoiser API — see [CLAUDE.md](../CLAUDE.md) M3.
