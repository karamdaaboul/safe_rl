# Stochastic decision horizons on the FH-DCMPO stack (VT-MPO + hybrid)

*2026-08-20 (Claude Code). Imports the modelling device of "Stochastic Decision Horizons for
Constrained Reinforcement Learning" (Milosevic, Franz, Haeufle, Martius, Scherf, Kolev;
arXiv:2602.04599v1, 4 Feb 2026) into this repo, as two arms. **Status: implemented and smoke-tested,
NOT yet evaluated.** No arm has reached an operating point, so there is no evidence here that it
helps — only that it runs and that the shaping is genuinely active.*

Related: [[fh-cost-target-and-duals]] (the measurements that motivated this),
[[cvpo-cost-critic-investigation]], [[cvpo-per-state-lambda]], [[cvpo-negative-result]].

---

## Why this paper, given our measurements

Our diagnosis across ~15 arms was not "the dual is mistuned" — it was that **there is almost
nothing for the dual to steer with**:

| measurement | value |
|---|---|
| `std_a(Q_c)` across the 64 candidate actions at a fixed state | **0.03–0.16** |
| `Q_c` level | ~16 (so the spread is **under 1%**) |
| `std_a(Q_r)` | 0.13 |
| true action effect on 900-step cost-to-go (16 actions, real sim) | **indistinguishable from rollout noise** |

The E-step softmax `exp((Q_r − λQ_c)/η)` is normalised per state, so only the across-action spread
reaches the policy. An additive `−λQ_c` term with <1% spread cannot move it, whatever λ does. That
is why the per-state homotopy stalled with a KKT residual of 1e-5 (a7: `d_b` median 52 against a
floor of 25, cost flat at 32–47 for 15k iterations) — the solver was correct and the signal was
absent.

**SDH changes the arithmetic.** A continuation probability `α(s,a) ∈ [0,1]` shapes reward and
discount

```
α(s,a) = exp(−λ_sdh · Σᵢ cᵢ(s,a)),   r̃ = α r,   γ̃ = γ α
```

so the cost becomes a **multiplicative attenuation of the entire future return** rather than a small
additive term, and it **compounds** along the horizon through `u_t = γ^t Πα`. A 1% per-step
difference in α scales all of `Q_surv` (~20) instead of adding ~0.1. Constraints enter **only
through the critic's Bellman target**; the MPO E-step and M-step are untouched and there is **no
Lagrange multiplier at all**. The variable-discount operator stays a contraction with modulus
`sup γ̃ ≤ γ`, so replay and target networks are unaffected.

**What it costs us.** SDH is *not* a CMDP: no budget of 25, no feasibility certificate. The
operating point is set by `λ_sdh`, which must be swept, and arms must be compared **at matched
realized cost**. Our branching probe also showed the *environment* barely separates actions by
cost, which bounds any method — so this is a promising mechanism, not a guaranteed fix.

---

## The two arms, and why both

`VTMPO` changes two things at once relative to our arms: it adds shaping **and** removes the dual.
If it wins we would not know which half did the work. So:

| arm | class | dual / budget | shaping | comparable on our metrics? |
|---|---|---|---|---|
| `VTMPO` | `VTMPO(MPO)` | **none** | reward critic | only at matched realized cost |
| hybrid | `FHDCMPOSurv(FHDCMPO)` | **unchanged** (η, λ, FH cost critic, budget 25) | reward critic | yes, directly |

The hybrid is one factor off `a1_lag_base`, so violation rate / CVaR₀.₉ / VaR coverage against the
limit of 25 stay directly comparable with every earlier arm.

---

## Implementation

Almost all of it is two existing mechanisms, not new machinery.

**`safe_rl/common/continuation.py`** — `exponential_continuation` (the paper's Safety-Gymnasium
mapping), `cat_continuation` (its normalised/saturating variant, for later multi-constraint work),
and `continuation_scale_at`, a linear schedule deliberately mirroring `fh_cost.kappa_at`.

**`safe_rl/storage/replay_storage.py::_gather_n_step`** — the survival-shaped n-step return, gated
on a mutable `survival_lambda` (`None` ⇒ the old path, byte-identical):

```python
alpha       = exponential_continuation(all_costs, self.survival_lambda)   # [B, n]
gamma_tilde = self.gamma * alpha
ubar = torch.cumprod(gamma_tilde, dim=-1)                       # prod_{j<=k}
u    = cat([ones, ubar[..., :-1]], -1)                          # prod_{j<k}, u_0 = 1
n_step_rewards    = (all_rewards * done_masks * u * alpha).sum(-1, keepdim=True)
survival_discount = ubar.gather(1, first_done.unsqueeze(-1))    # [B, 1]
```

Two deliberate choices:

* **Computed at sample time, unlike the paper.** The paper precomputes `R⁽ⁿ⁾` and `u_{t+n}` at
  rollout time and stores them. With a *scheduled* `λ_sdh` and a 1M-transition buffer, every
  replayed sample would then carry a stale scale. Reading the stored per-step costs at sample time
  uses the live one. Do not "optimise" this back into the rollout path.
* **`[B, 1]`, never `[B]`.** The standard-critic target multiplies `[batch, 1]` tensors; a `[batch]`
  discount broadcasts to `[batch, batch]` instead of failing.

**`safe_rl/algorithms/sac.py`** — `survival_discount` threaded exactly like the existing `bootstrap`
and `effective_n_steps` kwargs, default `None`. `_bootstrap_discount` returns it when present; note
it **replaces** `γ**n` rather than multiplying it, because the γ factors are already inside `u`.
`MPO._update_critic_distributional` (the only reward-critic override in the tree) forwards it too.

**`VTMPO`** additionally opts into the runner's cost plumbing (`num_costs`, a `store_transition`
override) because it needs the cost *signal* while having no cost *critic*, and discards the
runner's `current_costs` since it has no multiplier to update.

---

## Verification so far

* **The anchor:** at `λ_sdh = 0` the shaped return and the shaped discount equal the scalar-γ ones
  **exactly** (max |diff| 0.0, and `survival_discount == γ**n_eff`). Without this, arms trained
  before and after this feature would not be comparable and nothing else would notice.
* `survival_lambda = None` leaves `_gather_n_step`'s output byte-identical, keys included.
* A hand-rolled Python reference with the episode ending **inside** the window reproduces `R⁽ⁿ⁾`,
  `effective_n_steps` and `u_{t+n}`.
* Shaping without stored costs raises rather than silently running with α ≡ 1.
* 12 tests in `tests/test_survival_horizons.py`; the pre-existing per-sample-discount locks
  (`test_mpo.py`, `test_quantile_critic.py`, `test_cvpo.py` — 79 tests) still pass unchanged.
* **Smoke, 200 iterations on GPU, both arms** — shaping demonstrably active rather than a no-op:

| arm | `survival_discount` mean | γ¹⁰ (unshaped) | α mean | α min |
|---|---|---|---|---|
| VT-MPO (λ=0.3) | 0.861 | 0.904 | 0.952 | 0.057 |
| hybrid (λ=0.3) | 0.883 | 0.904 | 0.977 | 0.050 |

The silent failure of these arms is α ≡ 1 — a healthy-looking run that is secretly unshaped MPO —
so `sdh_alpha_mean` / `sdh_alpha_min` / `sdh_survival_discount_mean` are logged every update.

---

## Next: what would make this a result

1. **λ_sdh sweep** — fixed scales 0.1 / 0.3 / 0.6 / 1.0, 10k iters each, pick the value whose mean
   cost lands nearest 25.
2. **Both arms at that scale**, 20k iters, seed 2, evaluated with the corrected protocol
   (`--num_envs 1`, 200 episodes; `probe_fh_calibration.py` at 10k/15k/20k).
3. Compare against a1 (base), a2 (4:1) and FH-DCMPO **at matched cost**.

**The prediction, stated before the runs:** the across-action spread of the E-step score should rise
well above the reward-only ~0.13, and no per-state ask can stall because there is no ask. If cost
does **not** fall at matched reward, the multiplicative-signal hypothesis is wrong too, and that is
a real negative result about this environment rather than about the method.
