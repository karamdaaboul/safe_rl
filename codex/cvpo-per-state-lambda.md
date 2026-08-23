# CVPO with a global eta and a per-state lambda (M5)

Status: **implemented, exercised, not yet evaluated.** Solver, algorithm, dispersion gate and
amortized head are in, unit-tested (85 tests), and run end to end on `SafetyPointGoal1-v0` for 30
iterations across five arms. No arm has been trained to a reward/cost operating point, so there is
**no evidence yet that per-state λ helps** — only that it is correct, cheap, and does what the
design says. Numbers below are analytic, or measured on synthetic data carrying this repo's own
critic statistics, or from those 30-iteration runs; none are from a converged policy.

Two things surfaced only by running it, not by unit tests, and both changed the design:
the reachability-relative target (finding 3) and the KKT-diagnostic scoping bug under the gate
(see *Phase 2*).

## Why

[[cvpo-cost-critic-investigation]] closes with: *"What remains unsolved is the critic's inability
to discriminate states (slope 0.17 against 1.0); the constraint is currently enforced by a roughly
state-independent penalty. Whether per-state discrimination would buy anything is the natural next
question."* This is that question, and it is CLAUDE.md's milestone M5.

`eta` is a trust-region size and genuinely wants to be uniform across the update -- it regularises
the whole E-step. `lambda` is a tradeoff *rate* between two value scales and legitimately varies by
state. CVPO's authors ruled out a per-state dual because it meant leaving the vectorized path. It
does not: with `eta` fixed, each state is 1-D root-finding in a monotone function, so the whole
batch is ~30 bisection steps on device. No SciPy, no per-state loop, no learning rate on `lambda`,
no windup.

## Math

    q_b(a_k) ∝ exp((Q_r(s_b,a_k) − λ_b·Q_c(s_b,a_k)) / η)

    g(η,{λ_b}) = ηε + mean_b[ λ_b d_b + η·log (1/K) Σ_k exp((Q_r − λ_b Q_c)/η) ]

    ∂g/∂λ_b = d_b − E_{q_b}[Q_c]      ∂²g/∂λ_b² = Var_{q_b}(Q_c)/η ≥ 0
    ∂g/∂η   = ε   − mean_b KL(q_b‖π_old,b)

All four verified numerically against finite differences. `E_{q_b}[Q_c]` is non-increasing in λ_b,
`mean_b KL` non-increasing in η, so both blocks bisect exactly. Bisection on `[0, λ_max]` *is* the
KKT box projection, so the hot loop needs no data-dependent control flow and never synchronises.

`∂g/∂η = ε − KL` also means **η needs no SciPy either** — the whole solve is torch.

## Three findings that shaped the design

**1. A static per-state target `d_b = qc_thres` degenerates — and the solver is not at fault.**
Block-coordinate descent walks to η ≈ 169, λ_median ≈ 66, 26 % of states at λ_max. `g` decreased
monotonically at every one of 25 sweeps (−63.80 → −71.98) and converged. It is *correct*: for a
state whose sampled actions cannot reach `d_b`, `∂g/∂λ_b < 0` for every λ, so `λ_b → λ_max` really
is optimal; the induced `−λ_max·Q_c` spread is enormous; and the η that truly minimises `g` really
is large. Because η is shared, those states then contaminate every feasible state's solve.

> Consequence: **no solver-side fix is legitimate.** Damping η, clipping the sweep count or adding
> momentum would all be wrong and would violate "solved to optimality". The fix belongs to the
> problem.

**2. On the measured critic, a static target is hopeless.** Simulating this note's sibling's
statistics (Q_c level 3.12, per-state std *across actions* 0.038), a static target pins λ at the cap
at essentially every state and the constraint is **still violated by +1.28 at p90**. Saturation buys
nothing. A per-state reachable target `d_b = max(q_target, C_now_b − ask_b)` brings it to +0.002
with zero saturation.

**3. The ask must be sized against *reachability*, and nothing else transfers.** The reduction a
multiplier capped at λ_max can deliver is bounded, so asking for more makes every state infeasible
— finding 1 again, through a different door. Three parameterizations, three regimes (this repo's
critic statistics; a live 30-iteration early-training run; a large-action-spread fixture):

| ask | fixture A | live regime B | spread regime C |
|---|---|---|---|
| β = 0.01 of the level (0.82 σ) | λ 3.00, **98 % at cap** | — | — |
| κ = 0.25 σ | λ 0.36, ok | λ 2.65, **33 % at cap** | λ 1.56, ok |
| ρ = 0.50 of reachable | λ 0.30, ok | λ 2.25, **13 % at cap** | λ 1.30, ok |
| **ρ = 0.25 of reachable** | **λ 0.14, ok** | **λ 1.00, ok** | **λ 0.56, ok** |

A level-relative ask has to be re-derived whenever the critic's level moves — already paid for in
[[cvpo-qc-threshold-calibration]]. A σ-relative ask survives that but not much else: **the
KL-reachable drop `(C_now − reachable_qc_min)` measured 0.44 σ in all three regimes**, so it is set
by `eps`, not by the critic — meaning κ = 0.25 σ is silently asking for 57 % of reachability and
sits near the edge.

So the default is `dstate_beta_mode: reachable`, `ask_b = ρ·(C_now_b − reachable_qc_min_b)` with
ρ = 0.25. That fraction is attainable by construction, so it **cannot be mis-sized** and needs no
knowledge of the critic's level or spread at all. ρ must stay well under 1: the floor is what the
KL budget alone permits, but reaching it also needs λ → ∞, so λ_max binds first. `reachable_qc_min`
is the same bisection kernel with `Q_r = 0`, and it is shared with the feasibility probe.

Watch `dstate_ask_over_spread`: **negative means the floor is binding** — the policy is already
inside the budget and nothing is being asked of it — not that the ask is inverted.

## Deliberate differences from the scalar homotopy

A different batch of states arrives every update, so there is no per-state identity to carry state
on. Hence, unlike [[cvpo-feasible-threshold-homotopy]]:

* **No ratchet.** A per-state one is undefined; a scalar one applied uniformly is just a constant
  multiplier on `C_now_b`. The monotone pressure comes from the floor (`d_b ≥ q_target`, with
  `qc_target_max_rise` capping the floor's drift). `frac_states_at_floor → 1` is the observable that
  replaces it — the real budget now binds everywhere.
* **No `C_now` smoothing.** It existed to stop one noisy batch locking a persistent ratchet. With no
  ratchet there is nothing to lock, and `C_now_b` is a K = 64 mean of a quantity with std 0.038,
  i.e. precise to ~0.005 against a level of 3.12.

## λ_max

Sized from the E-step spread ratio, not the value scale — only the spread across candidate actions
survives the per-state softmax. `lambda_balanced` reads **1.86–2.35** here and a constant λ = 2.0
lands exactly on budget (reward 19.9, cost 25.3). λ = 4.0 is already where the pathology returns
(cost 34.74 vs λ = 3.0's 14.26, spread ratio collapsing to 0.02). **Default λ_max = 3.0, not 4.**

## What is in the tree

| | |
|---|---|
| `safe_rl/common/per_state_dual.py` | pure-torch batched solver; also a per-state `reachable_qc_min` |
| `safe_rl/algorithms/cvpo_per_state.py` | `CVPOPerState(CVPO)`; overrides `_estep_weights` only |
| `safe_rl/modules/lambda_head.py` | amortized `λ_ψ(s)` (3-way regime class + interior regression) |
| `config/safety_gymnasium_cvpo_perstate.yaml` | main arm |
| `config/safety_gymnasium_cvpo_perstate_static.yaml` | the documented failure arm |
| `tests/test_per_state_dual.py` (36) | M1 gates, per-state |
| `tests/test_cvpo_per_state.py` (49) | wiring, config validation, gate, head |

End to end on `SafetyPointGoal1-v0`, 30 iterations, `qc_thres` forced to 0.6 so the constraint
binds. Every arm: KKT interior residual ≤ 2e-7, solver-vs-MPO η residual 1.5e-9, **one** host sync
per E-step.

| arm | η | λ p10/p50/p90 | frac at cap | viol vs q_target p90 |
|---|---|---|---|---|
| reachable (default) | 0.159 | 0/0/0 | 0.00 | −0.064 (floor binding, constraint slack) |
| spread κ=0.25 | 0.099 | 0/0/0.04 | 0.09 | +1.878 |
| gated (spread, q=0.5) | 0.106 | 0/0/0 | 0.00 | −0.091 |
| head (observer) | 0.118 | 0/0/0 | 0.00 | −0.038 |
| **static arm** | 0.124 | **3.0/3.0/3.0** | **1.00** | **+0.114** |

The static arm reproduces finding 1 in a live training loop, not just synthetically: λ pinned at
the cap at *every* state, and still violating.

Cost: 12 ms (1 sweep) / 21 ms (2 sweeps) at B=256 K=64, CPU float32, 2 threads, against **7.8 ms**
for the SLSQP joint dual it replaces — and it removes the `.cpu().numpy()` sync CVPO pays per
E-step. Warm-started, one sweep reproduces the fixed point to 0.0e+00; from cold the descent needs
~16 sweeps, so a run's opening updates settle over ~16 *updates*.

## Phase 2 — dispersion gate

Where `std_k(Q_c)` is below the critic's noise floor the per-state root sits on a flat curve, so
λ_b is arbitrary and only adds dual variance to the M-step. Those states take a batch-level λ from
`solve_shared_lambda`. Thresholds are **batch-relative quantiles**, never absolute Q_c values.

`std_k(Q_c)` (spread across actions: "does λ_b have signal?") and `DistributionalCritic.get_var`
(the return distribution's own spread at one (s,a): "is the critic confident here?") are different
quantities answering different questions. The gate that decides identifiability is the first; the
second is a complementary second gate (`both`), not a substitute.

Gate per M5: worst-state violation must fall **without** ESS collapse. If it does not, reverting to
a shared λ is a legitimate result (CVPO's authors reached it first), not a failure.

**Diagnostic scoping, found by running it.** `dual_residual_lambda_interior_absmax` must be
restricted to *gated* interior states. An ungated state carries the batch-level λ, which satisfies
the batch constraint and has no reason to satisfy that state's own stationarity — counting it made
the headline KKT residual jump from 0 to ~1e-1 the moment the gate was switched on, reading as a
broken solver rather than as the gate working. Locked by
`test_kkt_residual_excludes_ungated_states`.

## Phase 3 — amortized λ_ψ(s)

The one carve-out to "duals are computed, never learned". The head has an optimizer *because its
loss is a supervised regression onto KKT targets the bisection already produced exactly*; it never
descends `∂g/∂λ`. Its step runs outside the E-step's `no_grad` block, and no gradient reaches Q_c.

3-way regime class (inactive / interior / infeasible) plus interior-only regression, because the
KKT solution has real mass on both corners where `log λ` is undefined or uninformative. Default
mode `observer`: it trains while the exact solve still drives the policy, so its error is measured
before it is trusted. Payoff is not compute (the E-step already evaluates all K cost values) but
generalisation across states, and `λ(s)` available at deployment as a boundary-proximity signal for
the M6 filter.

## Known gaps

* **m = 1 only.** Bisection does not generalise to multiple constraints (CLAUDE.md rule 5 wants
  m ≥ 1 from the start). Not papered over with an `[m,K,B]` signature; the m > 1 path is a batched
  projected Newton step on the m-dimensional convex subproblem.
* **Per-state feasibility ≠ budget satisfaction.** `d_b = max(q_target, C_now_b − ask_b)` guarantees
  monotone descent toward `q_target`, not `mean_b d_b ≤ qc_thres` at any given update — the same
  property the scalar homotopy has.
* **A shared η still couples the batch.** Phase 1 removes the *pathological* coupling by making
  every state feasible, not the coupling itself. `kl_dispersion_ratio` is the instrument;
  `per_state_eta: true` is the escape hatch (the same kernel with the reduction dropped).
* **`tests/test_smoke_regression.py` is currently red in this tree**, from the concurrent session's
  uncommitted `cvpo.py`/`mpo.py`/`replay_storage.py` changes, not from this work — verified by
  re-running it with `CVPOPerState` unimported.

## Next

Run the main arm and the static arm on `SafetyPointGoal1-v0`, ≥5 seeds. Watch
`frac_infeasible_lambda_cap` (≈0), `dstate_ask_over_spread` (≈0.25), `kl_dispersion_ratio`,
`ess_p10`, `dual_residual_lambda_interior_absmax` (≈0), and `viol_vs_qtarget_p90` — *not*
`viol_vs_dstate_p90`, which is ≈0 by construction once the target is reachable and only detects
solver failure. Reference points: FSRL CVPO reward 20.6 at cost 27.1 ([[m0-baselines]]); constant
λ = 2.0 at reward 19.9 / cost 25.3.

Cross-refs: [[cvpo-cost-critic-investigation]] (the question this answers),
[[cvpo-negative-result]], [[cvpo-feasible-threshold-homotopy]], [[mpo-estep-mstep-math]].
