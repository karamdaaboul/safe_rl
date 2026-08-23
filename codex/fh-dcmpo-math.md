# FH-DCMPO — the math, the bound, and a prediction registered before the first run

*2026-08-17. Companion to [[qr-dmpo-math]] (which this builds directly on),
[[why-mean-beat-cvar-on-pointgoal1]] and [[cvpo-negative-result]] (the two negative results this
design exists to answer). The MPO E-step/M-step derivation is not repeated — see
[[mpo-estep-mstep-math]]. Every formula below is traceable to a file:line.*

Config: `config/safety_gymnasium_fhdcmpo_goal1.yaml`, `..._cargoal1.yaml`,
`..._fhdcmpo_s1_mean_goal1.yaml`. Code: `safe_rl/algorithms/fhdcmpo.py`,
`safe_rl/common/fh_cost.py`, `safe_rl/envs/horizon_augmented_vec_env.py`.

---

## 0. The problem in one table

From [[qr-dmpo-math]] §6.1, 50-episode deterministic eval, budget 25:

| method | task | seeds | reward | cost | CVaR₀.₉ | **episodes over budget** |
|---|---|---|---|---|---|---|
| QR-DMPO | PointGoal1 | 3 | 23.84 | **18.97** | 56.9 | **32.0%** |
| DMPO (C51) | PointGoal1 | 3 | **25.70** | 24.68 | 71.5 | 43.3% |
| CVPO (scalar) | PointGoal1 | 3 | 23.31 | 26.73 | 73.1 | 43.3% |
| PPOL-PID | PointGoal1 | 3 | 21.65 | 26.50 | 76.7 | 40.7% |
| QR-DMPO | CarGoal1 | 1 | 32.30 | 28.68 | 121.2 | 34.0% |
| DMPO | CarGoal1 | 1 | 33.24 | 24.84 | 79.4 | 46.0% |

Every method violates in 32–46% of episodes while several report a compliant mean. The mean is not
a safety statistic. **Reducing that fraction, at unchanged reward, is the only objective here.**

## 1. Why the two obvious fixes already failed, and what survives of them

**Fix A — constrain a tail instead of the mean.** Tried; it lost badly
([[why-mean-beat-cvar-on-pointgoal1]]): reward 12.3 → 7.1 → 2.3 going mean → CVaR₀.₅ → CVaR₀.₉.
The diagnosis is the useful part, and it is three separate defects:

1. *Level mismatch.* Measured on a trained agent against `qc_thres = 2.50`: `E[Z_c] = 4.45`
   (1.78×), `CVaR₀.₅ = 6.89` (2.75×), `CVaR₀.₉ = 11.19` (**4.48×**). A tail mean sits above the
   mean *by construction*, so reusing a mean-calibrated threshold turns "be risk-aware" into "be
   4.5× more constrained". λ integrated to `λ_max` and stayed there for 13k of 15k iterations,
   which makes `exp((Q_r − λQ_c)/η)` stop depending on `Q_r` at all.
2. *Bad exchange rate.* Only the **spread across candidate actions at a state** survives the
   per-state softmax — a constant added to `Q_c` cancels. Going to CVaR₀.₉ multiplied the
   violation 2.5× to buy 1.60× in action-spread (0.0170 → 0.0273). You pay in constraint
   pressure, which is what damages reward, to buy discrimination.
3. *The tail was mis-estimated anyway.* Predicted std 3.21 vs realized 5.55 (2.1× too narrow),
   PIT KS 0.381 against a 0.008 critical value, `VaR₀.₉` covering 0.858 of realized returns. So
   the CVaR understated tail risk by 25–30% while charging the full price of a strict constraint.

**Fix B — solve the joint dual exactly.** Also tried; λ went bang-bang between 1e-6 and 1e5
([[cvpo-negative-result]]). That is *correct solver behaviour*: `∂g/∂λ = d − E_q[Q_c]`, so if no
reweighting of the sampled actions can reach `E_q[Q_c] = d`, the minimiser has no interior
stationary point and runs to a bound. Net constraint pressure ≈ 0.

**What survives.** Defect 2 is a real and permanent cost of tail statistics — it is not fixable,
only worth paying or not. Defects 1 and 3, and Fix B's infeasibility, all trace to the same root:
**the constraint is enforced in a different space from the one it is stated in.**

## 2. The root cause: `qc_scale`

The budget is episodic and undiscounted, `J_c = Σ_{t=0}^{T-1} c_t ≤ 25`. The critic learns
discounted cost-to-go `G_c = Σ_t γ^t c_t`. The bridge is one scalar
(`safe_rl/common/cost_scaling.py`, `measured_qc_scale`):

$$\texttt{qc\_scale}=\frac{\mathbb{E}[G_c(s_0)]}{\mathbb{E}[J_c]},\qquad d = 25\times\texttt{qc\_scale}$$

| task | measured `qc_scale` | `d` at limit 25 |
|---|---|---|
| SafetyPointGoal1 | 0.0764 | 1.91 |
| SafetyCarGoal1 | 0.09198 | 2.30 |
| analytic (uniform cost rate) | 0.100 | 2.50 |

Three things are wrong with this, and they compound:

- It is **per-task and measured**, so it is a calibration step that can be silently wrong. Running
  CarGoal1 with PointGoal1's 0.0764 enforces `d = 1.91` instead of 2.30 — a 20% tighter budget
  than the stated limit ([[qr-dmpo-math]] §3). The analytic value is 31% loose on PointGoal1.
- It is a **ratio of means**, so it maps `E[J_c]` to `E[G_c]` and *nothing else*. There is no
  scalar that maps a quantile of `J_c` to a quantile of `G_c`: discounting is a non-linear
  reweighting of the cost's position in time, so it changes the shape of the distribution, not
  just its scale. **A tail constraint in discounted space is not a tail constraint on the episodic
  cost, at any threshold.** This is the formal reason defect 1 is not a tuning problem.
- CGPO (arXiv:2412.11138 §3.2) makes the general version of the point: the Kakade–Langford identity
  behind every advantage-based surrogate requires `γ ≠ 1`, so finite-horizon undiscounted
  constraints are outside its scope, and "aforementioned deep Safe RL algorithms treat these
  constraints as if they were infinite-horizon… This leads to poor constraint satisfaction within
  these benchmarks." Its own remedy needs a differentiable simulator, which Safety-Gymnasium is
  not, so we take the diagnosis and not the method.

## 3. The object FH-DCMPO learns

$$Z_c^{FH}(s_t,u_t,a_t)\ \sim\ \text{distribution of}\ \sum_{t'=t}^{T-1}c_{t'},\qquad T=1000,\ \gamma_c=1$$

Undiscounted, finite-horizon. With `γ_c = 1` this is not a function of `(s,a)` alone, so the
critic input gains the normalized remaining horizon `u_t = (T−t)/T`
(`fh_cost.normalized_remaining_horizon`), the same device Sauté RL (ICML 2022) uses for the
remaining budget. `u = 0` is the boundary condition that pins the cost-to-go to zero.

**Two independent discounts have to be neutralised**, and missing either leaves the target
discounted no matter what the other says:

1. the bootstrap discount, `_cost_bootstrap_discount → 1.0` (`fhdcmpo.py`), a new cost-channel
   hook so `SAC._bootstrap_discount` still serves the reward channel at `γ = 0.99`;
2. the **n-step window sum** inside the buffer, which reused the reward's `γ^k` vector
   (`replay_storage.py:528,545`) — now `ReplayStorage(cost_gamma=1.0)`.

Both are asserted in `tests/test_fhdcmpo.py::test_both_cost_discounts_are_neutralised`.

Backup, per quantile location, with `γ_c^n = 1`:

$$(\mathcal{T}\theta')_j=\sum_{k=0}^{n-1}c_{t+k}+m\cdot\theta'_j(s_{t+n},u_{t+n},a'),\qquad a'\sim\pi$$

`m = bootstrap + (1 − done)` is the existing truncation-aware mask. **`n_step = 10` is kept and
matters more here than before:** with `γ_c = 1` there is no contraction anywhere in the recursion,
so value information propagates only from the `u = 0` boundary. At `n = 1` that is 1000 backups
deep; `n = 10` makes it 100.

**Quantile head only.** The categorical head hard-clips its support at `v_max` (50.0 by default,
`safe_actor_critic.py:136`) and undiscounted episodic cost runs well past any fixed upper edge.
`QuantileCritic` has no support bound. `FHDCMPO.__init__` raises rather than allowing the other.

## 4. The constraint statistic

$$\rho_{\kappa,\alpha}[Z]=\mathbb{E}[Z]+\kappa\big(\mathrm{CVaR}_\alpha[Z]-\mathbb{E}[Z]\big),\qquad \kappa\in[0,1]$$

`fh_cost.conservatism_statistic`. On `N` equal-mass sorted locations, `CVaR_α` takes exactly
`1−α` of mass from the top, splitting the boundary location (without the split, `α = 0.9` at
`N = 64` silently means `6/64 = 0.094` or `7/64 = 0.109`). The weights depend only on `(α, N)`,
never on `θ`, so **ρ is linear in the learned locations** — which is what makes it safe inside the
E-step exponent: the gradient reaches every location in the tail, undistorted, and none outside it.

Three properties, all unit-tested:
- `κ = 0` reproduces the mean constraint **bit-exactly** (`torch.equal`, not `allclose`) — that is
  what lets the S1 arm isolate the finite-horizon change from the risk change;
- `κ = 1` is pure `CVaR_α`;
- ρ is monotone non-decreasing in `κ`, since `CVaR_α ≥ E`.

**κ is ramped 0 → target** (`fh_cost.kappa_at`; warmup 5000 updates, ramp 15000). This is the
direct answer to defect 1: the constraint starts satisfiable and tightens, rather than starting
4.48× violated with λ pinned. The warmup exists for defect 3 — a tail read off an untrained,
under-dispersed critic is worse than no tail at all.

An EVT alternative is implemented behind `fh_risk_mode: evt`: EVO's extreme-quantile constraint
(arXiv:2601.12008 Eq. 15), a GPD fitted by MLE to the pooled peaks over a safety boundary,
$q_\mu + \frac{\sigma}{\xi}\big((1-\nu n/N_\mu)^{-\xi}-1\big)$. Structurally different from CVaR:
its tail correction is a **single global scalar** added to each state's mean, not a per-state tail.
Offered as an alternative, not the default.

## 5. The threshold and the dual

$$d = \texttt{cost\_limit} = 25\ \text{exactly.}$$

No `qc_scale` (`qc_thres: 25.0` set explicitly in the config; `FHDCMPO` refuses a `qc_scale_measured`
other than 1.0 and refuses `use_measured_qc_scale` outright). The statistic and the threshold are now
the same quantity in the same units, so there is nothing left to mis-calibrate, and the two tasks'
configs become **identical apart from the experiment name** — the per-task calibration step is gone.

The dual is unchanged and still **computed, not learned** (CLAUDE.md rule 1):

$$g(\eta,\lambda)=\eta\varepsilon+\lambda d+\eta\,\mathbb{E}_s\Big[\log\tfrac1N\textstyle\sum_j\exp\big(\tfrac{Q_r(s,a_j)-\lambda\rho_c(s,a_j)}{\eta}\big)\Big]$$

jointly convex, minimised by SLSQP over `η > 0`, `λ ∈ [0, λ_max]`, warm-started.

**Measured, not assumed:** at episodic scale the solver converges cleanly (status 0 in 4–5
iterations) and its solution agrees with a 400×400 grid search. With costs centred at 40 against
`d = 25` and enough KL budget to reach it, λ lands **interior** at 0.068 with a KKT residual
`|d − E_q[ρ_c]| < 1e-2`. With a narrow spread and a tight KL budget it pins at `λ_max` and the
feasibility probe reports it. Both are asserted in `tests/test_fhdcmpo.py`.

### 5.1 λ's scale — a wrong prediction, and the correction

**Predicted, before running:** `ρ_c` is order 10–40 where `Q_c` was 2–4, so the same λ has
≈ `1/qc_scale` ≈ 13× more effect inside the exponent; therefore divide every gain by 13 and take
`λ_max` from 4.0 to 0.35.

**Measured (S0 smoke, 2k iters, seed 1): that was wrong.**

```
std_a(Q_r) = 0.2385     std_a(rho_c) = 0.2883     lambda_balanced = 0.88
```

The error is a level/spread conflation, and it is the same mistake as §1's defect 2 in mirror image.
Only *differences across candidate actions at a state* survive the per-state softmax, so λ's balance
point is `std_a(Q_r) / std_a(ρ_c)` — a ratio of **spreads**, not of **levels**. The level went up
13×; the spreads turn out to be comparable (0.24 vs 0.29). So the QR-DMPO gains carry over
essentially unchanged, and `λ_max = 4.0` is right for the same reason it was right there.

**The rescaled version would have failed silently and looked like a weak method.** At `λ_max = 0.35`
the cap sits *below* the balance point 0.88, so the cost term can never perturb the softmax enough
for the constraint to bind — cost would have stayed high with `at_cap_frac = 0`, i.e. no saturation
symptom to point at. Every gain is restored; `lambda_balanced` stays logged and should be re-checked
as `std_a(Q_r)` grows during training.

*This is the second time in this file that reasoning about the level rather than the spread produced
a wrong prediction. The E-step compares actions at a state; the level almost never matters directly.*

## 6. The bound

Let `ρ̂` be the statistic read from the learned critic, `q*` the E-step solution, `π_θ` the M-step
projection. Then

$$J_c(\pi_\theta)\ \le\ d\ +\ \underbrace{\Delta_{\text{critic}}}_{\text{estimation}}\ +\ \underbrace{\Delta_{\text{estep}}}_{\text{feasibility}}\ +\ \underbrace{\Delta_{\text{mstep}}}_{\text{projection}}$$

**`Δ_critic`.** `CVaR_α` is `\frac{1}{1-\alpha}`-Lipschitz in `L¹` with respect to the quantile
function: if `‖F̂^{-1} − F^{-1}‖_1 ≤ e` then `|CVaR_α[Ẑ] − CVaR_α[Z]| ≤ e/(1-α)`. So
`Δ_critic ≤ κ·e/(1−α) + (1−κ)·e_mean`. **A tighter α buys tail protection at a price linear in
`1/(1−α)` in required critic accuracy.** This is the formal statement of defect 3: at `α = 0.9` a
2.1× dispersion error is amplified tenfold, which is why the earlier CVaR arm paid full price for a
constraint that did not deliver. It is also the argument for the κ ramp being tied to critic
calibration rather than to a fixed schedule.

**`Δ_estep`.** `q*` satisfies `E_{q*}[ρ̂_c] ≤ d` only when the constraint set is non-empty on the
*sampled support*. When it is not, the achieved level is the reachable floor
`min_{KL(q‖π_old)≤ε} E_q[ρ̂_c]`, and `Δ_estep` is the gap. Already instrumented as
`qc_reachable_min` / `estep_feasibility_margin` (`cvpo.py:599-659`) — this is the quantity
[[cvpo-negative-result]] identified as empty when every candidate action comes from an
already-unsafe policy, which no reweighting can repair.

**`Δ_mstep`.** `π_θ` is a KL-constrained projection of `q*`, not `q*`. With
`KL(π_θ‖q*) ≤ β`, Pinsker gives `TV ≤ sqrt(β/2)`, so
`Δ_mstep ≤ 2·sqrt(β/2)·(T·c_max)` in the worst case. This is the loosest of the three terms and the
one worth replacing with a path-space bound later (CLAUDE.md M4).

**The point of the finite-horizon formulation: there is no `1/(1−γ)` factor anywhere in this
bound.** CPO/CVPO/EVO-style bounds carry `\frac{1}{1-\gamma}` (EVO Thm 4.1 carries
`\frac{2\gamma\epsilon}{(1-\gamma)^2}`) because they bound a discounted surrogate and then relate
it to the real constraint. At `γ = 0.99` that is a factor of 100 sitting in front of the error
terms. Constraining the finite-horizon quantity directly removes it — the bound is stated in the
same units as the thing being bounded, so no conversion is needed.

### 6.1 The falsifiable claim, registered before the first run

`CVaR_α[Z] ≤ d` implies `VaR_α[Z] ≤ d` (since `CVaR_α ≥ VaR_α`), which implies

$$\boxed{P(J_c > d)\ \le\ 1-\alpha}$$

At `α = 0.9`: **at most 10% of episodes over budget**, against 32–46% for every method in §0.

This is the prediction on record. Two ways it can fail, and both are informative:

- **It holds.** Then the finite-horizon distributional constraint does what the tail statistic in
  discounted space could not, and the gap between the measured rate and 10% *is*
  `Δ_critic + Δ_estep + Δ_mstep` — logged term by term, so the bound is accountable rather than
  decorative. `fh_predicted_violation_rate` is logged next to the realized exceedance rate.
- **It fails.** Then one of the three terms dominates, and the diagnostics say which:
  `Δ_critic` shows up as a bad marginal PIT KS, `Δ_estep` as `estep_feasibility_margin < 0` and a
  pinned λ, `Δ_mstep` as a large `kl_path`. Reporting which term ate the guarantee is a result.

**What would make this a negative result rather than a bug:** reward collapsing below ~22 on
PointGoal1 while violations fall, i.e. paying for the tail purely in performance. That is defect 2
(the exchange rate) asserting itself, and it is not fixable by tuning. On Goal tasks the goals sit
near the hazards, so reward and cost partly share an axis — [[cvpo-negative-result]] found exactly
this on CarGoal2 ("goals-reached and cost are the same axis"). If the tail is irreducible at reward
≥ 23, the deliverable is the measured reward-vs-tail frontier, not a tuned number.

## 7. Staging

Each stage gates the next; a failed gate is recorded and reverts to the prior stage.

| stage | change | gate |
|---|---|---|
| S0 | derive; land `fh_cost.py`, the critic risk surface, `FHDCMPO` | unit tests green; full suite no worse than baseline; smoke run learns |
| S1 | finite horizon only, `fh_risk_mode: mean` (κ ≡ 0) | FH critic marginally calibrated; mean cost within seed noise (±3.05) of QR-DMPO's 18.97; `qc_scale` provably unused |
| S2 | κ ramp 0 → 1 at α = 0.9 | violations < 20% at reward ≥ 22; λ does not pin |
| S3 | in-training PIT recalibration | marginal PIT KS < 0.1; measured `P(J_c>25) ≤ 1−α` within noise |
| S4 | 3 seeds × {PointGoal1, CarGoal1} | final table, whichever gates failed included |

**S1 is not a formality.** If the undiscounted critic does not reproduce the mean-constraint
operating point, every tail number built on it is uninterpretable. This is the step that the
earlier CVaR attempt skipped.

### 7.1 S0 result — gate PASSED (2026-08-17, 2k iters, 8 envs, seed 1, Blackwell)

`config/safety_gymnasium_fhdcmpo_smoke.yaml`. Not a research arm; the κ schedule is compressed 25×
so the CVaR branch is reached inside the budget, which makes the reward/cost numbers meaningless.
What it establishes:

```
Qc:     thres=25.0000  scale=1.0000  thres_eff=25.0000     <- threshold IS the limit, no qc_scale
Fh:     kappa=1.0000  rho_over_limit=1.1634  alpha=0.9  predicted_violation_rate=0.1000
Lambda: mean=0.1240  balanced=0.8844  over_balanced=0.1402  at_cap_frac=0.0000
Dual:   residual_eta=3.1e-05  residual_lambda=-4.0858  solver status=0  iters=3
Estep:  feasible=0.0  feasibility_margin=-3.9604   (reachable_min=28.96 vs d=25)
Critic: cost_mean_Q=12.87  cost_spread=19.28  cost_zero_frac=1.3e-03
        actor input 61 = 60 obs + u;  critic input 63 = 61 + 2 actions
        episode cost 98.5, reward 8.1, goals 3.69 (unconverged, as expected at 2k)
```

Confirmed: horizon column present and reaching both actor and critics; `qc_scale` gone; the κ ramp
moves off 0 and reaches its target; the dual solves (`status=0`, `residual_eta ~ 3e-5`); every
required diagnostic reports. The λ-scale prediction was refuted — see §5.1.

**`Δ_estep` is already visible and is now interpretable without conversion**: `reachable_min = 28.96`
against `d = 25` says the E-step, at this point in training, cannot reweight its 64 sampled actions
to an episodic cost below 29.0 whatever λ does. Under the old discounted scheme that number was
`2.9` against `1.91` and had to be divided by a measured 0.0764 before it meant anything. This is
the practical payoff of stating the constraint in the units it is stated in.

### 7.2 Measuring `Δ_critic` directly — and a probe-design trap

The finite-horizon formulation makes `Δ_critic` **directly measurable**, which the discounted one did
not. The critic predicts `sum_{t'=t}^{T-1} c_t'`, and because episodes run a fixed `T`, the realized
value at every visited state is known once the episode ends:
`realized(t) = (episode total) - (cost before t)`. No discounting, no bootstrap, no `qc_scale`.
`scripts/eval/probe_fh_calibration.py` does this.

**The trap, which cost one wrong conclusion before it was caught.** *Which policy the probe rolls out
changes the verdict's sign.* Same checkpoint (S1, iteration 5000, 8 episodes, 320 states):

| rollout policy | pred mean | real mean | ratio | PIT KS | pred CVaR₀.₉ | real CVaR₀.₉ | VaR coverage | verdict |
|---|---|---|---|---|---|---|---|---|
| **stochastic** (training dist.) | 17.38 | 14.44 | **1.20** | 0.644 | 46.33 | **70.31** | 0.841 | **under**-disperses |
| deterministic (deployment dist.) | 16.91 | 2.81 | 6.01 | 0.812 | 42.70 | 15.00 | 1.000 | over-disperses |

The deterministic policy at this checkpoint is near-idle and realizes 2.81 mean cost-to-go against the
exploring policy's 14.44 — a 5× gap. Probing it against a critic trained on replay from the
stochastic policy manufactures a 6× "miscalibration" that is really a distribution mismatch. The
E-step queries the critic at *sampled* actions from the current stochastic policy, so **stochastic is
the mode that measures the bound's `Δ_critic`**; deterministic measures the deployment distribution,
which is a different (also useful) question. The probe now takes `--policy` and prints which it used.

**Two findings from the stochastic column, both early (iteration 5000 of 60000, 8 episodes — noisy,
not the gate):**

1. **The level is essentially right: ratio 1.20.** The discounted critic under-read its level by
   **2.2×** ([[cvpo-cost-critic-investigation]]), which is what kept λ from ever engaging. Getting the
   level for free is the main thing the finite-horizon change was supposed to buy, and at 5k
   iterations it looks bought.
2. **The tail still under-disperses, and by about the same margin as before.** Predicted `CVaR₀.₉`
   46.3 vs realized 70.3 (−34%), and `VaR₀.₉` coverage 0.841 against 0.858 for the old categorical
   critic ([[why-mean-beat-cvar-on-pointgoal1]]). **So correcting the units and the level did *not*
   fix the shape.** Under-dispersion in the unsafe direction is precisely what the `1/(1-α)`
   Lipschitz term in §6 says gets amplified tenfold at `α = 0.9`.

Consequence for the plan: **S3 recalibration moves from "if needed" to "expected to be needed."** The
prediction of §6.1 (`P(J_c > 25) ≤ 10%`) should be expected to *fail* at S2 with the raw critic, by
roughly the amount the tail is understated, and S3 is the step that would recover it. That is a
sharper, more falsifiable claim than the original staging made, and it was available before S2 ran.

### 7.3 `Δ_mstep` — RETRACTED: the measurement was confounded

Because everything is now in episodic units, the M-step projection slack is readable straight off the
logs as `realized J_c − E_q*[ρ_c]`: what the E-step *asked for* versus what the projected policy
*delivered*. S1, mean arm:

| iter | `E_q[ρ_c]` | realized `J_c` | slack = `Δ_mstep` |
|---|---|---|---|
| 1000 | 3.45 | 75.38 | **+71.93** |
| 2000 | 10.42 | 94.44 | **+84.02** |
| 3000 | 13.99 | 55.81 | +41.82 |
| 4000 | 16.34 | 108.41 | **+92.06** |
| 5000 | 19.23 | 43.25 | +24.02 |
| 6000 | 17.54 | 30.25 | +12.71 |
| 7000 | 16.57 | 13.47 | −3.10 |
| 9000 | 11.45 | 0.72 | −10.73 |
| 11000 | 9.89 | 3.00 | −6.89 |

> **RETRACTION (2026-08-18).** The table above is **not** a measurement of `Δ_mstep`, and the two
> conclusions drawn from it below are wrong. `E_q[ρ_c]` is a mean cost-**to-go** averaged over replay
> states, which span all timesteps of an episode; `J_c` is the **full-episode** total. If cost were
> spread evenly over an episode then mean cost-to-go is `J_c/2` *by construction*. Measured:
> `E_q[ρ_c] ≈ 11–12` against `J_c/2 ≈ 13–16`. So most of the "+72 to +92" and "+18 to +20" gaps are
> **horizon averaging, not projection loss**. Comparing a mid-episode quantity against an episode
> total and calling the difference an M-step failure was simply a bad measurement.
>
> **The valid comparison** is `E_π[ρ_c]` vs `E_q[ρ_c]` — the same states and the same quantity,
> before and after the E-step's reweighting. `E_π[ρ_c]` is already logged as `critic_cost_mean_Q`:
>
> | arm | iter | `E_π[ρ_c]` | `E_q[ρ_c]` | E-step reduction |
> |---|---|---|---|---|
> | s1b (λ retune) | 59999 | 11.81 | 11.75 | **0.6%** |
> | s2b (TD(λ)) | 19000 | 10.97 | 10.47 | **4.6%** |
>
> **The binding term is the E-step, not the M-step.** The E-step solves its dual correctly and
> reports `feasible = 1`, but the variational distribution it returns is within 1–5% of the current
> policy *on the cost axis*. There is almost nothing for the M-step to fail to project. The
> mechanism is the one already in the logs: only the spread across candidate actions survives the
> per-state softmax, and `λ·std_qc ≈ 1.16 × 0.14` is small against `std_qr ≈ 0.22`, so the cost term
> is a minor perturbation on the reward preference.
>
> Consequences for the plan, all of which reverse §7.3's original advice:
> * a richer *policy class* (diffusion, M3) targets the M-step and cannot fix this;
> * a better *critic* (TD(λ), S3 recalibration) helps only as a small multiplier — TD(λ) made the
>   E-step ~5x stronger (0.8% → 4.6%) and realized cost barely moved;
> * the lever is **how hard the cost term is allowed to push inside the E-step**: `λ` relative to its
>   balance point, and `λ_max`. S1 over-constrained at λ≈3.9 (cost → 0.7, reward → 0.95); s1b/s2b
>   under-constrain at λ≈1.2, sitting *at* `λ_balanced` where reward and cost spreads contribute
>   equally. The operating point is between, and has not been run.
>
> The original text is kept below because the *reasoning pattern* it encodes — "instrument each term
> of the bound separately" — is right, and because a retraction that deletes its own evidence is not
> a retraction. Only the arithmetic was wrong.

**Two things follow, and both matter more than the tail statistic.**

1. **`Δ_mstep` dwarfs `Δ_critic` early on — 72 to 92, against a budget of 25.** The bound of §6 is
   therefore *vacuous* during early training: no amount of critic accuracy or tail conservatism buys
   anything while the projection is losing 90 units of cost. Whatever safety is achieved in that
   regime comes from the λ feedback loop on *realized* cost, not from the E-step's guarantee.
2. **E-step feasibility is not sufficient for safety.** At iteration 3000 the E-step reported
   `feasible = 1.0` with `E_q[ρ_c] = 13.99 ≤ 25` while the policy realized 55.81. The constrained
   variational step genuinely solved its own problem; the projection then threw the solution away.
   This is the clearest possible argument for the bound having three terms rather than one, and against
   the tempting claim that an exactly-solved constrained E-step implies a safe policy.

**Corollary — do not "fix" the controller by switching `lambda_source` to `qspace`.** It looks
attractive now that the units and level are trustworthy (`qspace` was unusable before precisely
because the level was not), and it would remove the episode-length lag that caused the overshoot. But
`qspace` reads `E_q[ρ_c] − d`, i.e. what the E-step asked for: at iteration 3000 that is
`13.99 − 25 = −11`, reading as *comfortable slack* while the policy is violating by 2.2×. It would
systematically under-constrain by exactly `Δ_mstep`. `episodic` is the correct signal; the overshoot
is a gain-tuning problem, addressed in §7.4.

### 7.4 S1 over-constrained — the gains were tuned against a broken critic

S1 with the inherited QR-DMPO gains:

| iter | `J_c` | reward | goals | λ | at_cap% |
|---|---|---|---|---|---|
| 3000 | 55.8 | 9.25 | 4.38 | 2.23 | 0.0 |
| 5000 | 43.2 | 4.58 | 2.41 | 3.92 | 0.0 |
| 7000 | 13.5 | −0.42 | 1.22 | 3.76 | 18.3 |
| 9000 | 0.7 | 0.52 | 0.94 | 3.42 | 14.1 |
| 11000 | 3.0 | 0.95 | 1.22 | 2.78 | 11.5 |

Cost driven from 56 to **0.7 against a budget of 25**, with reward collapsing 9.25 → 0.95 and goals
4.4 → 1.2. That is the idle-policy failure ([[safety-goal-eval-gotchas]]: safe low-reward checkpoints
are usually idle, not skilled), not a safety result.

**This is the mirror image of every prior failure in this repo, and the diagnosis has a size rather
than a guess.** The old arms could not get cost below ~47; this one overshoots to 0.7. The inherited
gains were tuned against a cost critic that under-read its level by **2.2×**, so that arm's *effective*
pressure was `λ_max / 2.2 ≈ 1.8`. The finite-horizon critic reads the level correctly (§7.2, ratio
1.20), so the same nominal λ now bites ~2.2× harder. Preserving the old operating point means
`λ_max ≈ 1.8`, not 4.0 — and `Ki` halved, since the overshoot came from the integral winding up while
cost was ~100 and then needing ~10k iterations to unwind. `config/safety_gymnasium_fhdcmpo_s1b_mean_goal1.yaml`.

*Generalisation worth keeping: every controller gain in this repo was tuned against a mis-levelled cost
critic. Fixing the critic's level invalidates all of them by the size of the level error.*

### Note on recalibration (S3)

**A dispersion bias in the existing PIT feed, which S3 must not inherit.** The categorical path feeds
the recalibrator the PIT of the prediction evaluated at
`realized = costs + discount * mask * next_q`, where `next_q` is the **mean** of the target
distribution (`safe_sac.py`, the `_record_pit` call). A bootstrapped target built from a point
estimate has strictly less spread than the return it stands for, so this PIT source *itself*
understates dispersion — it will under-report exactly the defect it is meant to detect. For a quantile
head there is a cheap fix with no extra cost: draw a uniform index `k` and use
`theta'[:, k]` instead of the mean, which is an actual sample from the represented target
distribution and preserves its spread. Whatever S3 does, it should be fed a *sampled* target, not a
mean one — and the offline probe (§7.2), which compares against genuine realized Monte-Carlo returns,
stays the arbiter.

PIT recalibration of a *quantile* head is not the same operation as for a categorical head.
`recalibration.py`'s `recalibrate_distribution` remaps categorical probabilities; for a QR head the
CDF values are fixed by construction and the *support* is learned, so calibration means re-weighting
the equal-mass locations: `p_k = Ĝ(τ̂_k + 1/2N) − Ĝ(τ̂_k − 1/2N)` for the fitted PIT CDF `Ĝ`, and
then a mass-weighted CVaR. `FHDCMPO` raises `NotImplementedError` on `recalibrate_cvar` rather than
silently applying the categorical path. Calibration is judged on the **marginal**, never per-state
coverage — [[qr-dmpo-math]] §6.1 shows per-state coverage collapses toward 0.5 regardless of
correctness when the conditional is near-deterministic.

## 8. Positioning

| method | tail statistic | finite-horizon undiscounted | exactly-solved variational E-step |
|---|---|---|---|
| CVPO (ICML 2022) | — (expectation) | no | **yes** |
| WCSAC (AAAI 2021) | CVaR, Gaussian-fitted | no | no (Lagrangian) |
| SDAC (NeurIPS 2023) | CVaR, distributional | no | no (trust region) |
| EVO (ICML 2025) | **EVT/GPD extreme quantile** | no | no (CPO-style primal) |
| Sauté RL (ICML 2022) | — | **yes** (budget-augmented) | no |
| CGPO (2024) | — | **yes** (needs differentiable sim) | no |
| **FH-DCMPO** | **CVaR or EVT** | **yes** | **yes** |

The combination is the contribution, and the bound in §6 is what the combination buys: a safety
statement composing critic error, E-step feasibility and M-step slack, with no `1/(1−γ)`.

---

*Process note, in the spirit of [[qr-dmpo-math]]'s: the `1/(1−α)` Lipschitz term in §6 was derived
before launching anything, and it predicts that α and critic calibration cannot be tuned
independently. If S2 fails with a bad PIT KS, that is the term talking, and the fix is S3, not a
different α.*
