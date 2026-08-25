# Multi-env sweep: what was trained, what came out, and what every logged variable means

*2026-08-25. Runs on one workstation (Xeon 6520P, 16 cores, 31 GiB RAM, RTX PRO 4500 Blackwell).
wandb project `SafeRL-multienv-sweep`, entity `uqerh-kit`.*

---

## 1. What was trained

Two FH-DCMPO arms across four new Safety-Gymnasium environments, 60k iterations each,
cost limit 25, `num_envs=8`, two runs concurrent.

| arm | config | what it is |
|---|---|---|
| **a2** | `safety_gymnasium_fhdcmpo_a2_lag_cc4_60k.yaml` | baseline. FH-DCMPO with the episodic PID lambda controller, TD(lambda) cost targets over an L=64 window, quantile cost critic, `fh_risk_mode: mean` (kappa pinned to 0). |
| **c2** | `safety_gymnasium_fhdcmpo_c2_spreadmatch_med.yaml` | a2 **plus** per-state E-step cost spread matching, median-normalized (`estep_cost_spread_match: true`, `estep_spread_match_max: 10`). Also adds the deterministic eval panel. |

The two configs are byte-identical apart from those keys and the run names.

**Envs.** `SafetyPointGoal2-v0`, `SafetyCarGoal1-v0`, `SafetyPointButton1-v0`,
`SafetyPointPush1-v0`. All share horizon T=1000 and action dim 2; observation width differs
(60 / 72 / 76 / 76), so checkpoints are not portable between them. PointGoal1 was excluded --
it already had both arms from earlier work and is the reference point.

**Grid.** 2 arms x 3 seeds (2,3,4) x 4 envs = 24 cells, run by
`scripts/sweep/local_queue.py` at N=2 in priority order (seed 2 across all envs first, then
seeds 3 and 4).

**What actually completed: 12 of 24 cells.** Two more died partway
(`cargoal1_a2_s3` at 51k, `cargoal1_c2_s3` at 35k -- no traceback, killed externally at
11:36-11:53 on 2026-08-25; the box later rebooted at 13:50). Ten cells never started.
Seed 2 is complete on all four envs; seeds 3 and 4 only on PointGoal2.

---

## 2. Headline result: the constraint does not hold

**Cost limit is 25.** Nothing in the table below is reliably under it.

| cell | train rew | train cost | goals | eval rew | eval cost | eval p90 | eval viol |
|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | 28.83 | 22.91 | 13.88 | -- | -- | -- | -- |
| `cargoal1_c2_s2` | 21.45 | 41.69 | 10.00 | 25.84 | 33.75 | 61.00 | 0.50 |
| `cargoal1_a2_s3` | 27.36 | 26.81 | 12.97 | -- | -- | -- | -- |
| `cargoal1_c2_s3` | 19.17 | 34.81 | 9.22 | 19.25 | 31.50 | 71.00 | 0.50 |
| `pointbutton1_a2_s2` | 8.06 | 47.88 | 4.09 | -- | -- | -- | -- |
| `pointbutton1_c2_s2` | 2.79 | 25.25 | 1.53 | 8.83 | 74.50 | 141.00 | 0.75 |
| `pointgoal2_a2_s2` | 6.24 | 32.38 | 2.56 | -- | -- | -- | -- |
| `pointgoal2_c2_s2` | 5.35 | 37.16 | 2.19 | 3.38 | 6.00 | 24.00 | 0.00 |
| `pointgoal2_a2_s3` | 5.89 | 48.09 | 2.50 | -- | -- | -- | -- |
| `pointgoal2_c2_s3` | 3.02 | 30.62 | 1.19 | 4.47 | 29.25 | 117.00 | 0.25 |
| `pointgoal2_a2_s4` | 8.16 | 52.62 | 3.16 | -- | -- | -- | -- |
| `pointgoal2_c2_s4` | 6.54 | 21.66 | 2.69 | 12.14 | 92.25 | 268.00 | 0.75 |
| `pointpush1_a2_s2` | 2.28 | 24.50 | 0.53 | -- | -- | -- | -- |
| `pointpush1_c2_s2` | 3.44 | 50.28 | 0.88 | 1.93 | 8.00 | 32.00 | 0.25 |

Read this way:

* **Cost is out of control on every env.** Training-rollout cost ranges 21.7 to 52.6 against a
  limit of 25. The deterministic eval panel (c2 only) is worse: Button1 cost **74.5** at 75%
  violations, PointGoal2 seed 4 cost **92.3** at 75% violations, with a p90 of 268.
* **Seed variance swamps the arm difference.** PointGoal2 c2 gives eval cost 6.0 / 29.3 / 92.3
  across seeds 2/3/4. That is not a converged number.
* **c2 does not beat a2.** On PointGoal2, reward 5.3/3.0/6.5 (c2) vs 6.2/5.9/8.2 (a2), and cost
  is no lower. The spread-matching advantage measured on PointGoal1 does not reproduce here.
* **Only one cell is in budget:** `pointpush1_a2_s2` at cost 24.5 -- and see the diagnosis
  below, it is inert rather than controlled.

**Caveat on comparability.** Only c2 configs carry the eval panel; a2 does not. The `Episode/*`
columns are training-rollout means with exploration noise, so a2-vs-c2 comparisons here are
indicative, not matched. `scripts/analysis/sweep_report.py` produces matched checkpoint-based
metrics for both arms and has not been run on this set.

---

## 3. Diagnosis: the multiplier is saturated

This is the finding that explains the table above.

| cell | lambda | lam_balanced | lam/balanced | lam_delta | eta | dres_lambda | dres_eta | solver |
|---|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | 1.662 | 1.863 | 0.892 | 0.089 | 0.528 | 14.260 | -0.000 | 0.000 |
| `cargoal1_c2_s2` | 1.800 | 1.188 | 1.515 | 0.410 | 0.691 | 11.269 | -0.000 | 0.000 |
| `cargoal1_a2_s3` | 1.489 | 1.303 | 1.143 | 0.076 | 0.626 | 12.898 | -0.000 | 0.000 |
| `cargoal1_c2_s3` | 1.800 | 1.907 | 0.944 | 0.358 | 0.618 | 14.335 | -0.000 | 0.000 |
| `pointbutton1_a2_s2` | 1.800 | 1.366 | 1.318 | 0.727 | 1.032 | 6.612 | -0.000 | 0.000 |
| `pointbutton1_c2_s2` | 1.800 | 1.083 | 1.662 | -0.099 | 0.888 | 9.445 | -0.000 | 0.000 |
| `pointgoal2_a2_s2` | 1.794 | 1.532 | 1.171 | 0.135 | 0.938 | 13.976 | 0.000 | 0.000 |
| `pointgoal2_c2_s2` | 1.800 | 1.255 | 1.435 | 0.190 | 0.580 | 9.919 | 0.000 | 0.000 |
| `pointgoal2_a2_s3` | 1.800 | 1.502 | 1.198 | 0.278 | 1.027 | 11.216 | -0.000 | 0.000 |
| `pointgoal2_c2_s3` | 1.800 | 0.987 | 1.824 | 0.399 | 0.988 | 6.565 | -0.000 | 0.000 |
| `pointgoal2_a2_s4` | 1.786 | 1.641 | 1.088 | 0.490 | 1.158 | 10.038 | -0.000 | 0.000 |
| `pointgoal2_c2_s4` | 1.800 | 1.332 | 1.351 | 0.320 | 0.683 | 12.581 | -0.000 | 0.000 |
| `pointpush1_a2_s2` | 0.112 | 5601351168.000 | 0.000 | -0.410 | 0.157 | 15.824 | -0.000 | 0.000 |
| `pointpush1_c2_s2` | 1.702 | 21.353 | 0.080 | 0.894 | 0.223 | 8.121 | -0.000 | 0.000 |

`lambda` is pinned at **`lambda_max = 1.8` in 10 of 14 runs**, while cost sits at 2x the budget.
The controller is already pushing as hard as it is allowed to and cannot push harder. This is the
same failure mode documented in `codex/why-mean-beat-cvar-on-pointgoal1.md`: once lambda
integrates to its cap and stays there, the E-step weight is effectively "minimise cost, ignore
reward" -- except here even that is not enough to reach the budget.

`lambda_max: 1.8` is a **PointGoal1 number**: the c3 config header records it as the old 4.0
divided by 2.2, "the critic's old level error". These environments are simply more expensive --
a2 on Button1 ends at cost 47.9, on PointGoal2 at 52.6. A budget of 25 may not be reachable on
these tasks at any multiplier.

The two exceptions are informative:

* `pointpush1_a2_s2`: lambda **0.112**, cost 24.5 -- the only in-budget cell. But its
  `estep_spread_ratio` is **0.000**: the cost signal is flat across candidate actions, so the
  constraint cannot steer the policy at all. In budget by accident, not by control.
* `cargoal1_a2_s2`: lambda 1.66 (just below cap), cost 22.9 -- the one genuinely controlled run.

`dual_residual_lambda` = `qc_thres_eff - eqc` is large and positive everywhere (14.3 on
cargoal1_a2_s2), meaning the E-step's own cost expectation sits far below the threshold it is
solving against -- the constraint is not binding inside the E-step even while realized cost
overshoots. That gap between the E-step's view and reality is the core pathology.

---

## 4. Losses

| cell | Loss/actor | Loss/critic | Loss/cost_critic | lr |
|---|---|---|---|---|
| `cargoal1_a2_s2` | 3.0017 | 99.2249 | 169.5231 | -- |
| `cargoal1_c2_s2` | 3.2593 | 109.2637 | 211.5940 | -- |
| `cargoal1_a2_s3` | 2.8448 | 100.3534 | 171.0479 | -- |
| `cargoal1_c2_s3` | 2.9621 | 125.8870 | 202.1218 | -- |
| `pointbutton1_a2_s2` | 2.9585 | 144.9257 | 240.1351 | -- |
| `pointbutton1_c2_s2` | 3.0984 | 119.2012 | 207.3039 | -- |
| `pointgoal2_a2_s2` | 2.8539 | 128.1233 | 174.3954 | -- |
| `pointgoal2_c2_s2` | 2.6768 | 98.2205 | 217.2490 | -- |
| `pointgoal2_a2_s3` | 3.0238 | 121.1512 | 215.7477 | -- |
| `pointgoal2_c2_s3` | 2.9325 | 101.4490 | 253.5601 | -- |
| `pointgoal2_a2_s4` | 3.1186 | 129.5752 | 232.4760 | -- |
| `pointgoal2_c2_s4` | 2.9752 | 100.8515 | 204.7358 | -- |
| `pointpush1_a2_s2` | 2.6553 | 43.7802 | 63.2733 | -- |
| `pointpush1_c2_s2` | 2.5794 | 51.2264 | 123.9023 | -- |

* **`Loss/actor`** -- the M-step weighted-MLE loss: negative log-likelihood of the E-step's
  sampled actions under the new policy, weighted by the E-step weights, plus the Lagrangian
  terms enforcing the decoupled mean/covariance KL trust regions. It is *not* a return, so its
  absolute level is not meaningful; what matters is that it stays finite and does not drift.
* **`Loss/critic`** -- quantile Huber loss of the reward critic against its n-step TD target
  (n_step=10, twin critics). Scales with the square of reward magnitude, so CarGoal1 (~99) being
  larger than Push1 is expected, not pathological.
* **`Loss/cost_critic`** -- quantile Huber loss of the cost critic against the TD(lambda) cost
  target over the L=64 window, undiscounted. The large values (up to ~170 on CarGoal1) reflect
  episodic cost magnitudes of 20-50, not a broken critic.
* **`Loss/alpha`**, **`SafeRL/alpha`**, **`Train/alpha_mean/var`** -- SAC entropy-temperature
  loss and value. `auto_entropy_tuning` is **off** in these configs, so alpha is frozen at 0 and
  these are inert.
* **`Train/learning_rate`** -- constant 3e-4 for actor, critic and cost critic.

---

## 5. E-step: dual solve, weights, feasibility

| cell | ESS | ESS_min | std_a Qr | std_a Qc | spread_ratio | feasible | feas_margin | frac_sat |
|---|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | 57.385 | 6.323 | 0.064 | 0.037 | 0.892 | 1.000 | 14.280 | 0.705 |
| `cargoal1_c2_s2` | 56.983 | 4.597 | 0.122 | 0.087 | 1.515 | 1.000 | 11.316 | 0.559 |
| `cargoal1_a2_s3` | 56.352 | 14.477 | 0.090 | 0.075 | 1.143 | 1.000 | 12.923 | 0.615 |
| `cargoal1_c2_s3` | 57.610 | 2.713 | 0.146 | 0.075 | 0.944 | 1.000 | 14.448 | 0.508 |
| `pointbutton1_a2_s2` | 56.053 | 5.633 | 0.192 | 0.127 | 1.318 | 1.000 | 6.646 | 0.297 |
| `pointbutton1_c2_s2` | 57.319 | 3.821 | 0.118 | 0.114 | 1.662 | 1.000 | 9.516 | 0.285 |
| `pointgoal2_a2_s2` | 57.059 | 5.613 | 0.140 | 0.077 | 1.171 | 1.000 | 14.003 | 0.285 |
| `pointgoal2_c2_s2` | 56.645 | 9.232 | 0.111 | 0.068 | 1.435 | 1.000 | 10.011 | 0.244 |
| `pointgoal2_a2_s3` | 57.338 | 3.704 | 0.145 | 0.080 | 1.198 | 1.000 | 11.237 | 0.330 |
| `pointgoal2_c2_s3` | 56.603 | 8.732 | 0.148 | 0.132 | 1.824 | 1.000 | 6.657 | 0.246 |
| `pointgoal2_a2_s4` | 56.472 | 5.351 | 0.160 | 0.110 | 1.089 | 1.000 | 10.072 | 0.355 |
| `pointgoal2_c2_s4` | 56.552 | 8.366 | 0.144 | 0.092 | 1.352 | 1.000 | 12.673 | 0.338 |
| `pointpush1_a2_s2` | 57.243 | 6.175 | 0.025 | 0.000 | 0.000 | 1.000 | 15.883 | 0.281 |
| `pointpush1_c2_s2` | 57.698 | 7.156 | 0.032 | 0.001 | 0.081 | 1.000 | 8.202 | 0.244 |

* **`Train/eta`** -- the E-step temperature, the solution of the convex dual. Small eta = sharp
  weighting over sampled actions. Sitting at 0.5-0.9 throughout, which is healthy.
* **`Train/ess` / `ess_min`** -- effective sample size of the E-step weights,
  `(sum w)^2 / sum w^2`, out of `sample_action_num`. Collapse means the M-step is fitting
  essentially one action. `ess_min` is the worst state in the batch; values of 6 out of 64 at
  the worst state indicate some states are near-degenerate but the median is fine.
* **`Train/estep_std_qr` / `estep_std_qc`** -- the across-action spread of the reward and cost
  readouts at a state. **Only the spread survives the per-state softmax** -- a constant added to
  Q_c cancels in the normalization -- so these, not the levels, decide whether the constraint can
  re-rank actions. Both are tiny (0.03-0.07), which is the long-standing finding that the cost
  signal is nearly flat across candidate actions.
* **`Train/estep_spread_ratio`** = `lambda * std_a(Qc) / std_a(Qr)` -- the *delivered* relative
  influence of cost against reward in the exponent. ~1.0 means balanced. `pointpush1_a2_s2`
  reads 0.000: the constraint is completely inert there.
* **`SafeRL/lambda_balanced`** -- the lambda that *would* make cost and reward weigh equally,
  `median_s[std_a(Qr)/std_a(Qc)]`. **`lambda_over_balanced`** is `lambda / lambda_balanced`;
  below 1 means the multiplier is under-powered for the signal it is multiplying.
* **`Train/estep_feasible` / `feasibility_margin`** -- whether the cost threshold is attainable
  within the sampled action support at that state, and by how much. `Train/qc_reachable_min` is
  the best (lowest) cost achievable among sampled actions.
* **`Train/frac_saturated`** -- fraction of sampled actions pushed into the tanh saturation
  region. 0.70 on cargoal1_a2_s2 is high; a saturated actor has little room to respond to the
  penalty.
* **`Train/solver_status` / `solver_iters`** -- SLSQP convergence for the dual. Status 0 is
  success; a silent failure here would invalidate every downstream number.
* **`Train/dual_residual_eta`** = `eps - KL(q*||pi_old)`, ~0 at the optimum (it is, ~-0.0004).
  **`dual_residual_lambda`** = `qc_thres_eff - eqc`, ~0 when the constraint is active; large
  positive means inactive.

---

## 6. Cost critic and thresholds

| cell | cost_mean_Q | cost_spread | tgt_spread | zero_frac | mc_frac | eqc_episodic | qc_thres | c_now/thres | rho/limit |
|---|---|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | 11.913 | 25.756 | 53.974 | 0.301 | 0.035 | 10.740 | 25.000 | 0.432 | 0.430 |
| `cargoal1_c2_s2` | 14.898 | 33.497 | 67.198 | 0.343 | 0.035 | 13.731 | 25.000 | 0.552 | 0.549 |
| `cargoal1_a2_s3` | 11.639 | 26.496 | 55.993 | 0.298 | 0.052 | 12.102 | 25.000 | 0.488 | 0.484 |
| `cargoal1_c2_s3` | 11.352 | 28.536 | 68.934 | 0.450 | 0.038 | 10.665 | 25.000 | 0.428 | 0.427 |
| `pointbutton1_a2_s2` | 19.856 | 37.342 | 82.484 | 0.300 | 0.035 | 18.388 | 25.000 | 0.741 | 0.736 |
| `pointbutton1_c2_s2` | 15.123 | 31.514 | 80.783 | 0.521 | 0.035 | 15.555 | 25.000 | 0.627 | 0.622 |
| `pointgoal2_a2_s2` | 11.877 | 28.341 | 72.336 | 0.552 | 0.035 | 11.024 | 25.000 | 0.447 | 0.441 |
| `pointgoal2_c2_s2` | 14.684 | 32.639 | 100.739 | 0.655 | 0.035 | 15.081 | 25.000 | 0.607 | 0.603 |
| `pointgoal2_a2_s3` | 14.076 | 34.975 | 89.402 | 0.508 | 0.019 | 13.784 | 25.000 | 0.558 | 0.551 |
| `pointgoal2_c2_s3` | 20.042 | 39.975 | 103.023 | 0.555 | 0.019 | 18.435 | 25.000 | 0.744 | 0.737 |
| `pointgoal2_a2_s4` | 16.200 | 36.442 | 88.214 | 0.442 | 0.016 | 14.962 | 25.000 | 0.604 | 0.599 |
| `pointgoal2_c2_s4` | 13.835 | 29.321 | 82.711 | 0.611 | 0.016 | 12.419 | 25.000 | 0.502 | 0.497 |
| `pointpush1_a2_s2` | 10.547 | 7.879 | 47.893 | 0.907 | 0.035 | 9.176 | 25.000 | 0.372 | 0.367 |
| `pointpush1_c2_s2` | 12.965 | 14.952 | 90.406 | 0.896 | 0.035 | 16.879 | 25.000 | 0.683 | 0.675 |

* **`critic/cost_mean_Q`** -- mean predicted cost-to-go over the batch.
* **`critic/cost_spread`** vs **`cost_target_spread`** -- predicted vs target dispersion of the
  quantile head. A predicted spread far below the target's is the under-dispersion problem
  documented in `codex/cvpo-cost-critic-investigation.md`.
* **`critic/cost_zero_frac`** -- fraction of predicted quantiles sitting on the softplus floor
  (`nonneg: true`). High values mean a degenerate head; this is what drove the CVaR dose to its
  algebraic ceiling in the c3 arm.
* **`critic/cost_mc_frac`** -- fraction of TD(lambda) cost windows that reach an episode
  boundary and therefore carry a true Monte-Carlo atom rather than a bootstrap. At L=64 and
  T=1000 this is ~6% by construction; near 0 would mean TD(lambda) is only reshuffling
  bootstrapped atoms.
* **`Train/qc_thres` / `qc_thres_eff` / `qc_thres_static` / `qc_thres_target`** -- the cost-Q
  threshold the E-step solves against. FH-DCMPO is undiscounted with `qc_scale = 1`, so
  **`qc_thres_eff` equals the episodic limit, 25.0**, with no unit conversion. (`Train/qc_scale`
  is therefore 1.0 and `qc_scale_target_ema` is NaN -- both inert by design.)
* **`Train/eqc`** -- `E_q*[Q_c]`, the E-step's own expected cost under the new weights.
  **`eqc_as_episodic_cost`** is the same in episodic units, directly comparable with realized
  cost. It reads ~10.7 on cargoal1_a2_s2 while realized cost is 22.9 -- the E-step believes it is
  well inside budget when it is not.
* **`Train/c_now` / `c_now_over_thres`** -- current mean cost-Q and its ratio to the threshold.
* **`Train/fh_rho_over_limit`** -- the conservatism statistic over the episodic limit.
* **`Train/realized_cost_ema`** -- EMA of realized episodic cost; this is what the episodic PID
  controller actually feeds on. **`SafeRL/lambda_delta`** = `(realized - limit) / limit`, the
  controller's error signal.

---

## 7. Trust region and policy health

| cell | KL mean | KL var | KL q | std cond | std min | std max | pretanh absmax |
|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | 0.0071 | 0.0003 | 0.1004 | 1.7274 | 0.3466 | 5.1373 | -- |
| `cargoal1_c2_s2` | 0.0060 | 0.0002 | 0.1000 | 1.3131 | 0.4749 | 3.2600 | -- |
| `cargoal1_a2_s3` | 0.0068 | 0.0004 | 0.1002 | 1.6489 | 0.3940 | 4.5220 | -- |
| `cargoal1_c2_s3` | 0.0082 | 0.0003 | 0.1001 | 1.3088 | 0.4174 | 2.8752 | -- |
| `pointbutton1_a2_s2` | 0.0070 | 0.0003 | 0.1003 | 1.3595 | 0.3286 | 4.7885 | -- |
| `pointbutton1_c2_s2` | 0.0077 | 0.0003 | 0.1002 | 1.3083 | 0.4338 | 5.8113 | -- |
| `pointgoal2_a2_s2` | 0.0071 | 0.0003 | 0.1000 | 1.2717 | 0.3999 | 3.8120 | -- |
| `pointgoal2_c2_s2` | 0.0077 | 0.0003 | 0.0997 | 1.2355 | 0.3408 | 7.3891 | -- |
| `pointgoal2_a2_s3` | 0.0085 | 0.0003 | 0.1003 | 1.3057 | 0.5338 | 4.9064 | -- |
| `pointgoal2_c2_s3` | 0.0078 | 0.0003 | 0.1000 | 1.2290 | 0.4051 | 5.8633 | -- |
| `pointgoal2_a2_s4` | 0.0069 | 0.0003 | 0.1002 | 1.3890 | 0.4702 | 5.9459 | -- |
| `pointgoal2_c2_s4` | 0.0080 | 0.0003 | 0.1000 | 1.3008 | 0.4399 | 7.3891 | -- |
| `pointpush1_a2_s2` | 0.0056 | 0.0002 | 0.1001 | 1.3996 | 0.4520 | 4.6848 | -- |
| `pointpush1_c2_s2` | 0.0074 | 0.0003 | 0.1001 | 1.3287 | 0.3776 | 4.6869 | -- |

* **`Train/kl_mean` / `kl_var`** -- the decoupled M-step KL trust regions, mean and covariance
  parts, between the new policy and `pi_old`. **`kl_*_rel`** are these relative to their bounds;
  ~1 means the trust region is binding.
* **`Train/kl_q`** -- KL of the E-step's variational distribution `q*` from `pi_old`, the
  quantity the eta dual constrains to `eps`.
* **`Train/pi_std_min/max/cond`** -- per-dimension policy std and its condition number
  (max/min). A collapsed std means the policy has stopped exploring; a large condition number
  means one action dimension is far more deterministic than another.
* **`Train/pretanh_mean_absmax`** -- largest pre-tanh mean magnitude, the companion to
  `frac_saturated`: large values mean the actor is pushed into the flat region of the squash.
* **`Policy/mean_noise_std`** -- the exploration noise scale actually used.

---

## 8. Spread-matching diagnostics (c2 only)

| cell | m median | grip | dose | ratio_med | ref_med | capped | floored |
|---|---|---|---|---|---|---|---|
| `cargoal1_a2_s2` | -- | 2.546 | 1.000 | -- | -- | -- | -- |
| `cargoal1_c2_s2` | 1.000 | 1.474 | 1.000 | -- | -- | -- | -- |
| `cargoal1_a2_s3` | -- | 2.435 | 1.000 | -- | -- | -- | -- |
| `cargoal1_c2_s3` | 1.000 | 0.963 | 1.000 | -- | -- | -- | -- |
| `pointbutton1_a2_s2` | -- | 2.511 | 1.000 | -- | -- | -- | -- |
| `pointbutton1_c2_s2` | 1.000 | 1.611 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_a2_s2` | -- | 2.457 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_c2_s2` | 1.000 | 1.555 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_a2_s3` | -- | 2.728 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_c2_s3` | 1.000 | 1.861 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_a2_s4` | -- | 2.476 | 1.000 | -- | -- | -- | -- |
| `pointgoal2_c2_s4` | 1.000 | 1.433 | 1.000 | -- | -- | -- | -- |
| `pointpush1_a2_s2` | -- | 0.205 | 1.000 | -- | -- | -- | -- |
| `pointpush1_c2_s2` | 1.000 | 0.885 | 1.000 | -- | -- | -- | -- |

These are inert for a2 (the flag is off). For c2:

* **`estep_match_scale_median`** -- median of the per-state factor `m(s)` applied to the cost
  term. Under the default `active` normalizer this is 1 by construction: the median state keeps
  scale 1, so lambda's operating point is preserved while influence is equalized across states.
* **`estep_match_ratio_median`** -- `median_s[std_a(Qr)/std_a(Qc)]` on the active readout.
  **`estep_match_ref_median`** -- the same on the mean readout. Their ratio is the readout dose.
* **`estep_cost_dose_vs_mean_median`** -- how much more across-action cost signal the exponent's
  readout carries than the plain mean. **Exactly 1.0 here**, because both arms run
  `fh_risk_mode: mean` with kappa pinned to 0.
* **`estep_match_grip`** = `lambda * mean_s[m(s) * std_a(Qc)/std_a(Qr)]` -- the delivered
  relative grip of the cost term. This is the quantity that should stay put when the readout
  changes, and it is the headline diagnostic for the c4 arm (not run yet).
* **`estep_match_scale_capped_frac` / `floored_frac`** -- fraction of states hitting the upper
  cap (10) and the lower clamp (0.1). A high floored fraction means the correction is being
  silently truncated.

---

## 9. Throughput

| cell | fps | last iter |
|---|---|---|
| `cargoal1_a2_s2` | 55 | 59999 |
| `cargoal1_c2_s2` | 53 | 59999 |
| `cargoal1_a2_s3` | 56 | 51000 |
| `cargoal1_c2_s3` | 55 | 35000 |
| `pointbutton1_a2_s2` | 57 | 59999 |
| `pointbutton1_c2_s2` | 57 | 59999 |
| `pointgoal2_a2_s2` | 53 | 59999 |
| `pointgoal2_c2_s2` | 55 | 59999 |
| `pointgoal2_a2_s3` | 53 | 59999 |
| `pointgoal2_c2_s3` | 53 | 59999 |
| `pointgoal2_a2_s4` | 52 | 59999 |
| `pointgoal2_c2_s4` | 54 | 59999 |
| `pointpush1_a2_s2` | 57 | 59999 |
| `pointpush1_c2_s2` | 58 | 59999 |

Measured capacity on this box: **7.5 GiB RSS, 1.9 cores, 3.07 GiB VRAM and ~1.67 it/s per run**,
essentially flat up to N=3. RAM is the binding resource -- N=3 leaves only 18-19% free, under the
20% floor -- so the queue ran at **N=2**. One 60k run is ~10 hours.

Note: the runner's printed `Iteration time` (which reads 500+ s here) is **not** a per-iteration
time and should be ignored; real throughput was measured from log progress.

---

## 10. Conclusions and what to do next

**The result is negative, and it is a real result:** the a2/c2 setup, tuned on PointGoal1, does
not transfer to PointGoal2, CarGoal1, Button1 or Push1 at cost limit 25.

1. **`lambda_max = 1.8` is too small for these envs.** It is a PointGoal1-derived constant and it
   saturates in 10 of 14 runs while cost stays at 2x budget.
2. **Cost limit 25 may be unreachable** on Button1 and PointGoal2. Nothing here establishes what
   these tasks can actually achieve.
3. **c2 shows no advantage over a2 off PointGoal1**, and the seed spread (eval cost 6 -> 92 on
   one env/arm) is too large to conclude anything from single seeds.

**Recommended before spending another ~5 days on the remaining 10 cells:**

* Run one short **unconstrained** probe per env (`--lambda_max 0`) to establish each task's
  reachable cost floor. Hours, not days. This answers whether 25 is attainable at all.
* Then either raise `lambda_max` or set a **per-env** budget. Applying one PointGoal1 number to
  four different tasks is the defect this sweep exposed.
* Only then requeue. Under the standing rule -- no training arm without a probe predicting it can
  change the outcome -- the remaining cells do not currently qualify.

**Not yet done:** matched checkpoint-based metrics via `scripts/analysis/sweep_report.py`
(a2 lacks the in-training eval panel, so the arm comparison above is indicative only).
