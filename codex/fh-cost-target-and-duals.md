# The FH-DCMPO cost target and the duals: what the code is, and what the measurements say

*2026-08-18/19 (Claude Code). Two questions, answered separately because they need different
instruments: (1) does the cost critic's TARGET rule change its tail calibration — 1-step / n-step
vs distributional TD(λ), over λ and window L; (2) once the target is fixed, does the cost critic
lag a moving actor, and can more critic updates or a slower actor fix it. Plus a code walkthrough
of where η, λ (global and per-state) and the cost target actually live.*

Related: [[cvpo-cost-critic-investigation]] (why Q_c was near-constant), [[fh-dcmpo-math]] (the
bound these numbers are terms in), [[cvpo-per-state-lambda]] (the per-state dual, still
unevaluated), [[distributional-cost-critic]] (why the head is quantile, not categorical),
[[hazard-stratified-replay]].

---

## TL;DR

1. **TD(λ) beats the n-step target decisively on the tail, and what does the work is the
   effective horizon — not the window length.** Offline, one frozen policy, identical data and
   critic init: CVaR₀.₉ tail error **−26.55 → −2.10**, VaR₀.₉ coverage **0.747 → 0.900**.
2. **Bigger `L` helps only because it permits a longer horizon.** At matched effective horizon,
   4× the window buys 0.58 of tail error; raising the horizon at fixed window buys 24.2. λ is the
   knob, `L` is the permission slip.
3. **It is NOT "more episode terminals".** Deleting the terminal-anchored atoms doesn't degrade
   the tail, it destroys the critic (level ratio 7.5–8.7×) — with γ_c = 1 there is no contraction,
   so those atoms are the *only* thing pinning the value function. But ~3.5% of target mass is
   already enough; more adds nothing.
4. **Offline calibration does not transfer online.** The same target rule that reaches coverage
   0.900 on a frozen policy reads 0.77 mid-training. So we tested critic lag directly:
   **4× cost-critic updates ≈ neutral, 8× is actively harmful** (cost rises 25.3 → 32.3 over
   training), and **halving the actor lr made safety worse, not better** (violations 28% → 46%).
   Critic lag is not the binding constraint.
5. **What no target rule fixed: the marginal distribution.** PIT-KS is pinned at 0.446–0.450 for
   every TD(λ) arm against a 0.017 critical value. Tail calibration and distributional
   calibration are different problems.

---

## 1. The code

### 1.1 The cost-critic target — the thing under test

Two rules exist. Both live in the quantile path, and the second falls back to the first whenever
its window is absent, so TD(λ) is inert unless `cost_n_step` is configured.

**Baseline: plain n-step.** `safe_rl/algorithms/safe_sac.py:588` —

```python
costs = costs.squeeze(-1)
bootstrap_mask = self._cost_bootstrap_mask(dones, bootstrap).squeeze(-1)
discount = self._cost_bootstrap_discount(effective_n_steps)
...
with torch.no_grad():
    next_actions, _ = self.policy.sample_with_log_prob(next_obs)
    next_obs_norm = self.policy.critic_obs_normalizer(next_critic_obs)
    target_thetas = [
        costs.unsqueeze(-1) + disc * bootstrap_mask.unsqueeze(-1) * target(next_obs_norm, next_actions)
        for target in self.policy.cost_critic_targets
    ]
```

One bootstrap, at horizon `n`. The whole target distribution is the target network's own
distribution, shifted. **That is the mechanism of the failure**: a bootstrapped target can only
reproduce the spread it already has, which is why the tail stayed 2.2–2.4× too narrow across four
arms that changed the constraint, the controller and the loss shape but never the target.

FH-DCMPO makes two overrides that define what "finite-horizon" means here
(`safe_rl/algorithms/fhdcmpo.py`):

```python
def _cost_bootstrap_discount(self, effective_n_steps):
    return 1.0          # gamma_c = 1: undiscounted sum over the remaining horizon

def _cost_bootstrap_mask(self, dones, bootstrap):
    return 1.0 - dones  # the episode boundary is REAL for a finite-horizon cost
```

The second is not cosmetic. The shared mask is `bootstrap + (1 - done)`, which keeps
bootstrapping through a time-limit truncation — right for an infinite-horizon discounted
objective, wrong here: `J_c = Σ_{t<T} c_t` has no terms after `T`, so bootstrapping across the
boundary adds a whole extra episode's cost to every target beneath it.

**The arm under test: distributional TD(λ)** (SDAC, arXiv:2301.10923).
`safe_rl/algorithms/fhdcmpo.py:200` for the weights:

```python
def td_lambda_weights(self, length: int) -> torch.Tensor:
    """w_j = (1-lam) lam^(j-1) for j < L, with ALL remaining mass lam^(L-1) on the longest
    return. Truncating without that remainder would quietly renormalise the mixture toward
    short returns -- the opposite of the point."""
    lam = self.cost_td_lambda
    j = torch.arange(length, device=self.device, dtype=torch.float32)
    w = (1.0 - lam) * lam**j
    w[-1] = lam ** (length - 1)
    return w / w.sum()
```

and `fhdcmpo.py:213` for the target itself:

```python
B, L = cost_window_returns.shape
w = self.td_lambda_weights(L)                       # [L]

with torch.no_grad():
    flat_obs  = cost_window_next_obs.reshape(B * L, -1)
    flat_norm = self.policy.critic_obs_normalizer(flat_obs)
    flat_act, _ = self.policy.sample_with_log_prob(flat_obs)
    targets = []
    for tgt in self.policy.cost_critic_targets:
        z = tgt(flat_norm, flat_act).reshape(B, L, -1)                     # [B, L, Nq]
        g = cost_window_returns.unsqueeze(-1) + cost_window_mask.unsqueeze(-1) * z
        targets.append(g.reshape(B, -1))                                   # [B, L*Nq]
    nq = targets[0].shape[1] // L
    # Each component contributes Nq equally-weighted atoms carrying w_j of the mass.
    atom_w = (w / nq).repeat_interleave(nq).unsqueeze(0).expand(B, -1)
```

So the target is a **mixture over j-step undiscounted returns**
`G_j = Σ_{k<j} c_k + m_j Z_c(s_{t+j})` with geometric weights `w_j`. Two properties are the whole
reason for it:

* every component past an episode boundary has `m_j = 0`, so it is the **exact realized remaining
  episodic cost** — a pure Monte-Carlo atom with no critic error in it;
* the mixture's spread is therefore sourced from real returns rather than from the target
  network's own distribution.

The per-step window comes from the replay buffer, `safe_rl/storage/replay_storage.py:586`:

```python
costs_w = view2d(self._data["costs"])[wt, ee]          # [B, L, C]
dones_w = view2d(self._data["dones"])[wt, ee].squeeze(-1)

# alive_j = 1 while step j is still inside the ORIGINAL episode (done not yet seen).
dones_shifted = torch.cat([torch.zeros_like(dones_w[..., :1]), dones_w[..., :-1]], dim=-1)
alive   = torch.cumprod(1.0 - dones_shifted, dim=-1)   # [B, L]
returns = torch.cumsum(costs_w * alive, dim=-1)        # frozen after the boundary

# still_j = 1 only if the episode survives step j itself -> bootstrap there, else pure MC.
still = torch.cumprod(1.0 - dones_w, dim=-1)           # [B, L]
```

Undiscounted by construction — `ReplayStorage(cost_gamma=1.0)`, set by the algorithm. Note there
are **two independent discounts** that both have to be neutralised for the target to actually be
undiscounted: this window sum, and `_cost_bootstrap_discount`.

### 1.2 The loss function

`safe_rl/modules/critic.py:657`. Quantile Huber (Dabney et al. 2018), with the one addition that
TD(λ) requires:

```python
u = target.unsqueeze(1) - theta.unsqueeze(2)                      # [batch, N, M]
abs_u = u.abs()
huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
weight = (tau_hat.view(1, -1, 1) - (u.detach() < 0).float()).abs()
per_pair = weight * huber / kappa                                 # [batch, N, M]
if target_weights is None:
    return per_pair.mean(dim=2).sum(dim=1)
# Weighted target samples. Needed for a TD(lambda) target distribution, which is a MIXTURE over
# n-step returns with geometric weights -- the atoms are not equally weighted, so the plain mean
# over M would silently flatten the mixture into a uniform one and discard the lambda weighting.
w = target_weights.unsqueeze(1)                                   # [batch, 1, M]
return (per_pair * w).sum(dim=2).sum(dim=1)
```

Three details that are each load-bearing:

* **`target_weights` is what makes TD(λ) real.** Without it, `.mean(dim=2)` over the `L*Nq` atoms
  would weight every mixture component equally and λ would have no effect at all.
* **`u.detach()` in the indicator.** The asymmetric weight `|τ̂ − 1{u<0}|` is a selector, not a
  differentiable function of θ; a gradient through it is a bug.
* **Returns `[batch]`, not a scalar.** The cost channel multiplies by hazard-stratified importance
  weights before reducing, so the caller owns the final `.mean()`.

### 1.3 η — the E-step temperature

η is **solved**, never learned. Two implementations, both exact.

**Global (SLSQP), `safe_rl/algorithms/mpo.py:205`:**

```python
def dual_eta(x):
    eta = x[0]
    z = q_np / eta
    zmax = z.max(axis=0, keepdims=True)
    lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
    return eta * eps + eta * float(np.mean(lse))

res = minimize(dual_eta, np.array([max(self.eta, 1e-3)]), method="SLSQP", bounds=[(1e-6, 1e6)])
```

Warm-started from the previous iterate; `zmax` subtraction is the mandatory log-sum-exp
stabilisation. CVPO reuses it with the cost term folded into the exponent
(`cvpo.py:686`): `self._solve_eta(q_np - lam * qc_np)`.

**Per-state (bisection, GPU), `safe_rl/common/per_state_dual.py:201`:**

```python
reduce = (lambda x: x) if per_state else (lambda x: x.mean())
lo = torch.full(shape, math.log(eta_min)); hi = torch.full(shape, math.log(eta_max))
for _ in range(iters):
    mid = 0.5 * (lo + hi)
    too_cold = reduce(per_state_kl(estep_weights(q_r, q_c, mid.exp(), lam))) > eps
    lo = torch.where(too_cold, mid, lo)
    hi = torch.where(too_cold, hi, mid)
return (0.5 * (lo + hi)).exp()
```

`dg/dη = ε − KL`, and KL is non-increasing in η, so `g` is convex in η and the root of `KL(η) = ε`
is its minimiser. Log-space over a **fixed** bracket — a data-dependent bracket would be
marginally tighter and would reintroduce a host synchronisation.

### 1.4 λ — global, via a PID controller

This is what all the arms in this note actually ran. λ is a **scalar per update**, driven by
realized episodic cost, not by Q_c's level. `safe_rl/algorithms/cvpo.py:465`:

```python
if not np.isfinite(realized) or realized <= 0.0:
    return                     # empty cost buffer (runner reports 0.0), not a measurement
if self._last_realized_cost is not None and realized == self._last_realized_cost:
    return                     # one step per NEW measurement, else the loop rings
self._last_realized_cost = realized
self._lambda_reports += 1
if self._lambda_reports < self.lambda_episodic_warmup:
    return
limit = self._episodic_cost_limit()
self._lambda_delta = (realized - limit) / max(limit, 1e-12)
self.lam = self._lambda_ctrl.update(self._lambda_delta)
```

`delta = (J_c − d)/d` is dimensionless, so the gains don't need retuning per cost limit. The
controller itself, `safe_rl/common/lambda_controller.py:62`:

```python
saturated_high = self.lam >= self.lam_max - 1e-12 and delta > 0.0
saturated_low  = self.lam <= 1e-12 and delta < 0.0
if not (self.anti_windup and (saturated_high or saturated_low)):
    self.integral += self.ki * delta
derivative = 0.0 if self._prev_delta is None else (delta - self._prev_delta)
self._prev_delta = delta
raw = self.kp * delta + self.integral + self.kd * derivative
self.lam = float(np.clip(raw, 0.0, self.lam_max))
```

And where λ enters the objective — `cvpo.py:921`, the only place it acts:

```python
eta, lam = self._solve_dual(q_np, qc_np)
combined = rescale_advantage(q, qc, lam) if self.rescale_by_lambda else (q - lam * qc)
weights = torch.softmax(combined / eta, dim=0)     # [N, B], columns sum to 1
```

> **The consequence that matters for reading every result below.** That softmax is *per state*, so
> any constant added to `Q_c` cancels. Only the **spread of Q_c across the candidate actions at
> one state** reaches the policy. The code logs this directly:
> ```python
> std_qr, std_qc = q_np.std(axis=0), qc_np.std(axis=0)
> lam_balanced = float(np.median(std_qr / np.maximum(std_qc, 1e-12)))
> ```
> Measured on these arms, `std_a(Q_c)` is **0.03–0.16 against a level of ~16**. A single scalar λ
> multiplying a near-constant Q_c can only shift cost pressure uniformly.

### 1.5 λ(s) — per-state, where it lives (implemented, not yet evaluated)

Separate code path, `safe_rl/algorithms/cvpo_per_state.py` + `safe_rl/common/per_state_dual.py`.
**None of the arms in this note use it.** The solver, `per_state_dual.py:139`:

```python
lo = torch.zeros(b); hi = torch.full((b,), lam_max)
for _ in range(iters):
    mid = 0.5 * (lo + hi)
    violating = expected_qc(q_r, q_c, eta, mid) > d
    lo = torch.where(violating, mid, lo)
    hi = torch.where(violating, hi, mid)
lam = 0.5 * (lo + hi)
# snap the KKT corners exactly, else the `inactive` certificate lies
inactive = expected_qc(q_r, q_c, eta, torch.zeros_like(lam)) <= d
at_cap   = expected_qc(q_r, q_c, eta, torch.full_like(lam, lam_max)) > d
lam = torch.where(inactive, torch.zeros_like(lam), lam)
return torch.where(at_cap, torch.full_like(lam, lam_max), lam)
```

`E_{q_b}[Q_c]` is non-increasing in `λ_b` (derivative `−Var_{q_b}(Q_c)/η`), so the root is unique
where the constraint is active. Block-coordinate: exact λ block, exact η block, ending on a λ
block so `E_{q_b}[Q_c] = d_b` holds exactly at every interior state.

No optimizer, no learning rate, no PID — consistent with CLAUDE.md rule 1. `CVPOPerState`
**rejects** `lambda_source` / `lambda_update` / `lambda_kp` / `lambda_kd` / `lambda_lr` /
`lambda_init` rather than ignoring them.

Wiring for the FH critic is `FHDCMPOPerState` (`safe_rl/algorithms/fhdcmpo_per_state.py`), pure
composition on the MRO `FHDCMPOPerState → CVPOPerState → FHDCMPO → CVPO`: `CVPOPerState`
supplies the dual, `FHDCMPO._estep_cost` supplies the finite-horizon cost statistic, and the only
written code is a kappa-ramp override, because the per-state MRO bypasses
`FHDCMPO._estep_weights` where the ramp normally advances.

### 1.6 The one thing added for the lag experiment

`cost_critic_updates_per_step` in `SafeSAC` — repeats *only* the cost-critic gradient step, each
on its own fresh batch, leaving the reward critic and actor update counts untouched. Default `1`,
so no existing config changes behaviour.

```python
for rep in range(self.cost_critic_updates_per_step):
    if rep > 0:
        # Every extra pass draws its own batch: the point of a higher ratio is more independent
        # data per actor step, not more passes over the same one.
        batch = self.storage.sample(self.batch_size)
        ...
    losses.append(self._one_cost_critic_update(...))
return {"cost_critic": sum(losses) / len(losses)}
```

---

## 2. Experiment 1 — the target rule (offline, policy frozen)

### 2.1 Why offline, and why that is the stronger test

A training run changes the policy, the buffer contents and the target rule at once, so two runs
differ in ways that have nothing to do with the target. Harness:
`scratchpad/cost_target_ablation.py`.

* **Policy frozen** at s2b `model_20000.pt`; one 200k-transition dataset captured at
  `alg.store_transition` (i.e. after the runner resolved dones, repaired truncation observations
  and assembled the bootstrap channel), replayed into every arm's storage.
* **Byte-identical critic init** across arms; identical 12k-update budget; fresh Adam.
* **Ground truth is exact.** Safety-Gymnasium episodes run a fixed T = 1000, so the realized
  undiscounted cost-to-go at every visited state is `(episode cost) − (cost before that step)`.
  6400 eval states from 64 complete episodes, identical inputs for every arm (a paired comparison).
* The `drop-MC` diagnostic path is validated against the real target code to **9e-7** relative loss.

What it cannot measure: reward, mean episodic cost, violation rate — those are properties of a
*policy*, which this harness deliberately holds fixed. They come from §3.

### 2.2 Results

Realized CVaR₀.₉ = **55.64**, realized mean cost-to-go = **14.08** (identical for all arms).
KS 95% critical value = **0.017**. `tailErr = predicted − realized CVaR₀.₉`; negative =
understates the tail = **unsafe direction**. `effH = Σ_j w_j · j`.

| arm | n | effH | MCatom% | MCmass% | level | PIT-KS | predCVaR | **tailErr** | cover |
|---|---|---|---|---|---|---|---|---|---|
| **n-step (n=10)** | 2 | 10.0 | — | — | 0.969 | 0.488 | 29.09 | **−26.55** | 0.747 |
| L64 λ=0.95 | 2 | 19.2 | 3.15 | 1.89 | 1.220 | 0.474 | 45.39 | **−10.25** | 0.867 |
| **L64 λ=0.98** (shipped) | 3 | 36.3 | 3.15 | 3.52 | 1.234 | 0.450 | 53.53 | **−2.10 ± 0.44** | 0.900 |
| L128 λ=0.98 | 1 | 46.2 | 6.12 | 4.41 | 1.180 | 0.446 | 53.93 | **−1.71** | 0.900 |
| L256 λ=0.9725 | 1 | 36.3 | 12.47 | 3.56 | 1.239 | 0.450 | 53.83 | **−1.80** | 0.901 |
| L64 λ=0.99 | 3 | 47.4 | 3.15 | 4.60 | 1.188 | 0.446 | 54.21 | **−1.43 ± 0.66** | 0.903 |
| L256 λ=0.98 | 1 | 49.7 | 12.47 | 4.86 | 1.199 | 0.446 | 54.81 | **−0.83** | 0.906 |
| L64 λ=0.995 | 3 | 54.9 | 3.15 | 5.32 | 1.170 | 0.446 | 54.36 | **−1.28 ± 0.80** | 0.902 |
| L256 λ=0.995 | 1 | 144.6 | 12.47 | 13.98 | 1.139 | 0.446 | 58.72 | **+3.09** | 0.908 |
| L64 λ=0.98 **drop-MC** | 1 | 36.3 | 3.15 | 3.52 | **7.499** | 0.907 | 173.09 | **+117.45** | 1.000 |
| L256 λ=0.98 **drop-MC** | 1 | 49.7 | 12.47 | 4.86 | **8.704** | 0.930 | 195.88 | **+140.24** | 1.000 |

`MCatom%` = share of target **atoms** past a terminal (what training logs as
`critic/cost_mc_frac`); `MCmass%` = share of target **mass** on them.

### 2.3 1-step / n-step vs the λ-average, read directly

* **n-step (effH 10) → TD(λ) at effH 36:** tail error −26.55 → −2.10, coverage 0.747 → 0.900.
  That is ~92% of the tail error removed, and coverage landing exactly on target.
* The n-step arm has the **best level** (0.969 vs 1.17–1.24) and the **worst tail**. Getting the
  mean right and the tail wrong is exactly the bootstrapped-spread failure: a 1-step/n-step target
  regresses onto the target network's own distribution, so it cannot manufacture spread it doesn't
  already have.
* Every TD(λ) arm **over**-reads the level by 14–24% — the conservative direction — and the
  over-read shrinks monotonically as effH grows (1.234 → 1.170).
* λ = 0.95 is not enough: effH 19 still leaves −10.25 and coverage 0.867. The gain is not "TD(λ)
  vs n-step" as a label; it is the horizon the λ-average actually reaches.

### 2.4 Does larger L help for a real reason, or just less truncation? — two controls

**Control A, matched effH with different L.** `L=128, λ=0.98` (effH 46.2, 6.1% MC atoms) vs
`L=64, λ=0.99` (effH 47.4, 3.2%): **−1.71 vs −1.43**, coverage 0.900 vs 0.903. Two-fold different
window, two-fold different terminal share, **same answer** — at twice the target-network cost.

**Control B, matched horizon at 4× the window.** `L=256, λ=0.9725` is tuned so effH = 36.3,
exactly the shipped `L=64, λ=0.98` arm: **−1.80 vs −2.10**. So quadrupling `L` at a matched
horizon buys **0.30–0.58** of tail error, while raising effH from 10 → 36 buys **24.2**. The
horizon effect is ~40–80× larger.

**Verdict:** larger `L` helps because it *permits* a longer effective horizon (at λ = 0.98 the mass
`λ^(L−1)` stranded on the truncated return is 28% at L=64 but 0.6% at L=256), not because it
reaches more episode terminals. **λ is the knob; `L` only has to be large enough not to truncate
it.**

**A confound worth naming**, because it will fool anyone reading `cost_mc_frac`: MC *mass* is a
near-deterministic function of effH (≈ effH/1000, since a window of expected length *j* crosses a
terminal with probability *j*/1000 in a 1000-step episode) — 1.89 / 3.52 / 4.41 / 4.60 / 5.32 /
13.98% tracks effH exactly. "Longer horizon" and "more terminals" move together in every arm that
varies λ or L. Only the drop-MC arm and the matched-horizon arm separate them.

### 2.5 But the terminal-anchored atoms are load-bearing — just not marginally

Deleting them (renormalising the mixture over surviving atoms) does **not** degrade the tail
slightly. It destroys the critic: level ratio **7.5–8.7×**, Q_c level 104 instead of 16.5, PIT-KS
0.91–0.93, loss 561–672 vs 204.

Mechanism: with γ_c = 1 the cost backup is **not a contraction**, so the only thing pinning the
value function is the boundary condition `Z_c(s, u=0) = 0`, which reaches the loss *exclusively*
through the `mask = 0` atoms. 3.5% of target mass is enough to anchor it; 4.9% is no better.

This is why "L=256 reaches more terminals" is the wrong explanation for its benefit, **and** why
the MC anchor must never be removed as an optimisation.

### 2.6 Saturation, and the far end

effH ≈ 46–55 is the sweet spot. At effH 145 (`L=256, λ=0.995`) the sign flips to **+3.09** —
now over-stating the tail — with 14% MC mass and the highest loss (242.5). More horizon is not
monotonically better.

### 2.7 What no target rule fixed

PIT-KS is **0.446–0.450 for every TD(λ) arm** (0.488 for n-step) against a 0.017 critical value.
The marginal distribution is badly non-uniform in all of them. The target horizon moves the tail
statistic and the level; it does not make the critic distributionally calibrated.

---

## 3. Experiment 2 — is the critic lagging the actor? (online)

### 3.1 Why the question arose

The offline arms reach coverage 0.900 on a frozen policy. The same target rule read **0.77**
mid-training (`s2b_tdlam_L64_lam099_seed2`, predicted CVaR 42 vs realized 82, level 0.50–0.84).
Hypothesis: the actor changes faster than the critic can track it.

Cost target fixed at **L=64, λ=0.995** in every arm; seed 2; 20k iters; replay, networks, cost
limit and `qc_thres` untouched. One factor per arm.

### 3.2 Results

Deterministic policy, **single env** (see §4.1), 50 episodes for a1/a2/a4, 200 for a3:

| arm | change from base | reward | mean cost | violations | realized CVaR₀.₉ |
|---|---|---|---|---|---|
| **a1 base** | — (1 cost : 1 actor) | 14.09 | 18.34 | **28%** | 61.4 |
| **a2** | 4 cost : 1 actor | 14.16 | 17.66 | **26%** | 57.4 |
| **a3** | 8 cost : 1 actor | 12.77 | 32.26 | **40.5%** | 142.1 |
| **a4** | actor_lr 3e-4 → 1.5e-4 | 22.22 | 26.14 | **46%** | 68.2 |

a3 over training (200 ep per checkpoint, with calibration):

| a3 @ | reward | cost | violations | pred CVaR₀.₉ | real CVaR₀.₉ | VaR cover | level |
|---|---|---|---|---|---|---|---|
| 10k | 14.00 | 25.27 | 38.5% | 64.12 | 112.97 | 0.586 | 0.423 |
| 15k | 13.33 | 31.84 | 43.5% | 60.67 | 67.35 | 0.826 | 0.919 |
| 20k | 12.77 | 32.26 | 40.5% | 65.33 | 102.52 | 0.777 | 0.684 |

### 3.3 Reading

* **More cost-critic updates does not fix calibration, and past 4:1 it hurts.** 4:1 is
  approximately neutral (within noise of base on every metric). 8:1 makes cost *rise* over
  training, 25.3 → 32.3, with coverage still 0.78 and 40% violations. Cost per iteration is
  **5.1×** base (993 s vs 196 s per 1000 iters), so it is expensive as well as harmful.
* **Slowing the actor moved things the opposite way.** Halving actor_lr gave the *best* reward
  (22.22 vs 14.09) and the *worst* safety (46% violations vs 28%). It did not help the critic keep
  up; it let the policy exploit reward with less constraint pressure.
* **Non-stationarity is constant by construction, so it was never going to self-heal.** `kl_mean`
  sits at ~0.008 for all 60k iterations of the s2b reference — the M-step trust region binds at
  its `kl_mean_constraint: 0.01`. Policy drift per update does not decay, so critic lag does not
  decay either, and more iterations do not fix it.

**Conclusion: critic lag is not the binding constraint.** Neither more critic gradient steps nor a
slower actor recovers the offline calibration. The residual online error is not a
"not-enough-updates" problem.

### 3.4 What the evidence points at instead

The E-step can only use the **across-action spread** of Q_c at a fixed state (§1.4). Measured:

* `std_a(Q_c)` = **0.03–0.16** against a level of ~16, giving `lambda_balanced` ≈ 0.6–1.0.
* A 900-step, 16-action branching probe on the real simulator could **not separate the true action
  effect on cost-to-go from rollout noise** (noise sd 1.8, true effect variance measured as 0),
  with 37–58% of probed states incurring no cost on any branch.

So the constraint is being enforced by something close to a state-independent penalty — which is
the closing question of [[cvpo-cost-critic-investigation]] and the motivation for
[[cvpo-per-state-lambda]]. That path is wired to the FH critic now (§1.5) and queued; its M5 gate
is `ess_per_state`, and a smoke run shows ess 53/64 with no collapse.

Also relevant: every arm here runs `fh_risk_mode: mean`, so **κ = 0 throughout** — the E-step
constrains the *mean* cost-to-go. The tail metrics are diagnostics of the critic, not something
the constraint currently acts on.

---

## 4. Protocol notes — two measurement bugs that changed conclusions

### 4.1 `--num_envs 8` with one seed gives ~7 distinct episodes, not 50

`SafetyGymnasiumVecEnv.reset` tiles a single seed across sub-envs *deliberately* ("so all sub-envs
share the SAME task"). Evaluating with `--num_envs 8 --episodes 50` therefore reports 50 episodes
drawn from ~7 layouts. Same s2b checkpoint:

| protocol | reward | cost | violations | CVaR₀.₉ |
|---|---|---|---|---|
| `--num_envs 8` (≈7 distinct) | 15.61 | 26.88 | 80.0% | 48.0 |
| `--num_envs 1` (50 distinct) | 14.05 | 26.94 | 40.0% | 94.2 |

**Always `--num_envs 1` for tail statistics.** The evaluator already warns when distinct-episode
count is low.

### 4.2 Eval-seed noise exceeds the between-arm effects at n=50

Same s1b checkpoint, two eval seeds: cost **34.80 vs 23.68**, CVaR₀.₉ **102.8 vs 70.2**. At n=50,
CVaR₀.₉ is the mean of the 5 worst episodes. Report ≥200 episodes with bootstrap CIs before
claiming a violation-rate difference of a few points.

### 4.3 Repo fixes made along the way

* `scripts/eval/cost_critic_rank_probe.py` could not run on **any** FH arm — it built a raw env
  without the `horizon_feature` column (60 vs 61 obs, checkpoint load failure). Now reproduces
  `u_t = (T−t)/T` and rewinds the counter with every `sim.restore`.
* Same probe's `--crn` reseeded the RNG *before* sampling each branch action, so all branches drew
  the **identical** action — silently yielding `std_a(Q_c) = 0`, no defined ρ at any state, and
  "0% of the true action effect" regardless of critic quality. Fixed by drawing the action before
  the reseed.
* `scripts/train/train_safety_gymnasium.py` named the run dir by a second-resolution timestamp with
  no uniqueness guard. Two arms launched in the same second **shared a directory** and the slower
  run's checkpoints overwrote the faster one's, destroying an arm. Now claims the directory
  exclusively with a `_1`… suffix fallback.

---

## 5. Recommendation

**Smallest change that improves the tail without touching anything else:
`cost_td_lambda: 0.98 → 0.99` at `cost_n_step: 64`.** One YAML value, zero extra compute. Offline
it moves tail error −2.10 → −1.43 and coverage 0.900 → 0.903, and reduces the level over-read
1.234 → 1.188. `L=256` is measurably the best tail (−0.83) but costs 4× the target-network
evaluations — visible as 111 vs 170 steps/s in training — for 0.6 of CVaR that λ gets most of for
free.

**Stated honestly: at one refit seed this is inside noise.** With 3 refit seeds, λ ∈ {0.98, 0.99,
0.995} give −2.10 ± 0.44, −1.43 ± 0.66, −1.28 ± 0.80 — overlapping. λ = 0.995 is nominally best on
every offline metric. The online confirmation at λ = 0.99 (`q_lam099`, seed 2) came back
**neutral**: cost 26.94 → 25.42 and violations 40% → 36% vs s2b, but realized CVaR₀.₉ *worse*
(94.2 → 110.4) and reward flat (14.05 → 14.20).

**What is NOT worth spending on:** more cost-critic updates (§3), a slower actor (§3), or more
iterations to resolve the λ choice (the effect is smaller than the eval noise of §4.2).

**Where the remaining error actually is**, in priority order:
1. **Q_c cannot rank actions** (`std_a(Q_c)` ≈ 0.03–0.16 at level ~16), so a scalar λ applies an
   almost uniform penalty. Per-state λ(s) is wired and queued; its gate is `ess_per_state`, and
   reverting to shared λ is a legitimate result.
2. **Marginal calibration** (PIT-KS 0.45 vs 0.017 critical) is untouched by any target rule and
   needs the S3 recalibration path, which is explicitly not implemented for the quantile head.
3. **Online ≠ offline calibration** even at a fixed target and constant policy-drift rate — worth
   understanding before adding machinery on top.
