# QR-DMPO — the math actually implemented

*Written 2026-08-17 from the code, not from the papers. Every formula below is traceable to a
file:line. The MPO E-step/M-step derivation is **not** repeated here — see
[[mpo-estep-mstep-math]]. This note covers the two things QR-DMPO adds on top: a **quantile**
cost critic, and the **constrained** E-step that consumes it.*

QR-DMPO = MPO's actor (E-step / M-step) + CVPO's cost-constrained E-step + a QR-DQN quantile
critic on **both** channels. Config: `config/safety_gymnasium_qrdmpo_goal1.yaml`.

---

## 0. Notation

| symbol | meaning |
|---|---|
| $s, a$ | state, action |
| $Q_r(s,a)$, $Q_c(s,a)$ | scalar reward / cost action-values |
| $Z_c(s,a)$ | the **distribution** of discounted cost-to-go, not its mean |
| $N$ | number of quantiles (64) |
| $\hat\tau_k$ | the $k$-th quantile fraction |
| $\theta_k(s,a)$ | the $k$-th learned quantile **location** |
| $\eta$ | E-step temperature (dual variable for the KL bound $\varepsilon$) |
| $\lambda$ | cost multiplier |
| $d$ | cost threshold in $Q_c$ units (`qc_thres`) |
| $n$ | n-step horizon (10) |

---

## 1. The quantile cost critic

### 1.1 Representation

Where a categorical (C51) critic fixes the atom *positions* and learns their probabilities,
QR-DQN fixes the probabilities and learns the *positions* (`critic.py:682`):

$$Z_c(s,a)\;\approx\;\frac1N\sum_{k=1}^{N}\delta_{\theta_k(s,a)},\qquad
\hat\tau_k=\frac{2k-1}{2N}$$

Each of the $N$ locations carries equal mass $1/N$ and sits at the midpoint fraction
$\hat\tau_k$. Two implementation details (`critic.py:760-770`):

- `nonneg=True` applies a **softplus**, making $\theta_k \ge 0$ structurally — cost cannot be
  negative.
- The output is **sorted** before use. Nothing constrains the head to emit monotone values,
  and crossed quantiles corrupt every statistic read off them. Sorting only permutes, so it
  is differentiable w.r.t. the values.

The motivation for the swap is zero-inflation: on these tasks ~40–60% of episodes incur zero
cost, so a fixed support spends most of its atoms on a value carrying no information, and can
still saturate at the upper edge.

### 1.2 Distributional Bellman target

Per-quantile, with no projection step — shifting and scaling the locations *is* the backup
(`safe_sac.py`, `_update_cost_critic_quantile`):

$$(\mathcal{T}\theta')_j \;=\; c \;+\; \gamma^{\,n}\, m \cdot \theta'_j(s',a'),
\qquad a' \sim \pi(\cdot\mid s')$$

with $m = \text{bootstrap} + (1-\text{done})$ the truncation-aware mask, and $\gamma^n$ the
per-sample n-step discount. Two deliberate departures from the reward channel:

- **no entropy term** — MPO explores through the E-step KL trust region, not a temperature;
- **no min-over-twins** — understating cost is the *unsafe* direction, and the cost channel
  runs a single critic.

### 1.3 The loss — and the κ correction

Quantile Huber loss (`critic.py:656-679`), with $u_{jk} = (\mathcal{T}\theta')_j - \theta_k$:

$$\mathcal{L}
=\sum_{k=1}^{N}\ \frac1M\sum_{j=1}^{M}
\Big|\hat\tau_k-\mathbb{1}\{u_{jk}<0\}\Big|\cdot\frac{L_\kappa(u_{jk})}{\kappa},
\qquad
L_\kappa(u)=\begin{cases}\tfrac12u^2 & |u|\le\kappa\\[2pt]
\kappa\big(|u|-\tfrac12\kappa\big) & |u|>\kappa\end{cases}$$

Reduction is **mean over the $M$ target samples, sum over the $N$ predicted quantiles**. The
asymmetric weight is evaluated on `u.detach()`: the indicator is a selector, not a
differentiable function of $\theta$.

**The gradient, which is where I got it wrong.** Note the $/\kappa$ — it is present in the
code and it matters:

$$\frac{\partial}{\partial\theta_k}\ \frac{L_\kappa(u)}{\kappa}
=\begin{cases}-\,u/\kappa & |u|\le\kappa \quad(\text{slope }\le 1)\\[2pt]
-\operatorname{sign}(u) & |u|>\kappa \quad(\text{slope exactly }1)\end{cases}$$

So the gradient magnitude is **bounded by 1 regardless of $\kappa$**. Raising $\kappa$ does
**not** strengthen the signal on large errors — it only moves the quadratic/linear boundary
outward, and inside the quadratic region it *divides* the slope by $\kappa$.

> **Correction to an earlier claim.** I measured that on cost targets $>10$, 99% of quantile
> errors sit in the linear regime with mean $|u|\approx15$, and concluded "$\kappa=5$ gives
> the tail a 5× stronger signal". **That is wrong.** With the $/\kappa$ normalisation the
> linear-regime gradient is 1 for any $\kappa$; raising $\kappa$ from 1 to 5 *weakens* the
> gradient for errors in $(1,5]$ (from 1 to $u/5\le1$) and leaves errors $>5$ unchanged.
> The saturation measurement itself stands — the tail really does get a constant, scale-free
> push — but $\kappa$ is not the lever that fixes it.

**This was tested, and the prediction failed.** QR-DMPO / SafetyPointGoal1, seed 2, GPU 0,
`--deterministic`, 60k iterations, `kappa` the only change (cost critic only; the reward
critic stayed at 1.0). Prediction on record before launch: *"marginal tail ratio at $x=10$
rises from 0.25 toward 1.0, violations fall below 26%."*

| 50-episode eval | $\kappa=1$ | $\kappa=5$ | |
|---|---|---|---|
| reward | 24.06 | 23.73 | −0.33 |
| cost mean | **16.24** | **28.40** | **+12.16** |
| CVaR$_{0.9}$ | **57.40** | **111.80** | **+54.4** |
| over budget | **26.0%** | **38.0%** | **+12 pts** |

Training tail-average (last 10k): reward +0.80, cost +3.66, **$\lambda$ −1.05** (2.83 → 1.78).

The whole pattern follows from the gradient above. A larger $\kappa$ shrinks the gradient in
the mid-error range where most cost learning happens, so the critic becomes *less* accurate
and under-reports danger; $\lambda$ therefore settles **lower**, the constraint binds less,
and realised cost and its tail both rise. Reward is unchanged because the reward critic was
untouched.

**Conclusion: $\kappa$ is not the lever.** Confirmed by derivation and by experiment. The
saturation mechanism is real but needs a different fix — a loss whose gradient scales with
the error (e.g. dropping the $/\kappa$ normalisation, or a scale-aware target), not a larger
threshold. Config kept at `config/safety_gymnasium_qrdmpo_kappa5.yaml` for reproduction; do
not adopt it.

*Process note: the $/\kappa$ was visible in `critic.py:679` the whole time. Deriving the
gradient before launching would have cost minutes and saved a 2.2 h run.*

The real consequence of the capped gradient: **the critic climbs toward a large target at a
rate independent of how far away it is.** Reaching a value of 40 from 2 takes ~20× as many
updates as reaching 4 from 2. That is a plausible mechanism for a persistently thin upper
tail, and it is *not* addressed by $\kappa$.

### 1.4 Scalarisation

$$Q_c(s,a)=\mathbb{E}[Z_c]=\frac1N\sum_k\theta_k(s,a)$$

(`critic.py`, `get_value`). `get_dist` is the identity — for a quantile critic the forward
output already *is* the distribution — which is what lets both critic types be scalarised
through one call.

`get_cvar`, `get_quantile` and `risk_value` deliberately **raise `NotImplementedError`**
("Phase 2"). Explicit failure beats silently falling back to the mean, which would look like
"risk conditioning did nothing" rather than "risk conditioning is not wired".

---

## 2. The constrained E-step

MPO's E-step solves for a non-parametric $q$ maximising $\mathbb{E}_q[Q_r]$ under
$\mathrm{KL}(q\|\pi_{old})\le\varepsilon$. CVPO adds the cost constraint
$\mathbb{E}_q[Q_c]\le d$ (`cvpo.py`, `_estep_weights`).

### 2.1 Weights

With $N$ actions sampled per state, the closed-form solution is a softmax **over the action
axis**:

$$w_{ij}\;=\;\frac{\exp\!\big(A_{ij}/\eta\big)}{\sum_{j'}\exp\!\big(A_{ij'}/\eta\big)},
\qquad A_{ij}=Q_r(s_i,a_{ij})-\lambda\,Q_c(s_i,a_{ij})$$

Optionally (`rescale_by_lambda`, `lambda_controller.py:102`):

$$A_{ij}=\frac{Q_r-\lambda Q_c}{1+\lambda}$$

Dividing by a positive constant cannot flip a sign or reorder actions, so the softmax sees
the same preference ordering; only the effective temperature changes.

**Only the spread across candidate actions survives.** A constant added to $Q_c$ at a given
state cancels in the per-state normalisation. This is why absolute calibration of the cost
critic matters less than its *within-state ranking* — and why the diagnostic
`estep_std_qc` is logged.

### 2.2 The dual

$$g(\eta,\lambda)\;=\;\eta\varepsilon\;+\;\lambda d\;+\;
\eta\,\mathbb{E}_{s}\!\left[\log\frac1N\sum_j
\exp\!\Big(\tfrac{Q_r(s,a_j)-\lambda Q_c(s,a_j)}{\eta}\Big)\right]$$

minimised over $\eta>0$, $\lambda\in[0,\lambda_{max}]$, jointly convex. Implemented in
`cvpo.py`, `_solve_dual`, with a log-sum-exp stabilised by subtracting the per-state max.

Two modes:
- **`"dual"`** — joint SLSQP over $(\eta,\lambda)$ each batch. Snaps $\lambda$ to a bound
  whenever the sampled action set cannot reach $\mathbb{E}_q[Q_c]=d$ (bang-bang; see
  [[cvpo-negative-result]]).
- **`"grad"`** (default) — $\lambda$ held fixed, so $\lambda d$ is constant in $\eta$ and the
  objective reduces to MPO's 1-D dual on the shifted exponent $Q_r-\lambda Q_c$. $\lambda$ is
  then moved by a slow PID controller instead.

---

## 3. Cost units: `qc_scale`

The budget is **episodic** (e.g. 25 undiscounted), but the critic speaks **discounted
cost-to-go**. The conversion (`cost_scaling.py`, `measured_qc_scale`):

$$\texttt{qc\_scale}=\frac{\mathbb{E}[G_c(s_0)]}{\mathbb{E}[J_c]},\qquad
G_c=\sum_t\gamma^t c_t,\quad J_c=\sum_t c_t
\qquad\Longrightarrow\qquad
d=\texttt{cost\_limit}\times\texttt{qc\_scale}$$

Ratio of the **mean discounted return to the mean undiscounted cost** — not the mean of
per-episode ratios, because ~10–40% of episodes have $J_c=0$ and would divide by zero.

Analytic value for a uniform cost rate over $T=1000$, $\gamma=0.99$ is
$\tfrac{1-\gamma^{T}}{T(1-\gamma)}\approx0.10$. Measured values differ because cost is not
uniform in time:

| task | measured `qc_scale` | $d$ at limit 25 |
|---|---|---|
| SafetyPointGoal1 | 0.0764 | 1.91 |
| SafetyCarGoal1 | 0.09198 | 2.30 |

**A mis-set `qc_scale` silently changes the budget.** Running CarGoal1 with PointGoal1's
0.0764 enforces $d=1.91$ instead of 2.30 — a 20% *tighter* constraint than the stated limit
of 25. Correcting it therefore *loosens* the constraint and *raises* cost, which is what was
measured (+14.8 cost at matched iterations). It is a correctness fix, not a performance one.

---

## 4. The λ controller

In `"grad"` mode $\lambda$ follows a PID law on the constraint violation
(`lambda_controller.py`), with anti-windup at the bounds and $\lambda\in[0,\lambda_{max}]$.
`lambda_source` selects the error signal:

- `qspace` — $\mathbb{E}_q[Q_c]-d$, i.e. in critic units;
- `episodic` — realised $J_c-\texttt{cost\_limit}$, i.e. in true episodic units.

A useful diagnostic is $\lambda_{\text{balanced}}=\operatorname{median}_s
\big(\mathrm{std}_a Q_r / \mathrm{std}_a Q_c\big)$ — where $\lambda$ *should* settle to weigh
reward and cost spreads equally. Below that, the cost term barely perturbs the softmax; this
is the same scale argument as [[pid-lambda-advantage-scale]].

---

## 5. M-step

Unchanged from MPO: weighted maximum likelihood under decoupled mean/covariance KL trust
regions, multipliers by dual ascent. See [[mpo-estep-mstep-math]] §2. QR-DMPO changes nothing
here — the quantile critic affects the M-step only through the weights $w_{ij}$.

---

## 6. Measured properties (2026-08-17, seed 2)

| property | PointGoal1 | CarGoal1 |
|---|---|---|
| frac. of quantile errors in the linear regime (targets >10) | 0.982 | 0.991 |
| mean $\|u\|/\kappa$ in that bucket | 14.2 | 14.9 |
| predicted-marginal tail ratio $P_{pred}(>10)/P_{real}(>10)$ | 5.57 | 0.25 |
| realised cost-to-go, mean / max | 2.09 / 40.3 | 2.84 / 47.5 |

### 6.1 Reference results (50-episode deterministic eval, budget 25)

QR-DMPO's default configuration is the best safety point measured across four algorithms.

| method | task | n seeds | reward | cost | CVaR$_{0.9}$ | violations |
|---|---|---|---|---|---|---|
| **QR-DMPO** | PointGoal1 | 3 | 23.84 | **18.97** | **56.9** | **32.0%** |
| DMPO (categorical) | PointGoal1 | 3 | **25.70** | 24.68 | 71.5 | 43.3% |
| CVPO (scalar) | PointGoal1 | 3 | 23.31 | 26.73 | 73.1 | 43.3% |
| PPOL-PID (FSRL gains) | PointGoal1 | 3 | 21.65 | 26.50 | 76.7 | 40.7% |
| QR-DMPO | CarGoal1 | 1 | 32.30 | 28.68 | 121.2 | 34.0% |
| DMPO | CarGoal1 | 1 | 33.24 | 24.84 | 79.4 | 46.0% |

Note every method violates the budget in **32–54% of episodes** while several report a
compliant *mean* — the mean is not a safety statistic. QR-DMPO's seed spread on cost is
$\pm3.05$ (PointGoal1, n=3), the largest of the four, so single-seed differences below ~3 are
not interpretable.

**Health warnings on the last two rows.** The tail ratios come from 6–12 episode rollouts and
are noisy — the same policy measured twice gave max cost-to-go 18 and 47. An earlier claim
that the two tasks differ ~5× in cost scale did **not** survive a larger sample (2.09 vs
2.84). Treat the sign of the tail ratio as suggestive, not established.

**A separate methodological trap.** Per-state *coverage* (predicted $q_\alpha$ vs that state's
single realised return) is **not** a valid calibration test here: if $Z_c(s,a)$ is
near-deterministic, coverage collapses toward 0.5 at every $\alpha$ regardless of correctness.
Verified by refitting the critic on Monte-Carlo returns — correct by construction for the
conditional — which *lowered* coverage 0.675 → 0.572. The valid test is the **marginal**: the
mixture of per-state predicted distributions must equal the marginal of realised returns (law
of total probability), which holds however narrow each conditional is.
