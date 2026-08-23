# MPO E-step, M-step and action bounding — the math, and where our code differs from Acme

*2026-07-30 (Claude Code). Companion to [[mpo-vs-acme-reference]], which carries the code-level
diff and the mjlab Ant-Flat A/B. This note is the derivation: what each step actually optimizes,
the closed forms, and the exact mathematical content of each deviation from
[acme/jax/losses/mpo.py](https://github.com/google-deepmind/acme/blob/master/acme/jax/losses/mpo.py).*

Reference: Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation*, ICLR 2018
(arXiv:1806.06920); MO-MPO, arXiv:2005.07513.

## Notation

| symbol | meaning |
|---|---|
| $B$ | states $s$ in the replay minibatch (256) |
| $N$ | candidate actions sampled per state (64) |
| $D$ | action dimensions (8 for Ant) |
| $\pi_{\text{old}}$ | frozen Polyak target actor — samples the candidates, anchors both trust regions |
| $\pi_\theta$ | online actor being optimized |
| $x$ | **pre-tanh** latent, $x \sim \mathcal N(\mu(s), \operatorname{diag}\sigma(s)^2)$ |
| $a$ | action, $a = b + c \odot \tanh(x)$, with $b=\tfrac{high+low}{2}$, $c=\tfrac{high-low}{2}$ |

Both policies are diagonal Gaussians **in $x$**. The action distribution is the pushforward of that
Gaussian through $\tanh$. Section 3 shows why this distinction costs us nothing.

---

# 1. E-step

## 1.1 The problem

The E-step improves on the current policy *non-parametrically* — it does not touch $\theta$. It asks
for the best reweighting $q$ of the old policy that stays inside a KL trust region:

$$
\max_{q}\; \mathbb E_{s}\,\mathbb E_{a\sim q(\cdot|s)}\big[Q(s,a)\big]
\qquad\text{s.t.}\qquad
\mathbb E_{s}\big[\mathrm{KL}\big(q(\cdot|s)\,\|\,\pi_{\text{old}}(\cdot|s)\big)\big]\le\varepsilon .
$$

## 1.2 Closed form

Forming the Lagrangian and taking the variational derivative in $q$ gives a Gibbs tilting of the old
policy — an exponential reweighting by the tempered $Q$:

$$
q^*(a|s)=\frac{\pi_{\text{old}}(a|s)\,\exp\!\big(Q(s,a)/\eta\big)}{Z(s)},
\qquad
Z(s)=\int \pi_{\text{old}}(a|s)\,e^{Q(s,a)/\eta}\,da .
$$

$\eta>0$ is the multiplier on the KL constraint. **This is the structural reason no gradient ever
flows through the critic in MPO**: the safety/reward signal reaches the policy as *sample weights*,
never as $\nabla_a Q$.

## 1.3 The dual

Substituting $q^*$ back gives a convex, one-dimensional dual in $\eta$:

$$
g(\eta)=\eta\,\varepsilon+\eta\,\mathbb E_{s}\Big[\log\!\int \pi_{\text{old}}(a|s)\,e^{Q(s,a)/\eta}\,da\Big].
$$

With $a_1..a_N\sim\pi_{\text{old}}(\cdot|s)$ the integral is a plain Monte-Carlo mean, so the
estimator actually minimized in `mpo.py::_solve_eta` is

$$
\hat g(\eta)=\eta\,\varepsilon+\frac{\eta}{B}\sum_{s}\log\Big[\tfrac1N\sum_{i}e^{Q(s,a_i)/\eta}\Big],
$$

and the resulting weights are a softmax **over the $N$ samples, per state**:

$$
w_i(s)=\frac{e^{Q(s,a_i)/\eta^*}}{\sum_j e^{Q(s,a_j)/\eta^*}}
=\operatorname{softmax}_i\!\big(Q(s,a_i)/\eta^*\big),
\qquad \sum_i w_i(s)=1 .
$$

Note the $\tfrac1N$ inside the log: it is $\log\operatorname{mean}\exp$, not
$\log\operatorname{sum}\exp$. Using the sum instead is a common bug — it shifts the dual by
$\eta\log N$ and biases $\eta^*$. Our code uses `np.mean`; Acme subtracts `log_num_actions`
explicitly. Both are correct.

## 1.4 The identity that makes the solve verifiable

Let $L(\eta)=\tfrac1B\sum_s\log\big[\tfrac1N\sum_i e^{Q_i/\eta}\big]$. Then

$$
L'(\eta)=-\frac{1}{\eta^{2}}\cdot\frac1B\sum_s\sum_i w_i Q_i,
\qquad
g'(\eta)=\varepsilon+L(\eta)-\frac{1}{\eta}\cdot\frac1B\sum_s\sum_i w_i Q_i .
$$

Separately, the empirical KL of $q^*$ (weights $w_i$) against $\pi_{\text{old}}$ (uniform $1/N$ over
its own samples) is $\mathrm{KL}_s=\sum_i w_i\log(N w_i)$, and since
$\log(N w_i)=Q_i/\eta-\log\big[\tfrac1N\sum_j e^{Q_j/\eta}\big]$,

$$
\mathbb E_s[\mathrm{KL}_s]=\frac{1}{\eta}\cdot\frac1B\sum_s\sum_i w_i Q_i-L(\eta).
$$

Adding the two lines:

$$
\boxed{\;g'(\eta)=\varepsilon-\mathbb E_s\big[\mathrm{KL}(q^*\|\pi_{\text{old}})\big]\;}
$$

So at an interior optimum the **realized KL equals $\varepsilon$ exactly**. This is the KKT residual
`dual_residual_eta`, and `tests/test_mpo.py::test_mpo_dual_optimum_satisfies_kkt_on_eta` asserts it.
In all three live Ant-Flat runs it reads $\sim10^{-3}$ against $\varepsilon=0.1$, with SLSQP
converging in 1–2 iterations from the warm start.

## 1.5 Difference vs Acme: solver only

Acme's `compute_weights_and_temperature_loss` uses

$$
\text{loss}_\eta=\eta\Big(\varepsilon+\mathbb E_s\operatorname{logsumexp}_i(Q_i/\eta)-\log N\Big)
$$

which, since $\operatorname{logsumexp}-\log N\equiv\log\operatorname{mean}\exp$, is **the same
function** $\hat g(\eta)$. The difference is entirely in how it is minimized:

| | ours | Acme |
|---|---|---|
| $\eta$ is | output of a convex solve (SLSQP), warm-started | a trainable parameter, $\eta=\text{softplus}(\rho)+10^{-8}$ |
| update | solved to optimality every iteration | **one gradient step**: $\eta \leftarrow \eta-\text{lr}\,(\varepsilon-\mathrm{KL})$ |
| guarantees | $g'(\eta^*)\approx0$, so $\mathrm{KL}=\varepsilon$ each update | tracks $\varepsilon$ with lag; never exactly on it |

Acme's update *is* an integral controller on the constraint. Ours is the exact solve the C-TruDi
spec requires ("duals are computed, never learned"). Same objective, strictly better solution — and
this is the one axis where our implementation is unambiguously ahead of the reference.

---

# 2. M-step

## 2.1 The problem

Project the non-parametric $q^*$ back onto the parametric family, under a trust region against the
*old* policy:

$$
\max_\theta\;\mathbb E_s\sum_i w_i\log\pi_\theta(a_i|s)
\qquad\text{s.t.}\qquad
\mathbb E_s\big[\mathrm{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)\big]\le\beta .
$$

Note the KL direction is $\mathrm{KL}(\text{old}\,\|\,\text{new})$ — mode-covering, and the one with
a closed form in the old policy's statistics. It is solved as a penalized Lagrangian
$\mathcal L(\theta,\alpha)=J(\theta)+\alpha(\beta-\mathrm{KL})$ with $\alpha$ by dual ascent.

**This $\beta$ is not the E-step's $\varepsilon$.** Two different trust regions between two different
pairs of distributions; conflating them is a standing failure mode.

## 2.2 Why the KL is split in two

For diagonal Gaussians, MPO decomposes the single trust region into a mean part and a covariance
part, each with its own budget and its own multiplier:

$$
\mathrm{KL}_\mu=\mathbb E_s\,\mathrm{KL}\big(\mathcal N(\mu_{\text{old}},\Sigma_{\text{old}})\,\|\,\mathcal N(\mu_\theta,\Sigma_{\text{old}})\big),
\qquad
\mathrm{KL}_\Sigma=\mathbb E_s\,\mathrm{KL}\big(\mathcal N(\mu_{\text{old}},\Sigma_{\text{old}})\,\|\,\mathcal N(\mu_{\text{old}},\Sigma_\theta)\big).
$$

Using $\mathrm{KL}\big(\mathcal N(\mu_1,\sigma_1^2)\|\mathcal N(\mu_2,\sigma_2^2)\big)
=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$, the two
collapse to clean per-dimension forms:

$$
\mathrm{KL}_{\mu,d}=\frac{(\mu_{\text{old},d}-\mu_{\theta,d})^2}{2\,\sigma_{\text{old},d}^2},
\qquad
\mathrm{KL}_{\Sigma,d}=\log\frac{\sigma_{\theta,d}}{\sigma_{\text{old},d}}
+\frac{\sigma_{\text{old},d}^2}{2\,\sigma_{\theta,d}^2}-\frac12 .
$$

$\mathrm{KL}_\mu$ is a pure Mahalanobis step measured in units of the **old** $\sigma$;
$\mathrm{KL}_\Sigma$ is a pure scale divergence, independent of where the mean went.

**Why bother:** under a single coupled KL the optimizer can trade "move the mean far" against
"shrink $\sigma$" — and empirically $\sigma$ collapses to buy mean movement. Two separate budgets
forbid that trade.

## 2.3 The three deviations from Acme

### (a) Aggregation over action dimensions

$$
\text{ours:}\quad \alpha_\mu\sum_d \mathrm{KL}_{\mu,d}
\qquad\qquad
\text{Acme:}\quad \sum_d \alpha_{\mu,d}\,\mathrm{KL}_{\mu,d}
$$

We sum the per-dimension KLs into one scalar with one multiplier and one budget; Acme keeps the
$D$-vector, with $D$ multipliers and a per-dimension budget (`per_dim_constraining=True`, its
default, and its docstring recommends keeping it on).

Consequence: with a single multiplier only the *sum* is constrained, so one dimension blowing up is
paid for by the others staying still. With per-dimension multipliers, each $\alpha_{\mu,d}$ grows
when **that** dimension violates.

Also a units trap: our $\varepsilon_\mu=0.01$ bounds the sum over $D$ dims, so it is $D\times$
tighter than a per-dimension budget of the same numeric value. The mjlab A/B divides by
$D=8$ to keep the total budget matched.

### (b) Multiplier range — the one that actually broke

Both use projected dual ascent:

$$
\alpha \leftarrow \Big[\alpha + s\,(\mathrm{KL}-\varepsilon)\Big]_+ .
$$

This is an **integral controller**: while $\mathrm{KL}>\varepsilon$ it keeps accumulating until the
penalty $\alpha\,\mathrm{KL}$ is strong enough to pull $\mathrm{KL}$ back to $\varepsilon$. Its
authority depends on $\alpha$ being free to grow.

| | ours | Acme |
|---|---|---|
| parameterization | plain float | $\alpha=\text{softplus}(\rho)+10^{-8}$ |
| lower bound | $\alpha\ge0$ | $\rho\ge-18$ |
| **upper bound** | $\alpha\le\alpha_{\max}$ (**0.1** mean / **10** var) | **none** |

Clipping at $\alpha_{\max}$ does not merely weaken the constraint — it **opens the feedback loop**.
Once $\alpha=\alpha_{\max}$, the term $s(\mathrm{KL}-\varepsilon)$ has no effect, the integrator is
saturated, and $\mathrm{KL}$ is unconstrained no matter how badly it is violated. This is a
structural break of the control law, not a tuning nuance, and §4 shows it firing within 200
iterations on Ant-Flat.

### (c) The weighted-MLE term

Acme applies the *same* mean/covariance split to the objective, not just to the KL:

$$
\text{ours:}\quad J=\sum_i w_i\log\mathcal N(x_i\mid\mu_\theta,\sigma_\theta)
$$
$$
\text{Acme:}\quad J=\sum_i w_i\log\mathcal N(x_i\mid\mu_\theta,\sigma_{\text{old}})
+\sum_i w_i\log\mathcal N(x_i\mid\mu_{\text{old}},\sigma_\theta)
$$

Since $\partial_\mu\log\mathcal N(x|\mu,\sigma)=(x-\mu)/\sigma^2$, the mean gradients are

$$
\text{ours:}\quad \frac{\partial J}{\partial\mu_d}=\sum_i w_i\frac{x_{i,d}-\mu_d}{\sigma_{\theta,d}^{2}}
\qquad
\text{Acme:}\quad \frac{\partial J}{\partial\mu_d}=\sum_i w_i\frac{x_{i,d}-\mu_d}{\sigma_{\text{old},d}^{2}} .
$$

That $\sigma_{\text{old}}$ is **exactly** the $\sigma_{\text{old}}$ in
$\mathrm{KL}_{\mu,d}=(\Delta\mu_d)^2/2\sigma_{\text{old},d}^2$. So under Acme's form the objective's
mean-gradient and the mean trust region live in the *same metric*, and $\alpha_\mu$ has a stable,
interpretable scale. Under ours they drift apart as $\sigma_\theta$ shrinks: the mean gradient is
amplified by $1/\sigma_\theta^2$ while the trust region is still measured in $\sigma_{\text{old}}$.

**Scale note.** At $\theta=\theta_{\text{old}}$ the two Acme terms coincide, so
$J_{\text{decoupled}}=2\,J_{\text{coupled}}$ (unit-tested). Enabling it doubles the objective
relative to the KL penalty, so a proportionally larger $\alpha$ is needed for the same budget. Acme
absorbs the identical factor of 2 when `action_penalization` sums its two weight sets.

---

# 3. Action bounding, clipping, and the tanh

Three distinct things, routinely conflated.

## 3.1 Enforcing $a\in[\text{low},\text{high}]^D$

**Acme — soft penalty (MO-MPO).** Its policy is $\mathcal N(\mu,\sigma)$ on all of $\mathbb R^D$, so
samples leave the box. It defines a second objective

$$
C(s,a)=-\big\|\,a-\operatorname{clip}(a,-1,1)\,\big\|_2
$$

(zero inside the box, negative outside) and pushes it through the **same E-step machinery** with its
own temperature and budget:

$$
w^{\text{pen}}_i=\operatorname{softmax}_i\!\big(C_i/\eta_{\text{pen}}\big),\qquad
g_{\text{pen}}(\eta_{\text{pen}})=\eta_{\text{pen}}\varepsilon_{\text{pen}}
+\eta_{\text{pen}}\,\mathbb E_s\log\tfrac1N\textstyle\sum_i e^{C_i/\eta_{\text{pen}}},
\quad \varepsilon_{\text{pen}}=10^{-3},
$$

then sets $w_i \leftarrow w_i+w^{\text{pen}}_i$. Out-of-bound actions are *down-weighted*. It is a
**penalty, not a constraint** — nothing forbids them. (And $\sum_i(w_i+w^{\text{pen}}_i)=2$, which is
where Acme's own MLE-scale doubling comes from.)

**Ours — hard, by construction.** The policy is a pushforward: $x\sim\mathcal N(\mu_\theta,\sigma_\theta)$,
$a=b+c\odot\tanh(x)$. Since $\tanh:\mathbb R\to(-1,1)$ is a smooth bijection,
$a\in\operatorname{int}(A)$ with probability 1. No second dual, no $\varepsilon_{\text{pen}}$,
nothing to tune. There is no `clamp` on actions anywhere in `mpo.py`/`cvpo.py` or the runner, and
none is needed.

## 3.2 The change of variables — and why the M-step may omit it

For the diffeomorphism $f(x)=b+c\odot\tanh(x)$,

$$
\log\pi_\theta(a|s)=\log\mathcal N(x\mid\mu_\theta,\sigma_\theta)
-\underbrace{\sum_d\log\big(1-\tanh^2 x_d\big)-\sum_d\log c_d}_{=:\,\mathcal J(x)} .
$$

$\mathcal J$ is a function of $x$ **only — never of $\theta$**. Hence

$$
\sum_i w_i\log\pi_\theta(a_i|s)=\sum_i w_i\log\mathcal N(x_i\mid\mu_\theta,\sigma_\theta)-\underbrace{\sum_i w_i\mathcal J(x_i)}_{\text{constant in }\theta}
\;\Longrightarrow\;
\nabla_\theta \text{ identical.}
$$

And for every KL in the algorithm, the Jacobians cancel inside the log, so for any diffeomorphism $f$

$$
\boxed{\;\mathrm{KL}\big(f_{\#}p\,\|\,f_{\#}q\big)=\mathrm{KL}(p\,\|\,q)\;}
$$

Therefore $\mathrm{KL}_\mu$, $\mathrm{KL}_\Sigma$ computed on the **pre-tanh** Gaussians *are* the
KLs between the actual squashed action distributions, and the same holds for
$\mathrm{KL}(q^*\|\pi_{\text{old}})$ in the E-step. Our $\varepsilon$, $\varepsilon_\mu$,
$\varepsilon_\Sigma$ bound precisely the quantities we intend.

**So omitting the tanh correction in the M-step is exact, not an oversight.** Adding it back would
change nothing mathematically and only add Monte-Carlo noise. The apparent inconsistency —
`StochasticActor.sample` carries the full Jacobian while the M-step does not — is correct:
`sample()` returns an *absolute* log-probability, which SAC's entropy term needs as a value, whereas
MPO only ever consumes $\theta$-gradients and KLs, both invariant under the squash.

## 3.3 What the tanh costs instead: saturation

$$
\frac{\partial a_d}{\partial x_d}=c_d\big(1-\tanh^2 x_d\big)\;\longrightarrow\;0
\quad\text{as } |x_d|\to\infty .
$$

Once saturated, $Q(s,f(x))$ is flat in $x$, so the E-step weights stop discriminating among
large-$|x|$ samples and **nothing in the objective pulls $\mu$ back toward 0**. The only thing
limiting the drift is the mean trust region

$$
\sum_d\frac{(\Delta\mu_d)^2}{2\sigma_{\text{old},d}^2}\le\varepsilon_\mu ,
$$

which bounds each *step* but not the cumulative random walk. Hence: **an unenforced $\mathrm{KL}_\mu$
(a pinned $\alpha_\mu$, §2.3b) permits unbounded drift into saturation.** Acme's action penalization
is what plays this role in its unsquashed parameterization; our substitute is the
`frac_saturated` / `pretanh_mean_absmax` diagnostic.

## 3.4 Clipping that does exist in our code

Neither of these is an action bound:

- $\log\sigma$ clamped to $[-20,2]$ (`actor.py`). A hard clamp with **zero gradient at the bounds**;
  Acme's networks use a softplus scale with a small floor and no ceiling.
- Gradient-norm clipping to `max_grad_norm`.

---

# 4. What the mjlab Ant-Flat A/B measures

Three arms, identical except the flags; 256 envs, `Ant-Flat`, $D=8$. Arm C changes **only** the two
caps of §2.3b, isolating that single variable.

### Iteration ~200 (~51k steps) — the mechanism

| arm | $\mathrm{KL}_\mu$/budget | $\mathrm{KL}_\Sigma$/budget | $\alpha_\mu$ | $\alpha_\Sigma$ | E-step KL/$\varepsilon$ | ESS |
|---|---|---|---|---|---|---|
| A baseline | **3.43** | **5.27** | **0.100 pinned** | **10.0 pinned** | 1.00 | 53/64 |
| B acme-parity | 1.05 | 2.73 | 1.32 free | 19.1 free | 1.00 | 53/64 |
| C caps-only | **0.87** | **1.23** | 0.99 free | 51.1 free | 1.00 | 53/64 |

Both baseline multipliers saturate within 200 iterations and the trust region stops being enforced
— §2.3b, confirmed by measurement. Raising only the caps restores it. The E-step is identical and
healthy in all three, confirming §1.

### Iteration ~10,300 (~2.6M steps) — the consequence

| arm | reward | eplen | $\mathrm{KL}_\mu$/budget | $\alpha_\mu$ | **tanh saturated** |
|---|---|---|---|---|---|
| A baseline | **9.84** (down from 10.31) | 919 | 4.93 | 0.100 pinned | **70.6%** |
| B acme-parity | **12.05** (up from 10.50) | 888 | 0.77 | 2.56 free | 45.3% |
| C caps-only | 10.76 (up from 10.38) | 871 | 0.86 | 2.29 free | 36.6% |

The §3.3 prediction is playing out: the baseline's violation does not settle (3.19 → 4.93), its
pre-tanh mean walks into saturation fastest (6.8% → 70.6%), and its reward has **turned over** while
both corrected arms continue to climb. Runs are ongoing; final numbers go in
[[mpo-vs-acme-reference]].

---

# 5. Summary

| axis | ours vs Acme |
|---|---|
| **E-step KL** | Same dual $\hat g(\eta)$; we solve it exactly, Acme takes one gradient step. **Ours is better.** Verified live via $g'(\eta)=\varepsilon-\mathrm{KL}$. |
| **M-step KL** | Same decomposition and direction. Three deviations: scalar vs per-dimension multiplier (§2.3a); **capped vs floor-only multiplier (§2.3b — the one that breaks the controller)**; coupled vs decoupled weighted-MLE (§2.3c). |
| **Action clipping** | Acme soft-penalizes out-of-bound samples through a second dual; we make them impossible via $\tanh$. **Ours is cleaner**; the residual risk is saturation, not violation. |
| **tanh correction** | Omitting the Jacobian in the M-step is **exact** — it is $\theta$-independent, and KL is invariant under the bijection. Do not "fix" it. |

Cross-refs: [[mpo-vs-acme-reference]] (code diff, config, run ops), [[cvpo-negative-result]]
(why these diagnostics exist), [[m0-baselines]] (FSRL CVPO, whose dual is gradient ascent — §1.5's
weakness in the reference we benchmark against).
