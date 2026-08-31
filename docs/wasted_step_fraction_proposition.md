# Wasted-step fraction under inert padded coordinates

**Status.** Proposition and derivation for prospective use. This document distinguishes the exact expectation of a random ratio from a ratio of expected energies; they are not interchangeable.

**Revision history.** v1.0: as derived. v1.1 (2026-08-31, pre-outcome — before any Probe 0--7 result): metric convention resolved against the manuscript and the conditional removed (Sec. 1; former self-check 1 replaced by the state-pooling risk); self-check 2 routed through Probe 0 of the companion plan; \(\Sigma^{-1/2}\) mapping for (14) stated; typo in (11) fixed; equations (13)--(14) machine-verified by Monte Carlo with discrimination of the \(-(M-2)aa^\top\) term. Companion: `prospective_padding_error_field_analysis.md` (v1.1), committed together.

## 1. Setting and metric convention

Work in **pre-tanh** coordinates $y=(x,z)\in\mathbb R^{r+k}$, with $r=6$,

\[
y=\mu+\Sigma^{1/2}u,\qquad u\sim\mathcal N(0,I),\qquad
\Sigma=\operatorname{diag}(\Sigma_x,\Sigma_z)\succ0.
\]

The critic evaluated below is the composed function
\(F(u)=Q_\phi(s,\tanh(\mu+\Sigma^{1/2}u))\). The environment discards the post-tanh padded coordinates, hence for fixed \((s,x)\), \(Q^\pi(s,x,z)\) is constant in $z$ and \(\nabla_zQ^\pi=0\).

Unless an outer state distribution is written explicitly, all expectations below are conditional on a fixed state \(s\); state aggregation is performed only after the per-state fraction is formed.

For gradients use the dual norm

\[
\|g\|_\Sigma^2=g^\top\Sigma g.
\]

For mean displacements use the corresponding primal norm

\[
\|v\|_{\Sigma^{-1}}^2=v^\top\Sigma^{-1}v.
\]

The trust-region mean step is

\[
\Delta\mu=\sqrt{2\varepsilon}\,
\frac{\Sigma\hat g}{\|\hat g\|_\Sigma}.
\]

Accordingly, define the wasted-step fraction as

\[
L(\hat g)=
\frac{\|(\Delta\mu)_z\|_{\Sigma_z^{-1}}^2}
     {\|\Delta\mu\|_{\Sigma^{-1}}^2}.
\tag{1}
\]

The metric on displacements is fixed by the manuscript, not chosen here. The proof of Proposition 2 (blurring_v6, `prop:estep`) measures the step as \(\mathrm{KL}(q^\ast\|\pi_{\mathrm{old}})=\tfrac12\,\Delta\mu^\top\Sigma^{-1}\Delta\mu\); the appendix proof and the pathwise corollary (`cor:identical`) use the same form. \(\Sigma^{-1}\) on displacements is therefore the trust-region metric the paper already uses. **Resolved against the manuscript 2026-08-31; no alternative convention remains open.**

Let

\[
R=\Sigma^{1/2}\hat g=(R_x,R_z),\quad
X=\|R_x\|_2^2,\quad Z=\|R_z\|_2^2.
\]

Let \(P_x\) and \(P_z\) denote the orthogonal coordinate projectors onto the real and padded blocks in this whitened space.

Then, for every nonzero estimator realization,

\[
\boxed{L=\frac{Z}{X+Z}.}
\tag{2}
\]

The trust-region radius \(\varepsilon\) cancels. Therefore

\[
\boxed{\mathbb E[L]=\mathbb E\!\left[\frac{Z}{X+Z}\right],}
\tag{3}
\]

not \(\mathbb E[Z]/(\mathbb E[X]+\mathbb E[Z])\). Second moments determine the latter proxy but do **not** determine (3) without a distributional or concentration assumption. An exact distribution-free representation is

\[
\mathbb E[L]=\int_0^\infty
\mathbb E\!\left[Z\,e^{-t(X+Z)}\right]dt.
\tag{4}
\]

## 2. Pathwise estimator

Let

\[
H(u)=\Sigma^{1/2}\nabla_yQ_\phi(s,\tanh(\mu+\Sigma^{1/2}u)),
\quad h=\mathbb E[H],\quad C=\operatorname{Cov}(H),
\]

where the gradient includes the tanh Jacobian because the coordinates are pre-tanh. With iid samples,

\[
R_{\rm PW}=\frac1M\sum_{i=1}^M H(u_i),\qquad
\mathbb E[R_{\rm PW}]=h,\qquad
\operatorname{Cov}(R_{\rm PW})=\frac{C}{M}.
\tag{5}
\]

Thus the exact finite-\(M\) quantity is

\[
\boxed{
\mathbb E[L_{\rm PW}]
=\mathbb E\!\left[
\frac{\|M^{-1}\sum_i H_z(u_i)\|^2}
{\|M^{-1}\sum_i H_x(u_i)\|^2+\|M^{-1}\sum_i H_z(u_i)\|^2}
\right].}
\tag{6}
\]

Its block energy moments are

\[
\mathbb E[Z_{\rm PW}]=\|h_z\|^2+\frac{\operatorname{tr}C_{zz}}M,
\qquad
\mathbb E[X_{\rm PW}]=\|h_x\|^2+\frac{\operatorname{tr}C_{xx}}M.
\tag{7}
\]

The cross block \(C_{xz}\) does not enter (7), but it generally enters the expected ratio (6).

For the padded error field, under a specified reference distribution over \((s,x,z)\), define

\[
\tilde e_z=Q_\phi(s,x,z)-\mathbb E_zQ_\phi(s,x,z),
\]

\[
V_e=\mathbb E[\tilde e_z^2],\qquad
G_z^2=\mathbb E\!\left[
\|\Sigma_z^{1/2}\nabla_zQ_\phi\|^2
\right].
\tag{8}
\]

Here \(G_z^2=\|h_z\|^2+\operatorname{tr}C_{zz}\) when the same sampling law is used. The dimensionless centered padded-subspace frequency is

\[
\Omega_z^2=\frac{G_z^2}{V_e},
\tag{9}
\]

which equals \(\sigma_z^2\omega_z^2\) only for isotropic \(\Sigma_z=\sigma_z^2I\). It is not the full \(\omega\), because the constant-in-\(z\) critic bias is unidentified.

If $h\neq0$, bounded convergence gives

\[
\boxed{
\lim_{M\to\infty}\mathbb E[L_{\rm PW}]
=\frac{\|h_z\|^2}{\|h_x\|^2+\|h_z\|^2}.}
\tag{10}
\]

If $h=0$, (10) is undefined: both block energies are $O(M^{-1})$, their common scale cancels, and the limiting angular fraction depends on the full noise law (asymptotically on $C$), not on a deterministic mean.

## 3. Gaussian zeroth-order estimator with a sample-mean baseline

Let

\[
Q_i=F(u_i),\qquad \bar Q=M^{-1}\sum_iQ_i,
\]

and

\[
\hat g_{\rm ZO}=\Sigma^{-1/2}\hat a_M,\qquad
\hat a_M=\frac1M\sum_{i=1}^M(Q_i-\bar Q)u_i.
\tag{11}
\]

Set $q=Q-\mathbb E[Q]$ and define the population moments

\[
a=\mathbb E[qu],\qquad
\nu=\mathbb E[q^2],\qquad
D=\mathbb E[q^2uu^\top].
\tag{12}
\]

No independence between $Q$ and $u_z$ is assumed; their dependence is retained in $a$ and $D$. Assume iid samples and finite fourth moments. A direct sample-covariance calculation gives

\[
\boxed{\mathbb E[\hat a_M]=\left(1-\frac1M\right)a,}
\tag{13}
\]

and

\[
\boxed{
V_M:=\operatorname{Cov}(\hat a_M)
=\frac{M-1}{M^3}
\left[(M-1)D+\nu I-(M-2)aa^\top\right].}
\tag{14}
\]

Equation (14) is the finite-\(M\) covariance including the sample-mean-baseline correction. The \(1-1/M\) mean factor in (13), the joint fourth moment $D$, and the rank-one covariance term must all be retained.

**Mapping to original coordinates.** All \(L\)-statistics in this document are computed in the whitened space, where \(R_{\rm ZO}=\Sigma^{1/2}\hat g_{\rm ZO}=\hat a_M\); equation (14) therefore applies to them directly, with no mapping. If the covariance of the estimator itself is reported in original coordinates — as in the planned extension of the manuscript's Appendix A.2 — it is \(\operatorname{Cov}(\hat g_{\rm ZO})=\Sigma^{-1/2}V_M\Sigma^{-1/2}\). Do not apply \(V_M\) to the unwhitened estimator.

**Numerical verification.** Equations (13)--(14) were verified by direct Monte Carlo (`scripts/verify_wasted_step_moments.py`, committed alongside): a test function with a nonlinear \(q\)--\(u\) cross term, \(M\in\{2,8,32\}\), \(5\times10^5\) batches per \(M\). Agreement with (14) is at the Monte-Carlo noise floor (max deviation \(7\times10^{-4}\) at \(M=8\) against covariance entries of order \(0.2\)), and the check discriminates: the variant that drops the \(-(M-2)aa^\top\) term is rejected by a factor of \(\sim\!60\) at \(M=8\). At \(M=2\) that term vanishes identically, as (14) predicts.

To see the correction explicitly, let \(H_M=I-M^{-1}\mathbf1\mathbf1^\top\) and \(T=M\hat a_M=q^\top H_MU\). Expanding \(\mathbb E[TT^\top]\) over the nonzero index partitions (all indices equal, or two equal pairs) gives

\[
\mathbb E[TT^\top]
=\frac{(M-1)^2}{M}D
+\frac{M-1}{M}\nu I
+\frac{M-1}{M}\big[(M-1)^2+1\big]aa^\top.
\]

Since \(\mathbb E[T]=(M-1)a\), subtracting \(\mathbb E[T]\mathbb E[T]^\top\) and dividing by \(M^2\) yields (14).

Because \(R_{\rm ZO}=\Sigma^{1/2}\hat g_{\rm ZO}=\hat a_M\),

\[
\boxed{
\mathbb E[L_{\rm ZO}]
=\mathbb E\!\left[
\frac{\|\hat a_{M,z}\|^2}
{\|\hat a_{M,x}\|^2+\|\hat a_{M,z}\|^2}
\right].}
\tag{15}
\]

The exact block energy moments are

\[
\mathbb E[Z_{\rm ZO}]
=\left(1-\frac1M\right)^2\|a_z\|^2
+\operatorname{tr}(V_{M,zz}),
\tag{16}
\]

\[
\mathbb E[X_{\rm ZO}]
=\left(1-\frac1M\right)^2\|a_x\|^2
+\operatorname{tr}(V_{M,xx}).
\tag{17}
\]

Again, \(V_{M,xz}\) affects the random ratio even though it drops out of the two marginal energy expectations.

For a **restricted-\(z\) oracle** with \((s,x)\) fixed, replace $q$ by \(\tilde e_z\) and $u$ by $u_z$. Then

\[
\nu_z=\mathbb E_z[\tilde e_z^2],\quad
a_z=\mathbb E_z[\tilde e_z u_z],\quad
D_z=\mathbb E_z[\tilde e_z^2u_zu_z^\top]
\tag{18}
\]

are rollout-free but still estimated over sampled $z$. For the full estimator, $q$ also contains real-action signal and constant-in-\(z\) critic bias; those add measurable padded-component variance even though their expected correlation with $u_z$ is zero under block-independent sampling.

If $F$ is weakly differentiable and satisfies the Gaussian integration-by-parts conditions, Stein's identity gives

\[
a=\mathbb E[Fu]=\Sigma^{1/2}\mathbb E[\nabla_yQ_\phi]=h.
\tag{19}
\]

Consequently, if $a\neq0$,

\[
\boxed{
\lim_{M\to\infty}\mathbb E[L_{\rm ZO}]
=\frac{\|a_z\|^2}{\|a_x\|^2+\|a_z\|^2},}
\tag{20}
\]

which equals the pathwise limit in (10) when both use the same smoothed critic and sampling law. The distinction is finite-sample variance, nonlinear weighting, or a mismatch of sampling/coordinate conventions—not a different infinite-sample Gaussian-smoothed gradient.

For either estimator, if $R_M=h+M^{-1/2}\xi+o_p(M^{-1/2})$, \(\operatorname{Cov}(\xi)=C_*\), $s=\|h\|^2>0$, and $n=\|h_z\|^2$, the large-\(M\) expansion is

\[
\mathbb E[L]
=\frac ns+\frac1M\left[
\frac{\operatorname{tr}(P_zC_*)}{s}
-\frac{n\operatorname{tr}C_*+4h^\top P_zC_*h}{s^2}
+\frac{4n\,h^\top C_*h}{s^3}
\right]+o(M^{-1}).
\tag{21}
\]

For pathwise, $C_*=C$. For ZO, $C_*=D-aa^\top$ at leading order; the \(\nu I\) baseline-estimation term in (14) is $O(M^{-2})$. Equation (21) is an approximation, not a replacement for direct Monte Carlo evaluation of (6) or (15).

## 4. Implemented weighted-MLE update

For direct weighted fitting of sampled pre-tanh actions, write

\[
w_i=\frac{\exp(Q_i/\eta)}{\sum_j\exp(Q_j/\eta)},
\qquad
S_W=\Sigma^{-1/2}\Delta\mu_W.
\]

If the new mean is the exact weighted sample mean, then

\[
S_W=\sum_iw_iu_i,\qquad
L_W=\frac{\|(S_W)_z\|^2}{\|S_W\|^2}.
\tag{22}
\]

A subsequent scalar trust-region rescaling leaves $L_W$ unchanged. In the high-temperature regime,

\[
w_i=\frac1M+
\frac{Q_i-\bar Q}{M\eta}+O(\eta^{-2}),
\]

so

\[
S_W=\bar u+\eta^{-1}\hat a_M+O(\eta^{-2}).
\tag{23}
\]

The \(\bar u\) term disappears only if the implementation uses centered weights, antithetic samples, or an equivalent control variate; this must be established from code.

There is no analogue of (14) for the implemented operator without additional assumptions because:

1. softmax weights couple all $M$ samples and depend nonlinearly on $Q$;
2. per-state \(\eta\) is itself a random solution of the sampled dual;
3. finite \(\varepsilon_E\), clipping, covariance fitting, and the actor network's shared parameters can make the realized mean update differ from (22).

As $M\to\infty$, under regularity and convergence of the dual,

\[
S_W\longrightarrow
\frac{\mathbb E[e^{Q/\eta_*}u]}{\mathbb E[e^{Q/\eta_*}]},
\tag{24}
\]

not the canonical Gaussian-smoothed gradient unless the logits are in the linear regime. Therefore the implemented $\mathbb E[L_W]$, its deviation from the centered-$z$ reference, weight KL, ESS loss, and update cosine/MSE must be measured by repeated batches. Apart from $0\le L_W\le1$, no informative universal bound follows from the second moments alone; a useful bound additionally needs a lower-tail bound on total step energy or bounded logits/weights.

## 5. Assumptions

1. Samples are iid conditional on the state; $u_x$ and $u_z$ are independent standard-normal blocks.
2. \(\Sigma\) is fixed during each estimator evaluation and is positive diagonal.
3. All gradients and perturbations use pre-tanh coordinates; $Q_\phi$ is composed with tanh.
4. Required second/fourth moments exist; Stein's identity is used only when its differentiability and integrability conditions hold.
5. The true simulator and reward discard $z$, giving \(\nabla_zQ^\pi=0\).
6. Equations (11)--(20) describe the exact stated Gaussian estimator, not automatically the repository's weighted-MLE implementation.
7. The estimator is nonzero almost surely, or $L$ is declared undefined on zero-step realizations and their frequency is reported.

## 6. Adversarial self-check: three most likely failure points

1. **State pooling before the ratio.** Equation (3) is a per-state expectation of a random ratio. Vectorized analysis code can silently sum block energies over the whole state × batch tensor and form one global ratio — which is the energy-weighted proxy \(\mathbb E[Z]/(\mathbb E[X]+\mathbb E[Z])\) that (3) forbids, the Section 1 Jensen error committed across states instead of across batches. Guard: every script forms the per-state ratio first and aggregates after, and additionally reports the global-ratio proxy as a discrepancy check. (The metric-mismatch risk previously listed here is resolved; see Section 1.)
2. **Estimator mismatch hidden by notation.** The code may use post-tanh actions, antithetic or normalized directions, clipping, a different baseline, shared random keys, per-state or per-batch temperatures, or a sphere estimator carrying a different dimension factor. Any of these invalidates the direct application of (11)--(14), even though the sample-covariance algebra itself is correct. This risk is now gated by **Probe 0 (code-convention audit)** in `prospective_padding_error_field_analysis.md`: its five answers must be appended there (Amendment A, with audited commit hash and file:line) before Probe 1 runs, and until they are on record no formula in Sections 3--4 may be applied to repository outputs.
3. **Replacing an expected ratio by energy moments—or mapping it to weighted MLE.** Equations (7), (16), and (17) do not determine \(\mathbb E[L]\). Likewise, the softmax/dual/network update need not equal the canonical estimator. Direct repeated-batch measurement of the actual ratio is required, with the analytic moments used as checks rather than substitutes.
