# Prospective analysis plan: padded-subspace critic error

**Status:** commit before new gradient, repeated-estimator, crossed-operator, shuffle, or common-state results. This follows exploratory C5 amplitudes and is not a blind preregistration.

**Revision history.** v1.0: as drafted. v1.1 (2026-08-31, pre-outcome — before any Probe 0--7 result): Probe 0 (code-convention audit) inserted as an execution gate; chat-shorthand references replaced by the companion filename; provenance extended to cover Amendment A. Companion derivation: `wasted_step_fraction_proposition.md` (v1.1), committed together with its verification script.

## 1. Locked definitions

### Coordinates and distributions

- Use **pre-tanh** coordinates $y=(x,z)$, with \(x\in\mathbb R^6\), \(z\in\mathbb R^k\), and \(\tilde Q_\phi(s,y)=Q_\phi(s,\tanh y)\). Gradients include the tanh Jacobian.
- Restricted-$z$ probes hold \((s,x)\) fixed and vary only $z$. Full-estimator probes vary all $d=6+k$ coordinates. Results will label $k$ and $d$ explicitly; they will not substitute one for the other.
- Two padded reference laws are mandatory:
  1. **Checkpoint law:** $z\sim\mathcal N(\mu_{z,c},\Sigma_{z,c})$ from checkpoint $c$.
  2. **Common standardized law:** $z\sim\mathcal N(0,I_k)$ in pre-tanh coordinates for every checkpoint.
  In the metric, \(\Sigma_z\) is the covariance of the active reference law. Checkpoint-law results are primary for operator relevance; the common law is the mandatory cross-arm sensitivity analysis.

### Error field and norms

For each \((s,x)\) and chosen \(\rho_z\),

\[
\bar Q_\phi(s,x)=\mathbb E_{z\sim\rho_z}Q_\phi(s,x,z),\qquad
\tilde e_z=Q_\phi-\bar Q_\phi.
\]

The true padded gradient is zero. These identify centered $z$-varying error and its derivatives, not constant-in-$z$ bias or full \(e=Q_\phi-Q^\pi\).

Use the companion proposition's metric (`wasted_step_fraction_proposition.md`, Sec. 1): gradient energy \(\|g\|_\Sigma^2=g^\top\Sigma g\), step energy \(\|v\|_{\Sigma^{-1}}^2=v^\top\Sigma^{-1}v\), and

\[
L=\frac{\|(\Delta\mu)_z\|_{\Sigma_z^{-1}}^2}
        {\|\Delta\mu\|_{\Sigma^{-1}}^2}.
\]

Report repeated-batch \(\mathbb E[L]\); label the ratio of expected block energies as secondary. Define

\[
V_e=\mathbb E[\tilde e_z^2],\quad
G_z^2=\mathbb E\|\Sigma_z^{1/2}\nabla_zQ_\phi\|^2,\quad
\Omega_z^2=G_z^2/V_e.
\]

\(\Omega_z\) is centered padded-subspace frequency, not full \(\omega\). Keep the existing sup-norm claim separate unless an RMS proposition is proved.

For the pathwise interaction channel report separately

\[
N_{PW}^2=\mathbb E_{s,x,z}\|\nabla_xQ_\phi-\mathbb E_z\nabla_xQ_\phi\|_{\Sigma_x}^2,
\quad
S_{PW}^2=\mathbb E_{s,x}\|\mathbb E_z\nabla_xQ_\phi\|_{\Sigma_x}^2.
\]

Report \((N_{PW},S_{PW})\) primarily; report \(C_{PW}=N_{PW}/S_{PW}\) only for \(S_{PW}>0\), without \(\epsilon\).

### States, seeds, and uncertainty

- Use matched k=16 seeds 0--4; report every seed and arm.
- Within seed, use equal complete lanes from A and B, retaining temporal chunks. Apply each critic's normalizer to identical raw states. Common actions contain equal A/B-policy samples and retain source labels.
- Arm summary: seed median; mean secondary. Paired contrasts resample matched IDs.
- Use 10,000 hierarchical bootstrap replicates with `np.random.default_rng(20260831)`: seeds, whole lanes within seed, then estimator batches/$z$-samples within lane. Report percentile 95% intervals, seed values, zero-step frequency, and paired exact sign tests. Do not portray five seeds as precise population evidence.

## 2. Ordered probes and committed interpretation

### Probe 0 — code-convention audit (gates all formulas)

Five facts about the repository, each recorded as: audited commit hash, file:line, one-sentence answer. Answers are appended to this document as **Amendment A** before Probe 1 runs. They are factual conventions, not analytic choices: they change formulas where they must, never predictions.

1. **Coordinates.** Does the actor loss / E-step evaluate \(Q_\phi\) on pre-tanh \(y\) or post-tanh \(\tanh y\)? Are Q-inputs clipped before evaluation (this ties to the known clipped-log-probability defect)? Gates the tanh-Jacobian composition and every gradient formula.
2. **Weight centering.** Does the weighted-MLE mean update use raw softmax weights — retaining the \(\bar u\) noise term of the proposition's eq. (23) — or a centered / baselined / antithetic form? Gates the implemented-operator noise model; if uncentered, the isotropic \(\bar u\) term is a \(d\)-dimensional noise source in the operator itself and must be measured in Probes 2 and 4.
3. **Sampling.** Fresh iid Gaussian draws per state, or antithetic pairs / PRNG keys shared across states or across the batch? Gates the iid assumption behind eqs (5) and (14).
4. **Temperature.** Is \(\eta\) solved per state or per batch, from which dual, on which samples, and is it stochastic at probe time? Gates offline replication of the weights in Probes 4 and 7.
5. **Direction law.** Gaussian \(u\sim\mathcal N(0,I)\) or unit-sphere directions (different dimension factor)? Gates eq (11) and all moment formulas.

Also record: \(M\), \(\varepsilon_E\), whether \(\Sigma\) is diagonal in code, and the exact function evaluated (raw critic, categorical-head mean, target network).

| Order | Probe | Primary outcome | Prediction / falsification rule |
|---:|---|---|---|
| 0 | Code-convention audit | Amendment A on record: five convention facts + recordables, each with commit hash and file:line | **Gate, no prediction.** Probe 1 may not start until Amendment A is committed. Any formula contradicted by an answer is amended in the proposition before use, with the change logged in its revision history. |
| 1 | Restricted-$z$ oracle | PW/ZO padded error energy per component and full \(k\)-block; \(V_e,G_z,\Omega_z\) | **No winner predicted.** Do not use \(L\): a \(z\)-only step has \(L=1\). The proposition's eqs (13)--(14) moments must match repeated batches within MC uncertainty; failure rejects the code-to-derivation match. |
| 2 | Full-estimator oracle | Repeated-batch \(\mathbb E[L]\) and block errors over \(d=6+k\) | **Descriptive.** Difference from Probe 1 includes real signal and constant-in-\(z\) bias leakage, not \(\tilde e_z\) alone. |
| 3 | \(\Omega_z,C_{PW}\) | Frequency and PW interaction noise/signal pair | **Descriptive.** No crossover threshold: the companion proposition does not derive an RMS Claim 4. A sign alone cannot confirm it. |
| 4 | Crossed same-critic table vs \(\bar Q_\phi\) | Primary \(D\): squared \(\Sigma_x^{-1}\)-norm of real-update change; secondary \(L\), cosine, weight KL, ESS | Prediction: WML is affected more. Falsified if paired median \(D_{\rm WML}-D_{\rm PW}\le0\). Evaluate each critic under both operators. |
| 5 | Weight shuffle | Per-coordinate covariance; residual beyond \(1-1/\mathrm{ESS}\) | Prediction: shuffling preserves generic padded shrinkage and removes extra real contraction. Falsified by persistent real residual outside its permutation interval or systematic padded violation. |
| 6 | Common-state critic probe | Padded spread / real spread and / \(|Q_{p50}|\) | C5-motivated prediction: B>A. Falsified if paired median B-minus-A \(\le0\). Report both normalizations. |
| 7 | Synthetic ESS calibration | Exact-vs-linearized canonical/WML error over signal/error logits, \(M,\eta,\mathrm{ESS}\) | Prediction: error rises with error-logit scale and falling ESS. No cutoff committed. No monotone relation falsifies ESS as a one-dimensional validity proxy. |

Record fresh RNG streams. Report raw state/seed results before aggregates; omit no probe based on direction.

## 3. Provenance

Committed **after** the contaminated k=16 return, C5 Q-spread amplitudes, \(\sigma_{pad}/\sigma_{real}\), and the weight-leakage hypothesis; committed **before** padded gradients, repeated-estimator distributions, crossed-operator, shuffle, common-state, and synthetic-ESS outcomes. Predictions are prospective only for those unseen outcomes. Probe 0's answers (Amendment A) will be appended post-audit and pre-Probe-1; they are factual code conventions, not analytic choices, and cannot alter any prediction in Section 2.

## 4. Explicitly out of scope

- No return-level confirmation/refutation and no rehabilitation of the contaminated padding experiment.
- No claim that inert padding isolates raw action-dimension scaling.
- No estimate of full $e$, full manuscript \(\omega\), or constant-in-$z$ critic bias without Monte Carlo \(Q^\pi\).
- No replacement of the manuscript's sup-norm Claim 4 by RMS norms without a new proof.
- No causal A-vs-B critic-quality claim from own-policy state distributions.
- No masked training rerun decision until Probes 0--7 are all reported.

---

## Amendment A.0 — SUPERSEDED (audited wrong branch): Probe 0 on the torch re-implementation (2026-08-31)

> **Superseded by Amendment A below.** This audit was performed on `safe_rl/algorithms/reppo.py` — the local torch re-implementation (pathwise-only, M=1, no E-step) — which is **not** the code that produced the padded runs. Its statements are correct for that codebase and are retained unedited for provenance only; no probe may cite them.

Audited commit: `5371715` (branch `fhdcmpo-multienv-sweep`); every cited file verified unmodified in the working tree at audit time (`git diff --stat HEAD` empty for all three). Audited operator: the REPPO actor update, `safe_rl/algorithms/reppo.py`.

1. **Coordinates.** `5371715`, `safe_rl/algorithms/reppo.py:853-861` + `safe_rl/modules/stochastic_actor_critic_base.py:218` + `safe_rl/modules/reppo_actor_critic.py:153-156` — the actor loss evaluates \(Q_\phi\) on the **post-tanh** action \(a=\text{action\_scale}\cdot\tanh(y)\) (reparameterized `td.rsample()`), passed to `evaluate_q` **unclipped**: the \((1-10^{-6})\) clip (`stochastic_actor_critic_base.py:162-169`) is applied only inside `log_prob` (lines 206, 219 — the known clipped-log-probability convention), never to Q inputs.

2. **Weight centering.** `5371715`, `safe_rl/algorithms/reppo.py:846-863` — there is **no weighted-MLE / E-step update in REPPO**: the actor step is a pathwise reparameterized gradient (`primary = alpha_temp·log_prob − q_pi`, gradient through \(\partial Q/\partial a\)), so no softmax weights exist in the implemented operator, the \(\bar u\) term of the proposition's eq. (23) is **not live**, and the Probe-2/4 \(\bar u\) measurement is dropped as inapplicable — the pathwise channel \((N_{PW},S_{PW})\) is the operator-relevant one, and eqs (11)–(14) describe probe-constructed estimators only.

3. **Sampling.** `5371715`, `safe_rl/modules/stochastic_actor_critic_base.py:218` + `safe_rl/algorithms/reppo.py:873` — fresh iid draws from the global torch PRNG: one reparameterized action per state per minibatch pass for the loss, plus 16 fresh iid samples per state for the MC KL; no antithetic pairs, no keys shared across states or across the batch.

4. **Temperature.** `5371715`, `safe_rl/algorithms/reppo.py:97,260-261,929-936` — there is no E-step \(\eta\); the only temperature is the entropy dual \(\alpha_{temp}=\exp(\log\alpha_{temp})\), a **single scalar shared across the whole batch** (per-batch, never per-state), updated by one gradient step on \(\alpha_{temp}(H-H_{target})\) per minibatch and deterministic at probe time (a stored checkpoint parameter).

5. **Direction law.** `5371715`, `safe_rl/modules/stochastic_actor_critic_base.py:138-150,218` — perturbations are full **Gaussian** draws \(y=\mu+\sigma\odot u,\ u\sim\mathcal N(0,I)\) with diagonal (state-dependent) \(\sigma\), pushed through tanh; no unit-sphere or fixed-norm direction sampling exists in the code path.

**Recordables.** \(M=1\) action sample per state in the actor loss (the 16 belong to the KL estimator only); no \(\varepsilon_E\) exists — the trust-region bound is `desired_kl` (0.1 in the v24-lineage paper-bench configs, hard-gate "clipped" mode `reppo.py:899-901`); \(\Sigma\) is diagonal in code (`Normal(mean, std)`, `stochastic_actor_critic_base.py:138-150`); the function evaluated is the **live** critic's categorical-head mean — `evaluate_q` → `critic.get_value(critic.get_dist(logits))` (`safe_rl/modules/reppo_actor_critic.py:159-161`) — with no target network (REPPO has none).

**Formula consequences.** No amendment to the proposition is required: its self-check 6 already scopes eqs (11)–(20) to the stated Gaussian estimator, and answer 2 resolves that scoping — the ZO formulas apply only to estimators the probes construct themselves (whitened Gaussian \(u\), sample-mean baseline, per-state fresh iid draws, exactly eq. (13)'s \(\hat a_M\)); the pathwise formulas (proposition Sec. 2) apply to the training operator directly, composed with the tanh Jacobian at the post-tanh evaluation point per answer 1. Eqs (13)–(14) themselves are machine-verified in `scripts/analysis/verify_wasted_step_moments.py` (max cov residual 2.6e-3 at M=2 shrinking to 1.6e-4 at M=32; the variant dropping \(-(M-2)aa^\top\) rejected by 60–90× at M=8/32).


---

## Amendment A — Probe 0 code-convention audit (2026-08-31, corrected: training fork `reppo_original`)

Audited commit: `3b96deb` (branch `main`) of `~/workspaces/reppo_original` — the JAX fork that produced the padded runs. Provenance: `pad16_B_s*.log` carry `actor_update_mode: weighted_mle`, `estep_num_samples: 32`, `eps_e: 0.5`, `action_pad: 16`, `mstep_decoupled: false`, `actor_kl_clip_mode: clipped` (arm B = weighted-MLE + single KL clip), were started 2026-08-30 21:52 — after `3b96deb` (2026-08-30 15:52) — and export into the `*_weighted_mle_pad16_*` namespace that `3b96deb` itself introduced. Both audited files (`src/jaxrl/reppo.py`, `src/networks/jax_models.py`) verified unmodified in the fork's working tree at audit time (its only diffs touch `src/env_utils/jax_wrappers.py`, `src/torchrl/envs.py`).

1. **Coordinates.** `3b96deb` — pathwise arm `src/jaxrl/reppo.py:641-644`: \(Q_\phi\) sees the **post-tanh** action from the distrax tanh-Normal (`src/networks/jax_models.py:525`), **unclipped**; weighted-MLE arm `src/jaxrl/reppo.py:669,705`: the \(\pi_{old}\) samples are clipped to \(\pm(1-10^{-4})\) at line 669 **before** the critic call, so in arm B \(Q_\phi\) is evaluated on **clipped post-tanh** actions and the same clipped array feeds every log-prob — in this arm the clip is part of \(F(u)\), not only of the log-probability.

2. **Weight centering.** `3b96deb` — `src/jaxrl/reppo.py:134-155,708-711`: **raw self-normalized softmax** weights \(w_i=\mathrm{softmax}(q_i/\eta)\) in \(\text{objective}=-\sum_i w_i\log\pi_\theta(a_i)\), with no centering, no baseline, no antithetic pairing — **the \(\bar u\) noise term of proposition eq. (23) is live** and must be measured in Probes 2 and 4.

3. **Sampling.** `3b96deb` — `src/jaxrl/reppo.py:666-668` with key flow `src/jaxrl/reppo.py:915,925-927,933-938`: the \((M{=}32,B,d)\) proposal draws are iid Gaussian within one call, **but** the seed `key` is a closure constant over the minibatch `lax.scan` — every minibatch of an epoch reuses the **same** key (fresh key per epoch only), that key also seeds the arm's SAC-style sample (line 641), and the E-step actions are reused (never re-sampled) for the KL estimator and the critic alike — so the independence behind eqs (5)/(14) holds across epochs and offline redraws, **not** across minibatches within an epoch.

4. **Temperature.** `3b96deb` — `src/networks/jax_models.py:377,409-413` (mirrored at `507,539-543`) + `src/jaxrl/reppo.py:156-172,700,716`: \(\eta\) is a **single learned scalar shared across the whole batch** (per-batch, never per-state), softplus-parameterized and clipped to \([\eta_{min},\eta_{max}]\), updated by **one gradient step per minibatch** on the standard MPO dual \(g(\eta)=\eta\,\varepsilon_E+\eta\,\mathrm{mean}_j\,\mathrm{LSE}_i(q_{ji}/\eta)\) over the same \(M{=}32\) samples (max pulled out; \(q\) detached) — a gradient-tracked dual, **not** solved to optimality per iteration — detached inside the weights and **deterministic at probe time** (a stored checkpoint parameter).

5. **Direction law.** `3b96deb` — `src/networks/jax_models.py:523-525` + `src/jaxrl/reppo.py:666-668`: **Gaussian** — full-action proposals from \(\pi_{old}\)'s diagonal tanh-Normal, equivalently \(y=\mu_{old}+\sigma_{old}\odot u,\ u\sim\mathcal N(0,I)\) (the decoupled path draws \(u\) explicitly, `src/jaxrl/reppo.py:729-731`, unused in arm B); no unit-sphere or fixed-norm direction sampling exists.

**Recordables.** \(M=32\) (`estep_num_samples`); \(\varepsilon_E=0.5\) (`eps_e`); \(\Sigma\) diagonal and state-dependent (`loc, log_std` head split, \(\sigma=e^{\log\sigma}+\text{min\_std}\) with `actor_min_std: 0.0` in the padded runs, `src/networks/jax_models.py:523-524`); the Q head is the **live** critic's categorical (HL-Gauss) mean — \(\mathrm{softmax}(\text{logits})\cdot\mathrm{linspace}(v_{min}{=}0,v_{max}{=}150)\) (`src/networks/jax_models.py:305-310`); the local name `critic_target_model` (`src/jaxrl/reppo.py:630-633`) is merged from the live `train_state.critic.params` — there is no target critic — while the actor target is a true \(\pi_{old}\) snapshot refreshed once per learn step (`src/jaxrl/reppo.py:560-567`).

**Formula consequences.** Answer 2 activates the ZO/weighted-MLE track with the \(\bar u\) term live (measure it in Probes 2/4); answer 3 restricts eq. (14)'s repeated-batch use to across-epoch or offline re-drawn batches; answer 1 puts the \(\pm(1-10^{-4})\) clip inside \(F(u)\) for arm B (relevant where \(\sigma\) saturates \(|\tanh y|\to1\)); answer 4 means offline weight replication must read the checkpoint \(\eta\) verbatim rather than re-solving any dual; answer 5 and the diagonal \(\Sigma\) leave the whitened-coordinate mapping of the proposition's Sec. 1 unchanged.
