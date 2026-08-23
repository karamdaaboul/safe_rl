# Distributional (classification) cost critics in P3O — findings & integration guide

*2026-07-03, local A/B/C study (Claude Code overnight run). Question: how to integrate a
distributional safety critic (categorical two-hot vs HL-Gauss) into safe RL correctly, for good
performance — and why the categorical variant produced MORE cost than plain MSE.*

## TL;DR

**Use HL-Gauss for the cost critic. Do not use categorical/two-hot.** On SafetyCarGoal2-v0
(limit 25, 400 iters, 8 envs, seeds 1–2, configs identical except the cost-critic block):

| variant (cost critic) | seed | cost (last-50 mean) | reward (last-50) | final cost | iters ≤ 25 in last 100 |
|---|---|---|---|---|---|
| MSE (`StandardCritic`) | 1 | 33.9 | 0.91 | 43.4 | 65/100 |
| MSE | 2 | 36.9 | −1.04 | 30.0 | 5/100 |
| HL-Gauss | 1 | **8.0** | 0.28 | **13.4** | 65/100 |
| HL-Gauss | 2 | 29.4 | 0.72 | **7.8** | 45/100 |
| Categorical (two-hot) | 1 | 78.1 | −6.10 | 81.8 | **0/100** |
| Categorical (two-hot) | 2 | 100.4 | 1.16 | 95.2 | **0/100** |

Ordering **two-hot < MSE < HL-Gauss** exactly matches Farebrother et al. 2024, *"Stop Regressing:
Training Value Functions via Classification"* (arXiv:2403.03950), which reports the same ordering
for reward critics and attributes HL-Gauss's win to its label-smoothing-like use of the support's
ordinal structure. HL-Gauss was the only variant that *ended safe on both seeds*.

## Why two-hot produces more cost than MSE (mechanism)

The implementation is **not** the problem. `CategoricalCostCritic` (critic.py) passes unit tests
(exact two-hot mass split, decode round-trip) and three GPU probes in supervised settings —
clean-fit quality, generalization under GAE-like target noise, and shift absorption / decode bias
were all at parity with HL-Gauss and better than MSE (scratchpad probes: `fit_test_gpu.py`,
`noise_probe.py`, `bias_probe.py`). Two hypotheses (faster memorization; softmax-leak upward
decode bias) were tested and refuted.

The failure is **in the control loop**, where the cost critic feeds three signals at once:

1. **A_C = R_C − V_C** enters the P3O penalty surrogate — its *sign* per-sample decides which
   actions get pushed away (P3O deliberately does not mean-center cost advantages).
2. **Bootstrapped cost returns** (GAE with V_C) set the scale of Jc and the returns the critic is
   trained on next iteration (self-consistency loop).
3. The **gate/κ controller** reads mean episode cost vs limit; κ multiplies whatever gradient the
   (possibly wrong) A_C provides.

Two-hot's point-mass targets give cross-entropy no ordinal information: a prediction one bin off
is "fully wrong", and gradient reaches non-target bins only through the softmax normalizer. Under
non-stationary, noisy on-policy cost returns this yields a V_C whose *ranking* of states is
noisier than MSE's, even when its average decode error looks fine in supervised probes. With a
noisy-sign A_C, the κ-controller spirals: the gate stays open, κ pegs at `kappa_max` (100), and
the policy receives a large penalty gradient pointing in a partly wrong direction — so cost does
not come down (seed 2: never ≤ 25 in the last 100 iters, dips to 7.8 at it 198 then always
escapes to 75–118) or reward collapses without safety (seed 1: reward −10 at cost 80).
HL-Gauss spreads target mass over ~6 bins (σ/bin_width = 0.75), restoring metric structure in the
loss, which is exactly the ingredient two-hot lacks. "Fixing" two-hot by adding neighborhood label
smoothing *is* HL-Gauss with small σ — there is no reason to keep a separate categorical path.

## Integration checklist (what made HL-Gauss work here)

- **Critic class**: `HLGaussCostCritic` (`safe_rl/modules/critic.py`), selected via
  `policy.cost_critic_kwargs.loss_type: hlgauss`; P3O branches on
  `policy.is_distributional_cost_critic` (`actor_critic.py`, `p3o.py`).
- **Decode with `expected_value`** (mean of softmax over bin centers) and feed that scalar into
  the standard GAE pipeline — the rest of P3O stays untouched. Keep P3O's std-only advantage
  scaling (no mean-centering; the sign is the signal).
- **Support sizing**: `v_min: 0`, `v_max` ≈ 2–3× the largest plausible discounted cost return
  (here 60 with γ_cost 0.95, per-step cost ≤ ~1). Watch the logged `cost_return_clip_frac` —
  it should stay ≈ 0; if targets clip, widen `v_max`.
- **σ ratio**: `sigma = 0.75 × bin_width` (101 bins on [0, 60] → σ = 0.45). This is the paper's
  recommended ~0.75 ratio; going to σ→0 degrades toward two-hot.
- **Loss**: cross-entropy in fp32; `use_clipped_cost_loss` is force-disabled for classification
  critics (p3o.py:184) — value clipping is meaningless on logits.
- **Init**: last-layer weights ×0.01 and bias shifted so the initial decode sits near `v_min`
  (already in `HLGaussCostCritic`) — a random-init critic otherwise predicts mid-support (~30),
  which instantly opens the penalty gate at iteration 0.
- **Capacity/LR**: [512, 512, 512] + LayerNorm + elu, separate `cost_critic_lr: 1e-3`
  (policy lr 3e-4). Classification heads tolerate the higher LR well.
- **Controller caveat (all variants)**: on level-2 envs with `kappa_max: 100` the κ controller
  oscillates between "safe but low-reward" and "rewarding but unsafe" phases for *every* critic
  type; HL-Gauss recovers to safe, the others don't reliably. If steadier behavior is needed,
  lower `kappa_max` / soften the κ update (ρ), or try the HL-Gauss-only CVaR gate
  (`use_cvar_in_gate: true` + `cvar_alpha`; note `cvar_alpha` alone is inert — p3o.py:393).

## Status / caveats

- Evidence: 2 seeds × 3 variants × 400 iters, one env (SafetyCarGoal2-v0), local GPU. Consistent
  with the literature but confirm with a JUWELS multi-seed sweep before hard conclusions
  (template: `sweeps/p3o_hlgauss_sweep.yaml`).
- The categorical path (`CategoricalCostCritic`, `config/safety_gymnasium_p3o_categorical.yaml`,
  unit tests in `tests/test_core_components.py`) is kept as a correct, tested ablation baseline —
  recommended for comparisons only, not for actual training.
- Raw run logs + probe scripts lived in the session scratchpad (`runlogs/p3o_{mse_baseline,
  hlgauss,categorical}{,_s2}.log`, `fit_test_gpu.py`, `noise_probe.py`, `bias_probe.py`);
  scratchpads are ephemeral, numbers above are the durable record.

---

# Overnight campaign — "safe *and* rewarding" agent on SafetyCarGoal2-v0 (2026-07-04)

*Goal (user): stop settling for the safe-but-zero-reward corner. Run every promising variant and
deliver a checkpoint that is safe (episodic cost ≤ 25) **and** collects real reward. Bar to beat:
the plain-HL-Gauss run's `model_100` at reward 0.375 / cost 0.0 (10-ep eval).*

## ⚠️ Methodology first: small-sample eval on SafetyCarGoal2 is untrustworthy

The single most important finding of this campaign is a **measurement** one. On SafetyCarGoal2-v0
the per-episode cost has enormous variance (different hidden-goal layouts expose the agent to very
different hazard counts). A 5–20 episode eval of the *same checkpoint* swings wildly between runs —
e.g. CVaR `model_150` measured **cost 4.3 → 5.25 → 74.1 → 19.14** across four eval batches (10, 20,
10, 50 episodes). Any "winner" picked from a 10-episode eval is noise. **Only report numbers from
≥50 episodes at a fixed seed**, and treat even those as a wide band, not a point estimate. An earlier
draft of this section declared a 1.276-reward "winner" from a 10-ep eval; a 50-ep eval of that same
checkpoint (`model_100`) is cost **49.95 — unsafe**. That draft was wrong; the table below is the
corrected, 50-episode record.

## Verdict: there is NO safe-and-competent agent at this budget — "safe" here means "idle"

The honest conclusion, once you count **goals actually reached** instead of reward, is that no
checkpoint both does the task and stays safe. Reward on SafetyCarGoal is mostly distance-shaping, so
a timid agent that creeps toward the goal but never reaches it still scores ~0.4 reward. Counting
goal-completions (per-step reward spike ≥ 0.9) over 50 episodes at seed 0 exposes this:

| Checkpoint | Reward | Cost | Goals/ep | Eps reaching ≥1 goal | Safe (≤25)? |
|---|---|---|---|---|---|
| CVaR `model_200` | 0.13 | 0.0 | 0.02 | 1/50 | ✓ — **does essentially nothing** |
| CVaR `model_150` | 0.44 | 19.1 | 0.10 | 5/50 | ✓ — **barely engages** |
| CVaR `model_250` | 0.99 | 43.6 | 0.30 | 14/50 | ✗ |
| CVaR `model_100` | 1.06 | 31.5 | 0.32 | 16/50 | ✗ |
| plain HL-Gauss `model_100` | 0.47 | 24.9 | *(not counted)* | — | ✓ but at the edge |

Goals-reached and cost are **the same axis**: on SafetyCarGoal2 the goals sit among hazards, so the
only way to reach them is to enter hazards. Every checkpoint that reaches goals in even ~1/3 of
episodes is unsafe (cost 40–50); every safe checkpoint reaches a goal in ≤ 5/50 episodes. The
earlier "winner" (`model_150`, reward 0.44 / cost 19) is safe **because it barely moves** — 45/50
episodes reach zero goals. That is not a solution, it is the degenerate safe-RL collapse (minimize
cost by not participating). Do not ship it as a "winner."

**The penalty/Lagrangian family cannot escape this on this env at this budget.** κ / λ can only slide
the policy along the goals-vs-cost frontier; they cannot move the frontier itself. Getting an agent
that reaches goals *and* stays safe requires one of:

1. **Much more training** (400 iters × 8 envs is tiny for level 2) so the policy learns a *skilled*
   path that threads between hazards — moving the frontier, not sliding along it. Push to JUWELS.
2. **Runtime shielding instead of penalties** — let a reward-greedy policy chase goals while a
   safety filter blocks hazard-entering actions at execution time. The repo already has these:
   `safe_rl/cbf/cbf_filter.py`, the reachability filter (`safe_rl/filters/`, `--reach_filter` in the
   evaluator), `simplex_rta_wrapper`. This *decouples* task skill from safety and is the right tool
   when penalties just produce timidity — which is exactly what we observe here.
3. **Accept and state the trade-off** explicitly (e.g. cost limit 25 on SafetyCarGoal2 with this
   policy budget buys ~0.1 goals/episode).

## What the distributional/CVaR study still validly concludes

- **Cost-critic ranking (HL-Gauss > MSE > two-hot)** from the first study stands — that was about
  *estimation quality* and is independent of the goal-reaching collapse.
- **CVaR-gate does shift the safe operating point** (model_150 holds cost 19 with margin vs plain
  HL-Gauss model_100 pinned at 24.9), and only a distributional critic can feed a CVaR tail statistic
  into the gate. But it slides along the frontier; it does not create a safe *and* competent agent.

## Methodology takeaways (these are the durable wins of the night)

1. **Count goals, not just reward.** Dense shaping means reward ≠ task success; a "safe, positive
   reward" agent can be reaching goals ~0 times. Always report goals/episode on Goal tasks.
2. **Eval with ≥50 episodes at a fixed seed.** Per-episode cost variance is enormous — the same
   checkpoint measured cost 4.3 → 5.25 → 74.1 → 19.14 across 10/20/10/50-ep batches. A 10-ep eval
   once "found" a 1.276-reward safe winner that a 50-ep eval showed to be cost 50, unsafe, and
   reaching goals in only 16/50 episodes. Small-sample evals on this env are worthless.
3. **A checkpoint at cost 24.9/25 is not "safe."** Zero margin = violates on the next unlucky layout.

## Caveats

- Single seed per variant, one env, local GPU; even 50-ep seed-0 numbers are one point in a
  high-variance task distribution. No number here is a safety certificate. Confirm on JUWELS
  (multi-seed × many episodes) before any hard claim. Template: `sweeps/p3o_hlgauss_sweep.yaml`.
- Overnight-run configs: `config/overnight/`; checkpoints: `logs/overnight/<variant>_s1/`.
- Goal-counting script: session scratchpad `count_goals.py` (detects per-step reward spike ≥ 0.9).
