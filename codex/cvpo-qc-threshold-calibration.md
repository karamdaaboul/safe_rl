# CVPO on SafetyPointGoal1: the qc_thres proxy is mis-specified, and fixing it does not help

> **SUPERSEDED (2026-08-04) — see [[cvpo-cost-critic-investigation]].** Two claims below are
> wrong, corrected by direct measurement: (1) cost does **not** concentrate late — it is
> near-uniform in time (0.483 / 0.517 across episode halves over 200 episodes), so the
> uniform-cost assumption behind `qc_thres` is approximately valid; (2) the cost critic is
> **not** "accurate to 2%" — that comparison was aggregate-to-aggregate over truncated
> returns on a different arm. The real defect is that the critic is near-constant in state
> (calibration slope 0.094 against a true cost-to-go range of 0-38). Fixing that gets
> realized cost from ~47 to ~26. The conclusion below that the adaptive ratchet fails, and
> why, still stands.

*2026-08-03 (Claude Code). Follow-up to [[cvpo-negative-result]]. Question: after the MPO
M-step fixes ([[mpo-vs-acme-reference]]), does CVPO do better on `SafetyPointGoal1-v0` at
`cost_limit 25`? Answer: no — and the reason is now measured rather than guessed.*

## TL;DR

Three arms, 30k iterations, 8 envs, seed 1, identical except where noted. Final-phase
(last 5 logged blocks) training reward and episodic cost:

| arm | reward | cost (limit 25) |
|---|---|---|
| original M-step (caps 0.1/10, scalar KL) | **20.71** | 50.2 |
| improved M-step (uncapped, per-dim, decoupled, target-critic) | 18.11 | 46.4 |
| improved + `qc_thres` calibration | 6.92 | **81.2** |

**The MPO M-step fixes do not transfer to constrained CVPO here** — the improved arm is
slightly worse on reward and no better on cost. **The `qc_thres` calibration loop makes it
much worse.** No arm respects the budget.

## What the cost critic is actually doing (measured, not assumed)

Monte-Carlo probe (`scratchpad/cost_critic_probe.py`): roll the trained policy out, compute
`G_c(s_t) = sum_k gamma^k c_{t+k}` from visited states, compare with the critic's `Q_c(s_t,a_t)`.

```
cost per completed episode : 21.88     (limit 25)
Monte-Carlo Q_c mean       :  0.995
critic     Q_c mean        :  0.978     -> ratio 0.98
qc_thres (analytic, lim 25):  2.500
```

**The cost critic is accurate to 2%.** An earlier hypothesis that it underestimated by ~3x was
wrong. The Bellman update in `safe_sac.py::_update_cost_critic` is also correct
(`c + gamma^n * mask * Q_c'`, proper bootstrap mask).

**The mis-specification is in `qc_thres`.** It converts the episodic budget to Q-space as
`limit * (1 - gamma^H)/(1 - gamma)/H`, which assumes **costs are uniform over the episode**.
They are not: hazard contacts concentrate late, so states before t~500 discount them by
`gamma^500 ~ 0.007`. Measured `Q_c ~ 1.0` at an episodic cost of ~22, where a uniform spread
would give ~2.2. So the constraint reads satisfied in Q-space (`Eqc 2.12 < qc_thres 2.50`),
`lambda` goes to 0, and no cost pressure is applied while the budget is exceeded ~2x.

## The calibration loop: implemented, works mechanically, fails in outcome

`CVPO.update_lagrangian_multipliers` was a no-op (CVPO solves `lambda` in the E-step), but the
runner passes it the measured episodic cost every iteration. That hook now optionally closes
the loop (`qc_thres_adapt`, default **off**):

```
qc_thres <- clip(qc_thres - lr * (EMA[J_c] - limit) * scale,  0.05 * analytic,  analytic)
```

Ceiling at the analytic value (can only tighten, never loosen), floor at 5%, EMA-smoothed
because single-rollout cost variance is large (two probes of the same checkpoint: 3.0 and 21.9).

Every mechanism did what it was designed to do:

- `qc_thres` walked 2.50 -> **0.125** (the floor: maximum tightening).
- `Lambda` went **0 -> 12.09** — cost pressure fully engaged, the thing that was missing.
- `Eqc` fell to 0.17 — the E-step really is selecting its lowest-cost candidates.

And realized cost got **worse**: 81.2 vs 46-50, with reward collapsing 20.7 -> 6.9.

## Why (this is [[cvpo-negative-result]]'s mechanism, now with direct evidence)

All `sample_action_num` E-step candidates are drawn from the current, already-unsafe policy.
No reweighting of uniformly-unsafe actions reaches a safe target; raising `lambda` does not
create safe actions, it only deletes the reward signal. The agent then stops navigating
purposefully to goals and drifts — and drifting in a hazard-dense arena accumulates *more*
contact time than moving deliberately through it. Hence: maximum cost pressure, minimum
reward, higher cost.

**What this rules out.** The previous open question was whether `lambda` simply never engaged.
It now demonstrably can engage (0 -> 12), and constraint satisfaction still does not follow.
The binding limitation is own-policy candidate sampling, not the multiplier, not the critic,
and not the M-step.

## Code left in the tree (all default-off, tested)

- `CVPO(qc_thres_adapt, qc_thres_lr, qc_thres_min_frac, qc_ema)` — the loop above.
- `get_penalty_info()` reports `eqc_as_episodic_cost` (Eqc back in episodic units) and
  `realized_cost_ema`, so the proxy-vs-reality gap is visible every update instead of silent.
- The uniform-cost assumption is documented at the `qc_thres` computation.
- `SafeActorCritic` (`safe_rl/modules/safe_actor_critic.py`) — accepts standard **or**
  distributional reward critics, so CVPO can use the C51 stack that MPO benefits from. Cost
  critics stay scalar by design (the E-step needs `Q_c` as a plain expectation, and reward and
  cost scales do not share a categorical support). Untested on a full run: the planned
  distributional CVPO arm was cancelled once the defect was traced to the cost target rather
  than the reward critic.
- Tests: `tests/test_cvpo.py` covers both critic types, a CVPO update with distributional
  critics + n-step, and the calibration loop's tighten / never-loosen / static-when-off
  behaviour (19 tests).

## Recommendation

Do not pursue further reweighting-based fixes for CVPO on this task family. The evidence now
points where [[cvpo-negative-result]] already pointed, with the alternative explanations
eliminated: use runtime shielding that decouples safety from the policy's own action
distribution — the CBF projection (`safe_rl/cbf/`) or the learned reachability filter
(`safe_rl/filters/`). If CVPO is revisited, the prerequisite is a candidate-generation scheme
that can propose actions outside the current policy's support.

Cross-refs: [[cvpo-negative-result]] (CarGoal2, bang-bang lambda), [[m0-baselines]] (FSRL
reference: reward 20.6 at cost 27.1), [[mpo-vs-acme-reference]] (the M-step fixes that do
transfer, on unconstrained MPO).
