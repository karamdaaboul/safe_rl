# RCPPO — Reachability-Constrained PPO: math, implementation, and results

Status as of 2026-07-06. Covers the derivation of the algorithm, exactly what the code
computes, and the empirical validation (SafetyPointGoal1 vs SafetyHalfCheetahVelocity).
See also [[cvpo-negative-result]] and `plan: merry-weaving-cookie`.

---

## 1. Problem — Constrained MDP

A CMDP augments an MDP with a per-step cost `c_t ≥ 0` and a safety margin `h(s)` with
the convention `h(s) > 0 ⇔ s is unsafe`. The **classic** safe-RL objective is a
*cumulative-budget* constraint (this is what `P3O` / `CUP` / `PPOL_PID` solve):

```
maximize   J_r(π) = E[ Σ_t γ^t r_t ]
subject to J_c(π) = E[ Σ_t γ^t c_t ]  ≤  d          (episodic cost budget d, e.g. 25)
```

via the Lagrangian `L(π, λ) = J_r(π) − λ·(J_c(π) − d)`.

**Weakness (the motivation for RCPPO):** a cumulative bound lets the agent bank safe
steps and spend them on violations. It certifies nothing about *persistent, state-wise*
safety — you cannot guarantee "the agent never enters an unsafe region."

---

## 2. Reachability reformulation (RCRL, ICML 2022; HJ-reachability-in-RL)

Define a **safety value function** = worst *future* violation along the policy's path:

```
V_h^π(s) = max_{t ≥ 0}  h(s_t),      s_0 = s,   a_t ~ π
```

This is the model-free analogue of the Hamilton–Jacobi reachability value. It obeys a
**self-consistency (max-backup) fixed point**, contracted by a discount `γ_h → 1`:

```
V_h(s) = (1 − γ_h)·h(s)  +  γ_h · max( h(s),  V_h(s') )
```

Interpretation — `V_h` is a *learned Lyapunov-like certificate*:

- `V_h(s) ≤ 0  ⇔  s` is **feasible**: no future state under π ever violates.
- `{ s : V_h(s) ≤ 0 }` approximates the **largest control-invariant safe set**.

The safe-RL problem becomes **state-wise / persistent**:

```
maximize   J_r(π)     subject to     E_{s∼ρ}[ V_h^π(s) ]  ≤  ε
```

`ε` (= `cost_limits`, we use `0.1`) is a small **feasibility tolerance, NOT a budget**.
This is the key semantic difference from P3O: in RCPPO `cost_limits` is a threshold on a
per-state max-backup value, kept near 0.

---

## 3. What RCPPO actually computes

**Margin:** `h(s) = max(cost − cost_margin, 0)`, with `cost_margin = 0`.

**Cost-critic target** — backward recursion in `safe_rl/storage/reach_rollout_storage.py`,
shapes `(N, num_costs)`:

```
V_next    = last_cost_value                (t = T−1)   else   target[t+1]
V_eff     = (1 − done)·V_next  +  timeout·bootstrap(final_obs)
backup    = (1 − γ_h)·h_t  +  γ_h · max( h_t , V_eff )
target[t] = has_next·backup  +  true_done·h_t          # true terminal ⇒ V_h = h(s)
```

Critical subtlety (documented pitfall): the timeout bootstrap enters **inside** the
`max` through `V_eff`, and is **never added into the cost** — the max-backup is nonlinear,
so PPOL_PID's additive reward-style bootstrap would be wrong here.

- Cost critic `V_h` regresses these targets by MSE, with **value clipping forced off**
  (`use_clipped_cost_loss=false`) — clipping around a GAE mean is meaningless for a
  max-backup target.
- A second head `Q_h(s, a)` (`ActorCriticReachQ.evaluate_reach_q`) regresses the *same*
  targets at `(s_t, a_t)`; the runtime `ReachabilitySafetyFilter` ranks candidate actions
  by `Q_h` (least-restrictive: replace/blend only when `Q_h(s,a) > threshold`).

**Unit-test anchor** (`tests/test_rcppo.py::test_reachability_backup_hand_computed`):
γ_h = 0.9, costs `[0,2,0,1]`, `last_v = 3` ⇒ targets `[2.2212, 2.468, 2.52, 2.8]`
(t3 = 0.1·1 + 0.9·max(1,3) = 2.8; t2 = 0.9·2.8 = 2.52; t1 = 0.1·2 + 0.9·2.52 = 2.468;
t0 = 0.9·2.468 = 2.2212). The backup math is provably correct.

---

## 4. PID Lagrangian on E[V_h] (inherited from PPOL_PID)

Feedback error on the **reachability level**:

```
e_k = E[V_h] − ε
λ_k = clip(  Kp·e_k  +  Ki·Σ_j e_j  +  Kd·(e_k − e_{k−1}),   0,   λ_max  )
```

with `(Kp, Ki, Kd) = (0.05, 0.005, 0.1)`, `λ_max = 100`. The policy uses the
**sum-normalized** surrogate advantage:

```
A = (A_r − λ·A_c) / (1 + λ)
```

As `λ → ∞`, `A → −A_c` (pure safety-descent direction); the `1 + λ` denominator keeps the
step scale-invariant, so a large λ does **not** by itself blow up the update. "λ is large"
is therefore not evidence of a bug.

**Plumbing note (verified in code):** `RCPPO.update()` calls
`storage.get_mean_episode_costs()`, which for the reach storage returns
`cost_returns.mean()` — and `cost_returns` holds the V_h targets. So the PID regulates
`E[V_h]` (logged as `Loss/reach_level_mean`). The TensorBoard tag
`SafeRL/mean_cost_constraint_0` is the runner's episodic-cost-sum **logging channel only**;
it does **not** drive λ. Reading the two as if they were the same signal is a trap.

---

## 5. Experiments and results

Signals from TensorBoard. `E[V_h]` = `Loss/reach_level_mean` (**what the PID sees**).

### Env A — `SafetyPointGoal1-v0` (diffuse per-step hazard) — the MISMATCH

| signal        | it0  | it≈100 | it≈450–500                       |
|---------------|------|--------|----------------------------------|
| E[V_h]        | ~0   | 0.31   | 0.31 (pinned for 450 iters)      |
| λ             | 0    | rising | 0.57 (full) / **5.5 in diag ×35**|
| episodic cost | ~50  | ~50    | ~50 (flat)                       |
| reward        | low  | ~26    | ~26 (**unaffected by λ**)        |

λ rose **35×** and nothing moved — cost, E[V_h], reward all flat. **λ is decoupled from
safety** on this task.

### Env B — `SafetyHalfCheetahVelocity-v1` (crisp velocity boundary) — the GOOD FIT

200 iters, 16 envs, 1024 steps/env, ε = 0.1.

| signal (cell)              | it0  | it49 | it99 | it149 | it199        |
|----------------------------|------|------|------|-------|--------------|
| **A_control** Ki=0.005 E[V_h] | 0.01 | 0.09 | 0.70 | 0.85  | **0.905 ↑**  |
| A_control  λ               | 0    | 0    | 0.11 | 0.30  | 0.49         |
| A_control  reward          | −691 | 536  | 1156 | 1462  | 1742         |
| **B_highKi** Ki=0.05  E[V_h]  | 0.03 | 0.18 | 0.70 | 0.49  | **0.54 ↓**   |
| B_highKi  λ                | 0    | 0.01 | 1.13 | 2.35  | **3.31**     |
| B_highKi  reward           | −692 | 573  | 1228 | 1501  | 1670         |

---

## 6. What worked and what didn't

**✓ The machinery is correct.** Hand-computed backup test passes; `V_h` learns a real
dynamic range (0 → 0.9, not a constant); the constraint is correctly **dormant while the
agent is slow/safe** (it0–40: reward −690→535, λ=0 because velocity < limit ⇒ h=0 ⇒
E[V_h]≈0) and **activates the moment the velocity limit is crossed**. That dormant→active
transition is exactly right.

**✓ On velocity, λ COUPLES to safety — the implementation is validated.** At matched
reward, raising the integral gain 10× (A→B) drives λ 0.49 → 3.31, which **causally bends
E[V_h] down**: B humps at 0.70 (it99) then declines to 0.54, episodic cost 66 vs A's 117,
at a small reward cost (1670 vs 1742). This "rise-then-regulate" causal link is precisely
what PointGoal never produced.

**✗ Feasibility not yet reached on velocity.** Both cells sit at E[V_h] ∈ [0.49, 0.90] vs
ε = 0.1 after only 200 iters. Regulation works but needs more gain/time. B's peak-then-
decline shape says it will converge — a tuning gap, not a failure.

**✗ PointGoal is a genuine task–method mismatch, not a code bug.** Provable by
elimination: the *same code* couples λ→safety on velocity but not on PointGoal. Reason is
structural — PointGoal's cost is a spatially **diffuse per-step hazard** with no crisp
safe/unsafe boundary. `V_h` is then nearly **constant across states** (≈0.31 everywhere),
so its policy-gradient carries almost no directional information, and `{V_h ≤ 0.1}` is
nearly empty for *any* goal-reaching policy. Reachability/HJ methods need a boundary (a
wall, a velocity limit); they have no leverage on a fog of per-step penalties.

---

## 7. Direction

1. **Finish the velocity validation (cheap, high-confidence).** One longer run with the
   integral gain raised (Ki ≈ 0.05–0.1, or `lambda_init` ≈ 0.5, 400–500 iters) to confirm
   E[V_h] → ε ≈ 0.1 and map the reward/safety Pareto point, then evaluate the learned
   `Q_h` filter on that checkpoint. Turns "partially validated" into "validated
   end-to-end."

2. **To make reachability work on Goal-type tasks, fix the margin, not the algorithm.**
   Redefine `h(s)` as a continuous geometric margin, e.g.
   `h(s) = d_safe − dist(agent, nearest_hazard)`, instead of the sparse per-step cost.
   That gives `V_h` a real boundary and the filter something to certify. This is a
   vec-env/cost-path wrapper change, not an algorithm change.

3. **Scope the writeup honestly.** Defensible claim: *RCPPO delivers state-wise
   reachability safety on tasks with a well-defined safety boundary (velocity limits,
   walls) and reuses the learned safety value as a runtime filter; on diffuse per-step-
   hazard tasks, cumulative-cost Lagrangian methods (P3O / PPOL_PID) remain the right
   tool.* Do not overclaim PointGoal.

**Recommended order:** Direction 1 now (strongest evidence, least work), Direction 2 as
follow-up if the Goal-task story is also wanted.

**⚠ Revision 2026-07-06 (long-run results): the §6 coupling claim was over-optimistic.**
The 500-iter rerun of the B_highKi config (`logs/rcppo_vel/LONG_highKi/`) did **not**
regulate: λ ramped linearly 0 → 15.2 (pure integral windup, error ≈ 0.75 throughout)
while E[V_h] stayed pinned at 0.7–0.87, episodic cost grew 0 → 388 and reward 0 → 2666.
The 200-iter B cell's dip (0.70 → 0.54) was seed noise, not causal regulation.

Mechanistic diagnosis — **advantage-scale mismatch**: reward advantages are normalized
to std 1, but cost advantages A_c = V_h_target − V_h_pred inherit the *critic residual*
scale (cost-critic MSE 0.006 ⇒ std(A_c) ≈ 0.08). In the sum-normalized surrogate
(A_r − λ·A_c)/(1+λ), the constraint term reaches parity with the reward gradient only at
λ ≈ 1/std(A_c) ≈ 12 — which the PID (Ki=0.05) reached only at iteration ~500. The run
ended exactly when λ became operative.

**✓ CONFIRMED by cell `E_opLambda`** (lambda_init=10, Ki=0.5, 250 iters,
`logs/rcppo_vel/E_opLambda/`): with λ starting in the operative range, E[V_h] is held in
[0.00, 0.08] ≤ ε = 0.1 for the *entire* run, episodic cost stays ≤ **0.75** (vs 388
unregulated) and reward still climbs to **1448** (54% of the unconstrained 2666) — a
clean Pareto point. λ decays 10 → 3.2, i.e. the PID is in regulation (slightly negative
error, integral unwinding toward the holding value). **RCPPO is validated end-to-end on
the velocity boundary task**; the earlier failures were purely the λ warm-up path: a PID
started at λ≈0 with small Ki spends hundreds of iterations below the operative scale
while the policy entrenches an unsafe gait. Practical rule: set
`lambda_init ≈ 1/std(A_c)` (estimate std(A_c) ≈ √cost-critic-MSE) or raise Ki so λ
crosses it within ~50 iterations.

**⚠ Gait caveat on cell E (found by watching the eval video):** the E_opLambda policy
satisfies the constraint but locomotes *flipped on its back*. Frame comparison proves
it's the setup, not the env: the unregulated LONG_highKi policy runs upright. Cause:
λ=10 at iteration 0 penalizes any speed before a gait exists, and `entropy_coef: 0`
locks in the first slow gait found — a back-scoot (HalfCheetah has no posture term or
unhealthy termination, so nothing corrects it). Fix under test (cell F,
`rcppo_diag_F_upright.yaml`): λ *warm-starts near 0 but with Ki=0.5*, reaching the
operative scale within ~35 iterations of violation onset (instead of 450), plus
`entropy_coef: 0.01`; also running `SafetyWalker2dVelocity-v1` (limit 2.3415 m/s), where
unhealthy termination makes non-upright gaits impossible. Lesson: λ₀ at operative scale
from iteration 0 is too blunt — the right recipe is fast integral gain, so safety
pressure arrives strong but only after a locomotion prior exists. And always watch the
video before calling a policy good.

**Cell F results (both envs): constraint enforced, reward collapsed, posture unfixed.**
F_cheetah rescued the constraint (cost 3.4 → 1.4, E[V_h] → ε) but flipped again anyway
(entropy 0.01 changed the early trajectory; the flip is a seed lottery), and reward
ratcheted 905 → 86. F_walker: constraint held (cost 0.11) but reward 291 → −89 (falls
after ~1 s). Diagnosis of the ratchet: sitting *at* the boundary, the PID error hovers
around 0⁺, the integral never unwinds, and the pointwise λ·A_c gradient keeps eroding
speed — cell E escaped only because its error stayed clearly negative. Next probes:
**G cells** (`scale_cost_advantage: true`, A/B identical otherwise — running) and
**H_rescue**: fine-tune the upright/fast/unsafe LONG_highKi `model_499` with scaled
advantages and a fresh moderate PID (new `--resume_checkpoint` flag; checkpoints carry
no λ state, so the PID restarts by design). Rationale: constraint pressure applied to an
existing upright gait should slow it within the gait family instead of re-rolling
locomotion from scratch.

**G_cheetah — scale_cost_advantage VALIDATED (A/B vs F, only the flag differs):**
λ equilibrates at ~1.5 dimensionless (vs F's endless ratchet to 4.7+), constraint holds
at ε (cost 1.2/1000), and reward *recovers* to 1060 still rising (vs F's collapse to 86).
Logged `cost_adv_std` ≈ 0.2 confirms the scale math. → `scale_cost_advantage: true`
should be the RCPPO default going forward. (Posture still flipped — the flip predates λ
activation and is shared with F's early trajectory; orthogonal issue.)

**H_rescue — rescue of a converged policy FAILS for a new reason: exploration collapse.**
Resumed the upright/fast LONG policy with scaled advantages; λ ratcheted 0.6 → 12.6
(enormous in scaled units) with zero cost response (cost ~300, reward drifting back up).
Cause found in the checkpoint: `actor.std ≈ 0.02–0.10` (entropy 0 for 500 iters) — the
policy is near-deterministic, never samples slower actions, so A_c carries no usable
gradient and no λ can create one. **Lagrangian rescue requires exploration.** Probe H2
(running): `--resume_reset_std 0.3` re-inflates the std at resume. Also queued G_walker
(400 iters, scaled): Walker2d terminates when unhealthy, so posture is guaranteed — the
robust demo path.

**✓ H2_rescue_std — SUCCESS: the final upright + safe HalfCheetah policy.** The exact
corrections relative to the failed attempts, cumulatively:

1. *vs cell E (safe but flipped):* don't train from scratch under λ₀=10 — instead
   **fine-tune the already-upright unsafe LONG policy** (`--resume_checkpoint
   .../LONG_highKi/.../model_499.pt`), so the gait family is inherited, not re-rolled.
2. *vs H (rescue that didn't bite):* **re-inflate exploration at resume**
   (`--resume_reset_std 0.3`) — the converged policy's std had collapsed to 0.02–0.10.
3. *vs F/LONG (λ ratchet / reward collapse):* **`scale_cost_advantage: true`** so λ is
   dimensionless and can equilibrate, with fresh moderate PID
   (`lagrangian_pid [0.5, 0.1, 0.1]`, `lambda_init 0.5`, entropy 0.0;
   config `rcppo_diag_H_rescue.yaml`), 200 iters.

Trajectory: cost 59 → 216 (brief re-exploration burst) → **1.2**; E[V_h] 0.44 → 0.09 ≤ ε;
λ rises to 5.0 then stays *flat* for 80+ iters (true PID equilibrium — first time seen);
reward recovers to 1673 training / **1798 ± 33 eval (20 ep, 0.6 violating steps/ep,
0.06 %)**, posture verified upright from video frames. Checkpoint:
`logs/rcppo_vel/H2_rescue_std/.../model_698.pt`.

**✓ G_walker — SUCCESS from scratch on Walker2d (limit 2.3415 m/s).** Corrections vs
F_walker (which held the constraint but collapsed to reward −89 and fell): only
**`scale_cost_advantage: true`** (config `rcppo_diag_G_scaledAdv.yaml`: entropy 0.01,
Ki 0.5, λ₀≈0) **plus 400 iters instead of 300** — no rescue needed because unhealthy
termination already guarantees upright posture. λ regulates gently around 1.3–2.4,
cost ≤ 1.6 throughout, reward climbs monotonically to 1331 training / **1705 eval
(0.25 violating steps/ep, 0.03 %)**. Checkpoint: `logs/rcppo_vel/G_walker/.../model_399.pt`.

**✓ G-suite generalization (2026-07-06, same `rcppo_diag_G_scaledAdv.yaml` verbatim,
400 iters, 16 envs each):** the from-scratch recipe transfers across morphologies with
zero per-task tuning:

| env (limit m/s)        | E[V_h] end | λ end        | cost end | reward trajectory      |
|------------------------|-----------|--------------|----------|------------------------|
| Ant (2.62)             | 0.13 ≈ ε  | **0.99 flat**| 1.6      | −212 → 688, monotone   |
| Hopper (0.74)          | 0.11 ≈ ε  | 3.0 settled  | 2.2      | 21 → 1083 → 1013       |
| Swimmer (0.23, no term)| 0.20      | 11.7 rising  | 2.5      | 1 → 100 → 74 plateau   |

Ant is textbook (λ literally 1.0 — the dimensionless prediction exact). Hopper good.
Swimmer is the predicted weak case: cost 286 at it0 (limit far below natural speed) is
regulated down to ~2.5, but the near-empty feasible set keeps the error hovering ≥ 0 and
the integral drifts (λ 11.7 ↑) — the same slow ratchet as PointGoal; a small ε bump or
per-channel deadband would stabilize it. Logs: `logs/rcppo_vel/G_G_{Ant,Hopper,Swimmer}/`.

**✓ Unconstrained (λ ≡ 0) baselines — the per-task safety tax (2026-07-07).** Previously
only HalfCheetah had a matched unsafe reference (the LONG_highKi run), so the G-suite
generalization table showed the constraint is *enforced* but never quantified the reward it
*costs*. Fix: each G-suite env retrained with the **identical** `rcppo_diag_G_scaledAdv.yaml`,
changing only `--lambda_max 0` — this clamps the PID output to 0 every iteration
(`ppol_pid.py:247,255`), so the surrogate collapses to pure reward `A = A_r` while the
reach/cost critics still train (a trained-Q_h unconstrained baseline, as in the PointGoal
`baseline_eps0.05_unconstrained` run). 400 iters, 16 envs, seed 1 (Swimmer cut at 394/400
by a session teardown, already on its reward plateau). Logs:
`logs/rcppo_vel/UNSAFE_{walker,ant,hopper,swimmer}/`. Training reward / episodic true-cost
(last-5-iter mean) vs the constrained G runs:

| env (limit m/s)        | uncon R | con R (G) | reward Δ            | uncon cost/ep | con cost/ep | cost cut |
|------------------------|---------|-----------|--------------------|---------------|-------------|----------|
| Walker2d (2.34)        | 1115    | 1331      | **−16 % (helps)**  | 152           | ≤1.6        | ~95×     |
| Ant (2.62)             | 1113    | 688       | +38 % tax          | 7.4           | 1.6         | ~4.6×    |
| Hopper (0.74)          | 2101    | 1013      | +52 % tax          | 525           | 2.2         | ~240×    |
| Swimmer (0.23, no term)| 128     | 74        | +42 % tax          | 171           | 2.5         | ~68×     |

Reading — **the safety tax is strongly task-dependent and correlates with (a) how far the
env's natural speed sits above the limit and (b) whether posture-termination already
punishes the unsafe gait:**

- **Walker2d — the constraint is free-to-beneficial.** Unconstrained sprint-and-fall gives
  short episodes (ep_len 357 vs the constrained upright gait) so it scores *less* (1115)
  than the constrained policy (1331) while cutting cost ~95×. Here the reachability
  constraint doubles as a fall regularizer — a negative tax.
- **Hopper — largest tax, largest absolute win.** Unconstrained racks up **525**
  violation-steps/ep (the 0.74 m/s limit is far below its natural sprint), cut ~240× to
  2.2 for half the reward (2101 → 1013). The clearest genuine reward/safety Pareto point.
- **Ant — the constraint over-pays.** Unconstrained cost is *already* low (7.4/ep — Ant's
  gait barely exceeds 2.62 m/s), yet the constrained run gives up ~38 % reward to shave
  that to 1.6. Ant is near-feasible unconstrained, so the Lagrangian pressure mostly costs
  speed for little violation reduction — a candidate for a looser ε or per-task gain.
- **Swimmer — the weak case, confirmed.** No termination, natural speed ≫ 0.23 limit; pays
  42 % reward to cut cost ~68×, consistent with the near-empty feasible set (the same
  ratchet flagged above).

Bottom line: RCPPO buys **1–2 orders of magnitude fewer violations on every velocity env**,
at a reward cost that ranges from *negative* (Walker) to ~50 % (Hopper). This is the matched
unsafe-vs-safe comparison §6 was missing for the non-HalfCheetah envs. (Caveat: training
means, apples-to-apples with the G-suite table; eval-time violation-rate as done for the
HalfCheetah H2/G checkpoints would refine but not reorder these.)

Advisor-facing report for this safety-tax study (training plots + λ lever plot +
constrained-vs-unconstrained rollout videos for all four envs):
https://claude.ai/code/artifact/e8004c90-9ba9-444b-8a71-9748d5757c64 . Build assets and
scripts are in the session scratchpad (`make_plots.py`, `record_videos.sh`, transcoded
web clips); compact 8-video web set is ~2 MB, embedded self-contained (~3.5 MB page).

**Final velocity-task recipe (what actually works):** `scale_cost_advantage: true`
always; from scratch on envs with posture-safe termination (Walker) it just works; on
termination-free envs (HalfCheetah) train unconstrained first, then rescue-fine-tune
with `--resume_checkpoint` + `--resume_reset_std 0.3`. Comparison page (4 videos +
plots): https://claude.ai/code/artifact/65ae2887-65a0-45af-b8cc-9aa74022d4c4

**PointGoal scaled A/B (eps0_scaledAdv vs eps0_mmin0.1, 500 iters each): coupling
unlocked, safety outcome unchanged.** Scaled: E[V_h] halved (0.03–0.05 vs 0.07–0.10),
reward/goals halved (12.2/5.5 vs 24.2/11.5) — the first behavioral response to λ ever
seen on PointGoal, so the optimizer is validated here too. BUT true hazard cost is flat
(~50 vs ~55): the policy reduces the *mean-of-V_h metric* by changing its visitation mix
(fewer goal runs, more far-from-hazard wandering), not by avoiding hazards in transit —
PointGoal1's layout forces buffer crossings on goal paths, so the feasible set for a
goal-reaching policy is nearly empty and E_ρ[V_h] is gameable. Remaining gap is the
**constraint formulation** (statewise λ(s) as in RCRL, or a quantile/max constraint on
V_h), not the optimization. Also: with ε=0 unreachable the integral ratchets forever
(λ=9.5 rising at cutoff) — use ε≈0.03 for equilibrium in any longer run.

**Signed-margin pitfalls found on the first geom run** (kept as
`logs/rcppo_geom/baseline_eps0.05_unconstrained/`, useful as an unconstrained baseline
with a trained Q_h): (1) with a *signed* margin, feasibility is V_h ≤ 0, so ε must be
0.0 — ε=0.05 left λ=0 for all 274 iterations while true episodic cost sat at ~65;
(2) margin_min=−0.4 lets deep-safe states dilute E[V_h] below ε even while hazards are
clipped every episode — use an asymmetric clip (margin_min=−0.1). Corrected run:
`logs/rcppo_geom/eps0_mmin0.1/` (ε=0, Ki=0.5, lambda_init=1, margin_min=−0.1).

**Status 2026-07-06 — both directions running in parallel on the workstation:**

- *Direction 1*: `SafetyHalfCheetahVelocity-v1`, B_highKi config (Ki=0.05) extended to
  500 iters → `logs/rcppo_vel/LONG_highKi/`.
- *Direction 2 implemented*: `GeometricMarginWrapper` (`safe_rl/envs/geom_margin_wrapper.py`)
  replaces the cost channel with the signed margin
  `h(s) = clip(d_safe − min_dist(agent, hazard centers), margin_min, ·)`
  (defaults d_safe=0.4 > hazard radius 0.2, margin_min=−0.4); train flag `--geom_margin`;
  `RCPPO(signed_margin=true)` skips the h≥0 clamp so the safe region keeps its gradient
  and `{V_h ≤ 0}` is meaningful. The true (unwrapped) episodic hazard cost stays
  comparable via `Episode/true_episode_cost` (wrapper → `final_info` → vec-env `log`).
  Config: `config/safety_gymnasium_rcppo_geom.yaml` (ε=0.05, B_highKi PID gains).
  Run: `SafetyPointGoal1-v0`, 500 iters → `logs/rcppo_geom/signed_dsafe0.4/`.
  First iters: E[V_h] starts ≈ −0.16 (signed, real dynamic range at last),
  true episodic cost ≈ 18–29 baseline.

---

## 8. Evaluated alternative: Reach-Avoid Probability Certificates (RAPC)

Proposal: replace the worst-case value `V_h(s) = max_t h(s_t)` with a survival
probability, `P_safe(s) = (1 − P_fail(s)) · E[P_safe(s')]`, and constrain
`P_safe(s) ≥ 1 − δ`. Claim: multiplicative probabilities are smooth, so the gradient
landscape on diffuse-hazard tasks is restored.

**Assessment — partially right, but it does not fix PointGoal by itself:**

1. **It is a cumulative-cost method in disguise.** Take logs of the recursion:
   `−log P_safe(s) = −log(1 − P_fail(s)) + E[−log P_safe(s')]` — i.e. an (undiscounted)
   *cumulative sum* of the transformed per-step cost `c̃_t = −log(1 − P_fail(s_t))`.
   That is structurally the same family as P3O/PPOL_PID, which is exactly *why* it has
   good gradients on diffuse hazards — and also why it gives up the invariant-set
   certificate `{V_h ≤ 0}`. The safety claim weakens from "never leaves the safe set"
   to a chance constraint `P(fail) ≤ δ`. (Known in the literature as the *safety critic*,
   e.g. SQRL, Srinivasan et al. 2020; probabilistic reachability, arXiv 2002.10126.)

2. **The root cause on PointGoal is the per-step signal, not the backup operator.**
   Our `P_fail(s)` per step would be the binary in-hazard indicator — the same sparse
   signal `h(s) ∈ {0,1}` that made `V_h` flat. Under any current policy,
   P(ever touch a hazard within an episode) ≈ 1, so undiscounted `P_safe ≈ 0` for
   *every* state — saturated again, just at the other end. (Also `−log(1−1) = −∞`
   inside a hazard; needs clipping.) The measured `V_h ≈ 0.31 everywhere` is the same
   phenomenon: max, sum, or product of a binary state-independent signal cannot produce
   state-dependent values. **No backup operator can create information the per-step
   signal doesn't carry.**

3. **What actually restores the gradient is Direction 2:** a continuous margin,
   `h(s) = d_safe − dist(agent, nearest hazard)` — or, in the probabilistic version,
   a smooth `P_fail(s) = σ((d_safe − dist)/τ)`. Once the per-step signal varies with
   position, *both* formulations work; RAPC then becomes a reasonable variant
   (implementable as a storage-level change like `RolloutStorageReach`, using a
   discounted survival backup `P_safe = (1−γ) + γ·(1−P_fail)·E[P_safe(s')]` to avoid
   the everything-eventually-fails saturation).

**Conclusion:** RAPC alone ≈ re-deriving PPOL_PID with a log-cost; it inherits the
same PointGoal limitation unless combined with a continuous margin. Priority stays:
Direction 1 (finish velocity validation), then Direction 2 (continuous `h(s)`) — after
which RAPC is an optional third formulation to compare, not a prerequisite.

---

## 9. Scale-up: Unitree G1 mjlab navigation, 3 constraint channels (2026-07-07)

First multi-constraint, high-parallelism test of the validated recipe:
`Unitree-G1-Nav-Obstacles-Safe-Collision` (hierarchical nav — high-level policy
commands a frozen low-level walker), 4096 envs × 8 steps, 2000 iters (65.5M steps,
~2h on the RTX PRO 4500). Config `config/unitree_g1_nav_obstacles_rcppo.yaml`:
`scale_cost_advantage: true`, ε = [0.05, 0.1, 0.1] on E[V_h] for
[fall, collision, velocity], PID [0.05, 0.5, 0.1], λ₀ = 0.001, λ_max = 100 (safety
net only — no hand-cap needed, unlike the PPOL_PID series which required λ_max = 5).
Run: `logs/g1_nav_rcppo/g1_navigation_obstacles/2026-07-07_07-01-54/`.

**Environment restoration first**: the venv's mujoco had been collaterally upgraded
to 3.10.0 on Jul 3 by the safety-gymnasium fork install, breaking mujoco-warp 3.5.0
(the pairing `unitree_rl_mjlab` pins and the last successful run on Jul 2 used).
Fix: restore `mujoco==3.5.0 mujoco-warp==3.5.0 warp-lang==1.12.1` (safety-gymnasium
only needs mujoco>=3.0.0, so both stacks coexist). Upgrading forward instead
(mujoco-warp 3.10.0.1) does NOT work: mjlab 1.2.0 still sets `opt.ls_parallel`,
removed in warp 3.9.1. Also needed: `G1_VELOCITY_POLICY_PATH` env var pointing at
the frozen walker checkpoint (`unitree_rl_mjlab/logs/velocity/g1_flat`) when
launching from outside the unitree repo.

**Results at iteration 1999** (reward −80 → +1.14, episode length pegged at
timeout 60 — the robots do not fall):

| channel | ε | E[V_h] end | λ end | behavior |
|---|---|---|---|---|
| fall | 0.05 | 0.01 | 0.00 | satisfied from ~iter 200 onward; λ never needed |
| collision | 0.10 | 0.08 | 2.8 | **textbook equilibration**: λ spiked to ~13, drove cost 0.67→0.03, then unwound 13→5→2.8 as the constraint stayed satisfied — the scale-fix behavior, reproduced at 4096 envs / 3 channels |
| velocity | 0.10 | 0.36 | 0.23 | **limit cycle**: λ₂ rises → crushes cost (2.2→0.003 early) → fully unwinds to 0 while satisfied → cost climbs back over ε → repeat, period ~150 iters, orbiting cost ≈ 0.25 |

**The velocity limit cycle is a new failure mode**, distinct from both the inert-λ
(scale) and ratchet (unscaled-integral) pathologies — the loop *responds* but is
underdamped: the policy (the "plant") adapts over tens of iterations while the
integral unwinds to zero the moment the error goes negative, so the pair
overshoots in both directions forever. It appears here and not on the velocity
suite because this constraint is strongly reward-coupled (nav reward wants speed;
the equilibrium requires a *sustained* λ* > 0, which a fully-unwinding integral
cannot hold). Candidate fixes, untested: (a) per-channel Ki (lower for
reward-coupled channels ~0.05); (b) integral leak/deadband instead of hard unwind;
(c) accept it — time-averaged cost ≈ 0.25 is still an ~85% reduction from the
unregulated 2.2, and the velocity budget in the env cfg (2.0, weight 0.5) marks it
as the soft constraint anyway.

Net: **the two safety-critical constraints (fall, collision) are held at 4096-env
scale with zero per-task tuning of the recipe**, reward stays positive, and no
λ_max hand-cap was needed — the direct comparison being the PPOL_PID tuning
history in `config/unitree_g1_nav_obstacles_ppol_pid.yaml` (windup, face-plant
gaming, Ki throttled to 0.003). Video verification of the final policy: see
`logs/g1_nav_rcppo/videos/`.

## Sources

- Reachability Constrained RL — https://arxiv.org/abs/2205.07536 (ICML 2022)
- Iterative Reachability Estimation for Safe RL (RESPO) — https://arxiv.org/abs/2309.13528
- HJ Reachability in RL: A Survey — https://arxiv.org/abs/2407.09645
