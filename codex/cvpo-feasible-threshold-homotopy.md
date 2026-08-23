# CVPO: the feasible threshold homotopy is a negative result, and a constant lambda beats both controllers

*2026-08-06/07 (Claude Code). Builds on [[cvpo-cost-critic-investigation]], [[cvpo-negative-result]],
[[cvpo-qc-threshold-calibration]]. 13 training arms, 30k iterations each, `SafetyPointGoal1-v0`,
cost limit 25, 8 envs, GPU.*

## TL;DR

1. **The homotopy failed.** It reproduced the *unconstrained* policy: cost 49.28 ± 3.14 against a
   limit of 25, with `lambda` never leaving ~0.01. Mechanism identified and measured below.
2. **The `both_lam4` lambda controller is unreliable**, not wrong: cost 43.60 ± **14.39** across
   3 seeds (27.0 / 51.4 / 52.4). It works when the cost critic's level holds and fails when it
   collapses.
3. **A constant `lambda = 2.0` lands on the budget** — cost 25.32, reward 19.90 — matching the
   published `both_lam4` operating point (25.55 / 19.16) without any controller at all.
4. **Why**: both threshold schemes compare `E_q[Q_c]` against a threshold, so both depend on the
   critic's *level*, which is unreliable. A fixed lambda never makes that comparison and depends
   only on action *ranking*, which is weak but consistent.

## The fixed-lambda front (seed 1, 30k iters)

`lambda_lr = 0`, `lambda_init = L`, so the E-step exponent is exactly `(Q_r - L*Q_c)/eta` — no
controller, no threshold, no homotopy. All arms carry the critic-side fixes (`n_step: 10`,
`cost_critic_nonneg: true`, `qc_scale_source: measured`), identical to `both_lam4`.

| lambda | reward | cost | goals/ep | spread ratio | ess |
|---|---|---|---|---|---|
| 0.00 | 25.77 | 48.52 | 12.38 | 0.00 | 58.3 |
| 0.40 | 26.99 | 46.70 | 12.87 | 0.22 | 58.3 |
| 0.78 | 22.84 | 33.84 | 10.80 | 0.34 | 58.3 |
| 1.50 | 25.05 | 30.45 | 11.81 | 0.65 | 58.0 |
| **2.00** | **19.90** | **25.32** | 9.28 | 0.77 | — |
| 3.00 | 17.69 | 14.26 | 8.65 | 0.79 | 56.1 |
| 4.00 | 20.85 | 34.74 | 9.57 | 0.02 | — |

- `lambda = 0` reproduces the DMPO unconstrained ceiling (27.07 / 51.5), so the harness is sound.
- `lambda = 0.78` is dominated by `lambda = 1.5` on **both** axes — it was a bad operating point,
  not a trade-off. 0.78 came from [[cvpo-cost-critic-investigation]]'s
  `median_s std_a(Q_r)/std_a(Q_c)`, measured on a different arm. **It does not transfer**: the new
  `lambda_balanced` diagnostic reads 1.86–2.35 in these runs.
- `lambda = 4.0` breaks monotonicity (cost 34.74 > `lambda = 3.0`'s 14.26) with the spread ratio
  collapsing to 0.02 — the critic's action spread has gone, so the cost term goes inert. Same
  pathology that makes `lambda_max = 100` catastrophic.

**Read `estep_spread_ratio`, not `lambda`.** Only the spread across candidate actions survives the
per-state softmax; the level cancels. `spread = lambda / lambda_balanced`, so the balance point is
`lambda ~ 2` here. This is what the "128x" in [[cvpo-cost-critic-investigation]] measures
(`100/0.78`) — a spread ratio, not a level ratio.

## Head-to-head: homotopy vs both_lam4 (3 seeds each)

| arm | reward | cost | lambda (per seed) |
|---|---|---|---|
| homotopy | 26.29 ± 0.62 | **49.28 ± 3.14** | 0.006 / 0.019 / 0.019 |
| both_lam4 | 24.21 ± 4.79 | 43.60 ± 14.39 | 0.00 / 3.14 / 2.97 |
| *(fixed lambda=2.0)* | *19.90* | *25.32* | *2.0 pinned* |

Neither controller reaches the budget on average. The constant does.

### Why the homotopy failed

The threshold `thresh = max(q_target, (1-beta)*C_now)` chases the policy's own cost level, so the
constraint becomes **self-referential**. Measured on seed 1 at 30k:

```
c_now  5.193      thres  5.090      ->  thres tracks c_now to within 2%
dual_residual_lambda over training: -0.02 +0.57 -0.06 -0.08 +0.25 +0.001 -0.12 ... ~ ZERO MEAN
lambda: 0.002 0.000 0.006 0.010 0.000 0.014 ... 0.006     (never integrates)
episodic cost: 36.6 -> 71.2 -> 48.5                        (limit 25, never approached)
```

The intended violation signal is `beta * C_now ~ 0.025`. The per-batch noise in `E_q[Q_c]` is
**±0.1 to 0.6 — 4 to 20x larger**. So `lambda` random-walks near zero instead of integrating, and
the constraint is a no-op regardless of how unsafe the policy is.

What was described in the design as "self-limiting" (the ask is always just `beta` better than
where you are) is, quantitatively, **inert**. A threshold that tracks the current cost level
cannot generate a violation signal above its own batch noise. The `beta_max` reachability clamp
and the monotone ratchet are both irrelevant to this failure — the signal never accumulates in the
first place.

### Why both_lam4 is bimodal

Seed 1 trajectory: `lambda` spikes to 3.73 at step 2k, collapses to 0, stays 0 for 28k iterations.
Cause is visible in the same log — `c_now` falls to **0.05–0.9** against the static threshold 1.91,
so the constraint reads *satisfied* in Q-space while episodic cost sits at 50. That is exactly the
failure mode of [[cvpo-qc-threshold-calibration]] and [[cvpo-cost-critic-investigation]]: a
near-constant, level-deficient cost critic makes the threshold comparison meaningless.

Seed 2 held `lambda ~ 3.1` and reached cost 27.0. Seeds 1 and 3 did not (52.4, 51.4). The ±14.4
spread is entirely "did the critic's level survive".

## The structural point

| scheme | depends on | outcome |
|---|---|---|
| static threshold + controller (`both_lam4`) | critic **level** | bimodal, 43.6 ± 14.4 |
| homotopy threshold + controller | critic **level** (and its own noise floor) | inert, 49.3 ± 3.1 |
| **constant lambda** | critic **ranking** only | **25.32 at lambda = 2** |

Both threshold schemes convert the constraint into a comparison against `Q_c`'s absolute level.
That level is measured-unreliable. A constant `lambda` never performs the comparison, so it is
immune to level collapse and only needs the action ordering — weak but consistent.

## Cost-critic measurement (supporting, and a methodological warning)

`scripts/eval/cost_critic_rank_probe.py` branches K actions from each of N replay states and
correlates `Q_c` against Monte-Carlo cost-to-go **within state** (the only channel the softmax
sees — any per-state constant cancels).

**The M=1 trap. Do not repeat it.** One-way decomposition on the over-budget baseline checkpoint:

```
rollout noise (within-action) var 0.791  -> sd 0.889
TRUE action effect (unbiased)  var 0.298  -> sd 0.546   z = +4.04, 75% of states > 0
```

The action effect is real, but at M=1 the noise exceeds it, so the MC target is mostly noise and
observed rho attenuates by `sqrt(reliability)`:

| rollouts/action M | target reliability | ceiling on observed rho |
|---|---|---|
| 1 | 0.27 | 0.52 |
| 4 | 0.60 | 0.78 |
| 8 | 0.75 | 0.87 |
| 16 | 0.86 | 0.93 |

A "rho < 0.5" alarm would flag a **perfect** critic at M=1. Two probe runs (64 and 200 states) were
run at M=1 and their rho ~ 0 was uninformative; the conclusion drawn from them ("the critic cannot
rank, so no controller can work") was **wrong** and is retracted — the fixed-lambda front disproves
it directly, since `lambda = 2` cuts cost 48.5 -> 25.3 with the identical critic. A weak but
consistent signal integrated over 30k iterations is not noise.

The probe now defaults to `--repeats 8` and prints the decomposition, the reliability, the ceiling
and the de-attenuated rho natively. At M=4, n=36: de-attenuated rho **+0.021**, detectable floor
|rho| >= 0.16, and `std_a(Q_c)` is ~1% of the true action-effect sd. So the ranking is weak — but
"weak" is enough, as the front shows.

**CRN is blocked.** Common random numbers would cut the noise ~4x, but same-action branches still
diverge from step 0 even after restoring qpos/qvel/act/ctrl/qacc_warmstart, the env's
`np.random.RandomState`, the builder's terminated/truncated flags and the TimeLimit counter.
Something in the snapshot is still missing. `--crn` and `--same-action` exist and are wired; the
noise floor stays ~2.0 with them on.

## Recommendation

1. **Use a constant `lambda ~ 2`** on this task. It is the only scheme measured to hit the budget
   reliably. Confirm with 3+ seeds at `lambda = 2.0` (only seed 1 was run).
2. **Do not pursue the threshold homotopy as designed.** A threshold that tracks current cost
   cannot generate a violation signal above its own batch noise. If retried, the violation must be
   measured against the *static* target and the homotopy used only to schedule the controller gain
   — a different mechanism, not a tweak.
3. **The binding problem remains the cost critic's level stability**, which is what makes every
   threshold-based scheme bimodal. That is upstream of all controller work.

## Code state

All homotopy machinery is **default-off** and unit-tested (`tests/test_cvpo_homotopy.py`, 44 tests);
`tests/test_smoke_regression.py` byte-exact oracle passed with the flags off. Config keys:
`qc_thres_homotopy`, `homotopy_beta{,_max}`, `homotopy_ratchet`, `homotopy_cnow_ema`,
`qc_target_ema{,_horizon}`, `qc_target_jc_ema`, `qc_target_warmup_reports`, `qc_target_max_rise`,
`qc_target_{min,max}_frac`, `feasibility_probe_interval`, `lambda_init`.

Diagnostics that earned their place: `estep_spread_ratio`, `lambda_balanced`,
`lambda_over_balanced` (caught the 0.78 mis-centring), `dual_residual_lambda` (diagnosed the
homotopy failure), `c_now`, `qc_thres_eff`, `qc_thres_target`, `lambda_at_cap_frac`,
`estep_feasible`.

Configs: `safety_gymnasium_cvpo_fixlam_{0p0,0p4,0p78,1p5,2p0,3p0,4p0}.yaml`,
`safety_gymnasium_cvpo_homotopy.yaml`. Drivers: `scripts/run_fixlam_sweep.sh`,
`scripts/run_phase2_queue.sh`. Logs: `logs/fixlam_sweep/`, `logs/homotopy_ab/`.

**Caveats.** Fixed-lambda front is single-seed (repo convention is >=5); the head-to-head is 3
seeds. `lambda_init`'s byte-exactness re-check under the smoke oracle is still outstanding — it was
killed for CPU contention and not re-run.

Cross-refs: [[cvpo-cost-critic-investigation]], [[cvpo-negative-result]],
[[cvpo-qc-threshold-calibration]], [[why-mean-beat-cvar-on-pointgoal1]].
