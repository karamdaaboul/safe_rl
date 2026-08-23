# PointGoal2: why the cost-limit curriculum needed two gates and two quantiles

Measured 2026-08-12, CVPO + PID-lambda on `SafetyPointGoal2-v0`, seed 2, 8 envs.

## The task is not harder to solve, it is harder to make cheap

With a slack budget PointGoal2 reaches **reward 23.5-25.5 at 9.0-9.9 goals/episode** — on par
with level 1's ~27 ceiling. Earlier level-2 runs reporting reward ~5 at 0.23 goals were
lambda-crippled, not capability-limited; do not treat them as the ceiling.

The real gap is cost per goal:

| | cost | goals | cost/goal |
|---|---|---|---|
| PointGoal1 | 26.1 | 11.4 | **2.3** |
| PointGoal2 | ~20 | ~3 | **~6.7** |

Level 2 costs ~2.7x more per goal, so a budget of 20 structurally affords ~3 goals ≈ reward 7-8.
Equal-reward parity with level 1 at equal budget is not achievable; matching level 1's reward 23
would need ~9-10 goals ≈ cost 65.

## Failure 1 — a goal-only gate is a timer

`SafeCostLimitCurriculum` originally advanced on mean goals alone. A competent agent sits at
9.3 goals against a threshold of 3, so the gate never bound: the limit stepped down once per
`window` episodes regardless, i.e. every 5000 iterations. The budget slid straight through the
achievable cost (cost tracked 85-105 while the limit descended past it), and lambda — gaining
~0.4 per stage at Ki=0.1 while the budget dropped 5.0 — could never catch up.

Fix: `gate_on_cost`. Advance only when the window's cost is also at or under the current limit.
Both gates are needed; each alone admits the failure the other rules out (cost alone is
satisfied by an agent that stops moving — the goals-0.03 failure at a fixed limit of 25).

## Failure 2 — a mean is defeated by a bimodal policy

Per-episode eval of the gated run (20 episodes) showed the policy is bimodal: 10 episodes idle
(reward ~0, cost 0) and 10 productive but expensive (reward 5-22, cost up to 96). The reported
mean — reward 6.72, cost 15.15 — describes no episode that occurred.

A mean gate waves this through: `[10,0,0,10,0,0,10,0,0,10]` averages 4.0 and clears a threshold
of 3.0 with 60% empty episodes.

Fix: `success_quantile` / `cost_quantile`. **They run in opposite directions**, because the tail
that matters is on the opposite side for each:

- goals must be **high** → the **low** quantile is strict. `0.5` = half the episodes individually clear the bar.
- cost must be **low** → the **high** quantile is strict. `0.75` = three quarters individually fit the budget.

Using the median for cost would repeat the same mistake: on the measured distribution 12 of 20
episodes cost exactly 0, so the median is 0 and any budget passes. p75 was ~20, p90 ~59, making
0.75 the tightest quantile that remains satisfiable.

## Failure 3 — over-tightening costs both reward and safety

Per-stage averages from the gated run:

| limit | goals | cost | reward |
|---|---|---|---|
| 45 | 6.10 | 48.5 | 15.45 |
| 30 | 3.20 | 26.7 | 7.83 |
| **25** | **3.35** | **20.5** | **8.50** |
| 20 | 2.85 | 23.4 | 7.24 |

Limit 25 **dominates** limit 20 on both axes — lower cost *and* higher reward. Tightening past
the knee destabilises the policy, and an unstable policy is both worse and more expensive.

Both mean-gated runs ended badly at the floor: the hot-gain run finished at reward 2.79 with
lambda 7.77 (cap 8.0), the cool-gain run was stopped at reward 5.15 with cost 29.2 and lambda
still climbing. **The floor is the problem, not the controller tuning** — at the floor the goal
gate can only refuse to tighten further, it cannot stop lambda integrating, so a budget below
what the policy can sustain grinds it toward idle. Gains change how fast that happens, not
whether it does.

## Open items

- Per-sub-env curriculum instances are collapsed with `np.min`, so the enforced budget tracks
  the *luckiest* of 8 envs. Pooling all envs into one window would remove the order statistic.
  Not yet done; the quantile run still carries this bias.
- All results above are single-seed (seed 2). The repo convention is >=5 seeds per reported
  number, and none of the run-to-run differences here are separable from seed noise yet.
- Eval episodes may share seeds across parallel envs (the evaluator warns: "only 9 distinct cost
  values among 20 episodes"), so eval-derived numbers have a smaller effective sample than n.
