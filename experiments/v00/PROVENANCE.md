# V0 provenance — source drift during the training sweep

The V0 training sweep ran in a working tree shared with concurrent sessions. Two of the
frozen source files changed **while the baseline seeds were training**. This note
records exactly what changed, which runs are affected, and the evidence that the change
does not invalidate the comparison. It exists so the V0 numbers can be trusted (or
rejected) on evidence rather than assertion.

## Timeline

| time | event |
|---|---|
| 12:30:58 | `reppo_v00_s1` starts |
| 13:29:17 | `reppo_v00_s2` starts |
| 13:40:06 | `reppo_v00_s4` starts |
| **13:43:09** | **`safe_rl/modules/reppo_actor_critic.py` modified (concurrent session)** |
| **13:43:20** | **`safe_rl/algorithms/reppo.py` modified (concurrent session)** |
| 14:23:36 | `reppo_v00_s3` starts |
| 14:38:24 | `reppo_v00_s5` starts |

Python binds the module at process start, so **seeds 1, 2, 4 ran the frozen code and
seeds 3, 5 ran the modified code**. All five PPO seeds started before 13:43 and in any
case touch neither file.

## What changed

A new `action_scale` argument on `REPPOActorCritic` widens the squashed action range
from `(-1, 1)` to `(-scale, +scale)` by appending an `AffineTransform` to the
`TransformedDistribution`, with the matching `-n_act * log(scale)` entropy-target
correction in `REPPO.__init__`. Motivation (from the concurrent session): on
Unitree-Go2-Flat PPO's unbounded Gaussian reaches -2.86..+4.46 and spends 61-67% of
steps beyond `|a| > 1` on the calf joints, while a tanh policy is pinned at 1.00.

Every code path is guarded on `action_scale != 1.0`. The V0 baseline config does not
set `action_scale`, so it takes the default of `1.0`.

## Equivalence evidence

`experiments/v00/check_equivalence.py` loads the frozen file (from commit `7a54733`)
and the modified file side by side, forces identical weights, and compares outputs on
fixed inputs under fixed RNG seeds:

```
action_scale on new: 1.0
act               bitwise equal: True
act_inference     bitwise equal: True
sample_with_lp    bitwise equal: True
squashed log_prob bitwise equal: True
_clamp_squashed   bitwise equal: True
```

Bitwise, not approximate: `1.0 * tanh(x)` is exact in IEEE-754, and the new clamp bound
`1.0 - 1e-6 * 1.0` equals the old literal `1.0 - 1e-6`. The `target_entropy` correction
is inside `if action_scale != 1.0` and does not execute.

**Conclusion: seeds 3 and 5 are numerically identical to what the frozen code would
have produced. All five seeds are pooled.** Had any call path differed, seeds 3 and 5
would have been discarded and re-run rather than pooled.

## Residual risk

This is an argument about one specific diff, not a guarantee about the tree. The
protection going forward is `SOURCE_FREEZE.sha256`, which is checked before and after
each training batch; any mismatch is investigated the same way before results are
pooled. V1 onward should run in an isolated worktree if concurrent sessions continue.
