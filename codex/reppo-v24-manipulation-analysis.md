
c
---

## 8. The grasp-signal fix FAILED — and that rules out the §7.2 explanation

Both arms run on `Mjlab-Lift-Cube-Yam-Grasp`, 1024 envs, matched per-env budget:

| | per-env steps | `Episode_Reward/grasp` | `grasp_lift` | `episode_success` | `object_height` | mean reward |
|---|---|---|---|---|---|---|
| **PPO** (1400 x 24) | 33,600 | **0.9191** | **0.8941** | **0.6806** | 0.224 | 77.1 |
| **REPPO v25** (300 x 128) | 38,400 | **0.0000** | 0.0000 | 0.0000 | 0.0200 | 19.2 |

`0.0000` is the **only value `Episode_Reward/grasp` ever took for REPPO** — across
all 300 iterations, i.e. **39.3M environment steps over 1024 parallel envs without
a single two-finger pinch.** PPO reaches 0.919 of a 1.0 ceiling (holding the cube
~92% of the episode) and its `grasp` left zero by ~7,000 per-env steps.

REPPO's internals stayed healthy throughout: `frac_targets_clipped` 0.0000,
`q_bias` -0.15, entropy -6.81 on target, sigma 0.41, `lift` 1.016 (reaching solved).

### What this rules out

The §7.2 hypothesis — "REPPO fails because the reward has no grasp term" — is
**disproved**. The term is provably reachable and nearly saturable (PPO gets
0.919), yet adding it changed REPPO's outcome by exactly nothing. A reward signal
you never once trigger is worth zero, so the missing-gradient story was never the
binding constraint.

### What is left

A pure **exploration-bootstrap failure, specific to the gripper dimension**. The
circularity: no grasp ever occurs in the data -> the critic never learns that
closing pays -> `dQ/d(gripper)` stays ~0 -> the pathwise actor never closes -> no
grasp ever occurs. PPO breaks the same circle because its likelihood-ratio
surrogate reinforces a lucky closure directly from the realized return, without
needing the critic to have modelled it first.

Leading mechanism for why REPPO's exploration never lands one, worth testing
before anything else: **tanh saturation on the gripper action**. REPPO converges
hard onto reaching (`lift` 1.016 = reaching ~1.0), and if the actor's pre-tanh
mean for `left_finger` sits deep in the saturated region (say +2.5) then at
sigma 0.41 a sampled action is essentially always fully OPEN — closing would need
a ~6-sigma excursion. The gripper's usable band is only |a| < 0.457 wide
(scale 0.0866 m vs 0.0396 m travel, §3), so saturation there is cheap to reach and
catastrophic. Diagnostic: log the actor's mean pre-tanh gripper action and the
empirical distribution of the commanded finger position.

If that is confirmed, the fix is action-space conditioning (shrink
`YAM_ACTION_SCALE["left_finger"]` toward its true travel), NOT reward shaping and
NOT hyperparameters — consistent with the v21->v23 Ant finding that tanh
saturation starves the pathwise gradient.
