# REPPO v24 on mjlab manipulation (`Mjlab-Lift-Cube-Yam`) — what was done, and why it fails

Date: 2026-07-31. Branch `reppo_test`. All numbers below are measured on this
workstation (RTX PRO 4500 Blackwell / RTX 4000 Ada) unless marked otherwise.

---

## 1. What was done

### 1.1 Established that the task runs at all — no library change needed

| Requirement | Finding |
|---|---|
| Task exists | `Mjlab-Lift-Cube-Yam` (+ `-Rgb`, `-Depth`) ships in **mjlab 1.2.0**, already installed in `/home/human/venvs/agx_plain` and `/home/human/venvs/mjlab_ffs` |
| Cluster parity | JUWELS `mjlab311_booster` venv has the identical mjlab 1.2.0, the `manipulation/config/yam` package, and the 18 vendored YAM STL assets |
| Registration | `import mjlab.tasks` walks every subpackage, so `register_mjlab_task` fires as an import side effect. No `gym.register`, no `-v0` suffix |
| env_id routing | `safe_rl/envs/registry.py:31` already routes `Mjlab*` → `MjlabVecEnv`. Unchanged |
| Obs contract | Task uses obs groups named `actor`/`critic` — exactly what `MjlabVecEnv._convert_observations` expects. 29-D obs (joint_pos 8, joint_vel 8, ee_to_cube 3, cube_to_goal 3, actions 7), 7-D action |
| Assets | Vendored in the wheel; cube built procedurally. **No download** |
| SLURM | `scripts/slurm/reppo_mjlab.sh` already takes `ENV_ID` as positional arg 5 |
| PPO baseline arm | `--config none` → `convert_mjlab_ppo_cfg` reads only fields the task's `yam_lift_cube_ppo_runner_cfg()` supplies. Works out of the box |

**Answer to "how hard is it to run": under an hour, and zero library code.**

### 1.2 Three real fixes (all in this repo)

1. **`scripts/train/unitree_mjlab.py` + `scripts/eval/unitree_mjlab.py` — resilient
   optional task import.** `import src.tasks` walks every task package in the
   `unitree_rl_mjlab` checkout. A concurrent session's in-progress
   `g1_perception_ffs` task referenced `mdp.camera_depth_ffs_dr`, which existed in
   `ffs_dr.py` but was not exported from `velocity/mdp/__init__.py`. That
   `AttributeError` **aborted every mjlab run, not just manipulation**. Now warns
   and continues, since lift-cube comes from `mjlab.tasks` and never needs the
   fork. Also guarded `runner.add_git_repo_to_log(src.tasks.__file__)`, which
   `NameError`d on the same path. *(The other session has since exported the
   symbol; the guard remains as protection.)*
2. **`scripts/eval/unitree_mjlab.py` — report mjlab `Metrics/*`.** The evaluator
   printed only reward/cost/length. For manipulation the metric of record is
   success rate, and `reppo_mjlab.sh` auto-runs eval, so the six planned JUWELS
   jobs would have produced no usable number. Now collects any `Metrics/…` key
   generically from `extras["log"]` at episode boundaries.
3. **`--experiment_name` must be passed explicitly** — `load_safe_rl_yaml` pops and
   discards the YAML's, so logs otherwise land under mjlab's `yam_lift_cube/`.

### 1.3 Configs added

| File | Role |
|---|---|
| `config/mjlab_liftcube_reppo_v24.yaml` | Go2-v24 verbatim in `algorithm:`/`policy:` (verified byte-identical), only identity fields changed. The uncalibrated control |
| `config/mjlab_liftcube_reppo_v24_support.yaml` | `v_min -20 → -150`, `num_atoms 301 → 501`. **Superseded — its premise was disproven, see §2.2** |
| `config/mjlab_liftcube_reppo_v25.yaml` | Four levers tuned *for* manipulation. See §3 |

`pytest tests/test_config_resolution.py` → **390 passed**.

### 1.4 Runs executed (all local, 1024 envs)

| Run | iters × steps/env | per-env steps | wandb |
|---|---|---|---|
| REPPO v24 | 100 × 128 | 12,800 | `liftcube_v24_probe` |
| REPPO v24_support | 260 × 128 | 33,280 | `liftcube_v24_support_probe` |
| PPO baseline | 1400 × 24 | 33,600 | `liftcube_ppo_probe` |
| REPPO v25 | 300 × 128 | 38,400 | `liftcube_v25_probe` |

### 1.5 NOT done

- **Nothing pushed or submitted to JUWELS.** Sync verified by dry-run only.
- No env-side changes (action scales left alone).

---

## 2. Results

### 2.1 The headline, at matched per-env budget

| | per-env steps | `episode_success` | `object_height` | `position_error` | mean reward | σ |
|---|---|---|---|---|---|---|
| **PPO** (task's registered cfg) | 33,600 | **0.78 – 0.83** | 0.23 | 0.063 | 47.1 | 0.41 |
| **REPPO v24** | 33,280 | **0.0000** | 0.0200 | 0.347 | 17.8 | 0.46 |
| **REPPO v25** (tuned) | 38,400 | **0.0000** | 0.0201 | 0.346 | 19.3 | 0.41–0.42 |

`object_height` 0.0200 **is the cube's spawn height**. Across ~560 combined REPPO
iterations the cube never moved. `lift_precise` and `at_goal` were identically 0.

REPPO's internals were healthy the whole time — this is not a numerical failure:
`frac_targets_clipped` 0.0000, `q_bias` −0.25, KL 0.02 inside its bound, entropy
converged onto its target. It learned **reaching** (`Episode_Reward/lift`
plateaued at ~1.03–1.05, i.e. reaching ≈ 1.0 with bringing ≈ 0) and stopped.

### 2.2 Two of my own predictions were wrong — recorded so they aren't re-derived

- **The value-support floor scare did not materialise.** I predicted the
  `joint_vel_hinge` curriculum at weight −1.0 would drive returns to ≈ −32 and
  clip against `v_min: -20`. Measured: the penalty rate held flat at ≈ −0.026
  across all three stages (−0.01 → −0.1 → −1.0) because *the policy adapts as the
  weight rises*. `[-20, 150]` never clipped at either end. **`v24_support` was
  unnecessary.**
- **Curriculum timing** fires on reset with a strict `common_step_counter > stage`,
  and all envs reset in lockstep every 1000 steps → stages land at **iterations
  102 and 196** (steps 13000/25000), not the 94/188 obtained by naive division.
- **The early `frac_targets_clipped` = 0.99 spike is not the reward.**
  `compute_returns` folds the entropy bonus in at full weight *after*
  `reward_scale` (`safe_rl/algorithms/reppo.py:391,413`), so while `alpha_temp`
  decays from 1.0 the α·H term dominates the return. It self-resolves by ~it 20.
  Setting `init_alpha_temp: 0.1` in v25 removed it entirely (0.0000 all run).

---

## 3. The v25 tuning attempt, and its falsification

**Hypothesis:** the YAM gripper travels 0.0396 m (`left_finger` range
−0.00205…0.037524) but its action scale is `YAM_ACTION_SCALE["left_finger"] =
0.0866 m`, so tanh ±1 spans **2.19× the entire travel** and only |a| < 0.457 is
meaningful. v24's entropy dual pinned σ at 0.46–0.49 — *wider than the gripper's
whole controllable band* — so the gripper is slammed open/closed at random at
50 Hz. (`joint6` is oversaturated the same way: scale 5.52 vs range ±2.09.)

**v25 levers:** `target_entropy` −0.5 → −1.0 /dim; `init_alpha_temp` 1.0 → 0.1;
support `[-20,150]@301` → `[-40,100]@281` (bin 0.500); `desired_kl` 0.1 → 0.03.

**Outcome: every lever did what it was designed to do, and the task result did not
move.** Entropy hit −6.98 against its −7.0 target; σ fell to **0.41 — exactly
PPO's value**; clipping stayed 0.0000. Success stayed at 0.0000 and the cube never
left spawn height.

**So noise magnitude is definitively not the differentiator: PPO succeeds at
σ = 0.41 and REPPO fails at σ = 0.41.** (A secondary error of mine: I predicted
−1.0/dim would yield σ ≈ 0.18. It gave 0.41, because for a tanh-squashed Gaussian
the squash correction dominates the entropy, not log σ.)

---

## 4. Why locomotion works and manipulation does not

### 4.0 First, a caveat on the premise

"It works on Ant/Go2/Humanoid" deserves scrutiny. Measured today:

| task | v24 run | iters | mean reward |
|---|---|---|---|
| Ant-Flat | `v24_support` s2 / s3 | 381 | **8.28 / 4.91** |
| Unitree-Go2-Flat | `go2_v24_1h` | 300 | 49.45 (ep len 956/1000) |
| Humanoid-Flat | `humanoid_v24_4096` | 148 | 33.16 (ep len 942/1000) |

The PPO baseline on mjlab Ant-Flat is **~14** (from prior sessions' notes, not
re-verified today), and the REPPO authors' own trainer reaches 32.8 on this task.
**So v24 underperforms PPO on Ant too — by roughly 2×.** Go2 and Humanoid produce
healthy-looking curves with near-full episode lengths, but I have **no PPO
baseline for either** on this machine, so "works" there is unverified.

The honest framing is therefore a **gradient, not a cliff**: REPPO v24 is
mediocre-to-poor on Ant, plausibly fine on Go2/Humanoid, and *catastrophic* on
manipulation. Manipulation is the extreme end of an existing weakness, not a new
one.

### 4.1 The core mechanism — pathwise gradient needs a critic that has seen success

REPPO's actor is trained **pathwise**: maximize `Q(s, a_θ(s))` by differentiating
through the critic (`actor_q_reduction: q1`, `aux_loss_mult: 1.0` explicitly there
to smooth `dQ/da`, which the v23 header calls "the signal the pathwise actor
consumes"). PPO instead uses a likelihood-ratio surrogate weighted by *realized*
advantages.

That difference is decisive here:

- **Locomotion:** every action dimension changes the reward *smoothly and
  immediately*. Perturb a hip torque → base velocity changes → velocity-tracking
  reward changes. `dQ/da` is informative from the first update, everywhere in
  state space. The critic never needs to have seen a "success event" because there
  is no discrete success event — reward is a continuous function of the state.
- **Manipulation:** the reward has a **plateau with a cliff at its edge**.
  `staged_position_reward = reaching · (1 + bringing)` is dense in the *reaching*
  coordinate, and REPPO climbs it to reaching ≈ 1.0 without trouble. But
  `bringing` and `lift_precise` are `exp(−‖cube − goal‖²/std²)` with std 0.3 and
  **0.05** — and the cube only moves once a grasp physically succeeds. Until then
  those terms are *identically zero for every action*, so `∂Q/∂(gripper) ≈ 0`
  everywhere in the visited data. **The pathwise gradient has literally nothing to
  point at.** The actor is at a stationary point of the only signal it consumes.

PPO escapes the same plateau because one lucky episode in which random gripper
noise happens to pinch the cube produces a *realized* return above the batch
baseline, and the surrogate multiplies that trajectory's log-probability directly.
It needs no critic gradient w.r.t. the gripper — only a scalar advantage. That is
why a rare event bootstraps PPO and cannot bootstrap REPPO.

This is exactly the falsification test written into the v25 header, and it is the
hypothesis left standing after σ was equalised.

### 4.2 Supporting factors, in descending order of confidence

1. **Reward topology: dense-but-plateaued vs dense-everywhere.** Locomotion reward
   is improvable by infinitesimal action changes at every point. Lift-cube's
   second stage requires a *discrete, contact-mediated event* before any gradient
   appears. Shaping that is dense in "distance to cube" is not dense in "is the
   cube grasped".
2. **Contact discontinuity in the critic's target.** The distributional critic must
   represent a return distribution whose support jumps when contact is made. It is
   trained on data containing zero such transitions, so it cannot extrapolate to
   them — and the actor only ever sees the extrapolation-free region. Locomotion
   contacts are continuous, frequent, and present in the data from step 1.
3. **Action-space conditioning attacks the pathwise gradient specifically.** With
   the gripper scale at 2.19× travel and `joint6` at 2.6× range, a tanh-squashed
   actor sits in the saturated region on exactly the dimensions that matter for
   grasping, where `d tanh/dx → 0`. That multiplies the (already ~0) `dQ/da` by
   another vanishing factor. PPO's log-prob gradient does not carry this Jacobian
   factor in the same way. This is the mechanism the v21→v23 work already
   identified on Ant ("std creeping 0.60 → 0.76 … saturates tanh … starves the
   pathwise gradient") — **the same failure mode, worse, because manipulation's
   action scales are worse conditioned than a quadruped's.**
4. **Exploration structure.** Locomotion succeeds from unstructured noise: a
   randomly flailing Ant still translates and collects some reward. Grasping needs
   a *temporally extended, coordinated* sequence (approach → align → close →
   lift). Per-step i.i.d. Gaussian noise almost never produces it, and neither
   algorithm has temporally-correlated exploration — PPO just needs the event to
   happen once, REPPO needs it to happen often enough to shape a critic gradient.
5. **Entropy target tuned for locomotion.** `-0.5`/dim (the reference's
   `ent_target_mult 0.5`) keeps noise high, which is beneficial for locomotion
   robustness and harmful for precision. **Tested and falsified as the primary
   cause** (§3), but it is still the wrong default for this task class.

### 4.3 What is *not* the explanation (each measured, not assumed)

- ❌ Value support / `v_min` / `v_max` — `frac_targets_clipped` 0.0000, returns
  18–26 sitting comfortably inside; widening the floor changed nothing.
- ❌ `reward_scale` — 10.0 puts returns in the same 18–26 band Ant/Go2 converge in.
- ❌ KL trust region — measured 0.02, inside its bound the whole run; tightening
  0.1 → 0.03 changed nothing.
- ❌ Action-noise magnitude — equalised to PPO's 0.41 with no effect.
- ❌ Critic numerics — `q_bias` ≈ −0.25, no inflation, no clipping.
- ❌ Budget — REPPO had *more* per-env experience than PPO in every comparison.
- ❌ Episode/termination pathology — episode length pinned at 1000 (pure timeout),
  `ee_ground_collision` never fired, so the arm is not diving into the floor.

---

## 5. Recommendations (not executed)

1. **Attack the bootstrap, not the hyperparameters.** Seed the buffer with
   successful grasps (scripted or PPO-generated) so `∂Q/∂gripper` becomes non-zero.
   This is the only intervention that addresses §4.1 directly.
2. **Re-condition the action space** — shrink `left_finger`'s scale toward its
   0.0396 m travel and `joint6` toward its ±2.09 range. Cheap, and it also helps
   Ant per the v21→v23 findings. Note it changes the task, so PPO must be re-run
   for a fair comparison.
3. **If reporting a negative result**, run REPPO v24 + PPO at 3 seeds each on
   JUWELS at full budget. `scripts/slurm/reppo_mjlab.sh` is ready; both arms fit
   the 6 h wall comfortably (measured 8.6 s/iter REPPO, 0.55 s/iter PPO at 1024
   envs; learning dominates and is nearly `num_envs`-independent, so more envs is
   nearly free).
4. **Delete `config/mjlab_liftcube_reppo_v24_support.yaml`** — §2.2 disproved its
   premise, and keeping it invites re-deriving a wrong conclusion.
5. **Get a PPO baseline for Go2 and Humanoid** before claiming v24 "works" there.
   Given Ant measures 4.9–8.3 against a ~14 PPO baseline, that claim is currently
   unsupported for the whole locomotion set.

---

## 6. Reference-REPPO harness (set up 2026-07-31, ready to run)

To separate "REPPO the algorithm fails at contact-rich manipulation" from "the
safe_rl port has a bug", the authors' own trainer (`cvoelcker/reppo`, checked out
at `/home/human/workspaces/reppo_original`) now runs on the same task.

**Status: set up and smoke-tested end-to-end (~10,500 steps/s at 1024 envs).
No full comparison run has been made yet.**

### Venv — `/home/human/venvs/reppo_ref` (py3.12)

Fresh, isolated; nothing else on this box was modified.

```bash
uv venv --python 3.12 /home/human/venvs/reppo_ref
VIRTUAL_ENV=/home/human/venvs/reppo_ref uv pip install "mjlab==1.2.0" \
    hydra-core omegaconf torchinfo tqdm tensordict scipy
# CRITICAL PIN — uv otherwise resolves mujoco/mujoco-warp 3.11 and warp 1.15,
# and mjlab 1.2.0 then dies with
#   AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1.
VIRTUAL_ENV=/home/human/venvs/reppo_ref uv pip install \
    "mujoco==3.5.0" "mujoco-warp==3.5.0" "warp-lang==1.12.1" \
    "numpy==2.2.6" "scipy==1.15.3"
```

Their torch trainer imports no jax/brax/maniskill, and `make_envs` imports the
backend lazily per branch, so none of the heavy reference deps are needed.

### New files in `reppo_original/`

- `run_mjlab_liftcube.py` — builds the mjlab env, hands it to their
  `register_prebuilt`, then `runpy`s `src/torchrl/reppo.py` as `__main__`.
  Much simpler than `run_mjlab_ant.py`: that script's `sys.meta_path` surgery
  exists only because `unitree_rl_mjlab` owns a conflicting top-level `src`, and
  `Mjlab-Lift-Cube-Yam` is registered by mjlab itself, so unitree is never
  imported. Caller CLI args override the launcher's defaults (hydra treats a
  repeated key as a duplicate, not last-wins).
- `config/env/mjlab_liftcube.yaml` — `reward_scaling: 10.0`, `vmin: -20`,
  `vmax: 150`, deliberately matched to `config/mjlab_liftcube_reppo_v24.yaml` so
  any difference is attributable to the implementation rather than the support.
  Their Ant/brax default of `reward_scaling: 0.1` would put the discounted return
  near 0.2 against a 150-wide support.

### Run it

```bash
cd /home/human/workspaces/reppo_original
MUJOCO_GL=egl /home/human/venvs/reppo_ref/bin/python run_mjlab_liftcube.py \
    hyperparameters.total_time_steps=40000000
```

Their defaults already match our v24 closely (num_steps 128, num_mini_batches
128, num_envs 1024, num_epochs 4, gamma 0.99, lmbda 0.95, lr 3e-4, hidden 512);
`num_bins` is 151 vs our 301.

### Caveat worth knowing before trusting the result

`run_mjlab_ant.py` existed already but **every recorded reference run in
`outputs/` is `env=brax env.name=ant`** — the mjlab bridge had never actually been
executed, and the repo's own `.venv` has no mjlab. The adapter
(`src/env_utils/torch_wrappers/mjlab_env.py`) is nonetheless a good fit: its
`_split` reads `actor`/`policy` + `critic` groups, which is exactly what lift-cube
exposes, and it returns the 5-tuple step the trainer unpacks.

**Prediction to test:** if §4.1 is right, the reference should also plateau at
reaching with 0 success, because the pathwise-gradient-on-a-plateaued-reward
problem is structural and shared. If instead it lifts the cube, that points hard
at a bug in `safe_rl/algorithms/reppo.py` and is the more valuable outcome.

---

## 7. CORRECTION — REPPO is fine at manipulation; the mjlab reward is the problem

§4 framed this as "REPPO fails at contact-rich manipulation". **That is wrong**, and
the reference repo contains the disproof.

### 7.1 REPPO solves ManiSkill manipulation with default hyperparameters

`reppo_original/results/maniskill/*.csv` (normalized dense return, mean over
trials, 50M steps):

| task | start → final |
|---|---|
| PegInsertionSide-v1 | 0.000 → **0.980** |
| UnitreeG1TransportBox-v1 | 0.000 → **0.999** |
| PickSingleYCB-v1 | 0.428 → **0.802** |
| PokeCube-v1 | 0.102 → 0.733 |
| RollBall-v1 | 0.022 → 0.614 |
| UnitreeG1PlaceAppleInBowl-v1 | 0.000 → 0.583 |

And `config/experiment_overrides/maniskill.yaml` is three lines — `lmbda: 0.95`,
`num_epochs: 4`, `aux_loss_mult: 1.0` — i.e. the defaults. **No special tuning.**
Grasping tasks, learned from zero.

### 7.2 The difference is an explicit grasp term in the reward

ManiSkill `PickSingleYCB.compute_dense_reward`:

```python
reaching_reward = 1 - tanh(5 * tcp_to_obj_dist)
reward  = reaching_reward
reward += is_grasped                      # <-- +1, contact-based, immediate
reward += place_reward * is_grasped
reward += info["is_obj_placed"] * is_grasped
reward[info["success"]] = 6                # normalized_dense divides by 6
```

`is_grasped = agent.is_grasping(obj)` is a contact-force test on **both** fingers
(`min_force=0.5` N, `max_angle=85`). It pays out the instant the gripper closes,
whether or not the object has moved — so `dQ/d(gripper)` is non-zero from the
first update.

mjlab's lift-cube has no such term. Both positive terms are functions of
**positions only** (`reaching`, `bringing`, `exp(-||cube-goal||^2)`), so the
gripper dimension has identically zero gradient until the cube physically moves —
and nothing rewards the action that would move it. §4.1's mechanism was right;
its attribution was not. The cause is **reward shaping in this specific mjlab
task**, not REPPO.

Supporting env-side differences: ManiSkill `max_episode_steps: 50` vs mjlab's
1000 (at gamma 0.99 the horizon covers the whole ManiSkill episode but ~10% of a
lift-cube one), `normalized_dense` reward keeping returns tight enough for a
`vmin/vmax` of -15/+15, and `partial_reset: true` vs mjlab's lockstep resets.

### 7.3 New task: `Mjlab-Lift-Cube-Yam-Grasp`

`safe_rl/envs/mjlab_tasks/lift_cube_grasp.py` — registered via
`safe_rl.envs.mjlab_tasks.register_all()`, which `scripts/train/unitree_mjlab.py`
and `scripts/eval/unitree_mjlab.py` now call. Implemented in this repo rather than
by editing `site-packages`, so it survives an rsync to JUWELS.

Additive only — the five stock reward terms are untouched:

* two `ContactSensorCfg`s, left/right fingertip geoms (`[lr]f_down(6..11)_collision`)
  vs the `cube` body;
* `grasp` (weight +1.0) = 1.0 when **both** fingers press the cube above 0.5 N —
  mirrors `reward += is_grasped`;
* `grasp_lift` (weight +1.0) = `bringing * is_grasped` — mirrors
  `place_reward * is_grasped`; cannot be farmed by nudging the cube along the table;
* `is_grasped` appended to actor+critic obs (29 → **30 dims**).

Scaled return ceiling rises from `20*(2+1)=60` to `20*(2+1+1+1)=80` — still well
inside the `[-20, 150]` support, so the existing calibration carries over.

**Validated, not assumed:**
- sensors resolve (6 primary geoms per finger, `found` shape `[B, 6]`);
- `is_grasped` returns 1.0 and **sustains for 50 steps under free physics** on a
  genuine two-finger pinch;
- it correctly returns 0 when only one finger contacts. A scripted test initially
  looked like a 75% false-negative rate, but the 6 "failing" envs had one finger at
  *exactly* 0.0 N — the cube was resting on the closed gripper, not pinched. The
  `z > 0.05` proxy was wrong, not the sensor. When both fingers do touch, forces
  are >= 0.97 N, so the 0.5 N threshold separates cleanly;
- training runs end-to-end; `Episode_Reward/grasp` and `/grasp_lift` log correctly;
- `pytest tests/` → **512 passed**.

**Fair-comparison warning:** adding reward terms changes the task. The PPO
baseline must be re-run on `Mjlab-Lift-Cube-Yam-Grasp`; do not compare against the
stock-task PPO numbers in §2.1.

### 7.4 Reference-REPPO result on the STOCK task (§5 experiment, now conclusive)

RUN COMPLETE: full 50M env steps, `reward_scaling: 10` and `vmin/vmax` matched
to our v24. **All 20 evaluations reported `success rate: 0.00`.**

```
Eval return: 204.52  @  2.5M   success 0.00
        ...
Eval return: 187.62  @ 49.8M   success 0.00
```

Return drifts DOWN (raw 20.5 -> 18.8) rather than up, and 50M is 1.45x the 34.4M
steps at which PPO reaches 0.75 on the same task. So the authors' own
implementation reproduces our port's failure exactly (raw return ~19-20, success
0.00 at every eval) — **the safe_rl port is not buggy**, and the fault is the task
reward, per §7.2.

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
