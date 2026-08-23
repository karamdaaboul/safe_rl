# REPPO on Unitree-Go2-Flat: results, videos, and what actually moved the number

Date: 2026-08-04. Branch `reppo_test`. All numbers are the **length-normalized per-step
`|v_cmd - v_xy|`** printed by `scripts/eval/unitree_mjlab.py`. Do **not** use mjlab's own
`Metrics/twist/error_vel_xy` for cross-policy comparison — it is a cumulative sum over a
fixed 400-step constant, so it reads 2.5x the true mean on a full episode and scales with
episode length. See [[reppo-implementation-and-ppo-comparison]].

---

## 1. Headline

| arm | scripted | 5-seed mean | survival | 50-ep err | 50-ep reward | budget |
|---|---|---|---|---|---|---|
| PPO baseline | **0.215** | 0.270 | 5/5 | 0.287 | 51.6 | 196M |
| **v34 @200, seed 1** | **0.2335** | 0.327 | **5/5** | 0.306 | 53.2 | 52.4M |
| **v34 @200, seed 3** | **0.2404** | 0.348 | **5/5** | 0.327 | 52.8 | 52.4M |
| v33 (no parity switch) | 0.2256 | 0.293 | 4/5 | 0.271 | 54.1 | 157M |
| v34 @150, seed 1 | 0.2479 | 0.371 | 5/5 | 0.309 | 52.8 | 39.3M |
| PPO budget-matched | 0.2631 | 0.296 | 5/5 | 0.304 | 52.4 | 39.3M |
| clean_val (1024 env) | 0.2890 | — | 2/5 | 0.333 | — | 39.3M |
| v32 (`action_scale 3`) | 0.322 | 0.507 | 4/5 | — | — | 39.3M |
| reference REPPO | — | 0.617 | 5/5 | — | — | 39.3M |
| old REPPO (`action_scale 1`) | 0.607 *(falls @704)* | 0.755 | 5/5 | — | — | 39.3M |

**Start to finish the tracking error went 0.607 (falling over) -> 0.234 (5/5 survival)**, and
v34 beats the budget-matched PPO on the command-matched test (0.234 vs 0.263).

## 2. Caveats that belong next to those numbers

* **Only the "scripted" column is a controlled comparison.** It drives an identical
  command sequence via `--cmd_script` at the same seed. The 5-seed and 50-episode columns
  let each policy draw its own commands.
* **v34 @200 is 52.4M steps vs PPO's 39.3M** — a 1.33x data advantage. The budget-matched
  pair is v34 @150 (0.2479) vs PPO (0.2631).
* **Every PPO number is a single seed.** v34 has two (0.2335 / 0.2404, spread 0.007); a
  third is training. The claim "REPPO beats budget-matched PPO" rests on one PPO sample.
* **v34 wins scripted but loses the random-command average** (0.337 vs PPO's 0.296).
  Unexplained — see the open questions.
* Seed-to-seed survival in this project has swung **1/5 to 5/5 with no config change**, so
  differences below ~0.05 in these tables are not distinguishable from noise.

## 3. What actually moved the number, in order

1. **`action_scale 3`** — 0.607 -> 0.322, and it eliminated the falls. tanh caps actions at
   +-1, but mjlab Go2 sets `clip_actions: None` and PPO exploits that to command up to
   +-4.46, spending 61-67% of steps beyond |a| > 1 on the four **calf** joints. Our policy
   was clipped out of the gait it wanted, not choosing a conservative one: given the room,
   v32's calf usage landed almost exactly on PPO's (65-73% beyond +-1 vs PPO's 61-66%), and
   forward-velocity gain went 0.150 -> 0.576 (PPO 0.740).
2. **The authors' locomotion preset** (`experiment_overrides/isaaclab.yaml`: gamma 0.97,
   4096 envs, 64 steps, 8 epochs x 16 minibatches, `critic_hidden_dim` 512) plus their
   `env/isaaclab.yaml` (`reward_scaling 1.0`, support `[-10, 10]` x 151) — 0.322 -> 0.248.
3. **`init_alpha_temp: 0.01`** — turned a failing run into a working one (below).

**What did NOT move it:** the reference-parity switches. v28 (all three) 0.734, v29
(dual-clip) 0.696, v30 (`num_atoms`) 0.712, against v24's 0.755 — all inside the seed
spread. Also: lowering `target_entropy` made tracking monotonically **worse** (0.317 at
-0.75, 0.456 at -1.5), and training longer at gamma 0.99 made it worse
(`as3_full` 1200 iters: 0.467 scripted, 0.832 on 50 episodes).

## 4. The `init_alpha_temp` trap, because it will recur

With `reward_scale 1.0` and support `[-10, 10]`, the run **failed** at `init_alpha_temp: 1.0`:
97% of targets clipped through iteration 133, `q_value` pinned at 9.84, reward *degrading*
-10.5 -> -48.8.

Mechanism: the critic target is the **soft** return `r - gamma * alpha * log pi(a'|s')`, and
`action_scale 3` adds `n * log(3) = 13.2` to the entropy, so `log pi` is 13.2 more negative
per state than in a +-1 policy. At `alpha = 1.0` that bonus is ~7/step against a reward of
~0.05/step — 140x — and the lambda-return lands near 57 against a ceiling of 10. The
reference never sees this because their action range is +-1.

`init_alpha_temp: 0.01` fixes it and is **still a reference value**: the zip has
`ent_start: 1.0` with `# ent_start: 0.01` commented directly above it, and the GitHub
snapshot has 0.01 active. With it: zero clipping from iteration 1, `q_value` tracking
`returns_mean` to 4 decimals, reward -37 -> +51.

**Generalization:** `action_scale` inflates the entropy term, so it is *not* a free-standing
lever — it interacts with the value support through the soft reward. Changing one requires
re-checking the other.

## 5. Videos

All rendered at 720p (`--video_res`, added because the hardcoded 1080p buffers 6.2 GB of
frames in RAM and was repeatedly OOM-killed; 720p is 2.76 GB).

**`videos/cmp_scripted/`** — every clip is the *same* command sequence at seed 21, so they
are frame-comparable:

| clip | err | outcome |
|---|---|---|
| `ppo/` | 0.215 | survives |
| `reppo_v33/` | 0.2256 | survives (157M) |
| `reppo_v34/` | 0.2335 | survives |
| `reppo_v32_actscale3/` | 0.322 | survives |
| `reppo_v31_qa16/` | 0.607 | **falls @704** |
| `reppo_as3_full_{mode,qa1,qa4,qa16,qa64}/` | 0.467 / — | Q-greedy sweep |

**`videos/v34_5seed/`** — v34 seed 1 on the five evaluation seeds, **all survive 1000 steps**:

| seed | err |
|---|---|
| 3 | **0.178** |
| 7 | 0.526 |
| 11 | 0.310 |
| 21 | 0.279 |
| 33 | 0.341 |

Seed 21 is the interesting one: `clean_val` fell there at 401 and v33 at 434; v34 completes
it. Seed 7 (0.526) is the weakest and is what drags the 5-seed mean to 0.337.

Also on disk: `videos/clean_v32_5seed/` (2/5 survive — the falls are visible).

## 6. Open questions

1. **Why does v34 win the scripted test but lose the random-command average?** 0.234 vs
   PPO's 0.263 scripted, but 0.337 vs 0.296 over 5 random-command seeds. Either the scripted
   sequence happens to suit it, or it degrades on command draws the script does not contain.
   Seed 7's video is the place to look.
2. **PPO has no error bars.** Three seeds of PPO at matched settings would cost ~1 h and
   would decide whether the 0.234-vs-0.263 gap is real.
3. **Q-greedy was never applied to the good checkpoints.** On `as3_full` a *single* sampled
   candidate took tracking 0.467 -> 0.283 with no falls and no retraining. It has not been
   tried on v33/v34, and the `--q_argmax` path had an `action_scale` bug (fixed) that
   understated it everywhere it was used.
4. `dual_optim_mode: actor` is the last untested parity switch. Low expected value —
   the actor grad norm is 0.029 against a 0.5 clip, so the clip does not bind at convergence.

## 7. Reproduction

```bash
# the arm
python scripts/train/unitree_mjlab.py --env_id Unitree-Go2-Flat --num_envs 4096 \
  --config config/mjlab_go2_reppo_v34_lasttrunc.yaml --gpu_ids 0 --seed 1 --max_iterations 200

# the controlled comparison (identical commands across policies)
SC="150:1.0,0,0;150:0,0,1.0;150:-0.6,0,0;150:0,0.6,0;150:1.5,0,0;150:1.0,0,-1.0;100:0,0,0"
python scripts/eval/unitree_mjlab.py --env_id Unitree-Go2-Flat \
  --config config/mjlab_go2_reppo_v34_lasttrunc.yaml --checkpoint <ckpt> \
  --num_envs 1 --episodes 1 --seed 21 --headless --device cuda:0 --cmd_script "$SC"

# add --video --video_res 720p --video_dir <dir> to record it
```

Config lineage: `v32_actscale3` (action_scale) -> `v33_authorspreset` (locomotion preset +
`init_alpha_temp 0.01`) -> `v34_lasttrunc` (+ `force_last_step_truncated`).
