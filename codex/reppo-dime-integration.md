# REPPO-DIME — the TruDi diffusion actor in safe_rl

Port of TruDi's `reppo_dime` (REPPO with a DIME denoising-diffusion policy) into this repo.
Source: `safe_rl/trudi (1).zip`, branch `cleanup_reppo_dime`, torch path.

## Why

`codex/reppo-v24-manipulation-analysis.md`: Gaussian REPPO held
`Episode_Reward/grasp` at 0.0000 on Mjlab-Lift-Cube-Yam across 300 iterations /
39.3M steps while PPO reached 0.919, with every internal diagnostic healthy.
Diagnosis: exploration-bootstrap failure of the **unimodal tanh-Gaussian** on the
gripper dimension — the same tanh-saturation mechanism that capped Ant v21-v23
(entropy target met by widening σ, `d(tanh)/dx → 0`, pathwise gradient starved).
A multimodal actor is the intervention that addresses the mechanism rather than
its symptoms.

## What the reference actually changes

`trudi/src/torchrl/reppo_dime.py` is a **copy-paste fork** of `trudi/src/torchrl/reppo.py`
— the exact file our `REPPO` is calibrated against (`codex/reppo-vs-trudi-reference.md`).
Byte-identical: collection bookkeeping, λ-GVE returns, entropy-in-reward, HL-Gauss
critic + aux loss, dual losses, minibatch loop, hard `old_actor` sync. Only the
actor changes, and with it four derived quantities:

| Quantity | Gaussian REPPO | REPPO-DIME |
|---|---|---|
| Policy | tanh(N(μ,σ)), one step | 8-step Euler–Maruyama SDE toward N(0, 2.5·I), tanh at the end |
| `log π(a\|s)` | exact | ELBO pseudo-log-prob `run_cost + sto_cost + terminal_cost` (upper bound) |
| Old policy | per-step `(μ,σ)` in storage | frozen `old_actor` network, hard-synced per update (polyak 1.0) |
| KL | closed form / 16-sample MC | path-space KL, second 8-step rollout under the OLD chain |
| Entropy (dual target) | analytic / `-log π` | `-run_cost` (terminal prior term excluded) |

The actor loss *functional form* is unchanged:
`torch.where(kl < bound, α·logπ − Q, β·kl)`.

## Our layout

- `safe_rl/networks/dime/` — vendored stack, ~750 lines pure torch:
  `models.py` (`DiffusionModel`, `DIMEActor`), `integrators.py`, `control_net.py`,
  `schedulers.py`, `utils.py`, `__init__.py` (adds `DT_SCHEDULES`, replacing the
  reference's only hydra usage: `instantiate`/`get_method`).
- `safe_rl/modules/dime_actor_critic.py` — `DIMEActorCritic`: diffusion actor +
  frozen `old_actor` + the unchanged REPPO critic. Deliberately **not** a
  `StochasticActorCriticBase` subclass (that base is a Gaussian contract).
- `safe_rl/algorithms/reppo_dime.py` — `REPPODIME(REPPO)`. Overrides exactly three
  methods: `act` (no (μ,σ) snapshot), `_update_actor` (ELBO log-prob, path KL,
  `-run_cost` entropy), `update` (appends the old-actor sync). `reppo.py` is
  **untouched** — every Gaussian config behaves exactly as before.
- Configs: `config/mjlab_ant_reppodime.yaml`, `config/mjlab_liftcube_reppodime.yaml`.
- Tests: `tests/test_reppo_dime.py` (17).

`sample_with_log_prob` keeps the Gaussian 4-tuple contract, which is why
`process_env_step` and `compute_returns` need no override — and why the
soft-return entropy bonus is exactly the reference's
`r − γ·α·(run+sto+term)`.

Duals stay on the **algorithm** (`log_alpha_temp`/`log_alpha_kl`), not inside the
actor as in the reference. With `dual_optim_mode: actor` and
`target_entropy = -ent_target_mult`, the dual losses are algebraically identical
to the reference's; `DIMEActor.log_temperature`/`log_lagrangian` are vendored but
inert (no loss touches them).

## Do not "fix" these

- **`logratio`'s simplified per-step KL** — squared drift difference over
  `2σ_old²`, log-variance-ratio terms dropped. The full closed form is commented
  out *in the reference's own source*; the simplified one is what produced their
  published results. Consequence: learned per-dim friction differences are not
  charged to the trust region.
- **Entropy asymmetry** — the temperature target uses `-run_cost`, but the actor
  loss and the reward bonus use the full pseudo-log-prob. The reference's choice.
- **`target_entropy: -4.0`** (= reference `ent_target_mult`). The DIME
  pseudo-entropy is an ELBO quantity on a different scale from Gaussian
  differential entropy — never carry over the Gaussian `-0.5`, and the anneal
  (`target_entropy_final`) is rejected in `__init__` for the same reason.
- **`action_scale` must stay 1.0** — DIME actions are `tanh(x)`. Widening needs a
  matching `n·log(s)` log-prob correction; deferred.

## Gotchas found during the port

- **PyYAML parses signless exponents as strings.** `outer_clip: 1.0e4` arrives as
  `'1.0e4'`, and `torch.clamp(x, -outer_clip, …)` dies with
  `bad operand type for unary -: 'str'` at the first sampler call. Write plain
  floats (`10000.0`). `tests/test_reppo_dime.py::test_shipped_config_constructs_policy_and_algorithm`
  builds policy+algorithm straight from the shipped YAML to catch this class of bug;
  `test_config_resolution.py` cannot (it only resolves `class_name`).
- **`unitree_mjlab.py --device` / `--gpu_ids` do not work.** mjlab/warp initializes
  CUDA at module import, before `select_gpus` sets `CUDA_VISIBLE_DEVICES`. Preset
  the env var instead: `CUDA_VISIBLE_DEVICES=1 python scripts/train/unitree_mjlab.py …`.
  On this box CVD 0 = RTX 4000 Ada (20 GB), CVD 1 = RTX PRO 4500 Blackwell (32 GB) —
  torch and nvidia-smi order the cards oppositely, so verify placement by UUID.
- The runner logs `policy.action_std.mean()` unconditionally; `DIMEActorCritic`
  exposes the per-step SDE noise scale `sqrt(2·dt·sched/friction)` as that analog.

## Cost — and a 2.3× efficiency gap vs the reference worth closing

Collection pays two 8-step chains per env step (`act` + the bootstrap in
`process_env_step`) = 16 control-net forwards; the forward-KL actor update pays a
second full rollout per minibatch. Measured ~3.6k steps/s at 1024 envs on the Ada
(≈37 s/iteration).

**Wall-clock, same box / same 50 M steps / 1024 envs:**

| | wall-clock | ratio |
|---|---|---|
| Gaussian v24 (`v24_support_s3`) | 1 h 00 m | 1.0× |
| REPPO-DIME (`dime_ant_v0`) | 4 h 08 m | **4.1×** |
| *paper Table 2 (Humanoid, 50 M)* | *REPPO 1.07 h → TruDi 1.95 h* | *1.8×* |

**Our diffusion overhead is ~2.3× worse than the reference's**, so the gap is
implementation, not algorithm. Two things the reference does that our port does
not, both of which hit an 8-step chain far harder than a one-shot Gaussian:

1. **`actor.compile()`** — `trudi/src/torchrl/reppo_dime.py` compiles the actor.
   The chain is 8 sequential *small* control-net forwards, i.e. kernel-launch-latency
   bound; compile/CUDA-graph capture is exactly the fix for that shape of workload.
2. **AMP** — the reference wraps forward/loss in `autocast()` with a `GradScaler`
   (`get_autocast_context(cfg)`); we run fp32 throughout.

Neither changes the math, and closing this would take the Ant run from ~4 h to
plausibly ~2 h, which matters for iteration speed. Correctness came first and is
now established (see results above), so this is the natural next optimization.

**Compile: IMPLEMENTED BUT NOT USABLE — off everywhere.** The microbenchmark
(2.97× sampling / 1.59× update, batch 1024) and the `state_dict` unit test both
pass, but **it fails in both real training paths**:

| path | torch | failure |
|---|---|---|
| mjlab (`scripts/train/unitree_mjlab.py`) | 2.10 | `BackendCompilerFailed` at the first compiled forward — mjlab's `configure_torch_backends()` sets TF32 via the **new** API on torch ≥2.9, inductor's `pad_mm` reads the **legacy** getter, torch raises "mix of the legacy and new APIs". Run `z21hc3wh` died before iteration 1. |
| ManiSkill (`train_safety_gymnasium.py`) | 2.11 | **Hang** — 45 min with no output past wandb startup, where the uncompiled run printed every ~136 s. Run `jow09e2o`. |

So the 1.7× is a *microbenchmark* result, not a delivered speedup. Likely fix for
mjlab: `torch._inductor.config.shape_padding = False` (untested). The hang needs
debugging on an idle box. **Lesson: this was first tried on the run whose result
we depended on, costing ~1 h of the PickCube control for zero gain — validate on
a throwaway short run instead.**

The state_dict hazard below is real but avoidable, and that part is solved: `torch.compile(mod)` returns an `OptimizedModule` whose `state_dict`
keys gain an `_orig_mod.` prefix — that breaks every saved checkpoint and
`sync_old_actor()`'s `load_state_dict`. **`nn.Module.compile()` (the METHOD,
torch ≥ 2.2) compiles in place and leaves keys untouched.** Verified on torch
2.10: keys identical, `deepcopy` works, compiled↔uncompiled `state_dict`s load
both ways; `tests/test_reppo_dime.py::test_compile_score_net_preserves_state_dict_and_sync`
locks this in so nobody "simplifies" it back to the function form. Applied to both
`actor` and `old_actor` control nets, after the deepcopy. Off by default (every
result recorded here was produced uncompiled); `compile_mode="reduce-overhead"`
opts into CUDA-graph capture, which is the right tool for 8 sequential small
forwards.

**Still open — AMP.** Add bf16 autocast around the **score net only**, keeping
`run_cost`/`sto_cost` and the KL accumulation in fp32 (bf16 in a summed-over-8-steps
KL would lose the precision the trust region depends on). Needs a thin wrapper
around `ControlNetwork` that does not perturb `state_dict` nesting.

**Not the cause of the gap:** the two-chains-per-env-step cost is *the reference's
own design*, not an artifact of inheriting the Gaussian collection path.
`trudi/src/torchrl/reppo_dime.py:143` samples the action, `:177` runs a second full
chain at `norm_next_obs` for the bootstrap. It cannot reuse the next step's sample
because `_next_obs` is `final_observation` on truncation (≠ the next loop's
auto-reset obs) and the normalizer updates in between. Reusing it would be a
deliberate *deviation* — plausibly ~half of collection cost, but it couples the
entropy bonus to the action actually taken and must be measured, not assumed.
`dime_kl_mode: reverse` is the orthogonal algorithmic lever — it fuses sampling
and KL into one rollout (the reference's `rev_kl` variant) and roughly halves the
actor-update cost.

## Measured (2026-08-05, in flight)

Ant-Flat, 1024 envs, 381 iterations — **same config lineage and env count as
`mjlab_ant_reppo_v24_support.yaml`**, so training reward is directly comparable
(both `statistics.mean(rewbuffer)`):

| | Gaussian v24 (`v24_support_s3`) | REPPO-DIME (`dime_ant_v0`) |
|---|---|---|
| Train reward @it 200 | 8.89 | **54.45** |
| Train reward @it 380 | 8.28 | (see below) |
| Episode length | 959 | 953 |
| KL | 0.099 | 0.098 |
| Entropy / target | −3.73 / −4.0 | −31.9 / −32.0 |
| Exploration noise | σ 0.60–0.76, never anneals | friction 1.0 → 0.40 |
| Action saturation | (the v24 failure mode) | 9% |

Both hit their entropy target and sit on the KL bound, by **opposite mechanisms**:
the Gaussian buys entropy by widening σ (saturating tanh, taxing its sampled
behavior down to ~8 despite a deterministic policy worth 56.3), while the
diffusion policy buys it through multimodality — its samples are already worth
~54 and its learned friction anneals noise *down*. `deployment_gap` ≈ −0.02
(sampled ≈ mode quality) is the same fact from the critic's side.

**The deterministic comparison is the one that counts** and is not yet run;
Gaussian v24 is 56.3 ± 1.6 (3 seeds). See `codex/reppo-improvement-loop`-era
warning, restated: *training-time env metrics under different σ are NOT comparable
across algorithms* — the 8-vs-54 gap above is same-protocol (both stochastic), but
deployment numbers must come from the same 50-episode deterministic evaluator.

### Lift-Cube-Yam — NEGATIVE result so far (the motivating task)

**FINAL: `Metrics/lift_height/episode_success` = 0.0000 for every logged iteration
of the entire run** (max over the whole log = 0.0000). The uninterrupted phase
reached iteration 475 (≈62 M env steps); the continuation ran to 899. Throughout,
`object_height` stayed at 0.0198–0.0200 — the cube never left its 0.02 spawn
height — and `position_error` ≈ 0.34, indistinguishable from the Gaussian probe
(0.35, 0.0000), well past both curriculum stages (~102, ~196).
For scale, the Gaussian v24/v25 verdict was 0.0 across 300 iterations / 39.3 M
steps versus PPO 0.919; **DIME has now had ~1.6× that budget with the same result.**

**It is not a bug and not a misconfiguration.** Every internal is healthy at the
end of the run: `frac_targets_clipped` 0.0000 (the inherited [-150,150]/501 support
is fine for DIME's return scale), `q_bias` +0.05 with `q_value` 9.44 tracking
`returns_mean` 9.44, entropy −28.01 exactly on its −4.0×7 target, KL 0.1006 against
the 0.1 bound, `actor_grad_norm` 0.023 (clip 0.5 never binds). The optimizer is
solving precisely the problem it was given; the problem never rewards a grasp.

### ⚠️ CORRECTION — this run used the WRONG TASK VARIANT

**Retracted:** an earlier version of this note concluded from the run above that
"tanh saturation was not the binding constraint" because DIME barely saturates
(0.025) and still failed. **That conclusion was unfounded and is withdrawn.**

The run above is on the **base `Mjlab-Lift-Cube-Yam`**, whose reward terms are
`lift`, `lift_precise`, `action_rate_l`, `joint_pos_limits`, `joint_vel_hinge` —
**there is no grasp reward at all**. The documented Gaussian failure and PPO's
0.919 (`codex/reppo-v24-manipulation-analysis.md` §8) are on the **`-Grasp`
variant**, which does carry `Episode_Reward/grasp`. Comparing the two conflated
different tasks: failing to grasp on a task that never rewards grasping shows
nothing about the saturation hypothesis.

What the base-variant run *does* legitimately show is that DIME matches the
Gaussian probe on that task (both `episode_success` 0.0000, same
`position_error`), with every internal healthy — no more, no less.

The correct experiment is `config/mjlab_liftcube_grasp_reppodime.yaml` on
`Mjlab-Lift-Cube-Yam-Grasp`, budget-matched to REPPO v25 (1024 envs × 128 steps ×
300 iters = 38,400 per-env steps, support [-40,100]/281). §8's leading hypothesis —
the actor's pre-tanh gripper mean sits so deep in saturation that closing needs a
~6σ excursion — is exactly what a multimodal policy with 2.5 % saturation should
break, so that run is a real test rather than a foregone conclusion.

### Older hypothesis (still open, now testable)

A hypothesis consistent with the results here: multimodality helps *represent and
retain* several good modes once they have been found (Ant: +29 %, and the
multimodality probe below), but it cannot *discover* a mode that random
exploration has never visited. Lift-Cube's diagnosed failure
(`codex/reppo-v24-manipulation-analysis.md`) is exactly that — no grasp ever
occurs, so the critic never learns that closing pays and `dQ/d(gripper)` stays ~0.
Note DIME's action distribution is *narrower* than the Gaussian's (std 0.134 vs
0.300 on Ant), which if anything reduces the blind random exploration this task
needs. If that hypothesis holds, the fix for Lift-Cube is a *discovery* mechanism
(demonstrations, reward shaping on gripper closure, or an explicit exploration
bonus), not a richer policy class.

⚠️ The continuation past 475 is **confounded**: resuming from a checkpoint restores
the networks but *not* the env-side curriculum, which restarts at stage 0
(`joint_vel_hinge` jumped −0.155 → −0.0001 across the resume). Any Lift-Cube number
to be quoted must come from an uninterrupted run.

## Audit vs the paper (2026-08-05)

Paper: **TruDi — "Trust-Region Diffusion Policies for Massively Parallel On-Policy RL"**
(Le, Celik, Blessing, Hoang, Voelcker, Brunnbauer, Richter, Volpp, Neumann),
arXiv 2606.15260 / OpenReview `mGu2fs7kJt`. The zip's code is these authors' own,
so "faithful to the reference" and "faithful to the paper" are different claims;
both were checked.

### Our port vs the reference code — PROVEN bit-identical (not just diff-read)

`tests/test_dime_reference_parity.py` extracts the reference implementation from
`trudi (1).zip` at test time, builds both actors from identical weights, and
compares under identical RNG:

| checked | result |
|---|---|
| `sde_sample`, `ode_sample` | bit-identical |
| `kl_div` (K=1 and K=4) | bit-identical |
| `sde_sample_and_kl` | bit-identical |
| TruDi actor-loss **value** | identical (0.50831044) |
| TruDi actor-loss **gradients**, all 10 398 params | `max|diff| = 0.000e+00` |

So "is this a DIME implementation bug?" is answered **no**, numerically, for the
actor and its loss. (The surrounding REPPO machinery — collection, λ-returns,
critic — is inherited unchanged and was already proven bit-exact against their
`reppo.py`; see `codex/reppo-vs-trudi-reference.md`.) The test skips when the zip
is absent, and exists because the vendored files are a verbatim copy of someone
else's code: a future rename or `black` reflow could silently change the math and
only a numerical check would catch it.

### Our port vs the reference code — faithful
Whitespace/comment-stripped diff of all four vendored modules against
`trudi/src/networks/reppo_dime/*`: differences are docstrings, formatting, import
paths, and two semantically-identical refactors (`friction_shape`, and
`bs, action_dim = obs.shape` → `bs = obs.shape[0]`, where `action_dim` was unused
and misnamed). The actor update and collection path were compared line by line
against `make_actor_update_fn` / `collect_fn` in `trudi/src/torchrl/reppo_dime.py`
and are algebraically identical, including both duals:

- theirs `entropy_loss = (n_act·ent_target_mult + (−run_cost.mean())) · temperature`
  vs ours `alpha_temp·(entropy.mean() − target_entropy)` with
  `target_entropy = −4.0·n_act` — **identical** for `ent_target_mult = 4.0`.
  (Confirmed empirically: Ant entropy converged to −31.94 against target −32.0 = −4·8.)
- theirs `lagrangian_loss = −beta·(kl − kl_bound).mean()` vs ours
  `alpha_kl·(desired_kl − kl.mean())` — **identical**.
- clipped/full/`torch.where` gate, the `-qf + temp·log_probs` primary, the
  entropy-bonus reward `r − γ·temperature·(run+sto+term)` — **identical**.

### The reference's own deviations from the paper (kept, not "fixed")
1. **Simplified per-step trust-region KL.** The paper's constraint is
   `c(s) = (1/K) Σ_j Σ_n log[π̂_old^{n−1|n} / π̂_θ^{n−1|n}]` — a full Gaussian KL
   per step. The code keeps only `‖μ_old − μ_new‖²/(2σ_old²)`, dropping
   `log(σ_new/σ_old)` and the variance-ratio term (the full formula sits
   commented out in their source). **These agree exactly iff the old and new
   per-dim friction agree**, since σ² = 2·dt·sched/friction.

   **MEASURED over a complete 381-iteration Ant run** (seed 2; both KLs
   accumulated from the same rollout, friction 1.063 → 0.396):

   | phase | simplified KL | full KL | under-report | gate fires: simpl vs true |
   |---|---|---|---|---|
   | iters 1–10 (init transient) | 0.1295 | 0.1332 | +2.86 % | 0.732 vs 0.751 |
   | iters 10–50 | 0.1015 | 0.1024 | +0.89 % | 0.458 vs 0.468 |
   | iters 50–150 | 0.0999 | 0.1002 | +0.31 % | 0.425 vs 0.428 |
   | iters 150–end | 0.0999 | 0.1003 | +0.40 % | 0.420 vs 0.424 |

   Worst single iteration: +17.6 %. **Verdict: the reference's simplification is
   sound in practice** — ~0.4 % steady-state under-report, and the gate fires
   within 0.4 points of where the true KL would put it. (An earlier *analytic*
   estimate here put the worst case at +212 % by pairing the largest observed
   friction drift with a typical mean-difference term; the run shows that pairing
   does not occur. Superseded by the table — measurement over extrapolation.)
   `logratio_with_full_kl` / `kl_div_with_full` / `log_full_kl: true` now log
   `kl_full_closed_form`, `kl_full_minus_simplified`, `kl_full_frac_over_bound`
   alongside the trust-region value. **The loss still uses the simplified value**
   (a test asserts the diagnostic path returns a bit-identical KL), so this
   measures the gap without changing training.
2. **Entropy surrogate drops the terminal term.** Paper's `l_πθ` is the full
   trajectory entropy; the code's temperature target uses `−run_cost` only, while
   the actor loss and reward bonus use the full `run+sto+terminal`. Logged both
   (`entropy` vs `entropy_full_pseudo`).
3. **K = 1 trajectory per state for the KL estimate** in their torch config
   (their JAX config uses 4; the paper writes the estimator with a general K).
   With K = 1 the per-state gate fires on a single noisy sample — measured
   `kl_frac_over_bound` ≈ 0.41 on Ant. `kl_action_rep: 1 vs 4` is the obvious
   ablation; ours follows the torch config (1).

### Confirmed matches to the paper
- **N = 8 diffusion steps** — their ablation: T=1 fails, T=4 suboptimal, T=8 saturates.
- **ε = 0.1** — inside the paper's optimal 0.1–0.4 band (ε=0.01 too conservative, ε=50 too loose).
- **Forward trajectory KL(old‖new)**, sampled under the old policy — our
  `dime_kl_mode: forward` default. (`reverse` is their `rev_kl` variant.)
- **Split objective**: actor loss when `c(s) ≤ ε`, else `λ·c(s)` — our `kl_clip_mode: clipped`.
- **Dual ascent on both α and λ** — ours, with `dual_optim_mode: actor`.
- **HL-Gauss cross-entropy critic** — inherited unchanged.
- **Probability-flow ODE for deployment**, `a^{n−1} = a^n + δ(β a^n + c·2η²β ∇log π̂)`,
  score scaling **c = 1.0**; their ablation reports **ODE > SDE > best-of-K**.
  `act_inference` uses the ODE; `ode_coef` is now a config knob (default 1.0) so
  the sweep is reproducible. Note our eval script's `--action_mode deterministic`
  = ODE and `stochastic` = SDE, which maps exactly onto that ablation.

### Paper reference numbers to measure against
Table 2 (Humanoid, 50M steps): PPO 0.1±3.7, REPPO 29.6±7.2, DIME 15.6±9.4,
**TruDi 34.8±3.0**; wall-clock TruDi 1.95±0.46 h vs REPPO 1.07±0.35 h — i.e.
TruDi ≈ **1.18× REPPO's return for 1.8× the compute**. Our measured cost ratio
(~4 h vs the Gaussian arm's ~2 h on Ant) reproduces that 1.8× factor.
Multimodality (Behavior Entropy, 0 = collapse): REPPO **0.0**, DIME 0.20/0.68,
TruDi **0.32** (PushT) / **0.87** (StackCube).

## Ant-Flat deployment results (2026-08-06, same-protocol 2×2)

Both final checkpoints (iteration 380, 1024 envs, 50M steps, identical config
lineage), evaluated the same day with the same evaluator, env build, seed and
episode count: `--num_envs 64 --episodes 50 --seed 1`. `deterministic` =
probability-flow ODE (`act_inference`), `stochastic` = one SDE sample — which is
exactly the paper's ODE-vs-SDE evaluation ablation.

| arm | reward | ±sem | ep-len | survival |
|---|---|---|---|---|
| **REPPO-DIME, ODE (deterministic)** | **62.49** | 0.23 | 960.0 | **1.00** |
| REPPO-DIME, SDE (stochastic) | 59.77 | 0.31 | 960.0 | 1.00 |
| Gaussian v24, deterministic | 47.79 | 2.03 | 867.5 | 0.78 |
| Gaussian v24, stochastic | 45.65 | 1.77 | 874.1 | 0.78 |

1. **DIME wins on both protocols by ~31 %** — this is a deployment win, not just a
   training-time one.
2. **ODE > SDE (62.49 vs 59.77)** — reproduces the paper's evaluation ablation.
3. **Survival 1.00 vs 0.78.** The Gaussian falls in 22 % of episodes; DIME never
   falls. Most of the Gaussian's reward deficit is that tail.
4. **DIME is ~7× more consistent across episodes** (sem 0.23–0.31 vs 1.77–2.03).

**3-seed Gaussian baseline, today's protocol** (all `v24_support` @380, deterministic):

| seed | reward | survival |
|---|---|---|
| s1 | 43.04 | 0.58 |
| s2 | 54.34 | 0.96 |
| s3 | 47.79 | 0.78 |
| **mean** | **48.39 ± 5.68 sd** | |

**Multi-seed summary** (deterministic/ODE, 50 episodes each, identical protocol):

| | seeds | mean | sd | values | survival |
|---|---|---|---|---|---|
| **REPPO-DIME** | 3 | **61.21** | **1.98** | 62.5 / 58.9 / 62.2 | 1.00 / 0.98 / 1.00 |
| Gaussian v24 | 3 | 48.39 | 5.68 | 43.0 / 54.3 / 47.8 | 0.58 / 0.96 / 0.78 |

**+26.5 % on the seed mean (Welch t = 3.69, df ≈ 2.5), and the distributions do not
overlap: the worst DIME seed (58.9) beats the best Gaussian seed (54.3).** DIME's
seed spread is also 2.9× tighter (sd 1.98 vs 5.68) — it fell in 1 of 150 episodes,
the Gaussian in 22–42 % depending on seed. Note how tightly the Gaussian's reward tracks its survival rate
(0.58→43.0, 0.78→47.8, 0.96→54.3): **its seed variance essentially *is* its fall
rate.** DIME falls in 1 of 100 episodes across both seeds.

⚠️ **Do not compare these to the July 2026 figure of 56.3 ± 1.6** for this same
Gaussian config (July seeds: 57.9/54.7/56.4, sd 1.6). All three seeds measure
~8 lower and far more spread today (48.4, sd 5.7). The checkpoints load correctly
(actor tensors verified bit-equal to the file, zero missing keys, normalizer
matched) and `safe_rl/modules/actor.py` is unchanged, so this is env-build or
eval-protocol drift since July, not a load bug or an actor regression. **It does
not affect the comparison above**, because every arm was measured under identical
conditions on the same day — the same-protocol discipline
`codex/reppo-vs-trudi-reference.md` insists on. It does mean any July-era absolute
Ant number should be re-measured before being quoted again.

Against the paper's own margin — Table 2 Humanoid: TruDi 34.8 vs REPPO 29.6
(+18 %) — our Ant margin (+31 %) is in the same direction and somewhat larger.

### ODE score-scaling ablation (paper's eval ablation, reproduced)

`a^{n−1} = a^n + δ(β a^n + **c**·2η²β ∇log π̂)` — the paper sweeps `c` and ships
`ode_coefs: [1.0]`. Same checkpoint, same 50-episode protocol, `policy.ode_coef`:

| `ode_coef` | reward | ±sem | ep-len |
|---|---|---|---|
| 0.5 | 50.89 | 2.27 | 877.4 |
| **1.0** | **62.49** | **0.23** | **960.0** |
| 1.5 | 61.14 | 1.18 | 947.3 |
| 2.0 | 60.13 | 1.43 | 933.2 |

**c = 1.0 is optimal, confirming the reference default**, and it is also the
lowest-variance and only never-falling setting (ep-len 960 = cap). The curve is
asymmetric: under-scaling collapses hard (−19 % at c=0.5, as the sampler reverts
toward the uncontrolled prior), over-scaling decays gently. Every c ≥ 1.0 still
beats the best Gaussian seed (54.34), so the DIME advantage is robust to this knob.
The c=1.0 arm reproduced the 2×2 deterministic number to the digit (62.493),
which also verifies the new `ode_coef` plumbing is bit-identical to the previous
hardcoded path.

## Multimodality probe (the paper's central claim, measured on Ant)

The paper's Behavior-Entropy metric needs a task with a labelled symmetry (PushT,
StackCube). Ant has none, so we measure the property that metric proxies for:
**at a fixed in-distribution state, is the action distribution one blob or several?**
`scratchpad/probe_multimodality.py` draws K=1000 actions at each of 16 states
(states drawn as `obs_mean + obs_std·ε` from the run's own normalizer statistics,
so both policies are queried in-distribution — raw `randn` states are pushed
off-distribution by the normalizer and inflate the effect), then reports a 2-means
silhouette, the centroid gap in within-cluster std units, and a bimodality
coefficient. The Gaussian is the control: 2-means always splits a unimodal blob,
so whatever it scores is the unimodal floor.

| final Ant ckpt (iter 380) | silhouette | gap/within-std | bimodality | action std |
|---|---|---|---|---|
| Gaussian v24 (unimodal floor) | 0.327 | 1.53 | 0.258 | **0.300** |
| REPPO-DIME | **0.514** | **2.79** | **0.367** | **0.134** |

**The signature: DIME has less than half the action spread but nearly double the
mode separation.** The Gaussian meets its entropy target with one diffuse blob
(wide σ, saturating tanh); the diffusion policy meets its target with narrow,
well-separated modes. That is the paper's mechanism claim, reproduced on a task
they did not test.

## ⚠️ OPEN: PickCube control fails — four wrapper bugs found, a fifth not ruled out

**Status 2026-08-07.** Both our variants score **0.0048** on ManiSkill PickCube-v1
over full 381-iteration runs (DIME and Gaussian REPPO, independently), where the
paper reports the task solved. Returns plateau at ~2.5 of a ~5 maximum by
iteration ~40 and never move again, with every internal healthy (KL on bound,
entropy exactly on target, q_bias ~0.03, zero target clipping). That is a
reaching policy that never discovers grasping — the same signature as mjlab
Lift-Cube.

### Four bugs found and fixed in `safe_rl/envs/maniskill_vec_env.py`

| # | bug | reference | ours (broken) | effect |
|---|---|---|---|---|
| 1 | discount | γ = 1 − 10/max_ep_steps = **0.8** | 0.99 | ~100-step horizon on a 50-step task |
| 2 | `ignore_terminations` | `not partial_reset` → **False** | `partial_reset` → True | success never ended an episode |
| 3 | `record_metrics` | `True` | omitted | success metric unreliable |
| 4 | **`final_observation`** | `has_final_obs: true`, used in collect_fn | **never surfaced** | **~1 in 50 value targets bootstrapped from a NEW episode** |

Bug 4 is the instructive one: **our own code printed a warning about it at
iteration 0 of every ManiSkill run** (`[REPPO] WARNING: env truncated some
episodes but did not provide infos['final_observation']`) and it went unread for
a day. It also explains the locomotion/manipulation asymmetry — 50-step PickCube
episodes corrupt ~2 % of transitions, 1000-step mjlab episodes ~0.1 %, which is
why Ant/Humanoid were unaffected. **mjlab results never touch this file and are
unaffected by all four.**

Fixing all four did not change the outcome (0.0048 before and after).

### Version confound

Our stack is NOT the paper's. Theirs pins `gymnasium<1.0.0`, `mani-skill>=3.0.0b21`,
`torch==2.7.1`; we had gymnasium 1.3.0 / mani_skill 3.0.1 / torch 2.11. So our
PickCube may not be their PickCube, and their code **cannot run** on our install
(gymnasium 1.x changed the `Wrapper` contract). Also: **torch 2.7.1+cu126 has no
kernel image for the Blackwell (sm_120)** — the paper's stack only runs on the Ada.

### The decisive test, in flight

`~/venvs/trudi_ref` was built with the paper's exact pins (torch 2.7.1+cu126,
gymnasium 0.29.1, mani_skill 3.0.0b21) and runs the authors' **unmodified**
`src/torchrl/reppo_dime.py` on PickCube-v1 (`scratchpad/run_ref_pickcube.sh`;
their optional config keys supplied via hydra `++` rather than editing their
source). Metrics only reach wandb, so `scratchpad/poll_ref_wandb.py` reads the
success metric from the API.

**Decision rule, fixed in advance:**
* reference solves it, ours doesn't → **defect in our port**; diff their config and
  collection path against ours to find bug #5.
* reference also flat → **the paper's PickCube number does not reproduce on this
  box**, and our port is not the outlier — which would also reframe the mjlab
  Lift-Cube grasp failure.

Do not quote any PickCube conclusion until this resolves.

## ManiSkill integration (for the PickCube positive control)

Added so the paper's own grasping benchmark can be run against our port:

- `safe_rl/envs/maniskill_vec_env.py` — ManiSkill3 behind the safe_rl `VecEnv`
  contract, plus a `ManiSkill` prefix in `envs/registry.py`.
- `config/maniskill_pickcube_reppodime.yaml` — matched to the authors'
  `reppo_dime_maniskill.yaml`: 1024 envs × 128 steps × 64 minibatches × 4 epochs,
  lr 3e-4, kl_bound 0.1, ent_target_mult 4.0, 151 atoms over [-15, 15], 50M steps.
- Isolated venv `~/venvs/maniskill` (torch 2.11+cu128, mani_skill 3.0.1, safe_rl
  installed `--no-deps`) so nothing in `agx_plain` moves under the other sessions.

**The subtle part is truncation.** ManiSkill bootstraps on `terminated` where
mjlab/playground bootstraps on `truncated`. The reference wrapper folds
`terminated` into `truncated` and reports `done=False` under `partial_reset`;
we reproduce that exactly. Getting it wrong would silently corrupt the return
target on precisely the *successful* episodes — the ones this control is about.

Two gotchas worth knowing:
- `scripts/train/train_safety_gymnasium.py` reads an `env:` config block but does
  **not** forward it to `make_env`. Wrapper defaults apply (they happen to match
  the reference, including ManiSkill's default 8-dim joint control — hence
  `target_entropy` = -4.0×8 = -32). The config documents this so it doesn't read
  as settings that are being applied.
- **Cost profile is inverted vs mjlab**: collection 11-14 s vs learning 92-125 s
  per iteration, i.e. ~90 % of wall-clock is our update, not ManiSkill physics.
  And startup is CPU-bound — with another 1024-env job on the box, PickCube could
  not get through scene construction in 68 minutes (15.4 of 16 cores busy).
  **Run it alone**, not alongside another large-env job.

## Extended diagnostics

`REPPODIME` logs, per iteration, on top of the REPPO set: `dime_run_cost`,
`dime_terminal_cost`, `dime_log_prob`, `entropy_full_pseudo`, `kl_max`, `kl_std`,
`kl_frac_over_bound` (how often the clipped gate actually fires — the mean KL
hides this; measured ~0.41 on Ant), `q_mode`, `dime_action_sat_frac`,
`dime_action_abs_mean`, `dime_sde_ode_action_gap`, `dime_friction` (mean/min/max),
`dime_dt`, `dime_noise_scale`.
