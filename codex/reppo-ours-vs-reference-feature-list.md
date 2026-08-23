# REPPO: what our implementation has that the original does not

Date: 2026-08-04. Branch `reppo_test`.

**Reference of record:** the TruDi zip, mirrored read-only at
`/home/human/workspaces/trudi_ref/trudi` (`src/torchrl/reppo.py`,
`src/networks/torch_models.py`, `config/reppo.yaml`).

> Not `/home/human/workspaces/reppo_original` — that is the GitHub
> `cvoelcker/reppo` clone, a **different snapshot** (`ent_start 0.01`,
> `optim.AdamW` with wd 1e-2 vs the zip's `ent_start 1.0`, plain `optim.Adam`).
> Diffing against the clone produces phantom deviations.

Comparison is against their **torch** trainer, since that is what our port
mirrors. JAX-only reference features are listed separately at the end.

---

## 1. Structural / algorithmic additions

| ours | reference |
|---|---|
| `action_scale` — squashed action range widened to (−s, +s) via an `AffineTransform` folded into the `TransformedDistribution` (exact log-Jacobian) | tanh hard-capped at ±1 |
| `alpha_kl_min` — floor under the KL dual | no floor |
| `dual_optim_mode: separate` — duals get their own optimizer and are **not** grad-clipped | duals are `nn.Parameter`s **inside the Actor**, so they always ride the actor optimizer and sit inside `clip_grad_norm_(actor.parameters(), …)` |
| `critic_loss_denominator: mask` — divide by `mask.sum()` | only the `batch` convention (`(mask * ce).mean()`) |
| `force_last_step_truncated: false` — keep the real truncation flag | always mutates `truncated[-1] = 1.0` in place, so the last rollout step both bootstraps 1-step and is dropped from the critic loss |
| `target_entropy_final` + linear anneal schedule (held flat during early critic learning) | fixed target |
| closed-form Gaussian KL for `squash: none` | 16-sample MC estimator only |

**On π_old, to avoid a false "we added something" reading:** both sides freeze the
collecting policy for the KL term — they via the `old_actor` network, we via the
`(mu, sigma)` stored per step at collection. Those are equivalent for on-policy data,
and neither is an extra feature.

## 2. Knobs the reference hardcodes

- `optimizer_class` / `weight_decay` / `betas` — theirs is plain `Adam` with library defaults
- `critic_learning_rate` separate from the actor's — theirs is one shared `lr`
- `log_std_squash` ∈ {`clamp`, `tanh`, `sigmoid`} — theirs is plain unbounded `exp(log_std)`
- `critic_type` ∈ {`standard`, `distributional`, `reference`} — theirs is distributional-only
- `reward_scale` applied inside the algorithm — theirs multiplies in the env wrapper
  (same effect, different location; keeps runner/wandb reward logging raw)
- `reward_normalization` / `reward_norm_g_max` — adaptive running normalizer; theirs is a
  fixed per-env constant. (A `RewardNormalizer` exists in their `reppo_util.py` but is
  unused FastTD3 leftover.)

## 3. Network options

- **SimBa / SimbaV2** backbones for actor and critic — theirs is FCNN + RMSNorm + swish only
  (their `use_actor_skip` / `use_critic_skip` flags are JAX-only and default off)
- `zero_init_prior` bias-init variant on `DistributionalCritic` — theirs has only the
  additive learnable prior `logits + 40.9 · hl_gauss(0)`

## 4. Diagnostics (none exist upstream)

`q_bias`, `frac_targets_clipped`, `returns_mean` / `returns_max`,
`deployment_gap` (Δ_dep = `E_s[E_a[Q(s,a)] − Q(s, tanh μ(s))]`),
`target_entropy`, `actor_grad_norm` / `critic_grad_norm`.

## 5. Evaluation tooling (ours entirely)

- `--q_argmax N` — sampled Q-greedy action selection at eval
- `--cmd_script` — scripted velocity commands so different policies face an **identical**
  task (a shared `--seed` does *not* achieve this: the command RNG interleaves with other
  randomized events)
- `--dump_traj` — per-step commanded-vs-achieved velocity and per-dim actions
- length-normalized per-step tracking error, printed alongside mjlab's own
  `Metrics/twist/error_vel_xy`, which is a **cumulative sum over a fixed constant**
  (`resampling_time_range[1] / step_dt` = 400 steps for Go2), i.e. 2.5× the true mean on a
  full episode and scaling with episode length
- `scripts/eval/q_greedy_probe.py` — predicted-vs-realized Q-improvement probe

---

## Removed, so no longer a difference

**Target networks (`use_target_networks`, `tau`, `actor_target`, `critic_target`,
`soft_update_targets`, `evaluate_q_target`, `target_sample_with_log_prob`).** The
reference has no target critic: its bootstrap reads the live actor/critic
(`collect_fn:172,178`), and freezing `next_values` once per iteration at collection
*is* the target mechanism. Polyak targets were inherited SAC scaffolding; 0 of 56
configs enabled them, and 8 old configs were silently inheriting the constructor
default `True`. Removed 2026-08-04.

**Twin critics + `actor_q_reduction` (`min` / `mean` / `q1`).** The reference builds
exactly one `Critic` (`src/torchrl/reppo.py:945`), stores it as `train_state.critic`, and
both update functions read that same object. No `qf2`, no `torch.minimum` anywhere. We now
match: a single `self.critic`, and (after the target-network removal) no target copy.

Checkpoints written before the removal (keys `critics.0.*` / `critic_targets.0.*`) still
load via a `_load_from_state_dict` remap; a stored **second** critic is dropped with a
printed warning rather than silently ignored, since keeping only critic 1 would misreport
a twin-min policy as reproduced.

## Dead weight worth deleting for full cleanliness

`normalize_advantage_per_mini_batch` — accepted, stored, never used. REPPO's actor is
pathwise and has no advantages.

## The reverse direction: reference features we lack

- `"value"` KL-clip mode (reward term only, no KL)
- `reverse_kl` and the `reduce_kl` multiplier — **JAX only**
- per-env exploration-noise scaling with importance-weighted λ-returns and `lmbda_min` — **JAX only**
- aux head additionally predicting **reward**, and masking the aux loss by `(1 − done)` — **JAX only**
- `experiment_overrides` merging — present in their JAX `main`, **absent from the torch one**,
  so `experiment_overrides=…` silently does nothing under the torch entry point

## Shared defects (wrong in both, not a porting gap)

- No env wrapper forwards `infos["final_observation"]`, so a truncated step bootstraps `Q`
  on the auto-reset observation. The step is masked out of the critic loss, but its target
  still propagates one step backwards through the λ recursion.
- The entropy bonus is not masked on terminal steps (their JAX version masks it; the torch
  port has the masking commented out and we inherited that).

---

## Verification status

`tests/test_reppo.py` 22 passed (including a legacy-checkpoint remap test, a
no-target-networks test, and a privileged-critic-obs test); `tests/test_config_resolution.py`
486 passed across all 56 REPPO configs after stripping the removed keys. **The full suite
has not been run to completion yet.**

Configs that previously used twin-min (`brax_ant_reppo`, `mjlab_ant_reppo`,
`mjlab_ant_reppo_v2/v19/v20/v21/v25/v27`, `safety_gymnasium_reppo`,
`unitree_g1_flat_reppo*`) carry a header note: their twin-min arm is **no longer
reproducible** from those files.


---

## Refactor, 2026-08-04

`REPPOActorCritic` was rewritten to mirror the reference's object graph: an actor, a
single critic, and two `EmpiricalNormalization` instances — nothing else. Actor
construction, the observation normalizers, and the squashed action distribution moved
to `safe_rl/modules/stochastic_actor_critic_base.py` (`StochasticActorCriticBase`),
which deliberately owns **no** critic: REPPO needs one and SAC needs twin+targets, and
forcing a common layout would misrepresent both.

**Asymmetric observations are unaffected.** `num_critic_obs`, `critic_obs_normalizer`
(a separate instance with its own statistics), and normalization on every Q path are
all preserved — `critic_target` (the polyak network) was removed, `critic_obs` was not.

Status: `tests/test_reppo.py` 22 passed, `tests/test_config_resolution.py` 486 passed.
Full suite not yet re-run. `SACActorCritic` does **not** yet inherit from the base.
