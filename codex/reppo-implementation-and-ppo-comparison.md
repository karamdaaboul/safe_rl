# REPPO: how our implementation works, and how it compares to PPO on Go2

Date: 2026-08-03. Branch `reppo_test`. Task of record: mjlab `Unitree-Go2-Flat`.
All numbers measured on this workstation (RTX PRO 4500 Blackwell + RTX 4000 Ada).

This note has two halves:

1. **How REPPO is implemented here** — the algorithm, every design choice, and
   where it lives in the tree.
2. **Measured comparison against PPO** — with a correction to the metric that
   invalidates every tracking number reported before today.

Prior notes this supersedes in part: `reppo-vs-trudi-reference.md` (still correct
on the code diff; its tracking numbers are in the wrong units).

---

## Part 0 — the metric correction, read this first

**mjlab's `Metrics/twist/error_vel_xy` is not an average.** From
`mjlab/tasks/velocity/mdp/velocity_command.py::_update_metrics`:

```python
max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
self.metrics["error_vel_xy"] += torch.norm(
    self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
) / max_command_step
```

It is a **cumulative sum over the episode divided by a fixed constant** — for Go2,
`resampling_time_range = (3.0, 8.0)` and `step_dt = 0.02`, so the constant is 400
steps while an episode is 1000 steps. Therefore:

* For a full episode it reads **2.5x the true per-step mean**.
* It **scales with episode length**. A policy that falls at step 290 scores ~3.4x
  "better" than an identical policy that survives 1000 steps.

The authors' reference probe (`trudi_ref/eval_their_ckpt.py`) reports
`track_err_sum / ep_len` — a true per-step mean. **Every cross-implementation
tracking comparison made before today compared these two quantities directly**,
i.e. was off by 2.5x in our disfavour, with an additional episode-length bias.

Fixed in `scripts/eval/unitree_mjlab.py`: the evaluator now always computes and
prints `Mean per-step |v_cmd - v_xy|` (length-normalized) alongside mjlab's own
metric, with the trap documented at the computation site. Validated: 0.3513
per-step vs 0.8757 reported on the same rollout = exactly 2.493.

**Use the per-step number. Convert old numbers with `per_step = reported * 400 / episode_length`.**

---

## Part 1 — how REPPO is implemented

### 1.1 What REPPO is

REPPO (Relative-Entropy PPO, arXiv:2507.11019) is **not** a PPO variant despite the
name. It is an off-policy-style max-entropy actor-critic with an on-policy data
pipeline:

```
max_theta  E_s[ E_{a~pi_theta}[Q(s,a)] + alpha * H(pi_theta(.|s)) ]
s.t.       E_s[ KL(pi_old(.|s) || pi_theta(.|s)) ] <= eps
```

Three properties matter for everything below:

* **The actor learns pathwise**, not by likelihood ratio. With
  `a_theta = tanh(mu_theta(s) + sigma_theta(s) * xi)`, the gradient is
  `grad_theta E[Q] = E_xi[ dQ/da * da/dtheta ]` — the critic is *differentiated
  through the action*. PPO instead scores actions with a clipped ratio and never
  differentiates its value function w.r.t. the action.
* **Both multipliers are learned duals**, updated by gradient ascent every
  minibatch (512 times per iteration), not fixed coefficients.
* **The critic is a distributional Q(s,a)**, categorical over a fixed support,
  trained with HL-Gauss cross-entropy.

### 1.2 Where it lives

| Component | File | Key symbol |
|---|---|---|
| Algorithm | `safe_rl/algorithms/reppo.py` | `class REPPO` (`:15`) |
| Policy wrapper | `safe_rl/modules/reppo_actor_critic.py` | `REPPOActorCritic` (`:15`) |
| Actor | `safe_rl/modules/actor.py` | `StochasticActor` (`:203`) |
| Critic (reference-shaped) | `safe_rl/modules/critic.py` | `ReferenceREPPOCritic` (`:290`) |
| Critic (alternative) | `safe_rl/modules/critic.py` | `DistributionalCritic` (`:391`) |
| Obs normalizer | `safe_rl/modules/normalizer.py` | `EmpiricalNormalization` |
| Storage | `safe_rl/storage/rollout_storage.py` | `RolloutStorage(store_next_obs=True)` |
| Driver | `safe_rl/runners/on_policy_runner.py` | `OnPolicyRunner` |
| Configs | `config/mjlab_{ant,go2,humanoid}_reppo_v*.yaml` | |

### 1.3 The rollout (collection time)

`REPPO.act` (`reppo.py:282`) then `REPPO.process_env_step` (`:314`), per env step:

1. **Normalize once, store normalized.** Observations go through
   `actor_obs_normalizer` / `critic_obs_normalizer` and the *normalized* tensors are
   what land in storage. This is deliberate: if you store raw obs and re-normalize at
   update time, the normalizer statistics have moved during the rollout and the KL
   term measures *input drift* rather than policy change. (Measured 0.92 of mu-shift
   leaking into the first step's KL before this was fixed.)
2. **Sample and store `(mu, sigma)`** of the *base* Normal (pre-tanh). These are the
   `pi_old` parameters the KL is later measured against — no separate frozen policy
   snapshot is needed for on-policy collection.
3. **Reward shaping** (`:322`): `r <- r * reward_scale`, optionally through
   `RewardNormalization` first. `reward_scale: 10.0` on mjlab because dt-scaled sim
   rewards are ~0.05/step, which makes Q-differences tiny next to the fixed-scale KL
   cost and the actor barely moves.
4. **Bootstrap quantities computed here, not at update time** (`:380`, commit
   `919308d`): with the *collection-time* normalizer statistics and the
   *current* temperature, compute `a' ~ pi(s')`, `logp'`, `V' = Q(s',a')`, and the
   aux target `features(s',a')`. Also the entropy bonus
   `ent_bonus = -gamma * alpha * logp'`.
5. **Termination vs truncation**: `dones = terminated OR truncated`;
   `infos["time_outs"]` carries truncation separately
   (`safe_rl/envs/mjlab_vec_env.py:83`).

**Known gap:** no env wrapper in this repo forwards `infos["final_observation"]`,
so on a truncated step the Q-bootstrap uses the auto-reset observation. The step is
masked out of the critic loss, but its target still propagates one step backwards
through the lambda recursion. `reppo.py:353` warns once per run. The reference has
the identical gap (`has_final_obs: false` for mjlab), so this is parity, not a
regression — but it is wrong in both.

### 1.4 The critic target — soft-Q lambda return

`compute_returns` (`reppo.py:411`), per stored step `t`:

```
r'_t     = r_t - gamma * alpha * log pi(a'_t | s'_t)        # entropy in the REWARD, full weight
blend_t  = V'_t                                     if truncated_t
         = (1-lam) * V'_t + lam * target_{t+1}      otherwise
m_t      = max(1 - done_t, truncated_t)
target_t = r'_t + gamma * m_t * blend_t
```

Three subtleties, each of which was a bug at some point:

* **Entropy is counted once, at full weight, inside the reward.** Subtracting
  `alpha*log pi(a_t|s_t)` as well double-counts it; folding `-alpha*logp'` into the
  `(1-lam)`-blended bootstrap instead *undercounts* the future entropy chain by
  `(1-lam)`.
* **The lambda trace is cut on truncation** — a timeout ends the episode, so the
  buffered step `t+1` belongs to a new episode and blending its return backwards
  leaks it.
* **Truncated steps are masked out of the critic loss** (`:641`), and the mask
  denominator is a config switch (see 1.7).

Targets are embedded with HL-Gauss (`_hlgauss_embed`, `:696`): bin centres at
`linspace(v_min, v_max, num_atoms)`, edges extended half a bin beyond each end,
`sigma = 0.75 * bin_width`, targets **clamped** into `[v_min, v_max]`. The clamping
is why value support is a real hyperparameter: a clamped target makes cross-entropy
push probability mass into the edge bin, Q saturates at the ceiling, and `dQ/da` —
the actor's entire signal — goes flat. `frac_targets_clipped` is logged for exactly
this reason.

Critic loss = cross-entropy against the soft target + `aux_loss_mult` times a
**self-predictive auxiliary loss**: `pred_module(phi(s,a))` regressed onto
`sg[phi(s',a')]`. The predictor head matters — without it the loss degenerates into
pulling the critic's representation toward its own next-state features, which
smooths `dQ/da` directly.

### 1.5 The actor update

`_update_actor` (`reppo.py:723`):

```python
a_pi, logp_pi, mu_new, sigma_new = policy.sample_with_log_prob(obs)   # rsample, tanh-squashed
q_pi    = Q(critic_obs, a_pi)                                        # gradient flows through a_pi
primary = alpha_temp.detach() * logp_pi - q_pi
```

Critic parameters get `requires_grad = False` for the whole actor pass so the actor
step does not populate critic `.grad` buffers (`:737`, restored at `:806`). The
gradient still flows *through* the critic to the action — only parameter grads are
suppressed.

**KL term.** Under tanh squashing there is no closed form, so a 16-sample MC
estimate from the *old* policy (reconstructed from stored `mu_old, sigma_old`):

```
KL(pi_old || pi_new) ~= mean_{16 samples a~pi_old} [ log pi_old(a) - log pi_new(a) ]
```

Gradient flows only through `log pi_new`. For an unsquashed Gaussian the closed form
is used instead.

**Trust region — `kl_clip_mode`.** Two reference-supported modes:

* `"clipped"` (author default, what all our configs use): a **per-sample hard gate**
  ```python
  actor_loss = where(kl < desired_kl, primary, alpha_kl.detach() * kl).mean()
  ```
  When a sample's KL exceeds the bound its reward term is *replaced* by the KL
  penalty — a structural brake, not a soft cost.
* `"full"`: soft Lagrangian, `primary + alpha_kl * kl`.

**Duals.** `log_alpha_temp` and `log_alpha_kl` are `nn.Parameter`s; losses are
```python
alpha_temp_loss = alpha_temp * (entropy.mean().detach() - target_entropy)
alpha_kl_loss   = alpha_kl   * (desired_kl - kl.mean().detach())
```
so `alpha_temp` falls while entropy exceeds target, `alpha_kl` rises while KL
exceeds the bound. One dual step per minibatch = **512 per iteration**.
`target_entropy` is `-0.5 * num_actions` (= -6.0 for Go2's 12 actuators).

> Historical trap worth keeping: `num_actions` was once resolved via a
> `getattr(policy.actor, "num_actions", 1)` fallback that silently returned **1**,
> making the target -0.5 instead of -6.0. The temperature dual then actively *held*
> entropy at -0.5, pumping sigma back up whenever the policy tried to sharpen.
> Regression-tested now.

### 1.6 Networks

* **Actor** (`StochasticActor`, mlp path): `Linear -> RMSNorm -> swish` per hidden
  layer at width 512, then separate `mean_head` and `log_std_head`. State-dependent
  sigma. `log_std_squash: clamp` to `[-10, 2]`; additive `min_std` floor available
  (0.0 in all current configs). Tanh squashing is applied at the *wrapper* level
  (`TransformedDistribution` with `TanhTransform`); stored `(mu, sigma)` are the base
  Normal's.
* **Critic** (`ReferenceREPPOCritic`): a shared `feature_module` encoder feeding
  **two** heads — `critic_module` to `num_atoms` logits and `pred_module` to features
  for the aux loss. The split matters: the aux loss shapes the encoder two layers
  *below* the logits, so `aux_loss_mult: 1.0` does not smooth `dQ/da` directly.
  Plus an **additive learnable zero prior**: `logits = head(f) + 40.9 * zero_dist`
  with `zero_dist` initialised to `hl_gauss(0)`, so `E[Q] ~ 0` at init.
* Optimizers: plain Adam, one lr 3e-4, no weight decay, `max_grad_norm 0.5`, no LR
  schedule. Three optimizers (actor / critic / duals) or two, per `dual_optim_mode`.

### 1.7 Reference-parity switches (added today)

Four residual differences from the authors' torch trainer, each now a config flag
defaulting to the historical behaviour. All four are on in
`config/mjlab_go2_reppo_v28_refparity.yaml`.

| flag | default (ours) | reference | what it changes |
|---|---|---|---|
| `dual_optim_mode` | `separate` | `actor` | Their `log_temp`/`log_lagrange` are Actor `nn.Parameter`s, so they ride the single actor Adam **and** sit inside `clip_grad_norm_(actor.parameters(), 0.5)`. Ours had a separate optimizer and were never clipped. |
| `num_atoms` (config) | 301 | 151 | Their Go2 run used the hydra default over the same `[-20, 150]`. HL-Gauss sigma 0.85 vs our 0.43. |
| `force_last_step_truncated` | `false` | `true` | Their `compute_gve` mutates `truncated[-1] = 1.0` **in place**, so the last rollout step both bootstraps 1-step and is dropped from the critic loss, for every env. |
| `critic_loss_denominator` | `mask` | `batch` | They normalize the masked CE/aux losses by the full batch; we divided by `mask.sum()`. |

**Verified identical to the reference, do not re-litigate:** normalizer eps
placement (`sqrt(var + eps)` both sides), plain Adam (the *zip* uses `optim.Adam`;
the GitHub clone uses AdamW — different snapshots), `init_alpha_temp 1.0` = their
`ent_start 1.0`, the tanh transform (our `_clamp_squashed` builds a fresh tensor so
`cache_size=1` is bypassed and the `atanh` path is taken, same as theirs), the actor
head split (one `Linear(512 -> 2n)` is algebraically our two `Linear(512 -> n)`), and
the `log_std` clamp (never binds at sigma ~ 0.5).

**Reference of record is the zip (TruDi) copy**, mirrored read-only at
`/home/human/workspaces/trudi_ref/` — *not* `/home/human/workspaces/reppo_original`,
which is a different snapshot (`ent_start 0.01`, `AdamW` wd 1e-2) and will produce
phantom "deviations" if diffed.

### 1.8 Diagnostics logged every iteration

`value_function`, `surrogate`, `entropy`, `kl`, `q_value`, `alpha_temp`, `alpha_kl`,
`alpha_temp_loss`, `alpha_kl_loss`, `actor_grad_norm`, `critic_grad_norm` (added
today), `returns_mean`, `returns_max`, `q_bias` (E[Q - lambda-target] on the same
pairs; persistently positive = overestimation), `frac_targets_clipped`.

---

## Part 2 — REPPO vs PPO on Go2

### 2.1 Configuration differences

| | PPO (mjlab task-tuned) | REPPO v24 / v28 |
|---|---|---|
| trust region | `desired_kl` **0.01** + adaptive LR schedule | `desired_kl` **0.1**, fixed LR |
| entropy | `entropy_coef` 0.01, **fixed** | **hard target** -0.5/dim, dual-enforced |
| action std | scalar, state-independent; **anneals to ~0.1** | state-dependent; **pinned at ~0.50** |
| actor gradient | clipped likelihood ratio on GAE advantages | **pathwise** `dQ/da * da/dtheta` |
| critic | scalar V(s), MSE | distributional Q(s,a), 151-301 atoms, HL-Gauss CE |
| grad steps / iter | 20 (5 epochs x 4 minibatches) | **512** (4 x 128) |
| steps / env | 24 | 128 |
| lr | 1e-3, adaptive | 3e-4, fixed |
| net | [512, 256, 128] elu | [512, 512] swish + RMSNorm |
| gamma / lam | 0.99 / 0.95 | 0.99 / 0.95 (identical) |

### 2.2 Results — per-step tracking error

Protocol: `scripts/eval/unitree_mjlab.py`, `--num_envs 1 --episodes 1`, deterministic
action `tanh(mu)`, seeds 3 / 7 / 11 / 21 / 33. Error is the length-normalized
per-step `|v_cmd - v_xy|`. Reference numbers come from `trudi_ref/eval_their_ckpt.py`,
which uses the same definition.

| arm | survives | 3 | 7 | 11 | 21 | 33 | **mean** | yaw err | budget |
|---|---|---|---|---|---|---|---|---|---|
| **PPO** | 5/5 | 0.18 | 0.38 | 0.28 | 0.23 | 0.28 | **0.270** | 0.086 | ~200M |
| reference REPPO (zip) | 5/5 | 0.51 | 1.17 | 0.64 | 0.36 | 0.41 | **0.617** | ~0.16 | 39.3M |
| ours v28 (4 fixes) | 5/5 | 0.39 | 1.25 | 0.61 | 0.53 | 0.88 | **0.734** | ~0.13 | 39.3M |
| ours v24_normfix | 5/5 | 0.35 | 1.28 | 0.67 | 0.55 | 0.92 | **0.755** | ~0.12 | 39.3M |
| ours v24_stagger | 4/5 | 0.34 | 1.05 | 0.66 | 0.73 | 0.80 | 0.711 | | 39.3M |
| ours v25_s1 | **1/5** | — falls at 496 / 290 / 384 / 700 | | | | | | | 39.3M |

50-episode, seed 42: PPO reward 51.6, length 993.9, per-step error **0.287**.

**Caveat: PPO had ~5x the training budget** (200M vs 39.3M env steps). Part of the
gap is steps, not algorithm. A budget-matched PPO run has not been done.

### 2.3 What the numbers say

1. **The gap that matters is REPPO-vs-PPO, not ours-vs-reference.** We are 19% behind
   the authors' own code; *both* REPPO implementations are 2.3-2.7x behind PPO.
   Perfect reference parity lands at 0.617 — still ~2.3x worse than PPO. Chasing
   parity was never going to fix `error_vel_xy`.
2. **REPPO is much less consistent.** PPO's worst seed is 0.38; REPPO's worst is
   1.17-1.28. REPPO degrades badly on specific command sequences.
3. **Falls are checkpoint-specific, not systematic.** `v25_s1` falls 4/5;
   `v24_normfix` and `v28` survive 5/5, as does the reference. The original
   observation was correct about that one checkpoint.
4. **Every training-time statistic matches the reference** — entropy -5.94 vs -6.07,
   KL 0.099 vs 0.1005, alpha_kl 0.042 vs 0.0398, Q 49.9 vs 50.6, Q_max 57.7 vs 58.0,
   actor grad norm 0.029 vs 0.043, episode length 973 vs 981. Two implementations
   training identically, producing policies that track differently by 19%.

### 2.4 Behavioural decomposition

Per-step commanded-vs-achieved traces (`--dump_traj`, analysed with
`trudi_ref/analyze_traj.py`) decompose the miss into gain / bias / lag:

| | forward gain | lag (steps) | yaw gain |
|---|---|---|---|
| reference REPPO | 0.22 - 0.59 | 4 - 8 | 0.85 - 0.94 |
| ours (v24_normfix) | 0.05 - 0.28 | 26 - 28 | 0.86 - 0.89 |

Both **undershoot** the forward command badly (gain << 1); the reference is simply
less bad. Yaw tracking is equal and good on both. So the deficit is specifically
**forward-velocity responsiveness** — not stability, not turning. At high commanded
speed our gain collapses to ~0.05 (asked for >1.5 m/s, achieves 0.15 m/s).

### 2.5 Leading hypothesis for the REPPO-vs-PPO gap

REPPO's entropy dual **pins sigma at ~0.50 for the entire run** — that is what the
hard target -0.5/dim enforces, and both our runs and the reference converge to
entropy exactly -6.0. PPO's sigma decays to ~0.1.

At deployment both use the mean action. PPO's mean was optimized under near-zero
noise, so training and deployment agree. REPPO's mean is the centre of a wide
distribution whose Q was only ever evaluated under sigma = 0.5 noise — **the
deterministic action is off-distribution**. That predicts exactly what the trace
shows: a smooth, conservative, low-gain, laggy gait rather than a responsive one.

Two levers follow, in cost order:

* **Eval-time `--q_argmax N`** (already implemented, `scripts/eval/unitree_mjlab.py:208`):
  REPPO has an explicit Q, so sample N actions from `pi(.|s)` plus the mode and take
  the `argmax_a Q`. This closes the mode-vs-greedy gap with **no retraining**. PPO
  cannot do this — no action-value function. *Status: running.*
* **`target_entropy` -0.5 -> -1.5**, letting sigma anneal toward PPO's regime.
  Untested direction — the earlier `go2_t2_ent10` sweep moved entropy the *wrong*
  way (sigma up to 0.78, reward down to 38.4).

Secondary: budget-match to PPO's 200M steps before drawing final conclusions.

---

## Part 3 — reproduction

```bash
# train (39.3M steps, ~1h on one GPU)
python scripts/train/unitree_mjlab.py --env_id Unitree-Go2-Flat --num_envs 1024 \
  --config config/mjlab_go2_reppo_v28_refparity.yaml --gpu_ids 0 --seed 1

# per-step tracking error, seeded single-env (THE metric)
python scripts/eval/unitree_mjlab.py --env_id Unitree-Go2-Flat \
  --config config/mjlab_go2_reppo_v28_refparity.yaml --checkpoint <ckpt> \
  --num_envs 1 --episodes 1 --seed 7 --headless --device cuda:0

# behavioural trace
... --dump_traj /tmp/traj.csv
python /home/human/workspaces/trudi_ref/analyze_traj.py /tmp/traj.csv

# the authors' checkpoint under the same protocol
python /home/human/workspaces/trudi_ref/eval_their_ckpt.py --task Unitree-Go2-Flat \
  --ckpt /home/human/workspaces/trudi_ref/trudi_go2_ckpt/reppo_zip_go2_torch_Unitree-Go2-Flat_latest.pt \
  --num-envs 1 --episodes 1 --seed 7
```

Tests: 545 pass (`pytest`), including 4 new regression tests for the parity switches.

## Open items

* `--q_argmax 64` results on v28 / v24_normfix — running.
* v29 (`dual_optim_mode: actor` isolated) and v30 (`num_atoms: 151` isolated) —
  running, for attribution of the four parity levers.
* `target_entropy` sweep — not started.
* Budget-matched PPO (39.3M) — not started.
* `final_observation` forwarding in the mjlab wrapper — wrong in both
  implementations, unfixed.
