# Hidden-goal cMAML — goal-blind meta-learning on Safety-Gymnasium

Status as of 2026-07-09. Documents the design that makes the cMAML runs *true*
meta-learning (goal position absent from the observation) and the enforcement +
evaluation stack added around it. See also `safe_rl/runners/meta_on_policy_runner.py`
(class docstring, thesis sec. 7.4/7.5) and [[rcppo-reachability-math-and-results]]
for the constrained-RL background.

---

## 1. Why goal-lidar removal = true meta-learning

A standard `SafetyXGoal` task exposes the goal direction through the 16-bin
`goal_lidar` pseudo-lidar. With that in the observation, "tasks" (goal layouts) are
not latent: a single reactive policy solves every layout zero-shot, and any
"meta-learning" over layouts collapses to goal-conditioned RL. Removing the 16
goal-lidar dims makes the goal a latent variable inferable *only through the
distance-shaped reward* — the defining property of the meta-RL setting. Adaptation
(inner-loop gradient steps on per-task experience) then actually matters.

Goal lidar is the **only** goal-position channel in the default observation:
`Goal(is_lidar_observed=True, is_comp_observed=False)` in
`safety-gymnasium/safety_gymnasium/assets/geoms/goal.py`. The agent's proprioceptive
sensors (accelerometer, velocimeter, gyro, magnetometer) carry no goal information.

**Why a wrapper, not a safety-gymnasium config kwarg:** the `GoalLevelN` task classes
re-add their own `Goal(...)` in `__init__` *after* the config dict is parsed, clobbering
any config-injected `Goal(is_lidar_observed=False)`. So `HiddenGoalWrapper`
(`safe_rl/envs/hidden_goal_wrapper.py`) sets `task.goal.is_lidar_observed = False` and
calls `task.build_observation_space()` post-construction — the dims are physically
removed from the observation space, not masked/zeroed.

## 2. Seed → task mapping

A "task" is one fixed hidden-goal layout, fully determined by the reset seed
(safety-gymnasium re-seeds its per-env `RandomState` on `reset(seed=...)`; the same
seed reproduces the same layout+goal).

- `MetaOnPolicyRunner._sample_tasks()` draws seeds with replacement from
  `meta.seed_pool`; per task the outer loop calls `env.set_task(seed)` + `env.reset()`.
- `SafetyGymnasiumVecEnv.reset` **tiles** a single task seed to all sub-envs (a bare
  int would become `seed+i` = N different layouts).
- `HiddenGoalWrapper(fix_task=True)` remembers the task seed and reuses it on seedless
  auto-resets, so the goal stays fixed across episodes *within* a task — without this,
  each auto-reset would draw a new goal (a memoryless POMDP, nothing to adapt to).

## 3. Enforcement stack (added 2026-07-09)

Four layers guarantee meta runs are goal-blind; each fails loudly, never silently:

1. **Config default** — `config/safety_gymnasium_cmaml_p3o.yaml` has an `env:` block
   with `hidden_goal: true`; `train_safety_gymnasium.py` merges it with the CLI
   (`--no_hidden_goal` wins, for ablations).
2. **Physical check** — `SafetyGymnasiumVecEnv` (with `hidden_goal=True`) verifies via
   `env.call("obs_space_keys")` that `goal_lidar` is absent from every sub-env's
   observation space (defense against wrapper regressions).
3. **Runner guard** — `MetaOnPolicyRunner._validate_hidden_goal` raises unless the env
   is goal-blind or the config sets `meta: allow_goal_obs: true` (the explicit
   goal-conditioned-ablation escape hatch).
4. **Config test** — `tests/test_config_resolution.py::test_meta_configs_are_goal_blind`
   checks every `MetaOnPolicyRunner` YAML for `env.hidden_goal` (or `allow_goal_obs`)
   plus non-empty, disjoint `held_out_seeds`.

## 4. Parameter partition (thesis sec. 7.4)

The cost function is task-independent while the reward is not, so the three parameter
groups are treated differently:

- **actor** — meta-learned via Reptile interpolation (`meta_lr`; ANIL-style
  slow-trunk/fast-head via `meta_lr_head`);
- **cost critic** — one *global* critic trained on the meta-policy's pooled first
  rollouts across all tasks, used as the per-task starting point;
- **reward critic** — not meta-learned; re-initialized per task.

Optional eta meta-safety dual (sec. 7.5): `eta_adaptive` dual-ascends eta on the
meta-policy's own cost; `eta_penalized` applies an eta-weighted cost-reducing step to
the meta-actor (deadband-gated) so the *initialization* is safe, not just fast-adapting.

## 5. Checkpoints and the `load()` re-sync fix

`learn()` restores the meta parameters into the live policy before saving, so
checkpoints hold the meta weights. But the runner's meta snapshots
(`meta_actor_params` / `meta_cost_params`) were taken at construction and never
refreshed — loading a checkpoint used to clobber the loaded weights with the stale
random init on the next `_load_meta_into_policy()`. The `load()` override (2026-07-09)
re-syncs both snapshot groups from the loaded policy. Consequences:

- `--resume_checkpoint` on a meta run now actually continues from the checkpoint;
- eval-time adaptation starts from the trained meta-policy.

`value_init_params` (reward critic) deliberately stays the fresh random init — matching
the per-task re-init used during training.

## 6. Evaluation protocol — held-out task adaptation

`scripts/eval/eval_cmaml_adaptation.py`: for each seed in `meta.held_out_seeds`
(disjoint from `seed_pool`, enforced at runner construction *and* in the eval script):

1. **Zero-shot**: load meta-policy, `set_task(seed)`, deterministic rollout
   (`act_inference`), N episodes → return / cost / success (goal reached) / ep-len.
2. **Adapted**: reload meta-policy, K inner-loop P3O steps via the training-time
   `_collect_and_adapt_task` (fresh reward critic, reset optimizer/kappa state), then
   the same rollout.

Headline: `mean(adapted_return − zero_shot_return)` over held-out tasks, with adapted
cost vs `meta.cost_limit`. Success counting follows [[safety-goal-eval-gotchas]] logic:
count goals reached, not reward.

## 7. Commands

```bash
# Train (hidden_goal comes from the YAML env block; guard refuses goal-observing envs)
python scripts/train/train_safety_gymnasium.py --env_id SafetyPointGoal1-v0 --num_envs 8 \
    --config config/safety_gymnasium_cmaml_p3o.yaml

# Goal-conditioned ablation (must ALSO set meta.allow_goal_obs: true in the config)
python scripts/train/train_safety_gymnasium.py ... --no_hidden_goal

# Held-out adaptation eval
python scripts/eval/eval_cmaml_adaptation.py --env_id SafetyPointGoal1-v0 --num_envs 8 \
    --config config/safety_gymnasium_cmaml_p3o.yaml --checkpoint <model_N.pt> \
    --adapt_steps 3 --eval_episodes 10 --out_json results.json

# Tests
pytest tests/test_hidden_goal_wrapper.py tests/test_meta_on_policy_runner.py tests/test_config_resolution.py
```
