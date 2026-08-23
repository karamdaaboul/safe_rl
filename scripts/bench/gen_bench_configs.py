#!/usr/bin/env python3
"""Generate the paper-comparison benchmark configs.

Why generated rather than hand-written:

* The per-task values that actually matter here are *derived*, not chosen. ManiSkill's
  ``gamma`` is ``1 - 10/max_episode_steps`` (the reference derives it the same way,
  ``trudi/src/torchrl/envs.py:95-97``), so it differs per task — 0.80 / 0.875 / 0.90
  across the eight paper tasks. Getting one of them wrong is not a small error: using
  0.99 on a 50-step task was one of three bugs that made our first PickCube runs
  flatline at success ~0.005. Deriving it from the installed registry removes the
  entire class of mistake.
* Sixteen near-identical YAMLs differing in three fields is exactly where copy-paste
  drift happens.
* The files are still committed, so ``tests/test_config_resolution.py`` (which rglobs
  ``config/**``) covers them, and a run is reproducible from the repo alone.

``--check`` re-renders in memory and diffs against disk, so a hand-edit of a generated
file fails the test suite. It reads the derived values from ``_derived.json`` and so
runs in a bare CPU venv without ManiSkill or MuJoCo Playground installed.

The two suites live in different venvs, so run this once per suite:

    /home/human/venvs/maniskill/bin/python scripts/bench/gen_bench_configs.py --suite maniskill --write
    PYTHONPATH=$PWD /home/human/workspaces/reppo_original/.venv/bin/python \
        scripts/bench/gen_bench_configs.py --suite dmc --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "config" / "bench" / "paper"

# ---------------------------------------------------------------------------
# Task lists
# ---------------------------------------------------------------------------

# All eight ManiSkill tasks the paper reports (results/maniskill/*.csv).
# PickCube-v1 is deliberately ABSENT: it has no CSV, so it is a smoke vehicle only
# and must never appear in the comparison table.
MANISKILL_TASKS = [
    "LiftPegUpright-v1",
    "PegInsertionSide-v1",
    "PickSingleYCB-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "RollBall-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "UnitreeG1TransportBox-v1",
]

# A pre-registered 8-task subset of the paper's 23 DMC tasks. The full grid is
# ~7 GPU-days at 3 seeds and roughly half of it is saturated near the ceiling, so the
# subset is chosen for information per GPU-hour across their reported difficulty range.
# The rationale per task is recorded here because it is the pre-registration.
DMC_TASKS = [
    "CheetahRun",             # 924 +/- 52   saturated, low variance -> pipeline sanity
    "WalkerRun",              # 898 +/- 22   tightest sd in the suite -> highest-power test
    "WalkerWalk",             # 979 +/- 1    easy locomotion control
    "HumanoidRun",            # 693 +/- 91   high-dim; where critic/dual problems surface
    "HopperHop",              # 179 +/- 157  their own method barely solves it - do we
                              #              reproduce the FAILURE too?
    "AcrobotSwingupSparse",   # 11.5 +/- 13  sparse reward -> entropy/exploration duals
    "CartpoleSwingupSparse",  # 749 +/- 165  second sparse task, cheap sim
    "FingerTurnHard",         # 925 +/- 41   manipulation-flavoured, mid difficulty
]

# ---------------------------------------------------------------------------
# Derivation — read from the installed simulators, never hardcoded
# ---------------------------------------------------------------------------


def derive_maniskill() -> dict[str, dict]:
    import mani_skill.envs  # noqa: F401  (registers the tasks)
    from mani_skill.utils.registration import REGISTERED_ENVS

    out = {}
    for task in MANISKILL_TASKS:
        horizon = int(REGISTERED_ENVS[task].max_episode_steps)
        out[task] = {
            "max_episode_steps": horizon,
            # The reference derives the discount from the horizon rather than using a
            # locomotion-style 0.99 (trudi/src/torchrl/envs.py:97).
            "gamma": round(1.0 - 10.0 / horizon, 6),
            "source": "mani_skill.utils.registration.REGISTERED_ENVS[task].max_episode_steps",
        }
    return out


def derive_dmc() -> dict[str, dict]:
    from mujoco_playground import registry

    out = {}
    for task in DMC_TASKS:
        cfg = registry.get_default_config(task)
        out[task] = {
            "max_episode_steps": int(cfg.episode_length),
            # env/mjx_dmc.yaml keeps the global gamma; only ManiSkill derives it.
            "gamma": 0.99,
            "source": "mujoco_playground.registry.get_default_config(task).episode_length",
        }
    return out


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Shared preamble explaining what the parity block is for, so the reason survives in
# the generated file rather than living only here.
_PARITY_NOTE = """\
# Parity block: every knob our REPPO adds on top of the reference is pinned to its
# reference-equivalent value EXPLICITLY rather than left to our defaults, so a future
# change of default cannot silently invalidate the comparison. The list of additions is
# codex/reppo-ours-vs-reference-feature-list.md.
#   dual_optim_mode: actor          duals are nn.Parameters inside the reference Actor,
#                                   so they ride the actor optimizer and its grad clip
#   critic_loss_denominator: batch  reference uses (mask * ce).mean()
#   force_last_step_truncated: true reference always sets truncated[-1] = 1.0
#   target_entropy_final: null      reference has a fixed target, no anneal
#   reward_normalization: false     reference uses a fixed per-env constant
#   action_scale 1.0 / squash tanh  reference tanh is hard-capped at +/-1
#   optimizer adam, wd 0            reference is plain Adam with library defaults
#   critic_type: reference          the encoder/head-split critic with the learnable prior
"""

MANISKILL_TEMPLATE = """\
# AUTO-GENERATED by scripts/bench/gen_bench_configs.py -- do not hand-edit.
# `--check` (tests/test_bench_configs.py) fails if this file drifts from the generator.
#
# Paper-comparison arm: Gaussian REPPO on ManiSkill {task}.
# Compared against the authors' own result curve at
#   /home/human/workspaces/reppo_original/results/maniskill/{task}.csv
# which was produced by their TORCH trainer -- the same code family this port mirrors --
# so this is a PARITY claim: a gap here is our bug.
#
# derived: max_episode_steps={horizon} -> gamma={gamma}
#   via {source}
{parity_note}
runner_class_name: OnPolicyRunner
seed: 1

env:
  kwargs:
    # The reference evaluates on a SEPARATE env built with reconfiguration_freq=1, i.e.
    # assets and layout are resampled every reset, while the training env reuses one
    # scene. Evaluating on the training env would skip exactly the generalization
    # PickSingleYCB-v1 and PokeCube-v1 exist to measure and inflate our success rate
    # against a reference that reconfigured. 256 envs x 4 rounds = 1024 episodes, which
    # matches the 1/1024 quantization visible in every final row of their CSVs.
    num_eval_envs: 256
    eval_reconfiguration_freq: 1

algorithm:
  class_name: REPPO
  gamma: {gamma}
  lam: 0.95
  desired_kl: 0.1
  kl_clip_mode: clipped
  init_alpha_temp: 0.01          # reference ent_start
  init_alpha_kl: 0.01            # reference kl_start
  target_entropy: -0.5           # reference ent_target_mult
  learning_rate: 0.0003
  alpha_lr: 0.0003
  critic_learning_rate: null     # reference uses one shared lr
  max_grad_norm: 0.5
  aux_loss_mult: 1.0
  reward_scale: 1.0
  num_learning_epochs: 4
  # 128, not our PickCube config's 64. The reference global is 128 and the ManiSkill
  # override does not change it; 64 would double the minibatch to 2048.
  num_mini_batches: 128
  optimizer_class: adam
  weight_decay: 0.0
  betas: [0.9, 0.999]
  dual_optim_mode: actor
  critic_loss_denominator: batch
  force_last_step_truncated: true
  target_entropy_final: null
  reward_normalization: false

policy:
  class_name: REPPOActorCritic
  actor_type: stochastic
  critic_type: reference
  actor_obs_normalization: true
  critic_obs_normalization: true
  squash: tanh
  action_scale: 1.0
  min_std: 0.0
  actor_kwargs:
    network_type: mlp
    hidden_dims: [512, 512]
    activation: swish
    use_layer_norm: true
    norm_type: rmsnorm
    log_std_squash: clamp
    log_std_min: -10.0
    log_std_max: 2.0
    init_noise_std: 1.0
  critic_kwargs:
    num_atoms: 151
    v_min: -15.0                 # env/maniskill.yaml
    v_max: 15.0
    hidden_dim: 512
    encoder_layers: 2
    head_layers: 2
    pred_layers: 2
    activation: swish
    norm: rmsnorm
    prior_scale: 40.9

runner:
  num_steps_per_env: 128
  max_iterations: 381            # 1024 envs x 128 steps x 381 = 49,938,432 steps (~50M)
  eval_interval: 19              # 21 points over 0..380, last exactly at the final iter
  eval_episodes: 1024
  eval_modes: [ode]              # the paper's headline is the DETERMINISTIC eval only
  log_env_steps: true            # x-axis in env steps, to overlay on their curves
  empirical_normalization: false
  save_interval: 100
  experiment_name: bench_paper_maniskill_{slug}
  logger: tensorboard
"""

DMC_TEMPLATE = """\
# AUTO-GENERATED by scripts/bench/gen_bench_configs.py -- do not hand-edit.
# `--check` (tests/test_bench_configs.py) fails if this file drifts from the generator.
#
# Paper-comparison arm: Gaussian REPPO on MuJoCo Playground DMC {task}.
# Compared against the authors' own result curve at
#   /home/human/workspaces/reppo_original/results/mujoco_playground/{task}.csv
#
# IMPORTANT -- this is a BREADTH claim, not a parity claim, because those CSVs were
# produced by the authors' JAX trainer (reppo_original/src/jaxrl/reppo.py) while this is a
# torch port. The HYPERPARAMETERS below do match their effective DMC config
# (reppo.yaml + env/mjx_dmc.yaml + experiment_overrides/mjx_dmc_large_data.yaml), verified
# field by field, including two that are easy to get wrong:
#   * their `num_actor_layers: 3` builds in->512, 512->512, 512->out (FCNN in
#     src/networks/torch_models.py), i.e. TWO hidden layers -- what `[512, 512]` means here;
#   * their `clipped` actor loss is
#     where(kl < bound, log_prob*temp - value, kl*lagrangian*reduce_kl) with reduce_kl
#     defaulting to 1, which is exactly our torch.where(kl < desired_kl, primary,
#     alpha_kl*kl). `reduce_kl` is a multiplier, not a feature we lack.
# Two differences that DID exist here were closed on 2026-08-10 and are now pinned below:
#   * `aux_reward_pred: true` -- the reference aux loss stacks a next-embedding MSE *and*
#     a reward-prediction MSE, averaged over all D+1 slots and masked by (1 - done)
#     (jaxrl/reppo.py:493-501). The critic's prediction head emits hidden_dim + 1.
#   * `prior_scale: 40.0` -- the JAX critic adds `zero_dist * 40.0`
#     (networks/jax_models.py:298). The 40.9 we used is the TORCH reference's constant
#     (torch_models.py:282), and the DMC curves come from the JAX trainer.
#
# derived: max_episode_steps={horizon}, gamma={gamma} (env/mjx_dmc.yaml keeps the global)
#   via {source}
# Batch shape is the pre-registered `mjx_dmc_large_data` variant (num_steps 128,
# num_mini_batches 64, num_epochs 8). Their published curves appear to POOL the
# small/medium/large variants, which we cannot reproduce; the choice is declared here
# rather than selected after seeing results.
{parity_note}
runner_class_name: OnPolicyRunner
seed: 1

env:
  kwargs:
    # Separate eval env so the periodic deterministic eval does not reset and disturb
    # the training env 21 times per run. 128 envs is ample for a deterministic policy,
    # whose only source of return variance is the initial state.
    num_eval_envs: 128

algorithm:
  class_name: REPPO
  gamma: {gamma}
  lam: 0.95
  desired_kl: 0.1
  kl_clip_mode: clipped
  init_alpha_temp: 0.01
  init_alpha_kl: 0.01
  target_entropy: -0.5
  learning_rate: 0.0003
  alpha_lr: 0.0003
  critic_learning_rate: null
  max_grad_norm: 0.5
  aux_loss_mult: 1.0
  aux_reward_pred: true          # jaxrl/reppo.py:493-501 (embedding + reward, mean over D+1)
  reward_scale: 1.0              # env/mjx_dmc.yaml reward_scaling
  num_learning_epochs: 8         # mjx_dmc_large_data
  num_mini_batches: 64           # mjx_dmc_large_data
  optimizer_class: adam
  weight_decay: 0.0
  betas: [0.9, 0.999]
  dual_optim_mode: actor
  critic_loss_denominator: batch
  force_last_step_truncated: true
  target_entropy_final: null
  reward_normalization: false

policy:
  class_name: REPPOActorCritic
  actor_type: stochastic
  critic_type: reference
  actor_obs_normalization: true
  critic_obs_normalization: true
  squash: tanh
  action_scale: 1.0
  min_std: 0.0
  actor_kwargs:
    network_type: mlp
    hidden_dims: [512, 512]
    activation: swish
    use_layer_norm: true
    norm_type: rmsnorm
    log_std_squash: clamp
    log_std_min: -10.0
    log_std_max: 2.0
    init_noise_std: 1.0
  critic_kwargs:
    num_atoms: 151
    v_min: 0.0                   # env/mjx_dmc.yaml
    v_max: 150.0
    hidden_dim: 512
    encoder_layers: 2
    head_layers: 2
    pred_layers: 2
    activation: swish
    norm: rmsnorm
    prior_scale: 40.0            # jax_models.py:298 (torch reference uses 40.9)
    predict_reward: true         # prediction head emits hidden_dim + 1

runner:
  num_steps_per_env: 128
  max_iterations: 381            # 1024 envs x 128 steps x 381 = 49,938,432 steps (~50M)
  eval_interval: 19
  eval_episodes: 128
  eval_modes: [ode]
  log_env_steps: true
  empirical_normalization: false
  save_interval: 100
  experiment_name: bench_paper_dmc_{slug}
  logger: tensorboard
"""

SUITES = {
    "maniskill": (MANISKILL_TASKS, derive_maniskill, MANISKILL_TEMPLATE),
    "dmc": (DMC_TASKS, derive_dmc, DMC_TEMPLATE),
}


def render(suite: str, task: str, derived: dict) -> str:
    _, _, template = SUITES[suite]
    d = derived[task]
    return template.format(
        task=task,
        slug=task.replace("-", "_").replace(".", "_"),
        horizon=d["max_episode_steps"],
        gamma=d["gamma"],
        source=d["source"],
        parity_note=_PARITY_NOTE,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="render configs to disk")
    mode.add_argument("--check", action="store_true", help="fail if disk differs from a fresh render")
    args = ap.parse_args()

    tasks, derive, _ = SUITES[args.suite]
    out_dir = OUT_ROOT / args.suite
    derived_path = out_dir / "_derived.json"

    if args.write:
        derived = derive()
        out_dir.mkdir(parents=True, exist_ok=True)
        derived_path.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n")
        for task in tasks:
            (out_dir / f"{task}.yaml").write_text(render(args.suite, task, derived))
        print(f"wrote {len(tasks)} configs + _derived.json to {out_dir}")
        return 0

    # --check: read the derived values from disk so this runs without the simulators.
    if not derived_path.exists():
        print(f"missing {derived_path}; run --write with the {args.suite} interpreter", file=sys.stderr)
        return 1
    derived = json.loads(derived_path.read_text())
    drift = []
    for task in tasks:
        path = out_dir / f"{task}.yaml"
        if not path.exists():
            drift.append(f"{path} is missing")
        elif path.read_text() != render(args.suite, task, derived):
            drift.append(f"{path} differs from a fresh render")
    if drift:
        print("\n".join(drift), file=sys.stderr)
        return 1
    print(f"{args.suite}: {len(tasks)} configs match the generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
