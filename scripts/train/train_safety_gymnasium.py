from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Tuple

import yaml

from safe_rl.envs import make_env
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner
from safe_rl.utils.seeding import seed_everything

# Algorithms that use off-policy training
OFF_POLICY_ALGORITHMS = {"SAC", "TD3", "SafeSAC", "FastSAC", "FastTD3"}

# Algorithms that use on-policy training
ON_POLICY_ALGORITHMS = {"PPO", "P3O", "PPOL_PID", "RCPPO", "CUP", "REPPO", "Distillation"}


def load_train_cfg(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], int, str, str]:
    """Load training configuration from YAML file.

    Returns:
        Tuple of (train_cfg dict, env_cfg dict, max_iterations, runner_class_name, experiment_name)
    """
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # Environment-construction options; merged with CLI flags in main().
    env_cfg = cfg.get("env", {}) or {}

    algorithm_cfg = cfg["algorithm"]
    policy_cfg = cfg["policy"]
    runner_cfg = cfg.get("runner", {})
    experiment_name = runner_cfg.get("experiment_name", "")

    # Determine runner class from config or algorithm type
    algorithm_name = algorithm_cfg.get("class_name", "PPO")
    if algorithm_name in OFF_POLICY_ALGORITHMS:
        default_runner = "OffPolicyRunner"
    else:
        default_runner = "OnPolicyRunner"
    runner_class_name = cfg.get("runner_class_name", default_runner)

    # Build train_cfg based on runner type
    if runner_class_name == "OffPolicyRunner":
        train_cfg = {
            "algorithm": algorithm_cfg,
            "policy": policy_cfg,
            "runner": {
                "num_steps_per_env": runner_cfg.get("num_steps_per_env", 1),
                "save_interval": runner_cfg.get("save_interval", 50),
                "log_interval": runner_cfg.get("log_interval", 1),
                "empirical_normalization": runner_cfg.get("empirical_normalization", False),
                "obs_normalization_clip": runner_cfg.get("obs_normalization_clip", None),
                # Was missing from this whitelist, so `reward_normalization` / its mode were
                # silently dropped from every off-policy config and the runner always fell back
                # to its default (True, "empirical"). Defaults preserved here, so behaviour is
                # unchanged -- the key is simply configurable now instead of being a no-op.
                "reward_normalization": runner_cfg.get("reward_normalization", True),
                "reward_normalization_mode": runner_cfg.get("reward_normalization_mode", "empirical"),
                "logger": runner_cfg.get("logger", "tensorboard"),
                "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
                "wandb_entity": runner_cfg.get("wandb_entity"),
                "wandb_dir": runner_cfg.get("wandb_dir"),
                # Off-policy specific
                "max_size": runner_cfg.get("max_size", 1_000_000),
                "start_random_steps": runner_cfg.get("start_random_steps", 10000),
                "update_after": runner_cfg.get("update_after", 1000),
                "update_every": runner_cfg.get("update_every", 50),
                "n_step": runner_cfg.get("n_step", 1),
                # Off-policy mismatch diagnostic: stamp replay with behavior log-probs.
                # This dict is a WHITELIST -- a key absent here is silently dropped, so every
                # new runner flag must be added both in OffPolicyRunner and here.
                "store_behavior_logprob": runner_cfg.get("store_behavior_logprob", False),
                # Periodic deterministic evaluation (Eval/* in wandb); 0 = off.
                "eval_interval": runner_cfg.get("eval_interval", 0),
                "eval_episodes": runner_cfg.get("eval_episodes", 8),
                "eval_num_envs": runner_cfg.get("eval_num_envs", 2),
            },
        }
    else:
        train_cfg = {
            "algorithm": algorithm_cfg,
            "policy": policy_cfg,
            "num_steps_per_env": runner_cfg.get("num_steps_per_env", 24),
            "save_interval": runner_cfg.get("save_interval", 50),
            "empirical_normalization": runner_cfg.get("empirical_normalization", False),
            "obs_normalization_clip": runner_cfg.get("obs_normalization_clip", None),
            "logger": runner_cfg.get("logger", "tensorboard"),
            "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
            "wandb_entity": runner_cfg.get("wandb_entity"),
            "wandb_dir": runner_cfg.get("wandb_dir"),
            # train_cfg is an explicit WHITELIST — a runner key absent here is
            # silently dropped and the YAML setting has no effect. These three
            # exist so a run can be made wandb-comparable with another codebase:
            #   run_name / wandb_tags -> match the reference's run naming and tags
            #   log_env_steps         -> x-axis in env steps rather than iterations
            "run_name": runner_cfg.get("run_name"),
            "wandb_tags": runner_cfg.get("wandb_tags"),
            "log_env_steps": runner_cfg.get("log_env_steps", False),
            "eval_interval": runner_cfg.get("eval_interval", 0),
            "eval_episodes": runner_cfg.get("eval_episodes", 100),
            #   eval_modes            -> which eval passes to run; ["ode"] drops the
            #                            unused stochastic pass, which on fixed-length
            #                            episodes costs as much as a large slice of the
            #                            training budget
            "eval_modes": runner_cfg.get("eval_modes", ["ode", "sde"]),
        }
        # Handle symmetry config for on-policy algorithms
        symmetry_cfg = algorithm_cfg.get("symmetry_cfg")
        if symmetry_cfg is not None:
            if not symmetry_cfg.get("data_augmentation_func"):
                algorithm_cfg["symmetry_cfg"] = None
        # Carry the CBF config block through (used by OnPolicyRunner and make_env).
        train_cfg["cbf"] = cfg.get("cbf", None)
        # Carry the reachability-safety-filter block through (used by OnPolicyRunner).
        train_cfg["reach_filter"] = cfg.get("reach_filter", None)

    max_iterations = runner_cfg.get("max_iterations", 1000)
    return train_cfg, env_cfg, max_iterations, runner_class_name, experiment_name


def parse_cost_limits(cost_limits: str | None) -> list[float] | None:
    if cost_limits is None:
        return None
    return [float(value.strip()) for value in cost_limits.split(",") if value.strip()]


def resolve_fcsrl_options(args, env_cfg: Dict[str, Any]) -> Tuple[int, bool]:
    """Resolve the FCSRL-style env treatments; CLI wins over the config's env block.

    Both default to off. Each changes what a reported number means, so an enabled
    one is echoed rather than left for the reader to infer from the config.
    See codex/fcsrl-harness-tricks.md.
    """
    action_repeat = args.action_repeat if args.action_repeat is not None else int(env_cfg.get("action_repeat", 1))
    goal_pseudo_terminal = args.goal_pseudo_terminal or bool(env_cfg.get("goal_pseudo_terminal", False))

    if action_repeat > 1:
        print(
            f"[INFO] action_repeat={action_repeat}: one agent step covers {action_repeat} simulator steps "
            f"(episode horizon {1000 // action_repeat} decisions over 1000 simulator steps). "
            "Report the SIMULATOR-step budget, not the decision count."
        )
    if goal_pseudo_terminal:
        print(
            "[INFO] goal_pseudo_terminal: goal respawns cut the value bootstrap. "
            "Training only -- do not evaluate with this on."
        )
    return action_repeat, goal_pseudo_terminal


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Safe-RL agents on Safety-Gymnasium environments.")
    parser.add_argument("--env_id", type=str, required=True, help="Safety-Gymnasium env id (e.g. SafetyCarGoal1-v0).")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of vectorized environments.")
    parser.add_argument("--config", type=str, default="config/dummy_config.yaml", help="Path to training config.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for training.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations from config.")
    parser.add_argument("--cost_limits", type=str, default=None, help="Comma-separated cost limits.")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode (e.g. human, rgb_array).")
    parser.add_argument("--log_dir", type=str, default="logs/safety_gymnasium", help="Root log directory.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the environment AND the global torch/numpy/python RNGs "
        "(network init, action sampling, replay sampling).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic kernels (slower; some CUDA ops have no "
        "deterministic implementation). Used by the regression tests.",
    )
    parser.add_argument("--disable_rnd", action="store_true", help="Disable RND even if configured.")
    parser.add_argument(
        "--empirical_normalization",
        dest="empirical_normalization",
        action="store_true",
        default=None,
        help="Force running mean/std observation normalization on, overriding the config.",
    )
    parser.add_argument(
        "--no_empirical_normalization",
        dest="empirical_normalization",
        action="store_false",
        help="Force observation normalization off, overriding the config.",
    )
    parser.add_argument(
        "--obs_normalization_clip",
        type=float,
        default=None,
        help="Clamp normalized observations to [-C, C] (FCSRL uses 50). Guards against a near-constant channel dividing by a near-zero early std. Also settable as `runner: obs_normalization_clip`.",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Override wandb project name from config.")

    # Sweep-friendly hyperparameters (override config values)
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--num_learning_epochs", type=int, default=None, help="Number of learning epochs.")
    parser.add_argument("--num_mini_batches", type=int, default=None, help="Number of mini batches.")
    parser.add_argument("--clip_param", type=float, default=None, help="PPO clip parameter.")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--lam", type=float, default=None, help="GAE lambda.")
    parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm.")
    parser.add_argument("--num_steps_per_env", type=int, default=None, help="Steps per env per iteration.")

    # HL-Gauss cost-critic discretization (sweep-friendly; override policy.cost_critic_kwargs)
    parser.add_argument("--num_bins", type=int, default=None, help="HL-Gauss cost critic: number of bins.")
    parser.add_argument(
        "--sigma_to_bin_ratio",
        type=float,
        default=None,
        help="HL-Gauss cost critic: sigma as a multiple of bin width.",
    )
    parser.add_argument(
        "--support_transform",
        type=str,
        default=None,
        choices=["linear", "symlog"],
        help="HL-Gauss cost critic: support spacing.",
    )
    parser.add_argument("--cost_v_max", type=float, default=None, help="HL-Gauss cost critic: v_max upper bound.")

    parser.add_argument(
        "--geom_margin",
        action="store_true",
        help="Replace the sparse hazard cost with a signed geometric margin h(s) = d_safe - dist(agent, nearest hazard); pair with RCPPO signed_margin: true.",
    )
    parser.add_argument(
        "--geom_margin_d_safe",
        type=float,
        default=0.4,
        help="Safety distance from hazard centers for --geom_margin (must exceed the hazard radius).",
    )
    parser.add_argument(
        "--geom_margin_min", type=float, default=None, help="Lower clip for the signed margin (default: -d_safe)."
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=None,
        help="Apply each action for N simulator steps (FCSRL uses 4). Reward and cost are summed, so episode totals stay comparable, but the decision horizon shortens N-fold -- report it alongside any result. Also settable as `env: action_repeat` in the config.",
    )
    parser.add_argument(
        "--goal_pseudo_terminal",
        action="store_true",
        help="Treat a goal respawn as a value boundary (info['pseudo_terminated']) so the critic does not bootstrap across the teleporting goal. TRAINING ONLY -- never enable for evaluation. Also settable as `env: goal_pseudo_terminal` in the config.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a model_*.pt to warm-start the policy from (Lagrangian/PID state restarts fresh).",
    )
    parser.add_argument(
        "--resume_reset_std",
        type=float,
        default=None,
        help="With --resume_checkpoint: re-inflate the actor's action std to this value (converged policies have collapsed std and cannot explore toward the constraint).",
    )

    # Vision observations (Safety-Gymnasium *Vision-v0 envs)
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable image observations (auto-enabled when the env id contains 'Vision').",
    )
    parser.add_argument(
        "--vision_size", type=int, default=64, help="Rendered vision observation size (square, pixels)."
    )
    parser.add_argument(
        "--vision_encoder",
        type=str,
        default="resnet18",
        choices=["resnet18", "dinov2_vits14", "none"],
        help="Frozen pretrained encoder for image obs; 'none' passes raw images through (end-to-end CNN path).",
    )
    parser.add_argument(
        "--vision_encoder_weights",
        type=str,
        default=None,
        help="Local checkpoint path for the vision encoder (offline clusters).",
    )
    parser.add_argument("--vision_no_amp", action="store_true", help="Disable fp16 autocast for the vision encoder.")
    parser.add_argument(
        "--vision_proprio_keys",
        type=str,
        default=None,
        help="Comma-separated state keys appended to encoder features (default: all non-lidar keys).",
    )
    parser.add_argument(
        "--vision_mp_context",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start method for vision vector-env workers (default spawn: fork deadlocks with MuJoCo EGL rendering).",
    )

    # PPOL-PID specific parameters
    parser.add_argument("--pid_kp", type=float, default=None, help="PID proportional gain.")
    parser.add_argument("--pid_ki", type=float, default=None, help="PID integral gain.")
    parser.add_argument("--pid_kd", type=float, default=None, help="PID derivative gain.")
    parser.add_argument("--lambda_max", type=float, default=None, help="Maximum Lagrangian multiplier.")
    parser.add_argument("--pid_delta_p_ema_alpha", type=float, default=None, help="EMA alpha for P term.")
    parser.add_argument("--pid_delta_d_ema_alpha", type=float, default=None, help="EMA alpha for D term.")
    parser.add_argument("--pid_d_delay", type=int, default=None, help="Delay steps for D term.")

    args = parser.parse_args()

    # --device defaults to "cpu", so omitting it on a GPU box silently trains on CPU. That is
    # not a small penalty: PPOL-PID measured 16.7s/iter on cuda:0 vs 187s/iter on CPU, and a
    # 500-iteration run burned 12.5h before anyone noticed the GPUs were idle at 0%.
    if str(args.device).startswith("cpu"):
        import torch as _torch

        if _torch.cuda.is_available():
            print(
                f"\n[WARNING] --device is '{args.device}' but CUDA is available "
                f"({_torch.cuda.device_count()} device(s)). Training will run on CPU and be "
                "much slower. Pass --device cuda:0 if that is not intended.\n"
            )

    # Seed BEFORE anything constructs a module or samples: policy init, action sampling and
    # replay sampling all draw from the global torch generator. Previously --seed reached only
    # make_env, so runs were not reproducible (see safe_rl/utils/seeding.py).
    if args.seed is not None:
        seed_everything(args.seed, deterministic=args.deterministic)
        print(
            f"[INFO] Seeded torch/numpy/python with {args.seed}"
            f"{' (deterministic kernels)' if args.deterministic else ''}"
        )

    train_cfg, env_cfg, max_iterations, runner_class_name, experiment_name = load_train_cfg(args.config)
    algorithm_cfg = train_cfg["algorithm"]

    # Apply CLI overrides to algorithm config
    if args.learning_rate is not None:
        algorithm_cfg["learning_rate"] = args.learning_rate
    if args.num_learning_epochs is not None:
        algorithm_cfg["num_learning_epochs"] = args.num_learning_epochs
    if args.num_mini_batches is not None:
        algorithm_cfg["num_mini_batches"] = args.num_mini_batches
    if args.clip_param is not None:
        algorithm_cfg["clip_param"] = args.clip_param
    if args.gamma is not None:
        algorithm_cfg["gamma"] = args.gamma
    if args.lam is not None:
        algorithm_cfg["lam"] = args.lam
    if args.entropy_coef is not None:
        algorithm_cfg["entropy_coef"] = args.entropy_coef
    if args.max_grad_norm is not None:
        algorithm_cfg["max_grad_norm"] = args.max_grad_norm

    # Apply HL-Gauss cost-critic overrides (sweep-friendly)
    cost_critic_kwargs = train_cfg.get("policy", {}).get("cost_critic_kwargs")
    if cost_critic_kwargs is not None:
        if args.num_bins is not None:
            cost_critic_kwargs["num_bins"] = args.num_bins
        if args.sigma_to_bin_ratio is not None:
            # `sigma` and `sigma_to_bin_ratio` are mutually exclusive; clear the scalar.
            cost_critic_kwargs["sigma"] = None
            cost_critic_kwargs["sigma_to_bin_ratio"] = args.sigma_to_bin_ratio
        if args.support_transform is not None:
            cost_critic_kwargs["support_transform"] = args.support_transform
        if args.cost_v_max is not None:
            cost_critic_kwargs["v_max"] = args.cost_v_max

    # Apply runner config overrides. The two runners nest their settings
    # differently: OffPolicyRunner reads train_cfg["runner"], OnPolicyRunner reads
    # train_cfg directly.
    runner_overrides = train_cfg["runner"] if runner_class_name == "OffPolicyRunner" else train_cfg
    if args.num_steps_per_env is not None:
        runner_overrides["num_steps_per_env"] = args.num_steps_per_env
    if args.empirical_normalization is not None:
        runner_overrides["empirical_normalization"] = args.empirical_normalization
    if args.obs_normalization_clip is not None:
        runner_overrides["obs_normalization_clip"] = args.obs_normalization_clip
    if runner_overrides.get("empirical_normalization"):
        clip = runner_overrides.get("obs_normalization_clip")
        print(f"[INFO] empirical_normalization: running mean/std on observations (clip={clip}).")

    # Apply PPOL-PID specific overrides (RCPPO inherits the PID Lagrangian)
    if algorithm_cfg.get("class_name") in ("PPOL_PID", "RCPPO"):
        # Update PID gains if any are provided
        current_pid = algorithm_cfg.get("lagrangian_pid", [0.1, 0.01, 0.01])
        if args.pid_kp is not None:
            current_pid[0] = args.pid_kp
        if args.pid_ki is not None:
            current_pid[1] = args.pid_ki
        if args.pid_kd is not None:
            current_pid[2] = args.pid_kd
        algorithm_cfg["lagrangian_pid"] = current_pid

        if args.lambda_max is not None:
            algorithm_cfg["lambda_max"] = args.lambda_max
        if args.pid_delta_p_ema_alpha is not None:
            algorithm_cfg["pid_delta_p_ema_alpha"] = args.pid_delta_p_ema_alpha
        if args.pid_delta_d_ema_alpha is not None:
            algorithm_cfg["pid_delta_d_ema_alpha"] = args.pid_delta_d_ema_alpha
        if args.pid_d_delay is not None:
            algorithm_cfg["pid_d_delay"] = args.pid_d_delay

    # Handle RND config (only for on-policy algorithms)
    if runner_class_name == "OnPolicyRunner":
        rnd_cfg = algorithm_cfg.get("rnd_cfg")
        if args.disable_rnd or (rnd_cfg is not None and rnd_cfg.get("weight", 0.0) == 0.0):
            algorithm_cfg["rnd_cfg"] = None

    if args.max_iterations is not None:
        max_iterations = args.max_iterations

    if args.wandb_project is not None:
        if runner_class_name == "OffPolicyRunner":
            train_cfg["runner"]["wandb_project"] = args.wandb_project
        else:
            train_cfg["wandb_project"] = args.wandb_project

    # Resolve cost_limits: CLI takes precedence, then config, then None
    cost_limits = parse_cost_limits(args.cost_limits)
    if cost_limits is None and "cost_limits" in algorithm_cfg:
        # Use cost_limits from config if not provided via CLI
        cost_limits = algorithm_cfg["cost_limits"]

    # Pass cost_limits to algorithm config for Safe RL algorithms
    if (
        algorithm_cfg.get("class_name") in ("SafeSAC", "SafePPO", "PPOL_PID", "RCPPO", "P3O", "CUP")
        and cost_limits is not None
    ):
        algorithm_cfg["cost_limits"] = cost_limits
    # For RCPPO the limit is a feasibility threshold on the reachability value, not a budget.
    if algorithm_cfg.get("class_name") == "RCPPO" and cost_limits is not None:
        print(f"[INFO] RCPPO: cost_limits={cost_limits} acts as the feasibility threshold epsilon on E[V_h].")

    cbf_cfg = train_cfg.get("cbf", None)
    cbf_state = bool(cbf_cfg and cbf_cfg.get("enabled", False))

    action_repeat, goal_pseudo_terminal = resolve_fcsrl_options(args, env_cfg)

    vision = args.vision or "Vision" in args.env_id
    vec_kwargs = {}

    # Arbitrary per-env constructor kwargs from the YAML, namespaced under `env.kwargs`
    # so nothing else in the `env` block changes meaning. Needed for env-specific
    # settings that have no CLI flag — e.g. the ManiSkill benchmark configs' eval twin
    # (`num_eval_envs`, `eval_reconfiguration_freq`).
    extra_env_kwargs = dict(env_cfg.get("kwargs") or {})
    if extra_env_kwargs:
        print(f"[INFO] env kwargs from config: {extra_env_kwargs}")
        vec_kwargs.update(extra_env_kwargs)
    if vision:
        # Must be set before the vector-env subprocess workers spawn so each
        # worker gets a headless EGL rendering context.
        os.environ.setdefault("MUJOCO_GL", "egl")
        # fork + MuJoCo EGL rendering deadlocks in the async workers; spawn (a
        # clean interpreter per worker) is the default.
        vec_kwargs["mp_context"] = args.vision_mp_context

    env = make_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
        render_mode=args.render_mode,
        cost_limits=cost_limits,
        seed=args.seed,
        geom_margin=args.geom_margin,
        geom_margin_d_safe=args.geom_margin_d_safe,
        geom_margin_min=args.geom_margin_min,
        action_repeat=action_repeat,
        goal_pseudo_terminal=goal_pseudo_terminal,
        cost_limit_curriculum=env_cfg.get("cost_limit_curriculum"),
        risk_modes=int(env_cfg.get("risk_modes", 0)),
        cbf_state=cbf_state,
        vision=vision,
        vision_size=args.vision_size,
        **vec_kwargs,
    )

    if vision and args.vision_encoder != "none":
        from safe_rl.envs import VisionFeatureWrapper

        env = VisionFeatureWrapper(
            env,
            encoder=args.vision_encoder,
            encoder_weights=args.vision_encoder_weights,
            device=args.device,
            proprio_keys=args.vision_proprio_keys.split(",") if args.vision_proprio_keys else None,
            use_amp=not args.vision_no_amp,
        )
        print(
            f"[INFO] Vision: frozen {args.vision_encoder} at {args.vision_size}x{args.vision_size} -> "
            f"{env.num_features}-d features + {env.num_proprio}-d proprio (asymmetric critics on full state)."
        )

    alg_name = algorithm_cfg.get("class_name", "unknown")
    # Second-resolution timestamp alone is NOT unique: two runs launched in the same second (two
    # GPUs, one launcher script) landed in the SAME directory and the slower run's checkpoints
    # overwrote the faster one's, destroying an ablation arm. Claim the directory exclusively and
    # fall back to a suffix, so concurrent runs can never share one.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.log_dir, args.env_id, alg_name, stamp)
    log_dir = base
    for suffix in range(1, 100):
        try:
            os.makedirs(log_dir)
            break
        except FileExistsError:
            log_dir = f"{base}_{suffix}"
    else:
        raise RuntimeError(f"could not claim a unique log dir under {base}")

    # Set wandb run name from experiment_name + num_envs (+ cost_limit for single-constraint safe RL).
    # An EXPLICIT `run_name` in the YAML wins: this auto-name used to overwrite it
    # unconditionally, which silently discarded configs that set a specific name to
    # match another codebase's wandb naming (e.g. the TruDi reference's
    # "reppo_torch_PickCube-v1").
    _explicit_run_name = train_cfg.get("run_name") or train_cfg.get("runner", {}).get("run_name")
    if experiment_name and not _explicit_run_name:
        run_name = f"{experiment_name}_{args.num_envs}"
        if cost_limits is not None and len(cost_limits) == 1:
            cl = cost_limits[0]
            cl_str = str(int(cl)) if float(cl).is_integer() else str(cl)
            run_name = f"{run_name}_cl{cl_str}"
        if runner_class_name == "OffPolicyRunner":
            train_cfg["runner"]["run_name"] = run_name
        else:
            train_cfg["run_name"] = run_name

    # Select runner based on algorithm type
    if runner_class_name == "OffPolicyRunner":
        print(f"[INFO] Using OffPolicyRunner for algorithm: {algorithm_cfg.get('class_name')}")
        # Dedicated deterministic-evaluation env (runner `eval_interval > 0`): same observation
        # shaping as the training env, its own seed stream, and a small worker count -- the eval
        # panel needs a handful of episodes, not throughput. Never reuses the training env: that
        # would corrupt open episodes and the replay stream.
        eval_env = None
        if int(train_cfg.get("runner", {}).get("eval_interval", 0)) > 0:
            eval_num_envs = int(train_cfg["runner"].get("eval_num_envs", min(2, args.num_envs)))
            print(
                f"[INFO] Deterministic eval env: {eval_num_envs} envs, every "
                f"{train_cfg['runner']['eval_interval']} iterations"
            )
            eval_env = make_env(
                env_id=args.env_id,
                num_envs=eval_num_envs,
                device=args.device,
                cost_limits=cost_limits,
                seed=args.seed + 10_000,
                geom_margin=args.geom_margin,
                geom_margin_d_safe=args.geom_margin_d_safe,
                geom_margin_min=args.geom_margin_min,
                action_repeat=action_repeat,
                goal_pseudo_terminal=goal_pseudo_terminal,
                risk_modes=int(env_cfg.get("risk_modes", 0)),
                cbf_state=cbf_state,
                vision=vision,
                vision_size=args.vision_size,
                **vec_kwargs,
            )
        runner = OffPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device, eval_env=eval_env)
    else:
        print(f"[INFO] Using OnPolicyRunner for algorithm: {algorithm_cfg.get('class_name')}")
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    if args.resume_checkpoint:
        # Warm-start from a previous run's policy (and optimizer). Lagrangian/PID
        # state is not stored in checkpoints, so lambda restarts from the config's
        # lambda_init — intentional for constraint-rescue fine-tuning.
        print(f"[INFO] Resuming policy from {args.resume_checkpoint}")
        runner.load(args.resume_checkpoint, load_optimizer=True)
        if args.resume_reset_std is not None:
            import torch

            with torch.no_grad():
                runner.alg.policy.actor.std.fill_(args.resume_reset_std)
            print(f"[INFO] Actor std re-inflated to {args.resume_reset_std} for constraint-rescue exploration.")

    runner.learn(max_iterations)
    env.close()


if __name__ == "__main__":
    main()
