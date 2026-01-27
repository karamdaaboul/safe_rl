from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Tuple

import yaml

from safe_rl.env import SafetyGymnasiumVecEnv
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner

# Algorithms that use off-policy training
OFF_POLICY_ALGORITHMS = {"SAC", "TD3"}

# Algorithms that use on-policy training
ON_POLICY_ALGORITHMS = {"PPO", "P3O", "PPOL_PID", "CUP", "Distillation"}


def load_train_cfg(config_path: str) -> Tuple[Dict[str, Any], int, str]:
    """Load training configuration from YAML file.

    Returns:
        Tuple of (train_cfg dict, max_iterations, runner_class_name)
    """
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg["algorithm"]
    policy_cfg = cfg["policy"]
    runner_cfg = cfg.get("runner", {})

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
                "empirical_normalization": runner_cfg.get("empirical_normalization", False),
                "logger": runner_cfg.get("logger", "tensorboard"),
                "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
                "wandb_entity": runner_cfg.get("wandb_entity"),
                "wandb_dir": runner_cfg.get("wandb_dir"),
                # Off-policy specific
                "max_size": runner_cfg.get("max_size", 1_000_000),
                "start_random_steps": runner_cfg.get("start_random_steps", 10000),
                "update_after": runner_cfg.get("update_after", 1000),
                "update_every": runner_cfg.get("update_every", 50),
            },
        }
    else:
        train_cfg = {
            "algorithm": algorithm_cfg,
            "policy": policy_cfg,
            "num_steps_per_env": runner_cfg.get("num_steps_per_env", 24),
            "save_interval": runner_cfg.get("save_interval", 50),
            "empirical_normalization": runner_cfg.get("empirical_normalization", False),
            "logger": runner_cfg.get("logger", "tensorboard"),
            "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
            "wandb_entity": runner_cfg.get("wandb_entity"),
            "wandb_dir": runner_cfg.get("wandb_dir"),
        }
        # Handle symmetry config for on-policy algorithms
        symmetry_cfg = algorithm_cfg.get("symmetry_cfg")
        if symmetry_cfg is not None:
            if not symmetry_cfg.get("data_augmentation_func"):
                algorithm_cfg["symmetry_cfg"] = None

    max_iterations = runner_cfg.get("max_iterations", 1000)
    return train_cfg, max_iterations, runner_class_name


def parse_cost_limits(cost_limits: str | None) -> list[float] | None:
    if cost_limits is None:
        return None
    return [float(value.strip()) for value in cost_limits.split(",") if value.strip()]


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
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--disable_rnd", action="store_true", help="Disable RND even if configured.")

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

    # PPOL-PID specific parameters
    parser.add_argument("--pid_kp", type=float, default=None, help="PID proportional gain.")
    parser.add_argument("--pid_ki", type=float, default=None, help="PID integral gain.")
    parser.add_argument("--pid_kd", type=float, default=None, help="PID derivative gain.")
    parser.add_argument("--lambda_max", type=float, default=None, help="Maximum Lagrangian multiplier.")
    parser.add_argument("--pid_delta_p_ema_alpha", type=float, default=None, help="EMA alpha for P term.")
    parser.add_argument("--pid_delta_d_ema_alpha", type=float, default=None, help="EMA alpha for D term.")
    parser.add_argument("--pid_d_delay", type=int, default=None, help="Delay steps for D term.")

    args = parser.parse_args()

    train_cfg, max_iterations, runner_class_name = load_train_cfg(args.config)
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

    # Apply runner config overrides
    if args.num_steps_per_env is not None:
        if runner_class_name == "OffPolicyRunner":
            train_cfg["runner"]["num_steps_per_env"] = args.num_steps_per_env
        else:
            train_cfg["num_steps_per_env"] = args.num_steps_per_env

    # Apply PPOL-PID specific overrides
    if algorithm_cfg.get("class_name") == "PPOL_PID":
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

    cost_limits = parse_cost_limits(args.cost_limits)

    # Pass cost_limits to algorithm config for Safe RL algorithms
    if algorithm_cfg.get("class_name") in ("SafeSAC", "SafePPO", "PPOL_PID") and cost_limits is not None:
        algorithm_cfg["cost_limits"] = cost_limits
    env = SafetyGymnasiumVecEnv(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
        render_mode=args.render_mode,
        cost_limits=cost_limits,
        seed=args.seed,
    )

    alg_name = algorithm_cfg.get("class_name", "unknown")
    log_dir = os.path.join(args.log_dir, args.env_id, alg_name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Select runner based on algorithm type
    if runner_class_name == "OffPolicyRunner":
        print(f"[INFO] Using OffPolicyRunner for algorithm: {algorithm_cfg.get('class_name')}")
        runner = OffPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)
    else:
        print(f"[INFO] Using OnPolicyRunner for algorithm: {algorithm_cfg.get('class_name')}")
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    runner.learn(max_iterations)
    env.close()


if __name__ == "__main__":
    main()


