from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Tuple

import yaml

from safe_rl.env import SafetyGymnasiumVecEnv
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner

# Algorithms that use off-policy training
OFF_POLICY_ALGORITHMS = {"SAC", "DDPG", "TD3"}

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
    args = parser.parse_args()

    train_cfg, max_iterations, runner_class_name = load_train_cfg(args.config)
    algorithm_cfg = train_cfg["algorithm"]

    # Handle RND config (only for on-policy algorithms)
    if runner_class_name == "OnPolicyRunner":
        rnd_cfg = algorithm_cfg.get("rnd_cfg")
        if args.disable_rnd or (rnd_cfg is not None and rnd_cfg.get("weight", 0.0) == 0.0):
            algorithm_cfg["rnd_cfg"] = None

    if args.max_iterations is not None:
        max_iterations = args.max_iterations

    cost_limits = parse_cost_limits(args.cost_limits)
    env = SafetyGymnasiumVecEnv(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
        render_mode=args.render_mode,
        cost_limits=cost_limits,
        seed=args.seed,
    )

    log_dir = os.path.join(args.log_dir, args.env_id, time.strftime("%Y%m%d_%H%M%S"))
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


