from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import torch
import yaml

from safe_rl.envs import make_env
from safe_rl.runners import OnPolicyRunner


def load_train_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg["algorithm"]
    policy_cfg = cfg["policy"]
    runner_cfg = cfg.get("runner", {})

    symmetry_cfg = algorithm_cfg.get("symmetry_cfg")
    if symmetry_cfg is not None:
        if not symmetry_cfg.get("data_augmentation_func"):
            algorithm_cfg["symmetry_cfg"] = None

    rnd_cfg = algorithm_cfg.get("rnd_cfg")
    if rnd_cfg is not None and rnd_cfg.get("weight", 0.0) == 0.0:
        algorithm_cfg["rnd_cfg"] = None

    return {
        "algorithm": algorithm_cfg,
        "policy": policy_cfg,
        "num_steps_per_env": runner_cfg.get("num_steps_per_env", 24),
        "save_interval": runner_cfg.get("save_interval", 50),
        "empirical_normalization": runner_cfg.get("empirical_normalization", False),
        "logger": runner_cfg.get("logger", "tensorboard"),
        "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
    }


def parse_cost_limits(cost_limits: str | None) -> list[float] | None:
    if cost_limits is None:
        return None
    return [float(value.strip()) for value in cost_limits.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Safe-RL agents on Safety-Gymnasium environments.")
    parser.add_argument("--env_id", type=str, required=True, help="Safety-Gymnasium env id (e.g. SafetyCarGoal1-v0).")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of vectorized environments.")
    parser.add_argument("--config", type=str, default="config/dummy_config.yaml", help="Path to training config.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--cost_limits", type=str, default=None, help="Comma-separated cost limits.")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode (e.g. human, rgb_array).")
    parser.add_argument("--seed", type=str, default=None, help="Environment seed.")
    args = parser.parse_args()

    train_cfg = load_train_cfg(args.config)
    cost_limits = parse_cost_limits(args.cost_limits)
    env = make_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
        render_mode=args.render_mode,
        cost_limits=cost_limits,
        seed=args.seed,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.get_observations()
    obs = obs.to(runner.device)

    ep_rewards = []
    ep_costs = []
    reward_buf = torch.zeros(env.num_envs, device=runner.device)
    cost_buf = torch.zeros(env.num_envs, device=runner.device)

    while len(ep_rewards) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)
        obs = obs.to(runner.device)
        rewards = rewards.to(runner.device)
        dones = dones.to(runner.device)

        costs = infos.get("costs", torch.zeros_like(rewards)).to(runner.device)
        reward_buf += rewards
        cost_buf += costs

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            ep_rewards.extend(reward_buf[done_ids].cpu().tolist())
            ep_costs.extend(cost_buf[done_ids].cpu().tolist())
            reward_buf[done_ids] = 0.0
            cost_buf[done_ids] = 0.0

    mean_reward = sum(ep_rewards[: args.episodes]) / args.episodes
    mean_cost = sum(ep_costs[: args.episodes]) / args.episodes
    print(f"Evaluation over {args.episodes} episodes")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Mean cost: {mean_cost:.3f}")

    env.close()


if __name__ == "__main__":
    main()
