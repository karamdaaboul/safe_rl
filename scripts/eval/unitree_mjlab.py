from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _add_unitree_repo_to_path() -> None:
    explicit_repo = os.environ.get("UNITREE_RL_MJLAB_PATH")
    candidates = [explicit_repo] if explicit_repo else []
    candidates.extend(
        [
            "/opt/unitree_rl_mjlab",
            str(Path.home() / "workspaces" / "unitree_rl_mjlab"),
            str(Path(__file__).resolve().parents[3] / "unitree_rl_mjlab"),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
            break


_add_unitree_repo_to_path()

import mjlab  # noqa: E402
import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks  # noqa: E402
from mjlab.tasks.tracking.mdp import MotionCommandCfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

import src.tasks  # noqa: E402,F401

import torch  # noqa: E402
import yaml  # noqa: E402

from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner  # noqa: E402


OFF_POLICY_ALGORITHMS = {"SAC", "TD3", "SafeSAC", "FastSAC", "FastTD3"}



def convert_mjlab_ppo_cfg(agent_cfg: Any) -> dict[str, Any]:
    cfg = asdict(agent_cfg)
    actor_cfg = cfg["actor"]
    critic_cfg = cfg["critic"]
    algorithm_cfg = cfg["algorithm"]
    distribution_cfg = actor_cfg.get("distribution_cfg", {})

    return {
        "algorithm": {
            "class_name": "PPO",
            "normalize_advantage_per_mini_batch": algorithm_cfg.get("normalize_advantage_per_mini_batch", False),
            "value_loss_coef": algorithm_cfg["value_loss_coef"],
            "clip_param": algorithm_cfg["clip_param"],
            "use_clipped_value_loss": algorithm_cfg["use_clipped_value_loss"],
            "desired_kl": algorithm_cfg["desired_kl"],
            "entropy_coef": algorithm_cfg["entropy_coef"],
            "gamma": algorithm_cfg["gamma"],
            "lam": algorithm_cfg["lam"],
            "max_grad_norm": algorithm_cfg["max_grad_norm"],
            "learning_rate": algorithm_cfg["learning_rate"],
            "num_learning_epochs": algorithm_cfg["num_learning_epochs"],
            "num_mini_batches": algorithm_cfg["num_mini_batches"],
            "schedule": algorithm_cfg["schedule"],
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_type": "gaussian",
            "critic_type": "standard",
            "actor_obs_normalization": actor_cfg.get("obs_normalization", False),
            "critic_obs_normalization": critic_cfg.get("obs_normalization", False),
            "actor_kwargs": {
                "hidden_dims": list(actor_cfg["hidden_dims"]),
                "activation": actor_cfg["activation"],
                "init_noise_std": distribution_cfg.get("init_std", 1.0),
                "noise_std_type": distribution_cfg.get("std_type", "scalar"),
            },
            "critic_kwargs": {
                "hidden_dims": list(critic_cfg["hidden_dims"]),
                "activation": critic_cfg["activation"],
            },
        },
        "num_steps_per_env": cfg["num_steps_per_env"],
        "save_interval": cfg["save_interval"],
        "empirical_normalization": False,
        "logger": "tensorboard",
        "wandb_project": "safe_rl",
        "wandb_entity": None,
        "run_name": getattr(agent_cfg, "run_name", ""),
    }


def load_train_cfg(task_id: str, checkpoint_path: Path, train_cfg_path: str | None) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    if train_cfg_path is not None:
        candidate_paths.append(Path(train_cfg_path).expanduser().resolve())
    candidate_paths.append(checkpoint_path.parent / "params" / "agent.yaml")

    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file)
            cfg.setdefault("algorithm", {}).setdefault("class_name", "PPO")
            cfg.setdefault("policy", {}).setdefault("class_name", "ActorCritic")
            return cfg

    agent_cfg = load_rl_cfg(task_id)
    return convert_mjlab_ppo_cfg(agent_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate/play safe_rl PPO agents on Unitree mjlab tasks.")
    parser.add_argument("task_id", type=str, help="Registered mjlab task id, e.g. Unitree-G1-Flat.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--train_cfg", type=str, default=None, help="Optional path to saved params/agent.yaml.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of completed episodes to evaluate.")
    parser.add_argument("--motion_file", type=str, default=None, help="Required for tracking tasks.")
    parser.add_argument("--headless", action="store_true", help="Run without any rendering (metrics only).")
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=["auto", "native", "viser"],
        help="Interactive viewer backend (auto picks native if DISPLAY is set).",
    )
    parser.add_argument("--video", action="store_true", help="Record the first evaluation rollout to mp4.")
    parser.add_argument("--video_length", type=int, default=1000, help="Recorded video length in steps.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to store recorded evaluation videos.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--export_onnx", action="store_true", help="Export actor to ONNX alongside the checkpoint then exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.task_id not in list_tasks():
        raise ValueError(f"Unknown task_id '{args.task_id}'. Run with one of: {', '.join(list_tasks())}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_cfg = load_train_cfg(args.task_id, checkpoint_path, args.train_cfg)
    env_cfg: ManagerBasedRlEnvCfg = load_env_cfg(args.task_id)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed

    is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    if is_tracking_task:
        if not args.motion_file:
            raise ValueError("Tracking tasks require --motion_file.")
        motion_path = Path(args.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        env_cfg.commands["motion"].motion_file = str(motion_path)

    configure_torch_backends()

    interactive = not args.headless and not args.video
    env_render_mode = "rgb_array" if args.video else None
    if args.video:
        # Match the upstream play script's quality knobs, but use a larger default
        # automatically so evaluation recordings are easier to inspect.
        env_cfg.viewer.width = 1920
        env_cfg.viewer.height = 1080

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=env_render_mode)
    if args.video:
        default_video_dir = checkpoint_path.parent / "videos" / "eval"
        video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else default_video_dir
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to {video_dir}")

    agent_cfg = load_rl_cfg(args.task_id)
    vec_env = make_env(env_id=args.task_id, env=env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    alg_name = train_cfg.get("algorithm", {}).get("class_name", "PPO")
    if alg_name in OFF_POLICY_ALGORITHMS:
        print(f"[INFO] Using OffPolicyRunner for algorithm: {alg_name}")
        runner = OffPolicyRunner(vec_env, train_cfg, log_dir=None, device=args.device)
    else:
        print(f"[INFO] Using OnPolicyRunner for algorithm: {alg_name}")
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=args.device)
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    if args.export_onnx:
        runner.export_policy_to_onnx(
            path=str(checkpoint_path.parent),
            filename=checkpoint_path.stem + ".onnx",
        )
        vec_env.close()
        return

    if interactive:
        viewer = args.viewer
        if viewer == "auto":
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            viewer = "native" if has_display else "viser"
        if viewer == "native":
            NativeMujocoViewer(vec_env, policy).run()
        else:
            ViserPlayViewer(vec_env, policy).run()
        vec_env.close()
        return

    obs, _ = vec_env.get_observations()
    obs = obs.to(runner.device)

    ep_rewards: list[float] = []
    ep_costs: list[float] = []
    ep_lengths: list[int] = []
    reward_buf = torch.zeros(vec_env.num_envs, device=runner.device)
    cost_buf = torch.zeros(vec_env.num_envs, device=runner.device)
    length_buf = torch.zeros(vec_env.num_envs, dtype=torch.long, device=runner.device)

    while len(ep_rewards) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, infos = vec_env.step(actions)
        obs = obs.to(runner.device)
        rewards = rewards.to(runner.device)
        dones = dones.to(runner.device)

        costs = infos.get("costs", torch.zeros_like(rewards)).to(runner.device)
        reward_buf += rewards
        cost_buf += costs
        length_buf += 1

        runner.alg.policy.reset(dones=dones)

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            continue

        ep_rewards.extend(reward_buf[done_ids].detach().cpu().tolist())
        ep_costs.extend(cost_buf[done_ids].detach().cpu().tolist())
        ep_lengths.extend(length_buf[done_ids].detach().cpu().tolist())
        reward_buf[done_ids] = 0.0
        cost_buf[done_ids] = 0.0
        length_buf[done_ids] = 0

    ep_rewards = ep_rewards[: args.episodes]
    ep_costs = ep_costs[: args.episodes]
    ep_lengths = ep_lengths[: args.episodes]
    print(f"Evaluation over {args.episodes} episodes")
    print(f"Mean reward: {sum(ep_rewards) / len(ep_rewards):.3f}")
    print(f"Mean cost: {sum(ep_costs) / len(ep_costs):.3f}")
    print(f"Mean length: {sum(ep_lengths) / len(ep_lengths):.1f}")

    vec_env.close()


if __name__ == "__main__":
    main()
