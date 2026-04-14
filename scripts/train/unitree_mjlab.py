from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
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

# Import mjlab before torch: mjlab transitively loads libicui18n.so.78, which
# needs CXXABI_1.3.15 from the conda env's newer libstdc++. If torch imports
# first, it pins the host's older libstdc++ and later mjlab imports fail.
import mjlab  # noqa: E402
import mjlab.tasks  # noqa: E402,F401
import tyro  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks  # noqa: E402
from mjlab.tasks.tracking.mdp import MotionCommandCfg  # noqa: E402
from mjlab.utils.gpu import select_gpus  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import src.tasks  # noqa: E402,F401

import torch  # noqa: E402
import yaml  # noqa: E402

from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OnPolicyRunner  # noqa: E402


class _YamlDumper(yaml.SafeDumper):
    pass


def _represent_fallback(dumper: yaml.Dumper, data: Any) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", repr(data))


_YamlDumper.add_representer(None, _represent_fallback)


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, Dumper=_YamlDumper, sort_keys=False)


def convert_mjlab_ppo_cfg(agent_cfg: Any, logger: str, wandb_project: str, wandb_entity: str | None) -> dict[str, Any]:
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
        "logger": logger,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "run_name": getattr(agent_cfg, "run_name", ""),
    }


def apply_overrides(train_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    algorithm_cfg = train_cfg["algorithm"]

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
    if args.num_steps_per_env is not None:
        train_cfg["num_steps_per_env"] = args.num_steps_per_env
    if args.run_name is not None:
        train_cfg["run_name"] = args.run_name

    return train_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Unitree mjlab tasks with safe_rl PPO.")
    parser.add_argument("task_id", type=str, help="Registered mjlab task id, e.g. Unitree-Go2-Flat.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of vectorized environments.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for training.")
    parser.add_argument("--gpu_ids", nargs="*", default=["0"], help="GPU ids to use, or 'all'.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations.")
    parser.add_argument("--motion_file", type=str, default=None, help="Required for tracking tasks.")
    parser.add_argument("--video", action="store_true", help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--enable_nan_guard", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/safe_rl", help="Root directory for logs.")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "neptune"])
    parser.add_argument("--wandb_project", type=str, default="safe_rl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from a previous checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to load.")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_learning_epochs", type=int, default=None)
    parser.add_argument("--num_mini_batches", type=int, default=None)
    parser.add_argument("--clip_param", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    return parser


def resolve_gpu_ids(gpu_ids: list[str], device: str) -> list[int] | str | None:
    if device == "cpu":
        return None
    if len(gpu_ids) == 1 and gpu_ids[0] == "all":
        return "all"
    return [int(gpu_id) for gpu_id in gpu_ids]


def run_train(task_id: str, args: argparse.Namespace, log_dir: Path) -> None:
    env_cfg: ManagerBasedRlEnvCfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        rank = 0
        seed = agent_cfg.seed
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = agent_cfg.seed + local_rank

    configure_torch_backends()

    agent_cfg.seed = seed
    env_cfg.seed = seed

    is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    if is_tracking_task:
        if not args.motion_file:
            raise ValueError("Tracking tasks require --motion_file.")
        motion_path = Path(args.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        env_cfg.commands["motion"].motion_file = str(motion_path)

    if args.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True

    if rank == 0:
        print(f"[INFO] Training with safe_rl on task={task_id}, device={device}, seed={seed}")
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if args.video else None)
    if args.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "train",
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    vec_env = make_env(env_id=task_id, env=env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    train_cfg = convert_mjlab_ppo_cfg(agent_cfg, args.logger, args.wandb_project, args.wandb_entity)
    train_cfg = apply_overrides(train_cfg, args)

    if train_cfg["algorithm"]["class_name"] != "PPO":
        raise NotImplementedError("The mjlab safe_rl integration currently supports PPO only.")

    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(log_dir), device=device)
    runner.add_git_repo_to_log(__file__)
    runner.add_git_repo_to_log(src.tasks.__file__)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        runner.load(str(checkpoint_path))

    if rank == 0:
        _dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
        _dump_yaml(log_dir / "params" / "agent.yaml", train_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


def launch_training(task_id: str, args: argparse.Namespace) -> None:
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name

    log_root_path = Path(args.log_dir) / agent_cfg.experiment_name
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir_name += f"_{agent_cfg.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(resolve_gpu_ids(args.gpu_ids, args.device))
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task_id, args, log_dir)
    else:
        import torchrunx

        logging.basicConfig(level=logging.INFO)
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.task_id not in list_tasks():
        raise ValueError(f"Unknown task_id '{args.task_id}'. Run with one of: {', '.join(list_tasks())}")

    launch_training(args.task_id, args)


if __name__ == "__main__":
    main()
