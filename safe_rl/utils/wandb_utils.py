from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name from config if provided, otherwise use log_dir basename
        # For OffPolicyRunner, check cfg["runner"]["run_name"]
        # For OnPolicyRunner, check cfg["run_name"]
        run_name = None
        if "runner" in cfg and "run_name" in cfg["runner"] and cfg["runner"]["run_name"]:
            run_name = cfg["runner"]["run_name"]
        elif "run_name" in cfg and cfg["run_name"]:
            run_name = cfg["run_name"]

        # Fallback to timestamp-based name from log_dir
        if not run_name:
            run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        # Try to get entity from config first, then fall back to environment variable
        # If wandb_entity is explicitly set in config (even to null), use that value
        # Only fall back to env var if wandb_entity key doesn't exist in config
        if "wandb_entity" in cfg:
            entity = cfg["wandb_entity"]  # Can be None if set to null in yaml
        else:
            entity = os.environ.get("WANDB_USERNAME")

        # Get custom wandb directory if specified (default: None uses ./wandb)
        wandb_dir = cfg.get("wandb_dir", None)
        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)

        # Initialize wandb — disable git integration to avoid broken symlinks
        # for .diff files when running inside a container (offline mode).
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=wandb_dir,
            settings=wandb.Settings(disable_git=True),
        )

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        if isinstance(env_cfg, dict):
            wandb.config.update({"env_cfg": env_cfg})
            return
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        # Skip wandb.save for .pt files — they create symlinks that break
        # when the container's bind-mounted log directory is removed.
        if not model_path.endswith(".pt"):
            wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        if os.path.islink(path) and not os.path.exists(path):
            return  # skip broken symlinks
        if path.endswith(".diff"):
            return  # skip diff files — they create broken symlinks when synced outside the container
        wandb.save(path, base_path=os.path.dirname(path))

    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
