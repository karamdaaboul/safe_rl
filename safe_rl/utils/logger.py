from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict, deque
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

import safe_rl
from .utils import store_code_state


class Logger:
    """Dedicated logger for off-policy runners.

    Owns episode-level buffers (rewards, lengths, costs), writer initialisation,
    console output, and external logging. The runner only needs to call
    :meth:`process_env_step` during rollout and :meth:`log` after each iteration.
    """

    writer: SummaryWriter | None

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        runner_cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        num_costs: int = 0,
        device: str = "cpu",
    ) -> None:
        self.log_dir = log_dir
        self.cfg = cfg
        self.runner_cfg = runner_cfg
        self.num_envs = num_envs
        self.num_costs = num_costs
        self.device = device
        self.git_status_repos: list[str] = [safe_rl.__file__]

        # Timing / counters
        self.tot_timesteps = 0
        self.tot_time = 0.0

        # Episode-level buffers
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.ep_infos: list[dict[str, Any]] = []
        self.cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Safe RL cost buffers
        if num_costs > 0:
            self.costbuffers: list[deque[float]] = [deque(maxlen=100) for _ in range(num_costs)]
            self.cur_cost_sum = torch.zeros(num_envs, num_costs, dtype=torch.float, device=device)
        else:
            self.costbuffers = []
            self.cur_cost_sum = None

        # Prepare writer
        self.writer = None
        self.logger_type = "tensorboard"
        self._init_writer(env_cfg)

        # Store code state on init
        self._store_code_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict[str, Any],
        costs: torch.Tensor | None = None,
    ) -> None:
        """Update episode buffers after an environment step."""
        if self.log_dir is None:
            return

        # Collect episode extras
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        elif "log" in infos:
            self.ep_infos.append(infos["log"])

        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        if costs is not None and self.cur_cost_sum is not None:
            self.cur_cost_sum += costs

        # Flush completed episodes
        new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if new_ids.numel() > 0:
            self.rewbuffer.extend(self.cur_reward_sum[new_ids].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids].cpu().numpy().tolist())

            if costs is not None and self.cur_cost_sum is not None:
                for cost_idx in range(self.num_costs):
                    self.costbuffers[cost_idx].extend(
                        self.cur_cost_sum[new_ids, cost_idx].cpu().numpy().tolist()
                    )
                self.cur_cost_sum[new_ids] = 0

            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict[str, Any],
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Write metrics to the external logger and print console output."""
        if self.log_dir is None:
            return

        # -- Timing ----------------------------------------------------------
        iteration_time = collect_time + learn_time
        self.tot_time += iteration_time
        num_steps_per_env = int(self.runner_cfg.get("num_steps_per_env", 1))
        collection_size = num_steps_per_env * self.num_envs
        self.tot_timesteps += collection_size
        fps = int(collection_size / max(iteration_time, 1e-6))

        # -- Episode extras --------------------------------------------------
        extras_string = ""
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if key not in ep_info:
                        continue
                    val = ep_info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val])
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor).item()
                    if self.writer is not None:
                        if "/" in key:
                            self.writer.add_scalar(key, value, it)
                        else:
                            self.writer.add_scalar(f"Episode/{key}", value, it)
                    extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        # -- Scalar logging (convention-based routing) -----------------------
        if self.writer is not None:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                tb_key = self._route_key(key)
                self.writer.add_scalar(tb_key, value, it)

            # -- Performance
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            # -- Episode rewards / lengths
            if len(self.rewbuffer) > 0:
                mean_reward = statistics.mean(self.rewbuffer)
                mean_length = statistics.mean(self.lenbuffer)
                self.writer.add_scalar("Train/episode_reward", mean_reward, it)
                self.writer.add_scalar("Train/episode_length", mean_length, it)
                if self.logger_type != "wandb":
                    self.writer.add_scalar("Train/episode_reward/time", mean_reward, self.tot_time)
                    self.writer.add_scalar("Train/episode_length/time", mean_length, self.tot_time)

            # -- Episode costs
            for cost_idx, costbuffer in enumerate(self.costbuffers):
                if len(costbuffer) > 0:
                    mean_cost = statistics.mean(costbuffer)
                    self.writer.add_scalar(f"Train/episode_cost_{cost_idx}", mean_cost, it)

        # -- Console output --------------------------------------------------
        self._print_console(
            it, start_it, total_it, collect_time, learn_time, fps, loss_dict, extras_string, width, pad
        )

        # -- Clear episode infos for next iteration --------------------------
        self.ep_infos.clear()

    def save_model(self, path: str, it: int) -> None:
        """Forward model checkpoint to external logger."""
        if self.writer is not None and self.logger_type in ("wandb", "neptune"):
            self.writer.save_model(path, it)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route_key(self, key: str) -> str:
        """Map a metric key to a TensorBoard tag by convention.

        Rules:
            - ``*_loss``        -> ``Loss/<prefix>``   (e.g. ``critic_loss`` -> ``Loss/critic``)
            - ``lambda_*``      -> ``SafeRL/<key>``
            - ``alpha``         -> ``SafeRL/alpha``
            - ``noise_std``     -> ``Policy/mean_noise_std``
            - everything else   -> ``Train/<key>``
        """
        if key.endswith("_loss"):
            return f"Loss/{key[:-5]}"
        if key.startswith("lambda_") or key == "alpha":
            return f"SafeRL/{key}"
        if key == "noise_std":
            return "Policy/mean_noise_std"
        return f"Train/{key}"

    def _init_writer(self, env_cfg: dict | object) -> None:
        """Create the TensorBoard / wandb / Neptune writer."""
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.logger_type = self.runner_cfg.get("logger", "tensorboard").lower()

        # Build a flattened config for loggers that expect wandb_project at top level
        logger_cfg = {
            **self.cfg,
            "wandb_project": self.runner_cfg.get("wandb_project", "safe_rl"),
            "wandb_entity": self.runner_cfg.get("wandb_entity"),
            "wandb_dir": self.runner_cfg.get("wandb_dir"),
            "run_name": self.runner_cfg.get("run_name"),
        }

        if self.logger_type == "neptune":
            from safe_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=logger_cfg)
            self.writer.log_config(env_cfg, self.cfg, self.runner_cfg, {})
        elif self.logger_type == "wandb":
            from safe_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=logger_cfg)
            self.writer.log_config(env_cfg, self.cfg, self.runner_cfg, {})
        elif self.logger_type == "tensorboard":
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError(
                f"Logger type '{self.logger_type}' not found. Choose 'neptune', 'wandb', or 'tensorboard'."
            )

    def _store_code_state(self) -> None:
        """Save git diffs and upload them to external loggers."""
        if self.log_dir is None:
            return
        git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
        if self.logger_type in ("wandb", "neptune") and git_file_paths and self.writer is not None:
            for path in git_file_paths:
                if self.writer is not None:
                    self.writer.save_file(path)

    def _fmt(self, v: float) -> str:
        """Format a scalar value for console output."""
        return f"{v:.4e}" if (abs(v) < 1e-2 and v != 0.0) else f"{v:.4f}"

    def _print_console(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        fps: int,
        loss_dict: dict[str, Any],
        extras_string: str,
        width: int,
        pad: int,
    ) -> None:
        """Print formatted training metrics to the console."""
        iteration_time = collect_time + learn_time
        header = f" \033[1m Learning iteration {it}/{total_it} \033[0m "

        # -- Performance header
        log_string = (
            f"""{"#" * width}\n"""
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Total steps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Steps per second:":>{pad}} {fps:.0f}\n"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s\n"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s\n"""
        )

        # -- Losses and scalar metrics (grouped by prefix)
        prefix_groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            prefix = key.split("_")[0] if "_" in key else ""
            prefix_groups[prefix].append((key, value))

        for prefix, items in prefix_groups.items():
            if len(items) == 1 or not prefix:
                for key, value in items:
                    if key.endswith("_loss"):
                        label = f"Mean {key[:-5].replace('_', ' ')} loss:"
                    else:
                        label = f"{key.replace('_', ' ').capitalize()}:"
                    log_string += f"{label:>{pad}} {self._fmt(value)}\n"
            else:
                parts = [f"{key[len(prefix) + 1:]}={self._fmt(v)}" for key, v in items]
                label = f"{prefix.capitalize()}:"
                log_string += f"{label:>{pad}} {'  '.join(parts)}\n"

        # -- Episode rewards / lengths
        if len(self.rewbuffer) > 0:
            log_string += (
                f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
            )

        # -- Episode costs
        for cost_idx, costbuffer in enumerate(self.costbuffers):
            if len(costbuffer) > 0:
                mean_cost = statistics.mean(costbuffer)
                log_string += f"""{f"Episode cost {cost_idx} (mean):":>{pad}} {mean_cost:.4f}\n"""

        # -- Episode extras
        log_string += extras_string

        # -- Footer
        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / max(done_it, 1) * remaining_it
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
        )
        print(log_string)
