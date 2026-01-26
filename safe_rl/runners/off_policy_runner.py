# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import safe_rl
from safe_rl.env import VecEnv
from safe_rl.modules import EmpiricalNormalization
from safe_rl.utils import store_code_state


class OffPolicyRunner:
    """Off-policy runner for training and evaluation (e.g., DDPG, SAC, TD3)."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.runner_cfg = train_cfg["runner"]
        self.device = device
        self.env = env
        self.log_dir = log_dir

        # Resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # Resolve type of privileged observations (for critic)
        if "critic" in extras.get("observations", {}):
            self.privileged_obs_type = "critic"
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            self.privileged_obs_type = None
            num_critic_obs = num_obs

        # Build actor-critic model
        policy_class_name = self.policy_cfg.pop("class_name", "SACActorCritic")

        # Dynamic import based on policy class
        if policy_class_name == "SACActorCritic":
            from safe_rl.modules import SACActorCritic
            policy_class = SACActorCritic
        elif policy_class_name == "DDPGActorCritic":
            from rsl_rl.modules import DDPGActorCritic
            policy_class = DDPGActorCritic
        else:
            # Try to import from safe_rl.modules first, then fall back to eval
            try:
                import safe_rl.modules as modules
                policy_class = getattr(modules, policy_class_name)
            except AttributeError:
                policy_class = eval(policy_class_name)

        self.actor_critic = policy_class(
            num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # Initialize algorithm
        alg_class_name = self.alg_cfg.pop("class_name", "SAC")

        # Dynamic import based on algorithm class
        if alg_class_name == "SAC":
            from safe_rl.algorithms import SAC
            alg_class = SAC
        elif alg_class_name == "DDPG":
            from rsl_rl.algorithms import DDPG
            alg_class = DDPG
        else:
            # Try to import from safe_rl.algorithms first, then fall back to eval
            try:
                import safe_rl.algorithms as algorithms
                alg_class = getattr(algorithms, alg_class_name)
            except AttributeError:
                alg_class = eval(alg_class_name)

        self.alg = alg_class(self.actor_critic, device=self.device, **self.alg_cfg)

        # Initialize replay buffer
        buffer_size = int(self.runner_cfg.get("max_size", 1_000_000))
        self.alg.init_storage(
            buffer_size=buffer_size,
            num_envs=self.env.num_envs,
            obs_shape=[num_obs],
            act_shape=[self.env.num_actions],
        )

        # Training configuration
        self.num_steps_per_env = int(self.runner_cfg.get("num_steps_per_env", 1))
        self.save_interval = int(self.runner_cfg.get("save_interval", 50))
        self.start_random_steps = int(self.runner_cfg.get("start_random_steps", 10000))
        self.update_after = int(self.runner_cfg.get("update_after", 1000))
        self.update_every = int(self.runner_cfg.get("update_every", 50))
        self.gamma = float(self.alg_cfg.get("gamma", 0.99))

        # Empirical normalization
        self.empirical_normalization = self.runner_cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

        # Logging state
        self.writer = None
        self.logger_type = "tensorboard"
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [safe_rl.__file__]

        # Reset environment
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop."""
        # Initialize logger
        if self.log_dir is not None and self.writer is None:
            self._init_logger()

        # Randomize initial episode lengths (for exploration diversity)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
        if self.empirical_normalization:
            obs = self.obs_normalizer(obs)

        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Training loop
        global_step = self.tot_timesteps
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Initialize loss values for logging
            critic_loss = 0.0
            actor_loss = 0.0
            noise_std = self.actor_critic.std.mean().item() if hasattr(self.actor_critic, 'std') else 0.0

            # Collect data
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Select action
                    if global_step < self.start_random_steps:
                        action = self._sample_random_action()
                    else:
                        action = self.actor_critic.act_with_noise(obs)

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(action.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Apply normalization
                    if self.empirical_normalization:
                        next_obs = self.obs_normalizer(next_obs)

                    # Handle truncation (timeout) - for off-policy, truncated episodes
                    # should not be marked as terminal for proper bootstrapping
                    if "time_outs" in infos:
                        time_outs = infos["time_outs"].to(self.device)
                        # Mask out truncated episodes from done signal
                        # True terminal = done AND NOT truncated
                        terminal = dones * (1.0 - time_outs.float())
                    else:
                        terminal = dones

                    # Store transition in replay buffer
                    self.alg.store_transition(
                        obs,
                        action,
                        rewards,
                        terminal,  # Use terminal instead of dones for proper bootstrapping
                        next_obs,
                    )

                    # Update current observation
                    obs = next_obs
                    global_step += self.env.num_envs

                    # Book keeping for logging
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    cur_reward_sum += rewards
                    cur_episode_length += 1

                    # Handle episode completion
                    new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Update policy
            if global_step >= self.update_after and it % self.update_every == 0:
                update_result = self.alg.update()
                if update_result is not None:
                    if isinstance(update_result, tuple):
                        critic_loss, actor_loss, noise_std = update_result
                    elif isinstance(update_result, dict):
                        # Support both naming conventions (critic/critic_loss, actor/actor_loss)
                        critic_loss = update_result.get("critic", update_result.get("critic_loss", 0.0))
                        actor_loss = update_result.get("actor", update_result.get("actor_loss", 0.0))
                        noise_std = update_result.get("noise_std", noise_std)

            stop = time.time()
            learn_time = stop - start

            # Update counters
            self.tot_timesteps = global_step
            self.current_learning_iteration = it

            # Logging
            if self.log_dir is not None:
                self.log(
                    it=it,
                    tot_iter=tot_iter,
                    start_iter=start_iter,
                    num_learning_iterations=num_learning_iterations,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    critic_loss=critic_loss,
                    actor_loss=actor_loss,
                    noise_std=noise_std,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    ep_infos=ep_infos,
                )

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter and self.log_dir is not None:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths and self.writer is not None:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save final model
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(
        self,
        it: int,
        tot_iter: int,
        start_iter: int,
        num_learning_iterations: int,
        collection_time: float,
        learn_time: float,
        critic_loss: float,
        actor_loss: float,
        noise_std: float,
        rewbuffer: deque,
        lenbuffer: deque,
        ep_infos: list,
        width: int = 80,
        pad: int = 35,
    ):
        """Log training metrics."""
        # Update timing
        iteration_time = collection_time + learn_time
        self.tot_time += iteration_time

        # Compute FPS
        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))

        # Episode info logging
        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
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
                    if self.writer:
                        if "/" in key:
                            self.writer.add_scalar(key, value, it)
                        else:
                            self.writer.add_scalar(f"Episode/{key}", value, it)
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # Compute mean reward/length
        mean_reward = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0.0
        mean_length = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0

        # TensorBoard/Logger writing
        if self.writer:
            self.writer.add_scalar("Loss/critic", critic_loss, it)
            self.writer.add_scalar("Loss/actor", actor_loss, it)
            self.writer.add_scalar("Policy/mean_noise_std", noise_std, it)
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collection_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            if len(rewbuffer) > 0:
                self.writer.add_scalar("Train/mean_reward", mean_reward, it)
                self.writer.add_scalar("Train/mean_episode_length", mean_length, it)
                if self.logger_type != "wandb":
                    self.writer.add_scalar("Train/mean_reward/time", mean_reward, self.tot_time)
                    self.writer.add_scalar("Train/mean_episode_length/time", mean_length, self.tot_time)

        # Console output
        header = f" \033[1m Learning iteration {it}/{tot_iter} \033[0m "
        if len(rewbuffer) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {noise_std:.4f}\n"""
                f"""{'Mean critic loss:':>{pad}} {critic_loss:.4f}\n"""
                f"""{'Mean actor loss:':>{pad}} {actor_loss:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {mean_length:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {noise_std:.4f}\n"""
                f"""{'Mean critic loss:':>{pad}} {critic_loss:.4f}\n"""
                f"""{'Mean actor loss:':>{pad}} {actor_loss:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / max(it - start_iter + 1, 1)
                    * (start_iter + num_learning_iterations - it)
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None):
        """Save model checkpoint."""
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict() if hasattr(self.alg, "optimizer") else None,
            "iter": self.current_learning_iteration,
            "tot_timesteps": self.tot_timesteps,
            "infos": infos,
        }

        # Save critic optimizer if separate
        if hasattr(self.alg, "critic_optimizer"):
            saved_dict["critic_optimizer_state_dict"] = self.alg.critic_optimizer.state_dict()
        if hasattr(self.alg, "actor_optimizer"):
            saved_dict["actor_optimizer_state_dict"] = self.alg.actor_optimizer.state_dict()

        # Save normalizers
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()

        torch.save(saved_dict, path)
        print(f"[Model Saved] -> {path}")

        # Upload to external logger
        if self.logger_type in ["neptune", "wandb"] and self.writer is not None:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        """Load model checkpoint."""
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)

        # Load model
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        # Load normalizers
        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.empirical_normalization and "critic_obs_norm_state_dict" in loaded_dict:
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

        # Load optimizers
        if load_optimizer:
            if "optimizer_state_dict" in loaded_dict and loaded_dict["optimizer_state_dict"] is not None:
                if hasattr(self.alg, "optimizer"):
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "critic_optimizer_state_dict" in loaded_dict and hasattr(self.alg, "critic_optimizer"):
                self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if "actor_optimizer_state_dict" in loaded_dict and hasattr(self.alg, "actor_optimizer"):
                self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])

        # Restore training state
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        self.tot_timesteps = loaded_dict.get("tot_timesteps", 0)

        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None):
        """Get policy for inference/evaluation."""
        self.eval_mode()
        if device is not None:
            self.actor_critic.to(device)

        if self.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            return lambda x: self.actor_critic.act_inference(self.obs_normalizer(x))
        return self.actor_critic.act_inference

    def train_mode(self):
        """Switch to training mode."""
        self.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        self.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path: str):
        """Add a git repository to track for logging."""
        self.git_status_repos.append(repo_file_path)

    def _sample_random_action(self) -> torch.Tensor:
        """Sample random actions for initial exploration."""
        if hasattr(self.actor_critic, "sample_random_action"):
            return self.actor_critic.sample_random_action(self.env.num_envs)
        # Default: uniform random in [-1, 1]
        return torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2 - 1

    def _init_logger(self) -> None:
        """Initialize the logger based on configuration."""
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.logger_type = self.runner_cfg.get("logger", "tensorboard").lower()

        # Build a flattened config for loggers that expect wandb_project at top level
        logger_cfg = {
            **self.cfg,
            "wandb_project": self.runner_cfg.get("wandb_project", "safe_rl"),
            "wandb_entity": self.runner_cfg.get("wandb_entity"),
        }

        if self.logger_type == "neptune":
            from safe_rl.utils.neptune_utils import NeptuneSummaryWriter
            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=logger_cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from safe_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=logger_cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError(f"Logger type '{self.logger_type}' not found. Choose 'neptune', 'wandb', or 'tensorboard'.")
