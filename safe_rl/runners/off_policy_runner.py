from __future__ import annotations

import os
import statistics
import time

import torch

from safe_rl.envs import VecEnv
from safe_rl.modules import EmpiricalNormalization, RewardNormalization
from safe_rl.utils import NStepReturnAggregator
from safe_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for training and evaluation (e.g., SAC, TD3)."""

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
        elif policy_class_name == "SafeSACActorCritic":
            from safe_rl.modules import SafeSACActorCritic
            policy_class = SafeSACActorCritic
        elif policy_class_name == "TD3ActorCritic":
            from safe_rl.modules import TD3ActorCritic
            policy_class = TD3ActorCritic
        else:
            # Try to import from safe_rl.modules first, then fall back to eval
            try:
                import safe_rl.modules as modules
                policy_class = getattr(modules, policy_class_name)
            except AttributeError:
                policy_class = eval(policy_class_name)

        # TD3ActorCritic needs num_envs for its per-env exploration noise buffer.
        if policy_class_name == "TD3ActorCritic":
            self.policy_cfg.setdefault("num_envs", self.env.num_envs)

        self.actor_critic = policy_class(
            num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # Initialize algorithm
        alg_class_name = self.alg_cfg.pop("class_name", "SAC")

        # Set cost_limits for safe RL algorithms
        # Prioritize cost_limits from config, fall back to environment
        if alg_class_name == "SafeSAC":
            if "cost_limits" not in self.alg_cfg or self.alg_cfg["cost_limits"] is None:
                if hasattr(self.env, "cost_limits") and self.env.cost_limits is not None:
                    self.alg_cfg["cost_limits"] = self.env.cost_limits
                else:
                    raise ValueError(
                        f"cost_limits must be specified for safe RL algorithm {alg_class_name}. "
                        "Please specify cost_limits in the config file under 'algorithm.cost_limits' or "
                        "pass --cost_limits argument to the training script."
                    )

        # Dynamic import based on algorithm class
        if alg_class_name == "SAC":
            from safe_rl.algorithms import SAC
            alg_class = SAC
        elif alg_class_name == "FastSAC":
            from safe_rl.algorithms import FastSAC
            alg_class = FastSAC
        elif alg_class_name == "FastTD3":
            from safe_rl.algorithms import FastTD3
            alg_class = FastTD3
        elif alg_class_name == "SafeSAC":
            from safe_rl.algorithms import SafeSAC
            alg_class = SafeSAC
        else:
            # Try to import from safe_rl.algorithms first, then fall back to eval
            try:
                import safe_rl.algorithms as algorithms
                alg_class = getattr(algorithms, alg_class_name)
            except AttributeError:
                alg_class = eval(alg_class_name)

        # Forward n_step into the algorithm so it can set bellman_gamma = gamma ** n_step.
        self.n_step = int(self.runner_cfg.get("n_step", 1))
        if self.n_step > 1:
            self.alg_cfg["n_step"] = self.n_step

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
        self.gamma = float(self.alg_cfg.get("gamma", 0.99))

        # N-step return buffer (optional). When enabled, transitions are aggregated
        # into n-step returns before being written to the replay buffer.
        if self.n_step > 1:
            self.n_step_buffer: NStepReturnAggregator | None = NStepReturnAggregator(
                n_step=self.n_step,
                gamma=self.gamma,
                num_envs=self.env.num_envs,
                device=self.device,
            )
        else:
            self.n_step_buffer = None

        # Empirical normalization
        self.empirical_normalization = self.runner_cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

        # Reward normalization. Two modes:
        #   "empirical" (default): running std of raw rewards (EmpiricalNormalization)
        #   "return": running std of discounted returns (RewardNormalization)
        self.reward_normalization = self.runner_cfg.get("reward_normalization", True)
        reward_norm_mode = self.runner_cfg.get("reward_normalization_mode", "empirical")
        self.reward_normalization_mode = reward_norm_mode
        if self.reward_normalization:
            if reward_norm_mode == "return":
                self.reward_normalizer = RewardNormalization(
                    gamma=self.gamma,
                    g_max=float(self.runner_cfg.get("reward_normalization_g_max", 10.0)),
                ).to(self.device)
            else:
                self.reward_normalizer = EmpiricalNormalization(shape=[1], until=None).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity().to(self.device)

        # Safe RL detection
        self.is_safe_rl = hasattr(self.alg, 'num_costs') and self.alg.num_costs > 0
        num_costs = self.alg.num_costs if self.is_safe_rl else 0

        if self.is_safe_rl and self.n_step_buffer is not None:
            raise NotImplementedError(
                "n_step > 1 is not supported with safe RL algorithms yet "
                "(cost aggregation is not implemented in NStepReturnAggregator)."
            )

        # Create logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            runner_cfg=self.runner_cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            num_costs=num_costs,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self.tot_timesteps = 0

        # Reset environment
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop."""
        print("Starting to learn with:")
        print(str(self))

        # Randomize initial episode lengths (for exploration diversity)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations (store raw, normalize only for action selection)
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        critic_obs = extras.get("observations", {}).get(self.privileged_obs_type, obs)
        critic_obs = critic_obs.to(self.device) if isinstance(critic_obs, torch.Tensor) else obs
        if self.empirical_normalization:
            # Update normalizer stats but keep obs raw for buffer storage
            self.obs_normalizer(obs)
            self.critic_obs_normalizer(critic_obs)

        self.train_mode()

        # Training loop
        global_step = self.tot_timesteps
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Collect data
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Select action (normalize obs for policy, but keep raw for buffer)
                    if global_step < self.start_random_steps:
                        action = self._sample_random_action()
                    else:
                        # Normalize for policy without updating stats (already updated when obs arrived as next_obs)
                        if self.empirical_normalization:
                            with torch.no_grad():
                                obs_for_policy = (obs - self.obs_normalizer._mean) / (self.obs_normalizer._std + self.obs_normalizer.eps)
                        else:
                            obs_for_policy = obs
                        # Use algorithm's act method if available (e.g., for shielding)
                        if hasattr(self.alg, 'act'):
                            action = self.alg.act(obs_for_policy, eval_mode=False)
                        else:
                            action = self.actor_critic.act_with_noise(obs_for_policy)

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(action.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_critic_obs = infos.get("observations", {}).get(self.privileged_obs_type, next_obs)
                    next_critic_obs = next_critic_obs.to(self.device) if isinstance(next_critic_obs, torch.Tensor) else next_obs

                    # Update normalizer stats with raw data (don't transform for storage)
                    if self.empirical_normalization:
                        self.obs_normalizer(next_obs)
                        self.critic_obs_normalizer(next_critic_obs)
                    if self.reward_normalization:
                        if isinstance(self.reward_normalizer, RewardNormalization):
                            self.reward_normalizer.update(rewards, dones.float())
                        else:
                            self.reward_normalizer.update(rewards)

                    # Handle truncation (timeout) - for off-policy, truncated episodes
                    # should not be marked as terminal for proper bootstrapping
                    if "time_outs" in infos:
                        time_outs = infos["time_outs"].to(self.device).float()
                        # Mask out truncated episodes from done signal
                        # True terminal = done AND NOT truncated
                        terminal = dones * (1.0 - time_outs)
                    else:
                        time_outs = torch.zeros_like(dones, dtype=torch.float32)
                        terminal = dones

                    # Preserve final observations for timeouts when the env provides them.
                    final_obs = infos.get("observations", {}).get("final", {})
                    if isinstance(final_obs, dict) and "time_outs" in infos:
                        time_out_mask = infos["time_outs"].to(self.device).bool().unsqueeze(-1)
                        final_actor_obs = final_obs.get("actor_obs")
                        final_critic_obs = final_obs.get("critic_obs")
                        if isinstance(final_actor_obs, torch.Tensor):
                            final_actor_obs = final_actor_obs.to(self.device)
                            next_obs = torch.where(time_out_mask, final_actor_obs, next_obs)
                        if isinstance(final_critic_obs, torch.Tensor):
                            final_critic_obs = final_critic_obs.to(self.device)
                            next_critic_obs = torch.where(time_out_mask, final_critic_obs, next_critic_obs)

                    # Get costs if available (for Safe RL)
                    costs = None
                    if self.is_safe_rl and "costs" in infos:
                        costs = infos["costs"].to(self.device)
                        # Ensure costs have proper shape [num_envs, num_costs]
                        if costs.dim() == 1:
                            costs = costs.unsqueeze(-1)
                        if costs.shape[-1] != self.alg.num_costs:
                            costs = costs.expand(-1, self.alg.num_costs)

                    # Store transition in replay buffer (optionally aggregated via n-step buffer)
                    if self.is_safe_rl and hasattr(self.alg, 'store_transition'):
                        self.alg.store_transition(
                            obs,
                            action,
                            rewards,
                            terminal,
                            next_obs,
                            cost=costs,
                            critic_obs=critic_obs,
                            next_critic_obs=next_critic_obs,
                        )
                    elif self.n_step_buffer is not None:
                        self.n_step_buffer.push(
                            storage=self.alg.storage,
                            obs=obs,
                            action=action,
                            reward=rewards,
                            next_obs=next_obs,
                            terminal=terminal,
                            truncated=time_outs,
                            critic_obs=critic_obs,
                            next_critic_obs=next_critic_obs,
                        )
                    else:
                        self.alg.store_transition(
                            obs,
                            action,
                            rewards,
                            terminal,  # Use terminal instead of dones for proper bootstrapping
                            next_obs,
                            critic_obs=critic_obs,
                            next_critic_obs=next_critic_obs,
                        )

                    # Update current observation
                    obs = next_obs
                    critic_obs = next_critic_obs
                    global_step += self.env.num_envs

                    # Update logger episode buffers
                    self.logger.process_env_step(rewards, dones, infos, costs=costs)

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Update policy
            loss_dict: dict = {}
            if global_step >= self.update_after:
                # Pass normalizers so update normalizes at sample time
                norm_kwargs = {}
                if self.empirical_normalization:
                    norm_kwargs["obs_normalizer"] = self.obs_normalizer
                    norm_kwargs["critic_obs_normalizer"] = self.critic_obs_normalizer
                if self.reward_normalization:
                    norm_kwargs["reward_normalizer"] = self.reward_normalizer

                # For Safe RL, pass current episode costs for PID Lagrangian updates
                if self.is_safe_rl:
                    current_costs = [
                        statistics.mean(self.logger.costbuffers[i]) if len(self.logger.costbuffers[i]) > 0 else 0.0
                        for i in range(self.alg.num_costs)
                    ]
                    update_result = self.alg.update(current_costs=current_costs, **norm_kwargs)
                else:
                    update_result = self.alg.update(**norm_kwargs)

                if update_result is not None:
                    if isinstance(update_result, tuple):
                        loss_dict["critic_loss"] = update_result[0]
                        loss_dict["actor_loss"] = update_result[1]
                        loss_dict["noise_std"] = update_result[2]
                    elif isinstance(update_result, dict):
                        # Normalize key names to our convention
                        loss_dict["critic_loss"] = update_result.get("critic", update_result.get("critic_loss", 0.0))
                        loss_dict["actor_loss"] = update_result.get("actor", update_result.get("actor_loss", 0.0))
                        loss_dict["noise_std"] = update_result.get("noise_std", 0.0)
                        if self.is_safe_rl:
                            loss_dict["cost_critic_loss"] = update_result.get("cost_critic", update_result.get("cost_critic_loss", 0.0))

                # Merge safe RL penalty info into loss_dict
                if self.is_safe_rl:
                    if hasattr(self.alg, 'get_penalty_info'):
                        penalty_info = self.alg.get_penalty_info()
                        loss_dict["lambda_mean"] = penalty_info.get("lambda_mean", 0.0)
                        loss_dict["lambda_max"] = penalty_info.get("lambda_max", 0.0)
                        if "alpha" in penalty_info:
                            loss_dict["alpha"] = penalty_info["alpha"]

                    if hasattr(self.alg, 'get_shield_stats'):
                        shield_stats = self.alg.get_shield_stats()
                        loss_dict["shield_rejections"] = shield_stats.get("rejections", 0)
                        loss_dict["shield_total_samples"] = shield_stats.get("total_samples", 0)
                        loss_dict["shield_avg_resamples"] = shield_stats.get("avg_resamples", 0.0)

            # Fill in noise_std if not set by update
            if "noise_std" not in loss_dict:
                if hasattr(self.alg, 'get_actual_action_std'):
                    loss_dict["noise_std"] = self.alg.get_actual_action_std()
                elif hasattr(self.actor_critic, 'std'):
                    loss_dict["noise_std"] = self.actor_critic.std.mean().item()

            stop = time.time()
            learn_time = stop - start

            # Update counters
            self.tot_timesteps = global_step
            self.current_learning_iteration = it

            # Log
            self.logger.log(
                it=it,
                start_it=start_iter,
                total_it=tot_iter,
                collect_time=collection_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
            )

            # Save model
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # Save final model
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

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
        if self.reward_normalization:
            saved_dict["reward_norm_state_dict"] = self.reward_normalizer.state_dict()

        torch.save(saved_dict, path)
        print(f"[Model Saved] -> {path}")

        # Upload to external logger
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        """Load model checkpoint."""
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)

        # Load model
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)

        # Load normalizers
        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.empirical_normalization and "critic_obs_norm_state_dict" in loaded_dict:
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if self.reward_normalization and "reward_norm_state_dict" in loaded_dict:
            rn_sd = loaded_dict["reward_norm_state_dict"]
            if "G" in rn_sd and rn_sd["G"].shape != self.reward_normalizer.G.shape:
                rn_sd["G"] = torch.zeros_like(self.reward_normalizer.G)
            self.reward_normalizer.load_state_dict(rn_sd)

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
        self.logger.git_status_repos.append(repo_file_path)

    def __str__(self) -> str:
        sep = "─" * 56
        alg = self.alg
        ac = self.actor_critic
        n_envs = self.env.num_envs
        steps = self.num_steps_per_env

        lines = [
            sep,
            f"  {alg.__class__.__name__}  |  {n_envs:,} envs  |  {self.device}",
            sep,
            "  Runner",
            f"    {'num_actions:':<28} {self.env.num_actions}",
            f"    {'num_steps_per_env:':<28} {steps}",
            f"    {'start_random_steps:':<28} {self.start_random_steps}",
            f"    {'update_after:':<28} {self.update_after}",
            f"    {'transitions/iter:':<28} {n_envs * steps:,}",
            f"    {'empirical_normalization:':<28} {self.empirical_normalization}",
            f"    {'reward_normalization:':<28} {self.reward_normalization} ({self.reward_normalization_mode})",
            f"    {'n_step:':<28} {self.n_step}"
            + (f" (bellman_gamma = {self.gamma ** self.n_step:.5f})" if self.n_step > 1 else ""),
            "",
            f"  Policy  ({ac.__class__.__name__})",
            f"    {'actor_type:':<28} {ac.actor_type}",
            f"    {'critic_type:':<28} {ac.critic_type}",
            f"    {'num_critics:':<28} {ac.num_critics}",
            f"    {'actor:':<28} {ac.actor}",
            f"    {'critic:':<28} {ac.critics[0]}",
            "",
            f"  Algorithm  ({alg.__class__.__name__})",
            f"    {'batch_size:':<28} {alg.batch_size:,}",
            f"    {'gamma / tau:':<28} {alg.gamma}  /  {alg.tau}",
            f"    {'num_updates_per_step:':<28} {alg.num_updates_per_step}",
            f"    {'policy_frequency:':<28} {alg.policy_frequency}",
            f"    {'actor_lr / critic_lr:':<28} {alg.actor_optimizer.param_groups[0]['lr']}  /  {alg.critic_optimizer.param_groups[0]['lr']}",
        ]

        if hasattr(alg, "auto_entropy_tuning"):
            lines.append(f"    {'auto_entropy_tuning:':<28} {alg.auto_entropy_tuning}")
            lines.append(f"    {'alpha:':<28} {alg.alpha.item():.4f}")
            if alg.auto_entropy_tuning:
                lines.append(f"    {'target_entropy:':<28} {alg.target_entropy}")
        if hasattr(alg, "smoothing_noise"):
            lines.append(f"    {'target_smoothing_noise:':<28} {alg.smoothing_noise} (clip {alg.noise_clip})")

        # Safe RL info
        if self.is_safe_rl:
            lines += [
                "",
                "  Safe RL",
                f"    {'num_costs:':<28} {alg.num_costs}",
                f"    {'cost_limits:':<28} {alg.cost_limits}",
            ]
            if hasattr(alg, 'lambdas'):
                lambda_str = ", ".join([f"{l:.4f}" for l in alg.lambdas])
                lines.append(f"    {'lambdas:':<28} [{lambda_str}]")

        # Storage
        if alg.storage is not None:
            lines += [
                "",
                "  Storage",
                f"    {'max_size:':<28} {alg.storage.max_size:,}",
                f"    {'device:':<28} {alg.storage.device}",
            ]

        lines.append(sep)
        return "\n".join(lines)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export actor + obs normalizer to ONNX (mirrors rsl_rl OnPolicyRunner.export_policy_to_onnx)."""
        onnx_model = self.actor_critic.as_onnx(obs_normalizer=self.obs_normalizer, verbose=verbose)
        onnx_model.to("cpu").eval()

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
        )
        print(f"[ONNX Exported] -> {save_path}")

    def _sample_random_action(self) -> torch.Tensor:
        """Sample random actions for initial exploration."""
        if hasattr(self.actor_critic, "sample_random_action"):
            return self.actor_critic.sample_random_action(self.env.num_envs)
        # Default: uniform random in [-1, 1]
        return torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2 - 1
