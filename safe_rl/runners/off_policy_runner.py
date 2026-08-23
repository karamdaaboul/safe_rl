from __future__ import annotations

import math
import os
import statistics
import time
from copy import deepcopy

import torch

from safe_rl.envs import VecEnv
from safe_rl.modules import EmpiricalNormalization, RewardNormalization
from safe_rl.utils import NStepReturnAggregator
from safe_rl.utils.console import get_logger
from safe_rl.utils.logger import Logger

LOGGER = get_logger(__name__)


def _resolve_class(module_name: str, class_name: str, kind: str) -> type:
    """Resolve a config's ``class_name`` string to a class."""
    import importlib

    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        available = ", ".join(sorted(n for n in dir(module) if not n.startswith("_")))
        raise ValueError(f"Unknown {kind} class_name {class_name!r}. {module_name} exports: {available}") from None


class OffPolicyRunner:
    """Off-policy runner for training and evaluation (e.g., SAC, TD3)."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        eval_env: VecEnv | None = None,
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
        policy_class = _resolve_class("safe_rl.modules", policy_class_name, "policy")

        # TD3ActorCritic needs num_envs for its per-env exploration noise buffer.
        if policy_class_name == "TD3ActorCritic":
            self.policy_cfg.setdefault("num_envs", self.env.num_envs)

        self.actor_critic = policy_class(
            num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self._assert_critic_cfg_applied()

        # Initialize algorithm
        alg_class_name = self.alg_cfg.pop("class_name", "SAC")

        # Set cost_limits for safe RL algorithms
        # Prioritize cost_limits from config, fall back to environment
        if alg_class_name in ("SafeSAC", "CVPO", "CVPOPerState", "FHDCMPO", "FHDCMPODIME", "FHDCMPOPerState"):
            if "cost_limits" not in self.alg_cfg or self.alg_cfg["cost_limits"] is None:
                if hasattr(self.env, "cost_limits") and self.env.cost_limits is not None:
                    self.alg_cfg["cost_limits"] = self.env.cost_limits
                else:
                    raise ValueError(
                        f"cost_limits must be specified for safe RL algorithm {alg_class_name}. "
                        "Please specify cost_limits in the config file under 'algorithm.cost_limits' or "
                        "pass --cost_limits argument to the training script."
                    )

        alg_class = _resolve_class("safe_rl.algorithms", alg_class_name, "algorithm")

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
        # Console/scalar logging interval (in iterations). Off-policy iterations are
        # tiny (often num_steps_per_env=1), so logging every iteration floods the
        # console; only log every `log_interval` iterations (rsl_rl_sac default: 20).
        self.log_interval = int(self.runner_cfg.get("log_interval", 20))
        self.start_random_steps = int(self.runner_cfg.get("start_random_steps", 10000))
        self.update_after = int(self.runner_cfg.get("update_after", 1000))
        # Off-policy mismatch diagnostic (default off): stamp every stored transition with
        # log pi_behavior(a|s) and the collection-time iteration, so TD(lambda) targets can be
        # audited for behavior/current-policy drift. Requires the policy to expose
        # `action_log_prob`; costs one extra actor forward per collection step when enabled.
        self.store_behavior_logprob = bool(self.runner_cfg.get("store_behavior_logprob", False))
        if self.store_behavior_logprob and not hasattr(self.actor_critic, "action_log_prob"):
            raise ValueError(
                "store_behavior_logprob=True requires the policy to implement action_log_prob(); "
                f"{type(self.actor_critic).__name__} does not."
            )
        # Periodic deterministic evaluation (Eval/* section in wandb). Off by default: it needs a
        # dedicated eval env (never the training env -- stepping that with deterministic actions
        # would corrupt open episodes, the replay stream and the Episode/ statistics).
        self.eval_env = eval_env
        self.eval_interval = int(self.runner_cfg.get("eval_interval", 0))
        self.eval_episodes = int(self.runner_cfg.get("eval_episodes", 8))
        self._last_eval_metrics: dict[str, float] = {}
        if self.eval_interval > 0 and eval_env is None:
            LOGGER.warning(
                "eval_interval=%d but no eval_env was passed to the runner; deterministic "
                "evaluation is disabled. The training script builds one when the runner config "
                "sets eval_interval > 0.",
                self.eval_interval,
            )
        self.gamma = float(self.alg_cfg.get("gamma", 0.99))

        # Whether the algorithm aggregates n-step returns inside its own storage
        # (at sample time). If so, the runner stores raw 1-step transitions and
        # skips the runner-level NStepReturnAggregator.
        self.storage_native_n_step = self.n_step > 1 and getattr(self.alg, "supports_storage_n_step", False)
        # Whether the algorithm reads a separate bootstrap (timeout) channel, in
        # which case the runner stores truthful dones + a bootstrap flag instead
        # of the collapsed terminal signal.
        self.uses_bootstrap_channel = getattr(self.alg, "uses_bootstrap_channel", False)
        # One-shot guards so a missing/unusable terminal observation is reported
        # once per run rather than every step (see _terminal_observations).
        self._final_obs_warned = False
        self._final_obs_shape_warned = False

        # N-step return buffer (optional). When enabled, transitions are aggregated
        # into n-step returns before being written to the replay buffer. Skipped
        # when the algorithm handles n-step natively inside its storage.
        if self.n_step > 1 and not self.storage_native_n_step:
            self.n_step_buffer: NStepReturnAggregator | None = NStepReturnAggregator(
                n_step=self.n_step,
                gamma=self.gamma,
                num_envs=self.env.num_envs,
                device=self.device,
            )
        else:
            self.n_step_buffer = None

        # Empirical normalization. `obs_normalization_clip` bounds the normalized
        # output; see EmpiricalNormalization (FCSRL's equivalent clips at 50).
        self.empirical_normalization = self.runner_cfg.get("empirical_normalization", False)
        obs_clip = self.runner_cfg.get("obs_normalization_clip", None)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8, clip=obs_clip).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8, clip=obs_clip).to(
                self.device
            )
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
        self.is_safe_rl = hasattr(self.alg, "num_costs") and self.alg.num_costs > 0
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
        LOGGER.info("Starting to learn with:\n%s", self)

        # Randomize initial episode lengths (for exploration diversity)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, critic_obs = self._initial_observations()
        self.train_mode()

        # Training loop
        global_step = self.tot_timesteps
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Windowed logging accumulators (rsl_rl_sac style): timing is summed across
        # the iterations between two log points so reported FPS/timing covers the
        # whole window rather than a single iteration.
        log_window_collect_time = 0.0
        log_window_learn_time = 0.0
        log_window_iters = 0
        # Simulator steps in the window. Tracked here rather than recomputed by the
        # Logger, whose num_steps_per_env * num_envs assumption is wrong under
        # action repeat.
        log_window_steps = 0
        # Latest budget published by a cost-limit curriculum wrapper, if one is attached.
        curriculum_cost_limit: float | None = None

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Collect data
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Select action (normalize obs for policy, but keep raw for buffer)
                    if global_step < self.start_random_steps:
                        action = self._sample_random_action()
                        is_random = True
                    else:
                        obs_for_policy = self._obs_for_policy(obs)
                        # Use algorithm's act method if available (e.g., for shielding)
                        if hasattr(self.alg, "act"):
                            action = self.alg.act(obs_for_policy, eval_mode=False)
                        else:
                            action = self.actor_critic.act_with_noise(obs_for_policy)
                        is_random = False
                    behavior_log_prob = self._behavior_log_prob(obs, action, is_random)

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(action.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_critic_obs = self._privileged_obs(infos, next_obs)

                    # Update normalizer stats with raw data (don't transform for storage)
                    if self.empirical_normalization:
                        self.obs_normalizer(next_obs)
                        self.critic_obs_normalizer(next_critic_obs)
                    if self.reward_normalization:
                        if isinstance(self.reward_normalizer, RewardNormalization):
                            self.reward_normalizer.update(rewards, dones.float())
                        else:
                            self.reward_normalizer.update(rewards)

                    time_outs, terminal, store_done, store_bootstrap = self._episode_boundaries(infos, dones)

                    # Substitute the true terminal observation on truncation. After an
                    # auto-reset the env returns the *next* episode's first observation,
                    # so bootstrapping from it values a transition that never happened.
                    # Only the stored transition is patched: `next_obs` itself carries
                    # forward as the next step's policy input and must stay post-reset.
                    store_next_obs, store_next_critic_obs = self._terminal_observations(
                        infos, time_outs, next_obs, next_critic_obs
                    )

                    costs = self._costs(infos)
                    self._store_transition(
                        obs=obs,
                        action=action,
                        rewards=rewards,
                        costs=costs,
                        done=store_done,
                        terminal=terminal,
                        time_outs=time_outs,
                        bootstrap=store_bootstrap,
                        next_obs=store_next_obs,
                        critic_obs=critic_obs,
                        next_critic_obs=store_next_critic_obs,
                        behavior_log_prob=behavior_log_prob,
                        policy_version=None
                        if behavior_log_prob is None
                        else torch.full((obs.shape[0], 1), float(it), device=self.device),
                    )

                    # Update current observation
                    obs = next_obs
                    critic_obs = next_critic_obs
                    # Simulator steps, not decisions: action repeat makes these differ.
                    if "cost_limit" in infos:
                        curriculum_cost_limit = float(infos["cost_limit"])
                    step_delta = int(infos.get("sim_steps", self.env.num_envs))
                    global_step += step_delta
                    log_window_steps += step_delta

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
                    if curriculum_cost_limit is not None and hasattr(self.alg, "set_cost_limit"):
                        self.alg.set_cost_limit(curriculum_cost_limit)
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
                        # Entropy coefficient (-> SafeRL/alpha) and its loss (-> Loss/alpha).
                        # For safe RL these may be overridden by get_penalty_info() below.
                        if "alpha" in update_result:
                            loss_dict["alpha"] = update_result["alpha"]
                        if "alpha_loss" in update_result:
                            loss_dict["alpha_loss"] = update_result["alpha_loss"]
                        if self.is_safe_rl:
                            loss_dict["cost_critic_loss"] = update_result.get(
                                "cost_critic", update_result.get("cost_critic_loss", 0.0)
                            )

                # Merge algorithm diagnostics into loss_dict. Unconstrained algorithms expose
                # this too (MPO reports its E-step dual residual, ESS and KL ratios here), so
                # the scalar keys are merged for every algorithm that provides them; only the
                # multiplier/cost keys are safe-RL specific.
                if hasattr(self.alg, "get_penalty_info"):
                    penalty_info = self.alg.get_penalty_info()
                    for key, value in penalty_info.items():
                        if isinstance(value, (int, float)) and key not in loss_dict:
                            loss_dict[key] = float(value)
                    if self.is_safe_rl:
                        loss_dict["lambda_mean"] = penalty_info.get("lambda_mean", 0.0)
                        loss_dict["lambda_max"] = penalty_info.get("lambda_max", 0.0)
                    if "alpha" in penalty_info:
                        loss_dict["alpha"] = penalty_info["alpha"]

                if self.is_safe_rl:
                    if hasattr(self.alg, "get_shield_stats"):
                        shield_stats = self.alg.get_shield_stats()
                        loss_dict["shield_rejections"] = shield_stats.get("rejections", 0)
                        loss_dict["shield_total_samples"] = shield_stats.get("total_samples", 0)
                        loss_dict["shield_avg_resamples"] = shield_stats.get("avg_resamples", 0.0)

            # Fill in noise_std if not set by update
            if "noise_std" not in loss_dict:
                if hasattr(self.alg, "get_actual_action_std"):
                    loss_dict["noise_std"] = self.alg.get_actual_action_std()
                elif hasattr(self.actor_critic, "std"):
                    loss_dict["noise_std"] = self.actor_critic.std.mean().item()

            # Log the actor learning rate (rsl_rl_sac logs this each window).
            if "learning_rate" not in loss_dict:
                if hasattr(self.alg, "actor_learning_rate"):
                    loss_dict["learning_rate"] = self.alg.actor_learning_rate
                elif hasattr(self.alg, "actor_optimizer"):
                    loss_dict["learning_rate"] = self.alg.actor_optimizer.param_groups[0]["lr"]

            stop = time.time()
            learn_time = stop - start

            # Update counters
            self.tot_timesteps = global_step
            self.current_learning_iteration = it

            # Accumulate this iteration's timing into the current log window.
            log_window_collect_time += collection_time
            log_window_learn_time += learn_time
            log_window_iters += 1

            # Periodic deterministic evaluation (Eval/* in wandb; eval_interval = 0 disables).
            # Run BEFORE the log gate so a fresh panel is always attached to the next log point;
            # between eval points the last panel is re-attached, keeping the wandb series dense.
            if self.eval_env is not None and self.eval_interval > 0 and it > 0 and it % self.eval_interval == 0:
                self._last_eval_metrics = self._run_deterministic_eval()
            if self._last_eval_metrics:
                loss_dict.update(self._last_eval_metrics)

            # Log only every `log_interval` iterations (and on the final iteration),
            # passing the windowed timing and the number of iterations it covers.
            should_log = (it % self.log_interval == 0) or (it == tot_iter - 1)
            if should_log:
                self.logger.log(
                    it=it,
                    start_it=start_iter,
                    total_it=tot_iter,
                    collect_time=log_window_collect_time,
                    learn_time=log_window_learn_time,
                    loss_dict=loss_dict,
                    num_iters=log_window_iters,
                    collection_size=log_window_steps,
                )
                log_window_collect_time = 0.0
                log_window_learn_time = 0.0
                log_window_iters = 0
                log_window_steps = 0

            # Save model (skip iter 0: nothing has been learned yet)
            if self.log_dir is not None and it % self.save_interval == 0 and it != 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # Save final model
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _run_deterministic_eval(self) -> dict[str, float]:
        """``eval_episodes`` deterministic episodes on the dedicated eval env -> ``eval_*`` panel.

        Deterministic (``act_inference``) is the DEPLOYMENT distribution; the training-side
        ``Episode/`` numbers remain the stochastic behavior distribution, so the two series
        answer different questions and are logged side by side. The panel refreshes every
        ``eval_interval`` iterations and is re-attached to intermediate log points unchanged.
        """
        start = time.time()
        was_training = self.actor_critic.training
        self.eval_mode()
        n = self.eval_env.num_envs
        limit = float(self.alg.cost_limits[0]) if getattr(self.alg, "num_costs", 0) else float("inf")

        # Fresh episodes every panel: without the reset, the second eval would resume mid-episode
        # where the previous one stopped and the first "episode" totals would be partial.
        obs, _ = self.eval_env.reset()
        run_rew = torch.zeros(n, device=self.device)
        run_cost = torch.zeros(n, device=self.device)
        ep_rews: list[float] = []
        ep_costs: list[float] = []
        with torch.inference_mode():
            while len(ep_rews) < self.eval_episodes:
                obs_n = (
                    self.obs_normalizer(obs.to(self.device)) if self.empirical_normalization else obs.to(self.device)
                )
                actions = self.actor_critic.act_inference(obs_n)
                obs, rewards, dones, infos = self.eval_env.step(actions.to(self.eval_env.device))
                run_rew += rewards.to(self.device).reshape(-1)
                if "costs" in infos:
                    run_cost += infos["costs"].to(self.device).reshape(n, -1).sum(-1)
                for e in (dones > 0).nonzero(as_tuple=False).reshape(-1).tolist():
                    ep_rews.append(float(run_rew[e]))
                    ep_costs.append(float(run_cost[e]))
                    run_rew[e] = 0.0
                    run_cost[e] = 0.0
        if was_training:
            self.train_mode()

        costs_t = torch.tensor(ep_costs)
        metrics = {
            "eval_reward": float(torch.tensor(ep_rews).mean()),
            "eval_cost": float(costs_t.mean()),
            "eval_cost_p90": float(costs_t.quantile(0.9)) if len(ep_costs) > 1 else float(costs_t.max()),
            "eval_violation_rate": float((costs_t > limit).float().mean()),
            "eval_episodes_n": float(len(ep_rews)),
            "eval_time_s": time.time() - start,
        }
        return metrics

    def save(self, path: str, infos: dict | None = None):
        """Save model checkpoint."""
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict() if hasattr(self.alg, "optimizer") else None,
            "iter": self.current_learning_iteration,
            "tot_timesteps": self.tot_timesteps,
            "infos": infos,
            # Architecture provenance. Evaluation rebuilds the critics from a YAML `policy:`
            # block and then loads weights with strict=False, so a config that disagrees with
            # the checkpoint (wrong n_quantiles, wrong num_atoms) silently yields a partially
            # RANDOM critic. Recording what was actually built lets `load` say so out loud.
            # Purely informational: nothing reads it to construct anything.
            "policy_cfg": deepcopy(self.policy_cfg),
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
        LOGGER.info("[Model Saved] -> %s", path)

        # Upload to external logger
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        """Load model checkpoint."""
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)

        # Architecture check BEFORE loading: strict=False means a critic built to a different
        # shape than the checkpoint loads partially and silently, leaving randomly initialized
        # tensors behind. Warn rather than raise, so checkpoints predating this key (which
        # carry no `policy_cfg`) still load exactly as they did before.
        self._warn_on_policy_cfg_mismatch(loaded_dict.get("policy_cfg"), path)

        # Load model
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)

        # Load normalizers. A checkpoint trained with normalization is unusable
        # without its statistics — the policy would see inputs on a scale it never
        # saw in training — so a mismatch is reported rather than silently ignored.
        if not self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            LOGGER.warning(
                "checkpoint %s carries observation-normalizer statistics but this run has "
                "empirical_normalization disabled; the policy will see unnormalized "
                "observations it was never trained on.",
                path,
            )
        if self.empirical_normalization and "obs_norm_state_dict" not in loaded_dict:
            LOGGER.warning(
                "empirical_normalization is enabled but checkpoint %s carries no "
                "normalizer statistics; starting from empty statistics.",
                path,
            )
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
            f"    {'num_critics:':<28} {getattr(ac, 'num_critics', getattr(ac, 'num_reward_critics', '?'))}",
            f"    {'actor:':<28} {ac.actor}",
            f"    {'critic:':<28} {(ac.critics if hasattr(ac, 'critics') else ac.reward_critics)[0]}",
            "",
            f"  Algorithm  ({alg.__class__.__name__})",
            f"    {'batch_size:':<28} {alg.batch_size:,}",
            f"    {'gamma / tau:':<28} {alg.gamma}  /  {alg.tau}",
            f"    {'num_updates_per_step:':<28} {alg.num_updates_per_step}",
            f"    {'policy_frequency:':<28} {alg.policy_frequency}",
            (
                f"    {'actor_lr / critic_lr:':<28} {alg.actor_optimizer.param_groups[0]['lr']}  / "
                f" {alg.critic_optimizer.param_groups[0]['lr']}"
            ),
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
            if hasattr(alg, "lambdas"):
                lambda_str = ", ".join(f"{value:.4f}" for value in alg.lambdas)
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
        LOGGER.info("[ONNX Exported] -> %s", save_path)

    def _warn_on_policy_cfg_mismatch(self, saved_cfg: dict | None, path: str) -> None:
        """Report keys where this run's `policy:` block disagrees with the checkpoint's."""
        if not saved_cfg:
            return
        current = self.policy_cfg
        mismatches = []
        for key in sorted(set(saved_cfg) | set(current)):
            was, now = saved_cfg.get(key, "<absent>"), current.get(key, "<absent>")
            if was != now:
                mismatches.append(f"{key}: checkpoint={was!r} vs this run={now!r}")
        if mismatches:
            LOGGER.warning(
                "checkpoint %s was trained with a different policy config; weights are loaded "
                "with strict=False, so any layer whose shape disagrees stays RANDOMLY "
                "INITIALIZED. Differences:\n  %s",
                path,
                "\n  ".join(mismatches),
            )

    def _assert_critic_cfg_applied(self) -> None:
        """Print the RESOLVED critic config and assert it matches the YAML.

        This exists because of a real, repeated failure mode in this repo: a config key that
        never reaches the constructor leaves the model silently at its default (see the
        `n_step` note in config/safety_gymnasium_dmpo_costprobe.yaml). A run that reports
        `n_quantiles=64` because the YAML said so is indistinguishable from one that got 64
        by default -- unless something checks. Permanent by design; it costs one block of
        output per run and turns a silent misconfiguration into a startup failure.
        """
        critic_type = self.policy_cfg.get("critic_type")
        cost_critic_type = self.policy_cfg.get("cost_critic_type")
        if critic_type is None and cost_critic_type is None:
            return

        def _describe(critics, cfg_key: str, label: str) -> str:
            cfg = self.policy_cfg.get(cfg_key) or {}
            critic = critics[0]
            resolved = []
            for key in ("num_atoms", "v_min", "v_max", "n_quantiles", "kappa", "nonneg", "tqc_drop"):
                if not hasattr(critic, key):
                    continue
                actual = getattr(critic, key)
                resolved.append(f"{key}={actual}")
                if key in cfg and cfg[key] != actual:
                    raise AssertionError(
                        f"{label}: config asked for {key}={cfg[key]!r} but the built critic has "
                        f"{key}={actual!r} -- the key did not reach the constructor."
                    )
            return f"{label}: " + ", ".join(resolved)

        lines = []
        if critic_type is not None and getattr(self.actor_critic, "reward_critics", None):
            lines.append(
                f"critic_type={critic_type} | " + _describe(self.actor_critic.reward_critics, "critic_kwargs", "reward")
            )
        if cost_critic_type is not None and getattr(self.actor_critic, "cost_critics", None):
            lines.append(
                f"cost_critic_type={cost_critic_type} | "
                + _describe(self.actor_critic.cost_critics, "cost_critic_kwargs", "cost")
            )
        if lines:
            print("[INFO] Resolved critic config:\n    " + "\n    ".join(lines))

    def _sample_random_action(self) -> torch.Tensor:
        """Sample random actions for initial exploration."""
        if hasattr(self.actor_critic, "sample_random_action"):
            return self.actor_critic.sample_random_action(self.env.num_envs)
        # Default: uniform random in [-1, 1]
        return torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2 - 1

    def _behavior_log_prob(self, obs: torch.Tensor, action: torch.Tensor, is_random: bool) -> torch.Tensor | None:
        """Exact ``log pi_behavior(a|s)`` of the just-taken action, or None when the
        ``store_behavior_logprob`` flag is off (off-policy mismatch diagnostic).

        Warmup actions are uniform over the per-joint action box [b - c, b + c]
        (see :meth:`SACActorCritic.sample_random_action`), so their density is the
        constant ``prod_j 1/(2 c_j)`` — with unscaled actions, -A*log(2). Policy actions
        are evaluated through :meth:`action_log_prob`, i.e. the same atanh-inversion path
        any later mismatch measurement uses, so tanh saturation cancels exactly.
        """
        if not self.store_behavior_logprob:
            return None
        if not is_random:
            return self.actor_critic.action_log_prob(self._obs_for_policy(obs), action)
        log_p = -action.shape[-1] * math.log(2.0)
        actor = getattr(self.actor_critic, "actor", None)
        if actor is not None and getattr(actor, "scaled_actions", False):
            log_p += float(actor.neg_log_action_scale)
        return torch.full((action.shape[0], 1), log_p, device=self.device)

    def _privileged_obs(self, source: dict, obs: torch.Tensor) -> torch.Tensor:
        """Critic observation from an extras/infos dict, or ``obs`` itself (same object,
        which :meth:`_terminal_observations` uses to detect a symmetric setup)."""
        if self.privileged_obs_type is None:
            return obs
        candidate = source.get("observations", {}).get(self.privileged_obs_type)
        return candidate.to(self.device) if isinstance(candidate, torch.Tensor) else obs

    def _initial_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """First observations of a run, kept raw: the buffer outlives normalizer updates,
        so statistics are applied at sample time, not baked in here."""
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        critic_obs = self._privileged_obs(extras, obs)
        if self.empirical_normalization:
            self.obs_normalizer(obs)
            self.critic_obs_normalizer(critic_obs)
        return obs, critic_obs

    def _obs_for_policy(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize for action selection; stats already absorbed this obs as ``next_obs``."""
        if not self.empirical_normalization:
            return obs
        with torch.no_grad():
            return self.obs_normalizer.normalize(obs)

    def _episode_boundaries(
        self, infos: dict, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resolve the three kinds of episode boundary into storage flags.

        Termination cuts the bootstrap; truncation keeps it (the episode would have
        continued); pseudo-termination cuts it without ending the episode. Algorithms
        rebuild the mask as ``bootstrap + (1 - done)``, so a pseudo-terminal is encoded
        as done-without-bootstrap.
        """
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device).float()
        else:
            time_outs = torch.zeros_like(dones, dtype=torch.float32)

        pseudo = infos.get("pseudo_terminated")
        pseudo = torch.zeros_like(dones, dtype=torch.float32) if pseudo is None else pseudo.to(self.device).float()

        terminal = torch.clamp(dones * (1.0 - time_outs) + pseudo, max=1.0)
        if not self.uses_bootstrap_channel:
            return time_outs, terminal, terminal, None
        # Truncation bootstraps unless a pseudo-terminal landed on the same step.
        return time_outs, terminal, torch.clamp(dones.float() + pseudo, max=1.0), time_outs * (1.0 - pseudo)

    def _costs(self, infos: dict) -> torch.Tensor | None:
        """Per-env cost vector shaped ``[num_envs, num_costs]``, or None."""
        if not self.is_safe_rl or "costs" not in infos:
            return None
        costs = infos["costs"].to(self.device)
        if costs.dim() == 1:
            costs = costs.unsqueeze(-1)
        if costs.shape[-1] != self.alg.num_costs:
            costs = costs.expand(-1, self.alg.num_costs)
        return costs

    def _store_transition(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor | None,
        done: torch.Tensor,
        terminal: torch.Tensor,
        time_outs: torch.Tensor,
        bootstrap: torch.Tensor | None,
        next_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        behavior_log_prob: torch.Tensor | None = None,
        policy_version: torch.Tensor | None = None,
    ) -> None:
        """Write one transition to the replay buffer.

        Two storage paths, mutually exclusive by construction (``__init__`` rejects
        safe RL combined with runner-level n-step):

        - the n-step aggregator, which needs the raw ``terminal``/``truncated``
          split to know where to stop accumulating a return;
        - the algorithm's own ``store_transition``, which takes the already-resolved
          ``done`` plus, for algorithms that read the bootstrap channel, a separate
          timeout flag.
        """
        if self.n_step_buffer is not None:
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
            return

        kwargs: dict = {"critic_obs": critic_obs, "next_critic_obs": next_critic_obs}
        if self.is_safe_rl:
            kwargs["cost"] = costs
        if self.is_safe_rl or bootstrap is not None:
            kwargs["bootstrap"] = bootstrap
        if behavior_log_prob is not None:
            kwargs["behavior_log_prob"] = behavior_log_prob
        if policy_version is not None:
            kwargs["policy_version"] = policy_version
        self.alg.store_transition(obs, action, rewards, done, next_obs, **kwargs)

    def _terminal_observations(
        self,
        infos: dict,
        time_outs: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Observations to *store*, with truncations repaired.

        Vectorized envs auto-reset, so the obs returned with ``truncated=True`` belongs to
        the next episode; envs that can recover the real one publish
        ``infos["final_observation"]``. For storage only — the caller keeps the post-reset
        ``next_obs`` as the next step's policy input.
        """
        final_obs = infos.get("final_observation")
        if final_obs is None or not time_outs.any():
            if final_obs is None and time_outs.any() and not self._final_obs_warned:
                LOGGER.warning(
                    "the env truncated episodes but did not provide "
                    "infos['final_observation']. Q-bootstrap on truncation will use the "
                    "post-auto-reset observation, which belongs to the next episode. "
                    "Add final_observation forwarding to the env wrapper."
                )
                self._final_obs_warned = True
            return next_obs, next_critic_obs

        final_obs = final_obs.to(self.device)
        if final_obs.shape != next_obs.shape:
            # Vision runs publish the privileged state here, not an actor observation.
            if not self._final_obs_shape_warned:
                LOGGER.warning(
                    "infos['final_observation'] has shape %s but the actor observation "
                    "has shape %s; ignoring it and bootstrapping truncations from the "
                    "post-auto-reset observation.",
                    tuple(final_obs.shape),
                    tuple(next_obs.shape),
                )
                self._final_obs_shape_warned = True
            return next_obs, next_critic_obs

        mask = time_outs.bool().unsqueeze(-1)
        store_next_obs = torch.where(mask, final_obs, next_obs)
        # Only reusable when the critic obs IS the actor obs.
        store_next_critic_obs = store_next_obs if next_critic_obs is next_obs else next_critic_obs
        return store_next_obs, store_next_critic_obs
