from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.modules.sac_actor_critic import SACActorCritic
from safe_rl.storage.replay_storage import ReplayStorage


class SAC:
    """Soft Actor-Critic algorithm.

    Paper: https://arxiv.org/abs/1801.01290
    With automatic entropy tuning: https://arxiv.org/abs/1812.05905

    This implementation includes:
    - Twin Q-networks (double Q-learning)
    - Automatic entropy coefficient (alpha) tuning
    - Soft target updates (Polyak averaging)
    - Optional distributional critics (C51-style)
    """

    policy: SACActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy: SACActorCritic,
        # Learning rates
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # SAC parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float | None = None,
        # Training parameters
        batch_size: int = 256,
        num_updates_per_step: int = 1,
        policy_frequency: int = 1,
        max_grad_norm: float = 1.0,
        # Device
        device: str = "cpu",
        # Multi-GPU (for compatibility, not fully implemented for SAC)
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize SAC algorithm.

        Args:
            policy: SACActorCritic module.
            actor_lr: Learning rate for the actor.
            critic_lr: Learning rate for the critics.
            alpha_lr: Learning rate for the entropy coefficient.
            gamma: Discount factor.
            tau: Soft target update coefficient.
            alpha: Initial entropy regularization coefficient.
            auto_entropy_tuning: Whether to automatically tune alpha.
            target_entropy: Target entropy for auto-tuning. If None, uses -dim(action).
            batch_size: Mini-batch size for updates.
            num_updates_per_step: Number of gradient updates per environment step.
            policy_frequency: Frequency of actor/alpha updates relative to critic updates.
            max_grad_norm: Maximum gradient norm for clipping.
            device: Device to run on.
            multi_gpu_cfg: Multi-GPU configuration (for compatibility).
        """
        if kwargs:
            print(f"SAC.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        # SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_updates_per_step = num_updates_per_step
        self.policy_frequency = max(1, policy_frequency)
        self.max_grad_norm = max_grad_norm

        # Entropy coefficient (alpha)
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy: -dim(action) is a common heuristic
            if target_entropy is None:
                self.target_entropy = -policy.num_actions
            else:
                self.target_entropy = target_entropy
            # Log alpha for numerical stability
            self.log_alpha = torch.tensor([math.log(alpha)], requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor([alpha], device=device).log()
            self.target_entropy = None
            self.alpha_optimizer = None

        # Optimizers
        # Actor optimizer (only actor network parameters)
        actor_params = list(policy.actor.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)

        # Critic optimizer (both Q-networks)
        critic_params = list(policy.critic_1.parameters()) + list(policy.critic_2.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr)

        # Unified optimizer for compatibility
        self.optimizer = self.actor_optimizer

        # Storage (initialized later)
        self.storage: ReplayStorage | None = None

        # Multi-GPU (for compatibility)
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # For compatibility with PPO interface
        self.rnd = None
        self.learning_rate = actor_lr

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient."""
        return self.log_alpha.exp()

    def init_storage(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: list[int],
        act_shape: list[int],
    ) -> None:
        """Initialize the replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store.
            num_envs: Number of parallel environments.
            obs_shape: Shape of observations.
            act_shape: Shape of actions.
        """
        self.storage = ReplayStorage(
            num_envs=num_envs,
            max_size=buffer_size,
            obs_shape=obs_shape,
            action_shape=act_shape,
            device=self.device,
        )

    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        next_critic_obs: torch.Tensor | None = None,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: Current observations.
            action: Actions taken.
            reward: Rewards received.
            done: Done flags.
            next_obs: Next observations.
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage() first.")
        extras = {}
        if critic_obs is not None:
            extras["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            extras["next_critic_observations"] = next_critic_obs
        self.storage.add(obs, action, reward, done, next_obs, **extras)

    def update(self, obs_normalizer=None, critic_obs_normalizer=None, reward_normalizer=None) -> dict[str, float]:
        """Perform SAC update step.

        Args:
            obs_normalizer: Optional normalizer for actor observations.
                Applied at sample time (FastSAC-style) to keep raw obs in buffer.
            critic_obs_normalizer: Optional normalizer for critic observations.

        Returns:
            Dictionary containing loss values for logging.
        """
        if self.storage is None or len(self.storage) < self.batch_size:
            return {"critic": 0.0, "actor": 0.0, "alpha": 0.0}

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0

        actor_updates = 0

        for update_idx in range(self.num_updates_per_step):
            # Sample batch (raw obs from buffer)
            batch = self.storage.sample(self.batch_size)
            obs = batch["observations"]
            critic_obs = batch.get("critic_observations", obs)
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_observations"]
            next_critic_obs = batch.get("next_critic_observations", next_obs)

            # Normalize at sample time (FastSAC-style, eval mode to avoid updating stats)
            if obs_normalizer is not None:
                with torch.no_grad():
                    obs = (obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
                    next_obs = (next_obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
            if critic_obs_normalizer is not None:
                with torch.no_grad():
                    critic_obs = (critic_obs - critic_obs_normalizer._mean) / (critic_obs_normalizer._std + critic_obs_normalizer.eps)
                    next_critic_obs = (next_critic_obs - critic_obs_normalizer._mean) / (critic_obs_normalizer._std + critic_obs_normalizer.eps)
            # Normalize rewards by running std (don't shift mean)
            if reward_normalizer is not None:
                with torch.no_grad():
                    rewards = rewards / (reward_normalizer._std + reward_normalizer.eps)

            # Update critic
            critic_loss = self._update_critic(critic_obs, actions, rewards, dones, next_critic_obs)
            total_critic_loss += critic_loss

            # Update actor and alpha
            if update_idx % self.policy_frequency == 0:
                actor_loss, alpha_loss = self._update_actor_and_alpha(obs)
                total_actor_loss += actor_loss
                total_alpha_loss += alpha_loss
                actor_updates += 1

            # Soft update target networks
            self.policy.soft_update_targets(self.tau)

        # Average losses
        num_updates = self.num_updates_per_step
        actor_denominator = actor_updates if actor_updates > 0 else 1
        return {
            "critic": total_critic_loss / num_updates,
            "actor": total_actor_loss / actor_denominator,
            "alpha": total_alpha_loss / actor_denominator,
        }

    def _update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Update Q-networks.

        Args:
            obs: Current observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Done flags.
            next_obs: Next observations.

        Returns:
            Critic loss value.
        """
        if self.policy.is_distributional_critic:
            critic_loss = self._update_critic_distributional(obs, actions, rewards, dones, next_obs)
        else:
            critic_loss = self._update_critic_standard(obs, actions, rewards, dones, next_obs)

        return critic_loss

    def _update_critic_standard(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Update Q-networks with standard MSE loss.

        Args:
            obs: Current observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Done flags.
            next_obs: Next observations.

        Returns:
            Critic loss value.
        """
        with torch.no_grad():
            # Sample next actions and compute log probs
            next_actions, next_log_prob = self.policy.sample_with_log_prob(next_obs)

            # Compute target Q-values
            q1_target, q2_target = self.policy.evaluate_q_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # Soft Bellman backup
            # Q_target = r + γ * (1 - d) * (min Q_target - α * log π)
            target_q = rewards + self.gamma * (1 - dones) * (q_target - self.alpha.detach() * next_log_prob)

        # Compute current Q-values
        q1, q2 = self.policy.evaluate_q(obs, actions)

        # Critic loss (MSE with 0.5 factor per SAC paper)
        critic_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.critic_1.parameters()) + list(self.policy.critic_2.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_critic_distributional(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Update Q-networks with distributional (C51) cross-entropy loss.

        Args:
            obs: Current observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Done flags.
            next_obs: Next observations.

        Returns:
            Critic loss value.
        """
        # Squeeze to 1D for distributional critic: [batch, 1] -> [batch]
        rewards = rewards.squeeze(-1)
        bootstrap = (1.0 - dones).squeeze(-1)

        with torch.no_grad():
            # Sample next actions and compute log probs
            next_actions, next_log_prob = self.policy.sample_with_log_prob(next_obs)
            next_log_prob = next_log_prob.squeeze(-1)  # [batch, 1] -> [batch]

            # Modify rewards to include entropy bonus: r - γ * α * log π(a'|s')
            entropy_adjusted_rewards = rewards - self.gamma * bootstrap * self.alpha.detach() * next_log_prob

            # Normalize next_obs once (avoid redundant normalizer updates)
            next_obs_norm = self.policy.critic_obs_normalizer(next_obs)

            # Get target distributions from both critics
            logits_t1 = self.policy.critic_1_target(next_obs_norm, next_actions)
            logits_t2 = self.policy.critic_2_target(next_obs_norm, next_actions)
            dist_t1 = self.policy.critic_1_target.get_dist(logits_t1)
            dist_t2 = self.policy.critic_2_target.get_dist(logits_t2)

            # Double-Q trick: select distribution from the more pessimistic critic
            q1_val = self.policy.critic_1_target.get_value(dist_t1)
            q2_val = self.policy.critic_2_target.get_value(dist_t2)
            use_q1 = (q1_val < q2_val).unsqueeze(-1)
            min_dist = torch.where(use_q1, dist_t1, dist_t2)

            # Shared target distribution for both critics
            target_dist = self.policy.critic_1_target.project(
                next_dist=min_dist,
                rewards=entropy_adjusted_rewards,
                bootstrap=bootstrap,
                discount=self.gamma,
            )

        # Get current logits
        obs_normalized = self.policy.critic_obs_normalizer(obs)
        logits_1 = self.policy.critic_1(obs_normalized, actions)
        logits_2 = self.policy.critic_2(obs_normalized, actions)

        # Cross-entropy loss: -sum(target_dist * log_softmax(logits))
        log_probs_1 = F.log_softmax(logits_1, dim=-1)
        log_probs_2 = F.log_softmax(logits_2, dim=-1)

        critic_loss_1 = -torch.sum(target_dist * log_probs_1, dim=-1).mean()
        critic_loss_2 = -torch.sum(target_dist * log_probs_2, dim=-1).mean()
        critic_loss = critic_loss_1 + critic_loss_2

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.critic_1.parameters()) + list(self.policy.critic_2.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor_and_alpha(self, obs: torch.Tensor) -> tuple[float, float]:
        """Update actor and entropy coefficient.

        Args:
            obs: Current observations.

        Returns:
            Tuple of (actor_loss, alpha_loss).
        """
        # Sample actions and log probs for current policy
        actions, log_prob = self.policy.sample_with_log_prob(obs)

        # Compute Q-values for sampled actions
        q1, q2 = self.policy.evaluate_q(obs, actions)
        q_min = torch.min(q1, q2)

        # Actor loss: maximize Q - α * log π
        # Equivalently, minimize α * log π - Q
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(),
            self.max_grad_norm,
        )
        self.actor_optimizer.step()

        # Update entropy coefficient (alpha)
        alpha_loss_value = 0.0
        if self.auto_entropy_tuning and self.alpha_optimizer is not None:
            # Alpha loss: α * (-log π - target_entropy)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss_value = alpha_loss.item()

        return actor_loss.item(), alpha_loss_value

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (for multi-GPU training)."""
        if not self.is_multi_gpu:
            return

        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
