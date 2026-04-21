"""FastTD3 — Twin Delayed DDPG with optional distributional critics.

Port of the TD3 variant from Seo et al., "Learning Sim-to-Real Humanoid
Locomotion in 15 Minutes" (arXiv:2512.01996), adapted to safe_rl's interfaces.

Key features:
- Twin critics (standard scalar Q or C51 distributional)
- Delayed actor + soft-target update every ``policy_frequency`` critic steps
- Target policy smoothing (clipped Gaussian noise on target actions)
- ``num_updates_per_step`` gradient updates per collection (UTD ratio)
- AdamW, AMP, vectorized Polyak averaging
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from safe_rl.modules.reward_normalization import RewardNormalization
from safe_rl.modules.td3_actor_critic import TD3ActorCritic
from safe_rl.storage.replay_storage import ReplayStorage


@torch.compile
def _critic_loss_standard(
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    smoothing_noise: float,
    noise_clip: float,
    action_low: float,
    action_high: float,
    clipped_double_q: bool,
    actor: nn.Module,
    critic_1: nn.Module,
    critic_2: nn.Module,
    target_1: nn.Module,
    target_2: nn.Module,
    actor_obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
) -> torch.Tensor:
    with torch.no_grad():
        next_mean = actor(actor_obs_normalizer(next_obs))
        noise = torch.randn_like(next_mean).mul(smoothing_noise).clamp(-noise_clip, noise_clip)
        next_actions = (next_mean + noise).clamp(action_low, action_high)

        next_obs_norm = critic_obs_normalizer(next_obs)
        q1_t = target_1(next_obs_norm, next_actions)
        q2_t = target_2(next_obs_norm, next_actions)
        q_t = torch.min(q1_t, q2_t) if clipped_double_q else 0.5 * (q1_t + q2_t)
        target_q = rewards + gamma * (1.0 - dones) * q_t

    obs_norm = critic_obs_normalizer(obs)
    q1 = critic_1(obs_norm, actions)
    q2 = critic_2(obs_norm, actions)
    return 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)


@torch.compile
def _critic_loss_distributional(
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    smoothing_noise: float,
    noise_clip: float,
    action_low: float,
    action_high: float,
    clipped_double_q: bool,
    actor: nn.Module,
    critic_1: nn.Module,
    critic_2: nn.Module,
    target_1: nn.Module,
    target_2: nn.Module,
    actor_obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
) -> torch.Tensor:
    rewards_1d = rewards.squeeze(-1)
    bootstrap = (1.0 - dones).squeeze(-1)

    with torch.no_grad():
        next_mean = actor(actor_obs_normalizer(next_obs))
        noise = torch.randn_like(next_mean).mul(smoothing_noise).clamp(-noise_clip, noise_clip)
        next_actions = (next_mean + noise).clamp(action_low, action_high)

        next_obs_norm = critic_obs_normalizer(next_obs)
        logits_t1 = target_1(next_obs_norm, next_actions)
        logits_t2 = target_2(next_obs_norm, next_actions)
        dist_t1 = target_1.get_dist(logits_t1)
        dist_t2 = target_2.get_dist(logits_t2)

        if clipped_double_q:
            q1_val = target_1.get_value(dist_t1)
            q2_val = target_2.get_value(dist_t2)
            use_q1 = (q1_val < q2_val).unsqueeze(-1)
            min_dist = torch.where(use_q1, dist_t1, dist_t2)
        else:
            min_dist = 0.5 * (dist_t1 + dist_t2)

        target_dist = target_1.project(
            next_dist=min_dist,
            rewards=rewards_1d,
            bootstrap=bootstrap,
            discount=gamma,
        )

    obs_norm = critic_obs_normalizer(obs)
    logits_1 = critic_1(obs_norm, actions)
    logits_2 = critic_2(obs_norm, actions)

    log_probs_1 = F.log_softmax(logits_1, dim=-1)
    log_probs_2 = F.log_softmax(logits_2, dim=-1)
    loss_1 = -torch.sum(target_dist * log_probs_1, dim=-1).mean()
    loss_2 = -torch.sum(target_dist * log_probs_2, dim=-1).mean()
    return loss_1 + loss_2


@torch.compile
def _actor_loss_fn(
    obs: torch.Tensor,
    is_distributional: bool,
    clipped_double_q: bool,
    actor: nn.Module,
    critic_1: nn.Module,
    critic_2: nn.Module,
    actor_obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
) -> torch.Tensor:
    actions = actor(actor_obs_normalizer(obs))
    obs_norm = critic_obs_normalizer(obs)

    if is_distributional:
        logits_1 = critic_1(obs_norm, actions)
        logits_2 = critic_2(obs_norm, actions)
        dist_1 = critic_1.get_dist(logits_1)
        dist_2 = critic_2.get_dist(logits_2)
        q1 = critic_1.get_value(dist_1)
        q2 = critic_2.get_value(dist_2)
    else:
        q1 = critic_1(obs_norm, actions).squeeze(-1)
        q2 = critic_2(obs_norm, actions).squeeze(-1)

    q = torch.min(q1, q2) if clipped_double_q else 0.5 * (q1 + q2)
    return -q.mean()


class FastTD3:
    """TD3 with the FastSAC performance suite (AdamW, AMP, torch.compile)."""

    policy: TD3ActorCritic

    def __init__(
        self,
        policy: TD3ActorCritic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        num_updates_per_step: int = 1,
        policy_frequency: int = 2,
        n_step: int = 1,
        smoothing_noise: float = 0.2,
        noise_clip: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        weight_decay: float = 0.001,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        clipped_double_q: bool = True,
        amp: bool = False,
        amp_dtype: str = "bf16",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"FastTD3.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        self.gamma = gamma
        self.n_step = max(1, int(n_step))
        self.bellman_gamma = gamma ** self.n_step
        self.tau = tau
        self.batch_size = batch_size
        self.num_updates_per_step = num_updates_per_step
        self.policy_frequency = max(1, policy_frequency)
        self.smoothing_noise = smoothing_noise
        self.noise_clip = noise_clip
        self.action_low = action_low
        self.action_high = action_high
        self.clipped_double_q = clipped_double_q

        self.amp = amp
        self.amp_device = "cuda" if "cuda" in self.device else "cpu"
        if amp_dtype in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        elif amp_dtype in ("f16", "float16"):
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(enabled=self.amp and self.amp_dtype == torch.float16)

        self.actor_optimizer = optim.AdamW(
            policy.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            betas=adam_betas,
        )
        critic_params = list(policy.critic_1.parameters()) + list(policy.critic_2.parameters())
        self.critic_optimizer = optim.AdamW(
            critic_params,
            lr=critic_lr,
            weight_decay=weight_decay,
            betas=adam_betas,
        )
        self.optimizer = self.actor_optimizer

        self.storage: ReplayStorage | None = None

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.learning_rate = actor_lr
        self.rnd = None

    def init_storage(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: list[int],
        act_shape: list[int],
    ) -> None:
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
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage() first.")
        extras = {}
        if critic_obs is not None:
            extras["critic_observations"] = critic_obs
        if next_critic_obs is not None:
            extras["next_critic_observations"] = next_critic_obs
        self.storage.add(obs, action, reward, done, next_obs, **extras)

    def update(
        self,
        obs_normalizer=None,
        critic_obs_normalizer=None,
        reward_normalizer=None,
    ) -> dict[str, float]:
        if self.storage is None or len(self.storage) < self.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "noise_std": 0.0}

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        actor_updates = 0

        for update_idx in range(self.num_updates_per_step):
            batch = self.storage.sample(self.batch_size)
            obs = batch["observations"]
            critic_obs = batch.get("critic_observations", obs)
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_observations"]
            next_critic_obs = batch.get("next_critic_observations", next_obs)

            if obs_normalizer is not None:
                with torch.no_grad():
                    obs = (obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
                    next_obs = (next_obs - obs_normalizer._mean) / (obs_normalizer._std + obs_normalizer.eps)
            if critic_obs_normalizer is not None:
                with torch.no_grad():
                    critic_obs = (critic_obs - critic_obs_normalizer._mean) / (
                        critic_obs_normalizer._std + critic_obs_normalizer.eps
                    )
                    next_critic_obs = (next_critic_obs - critic_obs_normalizer._mean) / (
                        critic_obs_normalizer._std + critic_obs_normalizer.eps
                    )
            if reward_normalizer is not None:
                with torch.no_grad():
                    if isinstance(reward_normalizer, RewardNormalization):
                        rewards = reward_normalizer(rewards)
                    else:
                        rewards = rewards / (reward_normalizer._std + reward_normalizer.eps)

            critic_loss = self._update_critic(
                obs, critic_obs, actions, rewards, dones, next_obs, next_critic_obs
            )
            total_critic_loss += critic_loss

            if update_idx % self.policy_frequency == 0:
                actor_loss = self._update_actor(obs)
                total_actor_loss += actor_loss
                actor_updates += 1
                self._soft_update()

        num_updates = self.num_updates_per_step
        actor_denom = max(actor_updates, 1)

        noise_std = self.policy.actor.noise_scales.mean().item() if hasattr(self.policy.actor, "noise_scales") else 0.0

        return {
            "critic_loss": total_critic_loss / num_updates,
            "actor_loss": total_actor_loss / actor_denom,
            "noise_std": noise_std,
        }

    def _update_critic(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
    ) -> float:
        with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.amp):
            if self.policy.is_distributional_critic:
                loss = _critic_loss_distributional(
                    critic_obs, actions, rewards, dones, next_critic_obs,
                    self.bellman_gamma, self.smoothing_noise, self.noise_clip,
                    self.action_low, self.action_high, self.clipped_double_q,
                    self.policy.actor, self.policy.critic_1, self.policy.critic_2,
                    self.policy.critic_1_target, self.policy.critic_2_target,
                    self.policy.actor_obs_normalizer, self.policy.critic_obs_normalizer,
                )
            else:
                loss = _critic_loss_standard(
                    critic_obs, actions, rewards, dones, next_critic_obs,
                    self.bellman_gamma, self.smoothing_noise, self.noise_clip,
                    self.action_low, self.action_high, self.clipped_double_q,
                    self.policy.actor, self.policy.critic_1, self.policy.critic_2,
                    self.policy.critic_1_target, self.policy.critic_2_target,
                    self.policy.actor_obs_normalizer, self.policy.critic_obs_normalizer,
                )

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        return loss.item()

    def _update_actor(self, obs: torch.Tensor) -> float:
        with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.amp):
            actor_loss = _actor_loss_fn(
                obs, self.policy.is_distributional_critic, self.clipped_double_q,
                self.policy.actor, self.policy.critic_1, self.policy.critic_2,
                self.policy.actor_obs_normalizer, self.policy.critic_obs_normalizer,
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return actor_loss.item()

    @torch.no_grad()
    def _soft_update(self) -> None:
        for critic, target in zip(self.policy.critics, self.policy.critic_targets):
            src_ps = [p.data for p in critic.parameters()]
            tgt_ps = [p.data for p in target.parameters()]
            torch._foreach_mul_(tgt_ps, 1.0 - self.tau)
            torch._foreach_add_(tgt_ps, src_ps, alpha=self.tau)

            for (_, b_s), (_, b_t) in zip(
                critic.named_buffers(), target.named_buffers(), strict=False
            ):
                b_t.copy_(b_s)

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def get_actual_action_std(self) -> float:
        if hasattr(self.policy.actor, "noise_scales"):
            return self.policy.actor.noise_scales.mean().item()
        return 0.0
