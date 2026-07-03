from __future__ import annotations

import torch

from safe_rl.algorithms.ppo import PPO


class GRPO(PPO):
    """Group Relative Policy Optimization.

    Differences from PPO:
    - Reward-to-go returns (no GAE) when use_reward_to_go=True.
    - Advantages normalized per group (across parallel envs at each timestep) instead of globally.
    """

    def __init__(
        self,
        policy,
        use_reward_to_go: bool = True,
        group_size: int | None = None,
        normalize_advantage_per_group: bool = True,
        **ppo_kwargs,
    ):
        ppo_kwargs["normalize_advantage_per_mini_batch"] = False
        super().__init__(policy, **ppo_kwargs)
        self.use_reward_to_go = use_reward_to_go
        self.group_size = group_size
        self.normalize_advantage_per_group = normalize_advantage_per_group

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        if not self.use_reward_to_go:
            super().compute_returns(last_critic_obs)
            if not self.normalize_advantage_per_group:
                adv = self.storage.advantages
                self.storage.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
            return

        last_values = self.policy.evaluate(last_critic_obs).detach()
        returns = torch.zeros_like(self.storage.rewards)
        running_return = last_values

        for step in reversed(range(self.storage.num_transitions_per_env)):
            not_terminal = 1.0 - self.storage.dones[step].float()
            running_return = self.storage.rewards[step] + self.gamma * not_terminal * running_return
            returns[step] = running_return

        self.storage.returns = returns
        self.storage.advantages = returns - self.storage.values

        if not self.normalize_advantage_per_group:
            adv = self.storage.advantages
            self.storage.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def _apply_group_normalization(self) -> None:
        adv = self.storage.advantages  # [T, E, 1]
        T, E, _ = adv.shape
        group_size = self.group_size if self.group_size is not None else E

        if group_size >= E:
            mean = adv.mean(dim=1, keepdim=True)
            std = adv.std(dim=1, keepdim=True)
            self.storage.advantages = (adv - mean) / (std + 1e-8)
        else:
            num_groups = E // group_size
            tail = E % group_size
            adv_main = adv[:, : num_groups * group_size, :].view(T, num_groups, group_size, 1)
            mean = adv_main.mean(dim=2, keepdim=True)
            std = adv_main.std(dim=2, keepdim=True)
            adv[:, : num_groups * group_size, :] = ((adv_main - mean) / (std + 1e-8)).view(
                T, num_groups * group_size, 1
            )
            if tail > 0:
                adv_tail = adv[:, num_groups * group_size :, :]
                adv[:, num_groups * group_size :, :] = (adv_tail - adv_tail.mean(dim=1, keepdim=True)) / (
                    adv_tail.std(dim=1, keepdim=True) + 1e-8
                )
            self.storage.advantages = adv

    def update(self) -> dict[str, float]:
        if self.normalize_advantage_per_group:
            self._apply_group_normalization()
        return super().update()
