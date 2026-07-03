from __future__ import annotations

from copy import deepcopy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.actor import DeterministicActor
from safe_rl.modules.critic import DistributionalCritic, StandardCritic
from safe_rl.modules.normalizer import EmpiricalNormalization


class TD3ActorCritic(nn.Module):
    """Actor-Critic module for TD3 / FastTD3.

    - Deterministic actor with optional per-env exploration noise
    - Twin critics (standard scalar Q or distributional C51)
    - Frozen target networks updated via Polyak averaging
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_envs: int = 1,
        critic_type: str = "standard",
        num_critics: int = 2,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "TD3ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        actor_kwargs = deepcopy(actor_kwargs) if actor_kwargs is not None else {}
        critic_kwargs = deepcopy(critic_kwargs) if critic_kwargs is not None else {}

        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self.num_envs = num_envs
        self.actor_type = "deterministic"
        self.critic_type = critic_type
        self.num_critics = num_critics

        # -------- Actor --------
        actor_kwargs.setdefault("num_envs", num_envs)
        self.actor = DeterministicActor(
            num_obs=num_actor_obs,
            num_actions=num_actions,
            **actor_kwargs,
        )
        print(f"TD3 Actor: {self.actor}")

        # Exposed for runner logging compatibility (uses mean of noise_scales).
        self.std = nn.Parameter(
            self.actor.noise_scales.detach().mean().expand(num_actions).clone(),
            requires_grad=False,
        )

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # -------- Critics --------
        if critic_type == "standard":
            self.is_distributional_critic = False
            critic_kwargs.setdefault("hidden_dims", [256, 256])
            critic_kwargs.setdefault("activation", "relu")
            self.critics = nn.ModuleList(
                StandardCritic(
                    num_obs=num_critic_obs,
                    num_actions=num_actions,
                    **critic_kwargs,
                )
                for _ in range(num_critics)
            )
        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critics = nn.ModuleList(
                DistributionalCritic(
                    num_obs=num_critic_obs,
                    num_actions=num_actions,
                    **critic_kwargs,
                )
                for _ in range(num_critics)
            )
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'standard' or 'distributional'.")
        print(f"TD3 Critic: {self.critics[0]}")

        self.critic_targets = nn.ModuleList(deepcopy(critic) for critic in self.critics)
        for target in self.critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        self.critic_1 = self.critics[0]
        self.critic_2 = self.critics[1] if num_critics > 1 else self.critics[0]
        self.critic_1_target = self.critic_targets[0]
        self.critic_2_target = self.critic_targets[1] if num_critics > 1 else self.critic_targets[0]

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.actor.reset(dones)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = self.actor_obs_normalizer(obs)
        if deterministic:
            return self.actor.act_inference(obs)
        return self.actor.act(obs)

    def act_with_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(obs, deterministic=False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(obs, deterministic=True)

    def soft_update_targets(self, tau: float) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_random_action(self, num_envs: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.rand(num_envs, self.num_actions, device=device) * 2 - 1

    def as_onnx(self, obs_normalizer: nn.Module | None = None, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable actor: obs_normalizer → actor_obs_normalizer → network."""
        return self.actor.as_onnx(
            pre_normalizer=obs_normalizer,
            actor_normalizer=self.actor_obs_normalizer,
            verbose=verbose,
        )

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)
