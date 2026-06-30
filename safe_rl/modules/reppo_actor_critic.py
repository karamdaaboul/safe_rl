from __future__ import annotations

from copy import deepcopy
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributions import Normal

from safe_rl.modules.actor import GaussianActor, StochasticActor
from safe_rl.modules.critic import DistributionalCritic, StandardCritic
from safe_rl.modules.normalizer import EmpiricalNormalization


class REPPOActorCritic(nn.Module):
    """Actor-Critic for REPPO with twin Q critics, target nets, and a target actor.

    Differs from SACActorCritic in two ways:
    - Defaults to ``GaussianActor`` (raw, no tanh squashing) — matches the
      paper's convention of unbounded sampling with env-side action clipping.
    - Adds an ``actor_target`` (deep copy, frozen) used to compute the
      bootstrap target ``a' ~ π_target(s')`` for the soft-Q λ-returns.

    Mirrors SACActorCritic's Q-evaluation API (``evaluate_q``,
    ``evaluate_q_target``, ``evaluate_q_dist``, ``soft_update_targets``) so the
    REPPO algorithm class can call into the same surface.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_type: str = "gaussian",
        critic_type: str = "distributional",
        num_critics: int = 2,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "REPPOActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        actor_kwargs = deepcopy(actor_kwargs) if actor_kwargs is not None else {}
        critic_kwargs = deepcopy(critic_kwargs) if critic_kwargs is not None else {}

        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self.actor_type = actor_type
        self.critic_type = critic_type
        self.num_critics = num_critics

        # Actor
        if actor_type == "gaussian":
            self.actor = GaussianActor(num_actor_obs, num_actions, **actor_kwargs)
        elif actor_type == "stochastic":
            self.actor = StochasticActor(num_actor_obs, num_actions, **actor_kwargs)
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}. REPPO supports 'gaussian' or 'stochastic'.")

        print(f"REPPO Actor: {self.actor}")

        # Frozen target actor for bootstrap a' ~ π_target(s')
        self.actor_target = deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # Twin Q critics
        if critic_type == "standard":
            self.is_distributional_critic = False
            critic_kwargs.setdefault("hidden_dims", [256, 256])
            critic_kwargs.setdefault("activation", "relu")
            self.critics = nn.ModuleList(
                StandardCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
                for _ in range(num_critics)
            )
        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critics = nn.ModuleList(
                DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
                for _ in range(num_critics)
            )
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'standard' or 'distributional'.")

        print(f"REPPO Critic: {self.critics[0]}")

        # Frozen Q targets
        self.critic_targets = nn.ModuleList(deepcopy(c) for c in self.critics)
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

        # Cached state during act() for runner compatibility
        self._distribution: Normal | None = None
        self._last_action: torch.Tensor | None = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_std(self) -> torch.Tensor:
        """Current σ — used by the runner for logging."""
        if self.actor_type == "gaussian":
            if self.actor.noise_std_type == "scalar":
                return self.actor.std
            return torch.exp(self.actor.log_std)
        # StochasticActor: state-dependent σ; expose the last cached one if available.
        if self._distribution is not None:
            return self._distribution.scale.detach().mean(dim=0)
        return torch.zeros(self.num_actions, device=next(self.parameters()).device)

    @property
    def action_mean(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("action_mean accessed before act()")
        return self._distribution.mean

    @property
    def distribution(self) -> Normal:
        if self._distribution is None:
            raise RuntimeError("distribution accessed before act()")
        return self._distribution

    # ---------------- Sampling ----------------

    def _build_distribution(self, obs: torch.Tensor, target: bool = False) -> Normal:
        actor = self.actor_target if target else self.actor
        if self.actor_type == "gaussian":
            mean = actor.network(obs)
            std = actor.action_std(mean)
        else:
            mean, log_std = actor.forward(obs)
            std = log_std.exp()
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs, target=False)
        self._distribution = dist
        if deterministic:
            self._last_action = dist.mean
        else:
            self._last_action = dist.sample()
        return self._last_action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs, target=False)
        return dist.mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("get_actions_log_prob called before act()")
        return self._distribution.log_prob(actions).sum(dim=-1)

    def sample_with_log_prob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample with gradient → (action, log_prob, mean, std)."""
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs, target=False)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.mean, dist.scale

    def target_sample_with_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """No-grad sample from the frozen target actor → (action, log_prob)."""
        with torch.no_grad():
            obs = self.actor_obs_normalizer(obs)
            dist = self._build_distribution(obs, target=True)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def current_distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs, target=False)
        return dist.mean, dist.scale

    # ---------------- Q evaluation ----------------

    def evaluate_q(
        self, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(critic_obs)
        if self.is_distributional_critic:
            logits_1 = self.critic_1(obs, actions)
            logits_2 = self.critic_2(obs, actions)
            dist_1 = self.critic_1.get_dist(logits_1)
            dist_2 = self.critic_2.get_dist(logits_2)
            q1 = self.critic_1.get_value(dist_1).unsqueeze(-1)
            q2 = self.critic_2.get_value(dist_2).unsqueeze(-1)
        else:
            q1 = self.critic_1(obs, actions)
            q2 = self.critic_2(obs, actions)
        return q1, q2

    def evaluate_q_target(
        self, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(critic_obs)
        if self.is_distributional_critic:
            logits_1 = self.critic_1_target(obs, actions)
            logits_2 = self.critic_2_target(obs, actions)
            dist_1 = self.critic_1_target.get_dist(logits_1)
            dist_2 = self.critic_2_target.get_dist(logits_2)
            q1 = self.critic_1_target.get_value(dist_1).unsqueeze(-1)
            q2 = self.critic_2_target.get_value(dist_2).unsqueeze(-1)
        else:
            q1 = self.critic_1_target(obs, actions)
            q2 = self.critic_2_target(obs, actions)
        return q1, q2

    def evaluate_q_dist(
        self, critic_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_dist only available for distributional critics")
        obs = self.critic_obs_normalizer(critic_obs)
        return self.critic_1(obs, actions), self.critic_2(obs, actions)

    # ---------------- Polyak updates ----------------

    def soft_update_targets(self, tau: float) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def soft_update_actor_target(self, tau: float) -> None:
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)
