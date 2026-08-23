"""Shared actor + normalizer scaffolding for the stochastic actor-critic modules.

`REPPOActorCritic` and `SACActorCritic` had drifted into two copies of the same
actor construction, observation normalization, and (tanh-)squashed distribution
handling, differing only in defaults. That duplication is what let bugs land in
one and not the other. This base owns exactly the parts they genuinely share:

  * actor construction (`gaussian` | `stochastic`) and the actor obs normalizer
  * the critic obs normalizer
  * the action distribution: base Normal, optional additive `min_std`, optional
    tanh squashing widened to (-`action_scale`, +`action_scale`)
  * the act / sample / log-prob surface built on that distribution

It deliberately owns **no critic**. The two subclasses disagree about the critic
in ways that are algorithmic, not incidental: REPPO has a single critic and no
target network (matching the reference implementation), while SAC needs twin
critics with polyak targets because it is off-policy. Forcing a common critic
layout here would misrepresent both.
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributions import AffineTransform, Normal, TanhTransform, TransformedDistribution

from safe_rl.modules.actor import GaussianActor, StochasticActor
from safe_rl.modules.normalizer import EmpiricalNormalization


class StochasticActorCriticBase(nn.Module):
    """Actor, observation normalizers, and the squashed action distribution."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_type: str = "stochastic",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        min_std: float = 0.0,
        squash: str = "none",
        action_scale: float = 1.0,
        normalizer_eps_mode: str = "add_var",
        actor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.actor_type = actor_type

        # Additive std floor (reference REPPO: std = exp(log_std) + min_std).
        self.min_std = float(min_std)

        if squash not in ("none", "tanh"):
            raise ValueError(f"squash must be 'none' or 'tanh'; got {squash!r}")
        self.squash = squash

        # Half-width of the squashed action range: actions live in
        # (-action_scale, +action_scale) rather than (-1, 1). tanh caps |a| at 1,
        # but mjlab locomotion applies no action clipping, so an unbounded-Gaussian
        # policy commands far more -- measured on Unitree-Go2-Flat, PPO reaches
        # -2.86..+4.46 and spends 61-67% of steps beyond |a| > 1 on the four calf
        # joints. The affine is folded into the TransformedDistribution so log_prob
        # (hence entropy and the KL estimate) carries the exact -n*log(scale) term.
        self.action_scale = float(action_scale)
        if self.action_scale <= 0.0:
            raise ValueError(f"action_scale must be > 0; got {action_scale!r}")

        actor_kwargs = dict(actor_kwargs) if actor_kwargs else {}
        if actor_type == "gaussian":
            self.actor = GaussianActor(num_actor_obs, num_actions, **actor_kwargs)
        elif actor_type == "stochastic":
            self.actor = StochasticActor(num_actor_obs, num_actions, **actor_kwargs)
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}. Must be 'gaussian' or 'stochastic'.")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs, eps_mode=normalizer_eps_mode)
            if actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs, eps_mode=normalizer_eps_mode)
            if critic_obs_normalization
            else nn.Identity()
        )

        self._distribution: Normal | None = None
        self._last_action: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Runner compatibility
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def distribution(self) -> Normal | None:
        """Base (pre-squash) Normal cached by the last `act` — (mu, sigma) storage reads this."""
        return self._distribution

    @property
    def action_mean(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("action_mean requested before act()")
        return self._distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("action_std requested before act()")
        return self._distribution.stddev

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)

    # ------------------------------------------------------------------
    # Action distribution
    # ------------------------------------------------------------------

    def _build_distribution(self, obs: torch.Tensor) -> Normal:
        # GaussianActor carries a state-INDEPENDENT sigma and its forward returns
        # only the mean; StochasticActor is state-dependent and returns
        # (mean, log_std).
        if self.actor_type == "gaussian":
            mean = self.actor.network(obs)
            std = self.actor.action_std(mean)
        else:
            mean, log_std = self.actor.forward(obs)
            std = log_std.exp()
        if self.min_std > 0.0:
            std = std + self.min_std
        return Normal(mean, std)

    def squashed(self, base: Normal) -> TransformedDistribution:
        """tanh-transform the base Normal, then rescale to the action range.

        With `action_scale == 1` this is exactly the reference distribution.
        """
        transforms: list[Any] = [TanhTransform(cache_size=1)]
        if self.action_scale != 1.0:
            transforms.append(AffineTransform(loc=0.0, scale=self.action_scale))
        return TransformedDistribution(base, transforms)

    def _clamp_squashed(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip before log_prob so the inverse (atanh) stays finite.

        The reference does `pi.log_prob(actions.clip(-1 + 1e-6, 1 - 1e-6))`; the
        bound moves with the action scale.
        """
        bound = self.action_scale * (1.0 - 1e-6)
        return actions.clamp(-bound, bound)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, deterministic: bool = False, normalized: bool = False) -> torch.Tensor:
        """Sample an action.

        `normalized=True` means the caller already applied `actor_obs_normalizer`.
        REPPO normalizes once at collection and stores the result, so the KL at
        update time is measured on identical inputs; re-normalizing later with
        statistics that moved during the rollout manufactures KL out of input drift.
        """
        if not normalized:
            obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs)
        self._distribution = dist
        action = dist.mean if deterministic else dist.sample()
        if self.squash == "tanh":
            action = self.action_scale * torch.tanh(action)
        self._last_action = action
        return action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation/deployment."""
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs)
        if self.squash == "tanh":
            return self.action_scale * torch.tanh(dist.mean)
        return dist.mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("get_actions_log_prob called before act()")
        if self.squash == "tanh":
            td = self.squashed(self._distribution)
            return td.log_prob(self._clamp_squashed(actions)).sum(dim=-1)
        return self._distribution.log_prob(actions).sum(dim=-1)

    def sample_with_log_prob(
        self, obs: torch.Tensor, normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample -> (action, log_prob, base mean, base std)."""
        if not normalized:
            obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs)
        if self.squash == "tanh":
            td = self.squashed(dist)
            action = td.rsample()
            log_prob = td.log_prob(self._clamp_squashed(action)).sum(dim=-1)
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.mean, dist.scale
