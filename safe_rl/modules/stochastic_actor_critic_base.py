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

from safe_rl.modules.action_bounds import (
    fill_missing_action_bound_buffers,
    register_action_bound_buffers,
    resolve_action_bounds,
    split_action_bound_kwargs,
    squash_unit,
    sync_scaled_flag,
    uniform_scale_or_none,
)
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
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
        action_bounds: float | list[float] | None = None,
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

        # Per-joint action box a = b + c * tanh(u): actions live in [b - c, b + c]
        # rather than (-1, 1). tanh caps |a| at 1, but mjlab locomotion applies no
        # action clipping and the AGXCave joint-velocity terms scale 0.1 rad/s per
        # unit, so an unbounded-Gaussian policy commands far more -- PPO reaches
        # -2.86..+4.46 on Unitree-Go2-Flat and spends 5-9 % of steps beyond |a| > 1
        # on the excavator arm. The affine is folded into the TransformedDistribution
        # so log_prob (hence entropy and the KL estimate) carries the exact
        # -sum_j log c_j term. `action_scale` is the legacy scalar spelling; the box
        # lives on the POLICY (this module), so bound spellings that arrive inside
        # actor_kwargs are lifted out and the inner actor stays a unit tanh actor.
        actor_kwargs = dict(actor_kwargs) if actor_kwargs else {}
        lifted = split_action_bound_kwargs(actor_kwargs)
        box = dict(action_low=action_low, action_high=action_high, action_bounds=action_bounds, action_scale=action_scale)
        for key, value in lifted.items():
            current = box.get(key)
            if current is not None and not (key == "action_scale" and float(current) == 1.0):
                raise ValueError(f"{key} given both on the policy and in actor_kwargs.")
            box[key] = value
        b, c, _ = resolve_action_bounds(num_actions, **box)
        register_action_bound_buffers(self, b, c)
        # Legacy attribute: the scalar half-width when the box is uniform and symmetric
        # (1.0 when unbounded), None for a genuinely per-joint box.
        self.action_scale = uniform_scale_or_none(self)
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
    # Action box
    # ------------------------------------------------------------------

    @property
    def sum_log_action_scale(self) -> float:
        """``sum_j log c_j``: the differential-entropy offset of the widened box (0 when unbounded)."""
        return -float(self.neg_log_action_scale)

    def squash_unit_action(self, u: torch.Tensor) -> torch.Tensor:
        """Map a pre-squash value onto the action box: ``b + c * tanh(u)``.

        (Named to avoid the ``squash`` string attribute that selects 'none' / 'tanh'.)
        """
        return squash_unit(self, u)

    def as_onnx(self, obs_normalizer: nn.Module | None = None, verbose: bool = False) -> nn.Module:
        """ONNX/JIT-exportable deterministic path: obs_normalizer -> actor_obs_normalizer -> mean -> box."""
        return _OnnxSquashedPolicy(self, obs_normalizer, verbose)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before the box buffers keep the configured box (which for a
        # legacy scalar action_scale is the same range); a checkpoint carrying buffers
        # pins its own box.
        fill_missing_action_bound_buffers(self, state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        sync_scaled_flag(self)
        self.action_scale = uniform_scale_or_none(self)

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
        """tanh-transform the base Normal, then map onto the action box.

        Unbounded, this is exactly the reference distribution. The per-joint
        AffineTransform has event_dim 0, so its log-det-Jacobian is ``log c_j`` per
        dimension and callers' ``.sum(dim=-1)`` yields ``sum_j log c_j``.
        """
        transforms: list[Any] = [TanhTransform(cache_size=1)]
        if self.scaled_actions:
            transforms.append(AffineTransform(loc=self.action_b, scale=self.action_c))
        return TransformedDistribution(base, transforms)

    def _clamp_squashed(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip before log_prob so the inverse (atanh) stays finite.

        The reference does `pi.log_prob(actions.clip(-1 + 1e-6, 1 - 1e-6))`; the
        clip is applied in unit space and mapped back through the box.
        """
        bound = 1.0 - 1e-6
        if not self.scaled_actions:
            return actions.clamp(-bound, bound)
        unit = ((actions - self.action_b) / self.action_c).clamp(-bound, bound)
        return self.action_b + self.action_c * unit

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
            action = self.squash_unit_action(action)
        self._last_action = action
        return action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation/deployment."""
        obs = self.actor_obs_normalizer(obs)
        dist = self._build_distribution(obs)
        if self.squash == "tanh":
            return self.squash_unit_action(dist.mean)
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


class _OnnxSquashedPolicy(nn.Module):
    """Deterministic export path of a :class:`StochasticActorCriticBase` policy.

    obs -> obs_normalizer -> actor_obs_normalizer -> actor mean -> action box. Feed-forward
    only, so it can be traced for ONNX or TorchScript; the box buffers travel with it.
    """

    def __init__(self, policy: StochasticActorCriticBase, obs_normalizer: nn.Module | None, verbose: bool) -> None:
        super().__init__()
        from copy import deepcopy

        self.verbose = verbose
        self.pre_normalizer = deepcopy(obs_normalizer) if obs_normalizer is not None else nn.Identity()
        self.actor_normalizer = deepcopy(policy.actor_obs_normalizer)
        self.actor = deepcopy(policy.actor)
        self.actor_type = policy.actor_type
        self.apply_tanh = policy.squash == "tanh"
        self.input_size = policy.num_actor_obs
        self.register_buffer("action_b", policy.action_b.detach().clone())
        self.register_buffer("action_c", policy.action_c.detach().clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.actor_normalizer(self.pre_normalizer(obs))
        if self.actor_type == "gaussian":
            mean = self.actor.network(obs)
        else:
            mean, _ = self.actor.forward(obs)
        if not self.apply_tanh:
            return mean
        return self.action_b + self.action_c * torch.tanh(mean)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
