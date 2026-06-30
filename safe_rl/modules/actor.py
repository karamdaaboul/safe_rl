from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from safe_rl.networks import MLP, SimbaV2
from safe_rl.utils import resolve_nn_activation


def _build_actor_network(
    num_obs: int,
    num_actions: int,
    network_type: str,
    hidden_dims: list[int],
    activation: str,
    layer_norm: bool,
    network_kwargs: dict[str, Any] | None,
) -> nn.Module:
    """Build the actor trunk+head: MLP, or a faithful SimbaV2 (HyperTanhPolicy).

    For SimbaV2 the output head is ``HyperPredictor`` (scaler 1.0/1.0, no output
    L2-norm); the caller applies ``tanh`` on top, which together reproduce the
    reference ``HyperTanhPolicy``. ``hidden_dims``/``activation``/``layer_norm``
    are MLP-only and ignored for SimbaV2 (hyperspherical norm is built in).
    """
    if network_type == "mlp":
        return MLP(
            input_dim=num_obs,
            output_dim=num_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )
    if network_type == "simba":
        return SimbaV2(
            input_dim=num_obs,
            output_dim=num_actions,
            **(network_kwargs or {}),
        )
    raise ValueError(f"Unknown network_type: {network_type}. Must be 'mlp' or 'simba'.")


class DeterministicActor(nn.Module):
    """Deterministic actor network that outputs mean actions.

    Optionally supports per-environment exploration noise: if ``num_envs`` and
    a non-zero noise range are supplied, each env gets an independently
    sampled noise std from ``U[noise_std_min, noise_std_max]`` which is
    resampled on episode termination via :meth:`reset`.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        num_envs: int = 1,
        noise_std_min: float = 0.0,
        noise_std_max: float = 0.0,
        layer_norm: bool = False,
        network_type: str = "mlp",
        network_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "DeterministicActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.noise_std_min = float(noise_std_min)
        self.noise_std_max = float(noise_std_max)
        self.has_exploration_noise = self.noise_std_max > 0.0

        self.network = _build_actor_network(
            num_obs=num_obs,
            num_actions=num_actions,
            network_type=network_type,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            network_kwargs=network_kwargs,
        )

        # Per-env exploration noise scales, resampled on episode termination.
        init_scales = torch.empty(num_envs, 1).uniform_(self.noise_std_min, self.noise_std_max)
        self.register_buffer("noise_scales", init_scales)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Return tanh-bounded actions with per-env Gaussian exploration noise added."""
        actions = self.forward(obs)
        if self.has_exploration_noise and self.training:
            noise = torch.randn_like(actions) * self.noise_scales[: actions.shape[0]]
            actions = (actions + noise).clamp(-1.0, 1.0)
        return actions

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic tanh-bounded action without added noise."""
        return self.forward(obs)

    def as_onnx(self, pre_normalizer: nn.Module | None = None, actor_normalizer: nn.Module | None = None, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable wrapper: pre_normalizer → actor_normalizer → network."""
        return _OnnxDeterministicActor(self, pre_normalizer, actor_normalizer, verbose)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Resample exploration-noise scales for terminated environments."""
        if not self.has_exploration_noise or dones is None:
            return
        mask = dones.view(-1).bool()
        if not mask.any():
            return
        new_scales = torch.empty(int(mask.sum().item()), 1, device=self.noise_scales.device).uniform_(
            self.noise_std_min, self.noise_std_max
        )
        self.noise_scales[mask] = new_scales


class GaussianActor(nn.Module):
    """Gaussian policy with a state-independent learned σ (PPO-style).

    The mean is produced by an MLP over observations; σ is a free parameter
    shared across states — either a direct scalar (``noise_std_type='scalar'``)
    or parameterised in log-space (``'log'``).
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "GaussianActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type

        self.network = MLP(
            input_dim=num_obs,
            output_dim=num_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Must be 'scalar' or 'log'.")

        Normal.set_default_validate_args(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the policy mean for the given observations."""
        return self.network(obs)

    def action_std(self, mean: torch.Tensor) -> torch.Tensor:
        """Return σ expanded to match ``mean``'s shape."""
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def distribution(self, obs: torch.Tensor) -> Normal:
        """Build a Normal(mean(obs), σ) distribution."""
        mean = self.network(obs)
        return Normal(mean, self.action_std(mean))


class StochasticActor(nn.Module):
    """Stochastic actor network that outputs mean and log_std for a Gaussian policy."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_layer_norm: bool = False,
        zero_init_heads: bool = False,
        head_init: str = "default",
        init_noise_std: float = 1.0,
        log_std_squash: str = "clamp",
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StochasticActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = num_obs
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_squash = log_std_squash

        activation_fn = resolve_nn_activation(activation)

        # Backbone — inline rather than MLP: every hidden Linear needs a
        # [LayerNorm]+activation, including the final one feeding the heads.
        # MLP's last Linear skips LayerNorm, so it doesn't fit this shape.
        layers: list[nn.Module] = []
        input_dim = num_obs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn)
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(input_dim, num_actions)
        self.log_std_head = nn.Linear(input_dim, num_actions)

        # Back-compat: zero_init_heads=True is equivalent to head_init="zero".
        if zero_init_heads and head_init == "default":
            head_init = "zero"

        if head_init == "zero":
            nn.init.constant_(self.mean_head.weight, 0.0)
            nn.init.constant_(self.mean_head.bias, 0.0)
            nn.init.constant_(self.log_std_head.weight, 0.0)
            nn.init.constant_(self.log_std_head.bias, 0.0)
        elif head_init == "tuned":
            # Mirrors rsl_rl_sac SACActorModel: near-zero mean head so the initial
            # policy does not saturate tanh, and a log_std bias set to log(init_noise_std).
            nn.init.normal_(self.mean_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.mean_head.bias, 0.0)
            nn.init.constant_(self.log_std_head.weight, 0.0)
            nn.init.constant_(self.log_std_head.bias, math.log(init_noise_std + 1e-7))
        elif head_init != "default":
            raise ValueError(f"Unknown head_init '{head_init}'. Use 'default', 'zero', or 'tuned'.")

        # Per-joint bounded action scaling: a = b + c * tanh(x), with the matching
        # -sum(log c_j) Jacobian term in the log-prob. When bounds are unset, b=0/c=1
        # reduces exactly to plain tanh squashing in [-1, 1].
        self.scaled_actions = action_low is not None and action_high is not None
        if self.scaled_actions:
            low = torch.as_tensor(action_low, dtype=torch.float32)
            high = torch.as_tensor(action_high, dtype=torch.float32)
            if low.shape != (num_actions,) or high.shape != (num_actions,):
                raise ValueError(
                    f"action_low/action_high must have length num_actions={num_actions}, "
                    f"got {tuple(low.shape)} / {tuple(high.shape)}."
                )
            action_b = 0.5 * (high + low)
            action_c = 0.5 * (high - low)
        else:
            action_b = torch.zeros(num_actions)
            action_c = torch.ones(num_actions)
        self.register_buffer("action_b", action_b)
        self.register_buffer("action_c", action_c)
        # Scalar -sum(log c_j); c is clamped so a zero-range joint can't produce -inf.
        self.register_buffer("neg_log_action_scale", -action_c.clamp_min(1e-8).log().sum())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        if self.log_std_squash == "tanh":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        else:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities with tanh squashing."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = Normal(mean, std)
        x = dist.rsample()
        actions = self.action_b + self.action_c * torch.tanh(x)

        # Log probability with tanh squashing correction and the per-joint scale
        # Jacobian: -sum log(1 - tanh^2(x)) - sum log(c_j).
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (math.log(2.0) - x - nn.functional.softplus(-2 * x))).sum(
            dim=-1, keepdim=True
        )
        log_prob += self.neg_log_action_scale  # scalar; 0 when actions are unscaled

        return actions, log_prob

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return self.action_b + self.action_c * torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        return self.action_b + self.action_c * torch.tanh(x)

    def as_onnx(self, pre_normalizer: nn.Module | None = None, actor_normalizer: nn.Module | None = None, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable wrapper: pre_normalizer → actor_normalizer → backbone → tanh(mean)."""
        return _OnnxStochasticActor(self, pre_normalizer, actor_normalizer, verbose)


class _OnnxDeterministicActor(nn.Module):
    """ONNX-exportable deterministic actor (mirrors rsl_rl _OnnxMLPModel)."""

    def __init__(self, actor: DeterministicActor, pre_normalizer: nn.Module | None, actor_normalizer: nn.Module | None, verbose: bool) -> None:
        super().__init__()
        self.pre_normalizer = deepcopy(pre_normalizer) if pre_normalizer is not None else nn.Identity()
        self.actor_normalizer = deepcopy(actor_normalizer) if actor_normalizer is not None else nn.Identity()
        self.network = deepcopy(actor.network)
        self.input_size = actor.num_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.pre_normalizer(obs)
        obs = self.actor_normalizer(obs)
        return torch.tanh(self.network(obs))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _OnnxStochasticActor(nn.Module):
    """ONNX-exportable stochastic actor — exports deterministic (tanh mean) path."""

    def __init__(self, actor: StochasticActor, pre_normalizer: nn.Module | None, actor_normalizer: nn.Module | None, verbose: bool) -> None:
        super().__init__()
        self.pre_normalizer = deepcopy(pre_normalizer) if pre_normalizer is not None else nn.Identity()
        self.actor_normalizer = deepcopy(actor_normalizer) if actor_normalizer is not None else nn.Identity()
        self.backbone = deepcopy(actor.backbone)
        self.mean_head = deepcopy(actor.mean_head)
        self.input_size = actor.num_obs
        self.register_buffer("action_b", actor.action_b.detach().clone())
        self.register_buffer("action_c", actor.action_c.detach().clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.pre_normalizer(obs)
        obs = self.actor_normalizer(obs)
        return self.action_b + self.action_c * torch.tanh(self.mean_head(self.backbone(obs)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
