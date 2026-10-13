"""Actor-critic with Regularized Online Adaptation (ROA).

Port of the adaptation scheme in *Deep Whole-Body Control* (Fu, Cheng, Pathak, CoRL 2022,
https://github.com/MarkFzp/Deep-Whole-Body-Control). The actor conditions on a latent of the
environment's physical parameters, produced by one of two encoders:

* ``priv_encoder`` (teacher): MLP over the privileged parameters, ``z = mu(e)``;
* ``history_encoder`` (student): 1-D CNN over the proprioceptive history, ``z_hat = phi(h)``.

Both are trained in a single phase. PPO adds ``lambda * ||z - sg(z_hat)||`` so the teacher latent
stays reachable from history (regularize), and the student is regressed onto ``sg(z)`` on the
iterations where it drives the rollout (supervise). Deployment uses ``z_hat`` only.

The actor observation is the concatenation ``[obs | priv | history]`` built by
:class:`~safe_rl.envs.roa_obs_wrapper.ROAObsWrapper`; the critic is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from safe_rl.modules.actor_critic import ActorCritic
from safe_rl.modules.normalizer import EmpiricalNormalization
from safe_rl.utils import resolve_nn_activation

# (kernel, stride) per conv layer for each supported history length, as in the reference.
_HISTORY_CONV_LAYOUTS = {
    10: ((4, 2), (2, 1)),
    20: ((6, 2), (4, 2)),
    50: ((8, 4), (5, 1), (5, 1)),
}


class StateHistoryEncoder(nn.Module):
    """1-D temporal CNN over ``[batch, hist_len, hist_dim]`` from the DWBC reference."""

    def __init__(self, hist_dim: int, hist_len: int, output_size: int, activation: str = "elu", channel_size: int = 10):
        super().__init__()
        if hist_len not in _HISTORY_CONV_LAYOUTS:
            raise ValueError(f"hist_len must be one of {sorted(_HISTORY_CONV_LAYOUTS)}; got {hist_len}")
        self.hist_len = hist_len
        self.hist_dim = hist_dim

        self.encoder = nn.Sequential(nn.Linear(hist_dim, 3 * channel_size), resolve_nn_activation(activation))
        channels = (3 * channel_size, 2 * channel_size) + (channel_size,) * (len(_HISTORY_CONV_LAYOUTS[hist_len]) - 1)
        conv_layers: list[nn.Module] = []
        for (kernel, stride), c_in, c_out in zip(_HISTORY_CONV_LAYOUTS[hist_len], channels[:-1], channels[1:]):
            conv_layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride),
                resolve_nn_activation(activation),
            ]
        conv_layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*conv_layers)
        # Every layout above leaves 3 time steps of `channel_size` channels.
        self.linear_output = nn.Sequential(nn.Linear(3 * channel_size, output_size), resolve_nn_activation(activation))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch = history.shape[0]
        projection = self.encoder(history.reshape(batch * self.hist_len, self.hist_dim))
        features = self.conv_layers(projection.reshape(batch, self.hist_len, -1).permute(0, 2, 1))
        return self.linear_output(features)


class ActorCriticROA(ActorCritic):
    """Gaussian actor on ``[obs, z]`` with a privileged teacher encoder and a history student encoder."""

    requires_roa_obs = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_prop: int,
        num_priv: int,
        hist_len: int,
        hist_dim: int,
        priv_encoder_dims: tuple[int, ...] | list[int] = (64, 20),
        encoder_activation: str = "elu",
        history_channel_size: int = 10,
        actor_type: str = "gaussian",
        actor_obs_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the ROA actor-critic.

        Args:
            num_actor_obs: Width of the composed actor observation ``num_prop + num_priv + hist_len * hist_dim``.
            num_critic_obs: Width of the critic observation.
            num_actions: Action dimension.
            num_prop: Width of the current (deployable) actor observation.
            num_priv: Width of the privileged physical-parameter vector.
            hist_len: Number of history steps (10, 20 or 50).
            hist_dim: Width of one history step.
            priv_encoder_dims: Hidden/output sizes of the teacher MLP; the last entry is the latent size.
            encoder_activation: Activation used inside both encoders.
            history_channel_size: Base channel count of the history CNN.
            actor_type: Only ``"gaussian"`` is supported.
            actor_obs_normalization: Normalize the composed actor observation before splitting it.
        """
        if actor_type != "gaussian":
            raise ValueError("ActorCriticROA supports actor_type='gaussian' only.")
        expected = num_prop + num_priv + hist_len * hist_dim
        if num_actor_obs != expected:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} does not match num_prop + num_priv + hist_len * hist_dim = {expected}"
            )
        priv_encoder_dims = tuple(priv_encoder_dims)
        latent_dim = priv_encoder_dims[-1] if priv_encoder_dims else num_priv

        # The actor MLP sees [obs, z]; normalization applies to the composed input, so it is built here.
        super().__init__(
            num_actor_obs=num_prop + latent_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_type=actor_type,
            actor_obs_normalization=False,
            **kwargs,
        )
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)

        self.num_prop = num_prop
        self.num_priv = num_priv
        self.hist_len = hist_len
        self.hist_dim = hist_dim
        self.latent_dim = latent_dim

        if priv_encoder_dims:
            layers: list[nn.Module] = []
            for d_in, d_out in zip((num_priv,) + priv_encoder_dims[:-1], priv_encoder_dims):
                layers += [nn.Linear(d_in, d_out), resolve_nn_activation(encoder_activation)]
            self.priv_encoder = nn.Sequential(*layers)
        else:
            self.priv_encoder = nn.Identity()
        self.history_encoder = StateHistoryEncoder(
            hist_dim, hist_len, latent_dim, activation=encoder_activation, channel_size=history_channel_size
        )
        print(f"Privileged encoder: {self.priv_encoder}")
        print(f"History encoder: {self.history_encoder}")

        # Rollout mode: True on adaptation iterations, where the student latent drives the actor.
        self.use_history_latent = False
        self._last_obs: torch.Tensor | None = None
        self._last_priv_latent: torch.Tensor | None = None

    def _split(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prop = observations[:, : self.num_prop]
        priv = observations[:, self.num_prop : self.num_prop + self.num_priv]
        history = observations[:, self.num_prop + self.num_priv :].reshape(-1, self.hist_len, self.hist_dim)
        return prop, priv, history

    def infer_priv_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Teacher latent ``z`` from already-normalized observations."""
        return self.priv_encoder(self._split(observations)[1])

    def infer_hist_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Student latent ``z_hat`` from already-normalized observations."""
        return self.history_encoder(self._split(observations)[2])

    def _actor_input(self, observations: torch.Tensor, use_history: bool) -> torch.Tensor:
        if use_history:
            latent = self.infer_hist_latent(observations)
        else:
            latent = self.infer_priv_latent(observations)
            self._last_priv_latent = latent
        self._last_obs = observations
        return torch.cat([observations[:, : self.num_prop], latent], dim=-1)

    def act(self, observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        observations = self.actor_obs_normalizer(observations)
        self.distribution = self.actor.distribution(self._actor_input(observations, self.use_history_latent))
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action from the student latent, i.e. what runs on the robot."""
        observations = self.actor_obs_normalizer(observations)
        return self.actor(self._actor_input(observations, use_history=True))

    def act_inference_privileged(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action from the teacher latent (simulation-only upper bound)."""
        observations = self.actor_obs_normalizer(observations)
        return self.actor(self._actor_input(observations, use_history=False))

    def priv_reg_loss(self) -> torch.Tensor:
        """``||z - sg(z_hat)||_2`` on the batch of the last teacher-latent :meth:`act` call."""
        if self._last_priv_latent is None or self._last_obs is None:
            raise RuntimeError("priv_reg_loss() needs a preceding act() with use_history_latent=False.")
        with torch.no_grad():
            hist_latent = self.infer_hist_latent(self._last_obs)
        return (self._last_priv_latent - hist_latent).norm(p=2, dim=1).mean()

    def adaptation_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """``||sg(z) - z_hat||_2``; trains the history encoder only."""
        if self.actor_obs_normalization:
            observations = self.actor_obs_normalizer.normalize(observations)
        with torch.no_grad():
            priv_latent = self.infer_priv_latent(observations)
        hist_latent = self.infer_hist_latent(observations)
        return (priv_latent - hist_latent).norm(p=2, dim=1).mean()
