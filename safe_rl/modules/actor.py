# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from safe_rl.utils import resolve_nn_activation


class DeterministicActor(nn.Module):
    """Deterministic actor network that outputs mean actions."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "DeterministicActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        activation_fn = resolve_nn_activation(activation)

        layers = []
        layers.append(nn.Linear(num_obs, hidden_dims[0]))
        layers.append(activation_fn)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dims[-1], num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


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
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StochasticActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        activation_fn = resolve_nn_activation(activation)

        # Backbone
        layers = []
        input_dim = num_obs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn)
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(input_dim, num_actions)
        self.log_std_head = nn.Linear(input_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities with tanh squashing."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = Normal(mean, std)
        x = dist.rsample()
        actions = torch.tanh(x)

        # Log probability with tanh squashing correction
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - x - nn.functional.softplus(-2 * x))).sum(
            dim=-1, keepdim=True
        )

        return actions, log_prob

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        return torch.tanh(x)

