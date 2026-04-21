from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_rl.networks import MLP, SimbaV2


class StandardCritic(nn.Module):
    """MLP critic for V(s) or Q(s,a).

    With ``num_actions=0`` behaves as a V(s) estimator; otherwise Q(s,a) by
    concatenating ``obs`` and ``actions`` before the MLP. ``output_dim``
    controls the number of heads (e.g., ``num_costs`` for vector cost critics).
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int = 0,
        output_dim: int = 1,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        layer_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "StandardCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.output_dim = output_dim

        self.network = MLP(
            input_dim=num_obs + num_actions,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_actions == 0:
            return self.network(obs)
        return self.network(torch.cat([obs, actions], dim=-1))


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        network_type: str = "mlp",
        network_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms))
        self._device = device

        if network_kwargs is None:
            raise ValueError("`network_kwargs` is not allowed to be None")
        if network_type == "mlp":
            self.network = MLP(
                input_dim=num_obs + num_actions,
                output_dim=num_atoms,
                **network_kwargs,
            )
        elif network_type == "simba":
            self.network = SimbaV2(
                input_dim=num_obs + num_actions,
                output_dim=num_atoms,
                **network_kwargs,
            )
        else:
            raise ValueError(f"Unkown network type: {network_type}, must be 'mlp' or 'simba'")

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)

    def get_dist(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def get_value(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.sum(dist * self.q_support, dim=-1)

    # @torch.compile()
    def project(
        self,
        next_dist: torch.Tensor,  # [batch, num_atoms]
        rewards: torch.Tensor,  # [batch, ]
        bootstrap: torch.Tensor,  # [batch, ]
        discount: float,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_int = lower == upper
        l_mask = is_int & (lower > 0)
        u_mask = is_int & (lower == 0)

        lower = torch.where(l_mask, lower - 1, lower)
        upper = torch.where(u_mask, upper + 1, upper)

        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=next_dist.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1)
        )
        return proj_dist
