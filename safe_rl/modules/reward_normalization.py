from __future__ import annotations

import torch
import torch.nn as nn

from safe_rl.modules.normalizer import EmpiricalNormalization


class RewardNormalization(nn.Module):
    """Online reward normalisation based on the running discounted return.

    Maintains a per-environment running discounted return G, updates its
    empirical std σ_G, and tracks the running max |G|. Rewards are divided by

        denom = max(σ_G + ε,  G_r_max / g_max)

    The floor G_r_max / g_max keeps the denominator from collapsing before
    σ_G is well-estimated.
    """

    G: torch.Tensor
    G_r_max: torch.Tensor

    def __init__(self, gamma: float, g_max: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("G", torch.zeros(1))
        self.register_buffer("G_r_max", torch.zeros(1))
        self.G_rms = EmpiricalNormalization(shape=1)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms._std.squeeze() + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)
        return rewards / denominator

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Advance the running return estimate with one step of data.

        Args:
            rewards: Per-environment rewards, shape ``[num_envs]``.
            dones: Per-environment done flags (float, 1.0 = terminal),
                shape ``[num_envs]``.
        """
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))

        local_max = torch.max(torch.abs(self.G))
        self.G_r_max.copy_(torch.maximum(self.G_r_max, local_max.reshape_as(self.G_r_max)))
