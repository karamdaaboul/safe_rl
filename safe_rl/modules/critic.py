from __future__ import annotations

import math
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


class HLGaussCostCritic(nn.Module):
    """Multi-head HL-Gauss cost critic (Farebrother et al. 2024, "Stop Regressing").

    Replaces scalar regression with classification over a fixed support
    ``[v_min, v_max]`` discretized into ``num_bins`` bins. Each scalar target is
    converted to a soft histogram by integrating a Gaussian (std ``sigma``,
    centered on the target) over each bin; the network predicts logits and is
    trained with cross-entropy. The expected value over the predicted
    distribution is returned as the scalar value estimate.

    Output shape: ``[batch, num_costs, num_bins]`` logits (single MLP with
    ``output_dim = num_costs * num_bins``, reshaped).
    """

    support: torch.Tensor
    centers: torch.Tensor

    def __init__(
        self,
        num_obs: int,
        num_costs: int,
        num_bins: int = 101,
        v_min: float = 0.0,
        v_max: float = 100.0,
        sigma: float | None = None,
        sigma_to_bin_ratio: float | None = None,
        hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        layer_norm: bool = False,
        init_predicted_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        # `loss_skew` was a per-sample asymmetric reweighting that biased the critic's
        # expected value upward for safety. It violated the GAE unbiased-baseline contract
        # (§3.3) — the bias propagated into Â^C via δ^C, suppressing the gate. Pessimism now
        # comes from the detached CVaR anchor in p3o.py instead, which doesn't enter GAE.
        kwargs.pop("loss_skew", None)
        if kwargs:
            print(
                "HLGaussCostCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if v_max <= v_min:
            raise ValueError(f"v_max ({v_max}) must be > v_min ({v_min})")
        if num_bins < 2:
            raise ValueError(f"num_bins ({num_bins}) must be >= 2")
        if sigma is not None and sigma_to_bin_ratio is not None:
            raise ValueError("Pass either `sigma` or `sigma_to_bin_ratio`, not both.")

        self.num_obs = num_obs
        self.num_costs = num_costs
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        # Distance between adjacent bin centers — identical to the reference HL-Gauss formulation.
        self.bin_width = (v_max - v_min) / (num_bins - 1)
        if sigma is not None:
            self.sigma = float(sigma)
        else:
            ratio = sigma_to_bin_ratio if sigma_to_bin_ratio is not None else 0.75
            self.sigma = float(ratio) * self.bin_width
        if self.sigma <= 0.0:
            raise ValueError(f"sigma ({self.sigma}) must be > 0")
        self._sigma_sqrt_two = math.sqrt(2.0) * self.sigma

        # Bin edges extend half-bin-width beyond [v_min, v_max] so that targets at
        # the extremes receive symmetric Gaussian smoothing rather than truncation.
        # Bin centers (for expected-value decoding) sit exactly on linspace(v_min, v_max).
        half_bw = self.bin_width / 2.0
        self.register_buffer("support", torch.linspace(v_min - half_bw, v_max + half_bw, num_bins + 1))
        self.register_buffer("centers", torch.linspace(v_min, v_max, num_bins))

        self.network = MLP(
            input_dim=num_obs,
            output_dim=num_costs * num_bins,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

        # Bias head toward predicting `init_predicted_value` (default: v_min) at init.
        # Without this, random init -> uniform softmax -> expected_value = midpoint of
        # support, which gives MSE-incompatible V_cost predictions out of the box and
        # destabilizes early training (P3O κ saturates because cost-advantages are
        # baseline-shifted).
        target_v = v_min if init_predicted_value is None else float(init_predicted_value)
        target_v = max(v_min, min(v_max, target_v))
        target_bin = min(int((target_v - v_min) / self.bin_width), num_bins - 1)
        last_linear = [m for m in self.network.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last_linear.weight.data.mul_(0.01)
            bias = torch.zeros(num_costs, num_bins)
            bias[:, target_bin] = 20.0
            last_linear.bias.copy_(bias.flatten())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # [B, num_costs * num_bins] -> [B, num_costs, num_bins]
        return self.network(obs).view(-1, self.num_costs, self.num_bins)

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        # target: [B, num_costs] -> [B, num_costs, num_bins]
        target = target.clamp(self.v_min, self.v_max)
        cdf = torch.special.erf((self.support - target.unsqueeze(-1)) / self._sigma_sqrt_two)
        z = (cdf[..., -1] - cdf[..., 0]).clamp_min(1e-8)
        bin_probs = cdf[..., 1:] - cdf[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: [B, num_costs, num_bins] -> [B, num_costs]
        return torch.sum(probs * self.centers, dim=-1)

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        return self.transform_from_probs(F.softmax(logits, dim=-1))

    def cvar_value(self, logits: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """CVaR of the predicted distribution over the upper (worst-cost) tail.

        Accumulates probability mass from the highest-cost bin downward until
        the total reaches ``alpha``, then returns the re-normalized conditional
        expectation over that tail.  ``alpha=0.05`` isolates the worst 5 %.
        """
        probs = F.softmax(logits, dim=-1)  # [B, num_costs, num_bins]
        # Iterate from highest-cost bin; flip so index 0 = highest cost.
        probs_desc = probs.flip(-1)                                      # [B, num_costs, num_bins]
        centers_desc = self.centers.flip(0)                              # [num_bins]
        cum = probs_desc.cumsum(dim=-1)                                  # [B, num_costs, num_bins]
        cum_prev = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
        # Each bin contributes the probability mass it adds to the [0, alpha] window.
        bin_contrib = (cum.clamp(max=alpha) - cum_prev).clamp(min=0.0)  # [B, num_costs, num_bins]
        return (bin_contrib * centers_desc).sum(dim=-1) / alpha          # [B, num_costs]

    @torch.autocast("cuda", enabled=False)
    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Returns per-(batch, cost) cross-entropy [B, num_costs]; caller aggregates.
        # Autocast disabled: erf + log_softmax need fp32 for numerical stability.
        logits = logits.float()
        target_probs = self.transform_to_probs(target.float())
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_probs).sum(dim=-1)


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
