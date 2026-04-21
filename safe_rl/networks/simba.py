"""SimbaV2 architecture implementation.

Reference: "Hyperspherical Normalization for Scalable Deep Reinforcement Learning" (ICML 2025)
"""

import math
from typing import Callable

import torch
import torch.nn as nn


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize tensor along specified dimension."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class Scaler(nn.Module):
    """Learnable scaler with decoupled initialization and scaling.

    As described in Section 4.2 of the paper, the scaler uses decoupled
    initialization where the initial value and gradient scale are separate.
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.init = init
        self.scale = scale
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.init + self.scale * self.weight)


class HyperDense(nn.Module):
    """Hyperspherical dense layer with L2-normalized weights and outputs.

    Implements the hyperspherical weight normalization described in Section 4.2.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)

        self.scaler = Scaler(out_features, init=scaler_init, scale=scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize weights to unit hypersphere
        w = l2normalize(self.weight, dim=-1)
        # Linear transformation
        x = torch.nn.functional.linear(x, w)
        # Scale and normalize output
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)
        return x


class HyperMLP(nn.Module):
    """Hyperspherical MLP with inverted bottleneck structure.

    As described in Section 4.3, uses an expansion factor of 4 (inverted bottleneck).
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion = expansion
        expanded_dim = hidden_dim * expansion

        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.act = activation()

        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)
        return x


class HyperEmbedder(nn.Module):
    """Input embedding with shift and L2 normalization.

    Implements the input processing described in Section 4.1:
    Input → RSNorm → Shift + L2-Norm → Linear + Scaler + L2-Norm

    Note: RSNorm (Running Statistics Normalization) should be applied externally
    before passing inputs to this module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        c_shift: float = 3.0,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c_shift = c_shift

        self.dense = HyperDense(
            input_dim,
            hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shift and normalize input (Section 4.1)
        x = l2normalize(x + self.c_shift, dim=-1)
        x = self.dense(x)
        return x


class HyperLERPBlock(nn.Module):
    """LERP residual block with hyperspherical normalization.

    Implements the LERP residual connection described in Section 4.3:
    y = L2Norm(LERP(x, MLP(x), alpha))
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha_init: float = 0.5,
        alpha_scale: float = 1.0,
        expansion: int = 4,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = HyperMLP(
            hidden_dim,
            expansion=expansion,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            activation=activation,
        )

        # LERP alpha parameter with decoupled initialization
        self.alpha = Scaler(hidden_dim, init=alpha_init, scale=alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LERP: (1 - alpha) * x + alpha * mlp(x)
        # Compute alpha from a unit vector of shape [hidden_dim] — batch-independent.
        alpha = torch.sigmoid(self.alpha.init + self.alpha.scale * self.alpha.weight)
        y = (1 - alpha) * x + alpha * self.mlp(x)
        y = l2normalize(y, dim=-1)
        return y


class HyperPredictor(nn.Module):
    """Output prediction head with hyperspherical normalization."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dense = HyperDense(
            hidden_dim,
            output_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class SimbaV2(nn.Module):
    """SimbaV2 architecture with hyperspherical normalization.

    Implements the full SimbaV2 architecture as described in
    "Hyperspherical Normalization for Scalable Deep Reinforcement Learning" (ICML 2025).

    Architecture:
        Input → Embedder → [LERP Block × num_blocks] → Predictor → Output

    Key features:
        - L2 normalization of features (hyperspherical feature space)
        - Hyperspherical weight normalization
        - LERP residual connections
        - Decoupled scaler initialization

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output
        num_blocks: Number of LERP blocks
        expansion: Expansion factor for MLP (default: 4)
        c_shift: Shift constant for input embedding (default: 3.0)
        activation: Activation function class (default: nn.SiLU)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        expansion: int = 4,
        c_shift: float = 3.0,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Compute initialization values as per Table 3 of the paper
        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)

        # Input embedding layer
        self.embedder = HyperEmbedder(
            input_dim,
            hidden_dim,
            c_shift=c_shift,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
        )

        # LERP blocks
        self.blocks = nn.ModuleList([
            HyperLERPBlock(
                hidden_dim,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                expansion=expansion,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

        # Output prediction head - uses same scaler_init/scale as per Table 3
        self.predictor = HyperPredictor(
            hidden_dim,
            output_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.predictor(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features before the predictor head."""
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return x
