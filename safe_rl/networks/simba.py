"""SimbaV2 architecture implementation.

Faithful port of the reference FastTD3 SimbaV2 implementation
(younggyoseo/FastTD3, fast_td3/fast_td3_simbav2.py), which in turn implements
"Hyperspherical Normalization for Scalable Deep Reinforcement Learning" (ICML 2025).

Key design points carried over from the reference:
- ``HyperDense`` is a plain bias-free orthogonal linear; hyperspherical
  structure comes from L2-normalizing activations between layers, not from
  re-normalizing weights every forward.
- The whole trunk is bias-free; only the prediction head carries an explicit bias.
- ``HyperEmbedder`` appends a constant ``c_shift`` feature (homogeneous
  coordinate) before normalizing — it does not add the constant to every dim.
- ``Scaler`` uses the decoupled multiplicative parameterization.
- The prediction head (``HyperPredictor``) is ``w1 -> scaler -> w2 + bias`` with
  scaler_init = scaler_scale = 1.0 and NO output L2-normalization, so value/C51
  logits keep an O(1) scale.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize tensor along specified dimension."""
    return x / (torch.linalg.norm(x, ord=2, dim=dim, keepdim=True) + eps)


class Scaler(nn.Module):
    """Learnable scaler with decoupled (multiplicative) initialization.

    Stores ``init * scale`` as the parameter and multiplies the input by
    ``parameter * (init / scale)``; the ``init``/``scale`` split decouples the
    effective magnitude from the effective learning rate (Section 4.2).
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), init * scale))
        self.forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.to(x.dtype) * self.forward_scaler * x


class HyperDense(nn.Module):
    """Bias-free orthogonal linear layer (no per-forward normalization)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class HyperMLP(nn.Module):
    """Hyperspherical inverted-bottleneck MLP (Section 4.3).

    ``w1 -> scaler -> ReLU(+eps) -> w2 -> L2Norm``. The scaler acts on the
    expanded hidden dimension, before the activation. ``hidden_dim`` here is the
    expanded width (e.g. ``base_hidden * expansion``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` prevents a zero vector going into the next normalization.
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, dim=-1)
        return x


class HyperEmbedder(nn.Module):
    """Input embedding (Section 4.1).

    Appends a constant ``c_shift`` feature, then L2Norm -> HyperDense -> Scaler
    -> L2Norm. RSNorm (running-statistics normalization) is applied externally
    before this module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float = 3.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c_shift = c_shift

        self.w = HyperDense(input_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_axis = torch.full((*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype)
        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)
        return x


class HyperLERPBlock(nn.Module):
    """LERP residual block (Section 4.3).

    ``x <- L2Norm(x + alpha_scaler(MLP(x) - x))`` where ``alpha_scaler`` is a
    learnable per-dim Scaler. The internal MLP's scaler is divided by
    ``sqrt(expansion)`` (as in the reference).
    """

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
        )
        self.alpha_scaler = Scaler(hidden_dim, init=alpha_init, scale=alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mlp_out = self.mlp(x)
        x = residual + self.alpha_scaler(mlp_out - residual)
        x = l2normalize(x, dim=-1)
        return x


class HyperPredictor(nn.Module):
    """Output prediction head (matches the reference value/logit head).

    ``w1 -> scaler -> w2 + bias`` with NO output L2-normalization, so the head
    can emit logits/values at an O(1) scale. The scaler uses init = scale = 1.0
    (the reference passes 1.0/1.0 to the categorical value head), independent of
    the trunk's hyperspherical scaler magnitudes.
    """

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

        self.w1 = HyperDense(hidden_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(hidden_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = self.w2(x) + self.bias.to(x.dtype)
        return x


class SimbaV2(nn.Module):
    """SimbaV2 backbone + prediction head.

    Architecture: Input -> Embedder -> [LERP Block x num_blocks] -> Predictor.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Embedding / residual-stream width.
        output_dim: Output dimension (e.g. num_atoms for a C51 critic).
        num_blocks: Number of LERP blocks.
        expansion: Inverted-bottleneck expansion factor (default: 4).
        c_shift: Constant appended by the embedder (default: 3.0).
        activation: Accepted for interface compatibility; the reference trunk
            uses ReLU internally and this argument is ignored.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        expansion: int = 4,
        c_shift: float = 3.0,
        activation: Callable[[], nn.Module] | str | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Initialization values per Table 3 of the paper.
        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)

        self.embedder = HyperEmbedder(
            input_dim,
            hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        self.blocks = nn.ModuleList([
            HyperLERPBlock(
                hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                expansion=expansion,
            )
            for _ in range(num_blocks)
        ])

        # Prediction head: reference uses scaler_init = scaler_scale = 1.0 and no
        # output normalization.
        self.predictor = HyperPredictor(
            hidden_dim,
            output_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
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
