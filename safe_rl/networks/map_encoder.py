from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MapAttentionEncoder(nn.Module):
    """Proprioception-conditioned attention encoder for a terrain height-scan.

    Splits a flat observation into a proprioceptive part and a height-scan block,
    runs a small zero-padded CNN over the scan grid to produce per-cell tokens,
    and pools them with multi-head attention whose query is a linear embedding of
    the proprioceptive state. The pooled map embedding is concatenated back with
    the proprio state, so the module is a drop-in front-end for an actor/critic
    trunk: its :attr:`output_dim` (``state_dim + embed_dim``) is the width the
    following MLP should consume.

    Follows the "Attention-Based Map Encoding" recipe (Xu et al.,
    arXiv:2506.09588): proprio-as-query attention over CNN map features
    generalizes to unseen terrain markedly better than feeding the flattened scan
    straight into the MLP, and beats a plain CNN-downsample or a ViT.

    The scan is assumed to be a contiguous block of the observation. By default it
    is the *trailing* block (as mjlab exposes ``height_scan`` at the end of the
    ``policy`` obs group); pass ``scan_start`` for a different offset (e.g. a
    privileged critic obs where the scan sits before extra privileged terms).
    """

    def __init__(
        self,
        num_obs: int,
        scan_shape: tuple[int, int] = (11, 17),
        num_maps: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        scan_start: int | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print("MapAttentionEncoder.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs)))
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_obs = num_obs
        self.scan_shape = tuple(scan_shape)
        self.num_maps = num_maps
        self.embed_dim = embed_dim
        self.num_cells = self.scan_shape[0] * self.scan_shape[1]
        self.scan_dim = num_maps * self.num_cells
        # Default layout: the scan is the trailing block of the observation.
        self.scan_start = (num_obs - self.scan_dim) if scan_start is None else int(scan_start)
        if self.scan_start < 0 or self.scan_start + self.scan_dim > num_obs:
            raise ValueError(
                f"scan block [{self.scan_start}, {self.scan_start + self.scan_dim}) does not fit in num_obs={num_obs}."
            )
        self.state_dim = num_obs - self.scan_dim
        self.output_dim = self.state_dim + embed_dim

        pad = 2  # kernel size 5, keep the (H, W) dimensionality
        self.cnn = nn.Sequential(
            nn.Conv2d(num_maps, embed_dim, kernel_size=5, padding=pad),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=pad),
            nn.ReLU(),
        )
        # Learned positional embedding so attention can localize cells (conv is
        # translation-equivariant and carries no absolute position).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_cells, embed_dim))
        self.q_proj = nn.Linear(self.state_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.scan_start + self.scan_dim
        scan = obs[..., self.scan_start : end]
        state = torch.cat([obs[..., : self.scan_start], obs[..., end:]], dim=-1)
        return state, scan

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state, scan = self._split(obs)
        leading = state.shape[:-1]
        maps = scan.reshape(-1, self.num_maps, self.scan_shape[0], self.scan_shape[1])
        feat = self.cnn(maps)  # (B, embed_dim, H, W)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed  # (B, H*W, embed_dim)
        query = self.q_proj(state).reshape(-1, 1, self.embed_dim)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        map_embed = self.ln(attn_out).reshape(*leading, self.embed_dim)
        return torch.cat([state, map_embed], dim=-1)


def build_obs_encoder(
    encoder_type: str | None,
    num_obs: int,
    encoder_kwargs: dict[str, Any] | None,
) -> nn.Module | None:
    """Build an optional observation encoder placed in front of an actor/critic trunk.

    Returns ``None`` when ``encoder_type`` is falsy or ``"none"`` (the default —
    no behaviour change). A returned module exposes ``output_dim``, the width the
    caller should give the following network.
    """
    if not encoder_type or encoder_type == "none":
        return None
    if encoder_type == "map_attention":
        return MapAttentionEncoder(num_obs=num_obs, **(encoder_kwargs or {}))
    raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'none' or 'map_attention'.")
