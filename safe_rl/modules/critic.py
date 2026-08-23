from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_rl.common.fh_cost import quantile_cvar, quantile_var
from safe_rl.networks import MLP, SimbaV2, build_obs_encoder
from safe_rl.utils import resolve_nn_activation


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
        encoder_type: str = "none",
        encoder_kwargs: dict[str, Any] | None = None,
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

        # Optional observation encoder applied before concatenating the action.
        self.obs_encoder = build_obs_encoder(encoder_type, num_obs, encoder_kwargs)
        encoded_obs = self.obs_encoder.output_dim if self.obs_encoder is not None else num_obs

        self.network = MLP(
            input_dim=encoded_obs + num_actions,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
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
        support_transform: str = "linear",
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
        if support_transform not in ("linear", "symlog"):
            raise ValueError(f"support_transform must be 'linear' or 'symlog'; got {support_transform!r}")

        self.num_obs = num_obs
        self.num_costs = num_costs
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.support_transform = support_transform

        # Everything below lives in a *transformed* coordinate u = T(value):
        #   linear -> T = identity (u-space == value-space, bit-identical to before)
        #   symlog -> T(x) = sign(x)·ln(1+|x|), so bins are log-spaced in value space —
        #            fine resolution near v_min, coarse far out. Lets a fixed bin budget
        #            cover a wide range so v_max can be set large without manual tuning,
        #            and the worst-cost tail (what CVaR reads) is no longer truncated.
        u_min = self._to_u(v_min, support_transform)
        u_max = self._to_u(v_max, support_transform)

        # Uniform bin spacing in u-space; sigma is also expressed in u-space units.
        self.bin_width = (u_max - u_min) / (num_bins - 1)
        if sigma is not None:
            self.sigma = float(sigma)
        else:
            ratio = sigma_to_bin_ratio if sigma_to_bin_ratio is not None else 0.75
            self.sigma = float(ratio) * self.bin_width
        if self.sigma <= 0.0:
            raise ValueError(f"sigma ({self.sigma}) must be > 0")
        self._sigma_sqrt_two = math.sqrt(2.0) * self.sigma

        # Bin edges (`support`, in u-space) extend half-bin-width beyond [u_min, u_max] so
        # that targets at the extremes receive symmetric Gaussian smoothing rather than
        # truncation. Bin centers are taken in u-space then mapped back to value space via
        # T^{-1} (`centers`), so expected-value / CVaR decoding lives in real cost units.
        half_bw = self.bin_width / 2.0
        self.register_buffer("support", torch.linspace(u_min - half_bw, u_max + half_bw, num_bins + 1))
        centers_u = torch.linspace(u_min, u_max, num_bins)
        self.register_buffer("centers", self._from_u(centers_u, support_transform))

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
        # Locate the bin in u-space so the bias init lands correctly under symlog too.
        target_u = self._to_u(target_v, support_transform)
        target_bin = min(int((target_u - u_min) / self.bin_width), num_bins - 1)
        last_linear = [m for m in self.network.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last_linear.weight.data.mul_(0.01)
            bias = torch.zeros(num_costs, num_bins)
            bias[:, target_bin] = 20.0
            last_linear.bias.copy_(bias.flatten())

    @staticmethod
    def _to_u(x: Any, transform: str) -> Any:
        # Value space -> transformed (u) space. symlog(x) = sign(x)·ln(1+|x|).
        if transform == "linear":
            return x
        if isinstance(x, torch.Tensor):
            return torch.sign(x) * torch.log1p(x.abs())
        return math.copysign(math.log1p(abs(x)), x)

    @staticmethod
    def _from_u(u: Any, transform: str) -> Any:
        # Transformed (u) space -> value space. symexp(u) = sign(u)·(exp(|u|)-1).
        if transform == "linear":
            return u
        if isinstance(u, torch.Tensor):
            return torch.sign(u) * torch.expm1(u.abs())
        return math.copysign(math.expm1(abs(u)), u)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # [B, num_costs * num_bins] -> [B, num_costs, num_bins]
        return self.network(obs).view(-1, self.num_costs, self.num_bins)

    def clip_fraction(self, target: torch.Tensor) -> torch.Tensor:
        # Fraction of (pre-clamp) targets pinned at the upper support edge. Sustained > 0
        # means v_max is too low — raise it or set support_transform="symlog". Scalar tensor.
        return (target >= self.v_max).float().mean()

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        # target: [B, num_costs] -> [B, num_costs, num_bins]
        target = target.clamp(self.v_min, self.v_max)
        # Map into u-space, where `support` (bin edges) lives, before integrating the Gaussian.
        u = self._to_u(target, self.support_transform)
        cdf = torch.special.erf((self.support - u.unsqueeze(-1)) / self._sigma_sqrt_two)
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


class CategoricalCostCritic(HLGaussCostCritic):
    """Multi-head categorical (two-hot / C51-style) cost critic.

    Structurally identical to :class:`HLGaussCostCritic` (same ``[B, num_costs, num_bins]``
    network head, ``support``/``centers`` buffers, symlog support, bias-init, expected-value /
    CVaR decoding, and fp32 cross-entropy ``loss``). The *only* difference is how a scalar
    target becomes a soft histogram: instead of integrating a Gaussian over the bins, the mass
    is split *two-hot* between the two nearest bin centers.

    This is the categorical projection of C51 (Bellemare et al. 2017) applied directly to the
    Monte-Carlo/GAE cost return (the degenerate, no-bootstrap case of
    ``DistributionalCritic.project``), and matches the "Two-Hot / Categorical" ablation baseline
    in Farebrother et al. 2024, "Stop Regressing". The Gaussian ``sigma`` / ``sigma_to_bin_ratio``
    arguments are accepted for config compatibility but ignored.
    """

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        # target: [B, num_costs] -> [B, num_costs, num_bins] (two-hot over the two nearest centers).
        target = target.clamp(self.v_min, self.v_max)
        # Work in u-space (linear or symlog), where bin *centers* are uniformly spaced with
        # spacing self.bin_width starting at u_min.
        u = self._to_u(target, self.support_transform)
        u_min = self._to_u(self.v_min, self.support_transform)
        b = ((u - u_min) / self.bin_width).clamp(0.0, self.num_bins - 1)  # fractional center index
        lower = b.floor().long()
        upper = b.ceil().long()
        weight_lower = upper.to(b.dtype) - b  # -> center[lower]; 0 when target sits exactly on a bin
        weight_upper = b - lower.to(b.dtype)  # -> center[upper]

        probs = torch.zeros(*target.shape, self.num_bins, dtype=b.dtype, device=target.device)
        probs.scatter_add_(-1, lower.unsqueeze(-1), weight_lower.unsqueeze(-1))
        probs.scatter_add_(-1, upper.unsqueeze(-1), weight_upper.unsqueeze(-1))
        # Exact-hit bins get zero from both weights (lower == upper) — assign full mass there so
        # every row sums to 1.
        exact = (lower == upper).to(b.dtype)
        probs.scatter_add_(-1, lower.unsqueeze(-1), exact.unsqueeze(-1))
        return probs


class ReferenceREPPOCritic(nn.Module):
    """Critic mirroring the reference REPPO ``Critic`` (TruDi ``networks/torch_models.py``).

    Structural differences from :class:`DistributionalCritic` that this class exists
    to reproduce exactly:

    * **Encoder / head split.** A shared ``feature_module`` (``encoder_layers`` deep)
      feeds *two* independent heads: ``critic_module`` -> ``num_atoms`` logits and
      ``pred_module`` -> features (the self-predictive aux head). The aux loss
      therefore shapes the shared encoder, which sits ``head_layers`` below the
      logits — not the layer that directly produces the value, which is what a
      single-trunk critic gives you.
    * **Additive learnable zero prior.** ``logits = head(f) + prior_scale * zero_dist``
      with ``zero_dist`` a trainable parameter initialised to ``hl_gauss(0)``, so
      E[Q] starts at ~0. (Our ``zero_init_prior`` bias-init exists because this
      additive form cannot be overcome by SimbaV2's deliberately O(1) HyperPredictor
      logits; with an ordinary MLP head the reference form is the faithful one.)
    * **Pre-head activation** on both heads (reference ``input_activation=True``)
      and RMSNorm rather than LayerNorm.

    Exposes the same surface the REPPO algorithm consumes: ``forward`` -> logits,
    ``get_dist``, ``get_value``, ``features``, ``predict_features``, and the
    ``v_min`` / ``v_max`` / ``num_atoms`` attributes used for the HL-Gauss targets.
    """

    q_support: torch.Tensor

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        num_atoms: int = 151,
        v_min: float = -10.0,
        v_max: float = 50.0,
        hidden_dim: int = 512,
        encoder_layers: int = 2,
        head_layers: int = 2,
        pred_layers: int = 2,
        activation: str = "swish",
        norm: str = "rmsnorm",
        prior_scale: float = 40.9,
        predict_reward: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "ReferenceREPPOCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.prior_scale = float(prior_scale)
        # JAX reference (`networks/jax_models.py:312-323`) makes the prediction head
        # emit hidden_dim + 1 values: slot 0 is a one-step reward prediction, slots
        # 1: are the predicted next-state features. Off by default so every existing
        # config (and its checkpoints) keeps the hidden_dim-wide head it was
        # trained with; the DMC paper-parity arm turns it on.
        self.predict_reward = bool(predict_reward)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms))

        act = resolve_nn_activation(activation)

        # Reference FCNN(layers=N) == N Linear layers: the first N-1 are normed +
        # activated, the last is bare. MLP(hidden_dims=[h]*(N-1)) is that shape.
        def fcnn(in_dim: int, out_dim: int, layers: int) -> MLP:
            return MLP(
                input_dim=in_dim,
                output_dim=out_dim,
                hidden_dims=[hidden_dim] * max(layers - 1, 1),
                activation=activation,
                norm=norm,
            )

        self.feature_module = fcnn(num_obs + num_actions, hidden_dim, encoder_layers)
        # input_activation=True on both heads (reference)
        self.critic_module = nn.Sequential(act, fcnn(hidden_dim, num_atoms, head_layers))
        pred_out = hidden_dim + 1 if self.predict_reward else hidden_dim
        self.pred_module = nn.Sequential(act, fcnn(hidden_dim, pred_out, pred_layers))

        # Learnable zero prior: softmax(logits) starts at hl_gauss(0) => E[Q] ~ 0.
        delta_z = (v_max - v_min) / (num_atoms - 1)
        sigma_sqrt2 = 0.75 * delta_z * math.sqrt(2.0)
        edges = torch.linspace(v_min - delta_z / 2.0, v_max + delta_z / 2.0, num_atoms + 1)
        cdf = torch.erf(edges / sigma_sqrt2)
        probs = (cdf[1:] - cdf[:-1]) / (cdf[-1] - cdf[0]).clamp_min(1e-8)
        self.zero_dist = nn.Parameter(probs)

    def features(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.feature_module(torch.cat([obs, actions], dim=-1))

    def predict_features(self, features: torch.Tensor) -> torch.Tensor:
        pred = self.pred_module(features)
        return pred[..., 1:] if self.predict_reward else pred

    def predict_features_reward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the prediction head into (next-state features, reward), reference order.

        Reference: ``pred_rew = pred[..., :1]``, ``pred_features = pred[..., 1:]``.
        """
        if not self.predict_reward:
            raise RuntimeError("predict_features_reward requires predict_reward=True on the critic")
        pred = self.pred_module(features)
        return pred[..., 1:], pred[..., :1]

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic_module(self.features(obs, actions)) + self.prior_scale * self.zero_dist

    def get_dist(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def get_value(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.sum(dist * self.q_support, dim=-1)


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
        encoder_type: str = "none",
        encoder_kwargs: dict[str, Any] | None = None,
        zero_init_prior: bool = False,
        prior_scale: float = 40.9,
        aux_predictor: bool = False,
        aux_predictor_hidden_dims: list[int] | None = None,
        aux_predictor_activation: str = "elu",
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

        self._zero_init_prior = zero_init_prior
        self._fallback_prior_scale = prior_scale
        self.zero_dist = None
        self.prior_scale = 0.0

        # Optional observation encoder applied before concatenating the action.
        self.obs_encoder = build_obs_encoder(encoder_type, num_obs, encoder_kwargs)
        encoded_obs = self.obs_encoder.output_dim if self.obs_encoder is not None else num_obs

        if network_kwargs is None:
            raise ValueError("`network_kwargs` is not allowed to be None")
        if network_type == "mlp":
            self.network = MLP(
                input_dim=encoded_obs + num_actions,
                output_dim=num_atoms,
                **network_kwargs,
            )
        elif network_type == "simba":
            self.network = SimbaV2(
                input_dim=encoded_obs + num_actions,
                output_dim=num_atoms,
                **network_kwargs,
            )
        else:
            raise ValueError(f"Unkown network type: {network_type}, must be 'mlp' or 'simba'")

        # Zero-prior at initialization: make softmax(logits) start as the
        # hl_gauss embedding of 0 so the initial E[Q] ~ 0 instead of the support
        # mean (uniform init is optimistic whenever the support is asymmetric
        # around 0). Implemented by INITIALIZING the output-layer bias to
        # log hl_gauss(0) — the same starting point as the reference REPPO's
        # additive `prior_scale * hl_gauss(0)` term, but fully trainable
        # per-bin. The additive form is kept only as a fallback for networks
        # without an output bias; it must NOT be paired with an O(1)-scaled
        # head (SimbaV2 HyperPredictor), whose logits can never overcome a
        # fixed +40.9 prior — that combination froze E[Q] near 0 while true
        # returns were ~50 (observed: Ant-Flat, 2026-07-16).
        if self._zero_init_prior:
            delta_z = (v_max - v_min) / (num_atoms - 1)
            sigma_sqrt2 = 0.75 * delta_z * math.sqrt(2.0)
            edges = torch.linspace(v_min - delta_z / 2.0, v_max + delta_z / 2.0, num_atoms + 1)
            cdf = torch.erf(edges / sigma_sqrt2)
            probs = cdf[1:] - cdf[:-1]
            probs = probs / (cdf[-1] - cdf[0]).clamp_min(1e-8)
            out_bias = None
            for name, param in self.network.named_parameters():
                if name.endswith("bias") and param.shape == (num_atoms,):
                    out_bias = param  # last match = output layer
            if out_bias is not None:
                with torch.no_grad():
                    out_bias.copy_(torch.log(probs.clamp_min(1e-8)))
            else:
                self.zero_dist = nn.Parameter(probs)
                self.prior_scale = self._fallback_prior_scale

        # Self-predictive auxiliary head (reference REPPO critic `pred_module`):
        # predicts sg[features(s', a')] FROM features(s, a). The predictor is what
        # makes this a prediction objective — regressing the trunk features directly
        # onto the next-state features (what we did before) instead pulls the critic's
        # own representation toward its next-state value, smoothing dQ/da.
        self.aux_predictor: nn.Module | None = None
        if aux_predictor:
            if not hasattr(self.network, "get_features"):
                raise ValueError(
                    "aux_predictor requires a network exposing get_features (network_type: simba)"
                )
            feature_dim = self.network.hidden_dim
            hidden_dims = aux_predictor_hidden_dims if aux_predictor_hidden_dims is not None else [feature_dim]
            self.aux_predictor = MLP(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dims=hidden_dims,
                activation=aux_predictor_activation,
            )

    def _encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        return torch.cat([obs, actions], dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.network(self._encode(obs, actions))
        if self.zero_dist is not None:
            logits = logits + self.prior_scale * self.zero_dist
        return logits

    def features(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Trunk features of (obs, actions) — the aux loss's prediction target."""
        net = self.network
        if not hasattr(net, "get_features"):
            raise RuntimeError(
                "features() requires a network with get_features (network_type: simba)"
            )
        return net.get_features(self._encode(obs, actions))

    def predict_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map trunk features through the self-predictive head (identity if absent)."""
        if self.aux_predictor is None:
            return features
        return self.aux_predictor(features)

    def get_dist(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def get_value(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.sum(dist * self.q_support, dim=-1)

    def get_cdf(self, dist: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution over the atom support. Shape [..., num_atoms]."""
        return torch.cumsum(dist, dim=-1).clamp(max=1.0)

    def get_var(self, dist: torch.Tensor) -> torch.Tensor:
        """Variance of the return distribution (second moment minus squared mean)."""
        mean = self.get_value(dist)
        second = torch.sum(dist * self.q_support.pow(2), dim=-1)
        return (second - mean.pow(2)).clamp_min(0.0)

    def get_quantile(self, dist: torch.Tensor, alpha: float) -> torch.Tensor:
        """Value-at-Risk: the smallest atom whose CDF reaches ``alpha``. Shape [...]."""
        idx = torch.searchsorted(self.get_cdf(dist).contiguous(),
                                 torch.full(dist.shape[:-1] + (1,), float(alpha), device=dist.device))
        idx = idx.clamp(max=self.num_atoms - 1)
        return self.q_support[idx.squeeze(-1)]

    def risk_value(self, dist: torch.Tensor, level: float) -> torch.Tensor:
        """Distorted expectation: the mean over a tail fraction ``|level|`` of the return.

        Follows the distortion-measure form of DPPO (Schneider et al., arXiv:2309.14246),
        where a single scalar sweeps risk-seeking to risk-averse and the neutral setting is
        the plain mean rather than a special case:

        * ``level`` in ``(0, 1)`` -- mean of the WORST (highest) ``level`` fraction.
          Pessimistic about cost, i.e. risk-averse.
        * ``|level| == 1``        -- the whole distribution, i.e. ``get_value``.
        * ``level`` in ``(-1, 0)`` -- mean of the BEST (lowest) ``|level|`` fraction.
          Optimistic about cost, i.e. risk-seeking.

        Prefer this to :meth:`get_quantile`: a single quantile reads one atom, so on a
        discrete support it is coarse and jumps between atoms, while a tail mean uses every
        atom beyond the cut.
        """
        frac = abs(float(level))
        if frac >= 1.0:
            return self.get_value(dist)
        upper = level > 0
        alpha = (1.0 - frac) if upper else frac
        return self.get_cvar(dist, min(max(alpha, 0.0), 1.0 - 1e-6), upper=upper)

    def get_cvar(self, dist: torch.Tensor, alpha: float, upper: bool = True) -> torch.Tensor:
        """Conditional Value-at-Risk of the return distribution.

        ``upper=True`` (the cost convention) returns ``E[Z | Z >= VaR_alpha]`` — the mean of
        the worst ``1 - alpha`` fraction. ``upper=False`` gives the lower tail, which is the
        risk-averse direction for a *reward*.

        Computed exactly from the categorical atoms, with the partial mass at the VaR atom
        split correctly, so no Gaussian assumption is needed (contrast WCSAC, which fits a
        mean and a variance head and reads CVaR off a normal).
        """
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if alpha == 0.0:
            return self.get_value(dist)
        tail = 1.0 - alpha if upper else alpha
        cdf = self.get_cdf(dist)
        # Take exactly `tail` of probability mass from the requested end, splitting the
        # boundary atom: weight_i = min(p_i, max(0, tail - mass already beyond atom i)).
        beyond = (1.0 - cdf).clamp_min(0.0) if upper else (cdf - dist).clamp_min(0.0)
        w = torch.minimum(dist, torch.clamp(tail - beyond, min=0.0))
        total = w.sum(dim=-1).clamp_min(1e-12)
        return torch.sum(w * self.q_support, dim=-1) / total

    # @torch.compile()
    def project(
        self,
        next_dist: torch.Tensor,  # [batch, num_atoms]
        rewards: torch.Tensor,  # [batch, ]
        bootstrap: torch.Tensor,  # [batch, ]
        discount: float | torch.Tensor,  # scalar, or [batch] for per-sample n-step gamma**n
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        if isinstance(discount, torch.Tensor):
            discount = discount.reshape(-1, 1)  # [batch, 1] broadcasts against the atom support
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


def quantile_huber_loss(
    theta: torch.Tensor,  # [batch, N] predicted quantiles
    target: torch.Tensor,  # [batch, M] detached target samples
    tau_hat: torch.Tensor,  # [N] midpoint quantile fractions
    kappa: float = 1.0,
    target_weights: torch.Tensor | None = None,  # [batch, M], rows sum to 1
) -> torch.Tensor:
    """Quantile Huber loss (Dabney et al. 2018, QR-DQN), reduced per sample.

    Returns shape ``[batch]``, NOT a scalar: the cost channel multiplies it by the
    hazard-stratified importance weights before reducing, so the caller owns the final
    ``.mean()``. Reduction over the pair axes follows the paper -- mean over the M target
    samples, sum over the N predicted quantiles.

    The asymmetric weight ``|tau_hat - 1{u < 0}|`` is evaluated on ``u.detach()``: the
    indicator is a selector, not a differentiable function of theta, and letting a gradient
    through it would be a bug (the loss is piecewise-linear in that mask).
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    u = target.unsqueeze(1) - theta.unsqueeze(2)  # [batch, N, M]
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    weight = (tau_hat.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    per_pair = weight * huber / kappa  # [batch, N, M]
    if target_weights is None:
        return per_pair.mean(dim=2).sum(dim=1)
    # Weighted target samples. Needed for a TD(lambda) target distribution, which is a
    # MIXTURE over n-step returns with geometric weights -- the atoms are not equally
    # weighted, so the plain mean over M would silently flatten the mixture into a uniform
    # one and discard the lambda weighting entirely.
    if target_weights.shape != target.shape:
        raise ValueError(f"target_weights {tuple(target_weights.shape)} must match target {tuple(target.shape)}")
    w = target_weights.unsqueeze(1)  # [batch, 1, M]
    return (per_pair * w).sum(dim=2).sum(dim=1)


class QuantileCritic(nn.Module):
    """QR-DQN style critic: N learned quantile locations ``theta_k(s, a)``.

    Where :class:`DistributionalCritic` fixes the atom *positions* and learns their
    probabilities, this fixes the probabilities (``1/N`` each, at midpoint fractions
    ``tau_hat``) and learns the positions. That is the point of the swap: the cost return on
    SafetyPointGoal1 is zero-inflated, so a fixed support spends most of its atoms on a value
    that carries no information and can still saturate at the upper edge.

    Interface-compatible with :class:`DistributionalCritic` where CVPO touches it:
    ``forward`` returns the distribution representation and ``get_value`` the scalar mean.
    ``get_dist`` is the identity -- for a quantile critic the forward output already IS the
    distribution -- which is what lets ``SafeActorCritic._scalar_q`` scalarize both critic
    types through the same call.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        n_quantiles: int = 64,
        kappa: float = 1.0,
        nonneg: bool = False,
        tqc_drop: int = 0,
        network_type: str = "mlp",
        network_kwargs: dict[str, Any] | None = None,
        encoder_type: str = "none",
        encoder_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
    ):
        super().__init__()

        if n_quantiles < 1:
            raise ValueError(f"n_quantiles must be >= 1, got {n_quantiles}.")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {kappa}.")
        # TQC-style truncation of the top quantiles is deliberately out of scope for Phase 1;
        # the key exists so a config can declare it, and the assert stops it being set by
        # accident and silently doing nothing.
        if int(tqc_drop) != 0:
            raise ValueError(f"tqc_drop must be 0 in this phase, got {tqc_drop}.")

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.n_quantiles = int(n_quantiles)
        self.kappa = float(kappa)
        self.nonneg = bool(nonneg)
        self.tqc_drop = 0
        self._device = device

        # Midpoint fractions (2i+1)/2N. Registered as a buffer so it follows the module to
        # the GPU and survives deepcopy into the target network, exactly like `q_support`.
        self.register_buffer(
            "tau_hat", (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / self.n_quantiles
        )

        self.obs_encoder = build_obs_encoder(encoder_type, num_obs, encoder_kwargs)
        encoded_obs = self.obs_encoder.output_dim if self.obs_encoder is not None else num_obs

        if network_kwargs is None:
            raise ValueError("`network_kwargs` is not allowed to be None")
        if network_type == "mlp":
            self.network = MLP(input_dim=encoded_obs + num_actions, output_dim=self.n_quantiles, **network_kwargs)
        elif network_type == "simba":
            self.network = SimbaV2(input_dim=encoded_obs + num_actions, output_dim=self.n_quantiles, **network_kwargs)
        else:
            raise ValueError(f"Unkown network type: {network_type}, must be 'mlp' or 'simba'")

    def _encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        return torch.cat([obs, actions], dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Sorted quantile locations, shape ``[..., n_quantiles]``.

        ``softplus`` under ``nonneg`` is the quantile analogue of the categorical cost
        critic's one-sided support (``v_min=0``): it makes a negative ``Q_c`` structurally
        impossible rather than merely unlikely.

        The sort is mandatory, not cosmetic. Nothing constrains the head to emit monotone
        outputs, and crossed quantiles corrupt every statistic read off them. Sorting is
        differentiable w.r.t. the values (it only permutes them), so no special handling of
        the backward pass is needed.
        """
        theta = self.network(self._encode(obs, actions))
        if self.nonneg:
            theta = F.softplus(theta)
        return theta.sort(dim=-1).values

    def get_dist(self, theta: torch.Tensor) -> torch.Tensor:
        """Identity. The forward output already is the distribution representation."""
        return theta

    def get_value(self, theta: torch.Tensor) -> torch.Tensor:
        """Risk-neutral mean. Each quantile carries equal mass ``1/N``."""
        return theta.mean(dim=-1)

    def zero_frac(self, theta: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """Fraction of the represented mass below ``threshold`` (zero-inflation diagnostic)."""
        return (theta < threshold).to(theta.dtype).mean(dim=-1)

    def spread(self, theta: torch.Tensor, lo: float = 0.1, hi: float = 0.9) -> torch.Tensor:
        """``q_hi - q_lo`` of the represented distribution, by nearest tau_hat."""
        lo_i = int(torch.searchsorted(self.tau_hat, torch.tensor(lo, device=self.tau_hat.device)).clamp(
            max=self.n_quantiles - 1
        ))
        hi_i = int(torch.searchsorted(self.tau_hat, torch.tensor(hi, device=self.tau_hat.device)).clamp(
            max=self.n_quantiles - 1
        ))
        return theta[..., hi_i] - theta[..., lo_i]

    # -- Risk surface. Mirrors :class:`DistributionalCritic`'s semantics exactly, specialised to
    # equal mass ``1/N`` per location, so a config can swap the critic representation without
    # changing what ``cost_constraint_mode: cvar`` means.
    def get_cdf(self, theta: torch.Tensor) -> torch.Tensor:
        """Cumulative mass through each sorted location: ``(k + 1) / N``, independent of ``theta``.

        Present for interface parity with :class:`DistributionalCritic`. For a quantile critic
        the CDF *values* are fixed by construction and it is the support that is learned, which
        is the exact dual of the categorical case.
        """
        step = 1.0 / self.n_quantiles
        return (self.tau_hat + 0.5 * step).expand_as(theta)

    def get_quantile(self, dist: torch.Tensor, alpha: float) -> torch.Tensor:
        """Value-at-Risk: the smallest location whose cumulative mass reaches ``alpha``.

        Each location carries mass ``1/N`` and ``forward`` returns them sorted, so the index is
        available in closed form -- ``ceil(alpha * N) - 1`` -- with no search. Contrast
        :meth:`DistributionalCritic.get_quantile`, which must ``searchsorted`` a learned CDF
        over a fixed support.
        """
        return quantile_var(dist, alpha)

    def get_cvar(self, dist: torch.Tensor, alpha: float, upper: bool = True) -> torch.Tensor:
        """Conditional Value-at-Risk of the represented distribution.

        ``upper=True`` (the cost convention) returns ``E[Z | Z >= VaR_alpha]``, the mean of the
        worst ``1 - alpha`` fraction. ``upper=False`` gives the lower tail, which is the
        risk-averse direction for a *reward*.

        The tail-mass accounting is the same as :meth:`DistributionalCritic.get_cvar`: take
        exactly ``tail`` of probability from the requested end, **splitting the boundary
        location** rather than rounding to whole atoms -- otherwise ``alpha=0.9`` at ``N=64``
        would silently mean ``6/64 = 0.094`` or ``7/64 = 0.109``.

        The weights depend only on ``alpha`` and ``N``, never on ``dist``, so the statistic is
        *linear* in the learned locations. That is the property that makes it safe to put inside
        the E-step exponent: the gradient reaches every location in the tail undistorted.
        """
        return quantile_cvar(dist, alpha, upper=upper)

    def risk_value(self, dist: torch.Tensor, level: float) -> torch.Tensor:
        """Distorted expectation: the mean over a tail fraction ``|level|`` of the return.

        Identical distortion convention to :meth:`DistributionalCritic.risk_value` (DPPO,
        Schneider et al., arXiv:2309.14246), so a risk level transfers between critic types:

        * ``level`` in ``(0, 1)``  -- mean of the WORST (highest) ``level`` fraction; risk-averse
          about cost.
        * ``|level| == 1``         -- the whole distribution, i.e. :meth:`get_value`.
        * ``level`` in ``(-1, 0)`` -- mean of the BEST (lowest) ``|level|`` fraction; risk-seeking.
        """
        frac = abs(float(level))
        if frac >= 1.0:
            return self.get_value(dist)
        upper = level > 0
        alpha = (1.0 - frac) if upper else frac
        return self.get_cvar(dist, min(max(alpha, 0.0), 1.0 - 1e-6), upper=upper)
