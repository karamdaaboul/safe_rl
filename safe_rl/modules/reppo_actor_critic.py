"""Actor-critic for REPPO, shaped like the reference implementation.

The reference (`trudi/src/torchrl/reppo.py`) holds exactly four learned objects:
an `actor`, a frozen `old_actor` used only as the KL reference, a single
`critic`, and two `EmpiricalNormalization` instances. This class mirrors that.

Two deliberate simplifications relative to the reference's `TrainState`:

* **No `old_actor` network.** REPPO is on-policy, so the collecting policy is
  fully described by the `(mu, sigma)` stored per step at collection time, which
  is what the algorithm's KL term already uses. A second network would hold the
  same numbers.
* **No target networks.** The reference has none: its bootstrap reads the live
  actor and critic, and freezing `next_values` once per iteration at collection
  *is* the target mechanism. Polyak targets were inherited scaffolding from the
  SAC lineage this class was originally copied from, were never enabled in any
  shipped config, and are gone.

Everything shared with `SACActorCritic` -- actor construction, observation
normalizers, and the squashed action distribution -- lives in
`StochasticActorCriticBase`.
"""

from __future__ import annotations

from typing import Any

import torch

from safe_rl.modules.critic import DistributionalCritic, ReferenceREPPOCritic, StandardCritic
from safe_rl.modules.stochastic_actor_critic_base import StochasticActorCriticBase


class REPPOActorCritic(StochasticActorCriticBase):
    """Stochastic actor + a single Q(s,a) critic."""

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_type: str = "stochastic",
        critic_type: str = "reference",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        min_std: float = 0.0,
        squash: str = "none",
        action_scale: float = 1.0,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "REPPOActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_type=actor_type,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            min_std=min_std,
            squash=squash,
            action_scale=action_scale,
            # sqrt(var + eps), matching the reference normalizer -- caps the gain on
            # near-constant channels at 1/sqrt(eps) instead of the legacy add_std
            # form's up-to-100x amplification.
            normalizer_eps_mode="add_var",
            actor_kwargs=actor_kwargs,
        )
        print(f"REPPO Actor: {self.actor}")

        critic_kwargs = dict(critic_kwargs) if critic_kwargs else {}
        self.critic_type = critic_type
        if critic_type == "standard":
            self.is_distributional_critic = False
            critic_kwargs.setdefault("hidden_dims", [256, 256])
            critic_kwargs.setdefault("activation", "relu")
            self.critic = StandardCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critic = DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        elif critic_type == "reference":
            # Encoder/head-split critic matching the reference exactly: the aux
            # prediction head hangs off the shared encoder, not off the layer that
            # feeds the logits. See ReferenceREPPOCritic's docstring.
            self.is_distributional_critic = True
            self.critic = ReferenceREPPOCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        else:
            raise ValueError(
                f"Unknown critic_type: {critic_type}. Must be 'standard', 'distributional' or 'reference'."
            )
        print(f"REPPO Critic: {self.critic}")

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept checkpoints written before the twin-critic and target-net removal.

        Those stored the critic in an ``nn.ModuleList`` (``critics.0.*``) plus
        target copies (``critic_targets.*``, ``actor_target.*``). Remap the first
        critic onto ``critic.*`` and drop what no longer exists -- loudly for a
        second critic, since silently keeping only critic 1 would misreport a
        twin-min policy as reproduced.
        """
        # Old modules registered the critic under BOTH `critics.0.*` (the ModuleList)
        # and `critic_1.*` (a convenience alias assigned to the same object), so a
        # legacy state_dict carries every tensor twice. Take the first source that
        # appears, drop the rest.
        dropped_twin = 0
        remapped: set[str] = set()
        for key in [k for k in state_dict if k.startswith(prefix)]:
            tail = key[len(prefix):]
            source, rest = None, None
            if tail.startswith("critics."):
                idx, _, rest = tail[len("critics."):].partition(".")
                source = "keep" if idx == "0" else "twin"
            elif tail.startswith("critic_1."):
                source, rest = "keep", tail[len("critic_1."):]
            elif tail.startswith("critic_2."):
                source, rest = "twin", tail[len("critic_2."):]
            elif tail.startswith(
                ("critic_targets.", "critic_target.", "critic_1_target.", "critic_2_target.", "actor_target.")
            ):
                source = "drop"
            if source is None:
                continue
            value = state_dict.pop(key)
            if source == "keep" and rest not in remapped:
                state_dict[prefix + "critic." + rest] = value
                remapped.add(rest)
            elif source == "twin":
                # Count every second-critic tensor, whether or not its counterpart
                # was already remapped -- otherwise the warning silently never fires.
                dropped_twin += 1
        if dropped_twin:
            print(
                f"[REPPOActorCritic] dropped {dropped_twin} tensors for a second critic: this "
                "checkpoint was trained with twin critics, which are no longer supported. The "
                "first critic was loaded; a twin-min policy will NOT be reproduced."
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------
    # Q evaluation
    # ------------------------------------------------------------------

    def evaluate_q(
        self, obs: torch.Tensor, actions: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        """Scalar Q(s, a). Actions are passed through as-is (already squashed)."""
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        if self.is_distributional_critic:
            logits = self.critic(obs, actions)
            return self.critic.get_value(self.critic.get_dist(logits)).unsqueeze(-1)
        return self.critic(obs, actions)

    def evaluate_q_dist(
        self, obs: torch.Tensor, actions: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        """Raw categorical logits, for the HL-Gauss cross-entropy critic loss."""
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_dist requires a distributional critic")
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        return self.critic(obs, actions)

    def evaluate_q_features(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        normalized: bool = False,
        predict: bool = False,
    ) -> torch.Tensor:
        """Critic trunk features for the self-predictive aux loss.

        ``predict=True`` applies the prediction head (the online side); the target
        side uses the bare features.
        """
        if not hasattr(self.critic, "features"):
            raise RuntimeError(f"{type(self.critic).__name__} exposes no `features` for the aux loss")
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        features = self.critic.features(obs, actions)
        if predict:
            features = self.critic.predict_features(features)
        return features

    def evaluate_q_features_reward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        normalized: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Online prediction head split into (next-state features, reward prediction).

        Only the reference critic carries the reward slot; see
        ``ReferenceREPPOCritic.predict_features_reward``.
        """
        if not hasattr(self.critic, "predict_features_reward"):
            raise RuntimeError(
                f"{type(self.critic).__name__} has no reward-prediction head; "
                "set critic_kwargs.predict_reward=true on a `reference` critic"
            )
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        return self.critic.predict_features_reward(self.critic.features(obs, actions))
