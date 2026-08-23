from __future__ import annotations

from typing import Any

import torch

from safe_rl.modules.actor_critic_cost import ActorCriticCost
from safe_rl.modules.critic import StandardCritic


class ActorCriticReachQ(ActorCriticCost):
    """ActorCriticCost plus an action-conditioned Hamilton-Jacobi reachability head Q_h(s, a).

    "Reach" here means HJ *reachability* (Yu et al., ICML 2022 / RESPO), not a reaching task.

    Built for RCPPO: the inherited (state-only) cost critic learns the reachability value
    V_h(s) — the worst future constraint violation from s — while ``reach_critic`` learns
    the action-conditioned Q_h(s, a) on the same max-backup targets. The two differ only in
    that Q_h is fed the action (``num_actions=num_actions`` below, versus ``0`` for V_h).

    Q_h is what a runtime safety filter needs: ``ReachabilitySafetyFilter`` ranks candidate
    actions at a fixed state, and V_h(s) returns the same value for all of them. Living on
    the policy module, the head is saved/loaded through the standard checkpoint path with
    no runner changes.

    Feedforward only (``is_recurrent`` stays False); recurrent policies are not supported.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        reach_critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # ActorCriticCost enforces num_costs > 0, which Q_h needs for its output width.
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        reach_critic_kwargs = dict(reach_critic_kwargs) if reach_critic_kwargs is not None else {}
        self.reach_critic = StandardCritic(
            num_obs=num_critic_obs,
            num_actions=num_actions,
            output_dim=self.num_costs,
            **reach_critic_kwargs,
        )
        print(f"Reach critic (Q_h): {self.reach_critic}")

    def evaluate_reach_q(self, critic_observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q_h(s, a): predicted worst future constraint violation after taking ``actions``.

        Returns:
            reach_value: [batch_size, num_costs].
        """
        critic_observations = self.critic_obs_normalizer(critic_observations)
        return self.reach_critic(critic_observations, actions)
