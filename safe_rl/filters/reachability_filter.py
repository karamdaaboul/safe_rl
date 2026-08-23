from __future__ import annotations

import time

import torch


class ReachabilitySafetyFilter:
    """Least-restrictive runtime safety filter driven by a learned reachability critic.

    The HJ-reachability analogue of the analytic CBF filter (`safe_rl/cbf/cbf_filter.py`),
    but fully learned: it queries the policy's action-conditioned safety head Q_h(s, a)
    (see ``ActorCriticReachQ`` / RCPPO) instead of privileged simulator state and known
    dynamics. Proposed actions predicted to stay in the feasible set (Q_h <= threshold)
    pass through untouched; flagged actions are replaced (``mode="switch"``) or blended
    (``mode="blend"``) with the safest of a candidate set — the proposal, samples from the
    policy's current Gaussian, and uniform samples over the action box — ranked by Q_h.

    Mirrors the CBF filter integration contract: constructed by the runner from a config
    block, called between ``alg.act`` and ``env.step``, and exposing ``last_solve_ms`` /
    ``last_intervention_frac`` for logging. Pure torch; safe under torch.inference_mode().
    """

    def __init__(
        self,
        policy,
        threshold: float = 0.0,
        num_candidates: int = 16,
        mode: str = "switch",
        blend_temp: float = 0.5,
        perturb_std: float = 0.4,
        cost_index: int = 0,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if not hasattr(policy, "evaluate_reach_q"):
            raise ValueError(
                "ReachabilitySafetyFilter needs a policy with an evaluate_reach_q head; "
                "use policy class_name: ActorCriticReachQ (trained with RCPPO)."
            )
        if mode not in ("switch", "blend"):
            raise ValueError(f"mode must be 'switch' or 'blend', got {mode!r}")
        self.policy = policy
        self.threshold = threshold
        self.num_candidates = max(int(num_candidates), 2)
        self.mode = mode
        self.blend_temp = blend_temp
        self.perturb_std = perturb_std
        self.cost_index = cost_index
        self.action_low = action_low
        self.action_high = action_high
        self.device = device
        self.last_solve_ms = 0.0
        self.last_intervention_frac = 0.0

    def filter(self, actions: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Return actions with unsafe proposals replaced/blended toward argmin-Q_h candidates.

        Args:
            actions: proposed actions, shape (num_envs, num_actions).
            critic_obs: the (normalized) critic observations the actions were computed from.
        """
        start = time.perf_counter()
        q_prop = self.policy.evaluate_reach_q(critic_obs, actions)[:, self.cost_index]
        unsafe = q_prop > self.threshold
        num_unsafe = int(unsafe.sum().item())
        self.last_intervention_frac = num_unsafe / max(actions.shape[0], 1)
        if num_unsafe == 0:
            self.last_solve_ms = (time.perf_counter() - start) * 1000.0
            return actions

        flagged_obs = critic_obs[unsafe]
        flagged_actions = actions[unsafe]
        candidates = self._candidate_actions(flagged_actions, unsafe)  # (U, K, act)
        num_flagged, k, act_dim = candidates.shape

        obs_rep = flagged_obs.unsqueeze(1).expand(-1, k, -1).reshape(num_flagged * k, -1)
        q_cand = self.policy.evaluate_reach_q(obs_rep, candidates.reshape(num_flagged * k, act_dim))
        q_cand = q_cand[:, self.cost_index].view(num_flagged, k)
        best = q_cand.argmin(dim=1)
        safest = candidates[torch.arange(num_flagged, device=candidates.device), best]

        filtered = actions.clone()
        if self.mode == "switch":
            filtered[unsafe] = safest
        else:
            # Blend weight grows with the predicted violation above the threshold.
            w = ((q_prop[unsafe] - self.threshold) / self.blend_temp).clamp(0.0, 1.0).unsqueeze(-1)
            blended = (1.0 - w) * flagged_actions + w * safest
            filtered[unsafe] = blended.clamp(self.action_low, self.action_high)
        self.last_solve_ms = (time.perf_counter() - start) * 1000.0
        return filtered

    def _candidate_actions(self, proposed: torch.Tensor, unsafe: torch.Tensor) -> torch.Tensor:
        """Build the candidate set (U, K, act): proposal + policy samples + uniform samples.

        Samples from the policy's cached Gaussian directly via ``policy.distribution``
        WITHOUT calling ``policy.act`` (which would clobber the distribution the algorithm
        just cached for its rollout bookkeeping). At inference time ``act_inference``
        leaves the distribution unset (or stale from the update phase), so fall back to
        perturbing the proposal with ``perturb_std``.
        """
        num_flagged, act_dim = proposed.shape
        num_policy = (self.num_candidates - 1) // 2
        num_uniform = self.num_candidates - 1 - num_policy

        dist = getattr(self.policy, "distribution", None)
        if dist is not None and dist.mean.shape == (unsafe.shape[0], act_dim):
            mean, std = dist.mean[unsafe], dist.stddev[unsafe]
        else:
            mean, std = proposed, torch.full_like(proposed, self.perturb_std)

        noise = torch.randn(num_flagged, num_policy, act_dim, device=proposed.device, dtype=proposed.dtype)
        gauss = mean.unsqueeze(1) + std.unsqueeze(1) * noise
        uniform = torch.empty(
            num_flagged, num_uniform, act_dim, device=proposed.device, dtype=proposed.dtype
        ).uniform_(self.action_low, self.action_high)

        candidates = torch.cat([proposed.unsqueeze(1), gauss, uniform], dim=1)
        return candidates.clamp(self.action_low, self.action_high)
