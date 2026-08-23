"""Evaluation-harness helpers shared by the mjlab evaluator and its tests.

These live in the library rather than in ``scripts/eval/unitree_mjlab.py`` so they can
be unit-tested without importing mjlab (the eval script performs heavy mjlab/task
registration at module import). The evaluation protocol they implement is frozen in
``reports/EVAL_PROTOCOL.md``.
"""

from __future__ import annotations

from typing import Any

import torch


def make_q_argmax_policy(policy_module: Any, num_samples: int):
    """Return ``fn(actor_obs, critic_obs) -> action`` doing eval-time Q-argmax.

    REPPO learns an explicit Q, so at evaluation we can take one step of policy
    improvement: draw ``num_samples`` actions from ``pi(.|s)``, add the distribution
    mode as a guaranteed candidate, score every candidate with the clipped twin-Q
    ``min(Q1, Q2)``, and execute the argmax. PPO cannot do this — it has no
    action-value function — and it directly probes the max-entropy train/deploy gap
    where the entropy-inflated policy's *mode* need not be the Q-greedy action.

    Squashing is load-bearing, do not remove it
    ------------------------------------------
    ``_build_distribution`` returns the *base* Normal, i.e. pre-tanh. Training draws
    actions via ``TransformedDistribution(Normal, TanhTransform).rsample()`` and
    ``evaluate_q`` does **not** squash internally, so the critic has only ever seen
    actions in ``[-1, 1]``. Candidates must therefore be ``tanh``-ed before they are
    scored *and* before they are returned; the mode candidate becomes ``tanh(mu)``,
    which is exactly what ``REPPOActorCritic.act_inference`` executes at deployment.

    Scoring raw base-Normal samples would ask Q about actions far off its training
    support and would send ``|a| > 1`` to the environment, silently invalidating any
    Q-greedy-vs-mode comparison. Guarded by ``tests/test_eval_protocol.py``.
    """

    def select(actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        norm_actor_obs = policy_module.actor_obs_normalizer(actor_obs)
        dist = policy_module._build_distribution(norm_actor_obs)
        batch = actor_obs.shape[0]
        num_actions = policy_module.num_actions
        # Candidates: the mode (so Q-argmax never scores worse than the mode) + N samples.
        samples = dist.sample((num_samples,))                             # [N, B, A]
        candidates = torch.cat([dist.mean.unsqueeze(0), samples], dim=0)  # [N+1, B, A]
        if getattr(policy_module, "squash", "none") == "tanh":
            # action_scale is NOT optional here: the policy emits
            # `action_scale * tanh(u)`, so squashing to +-1 alone would hand the
            # critic and the env actions up to `action_scale` times too small --
            # silently crippling Q-greedy on any widened-range policy.
            candidates = getattr(policy_module, "action_scale", 1.0) * torch.tanh(candidates)
        num_cand = candidates.shape[0]
        # evaluate_q normalizes critic_obs internally; broadcast obs over candidates.
        flat_obs = critic_obs.unsqueeze(0).expand(num_cand, batch, -1).reshape(num_cand * batch, -1)
        flat_act = candidates.reshape(num_cand * batch, num_actions)
        q = policy_module.evaluate_q(flat_obs, flat_act).reshape(num_cand, batch)   # [N+1, B]
        best = q.argmax(dim=0)                                            # [B]
        return candidates[best, torch.arange(batch, device=candidates.device)]

    return select


def filter_recordable(
    done_ids: torch.Tensor,
    recorded: torch.Tensor,
    one_episode_per_env: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split finished envs into (episodes to record, envs to reset).

    Returns ``(record_ids, reset_ids)``. ``reset_ids`` is always every finished env —
    their accumulators must be cleared regardless — while ``record_ids`` is the subset
    whose episode enters the sample.

    Why this exists (protocol risk R7). The naive harness runs
    ``while len(episodes) < N`` and then truncates to ``N``. Every env starts at the
    same step, so the retained episodes are the *earliest finishers*: envs that fall
    early are over-represented and long clean episodes are thrown away. With
    ``one_episode_per_env=True`` each env contributes exactly its first completed
    episode, which makes the sample an unbiased draw over envs and lets survival rate
    and tracking error be read off the same set of episodes.

    ``recorded`` is mutated in place to mark newly recorded envs.
    """
    reset_ids = done_ids
    if one_episode_per_env and done_ids.numel() > 0:
        done_ids = done_ids[~recorded[done_ids]]
        recorded[done_ids] = True
    return done_ids, reset_ids
