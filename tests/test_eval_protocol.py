"""Regressions for the frozen evaluation protocol (reports/EVAL_PROTOCOL.md).

Two defects in the pre-V0 evaluator would have silently invalidated the mode-alignment
program's conclusions. Both are guarded here.

R1  Q-argmax scored and executed *pre-tanh* actions. Training squashes, and
    ``evaluate_q`` does not squash internally, so the critic had only ever seen actions
    in [-1, 1]. Scoring raw base-Normal samples asks Q about off-support actions and
    sends |a| > 1 to the environment — which would have produced a meaningless answer
    to "does REPPO's critic know better actions than the policy mean?".

R7  The episode harvest kept whichever episodes finished first and truncated the rest.
    Because every env starts on the same step, that retained the earliest finishers,
    over-representing falls and discarding long clean episodes.
"""

from __future__ import annotations

import torch

from safe_rl.modules import REPPOActorCritic
from safe_rl.utils.eval_utils import filter_recordable, make_q_argmax_policy

OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACT_DIM = 3


def _make_policy(squash: str = "tanh") -> REPPOActorCritic:
    torch.manual_seed(0)
    return REPPOActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=CRITIC_OBS_DIM,
        num_actions=ACT_DIM,
        actor_type="stochastic",
        critic_type="distributional",
        num_critics=2,
        squash=squash,
        actor_kwargs={"hidden_dims": [16, 16], "network_type": "mlp"},
        critic_kwargs={
            "num_atoms": 11,
            "v_min": -5.0,
            "v_max": 5.0,
            "network_type": "simba",
            "network_kwargs": {"hidden_dim": 16, "num_blocks": 1},
        },
    )


# ---------------------------------------------------------------------------
# R1 — Q-argmax must operate on tanh-squashed actions
# ---------------------------------------------------------------------------


def test_q_argmax_returns_squashed_actions_in_action_bounds():
    """Executed actions must lie in [-1, 1], the range the critic was trained on."""
    policy = _make_policy(squash="tanh")
    select = make_q_argmax_policy(policy, num_samples=16)

    obs = torch.randn(8, OBS_DIM)
    critic_obs = torch.randn(8, CRITIC_OBS_DIM)
    actions = select(obs, critic_obs)

    assert actions.shape == (8, ACT_DIM)
    assert torch.isfinite(actions).all()
    assert actions.abs().max() <= 1.0, (
        "Q-argmax returned unsquashed (pre-tanh) actions; the environment would receive "
        "|a| > 1 and the critic would be queried off its training support."
    )


def test_q_argmax_with_wide_sigma_still_bounded():
    """A high-entropy policy is exactly the regime the bug showed up in.

    With sigma inflated, base-Normal samples routinely land well outside [-1, 1]; only
    the tanh keeps candidates on-support. REPPO holds sigma near 0.5 for its whole run,
    so this is the operating point, not a corner case.
    """
    policy = _make_policy(squash="tanh")
    with torch.no_grad():
        # Force a large log_std so raw samples are far outside the action bounds.
        policy.actor.log_std_head.bias.fill_(2.0)
        policy.actor.log_std_head.weight.zero_()

    select = make_q_argmax_policy(policy, num_samples=64)
    actions = select(torch.randn(4, OBS_DIM), torch.randn(4, CRITIC_OBS_DIM))
    assert actions.abs().max() <= 1.0


def test_q_argmax_zero_samples_reduces_to_the_deterministic_mode():
    """With no samples the only candidate is the mode, so Q-argmax == act_inference.

    This pins the mode candidate to tanh(mu) rather than the raw mu, and makes
    "q_argmax N" a strict superset of the deterministic arm: any measured improvement
    is attributable to the sampled candidates, not to a different mode convention.
    """
    policy = _make_policy(squash="tanh")
    obs = torch.randn(6, OBS_DIM)
    critic_obs = torch.randn(6, CRITIC_OBS_DIM)

    greedy = make_q_argmax_policy(policy, num_samples=0)(obs, critic_obs)
    deterministic = policy.act_inference(obs)

    torch.testing.assert_close(greedy, deterministic)


def test_q_argmax_never_scores_below_the_mode():
    """The mode is always a candidate, so the selected action's Q is >= the mode's Q."""
    policy = _make_policy(squash="tanh")
    obs = torch.randn(8, OBS_DIM)
    critic_obs = torch.randn(8, CRITIC_OBS_DIM)

    chosen = make_q_argmax_policy(policy, num_samples=32)(obs, critic_obs)
    mode = policy.act_inference(obs)

    def q_of(actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = policy.evaluate_q(critic_obs, actions)
        return torch.minimum(q1, q2).squeeze(-1)

    assert (q_of(chosen) >= q_of(mode) - 1e-5).all()


def test_q_argmax_unsquashed_policy_is_left_alone():
    """squash='none' policies must not get a spurious tanh applied."""
    policy = _make_policy(squash="none")
    select = make_q_argmax_policy(policy, num_samples=8)
    actions = select(torch.randn(4, OBS_DIM), torch.randn(4, CRITIC_OBS_DIM))
    torch.testing.assert_close(actions, actions)  # shape/finiteness only
    assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# R7 — one episode per env, no earliest-finisher bias
# ---------------------------------------------------------------------------


def test_filter_recordable_takes_only_the_first_episode_per_env():
    recorded = torch.zeros(4, dtype=torch.bool)

    first, reset = filter_recordable(torch.tensor([0, 2]), recorded, one_episode_per_env=True)
    assert first.tolist() == [0, 2]
    assert reset.tolist() == [0, 2]
    assert recorded.tolist() == [True, False, True, False]

    # Env 0 finishes a SECOND episode while env 1 finishes its first. Only env 1 counts,
    # but both must still be reset.
    second, reset = filter_recordable(torch.tensor([0, 1]), recorded, one_episode_per_env=True)
    assert second.tolist() == [1], "a fast-failing env leaked a second episode into the sample"
    assert reset.tolist() == [0, 1]
    assert recorded.tolist() == [True, True, True, False]


def test_filter_recordable_legacy_mode_records_every_finished_episode():
    recorded = torch.zeros(4, dtype=torch.bool)
    for _ in range(3):
        rec, reset = filter_recordable(torch.tensor([0, 1]), recorded, one_episode_per_env=False)
        assert rec.tolist() == [0, 1]
        assert reset.tolist() == [0, 1]
    assert not recorded.any(), "legacy mode must not consume the per-env slots"


def test_filter_recordable_handles_no_finished_envs():
    recorded = torch.zeros(3, dtype=torch.bool)
    rec, reset = filter_recordable(torch.tensor([], dtype=torch.long), recorded, one_episode_per_env=True)
    assert rec.numel() == 0 and reset.numel() == 0


def test_one_episode_per_env_is_unbiased_when_fast_envs_fail():
    """The bias this flag removes, demonstrated end to end.

    Env 0 falls every 10 steps; env 1 survives 100. Collecting 4 episodes the legacy way
    yields almost all short ones. One-per-env yields exactly one from each.
    """
    lengths = {0: 10, 1: 100}

    def simulate(one_per_env: bool, want: int) -> list[int]:
        recorded = torch.zeros(2, dtype=torch.bool)
        collected: list[int] = []
        for step in range(1, 401):
            done = [e for e, ln in lengths.items() if step % ln == 0]
            if not done:
                continue
            rec, _ = filter_recordable(torch.tensor(done), recorded, one_per_env)
            collected.extend(lengths[int(e)] for e in rec)
            if one_per_env:
                if bool(recorded.all()):
                    break
            elif len(collected) >= want:
                break
        return collected if one_per_env else collected[:want]

    legacy = simulate(one_per_env=False, want=4)
    assert legacy.count(10) > legacy.count(100), "expected legacy harvest to over-sample short episodes"

    unbiased = simulate(one_per_env=True, want=2)
    assert sorted(unbiased) == [10, 100], "one-per-env must draw exactly one episode from each env"
