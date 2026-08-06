"""Item 4: evaluation must be read-only with respect to training.

An eval pass that consumed RNG or nudged a parameter would make every seeded run depend on
how often it was evaluated -- silently breaking the reproducibility item 0 just established.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from safe_rl.common.tail_eval import episode_cost_stats, write_episode_csv  # noqa: E402
from safe_rl.utils.seeding import rng_fingerprint, seed_everything  # noqa: E402

NUM_OBS, NUM_ACT = 6, 2


def _policy():
    from safe_rl.modules import SafeSACActorCritic

    return SafeSACActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=NUM_OBS, num_actions=NUM_ACT, num_costs=1,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"hidden_dims": [16, 16]},
        cost_critic_kwargs={"hidden_dims": [16, 16]},
    )


def test_stats_and_csv_do_not_consume_rng(tmp_path) -> None:
    seed_everything(3)
    before = rng_fingerprint()
    stats = episode_cost_stats([1.0, 40.0, 3.0], cost_limit=25.0, rewards=[1.0, 2.0, 3.0])
    write_episode_csv(tmp_path / "e.csv", [1.0, 40.0, 3.0], [1.0, 2.0, 3.0], [1000, 1000, 1000])
    assert rng_fingerprint() == before
    assert stats["n_episodes"] == 3


def test_deterministic_policy_evaluation_leaves_params_and_rng_untouched() -> None:
    """A greedy eval rollout must not perturb parameters or the RNG stream."""
    seed_everything(11)
    policy = _policy()
    params_before = torch.cat([p.detach().flatten().clone() for p in policy.parameters()])
    obs = torch.randn(32, NUM_OBS)          # consumes RNG deliberately, before the fingerprint
    fp_before = rng_fingerprint()

    with torch.no_grad():
        actions = policy.act(obs, deterministic=True)
        policy.evaluate_cost_q(obs, actions)
        policy.evaluate_q(obs, actions)

    params_after = torch.cat([p.detach().flatten() for p in policy.parameters()])
    assert torch.equal(params_before, params_after), "eval must not change parameters"
    assert rng_fingerprint() == fp_before, "deterministic eval must not consume RNG"


def test_eval_does_not_change_training_mode_flags() -> None:
    policy = _policy()
    policy.train()
    was_training = policy.training
    policy.eval()
    policy.train(was_training)
    assert policy.training == was_training


def test_stats_are_pure_functions() -> None:
    """Same input -> same output, and no hidden state between calls.

    Compared NaN-aware: reward fields are NaN when no rewards are supplied, and NaN != NaN
    would make a plain dict comparison fail on a function that is in fact pure.
    """
    costs = [1.0, 2.0, 90.0, 4.0]
    a = episode_cost_stats(costs, cost_limit=25.0)
    b = episode_cost_stats(costs, cost_limit=25.0)
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and np.isnan(x):
            assert np.isnan(y), f"{k}: {x} vs {y}"
        else:
            assert x == y, f"{k}: {x} vs {y}"
