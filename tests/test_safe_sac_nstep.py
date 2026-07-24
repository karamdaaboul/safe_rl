"""Tests for the SafeSAC <- SAC inheritance refactor and n-step cost aggregation.

SafeSAC inherits all non-safety mechanics from SAC (replay with in-buffer n-step,
reward-critic update, bootstrap channel) so the reward path cannot drift from the
MPO/SAC lineage. These tests pin the new safety-side guarantees:

- ReplayStorage aggregates ``costs`` over the n-step window exactly like ``rewards``
  (the FSRL reference CVPO applies n-step returns to reward and cost critics
  uniformly; a 1-step cost against an n-step-discounted bootstrap would be wrong).
- SafeSAC and CVPO run end-to-end with ``n_step > 1``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

NUM_OBS = 6
NUM_ACT = 2


def _make_safe_policy(num_costs: int = 1):
    from safe_rl.modules import SafeSACActorCritic

    return SafeSACActorCritic(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACT,
        num_costs=num_costs,
        actor_kwargs={"hidden_dims": [32, 32]},
        critic_kwargs={"hidden_dims": [32, 32]},
        cost_critic_kwargs={"hidden_dims": [32, 32]},
    )


def test_safe_sac_is_a_sac_subclass_with_shared_nonsafety_path() -> None:
    from safe_rl.algorithms import SAC, SafeSAC

    assert issubclass(SafeSAC, SAC)
    # The n-step contract is inherited, so the runner uses in-buffer aggregation
    # instead of raising "n_step > 1 is not supported with safe RL algorithms".
    assert SafeSAC.supports_storage_n_step
    assert SafeSAC.uses_bootstrap_channel
    # The cost-critic loss is reported through SAC.update()'s extra-critics hook.
    assert SafeSAC._extra_critic_keys == ("cost_critic",)
    # The reward-critic update itself is SAC's, not a reimplementation.
    assert SafeSAC._update_critic is SAC._update_critic
    assert "_update_reward_critic" not in vars(SafeSAC)


def test_replay_storage_nstep_aggregates_costs_like_rewards() -> None:
    from safe_rl.storage.replay_storage import ReplayStorage

    n_step, gamma, num_envs = 3, 0.9, 2
    storage = ReplayStorage(
        num_envs=num_envs,
        max_size=200,
        obs_shape=[NUM_OBS],
        action_shape=[NUM_ACT],
        device="cpu",
        n_step=n_step,
        gamma=gamma,
    )
    # Store costs identical to rewards: after aggregation the sampled n-step costs
    # must equal the sampled n-step rewards element-wise.
    torch.manual_seed(0)
    for _ in range(60):
        reward = torch.rand(num_envs)
        storage.add(
            torch.randn(num_envs, NUM_OBS),
            torch.rand(num_envs, NUM_ACT) * 2 - 1,
            reward,
            torch.zeros(num_envs),
            torch.randn(num_envs, NUM_OBS),
            costs=reward.unsqueeze(-1).clone(),
        )
    batch = storage.sample(32)
    assert batch["effective_n_steps"].max().item() == n_step  # no dones -> full horizon
    assert torch.allclose(batch["costs"], batch["rewards"], atol=1e-6)
    # Sanity: with 3 steps of rewards in [0, 1), the aggregated value can exceed 1,
    # which a 1-step (start-transition) cost never could.
    assert batch["rewards"].max().item() <= 1.0 + gamma + gamma**2


def test_replay_storage_nstep_costs_truncate_at_episode_end() -> None:
    from safe_rl.storage.replay_storage import ReplayStorage

    n_step, gamma, num_envs = 3, 0.9, 1
    storage = ReplayStorage(
        num_envs=num_envs,
        max_size=64,
        obs_shape=[1],
        action_shape=[1],
        device="cpu",
        n_step=n_step,
        gamma=gamma,
    )
    # Every transition terminates the episode: aggregation must truncate at the
    # first done, so the n-step cost stays the 1-step cost.
    for i in range(32):
        c = torch.full((1,), float(i % 5))
        storage.add(
            torch.zeros(1, 1), torch.zeros(1, 1), c, torch.ones(1), torch.zeros(1, 1),
            costs=c.unsqueeze(-1).clone(),
        )
    batch = storage.sample(16)
    assert (batch["effective_n_steps"] == 1).all()
    assert torch.allclose(batch["costs"], batch["rewards"], atol=1e-6)


def _fill_and_update(alg, num_envs: int, steps: int = 80):
    alg.init_storage(buffer_size=1000, num_envs=num_envs, obs_shape=[NUM_OBS], act_shape=[NUM_ACT])
    torch.manual_seed(0)
    for _ in range(steps):
        alg.store_transition(
            torch.randn(num_envs, NUM_OBS),
            torch.rand(num_envs, NUM_ACT) * 2 - 1,
            torch.randn(num_envs),
            torch.zeros(num_envs),
            torch.randn(num_envs, NUM_OBS),
            cost=torch.rand(num_envs),
        )
    return alg.update(current_costs=[30.0])


def test_safe_sac_nstep_update_runs_and_is_finite() -> None:
    from safe_rl.algorithms import SafeSAC

    alg = SafeSAC(
        _make_safe_policy(),
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=2,
        n_step=3,
        device="cpu",
    )
    assert alg.storage is None
    info = _fill_and_update(alg, num_envs=2)
    assert alg.storage.n_step == 3  # in-buffer aggregation is actually active
    for key in ("critic", "cost_critic", "actor", "alpha"):
        assert key in info
        assert torch.isfinite(torch.tensor(info[key]))


def test_cvpo_nstep_update_runs_and_is_finite() -> None:
    from safe_rl.algorithms import CVPO

    alg = CVPO(
        _make_safe_policy(),
        cost_limits=[25.0],
        batch_size=32,
        num_updates_per_step=1,
        sample_action_num=8,
        mstep_iteration_num=2,
        n_step=2,
        device="cpu",
    )
    info = _fill_and_update(alg, num_envs=2)
    assert alg.storage.n_step == 2
    for key in ("critic", "cost_critic", "actor"):
        assert torch.isfinite(torch.tensor(info[key]))
    assert alg.get_penalty_info()["eta"] > 0
