from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_resolve_nn_activation_returns_torch_module() -> None:
    from safe_rl.utils import resolve_nn_activation

    assert isinstance(resolve_nn_activation("elu"), torch.nn.ELU)
    assert isinstance(resolve_nn_activation("swish"), torch.nn.SiLU)
    with pytest.raises(ValueError):
        resolve_nn_activation("not_an_activation")


def test_rollout_storage_adds_transitions_and_computes_returns() -> None:
    from safe_rl.storage import RolloutStorage

    storage = RolloutStorage(
        training_type="rl",
        num_envs=2,
        num_transitions_per_env=2,
        obs_shape=(3,),
        privileged_obs_shape=None,
        actions_shape=(2,),
        device="cpu",
    )

    for step in range(2):
        transition = RolloutStorage.Transition()
        transition.observations = torch.full((2, 3), float(step))
        transition.actions = torch.zeros(2, 2)
        transition.rewards = torch.ones(2)
        transition.dones = torch.zeros(2) if step == 0 else torch.ones(2)
        transition.values = torch.zeros(2, 1)
        transition.actions_log_prob = torch.zeros(2)
        transition.action_mean = torch.zeros(2, 2)
        transition.action_sigma = torch.ones(2, 2)
        storage.add_transitions(transition)

    storage.compute_returns(last_values=torch.zeros(2, 1), gamma=1.0, lam=1.0, normalize_advantage=False)

    assert storage.step == 2
    assert storage.returns.shape == (2, 2, 1)
    torch.testing.assert_close(storage.returns[:, 0, 0], torch.tensor([2.0, 1.0]))


def test_replay_storage_adds_and_samples_expected_shapes() -> None:
    from safe_rl.storage import ReplayStorage

    storage = ReplayStorage(num_envs=2, max_size=4, obs_shape=[3], action_shape=[2], device="cpu", initial_size=3)
    assert not storage.initialized

    for _ in range(2):
        storage.add(
            obs=torch.zeros(2, 3),
            action=torch.ones(2, 2),
            reward=torch.ones(2),
            done=torch.zeros(2),
            next_obs=torch.full((2, 3), 2.0),
            costs=torch.zeros(2, 1),
        )

    assert len(storage) == 4
    assert storage.initialized

    batch = storage.sample(batch_size=3)
    assert batch["observations"].shape == (3, 3)
    assert batch["actions"].shape == (3, 2)
    assert batch["rewards"].shape == (3, 1)
    assert batch["next_observations"].shape == (3, 3)
    assert batch["costs"].shape == (3, 1)


def test_actor_critic_action_value_and_cost_shapes() -> None:
    from safe_rl.modules import ActorCritic

    policy = ActorCritic(
        num_actor_obs=3,
        num_critic_obs=5,
        num_actions=2,
        num_costs=1,
        actor_kwargs={"hidden_dims": [8], "activation": "elu", "init_noise_std": 0.5},
        critic_kwargs={"hidden_dims": [8], "activation": "elu"},
        cost_critic_kwargs={"hidden_dims": [8], "activation": "elu"},
    )

    actor_obs = torch.randn(4, 3)
    critic_obs = torch.randn(4, 5)

    actions = policy.act(actor_obs)
    values = policy.evaluate(critic_obs)
    costs = policy.evaluate_cost(critic_obs)
    log_probs = policy.get_actions_log_prob(actions)

    assert actions.shape == (4, 2)
    assert values.shape == (4, 1)
    assert costs.shape == (4, 1)
    assert log_probs.shape == (4,)
