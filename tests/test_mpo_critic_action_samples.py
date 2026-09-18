"""Monte Carlo integration for MPO's scalar critic, independent of E-step sampling."""

from copy import deepcopy

import pytest
import torch

from safe_rl.algorithms import MPO, SAC
from safe_rl.modules import SACActorCritic


def make_mpo(samples=2, **policy_kwargs):
    policy = SACActorCritic(
        6, 6, 2, actor_kwargs={"hidden_dims": [16]}, critic_kwargs={"hidden_dims": [16]}, **policy_kwargs
    )
    return MPO(policy, critic_action_samples=samples, gamma=0.5, critic_lr=0.0)


def fixed_samples(monkeypatch, alg):
    # Action-major layout; critics disagree on which action has the high value.
    actions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [1.5, 0.0], [1.0, 0.0]])
    expected_obs = torch.tensor([[-1.0], [0.0], [1.0]]).expand(3, 6).repeat(2, 1)

    def sample(obs):
        torch.testing.assert_close(obs, expected_obs)
        assert not torch.is_grad_enabled()
        return actions, torch.full((6, 1), 100000.0)

    monkeypatch.setattr(alg.actor_target, "sample", sample)
    monkeypatch.setattr(alg.policy.actor, "sample", lambda _: pytest.fail("Used online policy"))
    monkeypatch.setattr(alg.policy.critic_1_target, "forward", lambda obs, act: obs[:, :1] + act[:, :1])
    monkeypatch.setattr(alg.policy.critic_2_target, "forward", lambda obs, act: obs[:, :1] + 2 - act[:, :1])
    return expected_obs[:3], torch.tensor([[10.0], [20.0], [30.0]]).expand(3, 6)


def test_average_of_clipped_twin_values_not_min_of_averages(monkeypatch):
    alg = make_mpo()
    obs, critic_obs = fixed_samples(monkeypatch, alg)
    value = alg._compute_next_value(obs.requires_grad_(), critic_obs)
    torch.testing.assert_close(value, torch.tensor([[10.0], [20.5], [31.0]]))
    assert not value.requires_grad
    # min(E Q1, E Q2) would incorrectly produce [11, 21, 31].


def test_multisample_backup_preserves_timeout_and_nstep_semantics(monkeypatch):
    alg = make_mpo()
    obs, critic_obs = fixed_samples(monkeypatch, alg)
    captured = []
    original_loss = torch.nn.functional.mse_loss

    def capture(prediction, target):
        captured.append(target.clone())
        return original_loss(prediction, target)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", capture)
    loss = alg._update_critic_standard(
        obs,
        critic_obs,
        torch.zeros(3, 2),
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[0.0], [1.0], [1.0]]),
        obs,
        critic_obs,
        bootstrap=torch.tensor([[0.0], [1.0], [0.0]]),
        effective_n_steps=torch.tensor([[1], [2], [3]]),
    )
    assert len(captured) == 2
    for target in captured:
        torch.testing.assert_close(target, torch.tensor([[6.0], [7.125], [3.0]]))
    assert torch.isfinite(torch.tensor(loss))
    assert all(parameter.grad is None for parameter in alg.actor_target.parameters())


def test_normalizers_count_states_once_and_evaluate_normalized_critic_states(monkeypatch):
    alg = make_mpo(16, actor_obs_normalization=True, critic_obs_normalization=True)
    obs, critic_obs = torch.randn(4, 6), torch.randn(4, 6) + 10
    alg.policy.update_normalization(obs, critic_obs)
    seen = []
    original = alg.policy.critic_1_target.forward

    def record(normalized_obs, actions):
        seen.append(normalized_obs.clone())
        return original(normalized_obs, actions)

    monkeypatch.setattr(alg.policy.critic_1_target, "forward", record)
    value = alg._compute_next_value(obs, critic_obs)
    assert value.shape == (4, 1)
    assert alg.policy.actor_obs_normalizer.count == 4
    assert alg.policy.critic_obs_normalizer.count == 4
    expected = alg.policy.critic_obs_normalizer.normalize(critic_obs).repeat(16, 1)
    torch.testing.assert_close(seen[0], expected)


def test_one_sample_matches_original_mpo_target_bit_for_bit():
    alg = make_mpo(1)
    obs = torch.randn(8, 6)
    torch.manual_seed(42)
    with torch.no_grad():
        actions, _ = alg._sample_next_actions(obs)
        q1, q2 = alg.policy.evaluate_q_target(obs, actions)
        expected = torch.minimum(q1, q2)
    torch.manual_seed(42)
    assert torch.equal(alg._compute_next_value(obs, obs), expected)


def test_sac_standard_target_preserves_entropy_and_online_policy():
    alg = SAC(deepcopy(make_mpo(1).policy), alpha=0.2, auto_entropy_tuning=False)
    obs = torch.randn(8, 6)
    torch.manual_seed(3)
    with torch.no_grad():
        actions, log_prob = alg.policy.sample_with_log_prob(obs)
        q1, q2 = alg.policy.evaluate_q_target(obs, actions)
        expected = torch.minimum(q1, q2) - alg.alpha.detach() * log_prob
    torch.manual_seed(3)
    assert torch.equal(alg._compute_next_value(obs, obs), expected)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "16"])
def test_invalid_sample_count_fails(count):
    with pytest.raises(ValueError, match="positive integer"):
        make_mpo(count)


def test_multisample_distributional_critic_is_rejected_instead_of_ignored():
    policy = SACActorCritic(
        6,
        6,
        2,
        critic_type="distributional",
        critic_kwargs={"num_atoms": 11, "v_min": -10, "v_max": 10, "network_kwargs": {"hidden_dims": [16]}},
    )
    with pytest.raises(ValueError, match="standard scalar critics"):
        MPO(policy, critic_action_samples=16)
