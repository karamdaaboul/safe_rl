"""PPO can bootstrap timeouts from V(final_obs), independently of V(current_obs)."""

from types import SimpleNamespace

import pytest
import torch

from safe_rl.algorithms import PPO
from safe_rl.modules import ActorCritic, EmpiricalNormalization
from safe_rl.runners.on_policy_runner import OnPolicyRunner


def test_ppo_stores_reward_plus_final_value_and_cuts_the_episode():
    policy = ActorCritic(2, 3, 1, actor_kwargs={"hidden_dims": [8]}, critic_kwargs={"hidden_dims": [8]})
    alg = PPO(policy, gamma=0.9)
    alg.init_storage("rl", 2, 1, [2], [3], [1])
    alg.act(torch.zeros(2, 2), torch.zeros(2, 3))
    alg.transition.values.fill_(100)
    rewards = torch.tensor([1., 2.])
    alg.process_env_step(rewards, torch.tensor([1., 0.]), {
        "time_outs": torch.tensor([1., 0.]), "timeout_values": torch.tensor([[7.], [999.]]),
    })
    torch.testing.assert_close(alg.storage.rewards[0, :, 0], torch.tensor([7.3, 2.]))
    torch.testing.assert_close(rewards, torch.tensor([1., 2.]))
    assert alg.storage.dones[0, 0, 0]


@pytest.mark.parametrize("privileged", [False, True])
@pytest.mark.parametrize("empirical", [False, True])
def test_runner_evaluates_final_observations_without_updating_normalizers(privileged, empirical):
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.device = "cpu"
    runner.privileged_obs_type = "critic" if privileged else None
    runner.empirical_normalization = empirical
    runner.obs_normalizer = EmpiricalNormalization([2])
    runner.privileged_obs_normalizer = EmpiricalNormalization([2])
    internal = EmpiricalNormalization([2])
    internal(torch.tensor([[1., 2.], [3., 4.]]))
    before = {key: value.clone() for key, value in internal.state_dict().items()}
    policy = SimpleNamespace(is_recurrent=False, critic_obs_normalizer=internal)
    policy.evaluate = lambda obs: internal(obs).sum(-1, keepdim=True)
    runner.alg = SimpleNamespace(policy=policy, transition=SimpleNamespace(values=torch.zeros(2, 1)))
    actor_final = torch.tensor([[3., 7.], [10., 20.]])
    critic_final = actor_final + 10
    source = critic_final if privileged else actor_final
    if empirical:
        external = runner.privileged_obs_normalizer if privileged else runner.obs_normalizer
        source = external.normalize(source)
    expected = internal.normalize(source[:1]).sum(-1)
    info = {"time_outs": torch.tensor([1., 0.]), "final_observation": actor_final,
            "final_critic_observation": critic_final}
    result = runner._add_timeout_values(info)
    torch.testing.assert_close(result["timeout_values"][:, 0], torch.tensor([expected.item(), 0.]))
    assert "timeout_values" not in info
    assert internal.training
    for key, value in internal.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_missing_final_critic_observation_fails():
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.device = "cpu"
    runner.privileged_obs_type = "critic"
    runner.alg = SimpleNamespace(policy=SimpleNamespace(is_recurrent=False))
    with pytest.raises(RuntimeError, match="final_critic_observation"):
        runner._add_timeout_values({"time_outs": torch.ones(1), "final_observation": torch.zeros(1, 2)})
