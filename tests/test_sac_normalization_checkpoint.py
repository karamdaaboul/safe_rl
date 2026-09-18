"""Regressions for collected-only SAC/MPO normalization and entropy-state restoration."""
from copy import deepcopy

import pytest
import torch

from safe_rl.algorithms import MPO, SAC
from safe_rl.modules import SACActorCritic
from safe_rl.modules.normalizer import EmpiricalNormalization
from safe_rl.runners import OffPolicyRunner


def policy():
    return SACActorCritic(
        3,
        5,
        2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_kwargs={"hidden_dims": [16, 16]},
        critic_kwargs={"hidden_dims": [16, 16]},
    )


@pytest.mark.parametrize("samples", [1, 16, 64])
def test_mpo_queries_do_not_change_statistics(samples):
    torch.manual_seed(2)
    p = policy()
    obs, critic = torch.randn(8, 3), torch.randn(8, 5)
    p.update_normalization(obs, critic)
    before = {k: v.clone() for k, v in p.state_dict().items() if "normalizer." in k}
    alg = MPO(p, sample_action_num=samples, mstep_iteration_num=1, estep_use_target_critic=True)
    alg._update_actor_and_alpha(obs, critic)
    alg._compute_next_value(obs, critic)
    for k, v in before.items():
        assert torch.equal(v, p.state_dict()[k]), k
    assert p.critic_obs_normalizer.count == 8
    p.eval()
    p.update_normalization(obs + 100, critic + 100)
    for k, v in before.items():
        assert torch.equal(v, p.state_dict()[k]), k


def test_sac_replay_update_is_read_only_and_shared_ppo_normalizer_still_updates():
    p = policy()
    obs, critic = torch.randn(8, 3), torch.randn(8, 5)
    p.update_normalization(obs, critic)
    before = {k: v.clone() for k, v in p.state_dict().items() if "normalizer." in k}
    alg = SAC(p)
    alg._update_actor_and_alpha(obs, critic)
    alg._update_critic_standard(obs, critic, torch.zeros(8, 2), torch.ones(8, 1), torch.zeros(8, 1), obs, critic)
    for k, v in before.items():
        assert torch.equal(v, p.state_dict()[k]), k
    legacy = EmpiricalNormalization(3)
    legacy(obs)
    assert legacy.count == 8


def test_statistics_collected_in_inference_mode_can_be_reloaded():
    p = policy()
    with torch.inference_mode():
        p.update_normalization(torch.randn(8, 3), torch.randn(8, 5))
    assert not p.actor_obs_normalizer._std.is_inference()
    assert not p.critic_obs_normalizer._std.is_inference()
    state = deepcopy(p.state_dict())
    p.load_state_dict(state)
    p.update_normalization(torch.randn(8, 3), torch.randn(8, 5))
    assert p.actor_obs_normalizer.count == 16


class TimeoutEnv:
    device = "cpu"
    num_envs, num_actions, max_episode_length, cfg = 2, 2, 1, {}

    def reset(self):
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        return self.get_observations()

    def get_observations(self):
        return torch.zeros(2, 3), {"observations": {"critic": torch.zeros(2, 5)}}

    def step(self, actions):
        return (
            torch.full((2, 3), 100.0),
            torch.ones(2),
            torch.ones(2, dtype=torch.bool),
            {
                "observations": {"critic": torch.full((2, 5), 200.0)},
                "time_outs": torch.ones(2),
                "final_observation": torch.full((2, 3), 2.0),
                "final_critic_observation": torch.full((2, 5), 4.0),
            },
        )


def runner(auto=True, normalized=True):
    return OffPolicyRunner(
        TimeoutEnv(),
        {
            "algorithm": {"class_name": "SAC", "alpha": 0.2, "auto_entropy_tuning": auto, "batch_size": 2},
            "policy": {
                "class_name": "SACActorCritic",
                "actor_obs_normalization": normalized,
                "critic_obs_normalization": normalized,
                "actor_kwargs": {"hidden_dims": [16, 16]},
                "critic_kwargs": {"hidden_dims": [16, 16]},
            },
            "runner": {"max_size": 32, "reward_normalization": False, "start_random_steps": 100, "update_after": 100},
        },
        device="cpu",
    )


def test_runner_counts_collected_successors_including_true_timeout_state():
    r = runner()
    r.eval_mode()  # learn must enter training mode before initializing statistics
    r.learn(num_learning_iterations=1)
    assert r.actor_critic.actor_obs_normalizer.count == 4
    assert r.actor_critic.critic_obs_normalizer.count == 4
    torch.testing.assert_close(r.actor_critic.actor_obs_normalizer.mean, torch.ones(3))
    torch.testing.assert_close(r.actor_critic.critic_obs_normalizer.mean, torch.full((5,), 2.0))


@pytest.mark.parametrize("auto", [False, True])
def test_checkpoint_restores_alpha_and_next_update(tmp_path, auto):
    torch.manual_seed(5)
    a = runner(auto=auto, normalized=False)
    obs, critic = torch.randn(8, 3), torch.randn(8, 5)
    a.alg._update_actor_and_alpha(obs, critic)
    a.alg.log_alpha.data.fill_(-6.0)
    a.alg.update_step = 17
    a.save(str(tmp_path / "model.pt"))
    b = runner(auto=auto, normalized=False)
    b.load(str(tmp_path / "model.pt"))
    assert torch.equal(a.alg.log_alpha, b.alg.log_alpha)
    assert b.alg.update_step == 17
    torch.manual_seed(6)
    a.alg._update_actor_and_alpha(obs, critic)
    torch.manual_seed(6)
    b.alg._update_actor_and_alpha(obs, critic)
    torch.testing.assert_close(a.alg.log_alpha, b.alg.log_alpha, rtol=0, atol=0)
    for x, y in zip(a.actor_critic.parameters(), b.actor_critic.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_evaluation_load_restores_alpha_but_not_optimizer_and_legacy_loads(tmp_path):
    a = runner(normalized=False)
    a.alg._update_actor_and_alpha(torch.randn(8, 3), torch.randn(8, 5))
    a.alg.log_alpha.data.fill_(-6.0)
    path = tmp_path / "model.pt"
    a.save(str(path))
    b = runner(normalized=False)
    b.load(str(path), load_optimizer=False)
    assert torch.equal(a.alg.log_alpha, b.alg.log_alpha)
    assert not b.alg.alpha_optimizer.state
    legacy = torch.load(path, weights_only=False)
    legacy.pop("algorithm_state_dict")
    torch.save(legacy, path)
    c = runner(normalized=False)
    c.load(str(path))
    assert c.alg.alpha.item() == pytest.approx(0.2)
