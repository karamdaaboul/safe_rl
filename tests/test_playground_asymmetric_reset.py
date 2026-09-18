"""Keep privileged reset observations available before the first runner step."""

from types import SimpleNamespace

import torch

from safe_rl.envs.mujoco_playground_vec_env import MujocoPlaygroundVecEnv


def test_asymmetric_reset_preserves_and_copies_critic_observations():
    actor = torch.ones(2, 3)
    critic = torch.full((2, 5), 7.0)
    env = object.__new__(MujocoPlaygroundVecEnv)
    env.asymmetric_obs = True
    env.device = torch.device("cpu")
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env._inner = SimpleNamespace(reset_with_critic_obs=lambda: (actor, critic))
    obs, extras = env.reset()
    assert obs.shape == (2, 3)
    assert torch.equal(extras["observations"]["critic"], critic)
    actor.zero_()
    critic.zero_()
    cached, cached_extras = env.get_observations()
    assert cached.sum() == 6
    assert cached_extras["observations"]["critic"].sum() == 70
    assert not env.episode_length_buf.any()


def test_terminal_observations_match_runner_tensor_contract():
    env = object.__new__(MujocoPlaygroundVecEnv)
    env.device = torch.device("cpu")
    env.forward_final_observation = True
    actor = torch.ones(2, 3)
    critic = torch.full((2, 5), 7.0)
    extras = env._clean_extras(
        {"observations": {"raw": {"obs": actor, "critic_obs": critic}}, "time_outs": torch.ones(2)}
    )
    assert torch.equal(extras["final_observation"], actor)
    assert torch.equal(extras["final_critic_observation"], critic)
    actor.zero_()
    critic.zero_()
    assert extras["final_observation"].sum() == 6
    assert extras["final_critic_observation"].sum() == 70
    assert "raw" not in extras["observations"]
