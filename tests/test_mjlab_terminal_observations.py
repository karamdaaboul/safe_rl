"""mjlab auto-reset -> adapter -> SAC/MPO replay, without a GPU simulator.

Adapted from agxcave commit 9a4a35c's terminal-observation regression checks.
The fake env deliberately resets before observing, as the installed mjlab does.
"""

from types import SimpleNamespace

import pytest
import torch

from safe_rl.algorithms import MPO, SAC
from safe_rl.envs.mjlab_vec_env import MjlabVecEnv
from safe_rl.runners.off_policy_runner import OffPolicyRunner
from safe_rl.storage.replay_storage import ReplayStorage


def _state_observation(env):
    return env.derived


class ObservationManager:
    def __init__(self, env):
        self.env = env
        self.active_terms = {"actor": ["state"], "critic": ["state"]}
        self.cfg = {
            name: SimpleNamespace(history_length=None, enable_corruption=False, concatenate_terms=True)
            for name in self.active_terms
        }
        self.term = SimpleNamespace(history_length=0, delay_max_lag=0, func=_state_observation)

    def get_term_cfg(self, group, term):
        return self.term

    def compute_group(self, group, update_history=False):
        assert not update_history
        if group == "actor":
            return self.env.derived
        return torch.cat([10 * self.env.derived, torch.full((4, 1), 99.0)], dim=-1)


class BoundaryEnv:
    device = torch.device("cpu")
    num_envs = 4
    num_actions = 1
    step_dt = 0.02
    max_episode_length_s = 10.0
    max_episode_length = 500

    def __init__(self, native=False):
        self.cfg = SimpleNamespace(commands={}, events={})
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.state = torch.tensor([[-10.0, -20.0]] * 4)
        self.derived = self.state.clone()
        self.observation_manager = ObservationManager(self)
        self.sim = SimpleNamespace(forward=self.forward, sense=lambda: None)
        self.reset_terminated = torch.tensor([False, True, False, True])
        self.reset_time_outs = torch.tensor([True, False, False, True])
        self.native = native
        self.info = {}

    def forward(self):
        self.derived.copy_(self.state)

    def observe(self):
        return {group: self.observation_manager.compute_group(group) for group in ("actor", "critic")}

    def reset(self):
        self.state[:] = torch.tensor([-10.0, -20.0])
        self.forward()
        return self.observe(), {}

    def _reset_idx(self, env_ids):
        self.state[env_ids] = torch.tensor([-10.0, -20.0])

    def step(self, actions):
        self.state[:] = torch.tensor([3.0, 7.0])
        if self.native:
            self.forward()
            self.info["final_observation"] = {key: value.clone() for key, value in self.observe().items()}
        env_ids = (self.reset_terminated | self.reset_time_outs).nonzero().flatten()
        if len(env_ids):
            self._reset_idx(env_ids)
        self.forward()
        return self.observe(), torch.ones(4), self.reset_terminated, self.reset_time_outs, self.info

    def close(self):
        pass


def _runner(algorithm=SAC):
    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner.device = torch.device("cpu")
    runner.uses_bootstrap_channel = algorithm.uses_bootstrap_channel
    runner._final_obs_warned = runner._final_obs_shape_warned = False
    return runner


@pytest.mark.parametrize("algorithm", [SAC, MPO])
@pytest.mark.parametrize("native", [False, True])
def test_timeout_replay_uses_final_actor_and_critic_observations(algorithm, native):
    env = BoundaryEnv(native=native)
    if native:
        # A history-based task must use the native producer, not the snapshot hook.
        env.observation_manager.term.history_length = 2
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    obs, initial = wrapped.reset()
    next_obs, rewards, dones, infos = wrapped.step(torch.zeros(4, 1))
    expected_reset = torch.tensor([[-10.0, -20.0], [-10.0, -20.0], [3.0, 7.0], [-10.0, -20.0]])
    torch.testing.assert_close(next_obs, expected_reset)
    torch.testing.assert_close(infos["time_outs"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    runner = _runner(algorithm)
    timeouts, _, store_done, bootstrap = runner._episode_boundaries(infos, dones)
    next_critic = infos["observations"]["critic"]
    stored_obs, stored_critic = runner._terminal_observations(infos, timeouts, next_obs, next_critic)
    alg = algorithm.__new__(algorithm)
    alg.storage = ReplayStorage(num_envs=4, max_size=16, obs_shape=[2], action_shape=[1])
    alg.store_transition(
        obs,
        torch.zeros(4, 1),
        rewards,
        store_done,
        stored_obs,
        critic_obs=initial["observations"]["critic"],
        next_critic_obs=stored_critic,
        bootstrap=bootstrap,
    )
    data = alg.storage._data
    expected_stored = expected_reset.clone()
    expected_stored[0] = torch.tensor([3.0, 7.0])
    torch.testing.assert_close(data["next_observations"][:4], expected_stored)
    torch.testing.assert_close(
        data["next_critic_observations"][:4],
        torch.cat([10 * expected_stored, torch.full((4, 1), 99.0)], dim=-1),
    )
    torch.testing.assert_close(1 - data["dones"][:4, 0] + data["bootstrap"][:4, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    infos["final_observation"].fill_(99)
    infos["final_critic_observation"].fill_(99)
    torch.testing.assert_close(next_obs, expected_reset)
    torch.testing.assert_close(wrapped.get_observations()[0], expected_reset)
    assert "_reset_idx" not in vars(env)


def test_snapshot_is_cleared_on_the_next_step():
    env = BoundaryEnv()
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    wrapped.reset()
    assert "final_observation" in wrapped.step(torch.zeros(4, 1))[3]
    env.reset_time_outs.zero_()
    env.reset_terminated.zero_()
    assert "final_observation" not in wrapped.step(torch.zeros(4, 1))[3]


@pytest.mark.parametrize("unsupported", ["history", "delay", "corruption", "commands", "events", "stateful"])
def test_unsupported_snapshot_requires_native_final_observations(unsupported):
    env = BoundaryEnv()
    if unsupported == "history":
        env.observation_manager.term.history_length = 2
    elif unsupported == "delay":
        env.observation_manager.term.delay_max_lag = 1
    elif unsupported == "corruption":
        env.observation_manager.cfg["actor"].enable_corruption = True
    elif unsupported == "commands":
        env.cfg.commands = {"velocity": object()}
    elif unsupported == "events":
        env.cfg.events = {"push": SimpleNamespace(mode="interval")}
    else:
        env.observation_manager.term.func = object()
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    with pytest.raises(RuntimeError, match="truncated without final_observation"):
        wrapped.step(torch.zeros(4, 1))
    assert "_reset_idx" not in vars(env)


def test_reset_hook_is_restored_when_observation_capture_raises(monkeypatch):
    env = BoundaryEnv()
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()

    def fail(*args, **kwargs):
        raise ValueError("observation failed")

    monkeypatch.setattr(env.observation_manager, "compute_group", fail)
    with pytest.raises(ValueError, match="observation failed"):
        wrapped.step(torch.zeros(4, 1))
    assert "_reset_idx" not in vars(env)


@pytest.mark.parametrize("key", ["final_observation", "final_critic_observation"])
def test_invalid_final_observation_shape_fails(key):
    env = BoundaryEnv()
    env.info["final_observation"] = torch.ones(4, 2)
    env.info[key] = torch.ones(4, 7)
    with pytest.raises(ValueError, match="observation shape"):
        MjlabVecEnv(env).step(torch.zeros(4, 1))


def test_on_policy_use_does_not_capture_or_require_final_observations():
    wrapped = MjlabVecEnv(BoundaryEnv())
    assert "final_observation" not in wrapped.step(torch.zeros(4, 1))[3]


def test_missing_separate_final_critic_observation_fails():
    env = BoundaryEnv()
    env.info["final_observation"] = torch.ones(4, 2)
    env.cfg.commands = {"velocity": object()}
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    with pytest.raises(RuntimeError, match="final critic observation"):
        wrapped.step(torch.zeros(4, 1))


def test_true_terminations_do_not_require_final_observations():
    env = BoundaryEnv()
    env.cfg.commands = {"velocity": object()}
    env.reset_terminated[:] = True
    env.reset_time_outs[:] = True
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    _, _, done, info = wrapped.step(torch.zeros(4, 1))
    assert done.bool().all()
    assert not info["time_outs"].any()
    assert "final_observation" not in info


def test_reenabling_checks_the_current_observation_configuration():
    env = BoundaryEnv()
    wrapped = MjlabVecEnv(env)
    wrapped.enable_final_observations()
    env.observation_manager.term.history_length = 2
    wrapped.enable_final_observations()
    with pytest.raises(RuntimeError, match="truncated without final_observation"):
        wrapped.step(torch.zeros(4, 1))
