"""Eval-parity guards for the paper-comparison benchmark.

Two properties are locked in here, both of which silently corrupt a comparison
against the published REPPO numbers if they regress:

1. ``eval_modes`` — the reference reports a single DETERMINISTIC evaluation. Running
   the extra stochastic pass costs ``num_envs * max_episode_length`` env steps per
   eval point, which on fixed-length 1000-step tasks is comparable to the entire
   training budget.

2. The eval env is used when the wrapper offers one. ManiSkill's reference eval env
   reconfigures (resamples assets/layout) every reset; the training env does not.
   Evaluating on the training env skips exactly the generalization that tasks like
   ``PickSingleYCB-v1`` measure, biasing our success rate upward against a reference
   that reconfigured — i.e. it manufactures a false "we beat the paper".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safe_rl.runners import OnPolicyRunner  # noqa: E402


class _StubEnv:
    """Minimal VecEnv: fixed-length episodes, deterministic reward, counts its steps."""

    def __init__(self, num_envs: int = 4, num_obs: int = 3, num_actions: int = 2, horizon: int = 5):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.max_episode_length = horizon
        self.device = torch.device("cpu")
        self.cost_limits = None
        self.cfg = {}
        self.step_count = 0
        self.reset_count = 0
        self._t = 0

    def _obs(self):
        return torch.zeros(self.num_envs, self.num_obs), {"observations": {}}

    def get_observations(self):
        return self._obs()

    def reset(self):
        self.reset_count += 1
        self._t = 0
        return self._obs()

    def step(self, actions):
        self.step_count += 1
        self._t += 1
        done = self._t >= self.max_episode_length
        if done:
            self._t = 0
        dones = torch.full((self.num_envs,), bool(done), dtype=torch.bool)
        rewards = torch.ones(self.num_envs)
        extras = {
            "observations": {},
            "time_outs": dones.float(),
            "success_flag": torch.ones(self.num_envs, dtype=torch.bool) if done else torch.zeros(self.num_envs, dtype=torch.bool),
        }
        return torch.zeros(self.num_envs, self.num_obs), rewards, dones, extras


class _StubWriter:
    def __init__(self):
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag, value, step):  # noqa: D102
        self.scalars[tag] = value


def _cfg(**runner_overrides) -> dict:
    cfg = {
        "algorithm": {
            "class_name": "REPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "gamma": 0.99,
        },
        "policy": {
            "class_name": "REPPOActorCritic",
            "actor_kwargs": {"hidden_dims": [8, 8]},
            "critic_kwargs": {"hidden_dim": 8},
        },
        "num_steps_per_env": 4,
        "save_interval": 1000,
        "empirical_normalization": False,
        "eval_interval": 1,
        "eval_episodes": 4,
    }
    cfg.update(runner_overrides)
    return cfg


def _runner(env, **runner_overrides) -> OnPolicyRunner:
    runner = OnPolicyRunner(env, _cfg(**runner_overrides), log_dir=None, device="cpu")
    runner.writer = _StubWriter()
    return runner


def test_eval_modes_defaults_to_both_passes() -> None:
    runner = _runner(_StubEnv())
    runner._periodic_eval(0)
    tags = runner.writer.scalars
    assert "eval/success_ode_100" in tags, "deterministic pass must run by default"
    assert "eval/success" in tags, "stochastic pass must still run by default (unchanged behaviour)"


def test_eval_modes_ode_only_skips_the_stochastic_pass() -> None:
    env = _StubEnv()
    runner = _runner(env, eval_modes=["ode"])
    runner._periodic_eval(0)
    tags = runner.writer.scalars
    assert "eval/success_ode_100" in tags
    assert "eval/success" not in tags, "eval_modes=['ode'] must not run the SDE pass"
    assert "eval/episode_return" not in tags


def test_eval_modes_halves_the_env_steps_spent_on_eval() -> None:
    both = _StubEnv()
    _runner(both)._periodic_eval(0)
    ode_only = _StubEnv()
    _runner(ode_only, eval_modes=["ode"])._periodic_eval(0)
    assert ode_only.step_count * 2 == both.step_count


def test_invalid_eval_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="eval_modes"):
        _runner(_StubEnv(), eval_modes=["greedy"])


def test_eval_prefers_the_dedicated_eval_env_and_leaves_training_untouched() -> None:
    """The regression that would inflate PickSingleYCB/PokeCube against the paper."""
    train = _StubEnv()
    twin = _StubEnv()
    train.eval_env = twin

    runner = _runner(train, eval_modes=["ode"])
    steps_before = train.step_count
    result = runner._periodic_eval(0)

    assert twin.step_count > 0, "the eval twin must be the env that gets stepped"
    assert train.step_count == steps_before, "the training env must not be stepped during eval"
    assert train.reset_count == 0, "the training env must not be reset during eval"
    assert result is None, "with a separate eval env there is no stale training obs to refresh"


def test_eval_falls_back_to_the_training_env_when_no_twin_exists() -> None:
    train = _StubEnv()
    runner = _runner(train, eval_modes=["ode"])
    result = runner._periodic_eval(0)

    assert train.step_count > 0
    assert result is not None, "sharing the training env still requires refreshing the cached obs"
