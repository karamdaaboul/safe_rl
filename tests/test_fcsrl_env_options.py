"""FCSRL-style env treatments and the cost-limit curriculum, on our side of the wrapper.

Checks that the fork's wrappers reach the sub-envs, that their info keys are surfaced on
the VecEnv contract, and that the runner turns them into the right bootstrap flags.
See codex/fcsrl-harness-tricks.md.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safety_gymnasium")

ENV_ID = "SafetyPointGoal1-v0"


def _runner(uses_bootstrap_channel: bool):
    from safe_rl.runners.off_policy_runner import OffPolicyRunner

    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner.device = torch.device("cpu")
    runner.uses_bootstrap_channel = uses_bootstrap_channel
    return runner


def _flags(runner, *, dones, time_outs, pseudo=None):
    infos: dict = {"time_outs": torch.tensor(time_outs, dtype=torch.float32)}
    if pseudo is not None:
        infos["pseudo_terminated"] = torch.tensor(pseudo, dtype=torch.float32)
    return runner._episode_boundaries(infos, torch.tensor(dones, dtype=torch.float32))


# --------------------------------------------------------------------------
# _episode_boundaries: the three kinds of boundary must not be conflated
# --------------------------------------------------------------------------


def test_bootstrap_mask_per_boundary_kind() -> None:
    """Algorithms rebuild the mask as `bootstrap + (1 - done)`; check each case."""
    #                    ongoing, truncation, termination, pseudo-terminal
    _, _, done, bootstrap = _flags(
        _runner(True),
        dones=[0.0, 1.0, 1.0, 0.0],
        time_outs=[0.0, 1.0, 0.0, 0.0],
        pseudo=[0.0, 0.0, 0.0, 1.0],
    )
    mask = bootstrap + (1.0 - done)
    assert mask.tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
    ], "ongoing and truncated steps must bootstrap; terminations and goal respawns must not"


def test_pseudo_terminal_wins_over_a_simultaneous_truncation() -> None:
    """Reaching the goal on the last step is a value boundary, not a time-out."""
    _, _, done, bootstrap = _flags(_runner(True), dones=[1.0], time_outs=[1.0], pseudo=[1.0])
    assert (bootstrap + (1.0 - done)).tolist() == [0.0]


def test_collapsed_terminal_for_algorithms_without_the_bootstrap_channel() -> None:
    _, terminal, done, bootstrap = _flags(
        _runner(False), dones=[0.0, 1.0, 1.0], time_outs=[0.0, 1.0, 0.0], pseudo=[1.0, 0.0, 0.0]
    )
    assert bootstrap is None
    # pseudo-terminal and true termination collapse to terminal; truncation does not
    assert terminal.tolist() == [1.0, 0.0, 1.0]
    assert done is terminal


def test_absent_pseudo_key_reproduces_the_previous_behaviour() -> None:
    """The flag is opt-in: without it, nothing about the flags may change."""
    time_outs, terminal, done, bootstrap = _flags(_runner(True), dones=[0.0, 1.0, 1.0], time_outs=[0.0, 1.0, 0.0])
    assert terminal.tolist() == [0.0, 0.0, 1.0]
    assert done.tolist() == [0.0, 1.0, 1.0]
    assert bootstrap.tolist() == time_outs.tolist()


# --------------------------------------------------------------------------
# The wrappers actually reach the sub-envs
# --------------------------------------------------------------------------


@pytest.fixture
def vec_env_factory():
    envs = []

    def build(**kwargs):
        from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

        env = SafetyGymnasiumVecEnv(env_id=ENV_ID, num_envs=2, seed=0, **kwargs)
        envs.append(env)
        return env

    yield build
    for env in envs:
        env.close()


def test_action_repeat_multiplies_simulator_steps(vec_env_factory) -> None:
    env = vec_env_factory(action_repeat=4)
    env.reset()
    _, _, _, extras = env.step(torch.zeros(2, env.num_actions))
    # 2 envs x 4 simulator steps per agent step
    assert extras["sim_steps"] == pytest.approx(8.0)


def test_no_action_repeat_reports_no_sim_steps(vec_env_factory) -> None:
    """Default path is untouched — no wrapper, so no sim_steps key to account for."""
    env = vec_env_factory()
    env.reset()
    _, _, _, extras = env.step(torch.zeros(2, env.num_actions))
    assert "sim_steps" not in extras
    assert "pseudo_terminated" not in extras


def test_goal_pseudo_terminal_surfaces_the_flag(vec_env_factory) -> None:
    env = vec_env_factory(goal_pseudo_terminal=True)
    env.reset()
    _, _, _, extras = env.step(torch.zeros(2, env.num_actions))
    pseudo = extras["pseudo_terminated"]
    assert pseudo.shape == (2,)
    # Standing still reaches no goal, so the flag is off but present.
    assert pseudo.tolist() == [0.0, 0.0]


def test_action_repeat_must_be_positive(vec_env_factory) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        vec_env_factory(action_repeat=0)


def test_episode_totals_survive_action_repeat(vec_env_factory) -> None:
    """Summing reward/cost over the repeat is what keeps cost budgets comparable."""
    plain = vec_env_factory()
    repeated = vec_env_factory(action_repeat=4)
    action = torch.full((2, plain.num_actions), 0.5)

    plain.reset()
    plain_cost = sum(float(plain.step(action)[3]["costs"].sum()) for _ in range(4))
    repeated.reset()
    repeated_cost = float(repeated.step(action)[3]["costs"].sum())

    # Same 4 simulator steps from the same seed and action.
    assert repeated_cost == pytest.approx(plain_cost)


# --------------------------------------------------------------------------
# Cost-limit curriculum: the env anneals the budget, the algorithm consumes it
# --------------------------------------------------------------------------


def test_cost_limit_curriculum_publishes_the_budget(vec_env_factory) -> None:
    """The wrapper is expose-only: it reports the limit, it does not alter the env."""
    env = vec_env_factory(cost_limit_curriculum=dict(
        initial_cost_limit=100.0, min_cost_limit=20.0, decrement=15.0,
        success_threshold=1.5, window=20))
    env.reset()
    _, _, _, extras = env.step(torch.zeros(2, env.num_actions))
    assert extras["cost_limit"] == pytest.approx(100.0)


def test_no_curriculum_publishes_nothing(vec_env_factory) -> None:
    env = vec_env_factory()
    env.reset()
    _, _, _, extras = env.step(torch.zeros(2, env.num_actions))
    assert "cost_limit" not in extras


def test_episode_reward_and_cost_are_logged_on_the_boundary(vec_env_factory) -> None:
    """Episode/reward and Episode/cost must be per-episode totals, not per-step values."""
    env = vec_env_factory()
    env.reset()
    action = torch.full((2, env.num_actions), 0.5)

    reward_total = torch.zeros(2)
    cost_total = torch.zeros(2)
    boundary = None
    for _ in range(env.max_episode_length):
        _, rewards, dones, extras = env.step(action)
        reward_total += rewards.cpu()
        # `costs` is (num_envs,) for a single constraint and (num_envs, m) for several.
        step_cost = extras["costs"]
        cost_total += (step_cost.sum(dim=-1) if step_cost.dim() > 1 else step_cost).cpu()
        if dones.any():
            boundary = extras["log"]
            break

    assert boundary is not None, "no episode completed within max_episode_length"
    assert "reward" in boundary and "cost" in boundary
    assert boundary["reward"].cpu() == pytest.approx(reward_total, rel=1e-4)
    assert boundary["cost"].cpu() == pytest.approx(cost_total, rel=1e-4)


def test_episode_totals_reset_between_episodes(vec_env_factory) -> None:
    """A stale accumulator would make every episode look more expensive than the last."""
    env = vec_env_factory()
    env.reset()
    action = torch.full((2, env.num_actions), 0.5)

    totals = []
    for _ in range(2 * env.max_episode_length):
        _, _, dones, extras = env.step(action)
        if dones.any():
            totals.append(float(extras["log"]["cost"].sum()))
        if len(totals) == 2:
            break

    assert len(totals) == 2, "expected two episode boundaries"
    # Independent episodes of equal length: the second is not the running sum of both.
    assert totals[1] < totals[0] * 1.9


def test_curriculum_publishes_a_single_cost_limit_metric(vec_env_factory) -> None:
    """min/max across sub-envs were noise: they sit in lockstep except near a boundary."""
    import numpy as np
    from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

    env = SafetyGymnasiumVecEnv.__new__(SafetyGymnasiumVecEnv)
    env.device = torch.device("cpu")
    extras: dict = {}
    env._forward_fcsrl_info(extras, {"cost_limit": np.array([100.0, 80.0])})

    assert extras["log"]["cost_limit"] == pytest.approx(90.0), "the logged value is the mean"
    assert not any(k.startswith("cost_limit_") for k in extras["log"])


def test_vec_env_takes_the_tightest_sub_env_budget(monkeypatch) -> None:
    """The wrapper is per-sub-env, so 8 envs run 8 windows that advance at different
    times. The constraint is global, so the vec env must report the MINIMUM -- a mean
    would enforce a budget no single env has actually earned."""
    import numpy as np
    from safe_rl.envs.safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

    env = SafetyGymnasiumVecEnv.__new__(SafetyGymnasiumVecEnv)
    env.device = torch.device("cpu")
    extras: dict = {}
    env._forward_fcsrl_info(extras, {"cost_limit": np.array([100.0, 85.0, 70.0, 100.0])})
    assert extras["cost_limit"] == pytest.approx(70.0)


def test_cvpo_uses_the_external_limit_when_set() -> None:
    """`set_cost_limit` overrides the static config value for the episodic controller."""
    from tests.test_cvpo import _make_cvpo

    alg = _make_cvpo(lambda_source="episodic", cost_limits=[20.0])
    assert alg._episodic_cost_limit() == 20.0      # falls back to config
    alg.set_cost_limit(100.0)
    assert alg._episodic_cost_limit() == 100.0     # curriculum in force
    alg.set_cost_limit(None)
    assert alg._episodic_cost_limit() == 20.0      # detaches cleanly


def test_curriculum_metrics_only_appear_when_a_curriculum_is_attached() -> None:
    """Runs with a static budget -- and every algorithm without a curriculum -- must not
    get an extra flat-line metric on their dashboards."""
    from tests.test_cvpo import _make_cvpo

    alg = _make_cvpo(lambda_source="episodic", cost_limits=[20.0])
    assert not any(k.startswith("lambda_cost_") for k in alg.get_penalty_info())

    alg.set_cost_limit(80.0)
    info = alg.get_penalty_info()
    assert info["lambda_cost_limit"] == 80.0
    assert info["lambda_cost_limit_target"] == 20.0
    assert "lambda_cost_headroom" in info


def test_curriculum_metrics_route_to_the_safe_rl_group() -> None:
    """The `lambda_` prefix puts them beside the multiplier they drive, not in Train/."""
    from safe_rl.utils.logger import Logger

    route = Logger._route_key
    for key in ("lambda_cost_limit", "lambda_cost_limit_target", "lambda_cost_headroom"):
        assert route(None, key) == f"SafeRL/{key}"
