"""Off-policy Q-bootstrap must use the pre-reset observation on truncation.

Vectorized envs auto-reset, so the obs returned with ``truncated=True`` belongs to the
next episode. The runner used to look for it under ``infos["observations"]["final"]``, a
key no env ever sets, so the repair was dead on every off-policy Safety-Gymnasium run --
where nothing terminates and everything truncates.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

NUM_ENVS = 4
NUM_OBS = 3


@pytest.fixture
def log_records() -> list[logging.LogRecord]:
    """Records emitted under the `safe_rl` logger.

    Handlers attached to a logger run before propagation, so this captures the
    runner's warnings whether or not the package logger propagates to root —
    unlike capsys, which cannot see a handler bound to the pre-test stderr.
    """
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("safe_rl")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


def _runner():
    """A bare runner exposing only what `_terminal_observations` touches."""
    from safe_rl.runners.off_policy_runner import OffPolicyRunner

    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner.device = torch.device("cpu")
    runner._final_obs_warned = False
    runner._final_obs_shape_warned = False
    return runner


def _step(time_out_envs: list[int]):
    """Post-reset obs, plus the terminal obs the env would have published."""
    post_reset = torch.arange(NUM_ENVS * NUM_OBS, dtype=torch.float32).reshape(NUM_ENVS, NUM_OBS)
    terminal = post_reset + 100.0
    time_outs = torch.zeros(NUM_ENVS)
    time_outs[time_out_envs] = 1.0
    return post_reset, terminal, time_outs


def test_truncated_envs_store_the_pre_reset_observation() -> None:
    post_reset, terminal, time_outs = _step([1, 3])
    stored, stored_critic = _runner()._terminal_observations(
        {"final_observation": terminal}, time_outs, post_reset, post_reset
    )

    assert torch.equal(stored[[1, 3]], terminal[[1, 3]]), "truncated envs must bootstrap from the terminal obs"
    assert torch.equal(stored[[0, 2]], post_reset[[0, 2]]), "running envs must be left alone"
    # critic obs is the same tensor as the actor obs here, so it gets the same repair
    assert torch.equal(stored_critic, stored)


def test_next_obs_is_not_mutated() -> None:
    """The caller carries `next_obs` forward as the next step's policy input."""
    post_reset, terminal, time_outs = _step([0])
    before = post_reset.clone()
    _runner()._terminal_observations({"final_observation": terminal}, time_outs, post_reset, post_reset)
    assert torch.equal(post_reset, before)


def test_no_truncation_is_a_passthrough(log_records: list[logging.LogRecord]) -> None:
    post_reset, terminal, _ = _step([])
    runner = _runner()
    stored, _ = runner._terminal_observations(
        {"final_observation": terminal}, torch.zeros(NUM_ENVS), post_reset, post_reset
    )
    assert stored is post_reset
    assert not runner._final_obs_warned
    assert not log_records


def test_missing_final_observation_warns_once(log_records: list[logging.LogRecord]) -> None:
    post_reset, _, time_outs = _step([2])
    runner = _runner()
    for _ in range(3):
        stored, _ = runner._terminal_observations({}, time_outs, post_reset, post_reset)
        assert stored is post_reset
    warnings = [r for r in log_records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "a per-step warning would drown the training log"
    assert "final_observation" in warnings[0].getMessage()


def test_shape_mismatch_is_ignored_and_warned(log_records: list[logging.LogRecord]) -> None:
    """Vision runs publish the privileged state here; it is not an actor observation."""
    post_reset, _, time_outs = _step([1])
    privileged_terminal = torch.ones(NUM_ENVS, NUM_OBS + 5)
    runner = _runner()
    stored, _ = runner._terminal_observations(
        {"final_observation": privileged_terminal}, time_outs, post_reset, post_reset
    )
    assert stored is post_reset
    assert "ignoring it" in log_records[0].getMessage()


def test_distinct_privileged_critic_obs_is_left_alone() -> None:
    """Only the actor observation can be repaired from a single final_observation."""
    post_reset, terminal, time_outs = _step([0])
    critic_obs = torch.full((NUM_ENVS, NUM_OBS), -1.0)
    stored, stored_critic = _runner()._terminal_observations(
        {"final_observation": terminal}, time_outs, post_reset, critic_obs
    )
    assert torch.equal(stored[0], terminal[0])
    assert stored_critic is critic_obs


def test_no_env_publishes_the_legacy_nested_key() -> None:
    """Guard the deletion: `observations.final` had no producer anywhere."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "safe_rl"
    pattern = re.compile(r"""\[\s*["']final["']\s*\]|get\(\s*["']final["']""")
    offenders = [p for p in root.rglob("*.py") if pattern.search(p.read_text())]
    assert not offenders, f"unexpected 'final' observation key in {offenders}"
