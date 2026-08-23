"""Episode metrics emitted at different cadences must all reach the logger.

Keying the metric list off ``ep_infos[0]`` drops whatever is absent from the first entry
-- which is how Episode/goals_reached disappeared once a per-step key was added.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safe_rl.utils.logger import Logger  # noqa: E402


class _Writer:
    def __init__(self):
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag, value, it):
        self.scalars[tag] = value


def _logger_with(ep_infos):
    log = Logger.__new__(Logger)
    log.log_dir = "/tmp"
    log.device = "cpu"
    log.ep_infos = ep_infos
    log.writer = _Writer()
    log.logger_type = "tensorboard"
    return log


def _emit(log, it=0):
    """Run just the episode-metrics block the way Logger.log does."""
    for key in dict.fromkeys(k for ep_info in log.ep_infos for k in ep_info):
        infotensor = torch.tensor([], device=log.device)
        for ep_info in log.ep_infos:
            if key not in ep_info:
                continue
            val = ep_info[key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor([val])
            if len(val.shape) == 0:
                val = val.unsqueeze(0)
            infotensor = torch.cat((infotensor, val.to(log.device)))
        if infotensor.numel() > 0:
            log.writer.add_scalar(f"Episode/{key}", torch.mean(infotensor).item(), it)


def test_metric_absent_from_the_first_entry_is_still_emitted() -> None:
    # Step 1 has only the per-step curriculum key; goals arrive later, on episode end.
    log = _logger_with([
        {"cost_limit": 60.0},
        {"cost_limit": 60.0},
        {"cost_limit": 50.0, "goals_reached": torch.tensor([4.0, 6.0])},
    ])
    _emit(log)
    assert "Episode/goals_reached" in log.writer.scalars
    assert log.writer.scalars["Episode/goals_reached"] == pytest.approx(5.0)
    assert log.writer.scalars["Episode/cost_limit"] == pytest.approx(56.666667)


def test_all_keys_present_still_works() -> None:
    log = _logger_with([
        {"goals_reached": torch.tensor([2.0])},
        {"goals_reached": torch.tensor([4.0])},
    ])
    _emit(log)
    assert log.writer.scalars["Episode/goals_reached"] == pytest.approx(3.0)
