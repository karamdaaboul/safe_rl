"""Guards for the generated paper-comparison configs.

`config/bench/paper/**` is produced by `scripts/bench/gen_bench_configs.py`. These tests
fail if someone hand-edits a generated file, and they pin the values that would silently
break the comparison against the published REPPO numbers if they drifted.

Runs in a bare CPU venv: the generator's `--check` path reads the derived per-task values
from the committed `_derived.json` rather than importing ManiSkill or MuJoCo Playground.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "bench" / "gen_bench_configs.py"
PAPER_DIR = REPO / "config" / "bench" / "paper"

SUITES = [s for s in ("maniskill", "dmc") if (PAPER_DIR / s / "_derived.json").exists()]


@pytest.mark.parametrize("suite", SUITES)
def test_generated_configs_match_the_generator(suite: str) -> None:
    """A hand-edit of a generated config must fail here rather than silently ship."""
    result = subprocess.run(
        [sys.executable, str(GEN), "--suite", suite, "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, f"config drift in {suite}:\n{result.stdout}\n{result.stderr}"


def _configs(suite: str) -> list[Path]:
    return sorted((PAPER_DIR / suite).glob("*.yaml"))


@pytest.mark.skipif("maniskill" not in SUITES, reason="ManiSkill configs not generated")
def test_maniskill_gamma_is_derived_from_the_episode_horizon() -> None:
    """gamma = 1 - 10/max_episode_steps, per task.

    This is the value that flatlined a full 50M-step PickCube run when it was wrong,
    and it is NOT constant across the eight paper tasks (0.80 / 0.875 / 0.90).
    """
    derived = json.loads((PAPER_DIR / "maniskill" / "_derived.json").read_text())
    for path in _configs("maniskill"):
        cfg = yaml.safe_load(path.read_text())
        task = path.stem
        horizon = derived[task]["max_episode_steps"]
        assert cfg["algorithm"]["gamma"] == pytest.approx(1.0 - 10.0 / horizon), task


@pytest.mark.skipif("maniskill" not in SUITES, reason="ManiSkill configs not generated")
def test_maniskill_configs_cover_exactly_the_paper_tasks() -> None:
    """PickCube-v1 has no published CSV and must not be in the comparison set."""
    tasks = {p.stem for p in _configs("maniskill")}
    assert "PickCube-v1" not in tasks
    assert tasks == {
        "LiftPegUpright-v1",
        "PegInsertionSide-v1",
        "PickSingleYCB-v1",
        "PokeCube-v1",
        "PullCube-v1",
        "RollBall-v1",
        "UnitreeG1PlaceAppleInBowl-v1",
        "UnitreeG1TransportBox-v1",
    }


@pytest.mark.skipif("maniskill" not in SUITES, reason="ManiSkill configs not generated")
def test_maniskill_configs_request_a_reconfiguring_eval_env() -> None:
    """Evaluating on the training env would inflate success against the paper.

    The reference eval env uses reconfiguration_freq=1; the training env does not
    reconfigure at all. Without the twin, PickSingleYCB-v1 and PokeCube-v1 are scored on
    the single scene instantiation they trained against.
    """
    for path in _configs("maniskill"):
        cfg = yaml.safe_load(path.read_text())
        kwargs = cfg["env"]["kwargs"]
        assert kwargs["eval_reconfiguration_freq"] == 1, path.stem
        assert kwargs["num_eval_envs"] > 0, path.stem


@pytest.mark.parametrize("suite", SUITES)
def test_parity_knobs_are_pinned_explicitly(suite: str) -> None:
    """Every feature our REPPO adds over the reference is pinned, not left to defaults.

    Leaving these implicit means a future change of default silently voids the parity
    claim. `critic_loss_denominator` and `force_last_step_truncated` in particular take
    NON-reference defaults in our code.
    """
    expected_alg = {
        "dual_optim_mode": "actor",
        "critic_loss_denominator": "batch",
        "force_last_step_truncated": True,
        "target_entropy_final": None,
        "reward_normalization": False,
        "optimizer_class": "adam",
        "weight_decay": 0.0,
        "critic_learning_rate": None,
    }
    expected_policy = {"action_scale": 1.0, "squash": "tanh", "critic_type": "reference"}
    for path in _configs(suite):
        cfg = yaml.safe_load(path.read_text())
        for key, want in expected_alg.items():
            assert key in cfg["algorithm"], f"{path.stem}: {key} must be pinned explicitly"
            assert cfg["algorithm"][key] == want, f"{path.stem}: {key}"
        for key, want in expected_policy.items():
            assert cfg["policy"].get(key) == want, f"{path.stem}: {key}"


@pytest.mark.skipif("maniskill" not in SUITES, reason="ManiSkill configs not generated")
def test_bench_runner_keys_survive_the_train_cfg_whitelist() -> None:
    """`load_train_cfg` is an explicit whitelist that SILENTLY drops unlisted runner keys.

    Setting `eval_modes: [ode]` in a config and having it ignored is invisible at
    runtime — the run simply costs twice as much eval and logs a metric we do not
    compare. This asserts the keys the benchmark depends on actually reach the runner.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_train_sg", REPO / "scripts" / "train" / "train_safety_gymnasium.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg_path = PAPER_DIR / "maniskill" / "PullCube-v1.yaml"
    train_cfg, env_cfg, max_iterations, _, _ = module.load_train_cfg(str(cfg_path))

    assert train_cfg["eval_modes"] == ["ode"], "eval_modes was dropped by the whitelist"
    assert train_cfg["eval_interval"] == 19
    assert train_cfg["eval_episodes"] == 1024
    assert train_cfg["log_env_steps"] is True
    assert max_iterations == 381
    # env.kwargs must reach make_env, or the eval twin is never built.
    assert env_cfg["kwargs"]["num_eval_envs"] == 256
    assert env_cfg["kwargs"]["eval_reconfiguration_freq"] == 1


@pytest.mark.parametrize("suite", SUITES)
def test_budget_and_eval_grid_match_the_paper(suite: str) -> None:
    """50M env steps at 1024 envs, and a deterministic-only eval on ~20 points."""
    for path in _configs(suite):
        cfg = yaml.safe_load(path.read_text())["runner"]
        assert cfg["num_steps_per_env"] == 128
        assert cfg["max_iterations"] == 381
        assert 1024 * 128 * 381 == 49_938_432  # the budget the report must quote
        assert cfg["eval_modes"] == ["ode"], f"{path.stem}: the paper's headline is deterministic"
        assert cfg["log_env_steps"] is True
        n_points = len(range(0, cfg["max_iterations"], cfg["eval_interval"]))
        assert 18 <= n_points <= 22, f"{path.stem}: {n_points} eval points, want ~20"
