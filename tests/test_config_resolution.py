from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
SAFE_ALGORITHMS = {
    "CPO",
    "CUP",
    "FOCOPS",
    "FPPO",
    "P3O",
    "PCPO",
    "PCRPO",
    "PPOL_PID",
    "SafeSAC",
}


def _config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("config_path", _config_paths(), ids=lambda path: str(path.relative_to(PROJECT_ROOT)))
def test_config_class_names_resolve(config_path: Path) -> None:
    import safe_rl.algorithms as algorithms
    import safe_rl.modules as modules
    import safe_rl.runners as runners

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    algorithm_name = cfg["algorithm"]["class_name"]
    policy_name = cfg["policy"]["class_name"]
    runner_name = cfg.get("runner_class_name", "OnPolicyRunner")

    assert hasattr(algorithms, algorithm_name), f"{config_path} references unknown algorithm {algorithm_name!r}"
    assert hasattr(modules, policy_name), f"{config_path} references unknown policy {policy_name!r}"
    assert hasattr(runners, runner_name), f"{config_path} references unknown runner {runner_name!r}"


@pytest.mark.parametrize("config_path", _config_paths(), ids=lambda path: str(path.relative_to(PROJECT_ROOT)))
def test_safe_algorithm_cost_limits_are_valid_when_present(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg["algorithm"]
    algorithm_name = algorithm_cfg["class_name"]
    if algorithm_name not in SAFE_ALGORITHMS or "cost_limits" not in algorithm_cfg:
        return

    cost_limits = algorithm_cfg["cost_limits"]
    assert isinstance(cost_limits, list), f"{config_path} cost_limits must be a list"
    assert cost_limits, f"{config_path} cost_limits must not be empty"
    assert all(isinstance(limit, (float, int)) for limit in cost_limits), (
        f"{config_path} cost_limits must contain only numbers"
    )
