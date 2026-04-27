from __future__ import annotations

import importlib

import pytest

from tests.conftest import has_module

pytest.importorskip("torch")


@pytest.mark.parametrize(
    "module_name",
    [
        "safe_rl",
        "safe_rl.algorithms",
        "safe_rl.modules",
        "safe_rl.storage",
        "safe_rl.envs",
        "safe_rl.env",
        "safe_rl.runners",
        "safe_rl.utils",
    ],
)
def test_core_package_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_legacy_env_namespace_reexports_envs_symbols() -> None:
    envs = importlib.import_module("safe_rl.envs")
    legacy_env = importlib.import_module("safe_rl.env")

    assert legacy_env.VecEnv is envs.VecEnv
    assert legacy_env.MjlabVecEnv is envs.MjlabVecEnv
    assert legacy_env.SafetyGymnasiumVecEnv is envs.SafetyGymnasiumVecEnv


@pytest.mark.skipif(not has_module("mjlab"), reason="mjlab is optional")
def test_mjlab_wrapper_imports_when_mjlab_is_available() -> None:
    from safe_rl.envs import MjlabVecEnv

    assert MjlabVecEnv is not None


@pytest.mark.skipif(not has_module("safety_gymnasium"), reason="safety-gymnasium is optional")
def test_safety_gymnasium_wrapper_imports_when_available() -> None:
    from safe_rl.envs import SafetyGymnasiumVecEnv

    assert SafetyGymnasiumVecEnv is not None
