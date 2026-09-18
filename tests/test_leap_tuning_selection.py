"""Selection must respect the declared metric, baseline controls and compute cap."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("leap_tuning", Path(__file__).parents[1] / "scripts/queue/leap_tuning.py")
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)


@pytest.mark.parametrize("improves", [False, True])
def test_selection_and_budget(tmp_path, improves):
    c = tmp_path / "campaign"
    baseline = tmp_path / "baseline"
    manifest = {"baseline": str(baseline), "runs": []}
    for name, score in [("sac", 2), ("mpo", 1), ("mpo_one_step", 3)]:
        tuning.write(baseline / name / "eval_deterministic.json", {"mean_reward": score})
    for index, name in enumerate(tuning.VARIANTS):
        tuning.add_run(c, manifest, name + "_seed1", name, 1, 8000000, "screen")
        tuning.write(
            c / "runs" / f"{name}_seed1" / "eval_8000000_deterministic.json",
            {"mean_reward": index + 10 if improves else -10},
        )
        tuning.write(c / "runs" / f"{name}_seed1" / "eval_8000000_stochastic.json", {"mean_reward": 1000 - index})
    tuning.write(c / "manifest.json", manifest)
    tuning.select(c)
    selected = tuning.read(c / "selection.json")["chosen"]
    assert selected == (
        {"sac": "sac_fixed_alpha", "mpo": "mpo1_relaxed_kl"} if improves else {"sac": "sac", "mpo": "mpo_one_step"}
    )
    runs = tuning.read(c / "manifest.json")["runs"]
    assert len({r["name"] for r in runs}) == len(runs)
    assert sum(r["steps"] for r in runs) <= 208000000
    for family in ("sac", "mpo"):
        selected_runs = [r for r in runs if r["name"].startswith(family + "_selected_")]
        assert [r["seed"] for r in selected_runs] == [1, 2, 3]
        assert all(r["steps"] == 20000000 for r in selected_runs)
    controls = [r for r in runs if "_control_" in r["name"]]
    assert len(controls) == (2 if improves else 0)
