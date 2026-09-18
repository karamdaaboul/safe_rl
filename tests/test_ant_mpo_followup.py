"""Check gating, selection, and evaluation isolation for the unattended Ant follow-up."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

QUEUE = Path(__file__).resolve().parents[1] / "scripts/queue"
sys.path.insert(0, str(QUEUE))
spec = importlib.util.spec_from_file_location("ant_mpo_followup", QUEUE / "ant_mpo_followup.py")
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


def metrics(reward=100, survival=0.9):
    return {mode: {"mean_reward": reward, "survival_rate": survival, "mean_length": 900}
            for mode in followup.MODES}


def test_threshold_checks_both_modes_and_survival():
    reference = followup.aggregate([metrics()] * 3)
    candidate = followup.aggregate([metrics(95, 0.85)] * 3)
    assert followup.comparison(candidate, reference)["passes"]
    candidate["stochastic"]["mean_return"] = 94.9
    assert not followup.comparison(candidate, reference)["passes"]
    candidate["stochastic"]["mean_return"] = 100
    candidate["deterministic"]["survival"] = 0.849
    assert not followup.comparison(candidate, reference)["passes"]


def test_aggregate_uses_training_seeds_as_statistical_units():
    summary = followup.aggregate([metrics(10), metrics(20), metrics(30)])
    assert summary["deterministic"]["seeds"] == 3
    assert summary["deterministic"]["mean_return"] == 20
    assert summary["deterministic"]["seed_sd"] == 10


def test_evaluation_protocol_rejects_wrong_seed_or_episode_cap():
    result = {"env_id": "Ant-Flat", "episodes_collected": 50, "num_envs": 50,
              "one_episode_per_env": True, "action_mode": "deterministic", "seed": 20001,
              "max_episode_length": 960, **metrics()["deterministic"],
              "per_episode": {"reward": [100] * 50}}
    followup.validate_evaluation(result, "deterministic", 20001)
    for key, value in (("seed", 10001), ("max_episode_length", 1000), ("episodes_collected", 49)):
        changed = {**result, key: value}
        with pytest.raises(ValueError, match="protocol mismatch"):
            followup.validate_evaluation(changed, "deterministic", 20001)


def test_log_analysis_separates_constraint_usage_and_detects_failure(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Learning iteration 100/32000\n"
                   "Kl: mean_rel=0.8 var_rel=0.9 var_max_rel=1.5 q=0.1000\n"
                   "Solver: status=0.0000 iters=8.0000\n"
                   "Mean episode Episode_Reward/progress: 0.8\n"
                   "Learning iteration 31999/32000\n"
                   "Kl: mean_rel=0.6 var_rel=0.7 var_max_rel=1.1 q=0.1000\n"
                   "Solver: status=1.0000 iters=8.0000\nMean critic loss: nan\n")
    values = followup.telemetry(log)
    assert values["tail_mean"]["kl.mean_rel"] == pytest.approx(0.7)
    assert values["tail_mean"]["kl.var_max_rel"] == pytest.approx(1.3)
    assert values["solver_failure_blocks"] == 1
    assert values["nonfinite_logged_values"]


def test_wait_gate_never_runs_tuning_with_incomplete_baseline(tmp_path, monkeypatch):
    controller = object.__new__(followup.Controller)
    controller.baseline = tmp_path
    observed = []
    for name in ("ppo", "sac", "mpo"):
        for seed in (1, 2, 3):
            status = "training" if (name, seed) == ("mpo", 3) else "complete"
            followup.write_json(tmp_path / f"{name}_seed{seed}/status.json", {"status": status})
    controller.save = lambda **updates: observed.append(updates)

    def complete_after_poll(_):
        assert observed[-1]["phase"] == "waiting_for_baseline"
        followup.write_json(tmp_path / "mpo_seed3/status.json", {"status": "complete"})

    monkeypatch.setattr(followup.time, "sleep", complete_after_poll)
    controller.wait_for_baseline()
    assert len(observed) == 1
    followup.write_json(tmp_path / "mpo_seed3/status.json", {"status": "failed"})
    with pytest.raises(RuntimeError, match="Baseline failures"):
        controller.wait_for_baseline()


def test_frozen_selection_is_not_reopened_after_heldout(tmp_path):
    controller = object.__new__(followup.Controller)
    controller.state = {"phase": "needs_review", "selected": "wider_mean"}
    controller.wait_for_baseline = lambda: None
    controller.analyse = lambda: {"comparison": {"passes": False}, "returns": {"sac": {}}}
    controller.choose = lambda _: pytest.fail("Must not select again after held-out access")
    observed = []
    controller.heldout = observed.append
    controller.execute()
    assert observed == ["wider_mean"]


def test_finalist_selection_requires_three_seeds_and_prefers_passing_survival(tmp_path):
    controller = object.__new__(followup.Controller)
    controller.output = tmp_path
    (tmp_path / "configs").mkdir()
    controller.plan = {"variants": [{"name": name, "updates": updates, "reason": reason}
                                   for name, updates, reason in followup.INITIAL_VARIANTS], "max_finalists": 2}
    reference = followup.aggregate([metrics()] * 3)
    # Baseline has high return but misses the survival target.
    controller.state = {"trials": {}, "analysis": {"returns": {"mpo": followup.aggregate([metrics(110, 0.7)] * 3)}}}
    controller.save = lambda **updates: controller.state.update(updates)
    calls = []

    def train(name, seed):
        calls.append((name, seed))
        trial = {"name": name, "seed": seed, "status": "complete", "evaluations": metrics(96)}
        controller.state["trials"][f"{name}_seed{seed}"] = trial
        return trial

    controller.train = train
    for name in ("adaptive_regularized", "adaptive_mean"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("{}")
    winner = controller.choose(reference)
    assert winner != "baseline"
    assert {(winner, seed) for seed in (1, 2, 3)}.issubset(set(calls))
    assert len(calls) <= 12


@pytest.mark.parametrize("name,updates,reason", followup.INITIAL_VARIANTS)
def test_candidate_config_retains_training_budget_and_timeout_fix(tmp_path, name, updates, reason):
    baseline = Path(__file__).resolve().parents[1] / "config/ant_comparison/mpo.yaml"
    (tmp_path / "baseline_configs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "baseline_configs/mpo.yaml").write_text(baseline.read_text())
    config = yaml.safe_load(followup.write_config(tmp_path, name, updates).read_text())
    assert config["runner"]["max_iterations"] * config["runner"]["num_steps_per_env"] * 256 == 8192000
    assert config["algorithm"]["gamma"] == 0.99
    assert config["algorithm"]["alpha_mean_max"] is None
    assert config["algorithm"]["alpha_var_max"] is None
    assert config["algorithm"].get("estep_sample_std_scale", 1) == 1
    assert config["runner"]["n_step"] == 1
