"""Wait for the Ant comparison, diagnose MPO, and run a bounded adaptive tuning pass.

Uses the comparison's frozen code and existing training/evaluation CLIs. Selection uses
evaluation seed 10001; seed 20001 is opened only after the final configuration is frozen.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import yaml

from ant_comparison import MODES, ROOT, environment, now, read_json, write_json


INITIAL_VARIANTS = [
    ("wider_mean", {"kl_mean_constraint": 0.005}, "Allow faster changes to the policy mean."),
    ("hard_target", {"target_actor_update": "hard", "target_actor_period": 100},
     "Test a fixed policy anchor with a copy every 100 E/M cycles."),
    ("mean_regularized", {"mean_l2_coef": 0.001}, "Test whether reducing mean drift improves movement."),
    ("single_mstep", {"mstep_iteration_num": 1}, "Test less reuse of each weighted action batch."),
    ("tighter_variance", {"kl_var_constraint": 3.125e-6}, "Slow changes in exploration variance."),
    ("more_actions", {"sample_action_num": 128}, "Reduce Monte Carlo noise in E-step action integration."),
]
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def telemetry(log):
    """Summarize the last 20 logged update blocks, retaining full-run solver failures."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    chunks = re.split(r"Learning iteration\s+(\d+)/\d+", text)
    blocks = []
    for index in range(1, len(chunks), 2):
        values = {}
        for line in chunks[index + 1].splitlines():
            if ":" not in line:
                continue
            key, value = map(str.strip, line.split(":", 1))
            key = key.lower().replace(" ", "_")
            pairs = re.findall(r"([a-z_]+)=(" + NUMBER + r")", value)
            if pairs:
                values.update({f"{key}.{name}": float(number) for name, number in pairs})
            elif re.fullmatch(NUMBER, value):
                values[key] = float(value)
        blocks.append((int(chunks[index]), values))
    tail = blocks[-20:]
    keys = set().union(*(values.keys() for _, values in tail)) if tail else set()
    means = {key: statistics.mean(values[key] for _, values in tail if key in values) for key in sorted(keys)}
    return {
        "last_iteration": blocks[-1][0] if blocks else None,
        "tail_blocks": len(tail), "tail_mean": means,
        "solver_failure_blocks": sum(values.get("solver.status", 0) != 0 for _, values in blocks),
        "nonfinite_logged_values": bool(re.search(r"(?:=|:)\s*[-+]?(?:nan|inf)\b", text, re.I)),
    }


def validate_evaluation(result, mode, seed, episodes=50):
    expected = {"env_id": "Ant-Flat", "episodes_collected": episodes, "num_envs": episodes,
                "one_episode_per_env": True, "action_mode": mode, "seed": seed, "max_episode_length": 960}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Evaluation protocol mismatch for {key}: {result.get(key)} != {value}")
    for key in ("mean_reward", "mean_length", "survival_rate"):
        if not math.isfinite(result[key]):
            raise ValueError(f"Non-finite evaluation metric: {key}")
    if len(result["per_episode"]["reward"]) != episodes:
        raise ValueError("Missing per-episode returns")


def aggregate(results):
    summary = {}
    for mode in MODES:
        selected = [item[mode] for item in results]
        rewards = [item["mean_reward"] for item in selected]
        summary[mode] = {
            "seeds": len(rewards), "mean_return": statistics.mean(rewards),
            "seed_sd": statistics.stdev(rewards) if len(rewards) > 1 else None,
            "seed_returns": rewards,
            "survival": statistics.mean(item["survival_rate"] for item in selected),
            "mean_length": statistics.mean(item["mean_length"] for item in selected),
        }
    return summary


def comparison(candidate, reference, tolerance=0.05):
    checks = {}
    for mode in MODES:
        baseline = reference[mode]["mean_return"]
        difference = candidate[mode]["mean_return"] - baseline
        margin = tolerance * max(abs(baseline), 1.0)
        survival_gap = candidate[mode]["survival"] - reference[mode]["survival"]
        checks[mode] = {"return_difference": difference, "allowed_return_drop": margin,
                        "survival_difference": survival_gap,
                        "passes": difference >= -margin and survival_gap >= -0.05 - 1e-12}
    return {"passes": all(check["passes"] for check in checks.values()), "modes": checks}


def selection_score(summary, reference):
    # Score the weaker of deterministic and stochastic performance; avoid a mode-specific winner.
    return min((summary[mode]["mean_return"] - reference[mode]["mean_return"])
               / max(abs(reference[mode]["mean_return"]), 1.0) for mode in MODES)


def adaptive_variants(best):
    """Two local follow-ups around the best single-change screening result."""
    regularized = deepcopy(best)
    regularized["mean_l2_coef"] = 0.003 if best.get("mean_l2_coef", 0) else 0.001
    mean = deepcopy(best)
    mean["kl_mean_constraint"] = min(0.01, 2 * best.get("kl_mean_constraint", 0.00125))
    return [("adaptive_regularized", regularized, "Add or strengthen mean regularization around the screening winner."),
            ("adaptive_mean", mean, "Increase the mean KL budget around the screening winner.")]


def prepare(baseline, output, gpu):
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    manifest = read_json(baseline / "manifest.json")
    if manifest["seeds"] != [1, 2, 3] or manifest["steps_per_run"] != 8192000:
        raise ValueError("This follow-up requires the three-seed, 8.192M-step comparison")
    output.mkdir(parents=True)
    shutil.copytree(baseline / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(baseline / "configs", output / "baseline_configs")
    source_hashes = {str(path.relative_to(output / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (output / "source").rglob("*.py")}
    if source_hashes != manifest["source_sha256"]:
        raise RuntimeError("The source snapshot no longer matches the comparison manifest")
    (output / "configs").mkdir()
    for name, updates, _ in INITIAL_VARIANTS:
        write_config(output, name, updates)
    shutil.copy2(Path(__file__), output / "controller.py")
    shutil.copy2(Path(__file__).with_name("ant_comparison.py"), output / "ant_comparison.py")
    plan = {
        "created_utc": now(), "baseline": str(baseline), "python": manifest["python"], "gpu": gpu,
        "steps_per_run": 8192000, "training_seeds": [1, 2, 3], "screen_seed": 1,
        "development_eval_seed": 10001, "heldout_eval_seed": 20001, "evaluation_episodes": 50,
        "return_tolerance": 0.05, "survival_tolerance": 0.05,
        "max_screen_configs": 8, "max_finalists": 2, "max_additional_training_runs": 12,
        "source_sha256": source_hashes,
        "variants": [{"name": name, "updates": updates, "reason": reason}
                     for name, updates, reason in INITIAL_VARIANTS],
        "baseline_manifest": manifest,
    }
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"phase": "prepared", "updated_utc": now(), "trials": {}})
    render(output)


def write_config(output, name, updates):
    config = yaml.safe_load((output / "baseline_configs/mpo.yaml").read_text())
    config["algorithm"].update(updates)
    # Keep only periodic and final checkpoints; environment and optimizer budgets stay fixed.
    config["runner"]["save_interval"] = 16000
    path = output / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def render(output):
    state = read_json(output / "status.json")
    lines = ["# Ant MPO follow-up", "", f"Updated: {state['updated_utc']}", "",
             f"Status: **{state['phase']}**", "", state.get("message", ""), "",
             "## Protocol", "",
             "8,192,000 transitions per training run. Three training seeds for the final comparison.",
             "50 first episodes per evaluation, deterministic and stochastic actions.",
             "Development seed 10001; held-out seed 20001 is used only after selection is frozen.",
             "Target: within 5% of SAC return in both modes, and survival no more than 5 percentage points lower.",
             "Eight screening configurations and at most two three-seed finalists (12 additional runs maximum).", "",
             "## Trials", "", "| Trial | State | Deterministic return | Stochastic return |", "|---|---|---:|---:|"]
    for name, trial in state.get("trials", {}).items():
        values = trial.get("evaluations", {})
        scores = [f"{values[mode]['mean_reward']:.3f}" if mode in values else "pending" for mode in MODES]
        lines.append(f"| {name} | {trial['status']} | {scores[0]} | {scores[1]} |")
    analysis = state.get("analysis")
    if analysis:
        lines += ["", "## Baseline analysis", "", *[f"- {item}" for item in analysis["findings"]], ""]
        for name, summary in analysis["returns"].items():
            for mode in MODES:
                metric = summary[mode]
                lines.append(f"- {name.upper()} {mode}: {metric['mean_return']:.3f} ± {metric['seed_sd']:.3f}; "
                             f"survival {metric['survival']:.1%}.")
    if "selected" in state:
        lines += ["", "## Frozen selection", "", f"Configuration: **{state['selected']}**."]
    if "heldout" in state:
        lines += ["", "## Held-out evaluation", "",
                  "| Algorithm | Mode | Return (mean ± seed SD) | Survival |", "|---|---|---:|---:|"]
        for name in ("sac", "mpo"):
            for mode, metric in state["heldout"][name].items():
                lines.append(f"| {name.upper()} | {mode} | {metric['mean_return']:.3f} ± "
                             f"{metric['seed_sd']:.3f} | {metric['survival']:.1%} |")
        lines += ["", "This is a practical performance threshold, not a statistical equivalence claim."]
    temporary = output / "report.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "report.md")


class Controller:
    def __init__(self, output):
        self.output = output
        self.plan = read_json(output / "plan.json")
        self.baseline = Path(self.plan["baseline"])
        self.state = read_json(output / "status.json")
        self.env = environment(output, self.plan["gpu"])

    def save(self, **updates):
        self.state.update(updates, updated_utc=now())
        write_json(self.output / "status.json", self.state)
        render(self.output)

    def run(self, command, log, timeout=21600):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb", buffering=0) as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            self.save(active_command=command, active_pid=child.pid, active_log=str(log))
            started = time.monotonic()
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    self.save()
                    if time.monotonic() - started > timeout:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise TimeoutError(f"Command exceeded {timeout}s: {log}")
        self.save(active_pid=None)
        if code:
            raise RuntimeError(f"Command exited {code}: {log}")

    def wait_for_baseline(self):
        while True:
            statuses = {f"{name}_seed{seed}": read_json(self.baseline / f"{name}_seed{seed}" / "status.json")
                        for name in ("ppo", "sac", "mpo") for seed in (1, 2, 3)}
            failed = [name for name, status in statuses.items() if status["status"] == "failed"]
            if failed:
                raise RuntimeError(f"Baseline failures require recovery before tuning: {failed}")
            if all(status["status"] == "complete" for status in statuses.values()):
                return
            self.save(phase="waiting_for_baseline",
                      baseline_status={key: item["status"] for key, item in statuses.items()},
                      message="Waiting for all nine original runs and evaluations; no tuning training has started.")
            time.sleep(60)

    def baseline_results(self, name):
        results = []
        for seed in (1, 2, 3):
            modes = {}
            for mode in MODES:
                modes[mode] = read_json(self.baseline / f"{name}_seed{seed}" / f"eval_{mode}.json")
                validate_evaluation(modes[mode], mode, self.plan["development_eval_seed"])
            results.append(modes)
        return results

    def analyse(self):
        returns = {name: aggregate(self.baseline_results(name)) for name in ("sac", "mpo")}
        diagnostics = {f"{name}_seed{seed}": telemetry(self.baseline / f"{name}_seed{seed}" / "train.log")
                       for name in ("sac", "mpo") for seed in (1, 2, 3)}
        findings = []
        for name in ("mpo", "sac"):
            for metric in ("mean_episode_episode_reward/progress", "mean_episode_episode_reward/energy",
                           "frac_saturated", "kl.mean_max_rel", "kl.var_max_rel", "ess", "dual_residual_eta"):
                values = [diagnostics[f"{name}_seed{seed}"]["tail_mean"][metric] for seed in (1, 2, 3)
                          if metric in diagnostics[f"{name}_seed{seed}"]["tail_mean"]]
                if values:
                    findings.append(f"{name.upper()} {metric}: {statistics.mean(values):.6g} "
                                    "(mean over seed tail summaries).")
        failures = sum(item["solver_failure_blocks"] for name, item in diagnostics.items() if name.startswith("mpo"))
        findings.append(f"MPO logged E-step solver failure blocks: {failures}.")
        findings.append("Movement, saturation, and trust-region diagnostics guide hypotheses, not causal conclusions.")
        result = {"returns": returns, "diagnostics": diagnostics, "findings": findings,
                  "comparison": comparison(returns["mpo"], returns["sac"])}
        write_json(self.output / "baseline_analysis.json", result)
        self.save(phase="analysing", analysis=result, message="Baseline analysis complete.")
        if failures or any(item["nonfinite_logged_values"] for item in diagnostics.values()):
            raise RuntimeError("Non-finite values or solver failures require a code investigation before tuning")
        return result

    def evaluate(self, checkpoint, config, destination, evaluation_seed):
        results = {}
        for mode in MODES:
            path = destination / f"eval_{mode}.json"
            if not path.exists():
                command = [self.plan["python"], "-u", str(self.output / "source/scripts/eval/unitree_mjlab.py"),
                           "--env_id", "Ant-Flat", "--checkpoint", str(checkpoint), "--config", str(config),
                           "--num_envs", "50", "--episodes", "50", "--one_episode_per_env", "--headless",
                           "--device", "cuda:0", "--seed", str(evaluation_seed), "--action_mode", mode,
                           "--json_out", str(path)]
                self.run(command, destination / f"eval_{mode}.log", timeout=900)
            results[mode] = read_json(path)
            validate_evaluation(results[mode], mode, evaluation_seed)
            if Path(results[mode]["checkpoint"]).resolve() != Path(checkpoint).resolve():
                raise ValueError(f"Evaluation checkpoint mismatch: {path}")
        return results

    def train(self, name, seed):
        key = f"{name}_seed{seed}"
        destination = self.output / "trials" / key
        status = self.state["trials"].get(key)
        if status and status["status"] == "complete":
            return status
        if status and status["status"] == "failed":
            return None
        checkpoints = list((self.output / "runs" / key).glob("*/model_31999.pt"))
        if len(checkpoints) > 1:
            raise RuntimeError(f"Ambiguous final checkpoints: {key}")
        # A stopped controller must never silently start a second training process or overwrite a partial run.
        if (destination / "train.log").exists() and not checkpoints:
            raise RuntimeError(f"Incomplete existing training needs explicit recovery: {destination}")
        config = self.output / "configs" / f"{name}.yaml"
        status = {"status": "training", "name": name, "seed": seed, "config": str(config)}
        self.state["trials"][key] = status
        self.save(message=f"Running {key} at the original 8.192M-transition budget.")
        try:
            if not checkpoints:
                command = [self.plan["python"], "-u", str(self.output / "source/scripts/train/unitree_mjlab.py"),
                           "--env_id", "Ant-Flat", "--config", str(config), "--num_envs", "256", "--seed", str(seed),
                           "--gpu_ids", "0", "--device", "cuda:0", "--experiment_name", key,
                           "--run_name", f"mpo_followup_{key}", "--log_dir", str(self.output / "runs")]
                self.run(command, destination / "train.log")
                checkpoints = list((self.output / "runs" / key).glob("*/model_31999.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"Missing final checkpoint: {key}")
            status.update(status="evaluating", checkpoint=str(checkpoints[0]))
            self.save()
            results = self.evaluate(checkpoints[0], config, destination, self.plan["development_eval_seed"])
            diag = telemetry(destination / "train.log")
            if diag["last_iteration"] != 31999 or diag["solver_failure_blocks"] or diag["nonfinite_logged_values"]:
                raise RuntimeError(f"Training completeness/numerical diagnostics failed: {key}")
            status.update(status="complete", evaluations=results, diagnostics=diag, finished_utc=now())
        except (RuntimeError, ValueError, TimeoutError) as error:
            status.update(status="failed", error=str(error), finished_utc=now())
        self.save()
        return status if status["status"] == "complete" else None

    def choose(self, reference):
        candidates = self.plan["variants"][:]
        for variant in candidates:
            self.save(phase="screening", message=variant["reason"])
            self.train(variant["name"], 1)
        initial_names = {variant["name"] for variant in candidates}
        eligible = [trial for trial in self.state["trials"].values() if trial["status"] == "complete"
                    and trial["seed"] == 1 and trial["name"] in initial_names]
        if not eligible:
            raise RuntimeError("All initial screening runs failed")
        best = max(eligible, key=lambda trial: selection_score(aggregate([trial["evaluations"]]), reference))
        best_updates = next(variant["updates"] for variant in candidates if variant["name"] == best["name"])
        for name, updates, reason in adaptive_variants(best_updates):
            if not (self.output / "configs" / f"{name}.yaml").exists():
                write_config(self.output, name, updates)
            variant = {"name": name, "updates": updates, "reason": reason}
            candidates.append(variant)
            self.save(phase="adaptive_screening", adaptive_variants=candidates[6:], message=reason)
            self.train(name, 1)
        eligible = [trial for trial in self.state["trials"].values()
                    if trial["seed"] == 1 and trial["status"] == "complete"]
        eligible.sort(key=lambda trial: selection_score(aggregate([trial["evaluations"]]), reference), reverse=True)
        finalists = []
        for candidate in eligible[:self.plan["max_finalists"]]:
            name = candidate["name"]
            self.save(phase="confirming_three_seeds", message=f"Confirming {name} with training seeds 2 and 3.")
            trials = [candidate, self.train(name, 2), self.train(name, 3)]
            if all(trials):
                summary = aggregate([trial["evaluations"] for trial in trials])
                finalists.append((name, summary))
                if comparison(summary, reference)["passes"]:
                    break
        # Include the untouched baseline, so a failed sweep cannot silently replace it with a worse policy.
        finalists.append(("baseline", self.state["analysis"]["returns"]["mpo"]))
        self.save(finalists={name: summary for name, summary in finalists})
        return max(finalists, key=lambda pair: (comparison(pair[1], reference)["passes"],
                                               selection_score(pair[1], reference)))[0]

    def heldout(self, selected):
        self.save(phase="heldout_evaluation", selected=selected,
                  message="Configuration selection is frozen. Evaluating SAC and MPO on the unused evaluation seed.")
        summaries = {}
        for name in ("sac", "mpo"):
            results = []
            for seed in (1, 2, 3):
                if name == "sac" or selected == "baseline":
                    checkpoint = read_json(self.baseline / f"{name}_seed{seed}" / "status.json")["checkpoint"]
                    config = self.output / "baseline_configs" / f"{name}.yaml"
                else:
                    checkpoint = self.state["trials"][f"{selected}_seed{seed}"]["checkpoint"]
                    config = self.output / "configs" / f"{selected}.yaml"
                destination = self.output / "heldout" / f"{name}_seed{seed}"
                results.append(self.evaluate(checkpoint, config, destination, self.plan["heldout_eval_seed"]))
            summaries[name] = aggregate(results)
        check = comparison(summaries["mpo"], summaries["sac"])
        development = self.state.get("finalists", {}).get(selected, self.state["analysis"]["returns"]["mpo"])
        development_check = comparison(development, self.state["analysis"]["returns"]["sac"])
        passed = check["passes"] and development_check["passes"]
        write_json(self.output / "heldout_results.json", {**summaries, "comparison": check,
                                                          "development_comparison": development_check})
        self.save(phase="target_met" if passed else "target_not_met", heldout=summaries,
                  heldout_comparison=check, development_comparison=development_check,
                  message=("MPO met the predefined return and survival targets on both evaluation seeds."
                           if passed else "The tuning pass did not meet every target. Results are preserved; "
                           "further changes require a new investigation and a new held-out evaluation seed."))
        source = self.output / ("baseline_configs/mpo.yaml" if selected == "baseline" else f"configs/{selected}.yaml")
        shutil.copy2(source, self.output / "selected_config.yaml")

    def execute(self):
        if self.state["phase"] in ("target_met", "target_not_met"):
            return
        self.wait_for_baseline()
        analysis = self.analyse()
        if "selected" in self.state:
            selected = self.state["selected"]  # Never select again after looking at held-out results.
        elif analysis["comparison"]["passes"]:
            selected = "baseline"
        else:
            selected = self.choose(analysis["returns"]["sac"])
        self.heldout(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "report"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "logs/ant_ppo_sac_mpo_20260915")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/ant_mpo_followup_20260915")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(args.baseline.resolve(), output, args.gpu)
    elif args.action == "report":
        render(output)
    elif args.action == "launch":
        # A kernel-held lock also works when a sandbox cannot inspect host process IDs.
        with (output / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Follow-up worker already running") from error
            plan = read_json(output / "plan.json")
            with (output / "worker.log").open("ab", buffering=0) as log:
                child = subprocess.Popen([plan["python"], "-u", str(output / "controller.py"), "worker",
                                          "--output", str(output), "--lock-fd", str(lock.fileno())], cwd=ROOT,
                                         env=environment(output, plan["gpu"]), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock.fileno(),))
            (output / "worker.pid").write_text(f"{child.pid}\n")
        print(f"Started MPO follow-up worker PID {child.pid}; report: {output / 'report.md'}")
    else:
        lock_file = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (output / "worker.lock").open("a")
        with lock_file as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            controller = Controller(output)
            try:
                controller.execute()
            except Exception as error:
                controller.save(phase="needs_review", message=str(error))
                raise


if __name__ == "__main__":
    main()
