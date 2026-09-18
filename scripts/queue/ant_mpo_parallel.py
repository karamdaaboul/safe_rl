"""Run two requested MPO ablations concurrently using frozen code and existing CLIs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import difflib
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from ant_comparison import ROOT, MODES, environment, now, read_json, write_json
from ant_mpo_followup import Controller, aggregate


ARMS = {
    "single_mstep_regularized": {
        "gpu": 0, "updates": {"mstep_iteration_num": 1, "mean_l2_coef": 0.001, "critic_action_samples": 1},
        "description": "One M-step and mean L2=0.001; one action in the critic target.",
    },
    "critic_actions16": {
        "gpu": 1, "updates": {"critic_action_samples": 16},
        "description": "Baseline MPO with 16 next actions averaged in the scalar critic target.",
    },
}


def prepare(output, baseline):
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    baseline_manifest = read_json(baseline / "manifest.json")
    if baseline_manifest["steps_per_run"] != 8192000 or baseline_manifest["seeds"] != [1, 2, 3]:
        raise ValueError("Expected the original three-seed Ant comparison")
    output.mkdir(parents=True)
    shutil.copytree(baseline / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    patches = []
    for relative in ("safe_rl/algorithms/mpo.py", "safe_rl/algorithms/sac.py"):
        original = (output / "source" / relative).read_text()
        updated = (ROOT / relative).read_text()
        patches.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                           fromfile=f"baseline/{relative}", tofile=f"experiment/{relative}"))
        (output / "source" / relative).write_text(updated)
    (output / "critic_action_samples.patch").write_text("".join(patches))
    (output / "configs").mkdir()
    shutil.copytree(baseline / "configs", output / "baseline_configs")
    hashes = {str(path.relative_to(output / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (output / "source").rglob("*.py")}
    manifest = {"created_utc": now(), "baseline": str(baseline), "python": baseline_manifest["python"],
                "seeds": [1, 2, 3], "steps_per_run": 8192000, "development_eval_seed": 10001,
                "arms": ARMS, "source_sha256": hashes, "baseline_manifest": baseline_manifest}
    write_json(output / "manifest.json", manifest)
    for name, arm in ARMS.items():
        config = yaml.safe_load((output / "baseline_configs/mpo.yaml").read_text())
        config["algorithm"].update(arm["updates"])
        config["runner"]["save_interval"] = 16000
        (output / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        folder = output / name
        folder.mkdir()
        for shared in ("source", "configs"):
            (folder / shared).symlink_to(Path("..") / shared, target_is_directory=True)
        write_json(folder / "plan.json", {"baseline": str(baseline), "python": manifest["python"],
                                         "gpu": arm["gpu"], "development_eval_seed": 10001})
        write_json(folder / "status.json", {"phase": "queued", "updated_utc": now(), "trials": {}})
    for name in ("ant_mpo_parallel.py", "ant_mpo_followup.py", "ant_comparison.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    report(output)


def report(output):
    with (output / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_json(output / "manifest.json")
        lines = ["# Parallel MPO experiments", "", f"Updated: {now()}", "",
                 "Three training seeds per configuration; 8,192,000 transitions per seed.",
                 "Evaluation: 50 first episodes, seed 10001, deterministic and stochastic actions.",
                 "These are development evaluations; fresh held-out validation follows configuration selection.", "",
                 "| Experiment | State | Completed seeds | Deterministic return | Stochastic return |",
                 "|---|---|---:|---:|---:|"]
        summary = {"updated_utc": now(), "arms": {}}
        for name, arm in manifest["arms"].items():
            state = read_json(output / name / "status.json")
            trials = [trial for trial in state["trials"].values() if trial["status"] == "complete"]
            metrics = aggregate([trial["evaluations"] for trial in trials]) if trials else {}
            formatted = []
            for mode in MODES:
                value = metrics.get(mode)
                if value:
                    sd = value["seed_sd"]
                    suffix = f" ± {sd:.3f}" if sd is not None else " (1 seed)"
                    formatted.append(f"{value['mean_return']:.3f}" + suffix)
                else:
                    formatted.append("pending")
            lines.append(f"| {name} | {state['phase']} | {len(trials)}/3 | {formatted[0]} | {formatted[1]} |")
            summary["arms"][name] = {"state": state, "evaluation": metrics}
        lines += ["", "## Settings and progress", ""]
        for name, arm in manifest["arms"].items():
            state = summary["arms"][name]["state"]
            lines += [f"### {name}", "", f"GPU {arm['gpu']}: {arm['description']}", "", state.get("message", ""), ""]
            for key, trial in state["trials"].items():
                path = output / name / "trials" / key / "train.log"
                text = path.read_text(errors="replace") if path.exists() else ""
                progress = re.findall(r"Learning iteration\s+(\d+)/32000", text)
                urls = re.findall(r"https://wandb.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+", text)
                progress_text = f"; iteration {progress[-1]}/32000" if progress else ""
                link = f"; [wandb]({urls[-1]})" if urls else ""
                lines.append(f"- Seed {trial['seed']}: {trial['status']}{progress_text}{link}")
            if state.get("error"):
                lines.append(f"- Error: {state['error']}")
            lines.append("")
        original = read_json(Path(manifest["baseline"]) / "comparison.json")
        lines += ["## Original three-seed reference", ""]
        for name in ("sac", "mpo"):
            values = original["algorithms"][name]
            lines.append(f"- {name.upper()}: deterministic {values['deterministic']['mean_return']:.3f}; "
                         f"stochastic {values['stochastic']['mean_return']:.3f}.")
        write_json(output / "results.json", summary)
        temporary = output / "report.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(output / "report.md")


class ExperimentController(Controller):
    def __init__(self, output):
        super().__init__(output)
        self.env["WANDB_RUN_GROUP"] = self.output.parent.name

    def save(self, **updates):
        self.state.update(updates, updated_utc=now())
        write_json(self.output / "status.json", self.state)
        report(self.output.parent)


def launch(output):
    manifest = read_json(output / "manifest.json")
    with ExitStack() as stack:
        locks = {}
        for name in manifest["arms"]:
            lock = stack.enter_context((output / name / "worker.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks[name] = lock
        for name, arm in manifest["arms"].items():
            if read_json(output / name / "status.json")["phase"] == "complete":
                continue
            with (output / name / "worker.log").open("ab", buffering=0) as log:
                command = [manifest["python"], "-u", str(output / "ant_mpo_parallel.py"), "worker",
                           "--output", str(output), "--arm", name, "--lock-fd", str(locks[name].fileno())]
                child = subprocess.Popen(command, cwd=ROOT, env=environment(output, arm["gpu"]),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(locks[name].fileno(),))
            (output / name / "worker.pid").write_text(f"{child.pid}\n")
            print(f"Started {name}, worker PID {child.pid}, GPU {arm['gpu']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "logs/ant_mpo_parallel_20260916")
    parser.add_argument("--baseline", type=Path, default=ROOT / "logs/ant_ppo_sac_mpo_20260915")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(output, args.baseline.resolve())
    elif args.action == "launch":
        launch(output)
    elif args.action == "report":
        report(output)
    else:
        if args.arm is None:
            parser.error("worker requires --arm")
        folder = output / args.arm
        lock_file = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (folder / "worker.lock").open("a")
        with lock_file as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            controller = ExperimentController(folder)
            try:
                controller.save(phase="training")
                for seed in read_json(output / "manifest.json")["seeds"]:
                    if controller.train(args.arm, seed) is None:
                        raise RuntimeError(f"Seed {seed} failed; see its logs")
                controller.save(phase="complete", message="All three seeds trained and evaluated.")
            except Exception as error:
                controller.save(phase="needs_review", error=str(error))
                raise


if __name__ == "__main__":
    main()
