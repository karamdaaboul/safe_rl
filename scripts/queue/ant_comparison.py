"""Run and report a fixed-budget PPO/SAC/MPO comparison using the existing mjlab CLIs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("ppo", "sac", "mpo")
MODES = ("deterministic", "stochastic")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def prepare(campaign, seeds, env_id="Ant-Flat", config_dir=None, algorithms=ALGORITHMS):
    config_dir = config_dir or ROOT / "config/ant_comparison"
    if (campaign / "manifest.json").exists():
        raise RuntimeError(f"Campaign already exists: {campaign}")
    campaign.mkdir(parents=True, exist_ok=True)
    source = campaign / "source"
    paths = list((ROOT / "safe_rl").rglob("*.py"))
    paths += [ROOT / f"scripts/{folder}/unitree_mjlab.py" for folder in ("train", "eval")]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    for name in algorithms:
        destination = campaign / "configs" / f"{name}.yaml"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(config_dir / f"{name}.yaml", destination)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (campaign / "working_tree.diff").write_bytes(diff)
    manifest = {
        "created_utc": now(), "env_id": env_id, "algorithms": list(algorithms), "num_envs": 256,
        "steps_per_run": 8192000, "seeds": seeds, "evaluation_seed": 10001,
        "evaluation_episodes": 50, "evaluation_modes": list(MODES),
        "python": sys.executable, "workspace": str(ROOT), "source": str(source),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashes, "gpu_assignment": {"ppo": 0, "sac": 1, "mpo": 0, "mpo_one_step": 1},
        "notes": [
            "Equal environment-step budget; optimizer updates differ by algorithm.",
            "All backbones use [256, 256] ReLU and observation normalization.",
            "PPO uses a Gaussian policy; SAC and MPO use tanh-squashed Gaussians.",
            "PPO uses final-state V bootstrap; SAC/MPO use final actor/critic replay observations.",
            "Final checkpoint selection only; no best-checkpoint selection using evaluation scores.",
            "One initial episode per evaluation environment, with the same seed across policies.",
            "Wall-clock times use different/shared GPUs and are not a controlled speed comparison.",
        ],
    }
    write_json(campaign / "manifest.json", manifest)
    for name in algorithms:
        for seed in seeds:
            write_json(campaign / f"{name}_seed{seed}" / "status.json", {
                "algorithm": name, "seed": seed, "status": "queued", "updated_utc": now(),
            })
    report(campaign)
    print(f"Prepared {len(seeds) * len(algorithms)} runs in {campaign}")


def environment(campaign, gpu):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", MUJOCO_GL="egl",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH=str(campaign / "source"),
        UNITREE_RL_MJLAB_PATH="/home/human/workspaces/unitree_rl_mjlab",
        MPLCONFIGDIR=str(campaign / "matplotlib"), WANDB_RUN_GROUP=campaign.name,
    )
    return env


def launch(campaign, algorithm=None):
    manifest = read_json(campaign / "manifest.json")
    names = (algorithm,) if algorithm else manifest.get("algorithms", ALGORITHMS)
    for name in names:
        pid_path = campaign / f"{name}_worker.pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Worker {name} is still alive; refusing a duplicate launch.")
    for name in names:
        gpu = manifest["gpu_assignment"][name]
        with (campaign / f"{name}_worker.log").open("ab", buffering=0) as output:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                       "--campaign", str(campaign), "--algorithm", name, "--gpu", str(gpu)]
            process = subprocess.Popen(command, cwd=ROOT, env=environment(campaign, gpu),
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (campaign / f"{name}_worker.pid").write_text(f"{process.pid}\n")
        print(f"Started {name.upper()} worker PID {process.pid} on GPU {gpu}")


def run_command(command, log, env, status, status_path):
    with log.open("wb", buffering=0) as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT)
        status.update(pid=process.pid, command=command, log=str(log), updated_utc=now())
        write_json(status_path, status)
        campaign = status_path.parent.parent
        report(campaign)
        while True:
            try:
                return process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                report(campaign)


def worker(campaign, name, gpu):
    manifest = read_json(campaign / "manifest.json")
    env = environment(campaign, gpu)
    for seed in manifest["seeds"]:
        run = campaign / f"{name}_seed{seed}"
        status_path = run / "status.json"
        status = read_json(status_path)
        if status["status"] == "complete":
            continue
        try:
            status.update(status="training", started_utc=now(), gpu=gpu)
            config = campaign / "configs" / f"{name}.yaml"
            command = [manifest["python"], "-u", str(campaign / "source/scripts/train/unitree_mjlab.py"),
                       "--env_id", manifest["env_id"], "--config", str(config), "--num_envs", "256",
                       # environment() exposes one physical GPU; select index 0 in that list.
                       "--seed", str(seed), "--gpu_ids", "0", "--device", "cuda:0",
                       "--experiment_name", f"{name}_seed{seed}",
                       "--run_name", f"{campaign.name}_{name}_seed{seed}", "--log_dir", str(campaign / "runs")]
            started = time.monotonic()
            code = run_command(command, run / "train.log", env, status, status_path)
            status.update(training_seconds=time.monotonic() - started, train_exit_code=code)
            if code:
                raise RuntimeError(f"Training exited with {code}; see {run / 'train.log'}")
            final_index = 999 if name == "ppo" else 31999
            checkpoints = list((campaign / "runs" / f"{name}_seed{seed}").glob(f"*/model_{final_index}.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"Expected one final checkpoint; found {checkpoints}")
            status["checkpoint"] = str(checkpoints[0])
            for mode in MODES:
                status["status"] = f"evaluating_{mode}"
                command = [manifest["python"], "-u", str(campaign / "source/scripts/eval/unitree_mjlab.py"),
                           "--env_id", manifest["env_id"], "--checkpoint", str(checkpoints[0]),
                           "--config", str(config), "--num_envs", str(manifest["evaluation_episodes"]),
                           "--episodes", str(manifest["evaluation_episodes"]), "--one_episode_per_env",
                           "--headless", "--device", "cuda:0", "--seed", str(manifest["evaluation_seed"]),
                           "--action_mode", mode, "--json_out", str(run / f"eval_{mode}.json")]
                code = run_command(command, run / f"eval_{mode}.log", env, status, status_path)
                if code:
                    raise RuntimeError(f"{mode} evaluation exited with {code}")
                result = read_json(run / f"eval_{mode}.json")
                if result["episodes_collected"] != manifest["evaluation_episodes"]:
                    raise RuntimeError("Evaluation episode count does not match the protocol")
            status.update(status="complete", finished_utc=now())
        except Exception as exc:
            status.update(status="failed", error=str(exc), finished_utc=now())
            print(f"{name} seed {seed}: {exc}", flush=True)
        status["updated_utc"] = now()
        write_json(status_path, status)
        report(campaign)


def training_curve(log, name):
    if not log.exists():
        return [], []
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    blocks = re.split(r"Learning iteration\s+(\d+)/\d+", text)
    steps, rewards = [], []
    for index in range(1, len(blocks), 2):
        reward = re.search(r"^\s*Mean reward:\s*([-+\d.e]+)", blocks[index + 1], re.MULTILINE)
        if reward:
            steps.append((int(blocks[index]) + 1) * 256 * (32 if name == "ppo" else 1))
            rewards.append(float(reward[1]))
    return steps, rewards


def report(campaign):
    with (campaign / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _report(campaign)


def _report(campaign):
    manifest = read_json(campaign / "manifest.json")
    algorithms = manifest.get("algorithms", ALGORITHMS)
    rows, curves = [], {}
    for name in algorithms:
        for seed in manifest["seeds"]:
            run = campaign / f"{name}_seed{seed}"
            status = read_json(run / "status.json")
            curve = training_curve(run / "train.log", name)
            curves[(name, seed)] = curve
            row = {**status, "last_logged_steps": curve[0][-1] if curve[0] else 0}
            if (run / "train.log").exists():
                urls = re.findall(r"https://wandb.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+",
                                  (run / "train.log").read_text(errors="replace"))
                if urls:
                    row["wandb_url"] = urls[-1]
            for mode in MODES:
                path = run / f"eval_{mode}.json"
                if path.exists():
                    row[mode] = read_json(path)
            rows.append(row)
    complete = all(row["status"] == "complete" for row in rows)
    summary = {"updated_utc": now(), "complete": complete, "runs": rows, "algorithms": {}}
    lines = [f"# {manifest['env_id']}: " + " / ".join(algorithms), "", f"Updated: {summary['updated_utc']}", "",
             "Status: " + ("complete" if complete else "in progress or awaiting failed-run recovery"), "",
             f"Budget: {manifest['steps_per_run']:,} transitions per run; seeds {manifest['seeds']}.",
             "Evaluation: 50 environments, one first episode each, seed 10001; final checkpoints only.", "",
             "## Progress", "", "| Algorithm | Seed | State | Logged steps | wandb |",
             "|---|---:|---|---:|---|"]
    for row in rows:
        url = row.get("wandb_url")
        link = f"[run]({url})" if url else "pending"
        lines.append(f"| {row['algorithm'].upper()} | {row['seed']} | {row['status']} | "
                     f"{row['last_logged_steps']:,} | {link} |")
    for mode in MODES:
        lines += ["", f"## {mode.title()} final evaluation", "",
                  "Return is mean ± sample SD across training-seed means; incomplete sets are marked.", "",
                  "| Algorithm | Evaluated seeds | Return | Episode length | Survival |",
                  "|---|---:|---:|---:|---:|"]
        for name in algorithms:
            results = [row[mode] for row in rows if row["algorithm"] == name and mode in row]
            if not results:
                lines.append(f"| {name.upper()} | 0/{len(manifest['seeds'])} | pending | — | — |")
                continue
            returns = [result["mean_reward"] for result in results]
            avg, sd = statistics.mean(returns), statistics.stdev(returns) if len(returns) > 1 else None
            length = statistics.mean(result["mean_length"] for result in results)
            survival = statistics.mean(result["survival_rate"] for result in results)
            summary["algorithms"].setdefault(name, {})[mode] = {
                "seeds": len(results), "mean_return": avg, "seed_return_sd": sd,
                "mean_length": length, "survival_rate": survival,
            }
            label = f"{avg:.3f} ± {sd:.3f}" if sd is not None else f"{avg:.3f} (one seed)"
            lines.append(f"| {name.upper()} | {len(results)}/{len(manifest['seeds'])} | {label} | "
                         f"{length:.1f} | {survival:.1%} |")
    lines += ["", "## Interpretation", "", *[f"- {note}" for note in manifest["notes"]],
              "- Training curves show stochastic behavior returns; use final evaluation for the comparison.",
              "- Three seeds provide an initial variability check, not a definitive algorithm ranking.",
              "", "![Training curves](learning_curves.png)", "", "![Final returns](final_comparison.png)", ""]
    write_json(campaign / "comparison.json", summary)
    (campaign / "comparison.md").write_text("\n".join(lines))
    plot(campaign, curves, rows)


def plot(campaign, curves, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = read_json(campaign / "manifest.json")
    algorithms = manifest.get("algorithms", ALGORITHMS)
    colors = dict(mpo_one_step="#9255b5", ppo="#3066be", sac="#db6d28", mpo="#22845a")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for (name, seed), (steps, rewards) in curves.items():
        if not rewards:
            continue
        smoothed = [statistics.mean(rewards[max(0, i - 4):i + 1]) for i in range(len(rewards))]
        ax.plot(np.asarray(steps) / 1e6, smoothed, color=colors[name], alpha=0.75, label=f"{name.upper()} seed {seed}")
    ax.set(xlabel="Environment transitions (millions)", ylabel="Training episode return",
           title=f"{manifest['env_id']} · logged reward with a 5-point moving average")
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(fontsize=8, ncol=3)
    fig.savefig(campaign / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        for position, name in enumerate(algorithms):
            values = [row[mode]["mean_reward"] for row in rows if row["algorithm"] == name and mode in row]
            if values:
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0
                ax.bar(position, mean, yerr=sd, capsize=4, color=colors[name], alpha=0.55)
                ax.scatter([position] * len(values), values, color=colors[name], edgecolors="black", zorder=3)
            else:
                ax.text(position, 0, "pending", ha="center", va="bottom", fontsize=9)
        ax.set(xticks=range(len(algorithms)), xticklabels=[name.upper() for name in algorithms], title=mode.title(),
               ylabel="Final evaluation return (seed mean)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{manifest['env_id']} · 50 episodes per policy · error bars: SD across seeds")
    fig.savefig(campaign / "final_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "report"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--algorithm", choices=(*ALGORITHMS, "mpo_one_step"))
    parser.add_argument("--algorithms", nargs="+", choices=(*ALGORITHMS, "mpo_one_step"), default=ALGORITHMS)
    parser.add_argument("--env_id", default="Ant-Flat")
    parser.add_argument("--config_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(campaign, args.seeds, args.env_id, args.config_dir, args.algorithms)
    elif args.action == "launch":
        launch(campaign, args.algorithm)
    elif args.action == "worker":
        worker(campaign, args.algorithm, args.gpu)
    else:
        report(campaign)


if __name__ == "__main__":
    main()
