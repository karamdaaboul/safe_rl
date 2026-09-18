"""Run LEAP PPO and corrected-SAC seeds 2 and 3 with matched 8M budgets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("ppo", "sac_corrected")
SEEDS = (2, 3)
NUM_ENVS = 250
TRANSITIONS = 8_000_000
EVAL_SEED = 10_001
EVAL_EPISODES = 50
GPU_POOL = (0, 1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base(campaign: Path):
    module = load_module(campaign / "base_controller.py", "leap_seed_base")
    module.VERSIONS = ALGORITHMS
    module.GPU_POOL = GPU_POOL
    module.report = report
    return module


def source_hashes(source: Path) -> dict[str, str]:
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
    }


def prepare(campaign: Path) -> None:
    if (campaign / "manifest.json").exists():
        raise RuntimeError("Campaign already exists; refusing to overwrite it")
    pilot = ROOT / "logs/leap_zaxis_pilot_20260916"
    corrected = ROOT / "logs/leap_normalization_fix_20260917"
    campaign.mkdir(parents=True)
    shutil.copytree(pilot / "source", campaign / "sources/ppo")
    shutil.copytree(corrected / "source", campaign / "sources/sac_corrected")
    shutil.copy2(pilot / "configs/ppo.yaml", campaign / "ppo.yaml")
    shutil.copy2(corrected / "configs/sac.yaml", campaign / "sac_corrected.yaml")
    shutil.copy2(ROOT / "scripts/queue/leap_comparison.py", campaign / "pilot.py")
    shutil.copy2(ROOT / "scripts/queue/leap_mpo_normalization_seeds.py", campaign / "base_controller.py")
    shutil.copy2(__file__, campaign / "controller.py")
    (campaign / "dependencies").symlink_to(pilot / "dependencies", target_is_directory=True)
    (campaign / "jax_cache").symlink_to(pilot / "jax_cache", target_is_directory=True)

    runs = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            run = {
                "name": f"{algorithm}_seed{seed}",
                "version": algorithm,
                "algorithm": algorithm,
                "seed": seed,
                "transitions": TRANSITIONS,
            }
            runs.append(run)
            write(campaign / "runs" / run["name"] / "status.json", {"state": "queued", **run})
    manifest = {
        "created_utc": now(),
        "purpose": "complete three-seed LEAP PPO and corrected-SAC comparisons",
        "env_id": "MjxLeapCubeRotateZAxis",
        "num_envs": NUM_ENVS,
        "transitions_per_run": TRANSITIONS,
        "evaluation_seed": EVAL_SEED,
        "evaluation_episodes": EVAL_EPISODES,
        "execution_order": list(ALGORITHMS),
        "gpu_pool": list(GPU_POOL),
        "runs": runs,
        "python": sys.executable,
        "source_provenance": {"ppo": str(pilot.resolve()), "sac_corrected": str(corrected.resolve())},
        "source_sha256": {name: source_hashes(campaign / "sources" / name) for name in ALGORITHMS},
        "config_sha256": {
            name: hashlib.sha256((campaign / f"{name}.yaml").read_bytes()).hexdigest() for name in ALGORITHMS
        },
        "wandb_project": "leap",
        "wandb_group": campaign.name,
    }
    write(campaign / "manifest.json", manifest)
    write(campaign / "status.json", {"state": "prepared", "updated_utc": now()})
    report(campaign)


def build(campaign: Path, run: dict, env, *, smoke: bool, log: bool):
    from safe_rl.runners import OffPolicyRunner, OnPolicyRunner

    cfg = yaml.safe_load((campaign / f"{run['algorithm']}.yaml").read_text())
    cfg["runner"].update(
        experiment_name="leap_ppo_sac_additional_seeds",
        run_name=run["name"],
        wandb_project="leap",
        log_env_steps=True,
    )
    if run["algorithm"] == "ppo":
        cfg["runner"]["max_iterations"] = 1_000
        if smoke:
            cfg["runner"].update(logger="tensorboard", num_steps_per_env=4, save_interval=1)
            cfg["algorithm"].update(num_mini_batches=2, num_learning_epochs=1)
        cfg.update(cfg.pop("runner"))
        runner_class = OnPolicyRunner
    else:
        cfg["runner"].update(max_iterations=32_000, save_interval=31_999)
        if smoke:
            cfg["runner"].update(
                logger="tensorboard",
                max_size=128,
                start_random_steps=0,
                update_after=0,
                save_interval=1,
                log_interval=1,
            )
            cfg["algorithm"].update(batch_size=4, num_updates_per_step=1)
        runner_class = OffPolicyRunner
    log_dir = campaign / ("smoke" if smoke else "runs") / run["name"] / "checkpoints"
    return runner_class(env, cfg, log_dir=str(log_dir) if log else None, device="cuda:0")


def execute(campaign: Path, action: str, name: str, gpu: int) -> None:
    run = next(item for item in read(campaign / "manifest.json")["runs"] if item["name"] == name)
    common = base(campaign)
    common.configure(campaign, run["version"], gpu)
    import torch
    from safe_rl.utils.seeding import seed_everything

    torch.set_num_threads(4)
    seed_everything(run["seed"] if action != "eval" else EVAL_SEED)
    if action == "smoke_one":
        env = common.make(campaign, 4, run["seed"], horizon=2)
        runner = build(campaign, run, env, smoke=True, log=True)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        assert all(torch.isfinite(parameter).all() for parameter in runner.alg.policy.parameters())
        checkpoint = campaign / "smoke" / name / "checkpoints/model_1.pt"
        runner.load(str(checkpoint), load_optimizer=False)
        for mode in ("deterministic", "stochastic"):
            common.evaluate(runner, env, mode)
        print(f"PASS {name}: train, reload, deterministic and stochastic evaluation", flush=True)
        return

    env = common.make(
        campaign,
        NUM_ENVS if action == "train" else EVAL_EPISODES,
        run["seed"] if action == "train" else EVAL_SEED,
    )
    runner = build(campaign, run, env, smoke=False, log=action == "train")
    folder = campaign / "runs" / name
    if action == "train":
        iterations = 1_000 if run["algorithm"] == "ppo" else 32_000
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        assert runner.tot_timesteps == TRANSITIONS
    else:
        index = 999 if run["algorithm"] == "ppo" else 31_999
        runner.load(str(folder / "checkpoints" / f"model_{index}.pt"), load_optimizer=False)
        for mode in ("deterministic", "stochastic"):
            seed_everything(EVAL_SEED)
            write(folder / f"eval_{mode}.json", common.evaluate(runner, env, mode))


def smoke(campaign: Path, gpu: int) -> None:
    runs = read(campaign / "manifest.json")["runs"]
    representatives = [next(run for run in runs if run["algorithm"] == name) for name in ALGORITHMS]
    with (campaign / "smoke.log").open("wb") as output:
        for run in representatives:
            process = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(campaign / "controller.py"),
                    "smoke_one",
                    "--campaign",
                    str(campaign),
                    "--name",
                    run["name"],
                    "--gpu",
                    str(gpu),
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            if process.returncode:
                write(campaign / "smoke/status.json", {"state": "failed", "run": run["name"], "updated_utc": now()})
                raise RuntimeError(f"Smoke test failed for {run['name']}; see smoke.log")
    write(campaign / "smoke/status.json", {"state": "passed", "updated_utc": now()})


def coordinator(campaign: Path) -> None:
    common = base(campaign)
    try:
        for algorithm in ALGORITHMS:
            write(campaign / "status.json", {"state": f"running_{algorithm}", "updated_utc": now()})
            common.queue(campaign, algorithm)
        write(campaign / "status.json", {"state": "complete", "updated_utc": now()})
    except Exception as error:
        write(campaign / "status.json", {"state": "failed", "error": str(error), "updated_utc": now()})
    report(campaign)


def report(campaign: Path) -> None:
    lock_path = campaign / "report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for run in read(campaign / "manifest.json")["runs"]:
            folder = campaign / "runs" / run["name"]
            row = read(folder / "status.json")
            log = folder / "train.log"
            contents = log.read_text(errors="replace") if log.exists() else ""
            steps = re.findall(r"Total (?:timesteps|steps):\s+(\d+)", contents)
            row["logged_transitions"] = int(steps[-1]) if steps else 0
            urls = re.findall(r"https://wandb.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+", contents)
            if urls:
                row["wandb_url"] = urls[-1]
            for mode in ("deterministic", "stochastic"):
                path = folder / f"eval_{mode}.json"
                if path.exists():
                    row[mode] = read(path)
            rows.append(row)
        write(
            campaign / "comparison.json",
            {"updated_utc": now(), "campaign": read(campaign / "status.json"), "runs": rows},
        )
        lines = [
            "# LEAP PPO and corrected-SAC additional seeds",
            "",
            "250 environments, 8M transitions per run; seeds 2 and 3.",
            "Evaluation uses 50 first episodes per mode with seed 10001, matching seed 1.",
            "",
            "| Run | State | Transitions | Deterministic return | Stochastic return |",
            "|---|---|---:|---:|---:|",
        ]
        for row in rows:
            values = [
                f"{row[mode]['mean_reward']:.3f}" if mode in row else "pending"
                for mode in ("deterministic", "stochastic")
            ]
            lines.append(f"| {row['name']} | {row['state']} | {row['logged_transitions']:,} | {' | '.join(values)} |")
        (campaign / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "smoke", "smoke_one", "launch", "coordinator", "worker", "train", "eval", "report"),
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(campaign)
    elif args.action == "smoke":
        smoke(campaign, args.gpu)
    elif args.action == "launch":
        assert read(campaign / "status.json")["state"] == "prepared"
        assert read(campaign / "smoke/status.json")["state"] == "passed"
        write(campaign / "status.json", {"state": "launching", "updated_utc": now()})
        with (campaign / "coordinator.log").open("ab") as output:
            process = subprocess.Popen(
                [sys.executable, "-u", str(campaign / "controller.py"), "coordinator", "--campaign", str(campaign)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (campaign / "coordinator.pid").write_text(str(process.pid))
        print(f"Started LEAP PPO/SAC seed coordinator {process.pid}")
    elif args.action == "coordinator":
        coordinator(campaign)
    elif args.action == "worker":
        base(campaign).worker(campaign, args.name, args.gpu)
    elif args.action in ("smoke_one", "train", "eval"):
        execute(campaign, args.action, args.name, args.gpu)
    else:
        report(campaign)


if __name__ == "__main__":
    main()
