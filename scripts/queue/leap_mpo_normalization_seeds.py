"""Matched LEAP one-step MPO seeds for corrected and original normalization."""

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
VERSIONS = ("corrected", "original")
SEEDS = (2, 3)
TRANSITIONS = 8_000_000
NUM_ENVS = 250
EVAL_SEED = 10_001
EVAL_EPISODES = 50
GPU_POOL = (1,)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def helper(campaign: Path):
    spec = importlib.util.spec_from_file_location("leap_pilot", campaign / "pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    shutil.copytree(corrected / "source", campaign / "sources/corrected")
    shutil.copytree(pilot / "source", campaign / "sources/original")
    shutil.copy2(corrected / "configs/mpo_one_step.yaml", campaign / "config.yaml")
    shutil.copy2(ROOT / "scripts/queue/leap_comparison.py", campaign / "pilot.py")
    shutil.copy2(__file__, campaign / "controller.py")
    (campaign / "dependencies").symlink_to(pilot / "dependencies", target_is_directory=True)
    (campaign / "jax_cache").symlink_to(pilot / "jax_cache", target_is_directory=True)

    runs = []
    for version in VERSIONS:
        for seed in SEEDS:
            run = {
                "name": f"mpo1_{version}_seed{seed}",
                "version": version,
                "seed": seed,
                "transitions": TRANSITIONS,
            }
            runs.append(run)
            write(campaign / "runs" / run["name"] / "status.json", {"state": "queued", **run})

    manifest = {
        "created_utc": now(),
        "purpose": "two additional corrected seeds followed by two original-normalization seeds",
        "env_id": "MjxLeapCubeRotateZAxis",
        "num_envs": NUM_ENVS,
        "transitions_per_run": TRANSITIONS,
        "evaluation_seed": EVAL_SEED,
        "evaluation_episodes": EVAL_EPISODES,
        "gpu_pool": list(GPU_POOL),
        "execution_order": list(VERSIONS),
        "runs": runs,
        "python": sys.executable,
        "source_provenance": {
            "corrected": str(corrected.resolve()),
            "original": str(pilot.resolve()),
        },
        "source_sha256": {version: source_hashes(campaign / "sources" / version) for version in VERSIONS},
        "config_sha256": hashlib.sha256((campaign / "config.yaml").read_bytes()).hexdigest(),
        "wandb_project": "leap",
        "wandb_group": campaign.name,
    }
    write(campaign / "manifest.json", manifest)
    write(campaign / "status.json", {"state": "prepared", "updated_utc": now()})
    report(campaign)


def configure(campaign: Path, version: str, gpu: int) -> None:
    env = helper(campaign).runtime(campaign, gpu)
    paths = env["PYTHONPATH"].split(os.pathsep)
    paths[0] = str(campaign / "sources" / version)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["WANDB_RUN_GROUP"] = campaign.name
    os.environ.update(env)
    for path in reversed(paths):
        sys.path.insert(0, path)


def make(campaign: Path, count: int, seed: int, horizon: int = 500):
    return helper(campaign).make(count, seed, horizon=horizon)


def build(campaign: Path, run: dict, env, *, smoke: bool, log: bool):
    from safe_rl.runners import OffPolicyRunner

    cfg = yaml.safe_load((campaign / "config.yaml").read_text())
    cfg["runner"].update(
        experiment_name="leap_mpo_normalization_seeds",
        run_name=run["name"],
        wandb_project="leap",
        log_env_steps=True,
        max_iterations=TRANSITIONS // NUM_ENVS,
        save_interval=TRANSITIONS // NUM_ENVS - 1,
    )
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
    log_dir = campaign / ("smoke" if smoke else "runs") / run["name"] / "checkpoints"
    return OffPolicyRunner(env, cfg, log_dir=str(log_dir) if log else None, device="cuda:0")


def evaluate(runner, env, mode: str) -> dict:
    import torch

    runner.eval_mode()
    policy = runner.get_inference_policy()
    if mode == "stochastic":
        policy = runner.alg.policy.act
    obs, _ = env.reset()
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros_like(returns)
    rotation = torch.zeros_like(returns)
    entropy = torch.zeros_like(returns)
    drops = torch.zeros_like(active)
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            with torch.random.fork_rng(devices=[0]):
                _, log_probability = runner.alg.policy.sample_with_log_prob(obs)
            entropy += -log_probability.view(-1) * active
            obs, reward, done, info = env.step(policy(obs))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert torch.isfinite(log_probability).all()
            returns += reward * active
            lengths += active
            rotation += torch.utils.dlpack.from_dlpack(env._inner.env_state.metrics["reward/angvel"]) * active * 0.05
            drops |= active & done & ~info["time_outs"].bool()
            active &= ~done
            if not active.any():
                break
    assert not active.any(), "Evaluation did not finish all first episodes"
    return {
        "episodes_collected": env.num_envs,
        "mean_reward": returns.mean().item(),
        "mean_length": lengths.mean().item(),
        "drop_rate": drops.float().mean().item(),
        "mean_rotation_radians": rotation.mean().item(),
        "mean_action_entropy": (entropy.sum() / lengths.sum()).item(),
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
        "transitions": TRANSITIONS,
        "evaluation_seed": EVAL_SEED,
    }


def execute(campaign: Path, action: str, name: str, gpu: int) -> None:
    import torch

    run = next(item for item in read(campaign / "manifest.json")["runs"] if item["name"] == name)
    configure(campaign, run["version"], gpu)
    from safe_rl.utils.seeding import seed_everything

    torch.set_num_threads(4)
    if action == "smoke_one":
        seed_everything(run["seed"])
        env = make(campaign, 4, run["seed"], horizon=2)
        runner = build(campaign, run, env, smoke=True, log=True)
        runner.learn(num_learning_iterations=2)
        assert all(torch.isfinite(parameter).all() for parameter in runner.alg.policy.parameters())
        checkpoint = campaign / "smoke" / name / "checkpoints/model_1.pt"
        runner.load(str(checkpoint), load_optimizer=False)
        for mode in ("deterministic", "stochastic"):
            evaluate(runner, env, mode)
        print(f"PASS {name}: train, reload, deterministic and stochastic evaluation", flush=True)
        return

    seed_everything(run["seed"] if action == "train" else EVAL_SEED)
    env = make(
        campaign, NUM_ENVS if action == "train" else EVAL_EPISODES, run["seed"] if action == "train" else EVAL_SEED
    )
    runner = build(campaign, run, env, smoke=False, log=action == "train")
    folder = campaign / "runs" / name
    if action == "train":
        runner.learn(num_learning_iterations=TRANSITIONS // NUM_ENVS)
        assert runner.tot_timesteps == TRANSITIONS
    else:
        checkpoint = folder / "checkpoints" / f"model_{TRANSITIONS // NUM_ENVS - 1}.pt"
        runner.load(str(checkpoint), load_optimizer=False)
        for mode in ("deterministic", "stochastic"):
            seed_everything(EVAL_SEED)
            write(folder / f"eval_{mode}.json", evaluate(runner, env, mode))


def smoke(campaign: Path, gpu: int) -> None:
    runs = read(campaign / "manifest.json")["runs"]
    representatives = [next(run for run in runs if run["version"] == version) for version in VERSIONS]
    log_path = campaign / "smoke.log"
    with log_path.open("wb") as output:
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
                raise RuntimeError(f"Smoke test failed for {run['name']}; see {log_path}")
    write(campaign / "smoke/status.json", {"state": "passed", "updated_utc": now()})


def worker(campaign: Path, name: str, gpu: int) -> None:
    folder = campaign / "runs" / name
    status = read(folder / "status.json")
    try:
        for action in ("train", "eval"):
            status.update(state=action, updated_utc=now())
            with (folder / f"{action}.log").open("wb") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(campaign / "controller.py"),
                        action,
                        "--campaign",
                        str(campaign),
                        "--name",
                        name,
                        "--gpu",
                        str(gpu),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                status["pid"] = process.pid
                write(folder / "status.json", status)
                if process.wait():
                    raise RuntimeError(f"{action} failed; see {action}.log")
        status.update(state="complete", updated_utc=now())
    except Exception as error:
        status.update(state="failed", error=str(error), updated_utc=now())
    write(folder / "status.json", status)


def queue(campaign: Path, version: str) -> None:
    pending = [run for run in read(campaign / "manifest.json")["runs"] if run["version"] == version]
    active = {}
    while pending or active:
        for gpu in GPU_POOL:
            if gpu in active or not pending:
                continue
            run = pending.pop(0)
            state = read(campaign / "runs" / run["name"] / "status.json")["state"]
            if state == "complete":
                continue
            if state != "queued":
                raise RuntimeError(f"Refusing to overwrite {run['name']} in state {state}")
            with (campaign / "runs" / run["name"] / "worker.log").open("ab") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(campaign / "controller.py"),
                        "worker",
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
            active[gpu] = (process, run)
        report(campaign)
        for gpu, (process, run) in list(active.items()):
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                continue
            state = read(campaign / "runs" / run["name"] / "status.json")["state"]
            if code or state != "complete":
                raise RuntimeError(f"Run failed: {run['name']}")
            del active[gpu]
    report(campaign)


def coordinator(campaign: Path) -> None:
    try:
        for version in VERSIONS:
            write(campaign / "status.json", {"state": f"running_{version}", "updated_utc": now()})
            queue(campaign, version)
        write(campaign / "status.json", {"state": "complete", "updated_utc": now()})
    except Exception as error:
        write(campaign / "status.json", {"state": "failed", "error": str(error), "updated_utc": now()})
    report(campaign)


def report(campaign: Path) -> None:
    lock_path = campaign / "report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read(campaign / "manifest.json")
        rows = []
        for run in manifest["runs"]:
            folder = campaign / "runs" / run["name"]
            row = read(folder / "status.json")
            log = folder / "train.log"
            contents = log.read_text(errors="replace") if log.exists() else ""
            steps = re.findall(r"Total steps:\s+(\d+)", contents)
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
            "# LEAP one-step MPO normalization comparison: additional seeds",
            "",
            "250 environments, 8M transitions per run. Corrected seeds run before original-normalization seeds.",
            "Evaluation uses 50 first episodes per mode with seed 10001, matching seed 1.",
            "",
            "| Run | Code | State | Transitions | Deterministic return | Stochastic return |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in rows:
            values = [
                f"{row[mode]['mean_reward']:.3f}" if mode in row else "pending"
                for mode in ("deterministic", "stochastic")
            ]
            lines.append(
                f"| {row['name']} | {row['version']} | {row['state']} | "
                f"{row['logged_transitions']:,} | {' | '.join(values)} |"
            )
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
        print(f"Started LEAP MPO seed coordinator {process.pid}")
    elif args.action == "coordinator":
        coordinator(campaign)
    elif args.action == "worker":
        worker(campaign, args.name, args.gpu)
    elif args.action in ("smoke_one", "train", "eval"):
        execute(campaign, args.action, args.name, args.gpu)
    else:
        report(campaign)


if __name__ == "__main__":
    main()
