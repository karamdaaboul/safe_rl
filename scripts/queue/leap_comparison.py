"""Frozen-source LEAP pilot: prepare, smoke, launch, persistent report and final evaluation."""

from __future__ import annotations

import argparse
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("ppo", "sac", "mpo", "mpo_one_step")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(c):
    if (c / "manifest.json").exists():
        raise RuntimeError("Campaign already prepared")
    hashes = {}
    for p in (ROOT / "safe_rl").rglob("*.py"):
        target = c / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copy2(__file__, c / "leap_comparison.py")
    for name in NAMES:
        cfg = yaml.safe_load((ROOT / "config/humanoid_comparison" / f"{name}.yaml").read_text())
        cfg["runner"].update(
            experiment_name="leap_zaxis_pilot", run_name=f"{name}_seed1", wandb_project="leap", log_env_steps=True
        )
        (c / "configs").mkdir(exist_ok=True)
        (c / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        write(c / name / "status.json", {"algorithm": name, "state": "queued"})
    write(
        c / "manifest.json",
        dict(
            created_utc=now(),
            env_id="MjxLeapCubeRotateZAxis",
            num_envs=250,
            transitions=8000000,
            seed=1,
            evaluation_seed=10001,
            evaluation_episodes=50,
            python=sys.executable,
            algorithms=NAMES,
            gpu_assignment=dict(ppo=0, sac=1, mpo=0, mpo_one_step=1),
            source_sha256=hashes,
            playground_commit="1699a6d3d9996f2e8391791d55c2d91eccbe33ba",
            task_defaults=True,
            domain_randomization=False,
            actor_obs=32,
            critic_obs=105,
            forward_final_observation=True,
            checkpoint_selection="final",
        ),
    )
    report(c)


def runtime(c, gpu):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        MUJOCO_GL="egl",
        PYTHONPATH=os.pathsep.join(
            [str(c / "source"), str(c / "dependencies/fasttd3_playground"), str(c / "dependencies/python")]
        ),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        JAX_COMPILATION_CACHE_DIR=str(c / "jax_cache"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        WANDB_RUN_GROUP=c.name,
    )
    return env


def report(c):
    with (c / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _report(c)


def _report(c):
    rows = []
    for name in NAMES:
        row = json.loads((c / name / "status.json").read_text())
        log = c / name / "train.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        steps = re.findall(r"Total (?:timesteps|steps):\s+(\d+)", text)
        row["logged_transitions"] = int(steps[-1]) if steps else 0
        urls = re.findall(r"https://wandb.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+", text)
        if urls:
            row["wandb_url"] = urls[-1]
        for mode in ("deterministic", "stochastic"):
            path = c / name / f"eval_{mode}.json"
            if path.exists():
                row[mode] = json.loads(path.read_text())
        rows.append(row)
    write(
        c / "comparison.json", dict(updated_utc=now(), complete=all(r["state"] == "complete" for r in rows), runs=rows)
    )
    lines = [
        "# LEAP cube Z-axis pilot",
        "",
        "250 environments; 8M transitions each; training seed 1.",
        "Final evaluation: 50 first episodes, seed 10001, deterministic and stochastic actions.",
        "Single-seed pilot; results are not a reliable algorithm ranking.",
        "",
        "| Algorithm | State | Logged transitions | Deterministic return | Stochastic return |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        vals = [f"{r[m]['mean_reward']:.3f}" if m in r else "pending" for m in ("deterministic", "stochastic")]
        lines.append(f"| {r['algorithm']} | {r['state']} | {r['logged_transitions']:,} | {' | '.join(vals)} |")
    (c / "comparison.md").write_text("\n".join(lines) + "\n")


def build(c, name, env, smoke=False, log_dir=None):
    from safe_rl.runners import OnPolicyRunner, OffPolicyRunner

    raw = yaml.safe_load((c / "configs" / f"{name}.yaml").read_text())
    if smoke:
        raw["runner"].update(logger="tensorboard", save_interval=1)
        if name == "ppo":
            raw["runner"]["num_steps_per_env"] = 4
            raw["algorithm"].update(num_mini_batches=2, num_learning_epochs=1)
        else:
            raw["runner"].update(max_size=128, start_random_steps=0, update_after=0, log_interval=1)
            raw["algorithm"].update(batch_size=4, num_updates_per_step=1)
    cfg = copy.deepcopy(raw)
    if name == "ppo":
        cfg.update(cfg.pop("runner"))
    cls = OnPolicyRunner if name == "ppo" else OffPolicyRunner
    return cls(env, cfg, log_dir=str(log_dir) if log_dir else None, device="cuda:0")


def make(n, seed, horizon=500):
    from safe_rl.envs.mujoco_playground_vec_env import MujocoPlaygroundVecEnv

    return MujocoPlaygroundVecEnv(
        "MjxLeapCubeRotateZAxis", n, seed=seed, device="cuda:0", episode_length=horizon, forward_final_observation=True
    )


def evaluate(runner, env, mode):
    import torch

    runner.eval_mode()
    policy = runner.get_inference_policy()
    if mode == "stochastic":
        policy = runner.alg.policy.act
    obs, _ = env.reset()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros_like(returns)
    drops = torch.zeros_like(active)
    rotation = torch.zeros_like(returns)
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            obs, reward, done, info = env.step(policy(obs))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            returns += reward * active
            lengths += active
            angvel = torch.utils.dlpack.from_dlpack(env._inner.env_state.metrics["reward/angvel"])
            rotation += angvel * active * 0.05
            drops |= active & done & ~info["time_outs"].bool()
            active &= ~done
            if not active.any():
                break
    assert not active.any(), "Evaluation did not finish all first episodes"
    return dict(
        episodes_collected=env.num_envs,
        mean_reward=returns.mean().item(),
        mean_length=lengths.mean().item(),
        drop_rate=drops.float().mean().item(),
        mean_rotation_radians=rotation.mean().item(),
        returns=returns.tolist(),
        lengths=lengths.tolist(),
    )


def execute(c, action, name):
    import torch
    from safe_rl.utils.seeding import seed_everything

    torch.set_num_threads(4)
    seed_everything(1)
    if action == "smoke":
        env = make(4, 1, horizon=2)
        obs, info = env.reset()
        assert obs.shape == (4, 32) and info["observations"]["critic"].shape == (4, 105)
        initial = obs.clone()
        env.step(torch.zeros(4, 16, device="cuda:0"))
        obs, _, done, info = env.step(torch.zeros(4, 16, device="cuda:0"))
        assert done.all() and info["time_outs"].all()
        assert info["final_critic_observation"].shape == (4, 105)
        assert not torch.equal(info["final_observation"], initial)
        for name in NAMES:
            folder = c / "smoke" / name
            runner = build(c, name, env, smoke=True, log_dir=folder)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
            checkpoint = folder / "model_1.pt"
            assert checkpoint.exists()
            runner.load(str(checkpoint), load_optimizer=False)
            for mode in ("deterministic", "stochastic"):
                write(folder / f"eval_{mode}.json", evaluate(runner, env, mode))
            writer = getattr(runner, "writer", None)
            if writer is not None:
                writer.close()
            print(f"PASS {name}: timeout, train, reload, both evaluations", flush=True)
        write(c / "smoke/status.json", dict(state="passed", updated_utc=now()))
        return
    env = make(250 if action == "train" else 50, 1 if action == "train" else 10001)
    folder = c / name
    runner = build(c, name, env, log_dir=folder / "checkpoints" if action == "train" else None)
    if action == "train":
        runner.learn(num_learning_iterations=1000 if name == "ppo" else 32000, init_at_random_ep_len=False)
    else:
        runner.load(str(folder / "checkpoints" / f"model_{999 if name == 'ppo' else 31999}.pt"), load_optimizer=False)
        for mode in ("deterministic", "stochastic"):
            seed_everything(10001)
            write(folder / f"eval_{mode}.json", evaluate(runner, env, mode))


def worker(c, name):
    manifest = json.loads((c / "manifest.json").read_text())
    status = dict(algorithm=name, started_utc=now())
    for action in ("train", "eval"):
        status.update(state="training" if action == "train" else "evaluating", updated_utc=now())
        with (c / name / f"{action}.log").open("wb") as out:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(c / "leap_comparison.py"),
                    action,
                    "--campaign",
                    str(c),
                    "--algorithm",
                    name,
                ],
                env=runtime(c, manifest["gpu_assignment"][name]),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            status["pid"] = process.pid
            write(c / name / "status.json", status)
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    report(c)
        if code:
            status.update(state="failed", exit_code=code, failed_phase=action, updated_utc=now())
            write(c / name / "status.json", status)
            report(c)
            return
    status.update(state="complete", updated_utc=now())
    write(c / name / "status.json", status)
    report(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "smoke", "launch", "worker", "train", "eval", "report"))
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--algorithm", choices=NAMES)
    a = p.parse_args()
    c = a.campaign.resolve()
    if a.action == "prepare":
        prepare(c)
    elif a.action == "launch":
        assert json.loads((c / "smoke/status.json").read_text())["state"] == "passed"
        for name in NAMES:
            assert json.loads((c / name / "status.json").read_text())["state"] == "queued"
        for name in NAMES:
            with (c / f"{name}_worker.log").open("ab") as out:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(c / "leap_comparison.py"),
                        "worker",
                        "--campaign",
                        str(c),
                        "--algorithm",
                        name,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            (c / f"{name}_worker.pid").write_text(str(proc.pid))
            print(f"Started {name} worker PID {proc.pid}")
    elif a.action == "worker":
        worker(c, a.algorithm)
    elif a.action == "report":
        report(c)
    else:
        os.environ.update(runtime(c, os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
        for path in reversed(os.environ["PYTHONPATH"].split(os.pathsep)):
            sys.path.insert(0, path)
        execute(c, a.action, a.algorithm)


if __name__ == "__main__":
    main()
