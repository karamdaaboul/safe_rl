#!/usr/bin/env python3
"""Training sweep driver for the REPPO paper comparison.

Runs a (suite x task x seed) matrix, one subprocess per cell. Two lanes run
concurrently — ManiSkill on the Ada card, DMC on the Blackwell — and each lane executes
its own queue strictly serially.

Idioms are lifted from `scripts/eval/eval_matrix.py` (per-cell subprocess, child env
construction, skip-if-done resumability, never-silently-drop-a-failure) rather than
reinvented. What is new here is training-side: two lanes, per-suite interpreters, a RAM
guard, and PID bookkeeping for a shared box.

Hard constraints encoded here, each of which has already cost a run:

* ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is exported FIRST. Without it CUDA's indices are
  inverted relative to nvidia-smi (nvidia-smi 0 = Blackwell = CUDA 1).
* ManiSkill runs ONLY on the Ada card. SAPIEN PhysX predates Blackwell sm_120 and falls
  back to CPU physics *silently* — the run completes and produces numbers comparable to
  nothing.
* A RAM guard before each launch. This box has 32 GB, no swap, and the first pilot cell
  was SIGKILLed at 75 minutes with another session's jobs resident.
* Child PIDs are recorded in ``logs/.own_pids/``. Never ``pkill -f train_safety_gymnasium``
  on this box — other sessions use the same entry point.

Usage:

    nohup python -u scripts/bench/run_paper_bench.py --lanes maniskill,dmc --seeds 1,2,3 \\
        >> logs/paper_bench_driver.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "config" / "bench" / "paper"
DEFAULT_OUT = REPO / "experiments" / "paper_bench"

# Env steps per cell: 1024 envs x 128 steps x 381 iters. The report quotes this, not "50M".
BUDGET_ENV_STEPS = 49_938_432
FINAL_ITER = 380


class Lane:
    """One suite's execution settings: interpreter, GPU, env, and env-id prefix."""

    def __init__(self, name, python, gpu, env_prefix, extra_env, num_envs=1024):
        self.name = name
        self.python = python
        self.gpu = gpu
        self.env_prefix = env_prefix
        self.extra_env = extra_env
        self.num_envs = num_envs

    @property
    def config_dir(self) -> Path:
        return CONFIG_ROOT / self.name

    def tasks(self) -> list[str]:
        return sorted(p.stem for p in self.config_dir.glob("*.yaml"))


LANES = {
    # nvidia-smi index 1 = RTX 4000 Ada. Mandatory for ManiSkill (see module docstring).
    "maniskill": Lane(
        name="maniskill",
        python="/home/human/venvs/maniskill/bin/python",
        gpu=1,
        env_prefix="ManiSkill",
        extra_env={"MUJOCO_GL": "egl"},
    ),
    # nvidia-smi index 0 = RTX PRO 4500 Blackwell. jax and torch share this card, so
    # XLA preallocation must be off or jax takes ~75% before torch allocates.
    "dmc": Lane(
        name="dmc",
        python="/home/human/workspaces/reppo_original/.venv/bin/python",
        gpu=0,
        env_prefix="Mjx",
        extra_env={
            "MUJOCO_GL": "egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
            "JAX_COMPILATION_CACHE_DIR": str(Path.home() / ".cache" / "jax"),
            # Our package is not installed into the reference venv on purpose — it is a
            # reference of record, and re-resolving the git-pinned playground fork risks
            # silently picking a different commit.
            "PYTHONPATH": str(REPO),
        },
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cell_dir(out_root: Path, lane: str, task: str, seed: int) -> Path:
    return out_root / lane / task / f"s{seed}"


def is_done(cdir: Path) -> bool:
    """A cell counts as done only if the FINAL checkpoint exists.

    Keyed on the checkpoint rather than on DONE.json so that cells completed outside the
    driver (e.g. the hand-launched pilot) are also recognised and not re-run.
    """
    return any(cdir.rglob(f"model_{FINAL_ITER}.pt"))


def wait_for_ram(min_avail_mb: int, tag: str, poll_s: int = 60) -> None:
    """Block until MemAvailable clears the floor. Logs every wait; never launches blind."""
    waited = 0
    while True:
        avail = 0
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) // 1024
                break
        if avail >= min_avail_mb:
            if waited:
                print(f"[{_now()}] {tag}: RAM cleared after {waited}s ({avail} MB available)", flush=True)
            return
        print(f"[{_now()}] {tag}: waiting for RAM ({avail} MB < {min_avail_mb} MB floor)", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def build_cmd(lane: Lane, task: str, seed: int, cdir: Path, max_iterations: int | None) -> list[str]:
    cmd = [
        lane.python,
        "-u",
        "scripts/train/train_safety_gymnasium.py",
        "--env_id", f"{lane.env_prefix}{task}",
        "--num_envs", str(lane.num_envs),
        "--config", str(lane.config_dir / f"{task}.yaml"),
        "--device", "cuda:0",  # after CUDA_VISIBLE_DEVICES pinning, the lane's card is 0
        "--seed", str(seed),
        "--log_dir", str(cdir),
    ]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]
    return cmd


def child_env(lane: Lane) -> dict:
    env = dict(os.environ)
    # Order matters: PCI_BUS_ID must be set before CUDA_VISIBLE_DEVICES is interpreted.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(lane.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"
    env.update(lane.extra_env)
    return env


def run_cell(lane: Lane, task: str, seed: int, out_root: Path, args, pid_file: Path) -> dict:
    cdir = cell_dir(out_root, lane.name, task, seed)
    tag = f"{lane.name}/{task}/s{seed}"

    if is_done(cdir) and not args.force:
        print(f"[{_now()}] SKIP {tag} (final checkpoint present)", flush=True)
        return {"task": task, "seed": seed, "status": "skipped"}

    cdir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(lane, task, seed, cdir, args.max_iterations)

    if args.dry_run:
        print(f"[dry-run] {tag}\n  CUDA_VISIBLE_DEVICES={lane.gpu} {' '.join(cmd)}", flush=True)
        return {"task": task, "seed": seed, "status": "dry_run"}

    log_path = cdir / "train.log"
    started = time.time()

    # One retry, and only for SIGKILL (rc=137). This box is shared: another session's
    # jobs come and go, and the first pilot cell was OOM-killed at 75 minutes when their
    # footprint spiked. A retry from scratch is used rather than --resume_checkpoint
    # because resume does not restore the iteration counter, and a mis-resumed cell would
    # quietly carry the wrong env-step budget into the comparison.
    for attempt in range(1 + args.retries):
        wait_for_ram(args.min_ram_mb, tag)
        print(f"[{_now()}] START {tag}" + (f" (retry {attempt})" if attempt else ""), flush=True)
        with log_path.open("ab") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=child_env(lane), stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            with pid_file.open("a") as fh:
                fh.write(f"{proc.pid}\n")
            rc = proc.wait()
        if rc != -signal.SIGKILL and rc != 137:
            break
        print(f"[{_now()}] {tag} was SIGKILLed (rc={rc}) — almost certainly the OOM killer; "
              f"{'retrying' if attempt < args.retries else 'giving up'}", flush=True)

    wall = time.time() - started
    record = {
        "suite": lane.name,
        "task": task,
        "seed": seed,
        "returncode": rc,
        "wall_clock_s": round(wall, 1),
        "env_steps": BUDGET_ENV_STEPS if args.max_iterations is None else None,
        "config": str(lane.config_dir / f"{task}.yaml"),
        "cmd": " ".join(cmd),
        "run_dir": str(cdir),
        "finished": _now(),
        "final_checkpoint": is_done(cdir),
    }
    record["status"] = "ok" if (rc == 0 and record["final_checkpoint"]) else "failed"

    if record["status"] != "ok":
        # A failed cell is recorded loudly, never dropped: rc=137 in particular is a
        # SIGKILL (OOM), which leaves no Python traceback in the log.
        tail = log_path.read_bytes()[-8192:].decode("utf-8", "replace")
        (cdir / "FAILED.log").write_text(f"rc={rc}\ncmd={' '.join(cmd)}\n\n{tail}")
        print(f"[{_now()}] FAILED {tag} rc={rc} wall={wall/60:.1f}min "
              f"(rc=137 means SIGKILL/OOM) -> {cdir/'FAILED.log'}", flush=True)
    else:
        print(f"[{_now()}] DONE {tag} wall={wall/60:.1f}min", flush=True)

    (cdir / "DONE.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_lane(lane: Lane, seeds: list[int], out_root: Path, args, pid_file: Path, results: list) -> None:
    tasks = [t for t in lane.tasks() if not args.only or t in args.only]
    if not tasks:
        print(f"[{_now()}] {lane.name}: no matching tasks", flush=True)
        return
    # Seed-major: a complete 1-seed table for every task arrives before any task has
    # three seeds, so an early abort still yields a meaningful interim report.
    for seed in seeds:
        for task in tasks:
            results.append(run_cell(lane, task, seed, out_root, args, pid_file))
            write_manifest(out_root, results)


def write_manifest(out_root: Path, results: list) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="maniskill,dmc", help="comma-separated: maniskill,dmc")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", default="", help="comma-separated task names to restrict to")
    ap.add_argument("--max_iterations", type=int, default=None, help="override; for smoke runs")
    ap.add_argument("--min_ram_mb", type=int, default=12000,
                    help="do not launch a cell below this MemAvailable floor. A 1024-env "
                         "ManiSkill cell peaks near 7.5 GB RSS and this box is shared.")
    ap.add_argument("--retries", type=int, default=1,
                    help="retries for SIGKILLed (OOM) cells only")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run cells that already have a final checkpoint")
    args = ap.parse_args()
    args.only = [t for t in args.only.split(",") if t]

    lanes = [LANES[name] for name in args.lanes.split(",") if name]
    seeds = [int(s) for s in args.seeds.split(",")]

    pid_dir = REPO / "logs" / ".own_pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "run_paper_bench.pids"
    with pid_file.open("a") as fh:
        fh.write(f"{os.getpid()}\n")

    print(f"[{_now()}] driver start: lanes={[x.name for x in lanes]} seeds={seeds} out={args.out}", flush=True)
    for lane in lanes:
        print(f"  {lane.name}: gpu={lane.gpu} tasks={lane.tasks()}", flush=True)

    results: list = []
    threads = [
        threading.Thread(target=run_lane, args=(lane, seeds, args.out, args, pid_file, results), daemon=False)
        for lane in lanes
    ]

    def terminate(signum, _frame):
        # Do not orphan multi-hour GPU jobs when the driver is killed.
        print(f"[{_now()}] signal {signum}: terminating children", flush=True)
        for pid in {int(p) for p in pid_file.read_text().split() if p.isdigit()} - {os.getpid()}:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_manifest(args.out, results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = [f"{r['suite']}/{r['task']}/s{r['seed']}" for r in results if r.get("status") == "failed"]
    print(f"[{_now()}] driver done: {ok} ok, {len(failed)} failed", flush=True)
    if failed:
        print("  failed cells: " + ", ".join(failed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
