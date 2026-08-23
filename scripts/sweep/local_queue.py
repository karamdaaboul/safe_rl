#!/usr/bin/env python
"""Priority queue for the multi-env sweep on one workstation.

Keeps exactly N training runs alive, starting the next when one exits. Replaces the SLURM path:
there is no scheduler here, so the two things a scheduler would have done -- refusing to
oversubscribe, and surviving a restart -- are done explicitly.

RAM is the binding resource on this box (31.3 GiB total, ~7.5 GiB PSS per run), so the guardrail
is a *pre-launch* free-RAM check rather than a post-hoc kill: a run that is admitted and then
OOM-killed costs its whole elapsed wall-clock, which is what happened to the b1 arm. The check
also covers runs this queue did not start (the c4 arms, a manual run) -- it reads system-wide free
memory and counts every `train_safety_gymnasium.py` process, not just its own children, so an
externally launched training occupies a slot as far as the queue is concerned.

Restart tolerance: state lives in the run dirs and `sweep_status.json`, not in this process. On
start it classifies every manifest cell as finished / running / dead and continues.

NOTE -- dead runs are requeued FROM SCRATCH, not resumed. See `resume_supported()`.

Usage:
    python scripts/sweep/local_queue.py --manifest config/sweep/manifest.json
    python scripts/sweep/local_queue.py --status      # one-shot status dump, starts nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATUS = REPO / "logs" / "sweep" / "sweep_status.json"
RUNLOG = REPO / "logs" / "sweep"
PY = os.environ.get("SWEEP_PYTHON", "/home/human/venvs/agx_plain/bin/python")
TRAINER = "scripts/train/train_safety_gymnasium.py"
PROC_PAT = "train_safety_gymnasium.py"

# Measured on this box, 2026-08-22, one FH-DCMPO run at num_envs=8 on the Blackwell:
#   PSS 7.5 GiB | 1.9 cores | 3.07 GiB VRAM | 1.67-1.75 it/s (flat to N=3)
RUN_RSS_GIB = 7.5
DEFAULT_MAX_CONCURRENT = 2  # RAM-bound: N=3 leaves only 18% free, under the 20% floor
DEFAULT_MIN_FREE_GIB = 9.0  # must fit one more run (7.5) and still clear the 20% floor


def resume_supported() -> tuple[bool, str]:
    """Whether a dead run can be continued from its last checkpoint. It cannot.

    `--resume_checkpoint` exists and restores the model, optimizers, normalizers and the
    iteration counter -- but three things make it a warm-start for fine-tuning, not a resume:

    1. The replay buffer is never serialized (`OffPolicyRunner.save` writes model/optimizer/
       normalizer state only). On resume `global_step` is restored from the checkpoint, so the
       `update_after` gate is already open, and the first update samples an n-step window from an
       empty buffer -- `ReplayStorage._valid_start_t` raises "Not enough contiguous transitions
       for n-step sampling". The run dies at once rather than continuing.
    2. Lagrangian/PID state is not stored; lambda restarts from `lambda_init` (documented at
       train_safety_gymnasium.py:569-571). For these arms that discards the whole controller
       trajectory.
    3. `max_iterations` is additive on resume (`tot_iter = start_iter + num_learning_iterations`),
       so a resumed run overshoots its budget unless the caller compensates.

    So a partially-finished run is not resumable into a result comparable with its peers.
    """
    return False, "no replay-buffer serialization; lambda/PID reset; additive max_iterations"


def free_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def total_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def live_trainings() -> list[tuple[int, str]]:
    """Every training on the box -- ours and anyone else's (c4 arms count against N).

    Only PARENT processes. A vecenv run forks ~8-10 workers that inherit the parent's cmdline
    verbatim, so a naive match reports one run as nine and the queue concludes it is full and
    never launches again. Dropping any match whose parent is itself a match collapses each tree
    to its root.
    """
    matches: dict[int, tuple[int, str]] = {}
    try:
        res = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    for line in res.stdout.splitlines():
        line = line.strip()
        if PROC_PAT in line and " --env_id " in line and "ps -eo" not in line:
            pid_s, ppid_s, args = line.split(None, 2)
            matches[int(pid_s)] = (int(ppid_s), args)
    return [(pid, args) for pid, (ppid, args) in matches.items() if ppid not in matches]


def rss_gib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def run_dir_for(env_id: str, cell: str) -> Path | None:
    """Newest FHDCMPO run dir whose marker file names this cell."""
    root = REPO / "logs" / "safety_gymnasium" / env_id / "FHDCMPO"
    if not root.is_dir():
        return None
    cands = [d for d in root.iterdir() if (d / ".sweep_cell").is_file()
             and (d / ".sweep_cell").read_text().strip() == cell]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def last_iter(d: Path) -> int:
    its = [int(m.group(1)) for p in d.glob("model_*.pt")
           if (m := re.match(r"model_(\d+)\.pt$", p.name))]
    return max(its) if its else -1


def classify(cell: dict, iters: int, live: list[tuple[int, str]]) -> tuple[str, dict]:
    """finished | running | dead | pending, from run dirs + the live process list."""
    d = run_dir_for(cell["env_id"], cell["name"])
    info = {"run_dir": str(d.relative_to(REPO)) if d else None, "last_iter": -1}
    if d is None:
        return "pending", info
    info["last_iter"] = last_iter(d)
    for pid, args in live:
        if f"--config {cell['config']}" in args or cell["config"] in args:
            info["pid"] = pid
            info["rss_gib"] = round(rss_gib(pid), 2)
            return "running", info
    # A final checkpoint is written at max_iterations-1 by the runner's save cadence.
    if info["last_iter"] >= iters - 1:
        return "finished", info
    return "dead", info


def launch(cell: dict, iters: int, gpu: str, num_envs: int, cost_limits: str, extra: list[str]) -> subprocess.Popen:
    RUNLOG.mkdir(parents=True, exist_ok=True)
    log = RUNLOG / f"{cell['name']}.log"
    cmd = [PY, TRAINER, "--env_id", cell["env_id"], "--num_envs", str(num_envs),
           "--config", cell["config"], "--cost_limits", cost_limits,
           "--max_iterations", str(iters), "--seed", str(cell["seed"]),
           "--deterministic", "--device", "cuda:0", *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    fh = open(log, "ab")
    fh.write(f"\n=== {time.strftime('%F %T')} launching {cell['name']}: "
             f"{' '.join(shlex.quote(c) for c in cmd)}\n".encode())
    fh.flush()
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    # Stamp the run dir so a restart can map dirs back to cells. The trainer names the dir by
    # timestamp, so poll briefly for the one it just claimed.
    root = REPO / "logs" / "safety_gymnasium" / cell["env_id"] / "FHDCMPO"
    for _ in range(120):
        time.sleep(1)
        if root.is_dir():
            fresh = [d for d in root.iterdir() if d.is_dir() and not (d / ".sweep_cell").exists()
                     and d.stat().st_mtime > time.time() - 300]
            if fresh:
                newest = max(fresh, key=lambda d: d.stat().st_mtime)
                (newest / ".sweep_cell").write_text(cell["name"] + "\n")
                break
        if p.poll() is not None:
            break
    return p


def write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(STATUS)


def snapshot(cells: list[dict], iters: int, extra: dict) -> dict:
    live = live_trainings()
    rows, counts = [], {}
    for c in cells:
        state, info = classify(c, iters, live)
        counts[state] = counts.get(state, 0) + 1
        rows.append({**{k: c[k] for k in ("name", "arm", "env_id", "seed", "config")},
                     "state": state, **info})
    ours = sum(1 for r in rows if r["state"] == "running")
    return {
        "updated": time.strftime("%F %T"), "counts": counts,
        "free_gib": round(free_gib(), 2), "total_gib": round(total_gib(), 2),
        "live_trainings_on_box": len(live), "live_from_this_manifest": ours,
        "foreign_trainings": len(live) - ours, "runs": rows, **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/sweep/manifest.json")
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    ap.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    ap.add_argument("--iters", type=int, default=None, help="override the manifest's max_iterations")
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES; 1 = Blackwell (CUDA order)")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--cost-limits", default="25.0")
    ap.add_argument("--heartbeat-sec", type=int, default=300)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--status", action="store_true", help="dump status and exit")
    ap.add_argument("--dry-run", action="store_true", help="plan only; launch nothing")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    man = json.loads((REPO / args.manifest).read_text())
    cells, iters = man["runs"], args.iters or man.get("iters", 60000)

    if args.status:
        print(json.dumps(snapshot(cells, iters, {}), indent=2))
        return

    ok, why = resume_supported()
    print(f"[queue] resume supported: {ok} ({why})")
    print(f"[queue] {len(cells)} cells, iters={iters}, N={args.max_concurrent}, "
          f"min_free={args.min_free_gib} GiB, gpu={args.gpu}")

    # Carried into every status write so a heartbeat is self-describing after a restart.
    cfg = {"max_concurrent": args.max_concurrent, "min_free_gib": args.min_free_gib,
           "iters": iters, "gpu": args.gpu, "manifest": args.manifest}
    running: dict[str, subprocess.Popen] = {}
    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))

    last_hb = 0.0
    while not stop["now"]:
        live = live_trainings()
        states = {c["name"]: classify(c, iters, live)[0] for c in cells}
        for name, p in list(running.items()):
            if p.poll() is not None:
                print(f"[queue] {name} exited rc={p.returncode} -> {states.get(name)}")
                running.pop(name)

        n_ours = sum(1 for s in states.values() if s == "running")
        # Foreign trainings (c4 arms, manual runs) occupy slots too -- never oversubscribe.
        n_foreign = max(0, len(live) - n_ours)
        n_live = n_ours + n_foreign

        todo = [c for c in cells if states[c["name"]] in ("pending", "dead")]
        if not todo and n_ours == 0:
            print("[queue] all cells finished or running; nothing left to start.")
            write_status(snapshot(cells, iters, {"queue": "done", **cfg}))
            break

        if n_live < args.max_concurrent and todo:
            free = free_gib()
            if free < args.min_free_gib:
                print(f"[queue] HOLD: free RAM {free:.1f} GiB < {args.min_free_gib} GiB "
                      f"threshold; not launching (b1 OOM guardrail).")
            else:
                nxt = todo[0]
                if states[nxt["name"]] == "dead":
                    print(f"[queue] {nxt['name']} is DEAD -- restarting FROM SCRATCH "
                          f"(no resume support; prior partial run dir is left in place).")
                if args.dry_run:
                    print(f"[queue] DRY-RUN would launch {nxt['name']}")
                    todo.pop(0)
                    cells = [c for c in cells if c["name"] != nxt["name"]]
                    continue
                print(f"[queue] launching {nxt['name']} ({n_live}/{args.max_concurrent} slots used, "
                      f"free {free:.1f} GiB)")
                running[nxt["name"]] = launch(nxt, iters, args.gpu, args.num_envs,
                                              args.cost_limits, args.extra)

        if time.time() - last_hb >= args.heartbeat_sec or last_hb == 0.0:
            write_status(snapshot(cells, iters, {"queue": "running", **cfg}))
            last_hb = time.time()
        time.sleep(args.poll_sec)

    write_status(snapshot(cells, iters, {"queue": "stopped" if stop["now"] else "done", **cfg}))
    print("[queue] exiting; children keep running (start_new_session). Use stop_sweep.sh to kill.")


if __name__ == "__main__":
    main()
