"""Bounded LEAP screening followed by automatic three-seed, longer-budget validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "sac_entropy": ("sac", {"target_entropy_scale": 0.25}),
    "sac_fixed_alpha": ("sac", {"auto_entropy_tuning": False, "alpha": 0.001}),
    "mpo5_slow_actor": ("mpo", {"actor_lr": 0.0001}),
    "mpo1_slow_actor": ("mpo_one_step", {"actor_lr": 0.0001}),
    "mpo1_mean_l2": ("mpo_one_step", {"mean_l2_coef": 0.01}),
    "mpo1_relaxed_kl": ("mpo_one_step", {"kl_mean_constraint": 0.005, "kl_var_constraint": 0.00005}),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def helper(c):
    spec = importlib.util.spec_from_file_location("leap_pilot", c / "pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_run(c, manifest, name, config, seed, steps, phase):
    run = dict(name=name, config=config, seed=seed, steps=steps, phase=phase)
    manifest["runs"].append(run)
    write(c / "runs" / name / "status.json", dict(state="queued", **run))


def prepare(c, baseline, normalization_control=False):
    c.mkdir(parents=True, exist_ok=True)
    if (c / "manifest.json").exists():
        raise RuntimeError("Campaign exists; refusing to overwrite")
    baseline = baseline.resolve()
    # Freeze the exact pilot implementation, rather than unrelated current workspace changes.
    shutil.copytree(baseline / "source", c / "source")
    corrected_files = (
        (
            "safe_rl/modules/normalizer.py",
            "safe_rl/modules/sac_actor_critic.py",
            "safe_rl/algorithms/sac.py",
            "safe_rl/algorithms/mpo.py",
            "safe_rl/algorithms/mpo_dime.py",
            "safe_rl/runners/off_policy_runner.py",
        )
        if normalization_control
        else ()
    )
    for relative in corrected_files:
        shutil.copy2(ROOT / relative, c / "source" / relative)
    shutil.copy2(ROOT / "scripts/queue/leap_comparison.py", c / "pilot.py")
    shutil.copy2(__file__, c / "leap_tuning.py")
    (c / "dependencies").symlink_to(baseline / "dependencies", target_is_directory=True)
    (c / "jax_cache").symlink_to(baseline / "jax_cache", target_is_directory=True)
    (c / "configs").mkdir()
    for name in ("sac", "mpo", "mpo_one_step"):
        shutil.copy2(baseline / "configs" / f"{name}.yaml", c / "configs" / f"{name}.yaml")
    for name, (base, changes) in VARIANTS.items():
        cfg = yaml.safe_load((c / "configs" / f"{base}.yaml").read_text())
        cfg["algorithm"].update(changes)
        (c / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = dict(
        created_utc=now(),
        baseline=str(baseline),
        num_envs=250,
        runs=[],
        variants=VARIANTS,
        selection="highest deterministic return at 8M, seed 10001",
        validation_seeds=[1, 2, 3],
        validation_eval_seed=20001,
        validation_transitions=20000000,
        maximum_new_transitions=208000000,
        phases="six 8M screens; selected SAC/MPO three seeds at 20M; unchanged controls seed 1 at 20M",
        python=sys.executable,
        source_sha256=read(baseline / "manifest.json")["source_sha256"],
    )
    manifest["normalization_control"] = normalization_control
    if normalization_control:
        manifest.update(
            maximum_new_transitions=24000000,
            variants={},
            corrected_files=corrected_files,
            selection=None,
            validation_seeds=[],
            phases="three matched 8M baseline reruns",
        )
        for relative in corrected_files:
            manifest["source_sha256"][relative] = hashlib.sha256((c / "source" / relative).read_bytes()).hexdigest()
    for name in ("sac", "mpo", "mpo_one_step") if normalization_control else VARIANTS:
        add_run(c, manifest, name + "_seed1", name, 1, 8000000, "screen")
    write(c / "manifest.json", manifest)
    write(c / "status.json", dict(state="prepared", updated_utc=now()))
    diagnostics(c)
    report(c)


def diagnostics(c):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = Path(read(c / "manifest.json")["baseline"])
    tags = [
        "Train/episode_reward",
        "SafeRL/alpha",
        "Train/kl_mean_rel",
        "Train/kl_var_rel",
        "Train/frac_saturated",
        "Train/kl_q_rel",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    summary = {}
    for name in ("sac", "mpo", "mpo_one_step"):
        e = EventAccumulator(str(baseline / name / "checkpoints"), size_guidance={"scalars": 0})
        e.Reload()
        summary[name] = {}
        for tag, ax in zip(tags, axes.flat):
            ax.set_title(tag)
            ax.set_xlabel("Environment transitions (millions)")
            if tag not in e.Tags()["scalars"]:
                continue
            values = e.Scalars(tag)
            # Off-policy TensorBoard step is the iteration index.
            ax.plot([(v.step + 1) * 250 / 1e6 for v in values], [v.value for v in values], label=name)
            summary[name][tag] = dict(
                last=values[-1].value,
                final_10_percent_mean=sum(v.value for v in values[-max(1, len(values) // 10) :])
                / max(1, len(values) // 10),
            )
    axes.flat[1].set_yscale("log")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(c / "pilot_diagnostics.png", dpi=160)
    plt.close(fig)
    write(c / "pilot_diagnostics.json", summary)


def configure(c, gpu):
    env = helper(c).runtime(c, gpu)
    os.environ.update(env)
    for p in reversed(env["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, p)


def build(c, run, env, smoke=False, log=True):
    from safe_rl.runners import OffPolicyRunner

    cfg = yaml.safe_load((c / "configs" / f"{run['config']}.yaml").read_text())
    cfg["runner"].update(
        run_name=run["name"],
        experiment_name="leap_tuning",
        # Save exact 8M checkpoint as well as final 20M checkpoint.
        save_interval=31999,
        max_iterations=run["steps"] // 250,
    )
    if smoke:
        cfg["runner"].update(
            logger="tensorboard", max_size=128, start_random_steps=0, update_after=0, save_interval=1, log_interval=1
        )
        cfg["algorithm"].update(batch_size=4, num_updates_per_step=1)
    folder = c / ("smoke" if smoke else "runs") / run["name"] / "checkpoints"
    return OffPolicyRunner(env, cfg, log_dir=str(folder) if log else None, device="cuda:0")


def evaluate(runner, env, mode):
    # Measure differential action entropy without altering the evaluation RNG stream.
    import torch

    policy = runner.get_inference_policy()
    if mode == "stochastic":
        policy = runner.alg.policy.act
    obs, _ = env.reset()
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = returns.clone()
    rotation = returns.clone()
    entropy = returns.clone()
    drops = ~active
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            with torch.random.fork_rng(devices=[0]):
                _, logp = runner.alg.policy.sample_with_log_prob(obs)
            entropy += -logp.view(-1) * active
            obs, reward, done, info = env.step(policy(obs))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert torch.isfinite(logp).all()
            returns += reward * active
            lengths += active
            rotation += torch.utils.dlpack.from_dlpack(env._inner.env_state.metrics["reward/angvel"]) * active * 0.05
            drops |= active & done & ~info["time_outs"].bool()
            active &= ~done
            if not active.any():
                break
    assert not active.any()
    return dict(
        episodes_collected=env.num_envs,
        mean_reward=returns.mean().item(),
        mean_length=lengths.mean().item(),
        drop_rate=drops.float().mean().item(),
        mean_rotation_radians=rotation.mean().item(),
        mean_action_entropy=(entropy.sum() / lengths.sum()).item(),
        returns=returns.tolist(),
        lengths=lengths.tolist(),
    )


def execute(c, action, name, gpu):
    configure(c, gpu)
    import torch
    from safe_rl.utils.seeding import seed_everything

    torch.set_num_threads(4)
    m = read(c / "manifest.json")
    runs = m["runs"] if action == "smoke" else [next(r for r in m["runs"] if r["name"] == name)]
    if action == "smoke":
        runs = [r for r in runs if r["phase"] == "screen"]
        env = helper(c).make(4, 1, horizon=2)
        for run in runs:
            seed_everything(1)
            runner = build(c, run, env, smoke=True)
            runner.learn(num_learning_iterations=2)
            assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
            checkpoint = c / "smoke" / run["name"] / "checkpoints/model_1.pt"
            runner.load(str(checkpoint), load_optimizer=False)
            for mode in ("deterministic", "stochastic"):
                evaluate(runner, env, mode)
            print("PASS", run["config"], flush=True)
        write(c / "smoke/status.json", dict(state="passed", updated_utc=now()))
        return
    run = runs[0]
    seed_everything(run["seed"])
    eval_seed = 10001 if run["phase"] == "screen" else 20001
    env = helper(c).make(250 if action == "train" else 50, run["seed"] if action == "train" else eval_seed)
    runner = build(c, run, env, log=action == "train")
    folder = c / "runs" / name
    if action == "train":
        runner.learn(num_learning_iterations=run["steps"] // 250)
        assert runner.tot_timesteps == run["steps"]
    else:
        budgets = [run["steps"]] if run["phase"] == "screen" else [8000000, run["steps"]]
        for steps in budgets:
            runner.load(str(folder / "checkpoints" / f"model_{steps//250-1}.pt"), load_optimizer=False)
            for mode in ("deterministic", "stochastic"):
                seed_everything(eval_seed)
                result = evaluate(runner, env, mode)
                result.update(transitions=steps, evaluation_seed=eval_seed)
                write(folder / f"eval_{steps}_{mode}.json", result)


def report(c):
    m = read(c / "manifest.json")
    rows = []
    for run in m["runs"]:
        folder = c / "runs" / run["name"]
        row = read(folder / "status.json")
        log = folder / "train.log"
        txt = log.read_text(errors="replace") if log.exists() else ""
        counts = re.findall(r"Total steps:\s+(\d+)", txt)
        row["logged_transitions"] = int(counts[-1]) if counts else 0
        for p in folder.glob("eval_*.json"):
            row[p.stem] = read(p)
        rows.append(row)
    write(c / "comparison.json", dict(updated_utc=now(), campaign=read(c / "status.json"), runs=rows))
    lines = [
        "# LEAP SAC/MPO tuning",
        "",
        "250 environments. Six 8M screens, then selected settings at 20M, seeds 1–3.",
        "Selection: deterministic return, evaluation seed 10001. Validation: separate seed 20001.",
        "Longer-budget controls use unchanged SAC and one-step MPO settings (seed 1).",
        "Validation runs start fresh and save both 8M and 20M checkpoints.",
        "",
        "| Run | Phase | State | Transitions | Final deterministic return | Final stochastic return |",
        "|---|---|---|---:|---:|---:|",
    ]
    if m.get("normalization_control"):
        lines = [
            "# LEAP normalization correction: controlled reruns",
            "",
            "Same pilot configurations, seed 1, 250 environments, 8M transitions each.",
            "Only normalization/checkpoint corrections; no hyperparameter tuning or automatic extension.",
            "Evaluation: 50 first episodes per mode, seed 10001, matching the pilot.",
            "",
            "| Run | Phase | State | Transitions | Final deterministic return | Final stochastic return |",
            "|---|---|---|---:|---:|---:|",
        ]
    for r in rows:
        vals = []
        for mode in ("deterministic", "stochastic"):
            result = r.get(f"eval_{r['steps']}_{mode}")
            vals.append(f"{result['mean_reward']:.3f}" if result else "pending")
        lines.append(
            f"| {r['name']} | {r['phase']} | {r['state']} | {r['logged_transitions']:,} | {' | '.join(vals)} |"
        )
    # Append aggregate validation results only from the selected configurations.
    if m.get("normalization_control"):
        baseline = Path(m["baseline"])
        lines += ["", "## Original pilot references", ""]
        for name in ("sac", "mpo", "mpo_one_step"):
            vals = [
                read(baseline / name / f"eval_{mode}.json")["mean_reward"] for mode in ("deterministic", "stochastic")
            ]
            lines.append(f"- {name}: deterministic {vals[0]:.3f}, stochastic {vals[1]:.3f}.")
        (c / "comparison.md").write_text("\n".join(lines) + "\n")
        return
    for mode in ("deterministic", "stochastic"):
        lines += ["", f"## Selected settings: {mode}, 20M transitions", ""]
        import statistics

        for family in ("sac", "mpo"):
            vals = [
                r[f"eval_20000000_{mode}"]["mean_reward"]
                for r in rows
                if r["phase"] == "validation"
                and r["name"].startswith(family + "_selected_")
                and f"eval_20000000_{mode}" in r
            ]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else None
                lines.append(f"- {family}: {statistics.mean(vals):.3f}, seed SD {sd}, completed {len(vals)}/3 seeds.")
    (c / "comparison.md").write_text("\n".join(lines) + "\n")


def worker(c, name, gpu):
    folder = c / "runs" / name
    status = read(folder / "status.json")
    try:
        for action in ("train", "eval"):
            status.update(state=action, updated_utc=now())
            with (folder / f"{action}.log").open("wb") as out:
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(c / "leap_tuning.py"),
                        action,
                        "--campaign",
                        str(c),
                        "--name",
                        name,
                        "--gpu",
                        str(gpu),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
                status["pid"] = p.pid
                write(folder / "status.json", status)
                if p.wait():
                    raise RuntimeError(f"{action} failed; see {action}.log")
        status.update(state="complete", updated_utc=now())
    except Exception as exc:
        status.update(state="failed", error=str(exc), updated_utc=now())
    write(folder / "status.json", status)


def queue(c, phase):
    pending = [r for r in read(c / "manifest.json")["runs"] if r["phase"] == phase]
    active = {}
    while pending or active:
        for gpu in (0, 1):
            if gpu not in active and pending:
                r = pending.pop(0)
                status = read(c / "runs" / r["name"] / "status.json")
                if status["state"] in ("complete", "cancelled"):
                    continue
                if status["state"] != "queued":
                    raise RuntimeError(f"Refusing to overwrite {r['name']} in state {status['state']}")
                with (c / "runs" / r["name"] / "worker.log").open("ab") as out:
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(c / "leap_tuning.py"),
                            "worker",
                            "--campaign",
                            str(c),
                            "--name",
                            r["name"],
                            "--gpu",
                            str(gpu),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                active[gpu] = (p, r)
        report(c)
        for gpu, (p, r) in list(active.items()):
            try:
                code = p.wait(timeout=25)
            except subprocess.TimeoutExpired:
                continue
            if code or read(c / "runs" / r["name"] / "status.json")["state"] != "complete":
                raise RuntimeError(f"Run failed: {r['name']}")
            del active[gpu]
    report(c)


def select(c):
    m = read(c / "manifest.json")
    baseline = Path(m["baseline"])
    scores = {n: read(baseline / n / "eval_deterministic.json")["mean_reward"] for n in ("sac", "mpo", "mpo_one_step")}
    for run in m["runs"]:
        if run["phase"] == "screen":
            scores[run["config"]] = read(c / "runs" / run["name"] / "eval_8000000_deterministic.json")["mean_reward"]
    chosen = {}
    for family in ("sac", "mpo"):
        candidates = {n: v for n, v in scores.items() if n.startswith(family)}
        chosen[family] = max(candidates, key=candidates.get)
        for seed in (1, 2, 3):
            add_run(c, m, f"{family}_selected_seed{seed}", chosen[family], seed, 20000000, "validation")
        control = "sac" if family == "sac" else "mpo_one_step"
        if chosen[family] != control:
            add_run(c, m, f"{family}_control_seed1", control, 1, 20000000, "validation")
    write(c / "selection.json", dict(chosen=chosen, development_scores=scores, updated_utc=now()))
    write(c / "manifest.json", m)


def coordinator(c):
    try:
        if not (c / "selection.json").exists():
            write(c / "status.json", dict(state="screening", updated_utc=now()))
            queue(c, "screen")
        if not read(c / "manifest.json").get("normalization_control"):
            if not (c / "selection.json").exists():
                select(c)
            write(c / "status.json", dict(state="validating", updated_utc=now()))
            queue(c, "validation")
        write(c / "status.json", dict(state="complete", updated_utc=now()))
    except Exception as exc:
        write(c / "status.json", dict(state="failed", error=str(exc), updated_utc=now()))
    report(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("prepare", "smoke", "launch", "resume", "coordinator", "worker", "train", "eval", "report")
    )
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=ROOT / "logs/leap_zaxis_pilot_20260916")
    p.add_argument("--name")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--normalization_control", action="store_true")
    a = p.parse_args()
    c = a.campaign.resolve()
    if a.action == "prepare":
        prepare(c, a.baseline, a.normalization_control)
    elif a.action in ("launch", "resume"):
        assert read(c / "smoke/status.json")["state"] == "passed"
        expected = "prepared" if a.action == "launch" else "paused"
        assert read(c / "status.json")["state"] == expected, "Campaign must be prepared or explicitly paused"
        write(c / "status.json", dict(state="launching", updated_utc=now()))
        with (c / "coordinator.log").open("ab") as out:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "coordinator", "--campaign", str(c)],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (c / "coordinator.pid").write_text(str(proc.pid))
        print("Started LEAP tuning coordinator", proc.pid)
    elif a.action == "coordinator":
        coordinator(c)
    elif a.action == "worker":
        worker(c, a.name, a.gpu)
    elif a.action == "report":
        report(c)
    else:
        execute(c, a.action, a.name, a.gpu)


if __name__ == "__main__":
    main()
