"""Aggregate the original Ant MPO comparison and all follow-up variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "logs/ant_ppo_sac_mpo_20260915"
FOLLOWUP = ROOT / "logs/ant_mpo_followup_20260915"
PARALLEL = ROOT / "logs/ant_mpo_parallel_20260916"
OUTPUT = ROOT / "logs/ant_mpo_all_variants_20260916"
MODES = ("deterministic", "stochastic")
TERMINAL = {"complete", "target_met", "target_not_met", "needs_review"}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{path.stat().st_ino if path.exists() else 0}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def aggregate(results):
    output = {}
    for mode in MODES:
        values = [result[mode] for result in results]
        rewards = [value["mean_reward"] for value in values]
        output[mode] = {
            "n": len(values),
            "mean_return": statistics.mean(rewards),
            "return_sd": statistics.stdev(rewards) if len(rewards) > 1 else None,
            "seed_returns": rewards,
            "survival": statistics.mean(value["survival_rate"] for value in values),
            "length": statistics.mean(value["mean_length"] for value in values),
        }
    return output


def evaluation_pair(folder: Path):
    return {mode: read_json(folder / f"eval_{mode}.json") for mode in MODES}


def config_changes(path: Path, baseline: dict):
    current = yaml.safe_load(path.read_text())["algorithm"]
    defaults_added_after_baseline = {"critic_action_samples": 1}
    changes = {}
    for key, value in current.items():
        if key in baseline:
            baseline_value = baseline[key]
        elif key in defaults_added_after_baseline:
            baseline_value = defaults_added_after_baseline[key]
        else:
            baseline_value = None
        if baseline_value != value:
            changes[key] = value
    return changes


def collect():
    original = read_json(BASELINE / "comparison.json")
    baseline_config = yaml.safe_load((BASELINE / "configs/mpo.yaml").read_text())["algorithm"]
    variants = {
        "Original MPO": {
            "kind": "three_seed", "source": "original comparison",
            "config_changes": {},
            "metrics": {mode: {
                "n": original["algorithms"]["mpo"][mode]["seeds"],
                "mean_return": original["algorithms"]["mpo"][mode]["mean_return"],
                "return_sd": original["algorithms"]["mpo"][mode]["seed_return_sd"],
                "survival": original["algorithms"]["mpo"][mode]["survival_rate"],
                "length": original["algorithms"]["mpo"][mode]["mean_length"],
            } for mode in MODES},
        }
    }
    references = {
        algorithm: {mode: original["algorithms"][algorithm][mode] for mode in MODES}
        for algorithm in ("ppo", "sac")
    }

    parallel_names = {
        "single_mstep_regularized": "One M-step + mean L2",
        "critic_actions16": "16-action critic target",
    }
    parallel_states = {}
    for directory, label in parallel_names.items():
        root = PARALLEL / directory
        state = read_json(root / "status.json")
        parallel_states[directory] = state["phase"]
        trials = [trial for trial in state.get("trials", {}).values() if trial["status"] == "complete"]
        if trials:
            variants[label] = {
                "kind": "three_seed" if len(trials) == 3 else "partial",
                "source": "parallel experiment", "phase": state["phase"],
                "config_changes": config_changes(PARALLEL / "configs" / f"{directory}.yaml", baseline_config),
                "metrics": aggregate([trial["evaluations"] for trial in trials]),
            }

    followup_state = read_json(FOLLOWUP / "status.json")
    display = {
        "wider_mean": "Wider mean KL",
        "hard_target": "Hard target actor",
        "mean_regularized": "Mean L2",
        "single_mstep": "One M-step",
        "tighter_variance": "Tighter variance KL",
        "more_actions": "128 E-step actions",
        "adaptive_regularized": "Adaptive regularized",
        "adaptive_mean": "Adaptive mean KL",
    }
    grouped = {}
    for trial in followup_state.get("trials", {}).values():
        if trial["status"] == "complete":
            grouped.setdefault(trial["name"], []).append(trial)
    for name, trials in grouped.items():
        label = display.get(name, name)
        variants[label] = {
            "kind": ("three_seed" if len(trials) == 3
                     else "partial" if len(trials) > 1 else "screening"),
            "source": "adaptive follow-up", "phase": followup_state["phase"],
            "config_changes": config_changes(FOLLOWUP / "configs" / f"{name}.yaml", baseline_config),
            "metrics": aggregate([trial["evaluations"] for trial in trials]),
        }
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": all(phase in TERMINAL for phase in parallel_states.values())
                    and followup_state["phase"] in TERMINAL,
        "states": {"parallel": parallel_states, "followup": followup_state["phase"]},
        "references": references, "variants": variants,
        "heldout": followup_state.get("heldout"),
        "heldout_comparison": followup_state.get("heldout_comparison"),
        "development_comparison": followup_state.get("development_comparison"),
        "selected": followup_state.get("selected"),
        "selected_label": display.get(followup_state.get("selected"), followup_state.get("selected")),
    }


def label(metric):
    sd = metric["return_sd"]
    return f"{metric['mean_return']:.3f} ± {sd:.3f}" if sd is not None else f"{metric['mean_return']:.3f} (1 seed)"


def render(data):
    original = data["variants"]["Original MPO"]["metrics"]
    stopped_for_review = any(
        phase == "needs_review"
        for group in data["states"].values()
        for phase in (group.values() if isinstance(group, dict) else (group,))
    )
    if stopped_for_review:
        status = "Status: **stopped for review**"
    elif data["complete"]:
        status = "Status: **complete**"
    else:
        status = "Status: **training in progress**"
    lines = ["# Ant MPO: original and variant comparison", "", f"Updated: {data['updated_utc']}", "",
             status, "",
             "All training runs use 8,192,000 transitions. Development evaluation uses 50 first episodes per "
             "training seed with evaluation seed 10001. Screening rows have one training seed and cannot support "
             "a robustness conclusion. Original MPO uses 64 actions per state in the E-step, 5 M-step gradient "
             "iterations per sampled batch, and 1 sampled next action for its critic target.", "",
             "## Three-seed and partial results", "",
             "| Version | Seeds | Deterministic return | Stochastic return | Survival det./stoch. | "
             "Length det./stoch. | Δ det. vs original |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    robust = [(name, value) for name, value in data["variants"].items()
              if value["kind"] in ("three_seed", "partial")]
    robust.sort(key=lambda item: item[1]["metrics"]["deterministic"]["mean_return"], reverse=True)
    for name, value in robust:
        det, stoch = value["metrics"]["deterministic"], value["metrics"]["stochastic"]
        delta = det["mean_return"] - original["deterministic"]["mean_return"]
        lines.append(
            f"| {name} | {det['n']} | {label(det)} | {label(stoch)} | "
            f"{det['survival']:.1%} / {stoch['survival']:.1%} | "
            f"{det['length']:.1f} / {stoch['length']:.1f} | {delta:+.3f} |"
        )
    lines += ["", "Reference algorithms (same budget and evaluation protocol):", "",
              "| Algorithm | Seeds | Deterministic return | Stochastic return | Survival det./stoch. | "
              "Length det./stoch. |",
              "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("ppo", "sac"):
        reference = data["references"][algorithm]
        lines.append(
            f"| {algorithm.upper()} | {reference['deterministic']['seeds']} | "
            f"{reference['deterministic']['mean_return']:.3f} ± "
            f"{reference['deterministic']['seed_return_sd']:.3f} | "
            f"{reference['stochastic']['mean_return']:.3f} ± "
            f"{reference['stochastic']['seed_return_sd']:.3f} | "
            f"{reference['deterministic']['survival_rate']:.1%} / "
            f"{reference['stochastic']['survival_rate']:.1%} | "
            f"{reference['deterministic']['mean_length']:.1f} / "
            f"{reference['stochastic']['mean_length']:.1f} |"
        )
    lines += ["", "## One-seed screening", "",
              "| Version | Deterministic return | Stochastic return | Survival det./stoch. | "
              "Length det./stoch. | Δ det. vs original |",
              "|---|---:|---:|---:|---:|---:|"]
    screening = [(name, value) for name, value in data["variants"].items()
                 if value["kind"] == "screening"]
    screening.sort(key=lambda item: item[1]["metrics"]["deterministic"]["mean_return"], reverse=True)
    for name, value in screening:
        det, stoch = value["metrics"]["deterministic"], value["metrics"]["stochastic"]
        delta = det["mean_return"] - original["deterministic"]["mean_return"]
        lines.append(
            f"| {name} | {label(det)} | {label(stoch)} | "
            f"{det['survival']:.1%} / {stoch['survival']:.1%} | "
            f"{det['length']:.1f} / {stoch['length']:.1f} | {delta:+.3f} |"
        )
    lines += ["", "## Configuration differences", ""]
    for name, value in data["variants"].items():
        if name == "Original MPO":
            continue
        changes = ", ".join(f"`{key}={setting}`" for key, setting in value["config_changes"].items())
        lines.append(f"- **{name}** ({value['source']}): {changes or 'no algorithm changes'}.")
    signatures = {}
    for name, value in data["variants"].items():
        signatures.setdefault(json.dumps(value["config_changes"], sort_keys=True), []).append(name)
    repeated = [names for names in signatures.values() if len(names) > 1]
    if repeated:
        lines += ["", "Repeated effective configurations are reported separately and are not pooled as "
                  "independent seeds:"]
        lines.extend(f"- {', '.join(names)}." for names in repeated)
    if data["heldout"]:
        lines += ["", "## Held-out evaluation", "",
                  f"Selected configuration: **{data['selected_label']}** (`{data['selected']}`).", "",
                  "| Algorithm | Mode | Return | Survival | Episode length |",
                  "|---|---|---:|---:|---:|"]
        for algorithm in ("sac", "mpo"):
            for mode in MODES:
                metric = data["heldout"][algorithm][mode]
                lines.append(f"| {algorithm.upper()} | {mode} | {metric['mean_return']:.3f} ± "
                             f"{metric['seed_sd']:.3f} | {metric['survival']:.1%} | "
                             f"{metric['mean_length']:.1f} |")
        heldout_passes = data["heldout_comparison"]["passes"]
        development_passes = data["development_comparison"]["passes"]
        lines += ["", "Target: MPO return within 5% of SAC in both modes and survival no more than 5 "
                  "percentage points below SAC, on both development and held-out evaluations.",
                  f"Development target: **{'passed' if development_passes else 'failed'}**. "
                  f"Held-out target: **{'passed' if heldout_passes else 'failed'}**."]
    lines += ["", "## Interpretation", ""]
    completed = [item for item in robust if item[1]["kind"] == "three_seed"]
    if data["complete"] and completed:
        best = max(completed, key=lambda item: item[1]["metrics"]["deterministic"]["mean_return"])
        det = best[1]["metrics"]["deterministic"]
        sac_det = data["references"]["sac"]["deterministic"]["mean_return"]
        lines.append(f"- Highest deterministic return among completed three-seed MPO versions: **{best[0]}**, "
                     "return "
                     f"{det['mean_return']:.3f}; gap to SAC {det['mean_return'] - sac_det:+.3f}.")
        if data["heldout"]:
            selected = data["variants"].get(data["selected_label"])
            if selected:
                for mode in MODES:
                    metric = selected["metrics"][mode]
                    baseline = original[mode]["mean_return"]
                    sac = data["references"]["sac"][mode]["mean_return"]
                    lines.append(
                        f"- Selected **{data['selected_label']}** development {mode} return: "
                        f"{metric['mean_return']:.3f}; change from original MPO "
                        f"{metric['mean_return'] - baseline:+.3f}; gap to SAC "
                        f"{metric['mean_return'] - sac:+.3f}."
                    )
            for mode in MODES:
                mpo = data["heldout"]["mpo"][mode]
                sac = data["heldout"]["sac"][mode]
                lines.append(
                    f"- Held-out {mode}: MPO {mpo['mean_return']:.3f} versus SAC "
                    f"{sac['mean_return']:.3f}, a gap of "
                    f"{mpo['mean_return'] - sac['mean_return']:+.3f}; survival gap "
                    f"{mpo['survival'] - sac['survival']:+.1%}."
                )
            target_met = data["heldout_comparison"]["passes"] and data["development_comparison"]["passes"]
            lines.append(
                f"- Overall predefined SAC-similarity target: **{'met' if target_met else 'not met'}**."
            )
    else:
        lines.append("- Final ranking is pending remaining training, finalist confirmation, and held-out evaluation.")
    lines += ["- Three-seed means use training seeds as the statistical units; episode count is not treated as "
              "independent training replication.",
              "- Results compare final checkpoints under a common transition budget; no best-checkpoint selection.", ""]
    return "\n".join(lines)


def update():
    data = collect()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "comparison.json", data)
    temporary = OUTPUT / "comparison.tmp"
    temporary.write_text(render(data))
    temporary.replace(OUTPUT / "comparison.md")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Refresh every minute until all workers finish.")
    parser.add_argument("--launch", action="store_true", help="Start a detached report watcher.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        pid_path = OUTPUT / "watcher.pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Comparison watcher is already running")
        with (OUTPUT / "watcher.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--wait"],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        pid_path.write_text(f"{process.pid}\n")
        print(f"Started comparison watcher PID {process.pid}")
        return
    with (OUTPUT / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.wait else 0))
        while True:
            data = update()
            print(data["updated_utc"], data["states"], flush=True)
            if data["complete"] or not args.wait:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
