#!/usr/bin/env python
"""Pick the c-series winner from Eval telemetry and print its config path (for the 60k launch).

Decision rule, registered before the overnight c3 seeds finished (2026-08-20):

* For every run dir given: average the LAST 3 distinct Eval/violation_rate and Eval/reward
  points from its tfevents (deterministic panels; identical instrumentation in both arms).
* c3's score = the MEDIAN across its seeds (robust to one bad seed).
* c3 wins over c2 iff  (a) median violation rate < c2's by more than 0.05 (a real tail
  improvement, not panel noise), and (b) median eval reward >= 0.5 * c2's (no idle-policy
  collapse -- the c1 failure signature).
* Otherwise c2 wins.

Prints exactly two lines: `winner: <name>` and `config: <path>`, plus a reasoning line to
stderr-like stdout for the log. Exit code 0 always (the wrapper launches whatever is printed).
"""

from __future__ import annotations

import argparse
import glob
import statistics
from pathlib import Path


def last3(run_dir: str, tag: str) -> float | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(str(Path(run_dir) / "events.out.tfevents.*"))
    if not files:
        return None
    ea = EventAccumulator(files[0])
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    # Panels repeat between eval points; keep distinct consecutive values.
    vals: list[float] = []
    for e in ea.Scalars(tag):
        if not vals or vals[-1] != e.value:
            vals.append(e.value)
    if not vals:
        return None
    return float(sum(vals[-3:]) / len(vals[-3:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2_dir", required=True)
    ap.add_argument("--c3_dirs", nargs="+", required=True)
    ap.add_argument("--c2_config", default="config/safety_gymnasium_fhdcmpo_c2_spreadmatch_med.yaml")
    ap.add_argument("--c3_config", default="config/safety_gymnasium_fhdcmpo_c3_spreadmatch_cvar.yaml")
    args = ap.parse_args()

    c2_viol = last3(args.c2_dir, "Eval/violation_rate")
    c2_rew = last3(args.c2_dir, "Eval/reward")
    seeds = []
    for d in args.c3_dirs:
        v, r = last3(d, "Eval/violation_rate"), last3(d, "Eval/reward")
        if v is not None and r is not None:
            seeds.append((v, r))
        print(f"note: {d}: viol {v} reward {r}")
    print(f"note: c2 ({args.c2_dir}): viol {c2_viol} reward {c2_rew}")

    if c2_viol is None or c2_rew is None or not seeds:
        print("note: telemetry missing -> defaulting to c2")
        print("winner: c2")
        print(f"config: {args.c2_config}")
        return

    v_med = statistics.median(v for v, _ in seeds)
    r_med = statistics.median(r for _, r in seeds)
    better_tail = v_med < c2_viol - 0.05
    no_collapse = r_med >= 0.5 * c2_rew
    print(f"note: c3 median viol {v_med:.3f} vs c2 {c2_viol:.3f} (better_tail={better_tail}); "
          f"c3 median reward {r_med:.2f} vs c2 {c2_rew:.2f} (no_collapse={no_collapse})")
    if better_tail and no_collapse:
        print("winner: c3")
        print(f"config: {args.c3_config}")
    else:
        print("winner: c2")
        print(f"config: {args.c2_config}")


if __name__ == "__main__":
    main()
