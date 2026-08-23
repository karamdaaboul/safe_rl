#!/usr/bin/env bash
# Human-readable sweep status: queue state, live runs, RAM headroom.
set -u
cd "$(dirname "$0")/../.."
PY=${SWEEP_PYTHON:-/home/human/venvs/agx_plain/bin/python}
"$PY" - <<'PYEOF'
import json, pathlib, subprocess
p = pathlib.Path("logs/sweep/sweep_status.json")
if not p.exists():
    print("no sweep_status.json yet (queue not started, or no heartbeat written)"); raise SystemExit(0)
s = json.loads(p.read_text())
print(f"updated {s['updated']}   queue={s.get('queue','?')}   "
      f"RAM free {s['free_gib']:.1f}/{s['total_gib']:.1f} GiB")
print(f"counts: {s['counts']}   live on box: {s['live_trainings_on_box']} "
      f"(this manifest {s['live_from_this_manifest']}, foreign {s['foreign_trainings']})")
order = {"running": 0, "dead": 1, "pending": 2, "finished": 3}
for r in sorted(s["runs"], key=lambda r: (order.get(r["state"], 9), r["name"])):
    extra = f" iter={r['last_iter']}" if r["last_iter"] >= 0 else ""
    rss = f" rss={r['rss_gib']}G" if "rss_gib" in r else ""
    print(f"  {r['state']:9s} {r['name']:26s}{extra}{rss}")
PYEOF
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
