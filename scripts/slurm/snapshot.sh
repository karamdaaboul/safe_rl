#!/bin/bash
# Print a one-shot progress snapshot of the P3O HL-Gauss vs MSE comparison jobs.
LOGS=/p/scratch/hai_1075/safe_rl/logs
echo "=== SNAPSHOT $(date '+%F %T') ==="
echo "--- queue ---"
squeue --me --format="%.12i %.10j %.8T %.10M %R" | grep -E "g2pcpo_|g2cpo_" || echo "(no comparison jobs in queue)"
echo "--- progress (latest per job) ---"
for f in "$LOGS"/g2c40*.out; do
  [ -f "$f" ] || continue
  it=$(grep -aE "Learning iteration" "$f" | tail -1 | tr -s ' ')
  rew=$(grep -aE "Mean reward:" "$f" | tail -1 | tr -s ' ')
  cost=$(grep -aE "Constraint 0 cost:" "$f" | tail -1 | tr -s ' ')
  clip=$(grep -aE "cost_return_clip_frac" "$f" | tail -1 | tr -s ' ')
  echo "[$(basename "$f")] ${it} | ${rew} | ${cost} | ${clip}"
done
echo "--- errors (if any) ---"
for e in "$LOGS"/g2c40*.err; do
  [ -f "$e" ] || continue
  if grep -aqE "Traceback|Error|error|CUDA|Killed|oom" "$e"; then
    echo "[$(basename "$e")]"; grep -aE "Traceback|Error|error|CUDA|Killed|oom" "$e" | tail -4
  fi
done
echo "=== END SNAPSHOT ==="
