#!/usr/bin/env bash
# Tomorrow's 60k confirmation run: waits for the overnight c3 3-seed queue (pid $1), picks the
# winner (c2 vs c3) by the registered rule in pick_c_winner.py, launches winner at 60k seed 2.
set -u
cd "$(dirname "$0")/.."
PY=/home/human/venvs/agx_plain/bin/python
QUEUE_PID="${1:?pass the overnight queue pid}"
C2_DIR="${2:?pass the c2 run dir}"
MARKER="${3:?pass a marker file whose mtime predates the c3 runs}"

while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 300; done
echo "$(date +%T) overnight queue finished; picking winner"

# c3 run dirs = FHDCMPO dirs newer than the marker (the 3 overnight seeds).
mapfile -t C3_DIRS < <(find logs/safety_gymnasium/SafetyPointGoal1-v0/FHDCMPO -maxdepth 1 -mindepth 1 -type d -newer "$MARKER" | sort)
echo "c3 dirs: ${C3_DIRS[*]:-none}"

DECISION=$($PY scripts/analysis/pick_c_winner.py --c2_dir "$C2_DIR" --c3_dirs "${C3_DIRS[@]}" 2>&1)
echo "$DECISION"
CFG=$(echo "$DECISION" | grep "^config: " | cut -d' ' -f2)
echo "$(date +%T) launching 60k: $CFG"
CUDA_VISIBLE_DEVICES=1 $PY scripts/train/train_safety_gymnasium.py \
    --env_id SafetyPointGoal1-v0 --num_envs 8 --device cuda:0 \
    --config "$CFG" --cost_limits 25.0 --max_iterations 60000 --seed 2 --deterministic
echo "$(date +%T) 60k run exited with code $?"
