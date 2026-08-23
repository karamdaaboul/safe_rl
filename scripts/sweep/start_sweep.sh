#!/usr/bin/env bash
# Single entry point for the multi-env sweep. nohup/systemd friendly: the queue detaches its
# children (start_new_session), so killing the queue does NOT kill the trainings -- use
# stop_sweep.sh for that.
#
# Usage:
#   nohup bash scripts/sweep/start_sweep.sh > logs/sweep/queue.log 2>&1 &
#   N=3 ITERS=40000 nohup bash scripts/sweep/start_sweep.sh > logs/sweep/queue.log 2>&1 &
#
# Env overrides: N (max concurrent), ITERS, GPU (CUDA_VISIBLE_DEVICES; 1 = Blackwell in CUDA
# ordering, which is nvidia-smi's index 0), MIN_FREE (GiB), MANIFEST.
set -u
cd "$(dirname "$0")/../.."
PY=${SWEEP_PYTHON:-/home/human/venvs/agx_plain/bin/python}
mkdir -p logs/sweep

# Online wandb, explicitly. The cluster recipes export WANDB_MODE=offline and that value has a
# habit of following a shell around; this box is logged in (~/.netrc), so pin it rather than
# inherit it. The project itself is NOT set here -- `wandb_utils.init_wandb` reads
# `cfg["wandb_project"]` with no env fallback, so it lives in each generated config
# (SafeRL-multienv-sweep), keeping the sweep out of the main SafeRL project.
export WANDB_MODE=${WANDB_MODE:-online}

N=${N:-2}
ITERS=${ITERS:-}
GPU=${GPU:-1}
MIN_FREE=${MIN_FREE:-9.0}
MANIFEST=${MANIFEST:-config/sweep/manifest.json}

if [[ ! -f "$MANIFEST" ]]; then
    echo "no manifest at $MANIFEST -- run: $PY scripts/gen_sweep_configs.py" >&2
    exit 1
fi

ARGS=(--manifest "$MANIFEST" --max-concurrent "$N" --gpu "$GPU" --min-free-gib "$MIN_FREE")
[[ -n "$ITERS" ]] && ARGS+=(--iters "$ITERS")

PROJ=$(grep -h "wandb_project:" config/sweep/*.yaml | head -1 | sed "s/.*wandb_project: //")
echo "$(date +%F\ %T) starting sweep queue: N=$N GPU=$GPU MIN_FREE=${MIN_FREE}GiB ITERS=${ITERS:-<manifest>}"
echo "  wandb: mode=$WANDB_MODE project=$PROJ"
echo "  status:  bash scripts/sweep/status_sweep.sh"
echo "  stop:    bash scripts/sweep/stop_sweep.sh"
exec "$PY" scripts/sweep/local_queue.py "${ARGS[@]}"
