#!/usr/bin/env bash
# Stop the queue and, with --runs, the trainings it started.
# Default stops ONLY the queue: in-flight runs finish, nothing new starts -- the safe default,
# since killing a training discards its whole elapsed wall-clock.
set -u
cd "$(dirname "$0")/../.."
pkill -f "sweep/local_queue.py" && echo "queue stopped" || echo "no queue process found"
if [[ "${1:-}" == "--runs" ]]; then
    echo "stopping trainings started by the sweep..."
    pkill -f "config/sweep/.*\.yaml" && echo "trainings signalled" || echo "no sweep trainings found"
else
    n=$(pgrep -fc "config/sweep/.*\.yaml" 2>/dev/null || echo 0)
    echo "$n sweep training(s) still running; they will finish. Use --runs to kill them too."
fi
