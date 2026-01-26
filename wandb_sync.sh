#!/usr/bin/env bash
set -euo pipefail

# ---- modules ----
module --force purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3

# ---- env ----
source /p/project1/hai_1075/venvs/safe_rl311/bin/activate

# ---- sync ----
cd /p/project1/hai_1075/workspaces/safe_rl/wandb

for d in offline-run-2026012*; do
  rid="resync-$(basename "$d" | sed 's/offline-run-//')"
  wandb sync --id "$rid" --clean "$d"
done