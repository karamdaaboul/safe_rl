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

# Sync all unsynced runs in the current directory
# --sync-all automatically skips already synced runs
wandb sync --sync-all .