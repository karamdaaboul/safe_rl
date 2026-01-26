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

# Check if any directories exist to avoid errors if the glob matches nothing
if compgen -G "offline-run-20260*" > /dev/null; then
    for d in offline-run-20260*; do
        echo "Syncing $d ..."
        # Simply sync the directory; wandb will find the correct ID inside
        wandb sync "$d"
        
        # Optional: Only delete if you are absolutely sure. 
        # Better to do this manually after verifying the dashboard.
        # rm -rf "$d" 
    done
else
    echo "No offline runs found matching pattern."
fi