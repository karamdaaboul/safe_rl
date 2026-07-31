#!/bin/bash -l
#SBATCH --job-name=reppo_mjlab_setup
#SBATCH --account=hai_1074
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# One-time: build the Booster-native mjlab venv OFFLINE from the wheelhouse
# (compute nodes have no internet; wheels pre-downloaded on a login node into
# /p/project1/hai_1075/wheelhouse). Needed because venvs built on the JUWELS
# Cluster login symlink python into /p/software/juwels, absent on Booster.
set -euo pipefail

module --force purge
module load Stages/2024 GCCcore/.12.3.0 Python/3.11.3

VENV=/p/project1/hai_1075/venvs/mjlab311_booster
W=/p/project1/hai_1075/wheelhouse
export XDG_CACHE_HOME=/p/scratch/hai_1075/cache

rm -rf "$VENV"
python -m venv "$VENV"
"$VENV/bin/pip" install -q --no-index --find-links "$W" --upgrade pip
"$VENV/bin/pip" install -q --no-index --find-links "$W" \
    torch torchvision "mjlab==1.2.0" "mujoco==3.5.0" "warp-lang==1.12.1" \
    scipy wandb tensorboard pyyaml numpy GitPython onnx tqdm
"$VENV/bin/pip" install -q --no-index --no-deps -e /p/project1/hai_1075/workspaces/safe_rl

"$VENV/bin/python" - <<'EOF'
import sys
sys.path.insert(0, "/p/project1/hai_1075/workspaces/unitree_rl_mjlab")
import torch, mjlab.tasks
from mjlab.tasks.registry import list_tasks
import src.tasks
assert "Ant-Flat" in list_tasks(), list_tasks()
print("SETUP OK: torch", torch.__version__, "cuda:", torch.cuda.is_available(), "| Ant-Flat registered")
EOF
