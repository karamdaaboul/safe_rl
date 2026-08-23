#!/bin/bash -l
#SBATCH --job-name=mpo_x4
#SBATCH --account=hai_1074
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1075/safe_rl/logs/%x-%j.out
#SBATCH --error=/p/scratch/hai_1075/safe_rl/logs/%x-%j.err

# Packed mjlab job: up to 4 independent train+eval pipelines on one Booster node,
# one per A100 (Booster allocates full nodes, so single-GPU jobs waste 3 GPUs).
#
# Usage: sbatch mpo_mjlab_x4.sh SPEC [SPEC SPEC SPEC]
#   SPEC = config,seed,num_envs,run_name,env_id,max_iters   (comma-separated)
#   e.g. sbatch mpo_mjlab_x4.sh \
#     config/mjlab_ant_mpo_best.yaml,1,256,ant_best_s1,Ant-Flat,36000 \
#     config/mjlab_ant_mpo_best.yaml,2,256,ant_best_s2,Ant-Flat,36000
#
# Each slot trains, then runs the 50-episode deterministic eval on its final
# checkpoint (same pipeline as reppo_mjlab.sh). Per-run stdout goes to
# $SCR/safe_rl/logs/<run_name>.out so monitoring stays per-run.
set -uo pipefail

module --force purge
module load Stages/2024 GCCcore/.12.3.0 Python/3.11.3

# Booster-native venv (see reppo_mjlab_setup.sh; the Cluster-built one symlinks
# into /p/software/juwels, which Booster nodes don't mount).
source /p/project1/hai_1075/venvs/mjlab311_booster/bin/activate

export MUJOCO_GL=egl
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=offline
export WANDB_SILENT=true
# $HOME is over quota — keep every cache off it.
export MPLCONFIGDIR=/p/scratch/hai_1075/cache
export XDG_CACHE_HOME=/p/scratch/hai_1075/cache
export WANDB_DIR=/p/scratch/hai_1075/safe_rl/wandb

SCR=/p/scratch/hai_1075
LOGROOT=$SCR/safe_rl/reppo_test
RUNLOGS=$SCR/safe_rl/logs
mkdir -p "$RUNLOGS" $SCR/cache $SCR/safe_rl/wandb "$LOGROOT"
cd /p/project1/hai_1075/workspaces/safe_rl

run_slot() {  # $1=gpu index, $2=spec
    local gpu=$1
    IFS=, read -r config seed num_envs run_name env_id max_iters <<< "$2"
    echo "=== slot gpu=$gpu config=$config env=$env_id seed=$seed envs=$num_envs iters=$max_iters run=$run_name ==="
    export CUDA_VISIBLE_DEVICES=$gpu
    # 48 cores / 4 slots; mjlab sims on GPU, so 8 threads per slot is plenty.
    export OMP_NUM_THREADS=8
    local extra=()
    [ "$config" != "none" ] && extra+=(--config "$config")
    [ -n "$max_iters" ] && extra+=(--max_iterations "$max_iters")
    python -u scripts/train/unitree_mjlab.py \
        --env_id "$env_id" --num_envs "$num_envs" --seed "$seed" \
        --logger wandb --wandb_project mjlab \
        --run_name "$run_name" --log_dir "$LOGROOT" \
        "${extra[@]}"
    local rc=$?
    echo "=== TRAIN DONE rc=$rc run=$run_name ==="
    [ $rc -ne 0 ] && return $rc

    local ckpt
    ckpt=$(find "$LOGROOT" -path "*${run_name}*" -name "model_*.pt" ! -name "model_0.pt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2)
    if [ -z "$ckpt" ]; then
        echo "=== NO CHECKPOINT for $run_name ==="
        return 3
    fi
    echo "=== EVAL ckpt=$ckpt ==="
    local eval_extra=()
    [ "$config" != "none" ] && eval_extra+=(--config "$config")
    python -u scripts/eval/unitree_mjlab.py \
        --env_id "$env_id" --checkpoint "$ckpt" \
        --num_envs 64 --episodes 50 --headless --device cuda:0 \
        "${eval_extra[@]}"
    echo "=== EVAL DONE rc=$? run=$run_name ==="
}

pids=()
names=()
gpu=0
for spec in "$@"; do
    [ -z "$spec" ] && continue
    name=$(echo "$spec" | cut -d, -f4)
    run_slot "$gpu" "$spec" > "$RUNLOGS/${name}.out" 2>&1 &
    pids+=($!)
    names+=("$name")
    gpu=$((gpu + 1))
done

rc_all=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    echo "slot ${names[$i]}: rc=$rc"
    [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
