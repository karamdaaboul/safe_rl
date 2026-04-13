# JUWELS + `unitree_rl_mjlab`

This directory contains a JUWELS-ready bundle for running the upstream
`unitree_rl_mjlab` stack with `Apptainer`.

The compute nodes do not have internet access, so the intended flow is:

1. Log in on a JUWELS login node.
2. Build the container image there.
3. Submit offline Slurm jobs that use the prebuilt image.

## Files

- `build_unitree_rl_mjlab_image.sh`: clones/updates the upstream repo and builds the `.sif`.
- `unitree_rl_mjlab.def.template`: Apptainer definition template used by the build script.
- `unitree_rl_mjlab_interactive_check.sh`: requests one GPU and opens an interactive validation shell.
- `unitree_rl_mjlab_smoke.sbatch`: short smoke test job.
- `unitree_rl_mjlab_train.sbatch`: production training template.

## Default JUWELS layout

The scripts default to the `hai_1075` project layout used elsewhere in this repo:

- Project root: `/p/project1/hai_1075/unitree_rl_mjlab`
- Scratch root: `/p/scratch/hai_1075/unitree_rl_mjlab`
- Upstream source checkout: `/p/project1/hai_1075/unitree_rl_mjlab/src/unitree_rl_mjlab`
- Image output: `/p/project1/hai_1075/unitree_rl_mjlab/images`
- Logs/checkpoints: `/p/scratch/hai_1075/unitree_rl_mjlab`

Override these defaults by exporting environment variables before running the scripts.

## Login node commands

Once you have an authenticated `ssh juwels` session:

```bash
cd /p/project1/hai_1075/workspaces/safe_rl/experiments/juwels_unitree_rl_mjlab
bash build_unitree_rl_mjlab_image.sh
```

Optional overrides:

```bash
ACCOUNT=hai_1075 \
PROJECT_ROOT=/p/project1/hai_1075/unitree_rl_mjlab \
SCRATCH_ROOT=/p/scratch/hai_1075/unitree_rl_mjlab \
bash build_unitree_rl_mjlab_image.sh
```

## Interactive validation

This grabs one GPU, validates the image, lists Unitree tasks, and leaves you inside the container:

```bash
cd /p/project1/hai_1075/workspaces/safe_rl/experiments/juwels_unitree_rl_mjlab
bash unitree_rl_mjlab_interactive_check.sh
```

## Smoke test job

```bash
cd /p/project1/hai_1075/workspaces/safe_rl/experiments/juwels_unitree_rl_mjlab
sbatch unitree_rl_mjlab_smoke.sbatch
```

The smoke test uses `Unitree-Go2-Flat`, one GPU, a reduced env count, and a tiny number of iterations to validate startup.

## Production training

Example using four GPUs on JUWELS Booster:

```bash
cd /p/project1/hai_1075/workspaces/safe_rl/experiments/juwels_unitree_rl_mjlab
TASK_ID=Unitree-G1-Flat \
NUM_ENVS=4096 \
MAX_ITER=1500 \
GPU_IDS="0 1 2 3" \
sbatch unitree_rl_mjlab_train.sbatch
```

By default the training template targets the `booster` partition with four A100 GPUs.

## Notes

- The image is built on the login node so all package downloads happen before the job starts.
- Runtime jobs bind the upstream checkout into `/workspace/unitree_rl_mjlab`.
- Checkpoints and logs are written to scratch/project storage, not inside the image.
