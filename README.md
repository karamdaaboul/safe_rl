# Safe RL

A fast and simple implementation of safe reinforcement learning algorithms, designed to run fully on GPU.
This code is built on top of the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library and extends it with safe RL capabilities for constrained optimization problems.

## Implemented Algorithms

This library implements several **on-policy reinforcement learning algorithms** with a focus on safe RL:

### Standard RL Algorithms
* **PPO** (Proximal Policy Optimization) - The foundation algorithm for policy optimization
* **Student-Teacher Distillation** - Knowledge transfer from teacher to student policies

### Safe RL Algorithms
* **P3O** (Penalized Proximal Policy Optimization) - Safe RL using adaptive penalty methods for constraint handling
* **CUP** (Constrained Update Projection) - Two-phase safe RL with Lagrangian constraint projection ([Paper](https://arxiv.org/abs/2209.07089))
* **PPOL_PID** (PPO Lagrangian with PID Controller) - Safe RL using Lagrangian multipliers updated via PID control

### Algorithm Comparison

| Algorithm | Constraint Method | Key Feature |
|-----------|------------------|-------------|
| **P3O** | Adaptive penalty | Simple, single-phase update with adaptive κ |
| **CUP** | Lagrangian projection | Two-phase: PPO update → constraint projection |
| **PPOL_PID** | PID-controlled Lagrangian | Smooth constraint tracking with PID controller |

### Additional Features
* [Random Network Distillation (RND)](https://proceedings.mlr.press/v229/schwarke23a.html) - Encourages exploration by adding
  a curiosity driven intrinsic reward.
* [Symmetry-based Augmentation](https://arxiv.org/abs/2403.04359) - Makes the learned behaviors more symmetrical.

All algorithms are designed for **on-policy learning** and support cost-constrained environments for safe reinforcement learning.

We welcome contributions from the community. Please check our contribution guidelines for more
information.

**Built on**: [rsl_rl](https://github.com/leggedrobotics/rsl_rl) by Robotic Systems Lab, ETH Zurich & NVIDIA <br/>
**Extended for**: Safe Reinforcement Learning with multiple constraints handling capabilities



## Setup

Clone this repository and installing it with:

```bash
git clone git@git.algoryx.se:algoryx/external/xscave/safe-rl.git
cd safe_rl
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of PPO, please check the [dummy_config.yaml](config/dummy_config.yaml) file.

## Safety-Gymnasium usage

This repo includes a minimal wrapper and scripts to train/evaluate on
[Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/latest/).

Install Safety-Gymnasium:

```bash
pip install safety-gymnasium
```

Train PPO (standard RL):

```bash
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 8 \
  --config config/dummy_config.yaml
```

Train P3O (safe RL with adaptive penalty):

```bash
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 36 \
  --config config/safety_gymnasium_p3o.yaml \
  --cost_limits 25.0
```

Train CUP (safe RL with Lagrangian constraint projection):

```bash
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 36 \
  --config config/safety_gymnasium_cup.yaml \
  --cost_limits 25.0
```

Train PPOL-PID (safe RL with PID-controlled Lagrangian):

```bash
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 36 \
  --config config/safety_gymnasium_ppol_pid.yaml \
  --cost_limits 25.0
```

Evaluate a trained policy (single env, rendered):

```bash
python scripts/eval/eval_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 \
  --num_envs 1 \
  --render_mode human \
  --config config/dummy_config.yaml \
  --checkpoint logs/safety_gymnasium/SafetyCarGoal1-v0/<run>/model_<iter>.pt \
  --episodes 5
```


## Hyperparameter sweeps with Weights & Biases

Quick start (minimal PPO sweep):

```bash
# 1) Create the sweep and copy the returned SWEEP_ID
wandb sweep sweeps/quick_ppo_sweep.yaml

# 2) Launch one or more agents
wandb agent USERNAME/PROJECT/SWEEP_ID
```

Full PPO sweep:

```bash
wandb sweep sweeps/ppo_sweep.yaml
wandb agent USERNAME/PROJECT/SWEEP_ID
```

Safe RL sweep:

```bash
wandb sweep sweeps/safe_ppo_sweep.yaml
wandb agent USERNAME/PROJECT/SWEEP_ID
```

Notes:
- Ensure you are logged in: `wandb login`
- Replace `USERNAME/PROJECT` with your W&B entity and project.
- Run multiple `wandb agent` processes (or on multiple machines) for parallel sweeps.
