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
* **CUP** (Constrained Policy Optimization) - Alternative constraint handling approach
* **PPOL_PID** (PPO Lagrangian with PID Controller) - Safe RL using Lagrangian multipliers updated via PID control

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
