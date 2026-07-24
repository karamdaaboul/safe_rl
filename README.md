# Safe RL

A fast, GPU-first implementation of safe reinforcement learning algorithms, built on top of
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) and extended for constrained (cost-limited)
optimization.

## Implemented Algorithms

Safe RL algorithms maximize reward subject to a cost constraint; they differ mainly in *how* the
constraint is enforced.

| Algorithm | Type | Constraint handling / key feature | Paper |
|-----------|------|-----------------------------------|-------|
| **PPO** | RL (on-policy) | Clipped surrogate + GAE — the foundation | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| **SAC** | RL (off-policy) | Maximum-entropy actor–critic | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| **REPPO** | RL (on-policy) | Relative-entropy pathwise policy optimization | [arXiv:2507.11019](https://arxiv.org/abs/2507.11019) |
| **FastSAC** | RL (off-policy) | SAC + `torch.compile`/AMP/UTD, optional distributional critic | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| **FastTD3** | RL (off-policy) | TD3 + large-batch parallel updates, distributional critic | [arXiv:2505.22642](https://arxiv.org/abs/2505.22642) |
| **P3O** | Safe (on-policy) | Adaptive penalty κ on constraint violations | [arXiv:2205.11814](https://arxiv.org/abs/2205.11814) |
| **PPOL_PID** | Safe (on-policy) | PID-controlled Lagrangian multiplier | [arXiv:2007.03964](https://arxiv.org/abs/2007.03964) |
| **CPO** | Safe (on-policy) | TRPO-style trust-region constraint projection | [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) |
| **PCPO** | Safe (on-policy) | Projection-based CPO: reward step → cost projection | [arXiv:2010.03152](https://arxiv.org/abs/2010.03152) |
| **FPPO** | Safe (on-policy) | Predictor–corrector gradient projection | — |
| **SafeSAC** | Safe (off-policy) | SAC with a Lagrangian cost constraint | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| **Distillation** | Utility | Student–teacher policy distillation | — |

**Additional features:** [Random Network Distillation (RND)](https://proceedings.mlr.press/v229/schwarke23a.html)
for curiosity-driven exploration, and [symmetry-based augmentation](https://arxiv.org/abs/2403.04359).

> ⚠️ **Experimental — not validated:** the **CBF** (control-barrier-function) safety filter is a work
> in progress; please don't rely on it for experiments yet.

**Built on** [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (Robotic Systems Lab, ETH Zurich &
NVIDIA), extended for safe RL with multi-constraint support.

## Setup

```bash
git clone https://github.com/karamdaaboul/safe_rl.git
cd safe_rl
pip install -e .
```

Logging backends (set via the `logger` key): [TensorBoard](https://www.tensorflow.org/tensorboard/)
or [Weights & Biases](https://wandb.ai/site).

## Safety-Gymnasium usage

Train and evaluate on [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/latest/)
(`pip install safety-gymnasium`).

```bash
# Standard RL (PPO)
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 --num_envs 8 --config config/dummy_config.yaml

# Safe RL (P3O) — requires --cost_limits
python scripts/train/train_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 --num_envs 36 \
  --config config/safety_gymnasium_p3o.yaml --cost_limits 25.0

# Evaluate a checkpoint (single env, rendered)
python scripts/eval/eval_safety_gymnasium.py \
  --env_id SafetyCarGoal1-v0 --num_envs 1 --render_mode human \
  --config config/dummy_config.yaml \
  --checkpoint logs/safety_gymnasium/SafetyCarGoal1-v0/<run>/model_<iter>.pt --episodes 5
```

Safe RL algorithms require `--cost_limits` (or `algorithm.cost_limits` in the config); omitting it
silently disables constraint enforcement.

## Hyperparameter sweeps (Weights & Biases)

```bash
wandb login
wandb sweep sweeps/safe_ppo_sweep.yaml   # prints a SWEEP_ID
wandb agent USERNAME/PROJECT/SWEEP_ID    # launch one or more agents, optionally on several machines
```

Replace `USERNAME/PROJECT` with your W&B entity and project. Other sweep configs live in `sweeps/`.

We welcome contributions — please see the contribution guidelines before opening a PR.
