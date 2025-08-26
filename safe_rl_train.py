#!/usr/bin/env python3

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for Safe RL algorithms with Weights & Biases sweep support.

This script integrates with wandb sweeps for hyperparameter optimization.
It can be used standalone or as part of a wandb sweep.

Usage:
    # Standalone training
    python safe_rl_train.py --task Isaac-Velocity-Flat-H1-v0 --num_envs 4096
    
    # With wandb sweep
    wandb sweep sweeps/ppo_sweep.yaml
    wandb agent SWEEP_ID
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import copy
import yaml
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Add unitree_rl_lab to path if needed
unitree_rl_lab_path = "/home/dh1659/workspace/unitree_rl_lab"
if unitree_rl_lab_path not in sys.path:
    sys.path.append(unitree_rl_lab_path)

from isaaclab.app import AppLauncher

# local imports
try:
    import cli_args  # from unitree_rl_lab
except ImportError:
    print("Warning: cli_args not found. Creating minimal CLI args handler.")
    class cli_args:
        @staticmethod
        def add_rsl_rl_args(parser):
            # Minimal RSL-RL arguments
            arg_group = parser.add_argument_group("safe_rl", description="Arguments for Safe RL agent.")
            arg_group.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment folder.")
            arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
            arg_group.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
            arg_group.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
            arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
            arg_group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module.")
            arg_group.add_argument("--log_project_name", type=str, default=None, help="Logging project name.")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Safe RL agent with wandb sweep support.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

# PPO Algorithm Parameters
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--num_learning_epochs", type=int, help="Number of learning epochs")
parser.add_argument("--num_mini_batches", type=int, help="Number of mini batches")
parser.add_argument("--clip_param", type=float, help="PPO clipping parameter")
parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm")
parser.add_argument("--schedule", type=str, choices=["fixed", "adaptive"], help="Learning rate schedule")
parser.add_argument("--desired_kl", type=float, help="Desired KL divergence")
parser.add_argument("--gamma", type=float, help="Discount factor")
parser.add_argument("--lam", type=float, help="GAE lambda")
parser.add_argument("--value_loss_coef", type=float, help="Value loss coefficient")
parser.add_argument("--entropy_coef", type=float, help="Entropy coefficient")
parser.add_argument("--use_clipped_value_loss", action="store_true", help="Use clipped value loss")
parser.add_argument("--normalize_advantage_per_mini_batch", action="store_true", help="Normalize advantage per mini batch")
parser.add_argument("--algorithm_class", type=str, choices=["PPO", "P3O", "PPOL_PID"], help="Algorithm class")

# Policy Network Architecture
parser.add_argument("--actor_hidden_dims_0", type=int, help="First actor hidden layer size")
parser.add_argument("--actor_hidden_dims_1", type=int, help="Second actor hidden layer size")
parser.add_argument("--actor_hidden_dims_2", type=int, help="Third actor hidden layer size")
parser.add_argument("--critic_hidden_dims_0", type=int, help="First critic hidden layer size") 
parser.add_argument("--critic_hidden_dims_1", type=int, help="Second critic hidden layer size")
parser.add_argument("--critic_hidden_dims_2", type=int, help="Third critic hidden layer size")
parser.add_argument("--cost_critic_hidden_dims_0", type=int, help="First cost critic hidden layer size")
parser.add_argument("--cost_critic_hidden_dims_1", type=int, help="Second cost critic hidden layer size")
parser.add_argument("--activation", type=str, help="Network activation function")
parser.add_argument("--init_noise_std", type=float, help="Initial noise standard deviation")

# Runner Parameters
parser.add_argument("--num_steps_per_env", type=int, help="Number of steps per environment")

# RND Parameters
parser.add_argument("--rnd_weight", type=float, help="RND reward weight")
parser.add_argument("--rnd_learning_rate", type=float, help="RND learning rate")
parser.add_argument("--rnd_reward_normalization", action="store_true", help="RND reward normalization")
parser.add_argument("--rnd_state_normalization", action="store_true", help="RND state normalization")

# Safe RL specific parameters
parser.add_argument("--p3o_cost_limit", type=float, help="P3O cost limit")
parser.add_argument("--p3o_lambda_lr", type=float, help="P3O lambda learning rate")
parser.add_argument("--p3o_kl_early_stop", action="store_true", help="P3O KL early stopping")
parser.add_argument("--ppol_pid_kp", type=float, help="PPOL_PID proportional gain")
parser.add_argument("--ppol_pid_ki", type=float, help="PPOL_PID integral gain")
parser.add_argument("--ppol_pid_kd", type=float, help="PPOL_PID derivative gain")
parser.add_argument("--ppol_pid_cost_limit", type=float, help="PPOL_PID cost limit")

# Environment reward weight parameters
parser.add_argument("--track_lin_vel_xy_weight", type=float, help="Linear velocity tracking reward weight")
parser.add_argument("--track_ang_vel_z_weight", type=float, help="Angular velocity tracking reward weight")
parser.add_argument("--base_linear_velocity_weight", type=float, help="Base linear velocity penalty weight")
parser.add_argument("--base_angular_velocity_weight", type=float, help="Base angular velocity penalty weight")
parser.add_argument("--flat_orientation_l2_weight", type=float, help="Flat orientation penalty weight")
parser.add_argument("--joint_torques_weight", type=float, help="Joint torques penalty weight")
parser.add_argument("--action_rate_weight", type=float, help="Action rate penalty weight")
parser.add_argument("--energy_weight", type=float, help="Energy penalty weight")
parser.add_argument("--feet_air_time_weight", type=float, help="Feet air time reward weight")
parser.add_argument("--air_time_variance_weight", type=float, help="Air time variance penalty weight")

# Configuration file (fallback)
parser.add_argument("--config", type=str, default="config/dummy_config.yaml", help="Path to config file")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata
import platform
from packaging import version

import gymnasium as gym
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Sweep integration disabled.")

from safe_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl.safe_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from unitree_lab.envs import ManagerBasedSafeRLEnvCfg
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnvCfg
)
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config: Dict[str, Any], args_cli) -> Dict[str, Any]:
    """Update configuration with command line arguments and wandb sweep parameters."""
    config = copy.deepcopy(config)
    
    # Handle wandb sweep parameters
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb_config = dict(wandb.config)
        args_dict = vars(args_cli)
        
        # Update args with wandb config
        for key, value in wandb_config.items():
            if key in args_dict and value is not None:
                setattr(args_cli, key, value)
    
    # Update config with args
    if args_cli.seed is not None:
        config["seed"] = args_cli.seed
        
    # Algorithm parameters
    alg_updates = {}
    if hasattr(args_cli, 'learning_rate') and args_cli.learning_rate is not None:
        alg_updates["learning_rate"] = args_cli.learning_rate
    if hasattr(args_cli, 'num_learning_epochs') and args_cli.num_learning_epochs is not None:
        alg_updates["num_learning_epochs"] = args_cli.num_learning_epochs
    if hasattr(args_cli, 'num_mini_batches') and args_cli.num_mini_batches is not None:
        alg_updates["num_mini_batches"] = args_cli.num_mini_batches
    if hasattr(args_cli, 'clip_param') and args_cli.clip_param is not None:
        alg_updates["clip_param"] = args_cli.clip_param
    if hasattr(args_cli, 'max_grad_norm') and args_cli.max_grad_norm is not None:
        alg_updates["max_grad_norm"] = args_cli.max_grad_norm
    if hasattr(args_cli, 'schedule') and args_cli.schedule is not None:
        alg_updates["schedule"] = args_cli.schedule
    if hasattr(args_cli, 'desired_kl') and args_cli.desired_kl is not None:
        alg_updates["desired_kl"] = args_cli.desired_kl
    if hasattr(args_cli, 'gamma') and args_cli.gamma is not None:
        alg_updates["gamma"] = args_cli.gamma
    if hasattr(args_cli, 'lam') and args_cli.lam is not None:
        alg_updates["lam"] = args_cli.lam
    if hasattr(args_cli, 'value_loss_coef') and args_cli.value_loss_coef is not None:
        alg_updates["value_loss_coef"] = args_cli.value_loss_coef
    if hasattr(args_cli, 'entropy_coef') and args_cli.entropy_coef is not None:
        alg_updates["entropy_coef"] = args_cli.entropy_coef
    if hasattr(args_cli, 'use_clipped_value_loss') and args_cli.use_clipped_value_loss:
        alg_updates["use_clipped_value_loss"] = args_cli.use_clipped_value_loss
    if hasattr(args_cli, 'normalize_advantage_per_mini_batch') and args_cli.normalize_advantage_per_mini_batch:
        alg_updates["normalize_advantage_per_mini_batch"] = args_cli.normalize_advantage_per_mini_batch
    if hasattr(args_cli, 'algorithm_class') and args_cli.algorithm_class is not None:
        alg_updates["class_name"] = args_cli.algorithm_class
        
    # Update algorithm config
    if "algorithm" in config:
        config["algorithm"].update(alg_updates)
    
    # Policy network architecture
    policy_updates = {}
    
    # Handle actor hidden dims
    actor_dims = []
    if hasattr(args_cli, 'actor_hidden_dims_0') and args_cli.actor_hidden_dims_0 is not None:
        actor_dims.append(args_cli.actor_hidden_dims_0)
    if hasattr(args_cli, 'actor_hidden_dims_1') and args_cli.actor_hidden_dims_1 is not None:
        actor_dims.append(args_cli.actor_hidden_dims_1)
    if hasattr(args_cli, 'actor_hidden_dims_2') and args_cli.actor_hidden_dims_2 is not None:
        actor_dims.append(args_cli.actor_hidden_dims_2)
    if actor_dims:
        policy_updates["actor_hidden_dims"] = actor_dims
        
    # Handle critic hidden dims
    critic_dims = []
    if hasattr(args_cli, 'critic_hidden_dims_0') and args_cli.critic_hidden_dims_0 is not None:
        critic_dims.append(args_cli.critic_hidden_dims_0)
    if hasattr(args_cli, 'critic_hidden_dims_1') and args_cli.critic_hidden_dims_1 is not None:
        critic_dims.append(args_cli.critic_hidden_dims_1)
    if hasattr(args_cli, 'critic_hidden_dims_2') and args_cli.critic_hidden_dims_2 is not None:
        critic_dims.append(args_cli.critic_hidden_dims_2)
    if critic_dims:
        policy_updates["critic_hidden_dims"] = critic_dims
        
    # Handle cost critic hidden dims (for safe RL)
    cost_critic_dims = []
    if hasattr(args_cli, 'cost_critic_hidden_dims_0') and args_cli.cost_critic_hidden_dims_0 is not None:
        cost_critic_dims.append(args_cli.cost_critic_hidden_dims_0)
    if hasattr(args_cli, 'cost_critic_hidden_dims_1') and args_cli.cost_critic_hidden_dims_1 is not None:
        cost_critic_dims.append(args_cli.cost_critic_hidden_dims_1)
    if cost_critic_dims:
        policy_updates["cost_critic_hidden_dims"] = cost_critic_dims
        
    if hasattr(args_cli, 'activation') and args_cli.activation is not None:
        policy_updates["activation"] = args_cli.activation
    if hasattr(args_cli, 'init_noise_std') and args_cli.init_noise_std is not None:
        policy_updates["init_noise_std"] = args_cli.init_noise_std
        
    # Update policy config
    if "policy" in config:
        config["policy"].update(policy_updates)
    
    # Runner parameters
    runner_updates = {}
    if hasattr(args_cli, 'num_steps_per_env') and args_cli.num_steps_per_env is not None:
        runner_updates["num_steps_per_env"] = args_cli.num_steps_per_env
    if args_cli.max_iterations is not None:
        runner_updates["max_iterations"] = args_cli.max_iterations
        
    if "runner" in config:
        config["runner"].update(runner_updates)
    
    # RND parameters
    if "algorithm" in config and "rnd_cfg" in config["algorithm"] and config["algorithm"]["rnd_cfg"]:
        rnd_updates = {}
        if hasattr(args_cli, 'rnd_weight') and args_cli.rnd_weight is not None:
            rnd_updates["weight"] = args_cli.rnd_weight
        if hasattr(args_cli, 'rnd_learning_rate') and args_cli.rnd_learning_rate is not None:
            rnd_updates["learning_rate"] = args_cli.rnd_learning_rate
        if hasattr(args_cli, 'rnd_reward_normalization') and args_cli.rnd_reward_normalization:
            rnd_updates["reward_normalization"] = args_cli.rnd_reward_normalization
        if hasattr(args_cli, 'rnd_state_normalization') and args_cli.rnd_state_normalization:
            rnd_updates["state_normalization"] = args_cli.rnd_state_normalization
            
        config["algorithm"]["rnd_cfg"].update(rnd_updates)
    
    # Safe RL specific parameters
    if hasattr(args_cli, 'algorithm_class') and args_cli.algorithm_class == "P3O":
        if hasattr(args_cli, 'p3o_cost_limit') and args_cli.p3o_cost_limit is not None:
            config["algorithm"]["cost_limit"] = args_cli.p3o_cost_limit
        if hasattr(args_cli, 'p3o_lambda_lr') and args_cli.p3o_lambda_lr is not None:
            config["algorithm"]["lambda_lr"] = args_cli.p3o_lambda_lr
        if hasattr(args_cli, 'p3o_kl_early_stop') and args_cli.p3o_kl_early_stop:
            config["algorithm"]["kl_early_stop"] = args_cli.p3o_kl_early_stop
            
    elif hasattr(args_cli, 'algorithm_class') and args_cli.algorithm_class == "PPOL_PID":
        pid_updates = {}
        if hasattr(args_cli, 'ppol_pid_kp') and args_cli.ppol_pid_kp is not None:
            pid_updates["kp"] = args_cli.ppol_pid_kp
        if hasattr(args_cli, 'ppol_pid_ki') and args_cli.ppol_pid_ki is not None:
            pid_updates["ki"] = args_cli.ppol_pid_ki  
        if hasattr(args_cli, 'ppol_pid_kd') and args_cli.ppol_pid_kd is not None:
            pid_updates["kd"] = args_cli.ppol_pid_kd
        if hasattr(args_cli, 'ppol_pid_cost_limit') and args_cli.ppol_pid_cost_limit is not None:
            config["algorithm"]["cost_limit"] = args_cli.ppol_pid_cost_limit
            
        if pid_updates:
            config["algorithm"]["pid_cfg"] = config["algorithm"].get("pid_cfg", {})
            config["algorithm"]["pid_cfg"].update(pid_updates)
    
    return config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | ManagerBasedSafeRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with Safe RL agent and wandb sweep support."""
    try:
        # Initialize wandb if available and not already initialized
        if WANDB_AVAILABLE and wandb.run is None:
            # Only initialize if not running as part of a sweep
            run_id = os.getenv("WANDB_RUN_ID")
            if run_id is None:
                # Initialize without config first, we'll update it later
                wandb.init(project="safe_rl")
        
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # set the environment seed
        env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
        env_cfg.sim.device = args_cli.device if hasattr(args_cli, 'device') and args_cli.device is not None else env_cfg.sim.device

        # Update environment reward weights from CLI args and wandb sweep
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb_config = dict(wandb.config)
            # Update reward weights from wandb config
            if "track_lin_vel_xy_weight" in wandb_config and wandb_config["track_lin_vel_xy_weight"] is not None:
                env_cfg.rewards.track_lin_vel_xy.weight = wandb_config["track_lin_vel_xy_weight"]
            if "track_ang_vel_z_weight" in wandb_config and wandb_config["track_ang_vel_z_weight"] is not None:
                env_cfg.rewards.track_ang_vel_z.weight = wandb_config["track_ang_vel_z_weight"]
            if "base_linear_velocity_weight" in wandb_config and wandb_config["base_linear_velocity_weight"] is not None:
                env_cfg.rewards.base_linear_velocity.weight = wandb_config["base_linear_velocity_weight"]
            if "base_angular_velocity_weight" in wandb_config and wandb_config["base_angular_velocity_weight"] is not None:
                env_cfg.rewards.base_angular_velocity.weight = wandb_config["base_angular_velocity_weight"]
            if "flat_orientation_l2_weight" in wandb_config and wandb_config["flat_orientation_l2_weight"] is not None:
                env_cfg.rewards.flat_orientation_l2.weight = wandb_config["flat_orientation_l2_weight"]
            if "joint_torques_weight" in wandb_config and wandb_config["joint_torques_weight"] is not None:
                env_cfg.rewards.joint_torques.weight = wandb_config["joint_torques_weight"]
            if "action_rate_weight" in wandb_config and wandb_config["action_rate_weight"] is not None:
                env_cfg.rewards.action_rate.weight = wandb_config["action_rate_weight"]
            if "energy_weight" in wandb_config and wandb_config["energy_weight"] is not None:
                env_cfg.rewards.energy.weight = wandb_config["energy_weight"]
            if "feet_air_time_weight" in wandb_config and wandb_config["feet_air_time_weight"] is not None:
                env_cfg.rewards.feet_air_time.weight = wandb_config["feet_air_time_weight"]
            if "air_time_variance_weight" in wandb_config and wandb_config["air_time_variance_weight"] is not None:
                env_cfg.rewards.air_time_variance.weight = wandb_config["air_time_variance_weight"]

        # Update reward weights from CLI args (these take precedence)
        if hasattr(args_cli, 'track_lin_vel_xy_weight') and args_cli.track_lin_vel_xy_weight is not None:
            env_cfg.rewards.track_lin_vel_xy.weight = args_cli.track_lin_vel_xy_weight
        if hasattr(args_cli, 'track_ang_vel_z_weight') and args_cli.track_ang_vel_z_weight is not None:
            env_cfg.rewards.track_ang_vel_z.weight = args_cli.track_ang_vel_z_weight
        if hasattr(args_cli, 'base_linear_velocity_weight') and args_cli.base_linear_velocity_weight is not None:
            env_cfg.rewards.base_linear_velocity.weight = args_cli.base_linear_velocity_weight
        if hasattr(args_cli, 'base_angular_velocity_weight') and args_cli.base_angular_velocity_weight is not None:
            env_cfg.rewards.base_angular_velocity.weight = args_cli.base_angular_velocity_weight
        if hasattr(args_cli, 'flat_orientation_l2_weight') and args_cli.flat_orientation_l2_weight is not None:
            env_cfg.rewards.flat_orientation_l2.weight = args_cli.flat_orientation_l2_weight
        if hasattr(args_cli, 'joint_torques_weight') and args_cli.joint_torques_weight is not None:
            env_cfg.rewards.joint_torques.weight = args_cli.joint_torques_weight
        if hasattr(args_cli, 'action_rate_weight') and args_cli.action_rate_weight is not None:
            env_cfg.rewards.action_rate.weight = args_cli.action_rate_weight
        if hasattr(args_cli, 'energy_weight') and args_cli.energy_weight is not None:
            env_cfg.rewards.energy.weight = args_cli.energy_weight
        if hasattr(args_cli, 'feet_air_time_weight') and args_cli.feet_air_time_weight is not None:
            env_cfg.rewards.feet_air_time.weight = args_cli.feet_air_time_weight
        if hasattr(args_cli, 'air_time_variance_weight') and args_cli.air_time_variance_weight is not None:
            env_cfg.rewards.air_time_variance.weight = args_cli.air_time_variance_weight

        # multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"

            # set seed to have diversity in different threads
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # Update RSL-RL config with CLI args
        if hasattr(args_cli, 'experiment_name') and args_cli.experiment_name is not None:
            agent_cfg.experiment_name = args_cli.experiment_name
        if hasattr(args_cli, 'run_name') and args_cli.run_name is not None:
            agent_cfg.run_name = args_cli.run_name
        if hasattr(args_cli, 'logger') and args_cli.logger is not None:
            agent_cfg.logger = args_cli.logger
        if hasattr(args_cli, 'log_project_name') and args_cli.log_project_name is not None and agent_cfg.logger in {"wandb", "neptune"}:
            agent_cfg.wandb_project = args_cli.log_project_name
            agent_cfg.neptune_project = args_cli.log_project_name

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        
        # specify directory for logging runs
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Convert agent config to dict and update with sweep parameters
        config = agent_cfg.to_dict()
        config = update_config_from_args(config, args_cli)

        # create runner from safe_rl
        runner = OnPolicyRunner(env, config, log_dir=log_dir, device=agent_cfg.device)
        
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        
        # load the checkpoint
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), config)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), config)
        export_deploy_cfg(env.unwrapped, log_dir)

        # Update wandb config if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)

        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        # close the simulator
        env.close()

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()