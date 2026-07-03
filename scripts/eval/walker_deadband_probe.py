"""Probe the frozen low-level velocity walker's tracking + standing ability.

Bypasses the high-level policy: injects a FIXED velocity command into each env
(one command per env) and measures the actual base-frame velocity the walker
achieves, plus how far it drifts when commanded to stand (~0). This tells us
whether the "stops ~0.5m short / can't park on the goal" ceiling is a low-level
actuator limit (deadband / can't stand) rather than a reward-shaping problem.

Run from the safe_rl repo dir with G1_VELOCITY_POLICY_PATH set:
  python scripts/eval/walker_deadband_probe.py
"""

from __future__ import annotations

import os
import sys

import torch

# mjlab task registry + env builder.
import src.tasks.navigation  # noqa: F401  (registers the task)
from mjlab.tasks.registry import load_env_cfg
from src.envs import build_env

ENV_ID = "Unitree-G1-Nav-Obstacles-Safe-Collision"
DEVICE = "cuda:0"

# Per-axis scale the action term applies: velocity = action * scale.
VEL_SCALE = (1.0, 0.5, 0.8)

# (vx, vy, yaw) commands to test, one per env.
COMMANDS = [
    (0.0, 0.0, 0.0),   # stand still
    (0.1, 0.0, 0.0),   # right at the gait deadband (cmd_speed > 0.1 toggles stepping)
    (0.15, 0.0, 0.0),
    (0.2, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (1.0, 0.0, 0.0),   # full forward
]


def main() -> None:
    env_cfg = load_env_cfg(ENV_ID)
    n = len(COMMANDS)
    env_cfg.scene.num_envs = n
    env_cfg.seed = 0
    # Long episode so terminations don't interrupt the probe.
    env_cfg.episode_length_s = 1e9

    env = build_env(env_cfg, DEVICE, render_mode=None)
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = base.scene["robot"]

    scale = torch.tensor(VEL_SCALE, device=DEVICE)
    cmds = torch.tensor(COMMANDS, device=DEVICE)
    actions = cmds / scale  # invert: action = velocity / scale

    step_dt = base.step_dt
    settle_steps = int(round(3.0 / step_dt))   # let gait reach steady state
    measure_steps = int(round(2.0 / step_dt))  # average window
    print(f"step_dt={step_dt:.4f}s  settle={settle_steps} steps  measure={measure_steps} steps")

    env.reset()

    # Settle.
    for _ in range(settle_steps):
        env.step(actions)

    # Measure.
    vx_sum = torch.zeros(n, device=DEVICE)
    vy_sum = torch.zeros(n, device=DEVICE)
    speed_sum = torch.zeros(n, device=DEVICE)
    pos0 = robot.data.root_link_pos_w[:, :2].clone()
    fell = torch.zeros(n, device=DEVICE)
    for _ in range(measure_steps):
        env.step(actions)
        vb = robot.data.root_link_lin_vel_b[:, :2]
        vx_sum += vb[:, 0]
        vy_sum += vb[:, 1]
        speed_sum += torch.norm(vb, dim=-1)
        # crude fall check: projected-gravity z near -1 means upright
        gz = robot.data.projected_gravity_b[:, 2] if hasattr(robot.data, "projected_gravity_b") else None
        if gz is not None:
            fell += (gz > -0.5).float()
    pos1 = robot.data.root_link_pos_w[:, :2]
    drift = torch.norm(pos1 - pos0, dim=-1)  # XY travel during the 2s window

    vx_mean = (vx_sum / measure_steps).tolist()
    vy_mean = (vy_sum / measure_steps).tolist()
    speed_mean = (speed_sum / measure_steps).tolist()
    drift_l = drift.tolist()
    fell_l = (fell / measure_steps).tolist()

    print("\n cmd_vx | actual_vx | actual_speed | track%  | 2s_drift_m | fell%")
    print("-" * 66)
    for i, (cvx, cvy, cyaw) in enumerate(COMMANDS):
        track = (vx_mean[i] / cvx * 100) if cvx > 1e-6 else float("nan")
        print(f"  {cvx:4.2f}  |  {vx_mean[i]:+6.3f}   |   {speed_mean[i]:5.3f}      "
              f"| {track:6.1f}  |   {drift_l[i]:5.3f}    | {fell_l[i]*100:4.0f}")

    print("\nNotes:")
    print(" - cmd 0.00 row: 2s_drift is how far the robot wanders when told to STAND.")
    print("   Large drift => walker cannot hold position => can't park on the goal.")
    print(" - track% << 100 at small cmds => deadband/underactuation (can't inch in).")


if __name__ == "__main__":
    main()
