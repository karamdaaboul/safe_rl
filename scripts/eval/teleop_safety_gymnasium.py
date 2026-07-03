from __future__ import annotations

import argparse
import platform
import re
import sys

import numpy as np

# GLFW 3.x key codes.
KEY_ESCAPE = 256
KEY_Q = 81
KEY_N = 78
KEY_BACKSPACE = 259
KEY_W = 87
KEY_A = 65
KEY_S = 83
KEY_D = 68
KEY_I = 73
KEY_J = 74
KEY_K = 75
KEY_L = 76
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262

FORWARD_KEYS = {KEY_I, KEY_UP, KEY_W}
BACKWARD_KEYS = {KEY_K, KEY_DOWN, KEY_S}
LEFT_KEYS = {KEY_J, KEY_LEFT, KEY_A}
RIGHT_KEYS = {KEY_L, KEY_RIGHT, KEY_D}
RESET_KEYS = {KEY_N, KEY_BACKSPACE}
QUIT_KEYS = {KEY_ESCAPE, KEY_Q}
ALL_KEYS = FORWARD_KEYS | BACKWARD_KEYS | LEFT_KEYS | RIGHT_KEYS | RESET_KEYS | QUIT_KEYS


def to_debug_env_id(env_id: str) -> str:
    """Map ``SafetyCarGoal1-v0`` -> ``SafetyCarGoal1Debug-v0``."""
    if "Debug" in env_id:
        return env_id
    match = re.match(r"^(Safety.+?)(-v\d+)$", env_id)
    if not match:
        raise ValueError(
            f"Cannot derive a Debug env id from {env_id!r}. "
            "Use an id like SafetyCarGoal1-v0 or pass SafetyCarGoal1Debug-v0 directly."
        )
    return f"{match.group(1)}Debug{match.group(2)}"


def get_task(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base.unwrapped.task


def get_viewer_window(task):
    viewer = getattr(task, "viewer", None)
    if viewer is None and hasattr(task, "_viewers"):
        viewer = task._viewers.get("human")
    if viewer is None:
        return None
    return getattr(viewer, "window", None)


def _keys_pressed(keys: set[int], key_set: set[int]) -> bool:
    return any(key in keys for key in key_set)


def keys_to_car_action(keys: set[int]) -> np.ndarray:
    action = np.zeros(2, dtype=np.float64)
    if _keys_pressed(keys, FORWARD_KEYS):
        action += np.array([1.0, 1.0])
    if _keys_pressed(keys, BACKWARD_KEYS):
        action += np.array([-1.0, -1.0])
    if _keys_pressed(keys, LEFT_KEYS):
        return np.array([1.0, -1.0])
    if _keys_pressed(keys, RIGHT_KEYS):
        return np.array([-1.0, 1.0])
    return action


def keys_to_point_action(keys: set[int]) -> np.ndarray:
    action = np.zeros(2, dtype=np.float64)
    if _keys_pressed(keys, FORWARD_KEYS):
        action[0] += 1.0
    if _keys_pressed(keys, BACKWARD_KEYS):
        action[0] -= 1.0
    if _keys_pressed(keys, LEFT_KEYS):
        action[1] += 1.0
    if _keys_pressed(keys, RIGHT_KEYS):
        action[1] -= 1.0
    return action


def keys_to_action(agent_name: str, keys: set[int], action_space) -> np.ndarray:
    name = agent_name.lower()
    if name in {"car", "racecar"}:
        action = keys_to_car_action(keys)
    elif name == "point":
        action = keys_to_point_action(keys)
    else:
        action = np.zeros(action_space.shape, dtype=np.float64)
    return np.clip(action, action_space.low, action_space.high)


def poll_pressed_keys(window) -> set[int]:
    """Poll GLFW key state directly (works for arrows/WASD, not only I/J/K/L)."""
    import glfw

    glfw.poll_events()
    pressed: set[int] = set()
    for key in ALL_KEYS:
        if glfw.get_key(window, key) == glfw.PRESS:
            pressed.add(key)
    return pressed


def read_keys(task, agent, use_debug_env: bool) -> set[int]:
    window = get_viewer_window(task)
    if window is not None:
        return poll_pressed_keys(window)
    if use_debug_env:
        return set(agent.debug_info.keys)
    return set()


def print_platform_notes() -> None:
    if platform.system() != "Darwin":
        return
    print(
        "\n[macOS] GLFW keyboard input must run on the main thread.\n"
        "If keys do nothing, launch with mjpython instead of python:\n"
        "  mjpython scripts/eval/teleop_safety_gymnasium.py --env_id SafetyCarGoal1-v0\n"
        "(mjpython is installed with the mujoco package; it is a drop-in for python.)\n"
    )


def print_controls(agent_name: str) -> None:
    print("\nClick the MuJoCo viewer window so it has keyboard focus, then drive:")
    if agent_name.lower() in {"car", "racecar"}:
        print("  Up/I/W    : forward")
        print("  Down/K/S  : backward")
        print("  Left/J/A  : turn left (in place)")
        print("  Right/L/D : turn right (in place)")
    else:
        print("  Up/I/W    : forward")
        print("  Down/K/S  : backward")
        print("  Left/J/A  : turn left")
        print("  Right/L/D : turn right")
    print("  N / Backspace : reset episode")
    print("  Q / Esc       : quit")
    print("\nTip: arrow keys are the most reliable on Mac.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually drive a Safety-Gymnasium agent to the goal with the keyboard."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="SafetyCarGoal1-v0",
        help="Safety-Gymnasium env id (e.g. SafetyCarGoal1-v0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reset seed (controls obstacle/goal layout).")
    parser.add_argument(
        "--camera_name",
        type=str,
        default="track",
        choices=["vision", "track", "fixednear", "fixedfar", "human"],
        help="Safety-Gymnasium camera.",
    )
    parser.add_argument(
        "--hidden_goal",
        action="store_true",
        help="Wrap env with HiddenGoalWrapper (masks goal_lidar from observations).",
    )
    parser.add_argument(
        "--debug_env",
        action="store_true",
        help=(
            "Use Safety-Gymnasium's *Debug-v0 env (I/J/K/L only, via library callback). "
            "Default is GLFW key polling on the regular env, which supports arrows/WASD."
        ),
    )
    args = parser.parse_args()

    print_platform_notes()

    import safety_gymnasium

    env_id = to_debug_env_id(args.env_id) if args.debug_env else args.env_id
    env = safety_gymnasium.make(
        env_id,
        render_mode="human",
        camera_name=args.camera_name,
    )
    if args.hidden_goal:
        from safe_rl.envs.hidden_goal_wrapper import HiddenGoalWrapper

        env = HiddenGoalWrapper(env)

    task = get_task(env)
    agent = task.agent

    obs, info = env.reset(seed=args.seed)
    # Create the GLFW viewer before the control loop.
    env.step(np.zeros(env.action_space.shape, dtype=np.float64))

    print_controls(agent.__class__.__name__)
    print(f"env_id={env_id}")
    if args.seed is not None:
        print(f"seed={args.seed}")

    if get_viewer_window(task) is None:
        print(
            "[WARN] MuJoCo viewer window not found. Keyboard teleop may not work.\n"
            "On macOS, retry with: mjpython scripts/eval/teleop_safety_gymnasium.py ...",
            file=sys.stderr,
        )

    episode_reward = 0.0
    episode_cost = 0.0
    episode_len = 0

    try:
        while True:
            keys = read_keys(task, agent, use_debug_env=args.debug_env)

            if _keys_pressed(keys, QUIT_KEYS):
                break
            if _keys_pressed(keys, RESET_KEYS):
                obs, info = env.reset(seed=args.seed)
                episode_reward = 0.0
                episode_cost = 0.0
                episode_len = 0
                print("[INFO] Episode reset.")
                continue

            if args.debug_env:
                action = np.zeros(env.action_space.shape, dtype=np.float64)
            else:
                action = keys_to_action(agent.__class__.__name__, keys, env.action_space)

            obs, reward, cost, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_cost += float(cost)
            episode_len += 1

            if terminated or truncated:
                goal_met = "yes" if info.get("goal_met") else "no"
                print(
                    f"Episode done | len={episode_len} reward={episode_reward:.3f} "
                    f"cost={episode_cost:.3f} goal_reached={goal_met}"
                )
                obs, info = env.reset(seed=args.seed)
                episode_reward = 0.0
                episode_cost = 0.0
                episode_len = 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
