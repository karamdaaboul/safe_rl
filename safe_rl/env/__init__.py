"""Submodule defining the environment definitions."""

try:
    from safe_rl.envs import MjlabVecEnv
except ImportError:
    MjlabVecEnv = None
from safe_rl.envs import VecEnv

try:
    from safe_rl.envs import SafetyGymnasiumVecEnv
except ImportError:
    SafetyGymnasiumVecEnv = None


__all__ = ["VecEnv", "MjlabVecEnv", "SafetyGymnasiumVecEnv"]
