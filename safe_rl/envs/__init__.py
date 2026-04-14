"""Submodule defining the environment wrappers and registry."""

from .registry import make_env
from .vec_env import VecEnv

try:
    from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv
except ImportError:
    SafetyGymnasiumVecEnv = None

try:
    from .mjlab_vec_env import MjlabVecEnv
except ImportError:
    MjlabVecEnv = None

__all__ = ["VecEnv", "make_env", "SafetyGymnasiumVecEnv", "MjlabVecEnv"]
