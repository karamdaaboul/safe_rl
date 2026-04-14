"""Submodule defining the environment definitions."""

from .mjlab_vec_env import MjlabVecEnv
from .vec_env import VecEnv

try:
    from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv
except ImportError:
    SafetyGymnasiumVecEnv = None
    

__all__ = ["VecEnv", "MjlabVecEnv", "SafetyGymnasiumVecEnv"]
