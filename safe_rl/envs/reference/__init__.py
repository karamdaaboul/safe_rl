"""Reference-trajectory generators for CLF-RL reward shaping."""

from .hlip import HLIPReference, bezier5, bezier5_deriv, lip_velocity_profile

__all__ = ["HLIPReference", "bezier5", "bezier5_deriv", "lip_velocity_profile"]
