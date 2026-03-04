# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submodule defining the environment definitions."""

from .vec_env import VecEnv

try:
    from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv
except ImportError:
    SafetyGymnasiumVecEnv = None

__all__ = ["VecEnv", "SafetyGymnasiumVecEnv"]
