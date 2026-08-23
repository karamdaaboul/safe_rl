"""dt schedules for the DIME diffusion SDE.

Vendored verbatim from the TruDi reference
(`trudi/src/networks/reppo_dime/common/torch_schedulers.py`). Each factory
returns a callable `step -> scalar multiplier` applied to the softplus dt.
"""

from __future__ import annotations

import math

import torch


def get_linear_schedule(total_steps, min=0.01):
    def linear_noise_schedule(step):
        t = (total_steps - step) / total_steps
        return (1.0 - t) * min + t

    return linear_noise_schedule


def get_cosine_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        t = (total_steps - step) / total_steps
        offset = 1 + s
        return (1.0 - min) * torch.cos(0.5 * math.pi * (offset - t) / offset) ** pow + min

    return cosine_schedule


def get_constant_schedule():
    def constant_schedule(step):
        return torch.tensor(1.0)

    return constant_schedule
