"""Numerics helpers for the DIME diffusion policy.

Vendored from the TruDi reference (`trudi/src/networks/torch_utils.py`); only the
four functions the diffusion stack uses are kept — the source file also carries
module-level `wandb`/`matplotlib` plotting utilities that do not belong here.
"""

from __future__ import annotations

import torch


def inverse_softplus(x):
    """Numerically stable implementation of inverse softplus"""
    # Threshold above which the approximation log(e^x - 1) ≈ x is used
    threshold = 20.0
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def check_stop_grad(expression, stop_grad):
    """Stop gradients conditionally"""
    return expression.detach() if stop_grad else expression


def sample_kernel(mean, scale, device=None):
    """Sample from a normal distribution"""
    device = device or mean.device
    eps = torch.randn_like(mean, device=device)
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    """Compute log probability under normal distribution"""
    dist = torch.distributions.Independent(
        torch.distributions.Normal(loc=mean, scale=scale), 1
    )
    return dist.log_prob(x)
