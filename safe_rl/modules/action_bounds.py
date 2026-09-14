"""Per-joint action bounds for bounded-output actors.

Every tanh-squashed actor in safe_rl (``StochasticActor`` for SAC, ``DeterministicActor``
for FastTD3, the REPPO ``StochasticActorCriticBase`` and the DIME diffusion actor) maps its
unit output onto the same affine box::

    a = b + c * tanh(u),   b = (high + low) / 2,   c = (high - low) / 2

with the change-of-variables term ``-sum_j log c_j`` added to every log-probability. With no
bounds configured ``b = 0`` and ``c = 1``, which is bit-identical to plain tanh squashing.

Why this exists: an unbounded Gaussian PPO actor commands |a| > 1 on a few percent of steps
(fast travel bursts on the excavator arm), and the environments' action terms scale and clip
those commands. A tanh actor capped at +-1 reaches only a fraction of that joint speed and
plateaus. Widening the box in the actor -- instead of the env config -- keeps the env
byte-identical for every algorithm, and because ``b``/``c`` are persistent buffers the
replay buffer, the critic, ``play.py`` and the ONNX/JIT exports all see the same action.

Config spellings (at most one may be given; ``action_scale`` is the legacy REPPO scalar):

* ``action_bounds``: scalar or list of ``num_actions`` symmetric half-widths, ``[-x_j, +x_j]``
* ``action_low`` / ``action_high``: explicit per-joint lists (asymmetric boxes)
* ``action_scale``: scalar half-width (legacy; ``1.0`` means unset)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

BOUND_KEYS: tuple[str, ...] = ("action_bounds", "action_low", "action_high", "action_scale")
_BUFFER_NAMES: tuple[str, ...] = ("action_b", "action_c", "neg_log_action_scale")


def _as_vector(value: Any, num_actions: int, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vec = value.detach().to(dtype=torch.float32).reshape(-1)
        if vec.numel() == 1:
            vec = vec.expand(num_actions).clone()
    elif isinstance(value, (int, float)):
        vec = torch.full((num_actions,), float(value), dtype=torch.float32)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vec = torch.tensor([float(v) for v in value], dtype=torch.float32)
    else:
        raise TypeError(f"{name} must be a number, a sequence of numbers or a tensor; got {type(value).__name__}")
    if vec.shape != (num_actions,):
        raise ValueError(f"{name} must have length num_actions={num_actions}, got {tuple(vec.shape)}.")
    if not torch.isfinite(vec).all():
        raise ValueError(f"{name} must be finite; got {vec.tolist()}.")
    return vec


def resolve_action_bounds(
    num_actions: int,
    *,
    action_low: Any = None,
    action_high: Any = None,
    action_bounds: Any = None,
    action_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return ``(b, c, scaled)`` for the configured box; ``scaled`` is False for the unit box."""
    spellings: list[str] = []
    if action_low is not None or action_high is not None:
        if action_low is None or action_high is None:
            raise ValueError("action_low and action_high must be given together.")
        spellings.append("action_low/action_high")
    if action_bounds is not None:
        spellings.append("action_bounds")
    if action_scale is not None and float(action_scale) != 1.0:
        spellings.append("action_scale")
    if len(spellings) > 1:
        raise ValueError(f"Give at most one of {spellings}; they all define the same action box.")

    if action_low is not None:
        low = _as_vector(action_low, num_actions, "action_low")
        high = _as_vector(action_high, num_actions, "action_high")
    elif action_bounds is not None:
        high = _as_vector(action_bounds, num_actions, "action_bounds")
        low = -high
    elif action_scale is not None and float(action_scale) != 1.0:
        if float(action_scale) <= 0.0:
            raise ValueError(f"action_scale must be > 0; got {action_scale!r}")
        high = torch.full((num_actions,), float(action_scale), dtype=torch.float32)
        low = -high
    else:
        return torch.zeros(num_actions), torch.ones(num_actions), False

    if (high <= low).any():
        raise ValueError(f"action_high must exceed action_low on every joint; got low={low.tolist()} high={high.tolist()}.")
    b = 0.5 * (high + low)
    c = 0.5 * (high - low)
    scaled = bool((b != 0.0).any() or (c != 1.0).any())
    return b, c, scaled


def register_action_bound_buffers(module: nn.Module, b: torch.Tensor, c: torch.Tensor) -> None:
    """Attach the persistent ``action_b``/``action_c``/``neg_log_action_scale`` buffers."""
    module.register_buffer("action_b", b.detach().clone().float())
    module.register_buffer("action_c", c.detach().clone().float())
    module.register_buffer("neg_log_action_scale", -c.detach().clamp_min(1e-8).log().sum())
    module.scaled_actions = bool((b != 0.0).any() or (c != 1.0).any())


def fill_missing_action_bound_buffers(module: nn.Module, state_dict: dict, prefix: str) -> None:
    """``_load_from_state_dict`` helper: legacy checkpoints (no bound buffers) keep the
    constructor's values; a checkpoint that carries a different box wins, loudly."""
    for name in _BUFFER_NAMES:
        key = prefix + name
        current = getattr(module, name)
        if key not in state_dict:
            state_dict[key] = current.detach().clone()
        else:
            stored = state_dict[key].to(device=current.device, dtype=current.dtype)
            if stored.shape == current.shape and not torch.allclose(stored, current, atol=1e-6):
                print(
                    f"[action_bounds] {key}: checkpoint stores {stored.flatten().tolist()} but the config "
                    f"gives {current.flatten().tolist()}; the checkpoint's action box is used."
                )


def sync_scaled_flag(module: nn.Module) -> None:
    """Recompute ``scaled_actions`` from the (possibly just loaded) buffers."""
    module.scaled_actions = bool((module.action_b != 0.0).any() or (module.action_c != 1.0).any())


def split_action_bound_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop the bound spellings out of a kwargs dict and return them."""
    return {key: kwargs.pop(key) for key in BOUND_KEYS if key in kwargs}


def squash_unit(module: nn.Module, u: torch.Tensor) -> torch.Tensor:
    """``b + c * tanh(u)`` using the module's buffers (exactly ``tanh(u)`` when unbounded)."""
    if not getattr(module, "scaled_actions", False):
        return torch.tanh(u)
    return module.action_b + module.action_c * torch.tanh(u)


def uniform_in_box(module: nn.Module, num_envs: int, device: torch.device | str) -> torch.Tensor:
    """Uniform samples over the action box (warm-up exploration)."""
    unit = torch.rand(num_envs, module.action_b.numel(), device=device) * 2 - 1
    if not getattr(module, "scaled_actions", False):
        return unit
    return module.action_b.to(device) + module.action_c.to(device) * unit


def uniform_scale_or_none(module: nn.Module) -> float | None:
    """The scalar half-width when the box is symmetric and identical on every joint, else None."""
    b, c = module.action_b, module.action_c
    if bool((b == 0.0).all()) and bool((c == c[0]).all()):
        return float(c[0])
    return None


def format_box(module: nn.Module) -> str:
    low = (module.action_b - module.action_c).tolist()
    high = (module.action_b + module.action_c).tolist()
    return "[" + ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in zip(low, high)) + "]"
