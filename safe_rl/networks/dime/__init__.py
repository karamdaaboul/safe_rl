"""DIME diffusion-policy stack, vendored from the TruDi reference.

Source: `trudi/src/networks/reppo_dime/` (branch `cleanup_reppo_dime`), torch
path. The reference wires the dt schedule and integrators via
`hydra.utils.instantiate` / `get_method`; here they are plain imports plus the
`DT_SCHEDULES` registry, keyed by the config's `dt_schedule.type`.
"""

from safe_rl.networks.dime.control_net import ControlNetwork
from safe_rl.networks.dime.integrators import (
    logratio,
    ode_integrator,
    sde_integrator,
    sde_integrator_with_kl,
)
from safe_rl.networks.dime.models import DiffusionModel, DIMEActor
from safe_rl.networks.dime.schedulers import (
    get_constant_schedule,
    get_cosine_schedule,
    get_linear_schedule,
)

DT_SCHEDULES = {
    "cosine": get_cosine_schedule,
    "linear": get_linear_schedule,
    "constant": get_constant_schedule,
}

__all__ = [
    "ControlNetwork",
    "DiffusionModel",
    "DIMEActor",
    "DT_SCHEDULES",
    "get_constant_schedule",
    "get_cosine_schedule",
    "get_linear_schedule",
    "logratio",
    "ode_integrator",
    "sde_integrator",
    "sde_integrator_with_kl",
]
