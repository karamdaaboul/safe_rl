from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .cpo import CPO


class PCPO(CPO):
    """Projection-based CPO (Liu et al. 2021).

    Replaces CPO's 4-case Lagrangian analysis with a closed-form projection:
    take the full natural-gradient reward step, then subtract the minimum
    correction along the cost gradient direction needed to stay feasible.
    """

    def _compute_step_direction(
        self,
        xHx: torch.Tensor,
        x: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        c_hat: torch.Tensor,
        b_grads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        q_safe = torch.clamp_min(q, 1e-8)
        s_safe = torch.clamp_min(s, 1e-8)
        scale = torch.sqrt(2.0 * self.target_kl / q_safe)
        correction = torch.clamp_min((scale * r + c_hat) / s_safe, 0.0)
        step_direction = scale * x - correction * p
        one = torch.ones((), device=self.device)
        return step_direction, one, one, 1
