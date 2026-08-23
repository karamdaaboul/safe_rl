"""Continuation probabilities for stochastic decision horizons (SDH).

Implements the modelling device of *Stochastic Decision Horizons for Constrained Reinforcement
Learning* (Milosevic et al., arXiv:2602.04599): every state-action pair carries a continuation
probability ``alpha(s, a) in [0, 1]``, which shapes the reward and the discount

    r~(s, a) = alpha(s, a) r(s, a),        gamma~(s, a) = gamma * alpha(s, a)

so that constraint-violating choices shorten the effective planning horizon instead of being
budgeted by a Lagrange multiplier. The variable-discount Bellman operator stays a contraction with
modulus ``sup alpha * gamma <= gamma``, so off-policy replay and target networks are unaffected.

Why this is worth trying HERE specifically
------------------------------------------
Measured on this repo's FH cost critic: the across-action spread of ``Q_c`` at a fixed state is
0.03-0.16 against a level of ~16, i.e. under 1%, while ``std_a(Q_r)`` is ~0.13. The E-step softmax
``exp((Q_r - lambda Q_c)/eta)`` is normalised per state, so only that spread reaches the policy --
which is why a scalar ``lambda`` could never steer the constraint and why the per-state homotopy
stalled with its KKT residual at 1e-5 (nothing was broken; there was nothing to steer with).

Under SDH the cost stops being a small additive term and becomes a *multiplicative* attenuation of
the whole future return: a 1% difference in ``alpha`` scales all of ``Q_surv`` (~20) rather than
adding ~0.1, and it compounds along the horizon through ``u_t = gamma^t prod alpha``. That is the
mechanism this module exists to test.

What it is NOT: SDH is not a CMDP. There is no budget and no feasibility guarantee -- the scale
``lam`` trades return against violations, and the operating point has to be found by sweeping it.
"""

from __future__ import annotations

import torch


def exponential_continuation(costs: torch.Tensor, lam: float) -> torch.Tensor:
    """``alpha = exp(-lam * sum_i c_i)`` -- the paper's Safety-Gymnasium mapping.

    Args:
        costs: ``[..., C]`` per-step violation magnitudes ``c_i(s, a) >= 0``, one column per
            constraint. Summed over the constraint axis.
        lam: continuation scale ``>= 0``. ``lam = 0`` gives ``alpha == 1`` exactly, i.e. the
            unshaped MDP -- which is what makes it a usable "off" switch and a regression anchor.

    Returns:
        ``[...]`` continuation probabilities in ``(0, 1]``.

    Smooth and monotone in the violation magnitude, which the paper found more robust than
    indicator-style truncation. The clamp is defensive only: costs are non-negative by
    construction here, so ``alpha <= 1`` already holds, but a sign error upstream would otherwise
    silently produce a discount ABOVE gamma and break the contraction.
    """
    if lam < 0.0:
        raise ValueError(f"continuation scale must be >= 0, got {lam}")
    total = costs.sum(dim=-1) if costs.dim() > 1 else costs
    return torch.exp(-float(lam) * total).clamp(0.0, 1.0)


def cat_continuation(costs: torch.Tensor, c_max: torch.Tensor | float, p_max: float = 1.0) -> torch.Tensor:
    """CaT-style normalised, saturating continuation, aggregated over constraints by ``min``.

    ``alpha_i = 1 - p_max * clip(c_i / max(c_max, eps), 0, 1)``, then ``alpha = min_i alpha_i``.

    The paper uses this where violation magnitudes are heterogeneous and spiky (its musculoskeletal
    setting), normalising by a running scale so ``alpha`` is insensitive to raw cost units. Provided
    for the multi-constraint case; the Safety-Gymnasium arms use
    :func:`exponential_continuation`.
    """
    if not 0.0 <= p_max <= 1.0:
        raise ValueError(f"p_max must be in [0, 1], got {p_max}")
    scale = torch.as_tensor(c_max, dtype=costs.dtype, device=costs.device).clamp_min(1e-8)
    per_constraint = 1.0 - p_max * (costs / scale).clamp(0.0, 1.0)
    return per_constraint.min(dim=-1).values if costs.dim() > 1 else per_constraint


def continuation_scale_at(update: int, lam_final: float, warmup: int, ramp: int, lam_init: float = 0.0) -> float:
    """Linear schedule for the continuation scale, held at ``lam_init`` for ``warmup`` updates.

    The paper ramps ``lam`` linearly over training for every Safety-Gymnasium environment rather
    than fixing it, because a large ``lam`` from step 0 attenuates the return before the critic has
    learned anything to attenuate.

    Deliberately mirrors the signature and semantics of
    :func:`safe_rl.common.fh_cost.kappa_at`, so the two schedules in this codebase read the same
    way and can be reasoned about together.
    """
    if lam_final < 0.0 or lam_init < 0.0:
        raise ValueError(f"continuation scales must be >= 0, got init={lam_init}, final={lam_final}")
    if warmup < 0 or ramp < 0:
        raise ValueError(f"warmup and ramp must be >= 0, got {warmup}, {ramp}")
    if update < warmup:
        return float(lam_init)
    if ramp == 0 or update >= warmup + ramp:
        return float(lam_final)
    frac = (update - warmup) / float(ramp)
    return float(lam_init + frac * (lam_final - lam_init))
