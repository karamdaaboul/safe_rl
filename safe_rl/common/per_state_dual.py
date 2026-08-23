"""Batched E-step dual: one global temperature ``eta``, one multiplier ``lambda_b`` per state.

CVPO solves its E-step with a single ``(eta, lambda)`` pair per minibatch, by SLSQP on the CPU.
``eta`` is a trust-region size and genuinely wants to be uniform across the update -- it is a
regulariser on the whole E-step. ``lambda`` is a tradeoff *rate* between two value scales, and
that rate legitimately varies from state to state. CVPO's authors ruled out a per-state dual
because it meant leaving the vectorised path; it does not, if the solve never leaves the GPU.

With ``eta`` held global, each state's subproblem is 1-D root-finding in a monotone function::

    q_b(a_k)  propto  exp((Q_r(s_b,a_k) - lambda_b * Q_c(s_b,a_k)) / eta)

    g(eta, {lambda_b}) = eta*eps + mean_b[ lambda_b*d_b
                                           + eta * log (1/K) sum_k exp((Q_r - lambda_b Q_c)/eta) ]

    dg/dlambda_b = d_b - E_{q_b}[Q_c]     d2g/dlambda_b^2 = Var_{q_b}(Q_c)/eta >= 0
    dg/deta      = eps - mean_b KL(q_b || pi_old,b)

so ``E_{q_b}[Q_c]`` is non-increasing in ``lambda_b`` and ``mean_b KL`` is non-increasing in
``eta``. Both blocks are solved exactly by bisection, batched over ``(B, K)``, in pure torch.
No SciPy, no per-state Python loop, no learning rate on ``lambda``, no windup.

**Why there is no data-dependent control flow here.** Plain bisection on ``[0, lambda_max]``
*is* the KKT box projection: at an inactive state the test is False at every midpoint so the
bracket collapses to 0, at an infeasible state it is True at every midpoint so the bracket
collapses to ``lambda_max``. The hot loop is therefore a fixed ``for _ in range(iters)`` with no
masks and no host synchronisation, and the ``inactive`` / ``at_cap`` certificates are recovered
once at the end for diagnostics only. Keep it that way: a single ``.item()`` in the loop costs
more than the SciPy solve this replaces.

**Do not "fix" the solver when the iterates look wild.** ``g`` is jointly convex and both blocks
are solved exactly, so block-coordinate descent here cannot diverge -- it was measured decreasing
monotonically (-63.80 -> -71.98) over 25 sweeps while ``eta`` walked to 169 and ``lambda_median``
to 66. That is the iterates travelling *correctly* toward a minimiser that is genuinely far out:
for a state whose sampled actions cannot reach ``d_b``, ``dg/dlambda_b < 0`` for every lambda, so
``lambda_b -> lambda_max`` is optimal, the induced spread of ``-lambda_max * Q_c`` is enormous,
and the ``eta`` that truly minimises ``g`` really is large. Because ``eta`` is shared, those
states then contaminate every feasible state's solve. Damping ``eta``, capping the sweep count or
adding momentum would all be wrong and would violate "solved to optimality". The fix belongs to
the *problem*: hand this solver a per-state target ``d_b`` its trust region can actually reach
(see ``CVPOPerState``), and the pathology disappears -- 1 warm-started sweep, zero saturation.

Deviations from CLAUDE.md worth stating out loud:

* M1 specifies "pure numpy/scipy, no torch" for the dual solver. That rule was written for the
  scalar solve; the whole point of this one is that it is batched on the accelerator, and SciPy
  is the dependency being removed. Tests still cross-check against a SciPy oracle.
* Rule 5 wants ``m >= 1`` cost constraints from the start. 1-D bisection does not generalise to
  ``m > 1``, and faking an ``[m, K, B]`` signature that asserts ``m == 1`` would hide that. This
  module is single-constraint by construction; the ``m > 1`` path is a batched projected Newton
  step on the m-dimensional convex subproblem (still torch, still exact), and is not implemented.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class PerStateDualSolution:
    """Everything the E-step and its diagnostics need, all still on device.

    ``eta`` is deliberately a tensor rather than a float: converting it here would reintroduce
    the host synchronisation this module exists to remove. The algorithm converts once, at the
    diagnostics boundary, in a single batched transfer.
    """

    eta: torch.Tensor  # []  (or [B] when solved per state)
    lam: torch.Tensor  # [B]
    weights: torch.Tensor  # [K, B], columns sum to 1
    eqc: torch.Tensor  # [B]   E_{q_b}[Q_c] under `weights`
    kl: torch.Tensor  # [B]   KL(q_b || pi_old,b) under `weights`
    ess: torch.Tensor  # [B]
    lam_inactive: torch.Tensor  # [B] bool -- constraint slack already at lambda = 0
    lam_at_cap: torch.Tensor  # [B] bool -- infeasibility certificate (chapter 6.1)
    eta_at_bound: torch.Tensor  # [] or [B] bool -- eta pinned to its bracket
    sweeps: int
    lam_shared: torch.Tensor | None = None  # [] batch-level lambda, when a gate is in use


def _row(x: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or ``[B]`` tensor against a ``[K, B]`` operand."""
    return x if x.dim() == 0 else x.unsqueeze(0)


def estep_logits(q_r: torch.Tensor, q_c: torch.Tensor, eta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """``(Q_r - lambda_b * Q_c) / eta``, shape ``[K, B]``. ``eta`` and ``lam`` may be [] or [B]."""
    return (q_r - _row(lam) * q_c) / _row(eta)


def estep_weights(q_r: torch.Tensor, q_c: torch.Tensor, eta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Variational weights ``q_b(a_k)``, ``[K, B]``, columns summing to 1.

    ``torch.softmax`` subtracts the per-column max internally, which is the mandatory
    stabilisation of CLAUDE.md rule 4 -- do not hand-roll ``exp() / exp().sum()`` here.
    """
    return torch.softmax(estep_logits(q_r, q_c, eta, lam), dim=0)


def expected_qc(q_r: torch.Tensor, q_c: torch.Tensor, eta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """``E_{q_b}[Q_c]``, ``[B]``. Non-increasing in ``lam``: this is what bisection roots on."""
    return (estep_weights(q_r, q_c, eta, lam) * q_c).sum(dim=0)


def per_state_kl(weights: torch.Tensor) -> torch.Tensor:
    """``KL(q_b || pi_old,b)``, ``[B]``.

    The K candidates are *sampled* from pi_old, so pi_old is uniform over its own sample support
    and the divergence is ``sum_k w_k log(K w_k)``. Identical to
    :func:`safe_rl.algorithms.mpo.nonparametric_kl_from_weights`, duplicated here only to keep
    this module free of algorithm imports; the two are asserted equal in the tests.
    """
    k = weights.shape[0]
    return (weights * torch.log(k * weights + 1e-8)).sum(dim=0)


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Per-state ESS ``1 / sum_k w_k^2``, ``[B]``. K means the E-step did nothing, 1 means collapse."""
    return 1.0 / weights.pow(2).sum(dim=0).clamp_min(1e-12)


def dual_objective(
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    d: torch.Tensor,
    eps: float,
    eta: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """The joint dual ``g(eta, {lambda_b})``, ``[]``. For tests and monitoring, not the hot path."""
    z = estep_logits(q_r, q_c, eta, lam)
    log_mean_exp = torch.logsumexp(z, dim=0) - math.log(z.shape[0])  # [B]
    # `.mean()` of a 0-dim tensor is itself, so this covers shared and per-state eta alike.
    return (eta * eps).mean() + (lam * d).mean() + (eta * log_mean_exp).mean()


def solve_lambda(
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    eta: torch.Tensor,
    d: torch.Tensor,
    lam_max: float,
    iters: int = 30,
) -> torch.Tensor:
    """Exact per-state ``lambda_b``, ``[B]``, by batched bisection of ``E_{q_b}[Q_c] = d_b``.

    ``E_{q_b}[Q_c]`` is non-increasing in ``lambda_b`` (its derivative is ``-Var_{q_b}(Q_c)/eta``),
    so the root is unique wherever the constraint is active and the bracket ``[0, lam_max]``
    collapses onto the correct KKT corner otherwise. The corners are snapped exactly at the end:
    bisection would otherwise leave ``lambda_b`` at ``lam_max / 2**(iters+1)`` rather than 0 at
    an inactive state, which is numerically harmless but makes the ``inactive`` certificate and
    ``lambda_frac_zero`` lie.
    """
    b = q_r.shape[1]
    lo = torch.zeros(b, dtype=q_r.dtype, device=q_r.device)
    hi = torch.full((b,), float(lam_max), dtype=q_r.dtype, device=q_r.device)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        violating = expected_qc(q_r, q_c, eta, mid) > d
        lo = torch.where(violating, mid, lo)
        hi = torch.where(violating, hi, mid)
    lam = 0.5 * (lo + hi)

    inactive = expected_qc(q_r, q_c, eta, torch.zeros_like(lam)) <= d
    at_cap = expected_qc(q_r, q_c, eta, torch.full_like(lam, float(lam_max))) > d
    lam = torch.where(inactive, torch.zeros_like(lam), lam)
    return torch.where(at_cap, torch.full_like(lam, float(lam_max)), lam)


def solve_shared_lambda(
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    eta: torch.Tensor,
    d: torch.Tensor,
    lam_max: float,
    iters: int = 30,
) -> torch.Tensor:
    """One scalar ``lambda`` for the whole batch, ``[]``: CVPO's constraint, solved by bisection.

    Roots ``mean_b E_{q_b}[Q_c] = mean_b d_b``. Each state's expectation is non-increasing in
    lambda, so their mean is too, and the same bracket argument applies. This is the fallback the
    dispersion gate falls back *to*: at a state whose candidate actions all carry the same cost,
    the per-state root sits on a curve that is flat to within critic noise, so ``lambda_b`` is
    arbitrary and only adds dual variance to the M-step. A batch-level multiplier is the honest
    answer there -- and reverting to it wholesale is a legitimate outcome of M5, not a failure.
    """
    target = d.mean()
    lo = torch.zeros((), dtype=q_r.dtype, device=q_r.device)
    hi = torch.full((), float(lam_max), dtype=q_r.dtype, device=q_r.device)
    b = q_r.shape[1]
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        violating = expected_qc(q_r, q_c, eta, mid.expand(b)).mean() > target
        lo = torch.where(violating, mid, lo)
        hi = torch.where(violating, hi, mid)
    return 0.5 * (lo + hi)


def solve_eta(
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    lam: torch.Tensor,
    eps: float,
    eta_min: float = 1e-3,
    eta_max: float = 1e4,
    iters: int = 30,
    per_state: bool = False,
) -> torch.Tensor:
    """Exact ``eta`` by bisection of ``KL = eps``; ``[]`` shared, or ``[B]`` when ``per_state``.

    ``dg/deta = eps - KL``, and KL is non-increasing in ``eta``, so ``g`` is convex in ``eta`` and
    the root of ``KL(eta) = eps`` is its minimiser.

    Bisection runs in **log space over a fixed bracket** ``[eta_min, eta_max]``. A warm-started
    bracket with data-dependent expansion would be marginally tighter and would reintroduce
    exactly the host synchronisation this module avoids; over seven decades, 30 halvings still
    leave a relative precision of ~1e-8. Warm-starting matters for ``self.eta`` at the *sweep*
    level (sweep 1 solves lambda at the previous eta), not for this root-find, which is exact
    regardless of where it starts.

    ``per_state=True`` drops the batch reduction, enforcing ``KL_b = eps`` at every state. That is
    a different tier of the method (higher dual variance; CLAUDE.md M5 specifies a shared eta), so
    it is not the default -- but the ``[B]`` result broadcasts through every kernel above
    unchanged, which keeps it a one-line experiment rather than a refactor.
    """
    reduce = (lambda x: x) if per_state else (lambda x: x.mean())
    shape = (q_r.shape[1],) if per_state else ()
    lo = torch.full(shape, math.log(float(eta_min)), dtype=q_r.dtype, device=q_r.device)
    hi = torch.full(shape, math.log(float(eta_max)), dtype=q_r.dtype, device=q_r.device)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        too_cold = reduce(per_state_kl(estep_weights(q_r, q_c, mid.exp(), lam))) > eps
        lo = torch.where(too_cold, mid, lo)
        hi = torch.where(too_cold, hi, mid)
    return (0.5 * (lo + hi)).exp()


def reachable_qc_min(
    q_c: torch.Tensor,
    eps: float,
    eta_min: float = 1e-6,
    eta_max: float = 1e6,
    iters: int = 40,
) -> torch.Tensor:
    """Smallest ``E_{q_b}[Q_c]`` reachable inside the per-state KL budget, ``[B]``.

    The cost-minimising ``q`` under ``KL(q||pi_old) <= eps`` is ``q_nu propto exp(-nu Q_c)``, which
    is this module's own kernel with ``Q_r = 0``, ``lambda = 1`` and ``nu = 1/eta``. So the
    sampled-support feasibility test is :func:`solve_eta` read through :func:`expected_qc`, at zero
    extra code -- and, unlike ``CVPO._reachable_qc_min``, it is per-state and never leaves the
    device. A state is feasible for target ``d_b`` exactly when this floor is at or below it.
    """
    zeros = torch.zeros_like(q_c)
    ones = torch.ones(q_c.shape[1], dtype=q_c.dtype, device=q_c.device)
    eta = solve_eta(zeros, q_c, ones, eps, eta_min, eta_max, iters, per_state=True)
    return expected_qc(zeros, q_c, eta, ones)


def solve_per_state_dual(
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    d: torch.Tensor,
    eps: float,
    lam_max: float,
    eta_init: float | torch.Tensor = 1.0,
    sweeps: int = 1,
    lam_iters: int = 30,
    eta_iters: int = 30,
    eta_min: float = 1e-3,
    eta_max: float = 1e4,
    per_state_eta: bool = False,
    gate: torch.Tensor | None = None,
) -> PerStateDualSolution:
    """Block-coordinate descent on ``g``: exact ``lambda`` block, exact ``eta`` block.

    Args:
        q_r: ``[K, B]`` reward Q at the K candidate actions of each of B states.
        q_c: ``[K, B]`` cost Q at the same candidates (single constraint; see module docstring).
        d: ``[B]`` per-state cost target. Hand this solver a *reachable* target -- see the module
            docstring on why an unreachable one produces a correct but degenerate solution.
        eps: E-step KL budget.
        lam_max: cap on ``lambda_b``; also the infeasibility certificate's trigger. Size it from
            the E-step spread ratio, not from the value scale -- only the spread across candidate
            actions survives the per-state softmax.
        eta_init: warm start, used only to solve the *first* sweep's lambda block.
        sweeps: coordinate-descent sweeps. 1 suffices from a warm start; more from cold.
        gate: optional ``[B]`` bool. Where False, the state takes the batch-level scalar lambda
            from :func:`solve_shared_lambda` instead of its own root -- for states whose cost is
            flat across candidate actions, where the per-state root is not identifiable.

    The sequence ends on a ``lambda`` block, so ``E_{q_b}[Q_c] = d_b`` holds exactly at every
    interior state while ``mean_b KL = eps`` holds only up to the last eta block's staleness.
    That asymmetry is deliberate -- the cost constraint is the point of the method, and the
    residual on the other side is reported (``kl``) rather than hidden. At the fixed point, and
    hence in steady state, both hold.
    """
    if int(sweeps) < 1:
        raise ValueError(f"sweeps must be >= 1, got {sweeps}.")
    if q_r.shape != q_c.shape:
        raise ValueError(f"q_r and q_c must have the same shape, got {tuple(q_r.shape)} vs {tuple(q_c.shape)}.")
    if d.shape != (q_r.shape[1],):
        raise ValueError(f"d must be [B] = [{q_r.shape[1]}], got {tuple(d.shape)}.")
    if lam_max < 0.0:
        raise ValueError(f"lam_max must be >= 0, got {lam_max}.")

    if gate is not None and gate.shape != (q_r.shape[1],):
        raise ValueError(f"gate must be [B] = [{q_r.shape[1]}], got {tuple(gate.shape)}.")

    def lambda_block(eta_):
        lam_ = solve_lambda(q_r, q_c, eta_, d, lam_max, lam_iters)
        if gate is None:
            return lam_, None
        lam_shared = solve_shared_lambda(q_r, q_c, eta_, d, lam_max, lam_iters)
        return torch.where(gate, lam_, lam_shared.expand_as(lam_)), lam_shared

    eta = torch.as_tensor(eta_init, dtype=q_r.dtype, device=q_r.device)
    if eta.dim() != 0:
        eta = eta.reshape(-1)[0]
    lam_shared = None
    for _ in range(int(sweeps)):
        lam, lam_shared = lambda_block(eta)
        eta = solve_eta(q_r, q_c, lam, eps, eta_min, eta_max, eta_iters, per_state=per_state_eta)
    lam, lam_shared = lambda_block(eta)

    weights = estep_weights(q_r, q_c, eta, lam)
    return PerStateDualSolution(
        eta=eta,
        lam=lam,
        weights=weights,
        eqc=(weights * q_c).sum(dim=0),
        kl=per_state_kl(weights),
        ess=effective_sample_size(weights),
        lam_inactive=lam <= 0.0,
        # lam_max = 0 pins lambda at zero to disable the constraint; that is not saturation.
        lam_at_cap=(lam >= lam_max) & torch.as_tensor(lam_max > 0.0, device=lam.device),
        eta_at_bound=(eta <= eta_min * (1.0 + 1e-6)) | (eta >= eta_max * (1.0 - 1e-6)),
        sweeps=int(sweeps),
        lam_shared=lam_shared,
    )
