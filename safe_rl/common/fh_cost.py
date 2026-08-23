"""Finite-horizon distributional cost machinery for FH-DCMPO.

Three separable pieces, all framework-light and unit-testable without a training loop:

1. **Horizon features** -- the observation columns that make an *undiscounted* finite-horizon
   value function well-posed. With ``gamma_c = 1`` the cost-to-go
   ``sum_{t'=t}^{T-1} c_t'`` depends on how much episode is left, so ``(s, a)`` alone does not
   determine it and the regression target is non-stationary. Appending the normalized remaining
   horizon ``u_t = (T - t) / T`` restores the Markov property, exactly as Sauté RL
   (Sootla et al., ICML 2022) does for the remaining safety budget.

2. **Risk statistics on a quantile representation** -- ``VaR``, ``CVaR`` and the *conservatism
   statistic* ``rho_kappa,alpha = E[Z] + kappa (CVaR_alpha[Z] - E[Z])``. ``kappa = 0`` is the
   mean constraint *exactly*; ``kappa = 1`` is pure ``CVaR_alpha``. The interpolation is not
   cosmetic: ``codex/why-mean-beat-cvar-on-pointgoal1.md`` measured that jumping straight to
   ``CVaR_0.9`` puts the constraint 4.48x above its threshold on day one, so the multiplier pins
   at ``lambda_max`` for most of training and the E-step degenerates to "minimise cost, ignore
   reward". Ramping ``kappa`` keeps the constraint satisfiable while it tightens.

3. **An EVT/GPD tail alternative** -- the extreme-quantile constraint of EVO (Gao et al.,
   ICML 2025, arXiv:2601.12008), fitted to the pooled peaks over a safety boundary. Offered
   behind a flag as an alternative tail statistic, not as the default.

Nothing here reads or writes global state, and no function needs a critic instance -- which is
what lets ``tests/test_fh_cost.py`` check every statistic against brute-force Monte Carlo.
"""

from __future__ import annotations

import math
import numpy as np
import torch

__all__ = [
    "normalized_remaining_horizon",
    "normalized_remaining_budget",
    "horizon_feature_dim",
    "quantile_cvar",
    "quantile_cvar_weighted",
    "quantile_var",
    "pit_from_quantiles",
    "recalibrated_masses",
    "conservatism_statistic",
    "kappa_at",
    "fit_gpd",
    "gpd_excess_quantile",
    "evt_conservatism_statistic",
]


# --------------------------------------------------------------------------------------------
# 1. Horizon features
# --------------------------------------------------------------------------------------------


def horizon_feature_dim(include_budget: bool) -> int:
    """Number of appended observation columns: remaining horizon, optionally + remaining budget."""
    return 2 if include_budget else 1


def normalized_remaining_horizon(episode_length_buf: torch.Tensor, horizon: int) -> torch.Tensor:
    """``u_t = (T - t) / T`` in ``[0, 1]``, shape ``[num_envs, 1]``.

    ``u = 1`` at the first step of an episode and ``u = 0`` once ``t`` reaches ``T``. The value
    ``0`` matters: it is the boundary condition that pins the undiscounted cost-to-go to zero and
    gives the backup something to contract toward. Without it there is no contraction anywhere in
    the undiscounted recursion.

    ``episode_length_buf`` is the runner's per-env step counter, so this is exact rather than
    inferred from the observation.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    t = episode_length_buf.reshape(-1, 1).to(torch.float32)
    return ((horizon - t) / float(horizon)).clamp(0.0, 1.0)


def normalized_remaining_budget(cost_in_episode: torch.Tensor, cost_limit: float) -> torch.Tensor:
    """``b_t = (d - sum_{t' < t} c_t') / d`` clamped to ``[-1, 1]``, shape ``[num_envs, 1]``.

    Sauté RL's safety state. ``b = 1`` at episode start, ``0`` when the budget is exactly spent,
    and negative once it is blown -- the clamp at ``-1`` keeps the feature bounded without
    collapsing "just over" and "far over" onto the same value until 2x the budget.

    Optional, behind a config flag: it makes the *policy* budget-conditional, which is a real
    behavioural change on top of the finite-horizon critic and deserves its own ablation.
    """
    if cost_limit <= 0.0:
        raise ValueError(f"cost_limit must be positive, got {cost_limit}")
    spent = cost_in_episode.reshape(-1, 1).to(torch.float32)
    return ((cost_limit - spent) / float(cost_limit)).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------------------------
# 2. Risk statistics on N equal-mass quantile locations
# --------------------------------------------------------------------------------------------


def _sorted(theta: torch.Tensor) -> torch.Tensor:
    """Sort the locations ascending. Idempotent, differentiable, and **not** optional.

    ``QuantileCritic.forward`` already sorts, so in the training path this is a no-op. It is here
    anyway because reading a tail off unsorted locations does not fail -- it silently returns a
    number *below* the true tail mean, i.e. it understates cost, which is the unsafe direction and
    would look like a weak constraint rather than a bug. Sorting only permutes, so gradients are
    unaffected.
    """
    return theta.sort(dim=-1).values


def quantile_var(theta: torch.Tensor, alpha: float) -> torch.Tensor:
    """Value-at-Risk from equal-mass quantile locations. Shape ``theta[..., N] -> [...]``.

    Cumulative mass through location ``k`` is ``(k + 1) / N``, so the index is closed-form.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    n = theta.shape[-1]
    idx = min(max(math.ceil(alpha * n) - 1, 0), n - 1)
    return _sorted(theta)[..., idx]


def quantile_cvar(theta: torch.Tensor, alpha: float, upper: bool = True) -> torch.Tensor:
    """CVaR from sorted, equal-mass quantile locations. Shape ``theta[..., N] -> [...]``.

    ``upper=True`` (the cost convention) is ``E[Z | Z >= VaR_alpha]``, the mean of the worst
    ``1 - alpha`` fraction; ``upper=False`` takes the lower tail.

    Takes exactly ``tail`` of probability mass from the requested end, **splitting the boundary
    location** instead of rounding to whole atoms -- without the split, ``alpha = 0.9`` at
    ``N = 64`` would silently mean ``6/64 = 0.094`` or ``7/64 = 0.109``.

    The weights depend only on ``(alpha, N)`` and never on ``theta``, so the statistic is *linear*
    in the learned locations. That is what makes it safe inside the E-step exponent: the gradient
    reaches every location in the tail undistorted, and no location outside it.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if alpha == 0.0:
        return theta.mean(dim=-1)
    n = theta.shape[-1]
    tail = 1.0 - alpha if upper else alpha
    p = 1.0 / n
    k = torch.arange(n, device=theta.device, dtype=theta.dtype)
    # Mass lying strictly beyond location k, on the side the tail is taken from.
    beyond = (n - 1 - k) * p if upper else k * p
    w = torch.clamp(tail - beyond, min=0.0).clamp(max=p)
    return torch.sum(_sorted(theta) * w, dim=-1) / w.sum().clamp_min(1e-12)


def quantile_cvar_weighted(theta: torch.Tensor, probs: torch.Tensor, alpha: float, upper: bool = True) -> torch.Tensor:
    """CVaR over quantile locations carrying **non-uniform** mass ``probs``.

    The general form of :func:`quantile_cvar`, needed once PIT recalibration reweights the
    locations (S3): for a QR head the CDF values are fixed and the support is learned, so
    recalibrating means moving mass between locations rather than moving the locations.

    ``probs`` is ``[N]``, non-negative, summing to 1, aligned with ``theta``'s last axis. Same exact
    tail-mass accounting, with the boundary location split.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if probs.shape[-1] != theta.shape[-1]:
        raise ValueError(f"probs has {probs.shape[-1]} masses but theta has {theta.shape[-1]} locations")
    p = probs.to(theta.dtype).to(theta.device)
    order = theta.sort(dim=-1)
    th = order.values
    if alpha == 0.0:
        return torch.sum(th * p, dim=-1) / p.sum().clamp_min(1e-12)
    tail = 1.0 - alpha if upper else alpha
    total = p.sum()
    cum = torch.cumsum(p, dim=-1)
    # Mass lying strictly beyond location k on the side the tail is taken from.
    beyond = (total - cum) if upper else (cum - p)
    w = torch.clamp(tail - beyond, min=0.0).clamp(max=p)
    return torch.sum(th * w, dim=-1) / w.sum().clamp_min(1e-12)


def pit_from_quantiles(theta: torch.Tensor, realized: torch.Tensor) -> torch.Tensor:
    """Probability-integral transform of ``realized`` under the quantile prediction. Shape ``[...]``.

    ``PIT = F_hat(z)`` = the fraction of locations at or below the realized value. Uniform on
    ``[0, 1]`` exactly when the predicted distribution is calibrated, which is what
    :class:`~safe_rl.common.recalibration.PITRecalibrator` consumes.

    Judge these on the **marginal** (the pooled histogram over states), never per state:
    ``codex/qr-dmpo-math.md`` section 6.1 shows per-state coverage collapses toward 0.5 regardless of
    correctness when the conditional is near-deterministic, so a per-state test would reject a
    provably correct critic.
    """
    return (theta <= realized.unsqueeze(-1)).to(theta.dtype).mean(dim=-1)


def recalibrated_masses(cdf_map, n_quantiles: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Masses ``p_k`` for the ``N`` locations after recalibration. Shape ``[N]``, sums to 1.

    Kuleshov et al. (2018): the calibrated CDF is ``G_hat . F_hat`` for ``G_hat`` the fitted CDF of
    the PIT values. A quantile head has ``F_hat^{-1}(tau_k) = theta_k`` with nominal level
    boundaries ``c_k = (k+1)/N``, so the calibrated mass on location ``k`` is
    ``G_hat(c_k) - G_hat(c_{k-1})``. Telescopes to ``G_hat(1) - G_hat(0) = 1``.

    ``cdf_map`` is any callable applying ``G_hat`` elementwise to CDF values in ``[0, 1]``
    (``PITRecalibrator.apply``). A calibrated critic gives ``G_hat = identity`` and hence uniform
    ``1/N`` masses, i.e. this reduces exactly to :func:`quantile_cvar`.
    """
    if n_quantiles < 1:
        raise ValueError(f"n_quantiles must be >= 1, got {n_quantiles}")
    edges = np.arange(n_quantiles + 1, dtype=np.float64) / n_quantiles  # c_{-1}=0 .. c_{N-1}=1
    mapped = np.asarray(cdf_map(edges), dtype=np.float64).ravel()
    if mapped.shape[0] != n_quantiles + 1:
        raise ValueError(f"cdf_map returned {mapped.shape[0]} values for {n_quantiles + 1} edges")
    # Validate the map rather than repairing it. Blind repair is dangerous here: forcing the
    # endpoints onto a *decreasing* map yields all mass on the LOWEST location, i.e. CVaR = min,
    # which understates cost -- the unsafe direction, and invisible in the logs. A degenerate fit
    # must therefore fall back to the uniform masses (= no recalibration) rather than to an extreme.
    uniform = np.full(n_quantiles, 1.0 / n_quantiles)
    if not np.all(np.isfinite(mapped)):
        return torch.as_tensor(uniform, dtype=dtype, device=device)
    if np.any(np.diff(mapped) < -1e-9):
        return torch.as_tensor(uniform, dtype=dtype, device=device)  # not a CDF
    if mapped[-1] - mapped[0] < 1e-6:
        return torch.as_tensor(uniform, dtype=dtype, device=device)  # constant: carries no information

    mapped = np.clip(mapped, 0.0, 1.0)
    mapped[0], mapped[-1] = 0.0, 1.0
    masses = np.maximum(np.diff(mapped), 0.0)
    total = masses.sum()
    masses = uniform if (not np.isfinite(total) or total <= 0.0) else masses / total
    return torch.as_tensor(masses, dtype=dtype, device=device)


def conservatism_statistic(theta: torch.Tensor, alpha: float, kappa: float) -> torch.Tensor:
    """``rho = E[Z] + kappa * (CVaR_alpha[Z] - E[Z])``. Shape ``theta[..., N] -> [...]``.

    A convex combination of the mean and the tail mean, so ``rho`` is monotone in ``kappa`` (the
    tail mean is never below the mean) and:

    * ``kappa = 0`` reproduces the mean constraint **bit-exactly** -- the property that lets S1
      isolate the finite-horizon change from the risk change;
    * ``kappa = 1`` is pure ``CVaR_alpha``;
    * intermediate ``kappa`` is the homotopy path the schedule walks.

    Values above 1 are allowed (extrapolation past the tail mean) but are not the intended use.
    """
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    mean = theta.mean(dim=-1)
    if kappa == 0.0:
        return mean
    return mean + float(kappa) * (quantile_cvar(theta, alpha, upper=True) - mean)


def kappa_at(step: int, kappa_target: float, warmup_steps: int, ramp_steps: int) -> float:
    """Linear ``kappa`` ramp: ``0`` for ``warmup_steps``, then linear to ``kappa_target``.

    Held at ``0`` during warmup so the cost critic is levelled before any tail statistic is read
    off it -- reading a tail from an untrained, under-dispersed critic is precisely the failure
    ``codex/why-mean-beat-cvar-on-pointgoal1.md`` documents. ``ramp_steps <= 0`` means step change
    at the end of warmup.
    """
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return float(kappa_target)
    frac = min(1.0, (step - warmup_steps) / float(ramp_steps))
    return float(kappa_target) * frac


# --------------------------------------------------------------------------------------------
# 3. EVT / GPD tail (EVO, arXiv:2601.12008)
# --------------------------------------------------------------------------------------------


def fit_gpd(excesses: np.ndarray, max_iter: int = 200) -> tuple[float, float]:
    """MLE fit of a Generalized Pareto Distribution to peaks-over-threshold. Returns ``(xi, sigma)``.

    Maximises EVO Eq. 14,
    ``log L(xi, sigma) = -N log sigma - (1 + 1/xi) sum_i log(1 + xi/sigma * Y_i)``,
    by Grimshaw's reparameterisation: with ``b = xi / sigma`` the profile likelihood is a smooth
    1-D function of ``b`` alone, so this is a bracketed scalar solve rather than a 2-D search that
    can wander into the infeasible region ``1 + b*Y_i <= 0``.

    Falls back to the exponential limit ``xi -> 0`` (``sigma = mean``) when the sample is too
    small or degenerate, which is the conservative choice: it gives a lighter tail than a fitted
    positive ``xi``, so an EVT constraint never becomes *looser* because the fit failed silently.
    """
    y = np.asarray(excesses, dtype=np.float64).ravel()
    y = y[np.isfinite(y) & (y > 0.0)]
    if y.size < 10:
        return 0.0, float(max(y.mean(), 1e-8)) if y.size else 1e-8

    def neg_ll(b: float) -> float:
        z = 1.0 + b * y
        if np.any(z <= 1e-12):
            return np.inf
        xi = float(np.mean(np.log(z)))
        if abs(xi) < 1e-12:
            return np.inf
        sigma = xi / b
        if sigma <= 0.0:
            return np.inf
        return y.size * math.log(sigma) + (1.0 + 1.0 / xi) * float(np.sum(np.log(z)))

    # b must exceed -1/max(y) to keep every 1 + b*y_i positive.
    lo = -1.0 / y.max() + 1e-9
    hi = max(4.0 / y.mean(), lo + 1.0)
    grid = np.linspace(lo, hi, 256)
    vals = np.array([neg_ll(float(b)) for b in grid])
    if not np.any(np.isfinite(vals)):
        return 0.0, float(y.mean())
    b = float(grid[int(np.argmin(vals))])
    # Golden-section refinement around the grid minimum.
    span = (hi - lo) / 256.0
    a, c = b - span, b + span
    for _ in range(max_iter):
        if c - a < 1e-12:
            break
        m1, m2 = a + 0.382 * (c - a), a + 0.618 * (c - a)
        if neg_ll(m1) < neg_ll(m2):
            c = m2
        else:
            a = m1
    b = 0.5 * (a + c)
    z = 1.0 + b * y
    if np.any(z <= 1e-12):
        return 0.0, float(y.mean())
    xi = float(np.mean(np.log(z)))
    if abs(xi) < 1e-9 or xi / b <= 0.0:
        return 0.0, float(y.mean())
    return xi, float(xi / b)


def gpd_excess_quantile(xi: float, sigma: float, nu: float, n_peaks: int, n_total: int) -> float:
    """EVO Eq. 15: the risk-boundary offset ``(sigma/xi) * ((1 - nu*n/N_mu)^(-xi) - 1)``.

    ``nu`` is the exploitation range into the tail, ``n_peaks`` the number of samples above the
    safety boundary and ``n_total`` the sample count the boundary was computed from. Reduces to
    the exponential form ``-sigma * log(1 - nu*n/N_mu)`` as ``xi -> 0``, and returns ``0`` when
    the fit is degenerate or the requested range exceeds the available peak mass -- again the
    conservative direction is a *smaller* offset only when we genuinely cannot estimate it.
    """
    if n_peaks <= 0 or n_total <= 0 or sigma <= 0.0 or nu <= 0.0:
        return 0.0
    frac = nu * n_total / float(n_peaks)
    if not 0.0 < frac < 1.0:
        return 0.0
    if abs(xi) < 1e-9:
        return float(-sigma * math.log(1.0 - frac))
    return float((sigma / xi) * ((1.0 - frac) ** (-xi) - 1.0))


def evt_conservatism_statistic(
    theta: torch.Tensor,
    mu: float,
    nu: float,
    kappa: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """EVO-style statistic: per-state safety boundary + a **globally** fitted GPD tail offset.

    Following EVO Section 4.1, the safety boundary ``q_mu`` is the mean of the cost distribution
    and the excess beyond it is modelled by one GPD fitted to the pooled peaks across the batch.
    So the returned statistic is ``E[Z(s,a)] + kappa * z``, with ``z`` a single scalar offset --
    structurally different from :func:`conservatism_statistic`, whose tail is per-state.

    Returns the statistic and the fit diagnostics (``xi``, ``sigma``, ``n_peaks``, ``offset``),
    because a silently failed GPD fit is indistinguishable from a mean constraint at the logs.
    """
    mean = theta.mean(dim=-1)
    flat = theta.detach().reshape(-1).cpu().numpy().astype(np.float64)
    boundary = float(np.quantile(flat, mu)) if flat.size else 0.0
    peaks = flat[flat > boundary] - boundary
    xi, sigma = fit_gpd(peaks)
    offset = gpd_excess_quantile(xi, sigma, nu, int(peaks.size), int(flat.size))
    info = {
        "evt_xi": xi,
        "evt_sigma": sigma,
        "evt_n_peaks": float(peaks.size),
        "evt_offset": offset,
        "evt_boundary": boundary,
    }
    return mean + float(kappa) * offset, info
