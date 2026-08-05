"""Converting an episodic cost budget into the units a discounted cost critic predicts.

A cost limit is stated episodically ("<= 25 cost per 1000-step episode") but the critic
predicts a discounted return ``G_c(s_0) = sum_t gamma^t c_t``. The constraint therefore needs
a conversion factor ``qc_scale = G_c(s_0) / J_c``.

The analytic factor ``(1 - gamma^L)/(1 - gamma)/L`` assumes cost is spread uniformly in time.
On SafetyPointGoal1 the *total* cost mass really is near-uniform, but ``gamma^t`` weights the
first ``1/(1-gamma)`` steps most and those carry slightly less cost than average, so the
analytic factor over-reads: measured 0.0764 against an analytic 0.1. A threshold built on the
analytic value silently grants a **32.7** episodic budget when 25 was requested — which is
exactly what confounded the first CVaR-vs-mean comparison.

`make_thresholds` therefore always reports the *effective episodic budget* it implies, so the
mistake is visible in a config dump rather than buried three multiplications deep.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def analytic_qc_scale(gamma: float, horizon: int) -> float:
    """Uniform-cost conversion factor ``(1 - gamma^L)/(1 - gamma)/L``."""
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return float((1.0 - gamma**horizon) / (1.0 - gamma) / horizon)


def measured_qc_scale(
    episode_costs: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray],
    gamma: float,
) -> float:
    """Estimate ``G_c(s_0) / J_c`` empirically from rollout cost sequences.

    Args:
        episode_costs: per-episode 1-D cost sequences, or a ``[T, num_envs]`` array whose
            columns are treated as episodes.
        gamma: discount used by the cost critic.

    Returns the ratio of the *mean discounted return* to the *mean undiscounted cost*, not
    the mean of per-episode ratios: roughly 10% of real episodes incur zero cost, and a
    per-episode ratio would divide by zero on those.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")

    if isinstance(episode_costs, np.ndarray) and episode_costs.ndim == 2:
        episodes = [episode_costs[:, i] for i in range(episode_costs.shape[1])]
    else:
        episodes = [np.asarray(e, dtype=np.float64).reshape(-1) for e in episode_costs]

    if not episodes:
        raise ValueError("episode_costs is empty; cannot estimate qc_scale")

    total_discounted = 0.0
    total_undiscounted = 0.0
    for costs in episodes:
        disc = gamma ** np.arange(costs.size, dtype=np.float64)
        total_discounted += float(np.sum(costs * disc))
        total_undiscounted += float(np.sum(costs))

    if total_undiscounted <= 0.0:
        raise ValueError("all episodes have zero cost; qc_scale is undefined")
    return total_discounted / total_undiscounted


def make_thresholds(
    cost_lim: float,
    qc_scale: float,
    mode: str = "matched",
    ratio_stats: Mapping[float, float] | None = None,
) -> dict:
    """Build the cost-Q thresholds for the mean constraint and each CVaR level.

    Args:
        cost_lim: episodic cost budget (e.g. 25).
        qc_scale: episodic -> discounted conversion (analytic or measured).
        mode:
            ``"wcsac"``   — every CVaR level gets the *same* threshold as the mean, i.e. the
                            budget is applied to the tail statistic. WCSAC semantics, and
                            deliberately stricter: the implied episodic budget is
                            ``cost_lim / ratio``.
            ``"matched"`` — each CVaR threshold is scaled by the measured ``CVaR_a / mean``
                            ratio, so "CVaR at threshold" means "mean at budget". Same nominal
                            budget, shaped by risk.
        ratio_stats: ``{alpha: CVaR_alpha / mean}`` measured on a reference policy. Required
            for ``"matched"``; ignored for ``"wcsac"``.

    Returns a dict of thresholds plus, for every entry, the *effective episodic budget* it
    implies — the quantity to assert against in config tests.
    """
    if mode not in ("wcsac", "matched"):
        raise ValueError(f"mode must be 'wcsac' or 'matched', got {mode!r}")
    if qc_scale <= 0.0:
        raise ValueError(f"qc_scale must be positive, got {qc_scale}")
    if cost_lim <= 0.0:
        raise ValueError(f"cost_lim must be positive, got {cost_lim}")
    if mode == "matched" and not ratio_stats:
        raise ValueError("mode='matched' requires ratio_stats {alpha: CVaR_alpha/mean}")

    base = float(cost_lim) * float(qc_scale)
    out: dict = {
        "mode": mode,
        "cost_lim": float(cost_lim),
        "qc_scale": float(qc_scale),
        "mean": base,
        "effective_budget_mean": base / float(qc_scale),
    }

    for alpha, ratio in (ratio_stats or {}).items():
        if ratio <= 0.0:
            raise ValueError(f"ratio for alpha={alpha} must be positive, got {ratio}")
        thres = base if mode == "wcsac" else base * float(ratio)
        out[f"cvar_{alpha}"] = thres
        # What episodic mean cost a policy sitting exactly on this threshold would incur.
        out[f"effective_budget_cvar_{alpha}"] = thres / (float(qc_scale) * float(ratio))
    return out
