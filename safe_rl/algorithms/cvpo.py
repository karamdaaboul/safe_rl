from __future__ import annotations

import numpy as np
import torch
from copy import deepcopy
from typing import Any

from scipy.optimize import minimize

from safe_rl.common.cost_scaling import make_thresholds, measured_qc_scale
from safe_rl.common.lambda_controller import LambdaController, rescale_advantage
from safe_rl.common.recalibration import PITRecalibrator
from safe_rl.algorithms.mpo import MPO
from safe_rl.algorithms.safe_sac import SafeSAC
from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic


def _ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from uniform (0 = calibrated)."""
    if u.size == 0:
        return float("nan")
    s = np.sort(u)
    return float(np.max(np.abs(s - (np.arange(1, s.size + 1) / s.size))))


def _resolve_lambda_init(lambda_init: float | None, lambda_max: float) -> float:
    """Starting value for the E-step multiplier (1.0 by default, as before)."""
    if lambda_init is None:
        return 1.0
    if float(lambda_init) < 0.0:
        raise ValueError(f"lambda_init must be non-negative, got {lambda_init}.")
    if float(lambda_init) > lambda_max:
        raise ValueError(
            f"lambda_init {lambda_init} exceeds lambda_max {lambda_max}; it would be clipped on "
            "the first update and the pin would not hold."
        )
    return float(lambda_init)


def _validate_homotopy_config(
    homotopy_beta: float,
    homotopy_beta_max: float,
    homotopy_cnow_ema: float,
    qc_target_ema: bool,
    use_measured_qc_scale: bool,
    qc_target_ema_horizon: int,
    qc_target_jc_ema: float | None,
    qc_target_min_frac: float,
    qc_target_max_frac: float,
    qc_target_max_rise: float,
    feasibility_probe_interval: int,
) -> None:
    """Range-check the threshold-homotopy / measured-q_target settings.

    Module-level so ``CVPO.__init__`` stays under the repo's cyclomatic-complexity limit.
    """
    if not 0.0 <= float(homotopy_beta) < 1.0:
        raise ValueError(f"homotopy_beta must be in [0, 1), got {homotopy_beta}.")
    if not 0.0 <= float(homotopy_cnow_ema) <= 1.0:
        raise ValueError(f"homotopy_cnow_ema must be in [0, 1], got {homotopy_cnow_ema}.")
    if not 0.0 <= float(homotopy_beta_max) < 1.0:
        raise ValueError(f"homotopy_beta_max must be in [0, 1), got {homotopy_beta_max}.")
    if float(homotopy_beta_max) < float(homotopy_beta):
        raise ValueError(
            f"homotopy_beta_max must be >= homotopy_beta, got {homotopy_beta_max} < {homotopy_beta}: "
            "the reachability floor would override the ratchet on every update."
        )
    if float(qc_target_max_rise) < 0.0:
        raise ValueError(f"qc_target_max_rise must be >= 0, got {qc_target_max_rise}.")
    if qc_target_ema and use_measured_qc_scale:
        raise ValueError(
            "qc_target_ema and use_measured_qc_scale are mutually exclusive: the latter "
            "estimates qc_scale once and freezes it deliberately, and running both would "
            "move the threshold for two unrelated reasons."
        )
    if int(qc_target_ema_horizon) < 1:
        raise ValueError(f"qc_target_ema_horizon must be >= 1, got {qc_target_ema_horizon}.")
    if qc_target_jc_ema is not None and not 0.0 < float(qc_target_jc_ema) <= 1.0:
        raise ValueError(f"qc_target_jc_ema must be in (0, 1], got {qc_target_jc_ema}.")
    if float(qc_target_min_frac) <= 0.0 or float(qc_target_max_frac) < float(qc_target_min_frac):
        raise ValueError(
            f"require 0 < qc_target_min_frac <= qc_target_max_frac, got "
            f"{qc_target_min_frac} / {qc_target_max_frac}."
        )
    if int(feasibility_probe_interval) < 0:
        raise ValueError(f"feasibility_probe_interval must be >= 0, got {feasibility_probe_interval}.")


def _spread_match_ratio(q: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
    """Per-state ``s_b = std_a(Q_r) / std_a(Q_c)``. ``q``/``qc`` are ``[N, B]``; std is over actions."""
    return q.std(dim=0) / qc.std(dim=0).clamp_min(1e-6)


def _spread_match_scale(ratio: torch.Tensor, norm_median: torch.Tensor, match_max: float) -> torch.Tensor:
    """``m(s) = cap(s_b(s) / M)`` -- the factor the E-step exponent's cost term is multiplied by.

    Order of operations is load-bearing: per-state ratio -> divide by the batch normalizer M ->
    cap. The cap lands on the NORMALIZED quotient, not on ``s_b`` itself; keeping it there is what
    makes ``estep_match_normalizer="active"`` bit-identical to every run before the flag existed.

    Which median ``M`` is depends on the normalizer mode -- see `_estep_weights`. That choice is
    the only difference between the two modes.
    """
    return (ratio / norm_median.clamp_min(1e-6)).clamp(1.0 / match_max, match_max)


def _validate_spread_match(
    enabled: bool, match_max: float, lambda_mode: str, lambda_source: str, normalizer: str
) -> None:
    """Range/mode checks for ``estep_cost_spread_match`` (module-level: keeps __init__ under C901)."""
    if normalizer not in ("active", "reference"):
        raise ValueError(f"estep_match_normalizer must be 'active' or 'reference', got {normalizer!r}.")
    if not enabled:
        if normalizer != "active":
            raise ValueError(
                "estep_match_normalizer only has meaning when estep_cost_spread_match is on -- "
                "it selects which batch median normalizes the per-state match factor."
            )
        return
    if lambda_source != "episodic" or lambda_mode != "grad":
        raise ValueError(
            "estep_cost_spread_match rescales the E-step exponent per state, which breaks "
            "the LEVEL semantics the qspace controller and the joint dual solve rely on. "
            "Use lambda_mode='grad' with lambda_source='episodic'."
        )
    if match_max <= 0.0:
        raise ValueError(f"estep_spread_match_max must be > 0, got {match_max}")


class CVPO(MPO, SafeSAC):
    """Constrained Variational Policy Optimization (Liu et al., ICML 2022).

    https://arxiv.org/abs/2201.11927

    MPO with a cost constraint. Everything structural is inherited: the E-step sampling,
    the M-step weighted MLE under decoupled mean/covariance KL trust regions, and the
    target-actor sync come from :class:`MPO`; the cost critic and replay come from
    :class:`SafeSAC`. CVPO overrides one hook, :meth:`_estep_weights`, replacing

        q(a|s) ~ exp(Q_r / eta)     with     q(a|s) ~ exp((Q_r - lambda*Q_c) / eta)

    where ``(eta, lambda)`` solve the per-batch convex dual of

        max_q E_q[Q_r]  s.t.  KL(q || pi_old) <= eps,  E_q[Q_c] <= qc_thres.

    ``lambda_mode="dual"`` solves that jointly by SLSQP (the formulation as published);
    ``"grad"`` holds lambda over the batch and moves it with a controller, which avoids
    the bang-bang behaviour of the joint solve when the sampled actions cannot reach the
    threshold (codex/cvpo-negative-result.md). ``lambda_source="episodic"`` drives it from
    realized episodic cost instead, so it never reads Q_c's absolute level.
    """

    policy: SafeSACActorCritic

    def __init__(
        self,
        policy: SafeSACActorCritic,
        # --- CVPO-specific (the E/M-step arguments are MPO's, forwarded via kwargs) ---
        cost_horizon: int = 1000,  # episode length used to scale the episodic cost limit -> Q-space
        qc_thres: float | None = None,  # override the auto-computed cost-Q threshold
        lambda_mode: str = "grad",  # "grad" (graded projected ascent) or "dual" (per-batch joint SLSQP)
        lambda_lr: float = 0.03,  # step size for the graded-lambda controller
        lambda_init: float | None = None,  # starting lambda; with lambda_lr=0 this pins it
        # "qspace"  : delta = E_q[Q_c] - threshold        (reads the critic's LEVEL)
        # "episodic": delta = (J_c - limit) / limit       (reads only realized rollout cost)
        lambda_source: str = "qspace",
        lambda_episodic_warmup: int = 5,  # cost reports required before the controller engages
        lambda_max: float = 100.0,  # cap on lambda (also the dual upper bound)
        cost_constraint_mode: str = "mean",  # "mean" = E[Z_c] (CVPO) | "cvar" = tail mean (WCSAC-style)
        cvar_alpha: float = 0.9,  # tail level for cvar mode: constrain the worst 1-alpha fraction
        risk_levels: list | None = None,  # per-mode signed tail fraction: +averse, -seeking, 1 = mean
        risk_floor_frac: float = 0.3,  # floor a quantile at this fraction of the mean
        cost_critic_passive: bool = False,  # train Q_c but never let it influence the policy
        lambda_update: str = "sgd",  # "sgd" (legacy integral update) | "pid" (Stooke et al. 2020)
        lambda_kp: float = 0.0,  # PID proportional gain
        lambda_kd: float = 0.0,  # PID derivative gain
        lambda_anti_windup: bool = True,  # freeze integral while saturated (pid only)
        rescale_by_lambda: bool = False,  # use (Q_r - lam*Q_c)/(1+lam) in the E-step
        estep_cost_spread_match: bool = False,  # per-state cost spread matched to reward spread
        estep_spread_match_max: float = 10.0,  # cap on the per-state match factor (noise guard)
        estep_match_normalizer: str = "active",  # "active" | "reference" -- which batch median normalizes s_b
        recalibrate_cvar: bool = False,  # apply a learned monotone CDF map before reading CVaR
        recal_interval: int = 500,  # cost-critic updates between isotonic refits
        recal_capacity: int = 20000,  # PIT samples retained (FIFO)
        recal_min_samples: int = 500,  # refuse to fit below this
        use_measured_qc_scale: bool = False,  # estimate qc_scale from rollouts, then freeze
        qc_scale_estimate_episodes: int = 50,  # completed episodes to estimate it from
        qc_scale_source: str = "analytic",  # "analytic" | "measured" — how episodic -> Q-space is scaled
        qc_scale_measured: float | None = None,  # required when qc_scale_source == "measured"
        qc_scale_probe: str | None = None,  # provenance: which probe run the measured value came from
        qc_thres_adapt: bool = False,  # calibrate qc_thres from realized episodic cost
        qc_thres_lr: float = 2e-3,  # integral gain of that outer loop
        qc_thres_min_frac: float = 0.05,  # floor, as a fraction of the analytic qc_thres
        qc_ema: float = 0.05,  # EMA weight for the realized-cost estimate
        # --- feasible threshold homotopy ---
        qc_thres_homotopy: bool = False,  # ask only for a reachable reduction each update
        homotopy_beta: float = 0.005,  # fractional reduction demanded, per update
        homotopy_ratchet: bool = True,  # keep the threshold monotone non-increasing
        homotopy_beta_max: float = 0.02,  # hard cap on the per-update ask, so the ratchet stays reachable
        homotopy_cnow_ema: float = 0.01,  # EMA weight on C_now before the ratchet; 0 = raw
        # --- measured q_target (the homotopy floor) ---
        qc_target_ema: bool = False,  # derive q_target from EMA[C_now] / EMA[J_c]
        qc_target_ema_horizon: int = 200,  # updates; EMA weight for the Q_c numerator is 1/horizon
        qc_target_jc_ema: float | None = None,  # EMA weight for J_c; None -> reuse qc_ema
        qc_target_warmup_reports: int = 10,  # cost reports before q_target is trusted
        qc_target_max_rise: float = 0.01,  # cap on q_target's fractional increase per update; 0 = off
        qc_target_min_frac: float = 0.1,  # clamp band, as a fraction of the static qc_thres
        qc_target_max_frac: float = 10.0,
        # --- diagnostics ---
        feasibility_probe_interval: int = 1,  # updates between E-step feasibility probes; 0 = off
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        # CVPO handles exploration through the E-step KL trust region, not SAC entropy.
        # Disable the SAC entropy temperature (alpha ~ 0) and its auto-tuning.
        kwargs.pop("auto_entropy_tuning", None)
        kwargs.pop("alpha", None)
        super().__init__(policy, device=device, auto_entropy_tuning=False, alpha=1e-8, **kwargs)

        if self.num_costs != 1:
            raise ValueError(
                f"CVPO E-step dual solve supports a single cost constraint, got num_costs={self.num_costs}."
            )

        if lambda_mode not in ("grad", "dual"):
            raise ValueError(f"lambda_mode must be 'grad' or 'dual', got {lambda_mode!r}.")
        self.lambda_mode = lambda_mode
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        if lambda_source not in ("qspace", "episodic"):
            raise ValueError(f"lambda_source must be 'qspace' or 'episodic', got {lambda_source!r}.")
        # PID-Lagrangian on realized episodic cost (Stooke et al. 2020, arXiv:2007.03964).
        #
        # Both threshold schemes -- the static qc_thres and the homotopy -- drive lambda from
        # E_q[Q_c] vs a threshold, so both depend on the cost critic's absolute LEVEL. Measured,
        # that level is unreliable: it collapses to 0.05-0.9 against a threshold of 1.91 while
        # episodic cost sits at 50, so the constraint reads satisfied and lambda falls to 0
        # (2 of 3 both_lam4 seeds). The homotopy fails worse -- its threshold tracks the policy's
        # own cost, so the violation signal (~0.025) sits under its own batch noise (0.1-0.6) and
        # lambda random-walks. See codex/cvpo-feasible-threshold-homotopy.md.
        #
        # "episodic" never reads the level. delta = (J_c - d)/d is dimensionless (scale-invariant,
        # per Stooke), so the gains do not have to be retuned per cost limit, and the E-step keeps
        # using Q_c for RANKING only -- which the fixed-lambda front shows is sufficient.
        self.lambda_source = lambda_source
        self.lambda_episodic_warmup = int(lambda_episodic_warmup)
        self._external_cost_limit: float | None = None
        self._lambda_delta = float("nan")
        self._lambda_reports = 0
        # Last realized episodic cost the controller acted on, so an unchanged report is
        # not integrated again (see _update_lambda_episodic).
        self._last_realized_cost: float | None = None
        self.qc_thres_adapt = bool(qc_thres_adapt)
        self.qc_thres_lr = float(qc_thres_lr)
        self.qc_thres_min_frac = float(qc_thres_min_frac)
        self.qc_ema = float(qc_ema)
        self._realized_cost_ema: float | None = None

        # Convert the episodic cost limit (e.g. 25) into the discounted cost-Q scale the
        # cost critic actually predicts, i.e. the ratio G_c(s_0) / J_c.
        #
        # "analytic" assumes cost is spread uniformly over the episode, giving
        # (1 - gamma^H)/(1 - gamma)/H. On SafetyPointGoal1 the *total* cost mass really is
        # near-uniform in time, but gamma^t weights the first ~1/(1-gamma) steps most, and
        # those carry slightly less cost than average, so the analytic scale over-reads:
        # measured G_c(s_0)/J_c = 3.55/46.48 = 0.0764 against an analytic 0.1.
        #
        # "measured" substitutes a ratio measured by scratchpad/cost_critic_probe.py. This is
        # a static recalibration of the units, not a controller — `qc_thres_adapt` is separate
        # and stays off.
        if qc_scale_source not in ("analytic", "measured"):
            raise ValueError(f"qc_scale_source must be 'analytic' or 'measured', got {qc_scale_source!r}.")
        g, h = self.gamma, int(cost_horizon)
        self._qc_scale_analytic = (1.0 - g**h) / (1.0 - g) / max(h, 1)
        self.qc_scale_source = qc_scale_source
        self.qc_scale_probe = qc_scale_probe
        if qc_scale_source == "measured":
            if qc_scale_measured is None or float(qc_scale_measured) <= 0.0:
                raise ValueError("qc_scale_source='measured' requires a positive qc_scale_measured.")
            self._qc_scale = float(qc_scale_measured)
        else:
            self._qc_scale = self._qc_scale_analytic

        if qc_thres is not None:
            self.qc_thres = float(qc_thres)
        else:
            self.qc_thres = float(self.cost_limits[0]) * self._qc_scale
        self._qc_thres_initial = self.qc_thres
        print(
            f"CVPO qc_thres (cost-Q threshold) = {self.qc_thres:.4f}  "
            f"(episodic limit {self.cost_limits[0]}, qc_scale={self._qc_scale:.5f} "
            f"[{qc_scale_source}], analytic would be {self._qc_scale_analytic:.5f})"
        )
        if qc_scale_source == "measured":
            print(f"CVPO qc_scale provenance: {qc_scale_probe or '(unspecified)'}")

        # Online qc_scale estimation (item 1b). Estimated once from the first
        # `qc_scale_estimate_episodes` completed episodes, then FROZEN: a continuously moving
        # threshold would make the lambda controller's delta reflect target motion rather than
        # policy motion. Inert when the flag is off (`_qc_scale_frozen` starts True).
        self.use_measured_qc_scale = bool(use_measured_qc_scale)
        self.qc_scale_estimate_episodes = int(qc_scale_estimate_episodes)
        self._qc_scale_frozen = not self.use_measured_qc_scale
        self._qc_scale_open_eps: list[list[float]] | None = None
        self._qc_scale_done_eps: list[np.ndarray] = []
        if self.use_measured_qc_scale:
            print(
                f"CVPO use_measured_qc_scale: estimating qc_scale from the first "
                f"{self.qc_scale_estimate_episodes} completed episodes "
                f"(current {self._qc_scale:.5f}, analytic {self._qc_scale_analytic:.5f})"
            )

        # Feasible threshold homotopy. A fixed qc_thres asks the E-step for a cost level the
        # policy cannot reach inside its KL trust region: at ~1.8x the budget, one update can
        # shed a fraction of a percent of the gap, so the constraint set is empty and the dual
        # correctly parks lambda at its cap, where exp((Q_r - lam*Q_c)/eta) degenerates to
        # "minimise cost, ignore reward". Instead ask each update only for a reachable
        # reduction from where the policy actually is:
        #
        #     thresh = max(q_target, (1 - beta) * C_now),  C_now = E_{a~pi_old}[Q_c]
        #
        # This is self-limiting: thresh is pinned just under C_now, so it descends only as fast
        # as the policy descends -- beta does not compound against a stationary policy. It is
        # the opposite direction from `qc_thres_adapt`, which only ever tightened and walked the
        # threshold to its floor (codex/cvpo-qc-threshold-calibration.md).
        _validate_homotopy_config(
            homotopy_beta=homotopy_beta,
            homotopy_beta_max=homotopy_beta_max,
            homotopy_cnow_ema=homotopy_cnow_ema,
            qc_target_ema=qc_target_ema,
            use_measured_qc_scale=self.use_measured_qc_scale,
            qc_target_ema_horizon=qc_target_ema_horizon,
            qc_target_jc_ema=qc_target_jc_ema,
            qc_target_min_frac=qc_target_min_frac,
            qc_target_max_frac=qc_target_max_frac,
            qc_target_max_rise=qc_target_max_rise,
            feasibility_probe_interval=feasibility_probe_interval,
        )
        self.qc_thres_homotopy = bool(qc_thres_homotopy)
        self.homotopy_beta = float(homotopy_beta)
        self.homotopy_ratchet = bool(homotopy_ratchet)
        self.homotopy_beta_max = float(homotopy_beta_max)
        self.homotopy_cnow_ema = float(homotopy_cnow_ema)
        # Starts at +inf, not at qc_thres: the ratchet must not clamp before it has seen a
        # batch. Seeding it at the static target would make update 1 ratchet straight down to
        # the (unreachable) target and pin it there, disabling the homotopy entirely.
        self._qc_thres_eff = float("inf")
        self._cnow_ema: float | None = None

        # Measured q_target: the homotopy floor, in the units the cost critic actually speaks.
        # q_target = cost_lim * EMA[C_now] / EMA[J_c], so that
        #     C_now / q_target ~= J_c / cost_lim,
        # i.e. the Q-space overshoot equals the episodic overshoot. Neither static scale has
        # that property: the analytic 0.1 reads C_now = 2.12 as already satisfied (which is why
        # lambda sat at 0 in the baseline arms), and the MC units fix 0.0764 gives a Q-space
        # ratio of 1.11 against a true 1.86 because it leaves the critic's level bias in place
        # (Q_c(s0) = 2.12 against an MC G_c(s0) = 3.55). Dividing the critic's own reading by the
        # realized cost absorbs both. See codex/cvpo-cost-critic-investigation.md.
        self.qc_target_ema = bool(qc_target_ema)
        self.qc_target_ema_alpha = 1.0 / float(qc_target_ema_horizon)
        # Match the two EMA horizons. They tick in different clocks: C_now advances once per
        # gradient update, J_c once per update() call (= num_updates_per_step gradient updates).
        # Any mismatch makes q_target = EMA[C_now]/EMA[J_c] move for a reason that is not the
        # policy -- and because the q_target floor sits outside the ratchet, a spurious rise
        # loosens the threshold precisely when the policy is improving fastest. Derived unless
        # a weight is given explicitly.
        if qc_target_jc_ema is None:
            self.qc_target_jc_ema = min(1.0, float(self.num_updates_per_step) / float(qc_target_ema_horizon))
        else:
            self.qc_target_jc_ema = float(qc_target_jc_ema)
        self.qc_target_warmup_reports = int(qc_target_warmup_reports)
        # Belt-and-braces against the same failure: cap how fast q_target may rise, whatever
        # the horizons do. Falls are unclamped -- a tightening target is the intended direction.
        self.qc_target_max_rise = float(qc_target_max_rise)
        self._q_target_used: float | None = None
        self.qc_target_min_frac = float(qc_target_min_frac)
        self.qc_target_max_frac = float(qc_target_max_frac)
        self.feasibility_probe_interval = int(feasibility_probe_interval)
        self._qc_target_num_ema: float | None = None  # EMA[C_now], numerator of q_target
        self._cost_report_count = 0

        # Lambda saturation tracking: the cap is where the reward term is erased, so the
        # fraction of updates spent there is the headline symptom this change targets.
        self._lam_update_count = 0
        self._lam_at_cap_count = 0
        self._lam_at_cap = 0.0
        self._lam_at_cap_frac_ema: float | None = None
        self._feas_probe_count = 0
        self._last_feasibility: dict[str, float] = {}
        self._last_spread: dict[str, float] = {}
        self._last_spread_match: dict[str, float] = {}
        self._last_readout_diag: dict[str, Any] = {}
        if self.qc_thres_homotopy:
            print(
                f"CVPO qc_thres_homotopy: beta={self.homotopy_beta} "
                f"ratchet={self.homotopy_ratchet} cnow_ema={self.homotopy_cnow_ema}; "
                f"q_target={'measured EMA' if self.qc_target_ema else f'static {self.qc_thres:.4f}'}"
            )

        # Frozen target actor: the E-step samples from it and the M-step KL is measured
        # against it. Polyak-averaged toward the online actor after each actor update.
        self.actor_target = deepcopy(self.policy.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # M-step KL Lagrange multipliers (dual-ascent, warm-started across batches). They are
        # arrays so the scalar and per-dimension trust regions share one code path:
        # shape [1] vs [num_actions].
        dual_dim = self.policy.num_actions if self.per_dim_constraining else 1
        self.eta = 1.0  # E-step temperature (warm-start for SLSQP)
        # Passive mode: the cost critic is still trained every update, but lambda is pinned at
        # 0 so the E-step weight is exp(Q_r/eta) — identical to unconstrained MPO. This makes
        # Q_c a pure observer, which is what isolates "can the cost be learned" from "does the
        # cost signal change the policy".
        if cost_constraint_mode not in ("mean", "cvar", "risk"):
            raise ValueError(f"cost_constraint_mode must be 'mean', 'cvar' or 'risk', got {cost_constraint_mode!r}.")
        self.cost_constraint_mode = cost_constraint_mode
        if not 0.0 <= float(cvar_alpha) < 1.0:
            raise ValueError(f"cvar_alpha must be in [0, 1), got {cvar_alpha}.")
        self.cvar_alpha = float(cvar_alpha)
        # Risk-conditioned modes. The env appends the active mode's index to the observation,
        # so the selector is read straight off its tail. Each level is a SIGNED tail fraction
        # consumed by `DistributionalCritic.risk_value`: +f = mean of the worst f (averse),
        # -f = mean of the best f (seeking), |f| = 1 = the plain mean. Signing it this way
        # means neutral is not a special case -- CVaR over the whole distribution IS the mean.
        self.risk_levels = self._validate_risk_levels(risk_levels, self.cost_constraint_mode)
        self.num_risk_modes = len(self.risk_levels)
        self.risk_floor_frac = float(risk_floor_frac)
        self.cost_critic_passive = bool(cost_critic_passive)
        if lambda_update not in ("sgd", "pid"):
            raise ValueError(f"lambda_update must be 'sgd' or 'pid', got {lambda_update!r}.")
        self.lambda_update = lambda_update
        self.rescale_by_lambda = bool(rescale_by_lambda)

        # Per-state cost spread matching in the E-step exponent (2026-08-20, from
        # scripts/analysis/qc_spread_probe.py): std_a(rho_c) is ~0.5-1% of the cost LEVEL, the
        # per-state balance point std_a(Q_r)/std_a(rho_c) exceeds lambda_max at 23-46% of states,
        # and lambda_max moves the E-step weights by TV < 0.05 at another ~30% -- the constraint
        # is inert exactly where the spread collapses. With the flag on, the EXPONENT uses
        # ``qc * s_b`` with per-state ``s_b = std_a(Q_r) / std_a(Q_c)`` (capped), so lambda = 1
        # means "cost pulls as hard as reward" at EVERY state. Levels are untouched: eqc,
        # homotopy and every diagnostic keep raw units (the softmax discards per-state constants
        # anyway). Only meaningful -- and only allowed -- with the episodic lambda controller,
        # whose feedback is realized cost, not the (now rescaled) Q-space exponent.
        self.estep_cost_spread_match = bool(estep_cost_spread_match)
        self.estep_match_normalizer = str(estep_match_normalizer)
        self.estep_spread_match_max = float(estep_spread_match_max)
        _validate_spread_match(
            self.estep_cost_spread_match,
            self.estep_spread_match_max,
            lambda_mode,
            lambda_source,
            self.estep_match_normalizer,
        )

        # CVaR recalibration (item 2). CVaR-only by construction: the mean constraint never
        # reaches this code, so enabling it cannot shift the mean arm and re-confound the
        # comparison. Identity until the buffer reaches recal_min_samples.
        self.recalibrate_cvar = bool(recalibrate_cvar)
        self.recal_interval = int(recal_interval)
        self._recalibrator = (
            PITRecalibrator(capacity=int(recal_capacity), min_samples=int(recal_min_samples))
            if self.recalibrate_cvar
            else None
        )
        self._recal_updates = 0
        self._recal_x_t: torch.Tensor | None = None
        self._recal_y_t: torch.Tensor | None = None
        self._recal_ks: float = float("nan")
        # E-step cost multiplier. With lambda_lr = 0 and lambda_update = "sgd" the controller
        # is a no-op, so lambda_init pins lambda at a fixed value -- that is how the fixed-lambda
        # sweep traces the reward/cost front without a controller in the loop.
        lam0 = _resolve_lambda_init(lambda_init, self.lambda_max)
        self.lam = 0.0 if self.cost_critic_passive else lam0
        # Lambda controller (item 3). "sgd" reproduces the previous inline update exactly, so
        # the default path stays bit-identical and the smoke oracle still holds. Seeded from
        # self.lam so the two never diverge.
        self._lambda_ctrl = LambdaController(
            mode=self.lambda_update,
            lr=self.lambda_lr,
            kp=float(lambda_kp),
            ki=self.lambda_lr,
            kd=float(lambda_kd),
            lam_max=self.lambda_max,
            anti_windup=bool(lambda_anti_windup),
        )
        self._lambda_ctrl.lam = self.lam
        self._lambda_ctrl.integral = self.lam
        self.alpha_mean = np.zeros(dual_dim)  # M-step mean-KL multiplier(s)
        self.alpha_var = np.zeros(dual_dim)  # M-step var-KL multiplier(s)
        self._solver_status = -1.0
        self._solver_iters = 0.0
        self._last_actor_info: dict[str, float] = {}
        self._last_estep_info: dict[str, Any] = {}

    def update_lagrangian_multipliers(self, current_costs: list[float]) -> None:  # noqa: D401
        """Calibrate ``qc_thres`` against the realized episodic cost.

        CVPO's multiplier is solved in the E-step, so the inherited PID controller is
        unused. What this hook does instead is close the loop the analytic ``qc_thres``
        leaves open: that threshold converts the episodic budget to Q-space assuming
        costs are uniform over the episode, and when they are not, the constraint reads
        satisfied in Q-space while the episodic budget is exceeded.

        With ``qc_thres_adapt`` the threshold becomes the control variable and the
        realized episodic cost the measured output:

            qc_thres <- clip(qc_thres - lr * (EMA[J_c] - limit) * scale, floor, analytic)

        so a policy that overspends its budget tightens the Q-space target until the
        E-step's ``lambda`` engages. The ceiling is the analytic value, so the loop can
        only ever make the constraint stricter than requested, never looser.
        """
        if not current_costs:
            return
        if self.lambda_source == "episodic":
            self._update_lambda_episodic(float(current_costs[0]))
        # The realized-cost EMA also feeds the measured q_target, so it runs whenever either
        # consumer is enabled -- but the qc_thres ratchet below stays behind qc_thres_adapt.
        if not (self.qc_thres_adapt or self.qc_target_ema):
            return
        realized = float(current_costs[0])
        alpha = self.qc_target_jc_ema if self.qc_target_ema else self.qc_ema
        if self._realized_cost_ema is None:
            self._realized_cost_ema = realized
        else:
            self._realized_cost_ema += alpha * (realized - self._realized_cost_ema)
        self._cost_report_count += 1

        if not self.qc_thres_adapt:
            return

        limit = float(self.cost_limits[0])
        step = self.qc_thres_lr * (self._realized_cost_ema - limit) * self._qc_scale
        floor = self.qc_thres_min_frac * self._qc_thres_initial
        self.qc_thres = float(min(max(self.qc_thres - step, floor), self._qc_thres_initial))

    def _update_lambda_episodic(self, realized: float) -> None:
        """PI-Lagrangian step on realized episodic cost (Stooke et al. 2020).

        ``delta = (J_c - d) / d`` -- dimensionless, so the gains do not need retuning per cost
        limit, and ``lambda`` never reads ``Q_c``'s absolute level. The E-step still uses ``Q_c``,
        but only to *rank* candidate actions inside the per-state softmax.

        The empty-buffer guard is the point of failure this is most likely to hit. The runner
        computes ``current_costs = mean(costbuffers[i])`` and falls back to **0.0** when no
        episode has completed yet. Taken at face value that reads as ``delta = -1`` (maximum
        slack) on every iteration between ``update_after`` and the first completed episode --
        with 1000-step episodes that is hundreds of iterations of the integral being wound
        negative before a single real measurement arrives. Reports of exactly 0.0 are therefore
        discarded, and the controller waits ``lambda_episodic_warmup`` real reports before
        engaging. ``lambda_episodic_reports`` is logged so a stuck controller is visibly
        distinguishable from a starved one.
        """
        if self.cost_critic_passive:
            return
        if not np.isfinite(realized) or realized <= 0.0:
            return  # empty cost buffer (runner reports 0.0), not a measurement
        # One step per NEW measurement: `realized` only moves when an episode completes
        # (~every 250 iterations), and re-integrating it in between makes the loop ring.
        if self._last_realized_cost is not None and realized == self._last_realized_cost:
            return
        self._last_realized_cost = realized
        self._lambda_reports += 1
        if self._lambda_reports < self.lambda_episodic_warmup:
            return
        limit = self._episodic_cost_limit()
        self._lambda_delta = (realized - limit) / max(limit, 1e-12)
        self.lam = self._lambda_ctrl.update(self._lambda_delta)

    def _episodic_cost_limit(self) -> float:
        """Budget the episodic controller is held to; a curriculum's, if one is attached."""
        if self._external_cost_limit is not None:
            return float(self._external_cost_limit)
        return float(self.cost_limits[0])

    def set_cost_limit(self, limit: float | None) -> None:
        """Adopt an externally scheduled episodic cost limit."""
        self._external_cost_limit = None if limit is None else float(limit)

    def _effective_thres(self) -> float:
        """The cost-Q threshold this update's E-step is actually held to.

        Single accessor so the dual objective and the lambda controller cannot disagree, and
        so the homotopy is provably inert when switched off. Falls back to the static
        threshold until the first batch has set a finite value.
        """
        if not self.qc_thres_homotopy or not np.isfinite(self._qc_thres_eff):
            return self.qc_thres
        return self._qc_thres_eff

    def _current_q_target(self) -> float:
        """The homotopy floor: the real budget, in the units the cost critic reads.

        ``cost_lim * EMA[C_now] / EMA[J_c]``. Falls back to the static ``qc_thres`` until
        enough cost reports have arrived, and is clamped into
        ``[qc_target_min_frac, qc_target_max_frac] * qc_thres_static`` so a transient critic
        collapse cannot walk the target to zero the way ``qc_thres_adapt`` did.
        """
        if not self.qc_target_ema:
            return self.qc_thres
        if (
            self._cost_report_count < self.qc_target_warmup_reports
            or self._qc_target_num_ema is None
            or self._realized_cost_ema is None
            or self._realized_cost_ema <= 0.0
        ):
            return self.qc_thres
        target = float(self.cost_limits[0]) * (self._qc_target_num_ema / self._realized_cost_ema)
        lo = self.qc_target_min_frac * self._qc_thres_initial
        hi = self.qc_target_max_frac * self._qc_thres_initial
        return float(min(max(target, lo), hi))

    def _clamped_q_target(self) -> float:
        """``_current_q_target`` with its rate of *increase* capped, and the result latched.

        A rising q_target loosens the constraint, and it can rise for reasons that are not the
        policy (EMA horizon mismatch, a critic level drift). Falls are left unclamped: a
        tightening target is the direction the method is trying to move in.
        """
        raw = self._current_q_target()
        if self.qc_target_max_rise <= 0.0 or self._q_target_used is None:
            self._q_target_used = raw  # first valid value is accepted outright
            return raw
        ceiling = self._q_target_used * (1.0 + self.qc_target_max_rise)
        self._q_target_used = float(min(raw, ceiling))
        return self._q_target_used

    def _update_homotopy_threshold(self, c_now: float) -> float:
        """Advance the homotopy threshold from this batch's cost level; returns ``q_target``.

        ``c_now = E_{a~pi_old}[Q_c]``, the *unweighted* mean over the sampled candidates.
        """
        # The q_target numerator tracks C_now on every update, independent of the homotopy
        # switch, so the two flags stay orthogonal.
        if self.qc_target_ema:
            if self._qc_target_num_ema is None:
                self._qc_target_num_ema = c_now
            else:
                self._qc_target_num_ema += self.qc_target_ema_alpha * (c_now - self._qc_target_num_ema)

        q_target = self._clamped_q_target()
        if not self.qc_thres_homotopy:
            return q_target

        if self.homotopy_cnow_ema > 0.0:
            # Smooth before the ratchet: one noisy batch would otherwise lock the threshold
            # onto a level the policy never actually occupied.
            if self._cnow_ema is None:
                self._cnow_ema = c_now
            else:
                self._cnow_ema += self.homotopy_cnow_ema * (c_now - self._cnow_ema)
            c_use = self._cnow_ema
        else:
            c_use = c_now

        raw = (1.0 - self.homotopy_beta) * c_use
        if self.homotopy_ratchet:
            # Clamp the *ask*, not the level. A bare min(thresh_prev, .) is monotone but
            # decouples the demand from feasibility: if C_now rises, thresh_prev is now far
            # more than beta below it, and the per-update demand silently grows past what the
            # trust region can deliver -- the original bug, arriving slowly. The beta_max floor
            # caps the demand at a reachable fraction, so the threshold stays monotone while
            # feasible and loosens only just enough to stay reachable.
            raw = max((1.0 - self.homotopy_beta_max) * c_now, min(self._qc_thres_eff, raw))
        # The real budget always wins: never ask for less than q_target. Applied outside the
        # ratchet because q_target drifts with the critic level, and clamping inside would
        # leave the threshold stranded below a target that has since risen.
        self._qc_thres_eff = float(max(q_target, raw))
        return q_target

    def _reachable_qc_min(self, qc_np: np.ndarray) -> tuple[float, float]:
        """Smallest ``E_q[Q_c]`` the E-step could reach inside its KL budget.

        ``pi_old`` is uniform over the ``N`` sampled candidates, so the cost-minimising ``q``
        under ``KL(q||pi_old) <= eps_dual`` is ``q_nu ∝ exp(-nu·Q_c)``: ``E_{q_nu}[Q_c]`` is
        non-increasing in ``nu`` while the KL is non-decreasing, so bisect ``nu`` up to
        ``KL = eps_dual`` and read the cost off there. Returns ``(reachable_min, kl_at_nu)``.

        The E-step's constraint set is non-empty exactly when this floor is at or below the
        threshold. That is the *sampled-support* feasibility test — the quantity
        codex/cvpo-negative-result.md identified as empty when every candidate action is drawn
        from an already-unsafe policy, which no reweighting can repair.
        """
        eps = self.eps_dual
        n = qc_np.shape[0]

        def stats(nu: float) -> tuple[float, float]:
            z = -nu * qc_np
            z = z - z.max(axis=0, keepdims=True)
            w = np.exp(z)
            w /= w.sum(axis=0, keepdims=True)
            # KL(q || uniform) = sum_i w_i log(N w_i), matching nonparametric_kl_from_weights.
            kl = float(np.mean(np.sum(w * np.log(np.clip(w * n, 1e-300, None)), axis=0)))
            return float(np.mean(np.sum(w * qc_np, axis=0))), kl

        lo, hi = 0.0, 1.0
        _, kl_hi = stats(hi)
        for _ in range(60):
            if kl_hi >= eps:
                break
            lo, hi = hi, hi * 2.0
            _, kl_hi = stats(hi)
        else:
            # The KL never reaches eps: Q_c is (near-)constant across candidates, so the whole
            # simplex sits inside the trust region and the floor is the per-state minimum.
            return float(np.mean(qc_np.min(axis=0))), kl_hi

        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, kl_mid = stats(mid)
            if kl_mid < eps:
                lo = mid
            else:
                hi = mid
        return stats(lo)  # lo is the largest nu still inside the trust region

    def _probe_feasibility(self, qc_np: np.ndarray, thres: float) -> None:
        """Refresh the sampled-support feasibility diagnostics (every N updates)."""
        if self.feasibility_probe_interval <= 0:
            return
        run = self._feas_probe_count % self.feasibility_probe_interval == 0
        self._feas_probe_count += 1
        if not run:
            return
        reachable, kl_at = self._reachable_qc_min(qc_np)
        self._last_feasibility = {
            "qc_reachable_min": reachable,
            "estep_feasible": float(reachable <= thres),
            "estep_feasibility_margin": thres - reachable,
            "estep_feasibility_kl": kl_at,
        }

    def _solve_dual(self, q_np: np.ndarray, qc_np: np.ndarray) -> tuple[float, float]:
        """Solve the CVPO E-step dual for (eta, lambda) over the sampled Q / Qc.

        q_np, qc_np : [N, B] numpy arrays (N sampled actions per state, B states).

        Two modes (``self.lambda_mode``):

        * ``"grad"`` (default) — solve only the temperature ``eta`` from the dual with
          ``lambda`` held fixed at its current value; ``lambda`` is then updated by a slow
          projected-gradient controller in :meth:`_update_actor_and_alpha`. This avoids the
          bang-bang behaviour of the joint solve, which snaps ``lambda`` to a bound whenever
          the sampled action set can't reach ``E_q[Q_c] = qc_thres`` (see
          codex/cvpo-negative-result.md).
        * ``"dual"`` — the original per-batch joint SLSQP over both (eta, lambda), with the
          upper bound capped at ``lambda_max`` to limit M-step degeneracy.

        Returns the (eta, lambda) used to form this batch's variational weights, both > 0.
        """
        eps = self.eps_dual
        thres = self._effective_thres()

        if self.lambda_mode == "grad":
            # lambda is fixed this batch, so lam*thres is constant in eta and the objective
            # is MPO's with the exponent shifted by -lam*Qc.
            lam = self.lam
            return self._solve_eta(q_np - lam * qc_np), max(lam, 0.0)

        def dual(x: np.ndarray) -> float:
            eta, lam = x
            # z = (Q - lam * Qc) / eta, log-sum-exp over the N action samples, mean over states.
            z = (q_np - lam * qc_np) / eta
            zmax = z.max(axis=0, keepdims=True)
            lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
            return eta * eps + lam * thres + eta * float(np.mean(lse))

        x0 = np.array([max(self.eta, 1e-3), max(self.lam, 1e-3)], dtype=np.float64)
        bounds = [(1e-6, 1e6), (1e-6, self.lambda_max)]
        try:
            res = minimize(dual, x0, method="SLSQP", bounds=bounds)
            eta, lam = float(res.x[0]), float(res.x[1])
            self._solver_status = float(res.status)
            self._solver_iters = float(res.nit)
            if not np.isfinite(eta) or not np.isfinite(lam):
                raise ValueError("non-finite dual solution")
        except Exception as exc:  # pragma: no cover - numerical fallback
            print(f"CVPO dual solve failed ({exc}); keeping previous (eta, lambda).")
            eta, lam = self.eta, self.lam
            self._solver_status = -1.0
            self._solver_iters = 0.0
        return max(eta, 1e-6), max(lam, 1e-6)

    def store_transition(self, obs, action, reward, done, next_obs, cost=None, **kwargs) -> None:
        """Store a transition; additionally collect episode cost traces while estimating qc_scale.

        Collection is read-only with respect to training: it consumes no RNG and touches no
        parameters (asserted in tests/test_qc_scale_wiring.py).
        """
        super().store_transition(obs, action, reward, done, next_obs, cost=cost, **kwargs)
        if not self._qc_scale_frozen:
            self._collect_qc_scale_sample(cost, done)

    def _collect_qc_scale_sample(self, cost, done) -> None:
        """Accumulate per-env episode cost sequences; estimate and freeze once enough finish."""
        n_envs = int(done.shape[0])
        if self._qc_scale_open_eps is None:
            self._qc_scale_open_eps = [[] for _ in range(n_envs)]

        if cost is None:
            costs = np.zeros(n_envs, dtype=np.float64)
        else:
            costs = cost.detach().reshape(n_envs, -1)[:, 0].to("cpu").numpy().astype(np.float64)
        dones = done.detach().reshape(-1).to("cpu").numpy()

        for e in range(n_envs):
            self._qc_scale_open_eps[e].append(float(costs[e]))
            if dones[e] > 0:
                self._qc_scale_done_eps.append(np.asarray(self._qc_scale_open_eps[e]))
                self._qc_scale_open_eps[e] = []

        if len(self._qc_scale_done_eps) < self.qc_scale_estimate_episodes:
            return
        if sum(float(ep.sum()) for ep in self._qc_scale_done_eps) <= 0.0:
            return  # no cost observed yet; nothing to estimate from

        scale = measured_qc_scale(self._qc_scale_done_eps, self.gamma)
        thresholds = make_thresholds(float(self.cost_limits[0]), scale, mode="wcsac")
        old_thres, old_scale = self.qc_thres, self._qc_scale
        self._qc_scale = scale
        self.qc_thres = thresholds["mean"]
        self._qc_thres_initial = self.qc_thres
        self._qc_scale_frozen = True
        self._qc_scale_open_eps = None
        self._qc_scale_done_eps = []
        print(
            f"CVPO qc_scale measured from {self.qc_scale_estimate_episodes} episodes: "
            f"{old_scale:.5f} -> {scale:.5f} (analytic {self._qc_scale_analytic:.5f}); "
            f"qc_thres {old_thres:.4f} -> {self.qc_thres:.4f}; frozen"
        )

    def _update_lambda(self, eqc: float) -> float:
        """Advance the multiplier from the constraint violation ``E_q[Q_c] - thresh``.

        No-op under ``lambda_source="episodic"``. Gated here as well as at the call site so no
        future caller can quietly reintroduce the dependence on ``Q_c``'s absolute level -- that
        dependence is the measured cause of both threshold schemes failing.
        """
        if self.cost_critic_passive or self.lambda_source == "episodic":
            return self.lam
        self.lam = self._lambda_ctrl.update(float(eqc) - self._effective_thres())
        return self.lam

    def _track_lambda_saturation(self) -> None:
        """Record whether ``lambda`` is pinned at its cap this update.

        Called once per E-step, after both ``lambda_mode`` branches have settled on a value, so
        the joint-dual and graded-controller paths are measured the same way. Lambda at its cap
        is where ``exp((Q_r - lam*Q_c)/eta)`` stops depending on ``Q_r`` at all, so the fraction
        of updates spent there is the symptom the homotopy exists to remove -- it was ~0.87 in
        the CVaR arms (commit 9ac6018).
        """
        # lambda_max = 0 pins lambda at zero to disable the constraint; that is not saturation.
        at_cap = float(self.lambda_max > 0.0 and self.lam >= self.lambda_max - 1e-12)
        self._lam_at_cap = at_cap
        self._lam_update_count += 1
        self._lam_at_cap_count += int(at_cap)
        if self._lam_at_cap_frac_ema is None:
            self._lam_at_cap_frac_ema = at_cap
        else:
            self._lam_at_cap_frac_ema += 0.01 * (at_cap - self._lam_at_cap_frac_ema)

    def _recalibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Remap a categorical distribution's CDF through the fitted isotonic map (torch)."""
        r = self._recalibrator
        if r is None or not r.is_fitted:
            return probs
        if self._recal_x_t is None:
            self._recal_x_t = torch.as_tensor(r._x, dtype=probs.dtype, device=probs.device)
            self._recal_y_t = torch.as_tensor(r._y, dtype=probs.dtype, device=probs.device)
        x, y = self._recal_x_t, self._recal_y_t

        cdf = probs.cumsum(-1).clamp(0.0, 1.0)
        idx = torch.searchsorted(x, cdf.contiguous().reshape(-1)).clamp(1, x.numel() - 1)
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        frac = ((cdf.reshape(-1) - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
        new_cdf = (y0 + frac * (y1 - y0)).reshape(cdf.shape)
        new_cdf, _ = torch.cummax(new_cdf, dim=-1)
        new_cdf = new_cdf.clone()
        new_cdf[..., -1] = 1.0
        out = torch.diff(new_cdf, dim=-1, prepend=torch.zeros_like(new_cdf[..., :1])).clamp_min(0.0)
        total = out.sum(-1, keepdim=True)
        return torch.where(total > 0, out / total.clamp_min(1e-12), probs)

    def _record_pit(self, dist: torch.Tensor, realized: torch.Tensor, critic) -> None:
        """Store PIT values u = F_pred(realized) for the recalibration buffer."""
        if self._recalibrator is None:
            return
        z = critic.q_support
        cdf = critic.get_cdf(dist)
        idx = torch.searchsorted(z.contiguous(), realized.contiguous().clamp(z[0], z[-1]))
        idx = idx.clamp(1, z.numel() - 1)
        lo, hi = z[idx - 1], z[idx]
        frac = ((realized - lo) / (hi - lo).clamp_min(1e-9)).clamp(0.0, 1.0)
        c0 = torch.gather(cdf, 1, (idx - 1).unsqueeze(1)).squeeze(1)
        c1 = torch.gather(cdf, 1, idx.unsqueeze(1)).squeeze(1)
        pit = (c0 + frac * (c1 - c0)).detach().cpu().numpy()
        self._recalibrator.update(pit)
        self._recal_ks = _ks_uniform(pit)

        self._recal_updates += 1
        if self._recal_updates % self.recal_interval == 0 and self._recalibrator.refit():
            self._recal_x_t = None  # invalidate cached tensors after a refit
            self._recal_y_t = None

    @staticmethod
    def _validate_risk_levels(risk_levels: list | None, mode: str) -> list[float]:
        """Signed tail fractions for `DistributionalCritic.risk_value`, one per risk mode."""
        levels = [float(x) for x in risk_levels] if risk_levels else []
        for lvl in levels:
            if not 0.0 < abs(lvl) <= 1.0:
                raise ValueError(f"risk_levels entries must be a signed tail fraction with 0 < |x| <= 1, got {lvl}.")
        if mode == "risk" and not levels:
            raise ValueError("cost_constraint_mode='risk' requires a non-empty risk_levels.")
        return levels

    def _risk_cost(self, critic_obs: torch.Tensor, actions: torch.Tensor, target: bool) -> torch.Tensor:
        """Per-sample risk-adjusted Q_c, the mode read from the observation's last column.

        The distortion itself lives on the critic (`DistributionalCritic.risk_value`); this
        only routes each sample to its mode's level.

        The floor is not made redundant by CVaR: this cost distribution is zero-inflated
        (>50% of the mass on the zero atom), so the mean of ANY lower-tail fraction below
        ~0.57 is exactly zero, which would drop the cost term and leave that mode
        unconstrained rather than merely risk-seeking.
        """
        if not getattr(self.policy, "is_distributional_cost_critic", False):
            raise RuntimeError("cost_constraint_mode='risk' requires a distributional cost critic.")
        critics = self.policy.cost_critic_targets if target else self.policy.cost_critics
        normalizer = self.policy.critic_obs_normalizer
        obs_n = normalizer.normalize(critic_obs) if hasattr(normalizer, "normalize") else normalizer(critic_obs)

        critic = critics[0]
        dist = critic.get_dist(critic(obs_n, actions))
        mean = critic.get_value(dist)
        out = mean.clone()
        # The env appends the mode index rescaled to [0, 1]; undo that. Rounding is exact for
        # the levels sampled in training and snaps an interpolated eval-time level to the
        # nearest trained mode.
        denom = max(self.num_risk_modes - 1, 1)
        mode = (critic_obs[..., -1] * denom).round().long().clamp(0, self.num_risk_modes - 1)
        for i, level in enumerate(self.risk_levels):
            if abs(level) >= 1.0:  # neutral: risk_value would return this mean anyway
                continue
            sel = mode == i
            if bool(sel.any()):
                out[sel] = torch.maximum(critic.risk_value(dist[sel], level), self.risk_floor_frac * mean[sel])
        return out.unsqueeze(-1)

    def _estep_cost_readout_name(self) -> str:
        """Which cost readout the E-step exponent is currently built from. Diagnostics only."""
        return self.cost_constraint_mode

    def _estep_cost(self, critic_obs: torch.Tensor, actions: torch.Tensor, target: bool) -> torch.Tensor:
        """Cost signal the E-step constrains: either ``E[Z_c]`` or ``CVaR_alpha(Z_c)``.

        ``cost_constraint_mode="cvar"`` constrains the mean of the worst ``1 - alpha`` fraction
        of the cost return instead of its expectation, in the spirit of WCSAC
        (arXiv:2011.11814). Two differences from that paper: CVaR is read exactly off the
        categorical atoms rather than through a Gaussian fitted to a separate variance head,
        and the categorical critic is one network rather than two.

        Requires a distributional cost critic -- there is no distribution to take a tail of
        otherwise. Measured caveat: the raw categorical distribution is under-dispersed on
        SafetyPointGoal1 (predicted std ~3.2 vs realized ~5.5; PIT KS 0.38), so the raw CVaR
        understates tail risk by ~25-30%. See codex/cvpo-cost-critic-investigation.md; a
        post-hoc quantile recalibration fixes it (KS -> 0.06) but is not applied inside
        training here.
        """
        if self.cost_constraint_mode == "mean":
            fn = self.policy.evaluate_cost_q_target if target else self.policy.evaluate_cost_q
            return fn(critic_obs, actions)

        if self.cost_constraint_mode == "risk":
            return self._risk_cost(critic_obs, actions, target)

        if not getattr(self.policy, "is_distributional_cost_critic", False):
            raise RuntimeError(
                "cost_constraint_mode='cvar' requires a distributional cost critic "
                "(policy cost_critic_type='distributional')."
            )
        critics = self.policy.cost_critic_targets if target else self.policy.cost_critics
        # Normalize without moving the statistics: `critic_obs` is the [N*B] expanded
        # tensor, so updating here would count every state `sample_action_num` times.
        normalizer = self.policy.critic_obs_normalizer
        obs_n = normalizer.normalize(critic_obs) if hasattr(normalizer, "normalize") else normalizer(critic_obs)
        vals = []
        for c in critics:
            probs = c.get_dist(c(obs_n, actions))
            if self.recalibrate_cvar:
                probs = self._recalibrate_probs(probs)
            vals.append(c.get_cvar(probs, self.cvar_alpha))
        return torch.stack(vals, dim=0).mean(dim=0).unsqueeze(-1)

    def _estep_weights(self, q: torch.Tensor, actions: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """CVPO's constrained E-step: weights proportional to exp((Q_r - lambda*Q_c)/eta).

        Everything else -- sampling, the M-step, the target-actor sync -- is MPO's.
        """
        n, batch_size = q.shape
        cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
        qc_flat = self._estep_cost(cobs_exp, actions.reshape(n * batch_size, -1), self.estep_use_target_critic)
        qc = qc_flat[:, 0].reshape(n, batch_size)

        # Reference (mean) readout from the same forward pass, when the subclass provides one.
        # Identity is tracked on the PRE-reshape tensor: `reshape` returns a new object even for
        # an aliased input, so `qc_ref is qc` would be False exactly when the two are the same
        # readout. `ref_is_qc` is what makes the mean arm's dose exactly 1.0, not 1.0 + noise.
        ref_flat = getattr(self, "_last_estep_cost_ref", None)
        ref_is_qc = ref_flat is qc_flat
        qc_ref = qc if ref_is_qc else (None if ref_flat is None else ref_flat[:, 0].reshape(n, batch_size))

        # Spread matching (see __init__): the EXPONENT -- and therefore also the eta solve, which
        # must describe the same softmax -- sees qc * s_b; every level-consumer (eqc, homotopy,
        # feasibility probe, diagnostics) keeps the raw qc.
        #
        # MEDIAN-NORMALIZED (c2 revision, 2026-08-20): the raw ratio s_b = std_a(Qr)/std_a(Qc)
        # multiplied the TOTAL cost pressure by the batch's typical balance (~1-2x and rising as
        # spreads decay), on top of a lambda whose gains were tuned for the unscaled coupling --
        # the c1 arm (run 20260820_101946) collapsed into the idle-policy failure by iter 13k
        # (Eval reward ~ 0, cost 7 vs budget 25, training reward decaying). Dividing by the batch
        # median EQUALIZES influence across states -- which is the measured problem (~30% of
        # states inert) -- while the median state keeps s_b = 1, so lambda's tuned operating
        # point and the controller's plant gain are preserved.
        #
        # NORMALIZER (c4 revision, 2026-08-22): dividing by the ACTIVE readout's own median makes
        # m(s) invariant to a global rescale of that readout, so the mean -> CVaR switch passed
        # its full spread inflation (measured dose 2.3-10x, cap-saturating) straight into
        # lam * qc_exp against a lambda_max calibrated in mean units -- the c3 over-constraint.
        # `estep_match_normalizer="reference"` divides by the median of the MEAN-readout ratio
        # instead, which makes the delivered grip lam/M_ref: flat in the dose, and equal to what
        # the same lambda buys at kappa = 0. At kappa = 0 the two modes coincide exactly.
        qc_exp = qc
        if self.estep_cost_spread_match:
            ratio = _spread_match_ratio(q, qc)
            # `ratio` when the readouts are the same tensor -- keeps the mean arm bit-identical
            # instead of recomputing an equal-valued std.
            ratio_ref = ratio if (qc_ref is None or ref_is_qc) else _spread_match_ratio(q, qc_ref)
            if self.estep_match_normalizer == "reference" and qc_ref is None:
                raise RuntimeError(
                    "estep_match_normalizer='reference' needs a mean-readout reference from "
                    "`_estep_cost`, which this algorithm does not provide. FHDCMPO does; plain "
                    "CVPO's cvar/risk branches do not."
                )
            norm_median = ratio_ref.median() if self.estep_match_normalizer == "reference" else ratio.median()
            match_scale = _spread_match_scale(ratio, norm_median, self.estep_spread_match_max)
            qc_exp = qc * match_scale
            self._last_spread_match = {
                "estep_match_scale_median": float(match_scale.median()),
                "estep_match_scale_p90": float(match_scale.quantile(0.9)),
                # Upper cap only; the lower clamp is counted separately below. Under "reference"
                # the FLOOR is the one that binds -- m has median ~1/dose, so at dose ~= match_max
                # the median state sits on 1/match_max and the correction is silently truncated.
                # `floored_frac` is what makes that visible rather than a mystery.
                "estep_match_scale_capped_frac": float(
                    (match_scale >= self.estep_spread_match_max * (1 - 1e-6)).float().mean()
                ),
                "estep_match_scale_floored_frac": float(
                    (match_scale <= (1.0 / self.estep_spread_match_max) * (1 + 1e-6)).float().mean()
                ),
                # med(s_b) on the ACTIVE readout. Reads ~M_ref/dose under cvar in BOTH modes --
                # it describes the readout, not the normalizer.
                "estep_match_ratio_median": float(ratio.median()),
                # med(s_b) on the MEAN readout -- the c2-units normalizer. Logged in BOTH modes
                # (under "active" it is not what was divided by); ref/ratio medians = the dose.
                "estep_match_ref_median": float(ratio_ref.median()),
            }
            if qc_ref is not None:
                # s_b/M_active is invariant to a global rescale of the readout, so under "active"
                # this lands at ~1.0 however large the dose -- the standing evidence that median
                # normalization does NOT absorb it. Under "reference" it reads ~1/dose: that IS
                # the correction being applied.
                match_scale_ref = _spread_match_scale(ratio_ref, ratio_ref.median(), self.estep_spread_match_max)
                self._last_spread_match["estep_match_dose_vs_mean_median"] = float(
                    (match_scale / match_scale_ref.clamp_min(1e-12)).median()
                )

        # THE dose: how much more across-action cost signal the exponent's readout carries than
        # the plain mean. lambda multiplies this, so at fixed lambda it IS the change in E-step
        # constraint pressure -- and nothing downstream (the episodic lambda controller included)
        # adapts to it. See the c3 over-constraint in codex/offpolicy-mismatch-td-lambda.md.
        readout = self._estep_cost_readout_name()
        if ref_is_qc:
            dose = 1.0
        elif qc_ref is not None:
            dose = float((qc.std(dim=0) / qc_ref.std(dim=0).clamp_min(1e-12)).median())
        else:
            # Plain CVPO's own cvar/risk branches keep no cheap reference readout.
            dose = 1.0 if readout == "mean" else float("nan")
        self._last_readout_diag = {
            # String form, for probes and tests. The runner's logger filters `get_penalty_info`
            # to numeric values (off_policy_runner.py), so the 0/1 companion below is the one
            # that actually reaches tensorboard/wandb.
            "estep_match_readout": readout,
            "estep_match_readout_is_tail": 0.0 if readout == "mean" else 1.0,
            "estep_cost_dose_vs_mean_median": dose,
            "estep_match_normalizer": self.estep_match_normalizer,
            "estep_match_normalizer_is_reference": 1.0 if self.estep_match_normalizer == "reference" else 0.0,
        }

        # Delivered relative grip: how hard the cost term pulls against the reward term in the
        # exponent, per unit of reward spread, averaged over states. `m * std_a(C) / std_a(Q_r)`
        # is what the softmax actually sees, so this -- not lambda, and not the dose -- is the
        # quantity that must stay put when the readout changes. Under "reference" it collapses to
        # lam / M_ref (the cap aside), i.e. FLAT in the dose; under "active" it is dose x that.
        grip_per_state = (qc.std(dim=0) / q.std(dim=0).clamp_min(1e-12)) * (
            match_scale if self.estep_cost_spread_match else 1.0
        )

        q_np = q.cpu().numpy().astype(np.float64)
        qc_np = qc.cpu().numpy().astype(np.float64)  # RAW: homotopy/feasibility/spread read levels
        qc_exp_np = qc_np if qc_exp is qc else qc_exp.cpu().numpy().astype(np.float64)

        # Homotopy: aim just below the policy's own cost level so the E-step is asked for a
        # reduction it can reach, floored at the real target. Both consumers below read it.
        c_now = float(qc_np.mean())
        q_target = self._update_homotopy_threshold(c_now)
        thres_used = self._effective_thres()

        eta, lam = self._solve_dual(q_np, qc_exp_np)
        self.eta, self.lam = eta, lam
        lam_used = lam  # self.lam is stepped below, so diagnostics must use this one
        self._last_readout_diag["estep_match_grip"] = float(lam_used * grip_per_state.mean())

        combined = rescale_advantage(q, qc_exp, lam) if self.rescale_by_lambda else (q - lam * qc_exp)
        weights = torch.softmax(combined / eta, dim=0)  # [N, B], columns sum to 1

        eqc = (weights * qc).sum(dim=0).mean().item()  # E_q[Q_c] over states
        self._eqc = eqc
        if self.lambda_mode == "grad" and self.lambda_source == "qspace":
            self._update_lambda(eqc)
        self._track_lambda_saturation()
        self._probe_feasibility(qc_np, thres_used)

        # Only the SPREAD across candidate actions survives the per-state softmax: a constant
        # added to Q_c cancels in the normalisation. lambda_balanced is where lambda should
        # settle to weigh reward and cost equally.
        std_qr, std_qc = q_np.std(axis=0), qc_np.std(axis=0)
        lam_balanced = float(np.median(std_qr / np.maximum(std_qc, 1e-12)))
        self._last_spread = {
            "estep_std_qr": float(np.median(std_qr)),
            "estep_std_qc": float(np.median(std_qc)),
            "lambda_balanced": lam_balanced,
            "estep_spread_ratio": float(np.median(lam_used * std_qc / np.maximum(std_qr, 1e-12))),
            "lambda_over_balanced": lam_used / max(lam_balanced, 1e-12),
        }
        self._last_estep_info = {
            "lambda": self.lam,
            "eqc": eqc,
            # Eqc back in episodic-cost units, comparable with the runner's measured cost.
            "eqc_as_episodic_cost": eqc / max(self._qc_scale, 1e-12),
            "qc_thres_eff": thres_used,
            "qc_thres_target": q_target,
            "qc_thres_static": self.qc_thres,
            "c_now": c_now,
            "c_now_over_thres": c_now / max(thres_used, 1e-12),
            "qc_scale_target_ema": (
                self._qc_target_num_ema / self._realized_cost_ema
                if self._qc_target_num_ema is not None
                and self._realized_cost_ema is not None
                and self._realized_cost_ema > 0.0
                else float("nan")
            ),
            # dg/dlambda: ~0 when the constraint is active and interior.
            "dual_residual_lambda": thres_used - eqc,
        }
        return weights

    def _estep_extra_info(self) -> dict[str, Any]:
        return self._last_estep_info

    def get_penalty_info(self) -> dict[str, Any]:
        """CVPO logging: the per-batch dual variables and M-step KL diagnostics."""
        info = {
            "lambda_mean": self.lam,
            "lambda_max": self.lam,
            "lambda_min": self.lam,
            "lambda_list": [self.lam],
            "cost_limits": self.cost_limits.copy(),
            "qc_thres": self.qc_thres,
            "qc_scale": self._qc_scale,
            "realized_cost_ema": self._realized_cost_ema if self._realized_cost_ema is not None else float("nan"),
            "eta": self.eta,
            # Lambda saturation. Reported here rather than from the E-step block because these
            # are running totals over the algorithm's life, not per-batch values. The
            # cumulative figure cannot fall once a long saturated stretch is banked, so the EMA
            # is reported alongside it to show whether lambda is saturated *now*.
            "lambda_delta": self._lambda_delta,
            "lambda_episodic_reports": float(self._lambda_reports),
            "lambda_at_cap": self._lam_at_cap,
            "lambda_at_cap_frac": self._lam_at_cap_count / max(self._lam_update_count, 1),
            "lambda_at_cap_frac_ema": (
                self._lam_at_cap_frac_ema if self._lam_at_cap_frac_ema is not None else float("nan")
            ),
        }
        info.update(self._last_actor_info)
        # Sampled-support feasibility of the E-step (empty until the first probe runs).
        info.update(self._last_feasibility)
        # Reward-vs-cost spread in the E-step exponent (empty until the first actor update).
        info.update(self._last_spread)
        # Per-state spread-match factor (empty unless estep_cost_spread_match).
        info.update(self._last_spread_match)
        # Which cost readout the exponent is built from, and its spread relative to the mean.
        info.update(self._last_readout_diag)
        # Hazard-stratified replay composition (empty unless hazard_fraction > 0).
        info.update(self._last_replay_info)
        # Cost-critic representation diagnostics, computed identically for the categorical
        # and quantile arms so the two are comparable (empty until the first cost update).
        info.update(self._last_cost_critic_diag)
        # Only while a curriculum is attached; `lambda_` routes these to SafeRL/.
        if self._external_cost_limit is not None:
            info["lambda_cost_limit"] = float(self._external_cost_limit)
            info["lambda_cost_limit_target"] = float(self.cost_limits[0])
            # limit - J_c, in cost units: > 0 means inside the budget in force.
            info["lambda_cost_headroom"] = float(self._external_cost_limit) * -float(self._lambda_delta)
        return info
