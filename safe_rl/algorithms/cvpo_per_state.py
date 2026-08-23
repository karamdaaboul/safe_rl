"""CVPO with a global E-step temperature and a per-state cost multiplier.

CVPO holds one ``(eta, lambda)`` pair for the whole minibatch and solves for it with SLSQP on the
CPU. ``eta`` is a trust-region size, and wanting it uniform across the update is right -- it is a
regulariser on the whole E-step. ``lambda`` is a tradeoff *rate* between two value scales, and
that rate legitimately varies by state. Splitting them reduces each state to 1-D root-finding in
a monotone function, which vectorises trivially; see :mod:`safe_rl.common.per_state_dual` for the
solver and the convexity argument. This is CLAUDE.md's milestone M5 (chapter section 5.3-A).

Everything structural is inherited. Sampling, the M-step, the cost critic, the CVaR/risk cost
modes, recalibration and the qc_scale machinery are CVPO's; only :meth:`_estep_weights` changes.

**The per-state target is not a tuning knob -- it is what makes the method work.** Handing the
solver the static ``qc_thres`` at every state was measured to produce a *correct but degenerate*
solution: states whose sampled actions cannot reach the target have ``dg/dlambda_b < 0`` for every
lambda, so ``lambda_b -> lambda_max`` is genuinely optimal, and because ``eta`` is shared their
enormous ``-lambda_max * Q_c`` spread drags the batch-mean KL up and forces every *other* state's
multiplier up with it. On this repo's measured cost critic (level 3.12, per-state std across
actions only 0.038 -- codex/cvpo-cost-critic-investigation.md) that pins lambda at the cap at
essentially every state while the constraint is still violated by +1.28 at p90. Asking each state
only for a reduction its trust region can reach,

    d_b = max(q_target, C_now_b - ask_b),   C_now_b = E_{a ~ pi_old}[Q_c(s_b, a)]

brings that to +0.002, removes saturation entirely, and makes one warm-started coordinate sweep
enough. ``dstate_mode="static"`` keeps the failing arm available, because a documented negative
result is worth more than a deleted one.

This is the per-state analogue of CVPO's ``qc_thres_homotopy``, with two deliberate differences,
both forced by the same fact -- a different batch of states arrives every update, so there is no
per-state identity to carry state on:

* **No ratchet.** A per-state ratchet is undefined, and a scalar one applied uniformly is just a
  constant multiplier on ``C_now_b``. The monotone downward pressure comes from the floor instead
  (``d_b >= q_target`` always, with ``qc_target_max_rise`` capping the floor's own drift), and
  ``frac_states_at_floor -> 1`` is the observable that replaces it: the real budget now binds
  everywhere.
* **No ``C_now`` smoothing.** That existed to stop one noisy batch locking a persistent ratchet.
  With no ratchet there is nothing to lock, and ``C_now_b`` is a K-sample mean of a quantity whose
  across-action std is 0.038, i.e. precise to ~0.005 against a level of 3.12.

See ``codex/cvpo-negative-result.md``, ``codex/cvpo-cost-critic-investigation.md`` (whose closing
question -- "whether per-state discrimination would buy anything" -- this exists to answer) and
``codex/cvpo-feasible-threshold-homotopy.md``.
"""

from __future__ import annotations

import torch
from typing import Any

from safe_rl.algorithms.cvpo import CVPO
from safe_rl.common.per_state_dual import estep_weights, reachable_qc_min, solve_per_state_dual
from safe_rl.modules.lambda_head import LambdaHead, lambda_head_loss
from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic


def _validate_per_state_config(
    lambda_mode: str,
    lambda_source: str,
    lambda_lr: float,
    lambda_update: str,
    lambda_kp: float,
    lambda_kd: float,
    lambda_init: float | None,
    rescale_by_lambda: bool,
    qc_thres_homotopy: bool,
    dstate_mode: str,
    dstate_beta: float,
    dstate_beta_mode: str,
    dstate_spread_kappa: float,
    dstate_reachable_rho: float,
    lambda_max_mode: str,
    solver_dtype: str,
    dual_sweeps: int,
) -> None:
    """Reject inherited settings the per-state solve supersedes or is invalidated by.

    These raise rather than being silently ignored: a config that reads ``lambda_mode: grad`` and
    quietly does something else is the worst of the available failure modes. Module-level so
    ``__init__`` stays under the repo's cyclomatic-complexity limit.
    """
    if lambda_mode != "per_state":
        raise ValueError(
            f"CVPOPerState requires lambda_mode='per_state', got {lambda_mode!r}. 'grad' and 'dual' "
            "are CVPO's scalar-multiplier paths and are superseded here."
        )
    if lambda_source != "qspace":
        raise ValueError(
            f"CVPOPerState requires lambda_source='qspace', got {lambda_source!r}. An episodic "
            "PID controller would run alongside the computed lambda_b and both would write self.lam."
        )
    # CLAUDE.md rule 1, verbatim: "if `lambda_lr` appears in a config, you have misread the spec".
    # The dual here is solved to optimality every update; there is nothing for a gain to stabilise.
    if float(lambda_lr) != 0.0 or lambda_update != "sgd" or float(lambda_kp) != 0.0 or float(lambda_kd) != 0.0:
        raise ValueError(
            "CVPOPerState computes lambda, it does not learn it: lambda_lr, lambda_kp and lambda_kd "
            f"must be 0 and lambda_update must be 'sgd', got lr={lambda_lr} kp={lambda_kp} "
            f"kd={lambda_kd} update={lambda_update!r}."
        )
    if lambda_init is not None:
        raise ValueError(
            "lambda_init is meaningless here: bisection runs over a fixed bracket every update, so "
            "there is no lambda to warm-start and no lambda state to carry."
        )
    if rescale_by_lambda:
        # Not a style objection. With s = (Q_r - lam*Q_c)/(1+lam), ds/dlam = -(Q_c + Q_r)/(1+lam)^2,
        # whose sign depends on Q_r -- so E_q[Q_c] is no longer monotone in lambda (measured
        # non-monotone at 3 of 64 states) and bisection would silently return an arbitrary point
        # on a non-monotone curve.
        raise ValueError(
            "rescale_by_lambda breaks the monotonicity of E_q[Q_c] in lambda that the per-state "
            "bisection roots on; it would return an arbitrary point rather than the KKT solution."
        )
    if qc_thres_homotopy:
        raise ValueError(
            "qc_thres_homotopy is CVPO's scalar reachable-threshold scheme; the per-state target "
            "(dstate_mode) replaces it. Running both computes q_target through two paths."
        )
    if dstate_mode not in ("beta", "static"):
        raise ValueError(f"dstate_mode must be 'beta' or 'static', got {dstate_mode!r}.")
    if dstate_beta_mode not in ("reachable", "fixed", "spread"):
        raise ValueError(f"dstate_beta_mode must be 'reachable', 'fixed' or 'spread', got {dstate_beta_mode!r}.")
    if not 0.0 < float(dstate_reachable_rho) < 1.0:
        # rho >= 1 asks for the whole KL-reachable drop, which needs lambda -> inf: every
        # state would pin at lambda_max by construction.
        raise ValueError(f"dstate_reachable_rho must be in (0, 1), got {dstate_reachable_rho}.")
    if not 0.0 <= float(dstate_beta) < 1.0:
        raise ValueError(f"dstate_beta must be in [0, 1), got {dstate_beta}.")
    if float(dstate_spread_kappa) < 0.0:
        raise ValueError(f"dstate_spread_kappa must be >= 0, got {dstate_spread_kappa}.")
    if lambda_max_mode not in ("fixed", "balanced"):
        raise ValueError(f"lambda_max_mode must be 'fixed' or 'balanced', got {lambda_max_mode!r}.")
    if solver_dtype not in ("float32", "float64"):
        raise ValueError(f"solver_dtype must be 'float32' or 'float64', got {solver_dtype!r}.")
    if int(dual_sweeps) < 1:
        raise ValueError(f"dual_sweeps must be >= 1, got {dual_sweeps}.")


class CVPOPerState(CVPO):
    """CVPO whose E-step multiplier is solved once per state, by batched bisection on device."""

    policy: SafeSACActorCritic

    def __init__(
        self,
        policy: SafeSACActorCritic,
        # --- per-state dual solver ---
        dual_sweeps: int = 1,  # coordinate-descent sweeps; 1 suffices from a warm eta
        lambda_bisect_iters: int = 30,
        eta_bisect_iters: int = 30,
        eta_min: float = 1e-3,  # fixed bisection bracket: a data-dependent one would resync
        eta_max: float = 1e4,
        solver_dtype: str = "float32",  # float64 is for the unit tests, not for training
        per_state_eta: bool = False,  # KL_b = eps at every state instead of on average (5.3 tier up)
        # --- lambda_max, sized from the E-step spread ratio ---
        lambda_max_mode: str = "fixed",  # "fixed" | "balanced" (measure, then freeze)
        lambda_max_balanced_mult: float = 1.5,
        lambda_max_warmup_updates: int = 200,
        lambda_max_clamp: tuple = (0.5, 100.0),
        # --- per-state target d_b ---
        dstate_mode: str = "beta",  # "beta" = reachable ask | "static" = the documented failure arm
        dstate_beta_mode: str = "reachable",  # fraction of the KL-reachable drop; see _per_state_target
        dstate_reachable_rho: float = 0.25,
        dstate_spread_kappa: float = 0.25,  # dstate_beta_mode="spread" only
        dstate_beta: float = 0.003,  # ask as a fraction of C_now_b  (dstate_beta_mode="fixed" only)
        # --- dispersion gate: where lambda_b is not identifiable, fall back to a batch-level lambda ---
        lambda_gate_mode: str = "none",  # "none" | "spread" | "dispersion" | "both"
        gate_spread_quantile: float = 0.5,  # gate states at/above this quantile of std_k(Q_c)
        gate_dispersion_quantile: float = 0.5,  # same, on the cost distribution's own std
        # --- amortized lambda_psi(s), trained by regression to the bisection's KKT solution ---
        lambda_head_mode: str = "off",  # "off" | "observer" (train, don't use) | "amortized"
        lambda_head_hidden_dims: tuple | list = (64, 64),
        lambda_head_lr: float = 1e-3,  # trains a REGRESSOR, not the dual -- see lambda_head.py
        # --- diagnostics ---
        diag_interval: int = 1,  # updates between diagnostic reductions; each costs one transfer
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        _validate_per_state_config(
            lambda_mode=kwargs.get("lambda_mode", "per_state"),
            lambda_source=kwargs.get("lambda_source", "qspace"),
            lambda_lr=kwargs.get("lambda_lr", 0.0),
            lambda_update=kwargs.get("lambda_update", "sgd"),
            lambda_kp=kwargs.get("lambda_kp", 0.0),
            lambda_kd=kwargs.get("lambda_kd", 0.0),
            lambda_init=kwargs.get("lambda_init", None),
            rescale_by_lambda=kwargs.get("rescale_by_lambda", False),
            qc_thres_homotopy=kwargs.get("qc_thres_homotopy", False),
            dstate_mode=dstate_mode,
            dstate_beta=dstate_beta,
            dstate_beta_mode=dstate_beta_mode,
            dstate_spread_kappa=dstate_spread_kappa,
            dstate_reachable_rho=dstate_reachable_rho,
            lambda_max_mode=lambda_max_mode,
            solver_dtype=solver_dtype,
            dual_sweeps=dual_sweeps,
        )
        # The parent validates against its own vocabulary; hand it an inert scalar configuration.
        # Both scalar paths raise if anything ever reaches them (see _solve_dual / _update_lambda).
        kwargs.pop("lambda_mode", None)
        kwargs.pop("lambda_init", None)
        kwargs["lambda_lr"] = 0.0
        super().__init__(policy, device=device, lambda_mode="grad", **kwargs)

        self.dual_sweeps = int(dual_sweeps)
        self.lambda_bisect_iters = int(lambda_bisect_iters)
        self.eta_bisect_iters = int(eta_bisect_iters)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.solver_dtype = torch.float64 if solver_dtype == "float64" else torch.float32
        self.per_state_eta = bool(per_state_eta)

        self.lambda_max_mode = lambda_max_mode
        self.lambda_max_balanced_mult = float(lambda_max_balanced_mult)
        self.lambda_max_warmup_updates = int(lambda_max_warmup_updates)
        self.lambda_max_clamp = (float(lambda_max_clamp[0]), float(lambda_max_clamp[1]))
        # EMA of median_b std_k(Q_r)/std_k(Q_c) -- the multiplier at which the reward and cost terms
        # contribute equal spread to the E-step exponent. Only the spread across candidate actions
        # survives the per-state softmax, so this, not the value scale, is what sizes lambda_max.
        # Measured 1.86-2.35 on SafetyPointGoal1 (codex/cvpo-feasible-threshold-homotopy.md).
        self._lambda_balanced_ema: float | None = None
        self._lambda_max_used = float(self.lambda_max)
        self._lambda_max_frozen = lambda_max_mode == "fixed"

        self.dstate_mode = dstate_mode
        self.dstate_beta = float(dstate_beta)
        self.dstate_beta_mode = dstate_beta_mode
        self.dstate_spread_kappa = float(dstate_spread_kappa)
        self.dstate_reachable_rho = float(dstate_reachable_rho)

        if lambda_gate_mode not in ("none", "spread", "dispersion", "both"):
            raise ValueError(
                f"lambda_gate_mode must be 'none', 'spread', 'dispersion' or 'both', got {lambda_gate_mode!r}."
            )
        for name, q in (
            ("gate_spread_quantile", gate_spread_quantile),
            ("gate_dispersion_quantile", gate_dispersion_quantile),
        ):
            if not 0.0 <= float(q) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {q}.")
        if lambda_gate_mode in ("dispersion", "both") and not getattr(
            self.policy, "is_distributional_cost_critic", False
        ):
            raise ValueError(
                f"lambda_gate_mode={lambda_gate_mode!r} reads the cost return distribution's spread; "
                "it requires a distributional cost critic (policy cost_critic_type='distributional')."
            )
        self.lambda_gate_mode = lambda_gate_mode
        self.gate_spread_quantile = float(gate_spread_quantile)
        self.gate_dispersion_quantile = float(gate_dispersion_quantile)

        # Amortized lambda_psi(s). "observer" is the default when enabled at all, and deliberately:
        # the head trains while the E-step keeps using the exact bisection, so its regression error
        # is measured before it is ever allowed to drive the policy.
        if lambda_head_mode not in ("off", "observer", "amortized"):
            raise ValueError(f"lambda_head_mode must be 'off', 'observer' or 'amortized', got {lambda_head_mode!r}.")
        self.lambda_head_mode = lambda_head_mode
        self.lambda_head: LambdaHead | None = None
        self.lambda_head_optimizer: torch.optim.Optimizer | None = None
        self._last_head_info: dict[str, float] = {}
        self._head_batch: tuple | None = None
        if lambda_head_mode != "off":
            self.lambda_head = LambdaHead(
                num_obs=self.policy.num_critic_obs,
                lambda_max=self._lambda_max_used,
                hidden_dims=list(lambda_head_hidden_dims),
            ).to(self.device)
            # This optimizer descends a SUPERVISED REGRESSION loss onto KKT targets the convex
            # solve already produced. It never touches the dual. See safe_rl/modules/lambda_head.py.
            self.lambda_head_optimizer = torch.optim.Adam(self.lambda_head.parameters(), lr=float(lambda_head_lr))

        self.diag_interval = max(int(diag_interval), 1)
        self._estep_calls = 0
        self._host_syncs = 0  # per E-step; asserted not to scale with B or K in the tests

        # lambda is computed, never carried: start it at zero rather than at CVPO's default 1.0,
        # so a run that somehow never solves is visibly inert rather than silently penalised.
        self.lam = 0.0
        self._lambda_ctrl.lam = 0.0
        self._lambda_ctrl.integral = 0.0
        # Warm start for sweep 1's lambda block. Kept as a tensor so the solve never synchronises;
        # self.eta (the float MPO logs) is refreshed from it in the diagnostics reduction.
        self._eta_t = torch.tensor(float(self.eta), dtype=self.solver_dtype, device=self.device)
        print(
            "CVPOPerState: global eta, per-state lambda by batched bisection "
            f"({self.dual_sweeps} sweep(s), {self.lambda_bisect_iters}/{self.eta_bisect_iters} iters, "
            f"{solver_dtype}); target d_b mode={self.dstate_mode}"
            + (f" ask={self.dstate_beta_mode}" if self.dstate_mode == "beta" else "")
            + f"; lambda_max={self._lambda_max_used:.3f} ({self.lambda_max_mode})"
        )

    # -- paths the per-state solve supersedes -------------------------------------------------

    def _solve_dual(self, q_np, qc_np):  # noqa: D401
        """Unreachable: :meth:`_estep_weights` never calls it. Raising keeps it that way."""
        raise RuntimeError(
            "CVPOPerState does not use CVPO's scalar SLSQP dual; the per-state solve replaces it. "
            "If you reached this, _estep_weights was bypassed."
        )

    def _update_lambda(self, eqc):  # noqa: D401
        """Unreachable: lambda is computed by bisection, not integrated by a controller."""
        raise RuntimeError(
            "CVPOPerState computes lambda per state in the E-step; the LambdaController path is "
            "not part of this algorithm (CLAUDE.md rule 1)."
        )

    # -- lambda_max and the per-state target ---------------------------------------------------

    def _current_lambda_max(self) -> float:
        """The cap in force this update.

        ``"balanced"`` measures ``lambda_balanced`` for ``lambda_max_warmup_updates`` and then
        **freezes**, for the reason ``use_measured_qc_scale`` freezes (cvpo.py): a continuously
        moving bound would make the saturation fraction reflect the bound's motion rather than the
        policy's, and ``lambda_frac_at_cap`` incomparable across time. The value it reads is the
        previous update's EMA, so sizing the cap never costs a synchronisation of its own.
        """
        if self._lambda_max_frozen or self._lambda_balanced_ema is None:
            return self._lambda_max_used
        lo, hi = self.lambda_max_clamp
        self._lambda_max_used = float(min(max(self.lambda_max_balanced_mult * self._lambda_balanced_ema, lo), hi))
        if self._estep_calls >= self.lambda_max_warmup_updates:
            self._lambda_max_frozen = True
            print(
                f"CVPOPerState lambda_max frozen at {self._lambda_max_used:.4f} "
                f"({self.lambda_max_balanced_mult} x measured lambda_balanced "
                f"{self._lambda_balanced_ema:.4f}) after {self._estep_calls} updates."
            )
        return self._lambda_max_used

    def _per_state_target(self, q_c: torch.Tensor, q_target: float, reachable=None) -> torch.Tensor:
        """``d_b``, the cost target this update's E-step holds each state to. Shape ``[B]``.

        ``clamp_min(., q_target)`` is the per-state analogue of CVPO's ``max(q_target, raw)``: the
        real budget always wins, and it is applied outside any smoothing for the reason documented
        there.

        **Ask only for a reduction the trust region can actually deliver.** Ask for more and every
        state is infeasible, ``lambda_b`` pins at the cap, and you have reproduced the static
        target's degeneracy through a different door. The three modes differ in how much
        calibration that sizing needs:

        * ``"reachable"`` (default) -- ``ask_b = rho * (C_now_b - reachable_qc_min_b)``, where the
          floor is the smallest ``E_{q_b}[Q_c]`` attainable *inside the KL budget*. This asks for a
          fixed fraction of a provably attainable reduction, so it **cannot** be mis-sized and needs
          no knowledge of the critic's level or spread at all. ``rho`` must stay well under 1: the
          floor is what the KL budget alone permits, while reaching it also needs ``lambda -> inf``,
          so ``lambda_max`` binds first.
        * ``"spread"`` -- ``ask_b = kappa * std_k(Q_c)``. Scale-free in the level, but not in eps.
        * ``"fixed"`` -- ``ask_b = beta * C_now_b``. Needs re-deriving whenever the level moves,
          which is the failure already paid for in codex/cvpo-qc-threshold-calibration.md.

        Measured across three regimes (this repo's critic statistics; a live 30-iteration early
        run; and a large-action-spread fixture), the reachable drop was **0.44 sigma in all three**
        -- it is set by ``eps``, not by the critic. So a sigma-sized ask is implicitly a fraction of
        reachability, and a badly chosen one:

            mode                  fixture A        live regime B     spread regime C
            kappa = 0.25 sigma    lam 0.36, ok     lam 2.65, 33% AT CAP   lam 1.56, ok
            rho   = 0.50          lam 0.30, ok     lam 2.25, 13% AT CAP   lam 1.30, ok
            rho   = 0.25          lam 0.14, ok     lam 1.00, ok           lam 0.56, ok

        Hence ``rho = 0.25``. Watch ``dstate_ask_over_spread``: negative means the floor is binding
        (the policy is already inside the budget and nothing is being asked of it), not that the ask
        is inverted.
        """
        if self.dstate_mode == "static":
            return torch.full((q_c.shape[1],), self._effective_thres(), dtype=q_c.dtype, device=q_c.device)
        c_now_b = q_c.mean(dim=0)
        if self.dstate_beta_mode == "reachable":
            floor = reachable if reachable is not None else reachable_qc_min(q_c, self.eps_dual)
            ask = self.dstate_reachable_rho * (c_now_b - floor).clamp_min(0.0)
        elif self.dstate_beta_mode == "spread":
            ask = self.dstate_spread_kappa * q_c.std(dim=0)
        else:
            ask = self.dstate_beta * c_now_b
        return torch.clamp_min(c_now_b - ask, q_target)

    def _cost_dispersion(self, critic_obs_exp: torch.Tensor, actions_flat: torch.Tensor, n: int, b: int):
        """Per-state std of the cost RETURN distribution, ``[B]``, averaged over the K candidates.

        Distinct from ``std_k(Q_c)``, and the distinction matters. This is the aleatoric spread of
        ``Z_c`` at a single ``(s, a)`` -- "how uncertain is the outcome here". ``std_k(Q_c)`` is the
        spread *across actions* at a state -- "does the multiplier have anything to act on". Only
        the second decides whether ``lambda_b`` is identifiable, so it is the primary gate; this one
        is a complementary signal about whether the critic is saying anything at all at this state,
        not a substitute for it.
        """
        critic = self.policy.cost_critics[0]
        normalizer = self.policy.critic_obs_normalizer
        obs_n = normalizer.normalize(critic_obs_exp) if hasattr(normalizer, "normalize") else normalizer(critic_obs_exp)
        dist = critic.get_dist(critic(obs_n, actions_flat))
        return critic.get_var(dist).clamp_min(0.0).sqrt().reshape(n, b).mean(dim=0)

    def _gate_mask(self, q_c: torch.Tensor, dispersion: torch.Tensor | None):
        """``[B]`` bool: True where ``lambda_b`` is solved per state, False where it takes the batch's.

        The thresholds are **batch-relative quantiles, never absolute values in Q_c units**. An
        absolute threshold would have to be re-derived every time the cost critic's level moved,
        which is the failure this repo has already paid for twice (see
        codex/cvpo-qc-threshold-calibration.md and the qc_scale saga).
        """
        if self.lambda_gate_mode == "none":
            return None, None
        spread = q_c.std(dim=0)
        gate = torch.ones_like(spread, dtype=torch.bool)
        thr = torch.zeros((), dtype=q_c.dtype, device=q_c.device)
        if self.lambda_gate_mode in ("spread", "both"):
            thr = torch.quantile(spread, self.gate_spread_quantile)
            gate = gate & (spread >= thr)
        if self.lambda_gate_mode in ("dispersion", "both") and dispersion is not None:
            disp = dispersion.to(q_c.dtype)
            gate = gate & (disp >= torch.quantile(disp, self.gate_dispersion_quantile))
        return gate, thr

    # -- the E-step ----------------------------------------------------------------------------

    def _estep_weights(self, q: torch.Tensor, actions: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Weights ``exp((Q_r - lambda_b Q_c)/eta)``, with ``lambda_b`` solved per state."""
        n, batch_size = q.shape
        cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
        act_flat = actions.reshape(n * batch_size, -1)
        qc_flat = self._estep_cost(cobs_exp, act_flat, self.estep_use_target_critic)
        qc = qc_flat[:, 0].reshape(n, batch_size)

        q_r = q.to(self.solver_dtype)
        q_c = qc.to(self.solver_dtype)

        dispersion = (
            self._cost_dispersion(cobs_exp, act_flat, n, batch_size)
            if self.lambda_gate_mode in ("dispersion", "both")
            else None
        )
        gate, gate_thr = self._gate_mask(q_c, dispersion)

        # One scalar synchronisation per update, and the only one before the solve: the inherited
        # q_target machinery is Python-side EMA state over hundreds of updates, so its numerator
        # has to reach the host. It does not scale with B or K, and it replaces CVPO's transfer of
        # two [N, B] arrays plus a SciPy call.
        c_now_mean = float(q_c.mean())
        self._host_syncs = 1
        q_target = self._update_homotopy_threshold(c_now_mean)

        need_reach = self.dstate_mode == "beta" and self.dstate_beta_mode == "reachable"
        reachable = reachable_qc_min(q_c, self.eps_dual) if need_reach else None
        d = self._per_state_target(q_c, q_target, reachable)
        lam_max = self._current_lambda_max()
        sol = solve_per_state_dual(
            q_r,
            q_c,
            d,
            eps=self.eps_dual,
            lam_max=lam_max,
            eta_init=self._eta_t,
            sweeps=self.dual_sweeps,
            lam_iters=self.lambda_bisect_iters,
            eta_iters=self.eta_bisect_iters,
            eta_min=self.eta_min,
            eta_max=self.eta_max,
            per_state_eta=self.per_state_eta,
            gate=gate,
        )
        self._eta_t = sol.eta.detach()

        weights = sol.weights
        if self.lambda_head is not None:
            # Stash detached regression targets; the optimizer step happens in
            # _update_actor_and_alpha, OUTSIDE the no_grad block the E-step runs under.
            self._head_batch = (critic_obs.detach(), sol.lam.detach(), sol.lam_inactive, sol.lam_at_cap)
            if self.lambda_head_mode == "amortized":
                # The head supplies lambda, but the bisection above still ran -- it produced the
                # targets and every KKT diagnostic, so exact and amortized stay directly comparable.
                lam_hat = self.lambda_head.predict(critic_obs.detach()).to(q_c.dtype)
                weights = estep_weights(q_r, q_c, sol.eta, lam_hat)

        self._estep_calls += 1
        if self._estep_calls % self.diag_interval == 0:
            self._reduce_diagnostics(sol, q_r, q_c, d, q_target, c_now_mean, lam_max, gate, gate_thr, reachable)
        return weights.to(q.dtype)

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None):
        """MPO's E/M steps, then the amortized head's regression step.

        The head is trained *here* rather than inside :meth:`_estep_weights` for a concrete
        reason: MPO wraps the whole E-step in ``torch.no_grad()``, so a loss built there has no
        graph to backpropagate. Training it outside also makes the separation legible -- the
        E-step is a solve, this is a regression, and they do not share a gradient.
        """
        out = super()._update_actor_and_alpha(obs, critic_obs)
        if self.lambda_head is not None and self._head_batch is not None:
            self._last_actor_info.update(self._train_lambda_head())
        return out

    def _train_lambda_head(self) -> dict[str, float]:
        """One supervised step onto this batch's KKT solution. Returns its diagnostics.

        The gradient reaches the head and nothing else. It cannot reach ``Q_c``: the targets come
        from a solve that ran under ``no_grad`` and are detached, and the input observation is
        detached. That separation is the whole point -- the safety signal still reaches the policy
        through sample weights, never through a gradient chained into a critic (CLAUDE.md rule 3).
        """
        critic_obs, lam, inactive, at_cap = self._head_batch
        self._head_batch = None
        self.lambda_head.set_lambda_max(self._lambda_max_used)
        dtype = next(self.lambda_head.parameters()).dtype
        loss, diag = lambda_head_loss(self.lambda_head, critic_obs.to(dtype), lam.to(dtype), inactive, at_cap)
        self.lambda_head_optimizer.zero_grad()
        loss.backward()
        self.lambda_head_optimizer.step()
        self._last_head_info = diag
        return diag

    # -- diagnostics ---------------------------------------------------------------------------

    def _reduce_diagnostics(
        self, sol, q_r, q_c, d, q_target, c_now_mean, lam_max, gate=None, gate_thr=None, reachable=None
    ) -> None:
        """Reduce every per-state statistic to one device tensor, then transfer it once.

        This shape is not incidental. Thirty ``torch.quantile(...).item()`` calls would be thirty
        synchronisations per update -- strictly worse than the single ``.cpu().numpy()`` the
        per-state solver was written to remove. Add statistics here, never ``.item()`` at the call
        site.
        """
        lam, kl, ess, eqc = sol.lam, sol.kl, sol.ess, sol.eqc
        # Interior for the KKT gate means "this state's own root, strictly inside the box". An
        # UNGATED state carries the batch-level lambda, which satisfies the batch constraint and
        # has no reason to satisfy that state's stationarity -- counting it here would make the
        # headline KKT residual read as broken the moment the gate is switched on.
        interior = ~(sol.lam_inactive | sol.lam_at_cap)
        if gate is not None:
            interior = interior & gate
        residual = d - eqc  # dg/dlambda_b
        qs = torch.tensor([0.1, 0.5, 0.9], dtype=lam.dtype, device=lam.device)

        # Only the spread across candidate actions survives the per-state softmax; a constant added
        # to Q_c cancels in the normalisation. lambda_balanced is where lambda weighs the two equally.
        std_qr, std_qc = q_r.std(dim=0), q_c.std(dim=0)
        lam_balanced = (std_qr / std_qc.clamp_min(1e-12)).median()

        feasible_now = self.feasibility_probe_interval > 0 and (
            self._feas_probe_count % self.feasibility_probe_interval == 0
        )
        self._feas_probe_count += 1
        # The target may already have paid for this bisection; do not run it twice.
        if reachable is not None:
            reach, feasible_now = reachable, True
        else:
            reach = reachable_qc_min(q_c, self.eps_dual) if feasible_now else torch.full_like(eqc, float("nan"))

        # Order must match _DIAG_NAMES exactly.
        packed = torch.stack(
            [
                sol.eta.mean(),
                sol.eta_at_bound.to(lam.dtype).mean(),
                lam.mean(),
                lam.median(),
                lam.min(),
                lam.max(),
                lam.std(),
                torch.quantile(lam, qs[0]),
                torch.quantile(lam, qs[2]),
                sol.lam_inactive.to(lam.dtype).mean(),
                sol.lam_at_cap.to(lam.dtype).mean(),
                lam.median() / lam_balanced.clamp_min(1e-12),
                residual.mean(),
                torch.where(interior, residual.abs(), torch.zeros_like(residual)).max(),
                _kkt_residual_max(residual, sol.lam_inactive, sol.lam_at_cap),
                # viol vs d_b: ~0 by construction once the target is reachable -- a solver check only.
                torch.quantile(eqc - d, qs[1]),
                torch.quantile(eqc - d, qs[2]),
                (eqc - d).max(),
                # viol vs q_target: the number that actually means something.
                torch.quantile(eqc - q_target, qs[1]),
                torch.quantile(eqc - q_target, qs[2]),
                (eqc - q_target).max(),
                (eqc > q_target).to(lam.dtype).mean(),
                (d <= q_target + 1e-12).to(lam.dtype).mean(),
                (reach > d).to(lam.dtype).mean(),
                ess.mean(),
                ess.min(),
                torch.quantile(ess, qs[0]),
                ess.median(),  # noqa: E501 -- ess_solver
                (ess < 4.0).to(lam.dtype).mean(),
                torch.quantile(kl, qs[0]),
                kl.median(),
                torch.quantile(kl, qs[2]),
                kl.max(),
                (kl > self.eps_dual).to(lam.dtype).mean(),
                torch.quantile(kl, qs[2]) / torch.quantile(kl, qs[0]).clamp_min(1e-12),
                torch.quantile(d, qs[0]),
                d.median(),
                torch.quantile(d, qs[2]),
                d.mean(),
                ((q_c.mean(dim=0) - d) / std_qc.clamp_min(1e-12)).median(),
                eqc.mean(),
                std_qr.median(),
                std_qc.median(),
                lam_balanced,
                (lam.median() * std_qc.median() / std_qr.median().clamp_min(1e-12)),
                self.eps_dual - kl.mean(),
                # -- dispersion gate. All-True mask and a NaN shared lambda when the gate is off,
                #    so the keys stay present and a disabled gate is visibly disabled rather than
                #    silently reading as "everything gated at lambda 0".
                _gate_frac(gate, lam),
                sol.lam_shared if sol.lam_shared is not None else _nan_like(lam),
                gate_thr if gate_thr is not None else _nan_like(lam),
                _masked_mean(ess, gate),
                _masked_mean(ess, _invert(gate, ess)),
                _masked_mean(eqc - d, gate),
                _masked_mean(eqc - d, _invert(gate, eqc)),
            ]
        )
        assert packed.numel() == len(_DIAG_NAMES), (
            f"diagnostic tensor has {packed.numel()} entries but _DIAG_NAMES has {len(_DIAG_NAMES)}; "
            "the two are positional and must be edited together."
        )
        vals = [float(v) for v in packed.cpu()]  # the one transfer
        info = dict(zip(_DIAG_NAMES, vals))
        # The lambda-cap certificate (chapter 6.1) under the name the spec uses. It is a different
        # quantity from frac_infeasible_support -- one says "the multiplier ran out of room", the
        # other "no reweighting of these candidates could have reached the target" -- and they
        # will disagree, which is informative rather than a bug.
        info["frac_infeasible_lambda_cap"] = info["lambda_frac_at_cap"]
        if not feasible_now:
            info["frac_infeasible_support"] = float("nan")

        self.eta = info["eta_star"]
        self.lam = info["lambda_mean"]
        self._eqc = info["eqc"]
        self._solver_iters = float(self.dual_sweeps * (self.lambda_bisect_iters + self.eta_bisect_iters))
        self._solver_status = float(
            1.0 if info["eta_at_bound_frac"] > 0.0 else (2.0 if info["lambda_frac_at_cap"] > 0.05 else 0.0)
        )
        self._track_lambda_saturation_per_state(info["lambda_frac_at_cap"])

        if self._lambda_balanced_ema is None:
            self._lambda_balanced_ema = info["lambda_balanced"]
        else:
            self._lambda_balanced_ema += 0.01 * (info["lambda_balanced"] - self._lambda_balanced_ema)

        if feasible_now:
            self._last_feasibility = {
                "estep_feasible": 1.0 - info["frac_infeasible_support"],
                "frac_infeasible_support": info["frac_infeasible_support"],
            }
        self._last_spread = {
            "estep_std_qr": info["estep_std_qr"],
            "estep_std_qc": info["estep_std_qc"],
            "lambda_balanced": info["lambda_balanced"],
            "estep_spread_ratio": info["estep_spread_ratio"],
            "lambda_over_balanced": info["lambda_over_balanced"],
        }
        info.update(
            {
                "solver_sweeps": float(self.dual_sweeps),
                "solver_iters": self._solver_iters,
                "solver_status": self._solver_status,
                "lambda_max_used": lam_max,
                "qc_thres_eff": self._effective_thres(),
                "qc_thres_target": q_target,
                "qc_thres_static": self.qc_thres,
                "c_now": c_now_mean,
                "c_now_over_thres": c_now_mean / max(q_target, 1e-12),
                "eqc_as_episodic_cost": info["eqc"] / max(self._qc_scale, 1e-12),
                # CVPO's key name, kept so existing dashboards keep resolving. Here it is the
                # mean over states of d_b - E_{q_b}[Q_c], not a single batch-level residual.
                "dual_residual_lambda": info["dual_residual_lambda_mean"],
                "estep_host_syncs": float(self._host_syncs),
            }
        )
        self._last_estep_info = info

    def _track_lambda_saturation_per_state(self, frac_at_cap: float) -> None:
        """Saturation tracking, with ``at_cap`` now a *fraction of states*, not a 0/1 flag.

        The counter names are CVPO's on purpose, so dashboards and comparisons against the scalar
        arms keep working -- but read them accordingly: ``lambda_at_cap`` is the fraction of states
        pinned this update, and ``lambda_at_cap_frac`` is its running mean over updates.
        """
        self._lam_at_cap = float(frac_at_cap)
        self._lam_update_count += 1
        self._lam_at_cap_count += float(frac_at_cap)
        if self._lam_at_cap_frac_ema is None:
            self._lam_at_cap_frac_ema = float(frac_at_cap)
        else:
            self._lam_at_cap_frac_ema += 0.01 * (float(frac_at_cap) - self._lam_at_cap_frac_ema)

    def _track_lambda_saturation(self) -> None:  # noqa: D401
        """Disabled: the scalar version reads ``self.lam``, which is a batch mean here."""
        raise RuntimeError("CVPOPerState uses _track_lambda_saturation_per_state; lambda is a vector.")

    def get_penalty_info(self) -> dict[str, Any]:
        """CVPO's logging surface, with the multiplier keys made per-state.

        Set explicitly rather than relying on ``self._last_estep_info`` shadowing them through the
        parent's ``info.update(self._last_actor_info)``: that ordering happens to work today and
        would be silent if it stopped.
        """
        info = super().get_penalty_info()
        e = self._last_estep_info
        if e:
            info.update(
                {
                    "lambda_mean": e["lambda_mean"],
                    "lambda_median": e["lambda_median"],
                    "lambda_min": e["lambda_min"],
                    "lambda_max": e["lambda_max"],  # per-state max; the *bound* is lambda_max_used
                    "lambda_list": [e["lambda_median"]],
                    "lambda_max_used": e["lambda_max_used"],
                }
            )
        return info


def _nan_like(x: torch.Tensor) -> torch.Tensor:
    return torch.full((), float("nan"), dtype=x.dtype, device=x.device)


def _invert(gate: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """The ungated states. With no gate there are none, so this is all-False, not all-True --
    otherwise ``ess_ungated`` would report a real number for a population that does not exist."""
    return torch.zeros_like(like, dtype=torch.bool) if gate is None else ~gate


def _gate_frac(gate: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """Fraction of states solved per-state. 1.0 when no gate is in use -- which is the truth."""
    if gate is None:
        return torch.ones((), dtype=like.dtype, device=like.device)
    return gate.to(like.dtype).mean()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``x`` over ``mask``; NaN if the mask is empty, so "no such states" never reads as 0."""
    if mask is None:
        return x.mean()
    n = mask.sum()
    return torch.where(n > 0, (x * mask.to(x.dtype)).sum() / n.clamp_min(1).to(x.dtype), _nan_like(x))


def _kkt_residual_max(residual: torch.Tensor, inactive: torch.Tensor, at_cap: torch.Tensor) -> torch.Tensor:
    """Box-projected KKT residual, ``max_b |.|``.

    ``dg/dlambda_b`` only has to vanish at *interior* states. At ``lambda_b = 0`` optimality needs
    it non-negative, at ``lambda_b = lambda_max`` non-positive; the projection is what turns those
    one-sided conditions into a single number that is zero exactly at the optimum.
    """
    proj = torch.where(inactive, residual.clamp_max(0.0), residual)
    proj = torch.where(at_cap, residual.clamp_min(0.0), proj)
    return proj.abs().max()


_DIAG_NAMES = (
    "eta_star",
    "eta_at_bound_frac",
    "lambda_mean",
    "lambda_median",
    "lambda_min",
    "lambda_max",
    "lambda_std",
    "lambda_p10",
    "lambda_p90",
    "lambda_frac_zero",
    "lambda_frac_at_cap",
    "lambda_over_balanced",
    "dual_residual_lambda_mean",
    "dual_residual_lambda_interior_absmax",
    "kkt_residual_max",
    "viol_vs_dstate_p50",
    "viol_vs_dstate_p90",
    "viol_vs_dstate_max",
    "viol_vs_qtarget_p50",
    "viol_vs_qtarget_p90",
    "viol_vs_qtarget_max",
    "frac_states_violating",
    "frac_states_at_floor",
    "frac_infeasible_support",
    # `ess_solver`, not `ess`: MPO already logs `ess` from the returned weights and the two must
    # stay independently visible rather than one silently shadowing the other.
    "ess_solver",
    "ess_min_solver",
    "ess_p10",
    "ess_median",
    "ess_frac_below_4",
    "kl_q_p10",
    "kl_q_p50",
    "kl_q_p90",
    "kl_q_max",
    "kl_q_frac_over_eps",
    "kl_dispersion_ratio",
    "dstate_p10",
    "dstate_p50",
    "dstate_p90",
    "dstate_mean",
    "dstate_ask_over_spread",
    "eqc",
    "estep_std_qr",
    "estep_std_qc",
    "lambda_balanced",
    "estep_spread_ratio",
    "dual_residual_eta_solver",
    "frac_states_gated",
    "lambda_shared",
    "gate_threshold",
    "ess_gated",
    "ess_ungated",
    "viol_gated_mean",
    "viol_ungated_mean",
)
