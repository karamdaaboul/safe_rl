"""FH-DCMPO — Finite-Horizon Distributional Constrained MPO.

CVPO's constrained variational E-step, but the cost channel is moved into the units the
benchmark actually constrains. Three changes, and nothing else:

1. **Finite-horizon, undiscounted cost critic.** ``Z_c`` becomes the distribution of
   ``sum_{t'=t}^{T-1} c_t'`` -- no ``gamma_c``, no ``qc_scale``. Requires the horizon feature that
   :class:`~safe_rl.envs.horizon_augmented_vec_env.HorizonAugmentedVecEnv` appends, since the
   undiscounted cost-to-go is not a function of ``(s, a)`` alone.
2. **A distributional risk statistic inside the E-step**, read off the quantile head:
   ``rho = E[Z_c] + kappa (CVaR_alpha[Z_c] - E[Z_c])``, with ``kappa`` ramped from 0.
3. **The threshold is the cost limit**, literally: ``qc_thres = cost_limit = 25``.

Why this is not just a units cleanup. Everything the E-step does is compare candidate actions at
one state, so what matters is the *spread* of the cost signal across actions and the *level*
relative to the threshold. Under the old scheme the level came through a scalar ``qc_scale``
measured per task (0.0764 on PointGoal1, 0.09198 on CarGoal1); getting it wrong moved the real
budget by 20-31% silently (``codex/qr-dmpo-math.md`` section 3), and a *tail* statistic inherits
that error amplified -- ``codex/why-mean-beat-cvar-on-pointgoal1.md`` measured ``CVaR_0.9`` sitting
4.48x above a mean-calibrated threshold, which pinned ``lambda`` at its cap for 13k of 15k
iterations and turned the E-step into "minimise cost, ignore reward". Here the statistic and the
threshold are the same quantity in the same units, so there is nothing left to mis-scale.

**The claim this algorithm exists to test**, registered before any run: if the constraint
``CVaR_alpha[J_c] <= d`` is actually enforced then ``VaR_alpha <= d``, hence
``P(J_c > d) <= 1 - alpha``. At ``alpha = 0.9`` that is at most 10% of episodes over budget,
against 32-46% measured for every method in the repo. The shortfall from 10% is exactly the sum of
critic error, E-step infeasibility and M-step projection slack -- all three already logged.

See ``codex/fh-dcmpo-math.md`` for the derivation and the bound.
"""

from __future__ import annotations

import torch
from copy import deepcopy
from typing import Any

from safe_rl.common.fh_cost import conservatism_statistic, evt_conservatism_statistic, kappa_at
from safe_rl.modules.critic import quantile_huber_loss

from .cvpo import CVPO


class FHDCMPO(CVPO):
    """CVPO with a finite-horizon undiscounted distributional cost constraint.

    Args:
        fh_risk_mode: ``"cvar"`` (default) uses the per-state tail mean of the learned quantile
            distribution. ``"evt"`` uses EVO's GPD extreme-quantile offset fitted to the pooled
            batch peaks (arXiv:2601.12008, Eq. 15) -- structurally different in that its tail
            correction is one global scalar rather than per-state. ``"mean"`` pins ``kappa = 0``
            and is the S1 arm: finite horizon, no risk statistic.
        fh_alpha: Tail level ``alpha`` for the CVaR. ``0.9`` constrains the worst 10%.
        fh_kappa: Target ``kappa``. ``1.0`` is pure ``CVaR_alpha``; ``0.0`` the plain mean.
        fh_kappa_warmup: E-step updates held at ``kappa = 0`` before the ramp starts. The cost
            critic must be levelled before a tail is read off it -- reading a tail from an
            under-dispersed critic is the documented failure mode, not a hypothetical one.
        fh_kappa_ramp: E-step updates spent ramping ``kappa`` linearly to ``fh_kappa``.
        fh_evt_mu / fh_evt_nu: EVO's safety-boundary quantile and tail exploitation range.
        cost_horizon: Episode length ``T``. Used only for reporting here -- the finite horizon
            enters through the observation feature and the undiscounted backup, not through a
            scale factor.
    """

    def __init__(
        self,
        policy,
        fh_risk_mode: str = "cvar",
        fh_alpha: float = 0.9,
        fh_kappa: float = 1.0,
        fh_kappa_warmup: int = 5000,
        fh_kappa_ramp: int = 15000,
        fh_evt_mu: float = 0.9,
        fh_evt_nu: float = 0.05,
        cost_n_step: int | None = None,
        cost_td_lambda: float = 0.95,
        cost_boot_ema_tau: float | None = None,
        offpolicy_diag: bool = False,
        offpolicy_diag_interval: int = 8,
        **kwargs: Any,
    ) -> None:
        if fh_risk_mode not in ("mean", "cvar", "evt"):
            raise ValueError(f"fh_risk_mode must be 'mean', 'cvar' or 'evt', got {fh_risk_mode!r}.")
        if not 0.0 <= fh_alpha < 1.0:
            raise ValueError(f"fh_alpha must be in [0, 1), got {fh_alpha}.")
        if fh_kappa < 0.0:
            raise ValueError(f"fh_kappa must be non-negative, got {fh_kappa}.")

        # The cost channel is undiscounted, so the episodic limit needs no conversion. Expressed
        # through the parent's own units mechanism (qc_scale = 1) rather than by reaching into its
        # internals, so the constructor's printout reports the truth: qc_thres == cost_limit.
        kwargs.setdefault("qc_scale_source", "measured")
        if float(kwargs.get("qc_scale_measured", 1.0)) != 1.0:
            raise ValueError(
                "FH-DCMPO does not use qc_scale: the cost critic is undiscounted, so the episodic "
                "limit IS the threshold. Remove qc_scale_measured (or set it to 1.0) and let "
                "qc_thres default to cost_limit."
            )
        kwargs["qc_scale_measured"] = 1.0
        kwargs.setdefault("qc_scale_probe", "not applicable: undiscounted finite-horizon cost critic")
        if kwargs.get("use_measured_qc_scale"):
            raise ValueError(
                "use_measured_qc_scale is meaningless for FH-DCMPO -- it estimates the "
                "discounted-to-episodic ratio, which is 1 by construction here."
            )
        # The parent validates this field against its own modes; we override `_estep_cost`
        # wholesale, so it is told "mean" and the real mode is kept in `self.fh_risk_mode`.
        kwargs["cost_constraint_mode"] = "mean"
        if kwargs.pop("recalibrate_cvar", False):
            raise NotImplementedError(
                "recalibrate_cvar is not wired for the quantile head yet (S3). PIT recalibration "
                "of a quantile critic means re-weighting the equal-mass locations, not remapping "
                "categorical probabilities; see codex/fh-dcmpo-math.md."
            )

        super().__init__(policy, **kwargs)

        self.fh_risk_mode = fh_risk_mode
        self.fh_alpha = float(fh_alpha)
        self.fh_kappa_target = 0.0 if fh_risk_mode == "mean" else float(fh_kappa)
        self.fh_kappa_warmup = int(fh_kappa_warmup)
        self.fh_kappa_ramp = int(fh_kappa_ramp)
        self.fh_evt_mu = float(fh_evt_mu)
        self.fh_evt_nu = float(fh_evt_nu)
        self._fh_updates = 0
        self._fh_kappa = 0.0
        self._last_evt_info: dict[str, float] = {}
        # Diagnostics only: the kappa = 0 (mean) readout, taken from the SAME critic forward pass
        # as the statistic `_estep_cost` actually returns. Lets the E-step report how much extra
        # across-action cost signal the tail statistic carries -- the constraint "dose" -- without
        # a second forward pass, and without any tensor that feeds the weights changing.
        self._last_estep_cost_ref: torch.Tensor | None = None

        # TD(lambda) cost target (SDAC, arXiv:2301.10923). `cost_n_step = None` keeps the plain
        # n-step target, i.e. exactly the behaviour of every arm before this.
        self.cost_n_step = None if cost_n_step is None else max(1, int(cost_n_step))
        self.cost_td_lambda = float(cost_td_lambda)
        if not 0.0 <= self.cost_td_lambda <= 1.0:
            raise ValueError(f"cost_td_lambda must be in [0, 1], got {cost_td_lambda}")
        if self.cost_n_step is not None and self.cost_n_step <= self.n_step:
            raise ValueError(
                f"cost_n_step ({self.cost_n_step}) must exceed n_step ({self.n_step}) to be worth "
                "anything: TD(lambda) over a window no longer than the existing n-step target "
                "reduces to that target at lambda=1 and mixes in SHORTER (less informative) "
                "returns below it, so it would inject less real cost variance, not more."
            )

        # Undiscounted n-step cost aggregation in the replay buffer. Read by
        # `SAC.init_storage` via getattr, so this attribute is the entire plumbing.
        self.cost_gamma = 1.0

        # -- EMA bootstrap policy for the cost target (bootstrap-stability experiment) ----------
        # Motivation, registered before the runs (codex/offpolicy-mismatch-td-lambda.md): the
        # TD(lambda) tail is well calibrated offline with a frozen policy even on maximally
        # off-policy data, and recent-only replay does NOT fix the online failure -- pointing at
        # the NON-STATIONARITY of the bootstrap policy, not at data staleness. This arm slows
        # ONLY the policy that draws the bootstrap action a' ~ pi_boot(s_{t+j}) in the cost
        # target: pi_boot <- (1 - tau_b) pi_boot + tau_b pi_current after every actor update.
        # pi_boot touches NOTHING else -- not environment actions, not the E-step (whose anchor
        # is MPO's own `actor_target`), not the reward critic. `None` (default) is the exact
        # baseline: bootstrap from pi_current, byte-identical to every arm before this.
        # Effective lag ~ 1/tau_b actor updates (= 1/(2 tau_b) iterations at
        # num_updates_per_step=2). NOTE: pi_boot lives on the algorithm, so checkpoints do not
        # carry it; a resumed run re-seeds it from the resumed actor.
        self.cost_boot_ema_tau = None if cost_boot_ema_tau is None else float(cost_boot_ema_tau)
        self.cost_boot_actor = None
        self._last_boot_diag: dict[str, float] = {}
        if self.cost_boot_ema_tau is not None:
            if not 0.0 < self.cost_boot_ema_tau <= 1.0:
                raise ValueError(f"cost_boot_ema_tau must be in (0, 1], got {cost_boot_ema_tau}")
            if self.cost_n_step is None:
                raise ValueError(
                    "cost_boot_ema_tau targets the TD(lambda) window bootstrap; it requires "
                    "cost_n_step (the plain n-step cost path is the parent's and is left alone)."
                )
            self.cost_boot_actor = deepcopy(self.policy.actor).to(self.device)
            for p in self.cost_boot_actor.parameters():
                p.requires_grad = False

        # Off-policy mismatch DIAGNOSTIC (default off; measures, never corrects). When on,
        # `SAC.init_storage` passes cost_window_extras=True to the replay buffer and this class
        # logs, every `offpolicy_diag_interval`-th cost-critic update, how far the current policy
        # has drifted from the behavior policy along the stored TD(lambda) windows
        # (`opd_*` keys in get_penalty_info). Ratios need `runner.store_behavior_logprob: true`
        # in the config; without it only action/age statistics are available.
        self.offpolicy_diag = bool(offpolicy_diag)
        self.offpolicy_diag_interval = max(1, int(offpolicy_diag_interval))
        self._opd_counter = 0
        self._last_opd_diag: dict[str, float] = {}
        if self.offpolicy_diag and self.cost_n_step is None:
            raise ValueError("offpolicy_diag measures the TD(lambda) window; it requires cost_n_step.")

        if not getattr(self.policy, "is_quantile_cost_critic", False):
            raise RuntimeError(
                "FH-DCMPO requires a quantile cost critic (policy `cost_critic_type: quantile`). "
                "The categorical head hard-clips its support at v_max, and undiscounted episodic "
                "costs run well past any fixed upper edge."
            )

        print(
            "FH-DCMPO: undiscounted finite-horizon cost critic (gamma_c = 1), "
            f"qc_thres = {self.qc_thres:.3f} = cost_limit (no qc_scale), "
            f"risk={self.fh_risk_mode} alpha={self.fh_alpha} kappa 0 -> {self.fh_kappa_target} "
            f"over [{self.fh_kappa_warmup}, {self.fh_kappa_warmup + self.fh_kappa_ramp}] updates"
        )
        print(
            "FH-DCMPO scale note: rho_c is episodic (order 10-40) where the discounted arms had Q_c "
            "of order 2-4, but lambda's balance point is std_a(Q_r)/std_a(rho_c) -- a ratio of "
            "action SPREADS, not of levels -- and those are comparable, so the qrdmpo gains carry "
            f"over. Measured balance point 0.88 (S0 smoke); lambda_max is {self.lambda_max}. If "
            "`lambda_balanced` in the logs drifts above lambda_max, the cost term can no longer "
            "perturb the softmax enough to bind and cost will rise with at_cap_frac still at 0."
        )

    # -- EMA bootstrap policy (cost target only) -----------------------------------------------

    def _sync_target_actor(self) -> None:
        """MPO's E-step anchor update, plus the cost-bootstrap EMA (same cadence: per actor update).

        The two averages are deliberately independent: `actor_target` (rate ``self.tau``) is the
        E-step trust-region anchor and must keep tracking the actor as before; `cost_boot_actor`
        (rate ``cost_boot_ema_tau``) exists to LAG, and only the cost target reads it.
        """
        super()._sync_target_actor()
        if self.cost_boot_actor is None:
            return
        tau_b = self.cost_boot_ema_tau
        with torch.no_grad():
            for p, bp in zip(self.policy.actor.parameters(), self.cost_boot_actor.parameters()):
                bp.data.mul_(1.0 - tau_b).add_(p.data, alpha=tau_b)

    def _cost_bootstrap_action(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """The action the cost target bootstraps with: ``a' ~ pi_boot`` when the EMA is on,
        else ``a' ~ pi_current`` (the exact pre-existing behaviour)."""
        if self.cost_boot_actor is None:
            return self.policy.sample_with_log_prob(flat_obs)[0]
        actions, _ = self.cost_boot_actor.sample(self.policy.actor_obs_normalizer(flat_obs))
        return actions

    def _boot_drift_diag(self, obs: torch.Tensor) -> None:
        """KL(pi_current || pi_boot) on the batch states -- the lag the experiment must report.

        Closed-form diagonal-Gaussian KL on the pre-tanh distributions (the tanh is a shared
        bijection, so the KL is identical after squashing), summed over action dims, averaged
        over states. Also logs the mean-parameter distance so a pure variance drift is visible.
        """
        norm_obs = self.policy.actor_obs_normalizer(obs)
        mean_c, log_std_c = self.policy.actor(norm_obs)
        mean_b, log_std_b = self.cost_boot_actor(norm_obs)
        var_c, var_b = (2 * log_std_c).exp(), (2 * log_std_b).exp()
        kl = (log_std_b - log_std_c + (var_c + (mean_c - mean_b).pow(2)) / (2 * var_b) - 0.5).sum(-1)
        self._last_boot_diag = {
            "boot_kl_mean": float(kl.mean()),
            "boot_kl_p90": float(kl.quantile(0.9)),
            "boot_mean_l2": float((mean_c - mean_b).norm(dim=-1).mean()),
            "boot_ema_tau": float(self.cost_boot_ema_tau),
        }

    # -- Critic side -------------------------------------------------------------------------

    def _cost_bootstrap_discount(self, effective_n_steps: torch.Tensor | None) -> torch.Tensor | float:
        """``gamma_c = 1``: the cost target is an undiscounted sum over the remaining horizon.

        Returned as a plain float, which every cost call site already handles (it is the shape the
        1-step path uses). The n-step *window* sum is made undiscounted separately, by
        ``ReplayStorage(cost_gamma=1.0)`` -- two independent discounts, both of which have to be
        neutralised for the target to actually be undiscounted.
        """
        return 1.0

    def _cost_bootstrap_mask(self, dones: torch.Tensor, bootstrap: torch.Tensor | None) -> torch.Tensor:
        """``1 - done``: the episode boundary is **real** for a finite-horizon cost.

        The shared mask is ``bootstrap + (1 - done)``, which keeps bootstrapping through a
        time-limit truncation. That is right for an infinite-horizon discounted objective, where
        cutting the episode at ``T`` is an artifact of the simulator and the true value continues
        past it. It is **wrong** here: ``J_c = sum_{t=0}^{T-1} c_t`` has no terms after ``T``, so
        the cost-to-go at the boundary is exactly zero and bootstrapping across it adds a whole
        extra episode's worth of cost to every target beneath it.

        Dropping the bootstrap instead makes the final window's target the exact realized remaining
        cost sum -- a Monte-Carlo anchor at the one place where the answer is known without a
        critic. That anchor is also what *enforces* the ``u = 0`` boundary condition rather than
        merely hoping the network learns it: ``theta(s, u=0)`` is otherwise only ever read as a
        bootstrap value and never regressed toward anything, so with a ``softplus`` head it could
        sit at some positive constant and silently inflate the whole value function.

        Note this is the cost channel only. The reward critic keeps the standard truncation-aware
        mask, because the reward objective *is* infinite-horizon discounted.
        """
        return 1.0 - dones

    def td_lambda_weights(self, length: int) -> torch.Tensor:
        """Geometric TD(lambda) mixture weights over j = 1..L, summing to 1.

        ``w_j = (1-lam) lam^(j-1)`` for ``j < L``, with ALL the remaining mass ``lam^(L-1)`` on the
        longest return. Truncating without that remainder term would quietly renormalise the
        mixture toward short returns -- the opposite of the point.
        """
        lam = self.cost_td_lambda
        j = torch.arange(length, device=self.device, dtype=torch.float32)
        w = (1.0 - lam) * lam**j
        w[-1] = lam ** (length - 1)
        return w / w.sum()

    def _update_cost_critic_quantile(
        self,
        obs,
        critic_obs,
        actions,
        costs,
        dones,
        next_obs,
        next_critic_obs,
        bootstrap=None,
        effective_n_steps=None,
        cost_is_weights=None,
        cost_window_returns=None,
        cost_window_next_obs=None,
        cost_window_mask=None,
        **unused,
    ) -> float:
        """Quantile cost-critic update against a **TD(lambda) target distribution** (SDAC).

        Falls back to the parent's plain n-step target whenever the longer window is absent, so
        this is inert unless ``cost_n_step`` is configured.

        The target is a MIXTURE over j-step undiscounted returns,
        ``G_j = sum_{k<j} c_k + m_j Z_c(s_{t+j})``, with geometric weights. Two properties are the
        whole reason for it:

        * every component past an episode boundary has ``m_j = 0``, so it is the *exact realized*
          remaining episodic cost -- a pure Monte-Carlo atom with no critic error in it;
        * the mixture's spread is therefore sourced from real returns rather than from the target
          network's own distribution. A one-step bootstrap can only ever reproduce the spread it
          already has, which is why the tail stayed 2.2-2.4x too narrow across four arms that
          changed the constraint, the controller and the loss shape but never the target.
        """
        if cost_window_returns is None:
            return super()._update_cost_critic_quantile(
                obs,
                critic_obs,
                actions,
                costs,
                dones,
                next_obs,
                next_critic_obs,
                bootstrap=bootstrap,
                effective_n_steps=effective_n_steps,
                cost_is_weights=cost_is_weights,
            )

        if self.offpolicy_diag and "cost_window_actions" in unused:
            self._opd_counter += 1
            if self._opd_counter % self.offpolicy_diag_interval == 0:
                with torch.no_grad():
                    self._offpolicy_mismatch_diag(unused, cost_window_returns)

        B, L = cost_window_returns.shape
        w = self.td_lambda_weights(L)  # [L]

        with torch.no_grad():
            flat_obs = cost_window_next_obs.reshape(B * L, -1)
            flat_norm = self.policy.critic_obs_normalizer(flat_obs)
            # a' ~ pi_boot when the EMA bootstrap is configured, else pi_current (baseline).
            flat_act = self._cost_bootstrap_action(flat_obs)
            if self.cost_boot_actor is not None:
                self._boot_drift_diag(obs)
            targets = []
            for tgt in self.policy.cost_critic_targets:
                z = tgt(flat_norm, flat_act).reshape(B, L, -1)  # [B, L, Nq]
                g = cost_window_returns.unsqueeze(-1) + cost_window_mask.unsqueeze(-1) * z
                targets.append(g.reshape(B, -1))  # [B, L*Nq]
            nq = targets[0].shape[1] // L
            # Each component contributes Nq equally-weighted atoms carrying w_j of the mass.
            atom_w = (w / nq).repeat_interleave(nq).unsqueeze(0).expand(B, -1)

        obs_normalized = self.policy.critic_obs_normalizer(critic_obs)
        weights = 1.0 if cost_is_weights is None else cost_is_weights.view(-1)
        cost_critic_loss = 0.0
        for i, (critic, target_theta) in enumerate(zip(self.policy.cost_critics, targets)):
            theta = critic(obs_normalized, actions)
            if self.cost_loss_scale_norm:
                sc = self._cost_loss_scale(target_theta)
                per_sample = sc * quantile_huber_loss(
                    theta / sc, target_theta / sc, critic.tau_hat, critic.kappa, target_weights=atom_w
                )
            else:
                per_sample = quantile_huber_loss(
                    theta, target_theta, critic.tau_hat, critic.kappa, target_weights=atom_w
                )
            cost_critic_loss = cost_critic_loss + (weights * per_sample).mean()
            if i == 0:
                with torch.no_grad():
                    mc_frac = float((cost_window_mask == 0).float().mean())
                    self._last_cost_critic_diag = {
                        "critic_cost_mean_Q": float(critic.get_value(theta).mean()),
                        "critic_cost_zero_frac": float(critic.zero_frac(theta).mean()),
                        "critic_cost_spread": float(critic.spread(theta).mean()),
                        # Share of target atoms that are pure Monte-Carlo (episode already ended).
                        # If this is ~0 the window never reaches an episode boundary and TD(lambda)
                        # is still only reshuffling bootstrapped atoms.
                        "critic_cost_mc_frac": mc_frac,
                        "critic_cost_target_spread": float(
                            (target_theta.max(dim=1).values - target_theta.min(dim=1).values).mean()
                        ),
                    }

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        params = [p for c in self.policy.cost_critics for p in c.parameters()]
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.cost_critic_optimizer.step()
        return cost_critic_loss.item()

    # -- Off-policy mismatch diagnostic --------------------------------------------------------

    _OPD_DEPTHS = (1, 10, 25, 35, 50, 63)

    def _offpolicy_mismatch_diag(self, window: dict, returns: torch.Tensor) -> None:
        """Measure behavior/current-policy drift along the TD(lambda) window. Pure telemetry.

        The j-step target ``G_j = sum_{k<j} c_k + m_j Z_c(s_{t+j})`` realizes its costs under
        the BEHAVIOR policy's actions ``a_{t+1} .. a_{t+j-1}`` (offset 0 is the conditioning
        action of ``Q_c(s_t, a_t)`` and carries no mismatch; the bootstrap action at the far end
        is drawn from the current policy). This computes, per window offset ``j >= 1`` still
        inside the episode::

            log_ratio_j = log pi_current(a_j | s_j) - log pi_behavior(a_j | s_j)

        and reports its distribution, the cumulative window log-ratio, the effective sample
        size a hypothetical importance correction would retain, and the correlation between
        mismatch and realized window cost. All keys are prefixed ``opd_``. Depth anchors are
        offsets into the window; offset 63 is the deepest stored behavior action of an L=64
        window (the "j=64" action is the current-policy bootstrap by construction).

        Requires stored ``behavior_log_prob`` (runner flag ``store_behavior_logprob``); with
        only ``cost_window_extras`` the age statistics are still reported.
        """
        diag: dict[str, float] = {}
        age = window.get("cost_window_age")
        if age is not None:
            diag["opd_age_mean_transitions"] = float(age.mean())
            diag["opd_age_p90_transitions"] = float(age.quantile(0.9))

        blp = window.get("cost_window_blp")
        if blp is None:
            self._last_opd_diag = diag
            return

        obs_w = window["cost_window_obs"]  # [B, L, obs]
        act_w = window["cost_window_actions"]  # [B, L, A]
        alive = window["cost_window_alive"]  # [B, L]
        B, L = blp.shape

        # log pi_current at every window step; offset 0 excluded from mismatch (see docstring).
        lp_now = self.policy.action_log_prob(obs_w.reshape(B * L, -1), act_w.reshape(B * L, -1))
        lr = (lp_now.reshape(B, L) - blp)[:, 1:]  # [B, L-1]
        m = alive[:, 1:] > 0  # mismatch only matters while the return is still accumulating

        flat = lr[m]
        if flat.numel() == 0:
            self._last_opd_diag = diag
            return
        diag["opd_logratio_mean"] = float(flat.mean())
        diag["opd_logratio_median"] = float(flat.median())
        diag["opd_logratio_p10"] = float(flat.quantile(0.1))
        diag["opd_logratio_p90"] = float(flat.quantile(0.9))
        log2, log10 = 0.6931471805599453, 2.302585092994046
        diag["opd_frac_ratio_below_half"] = float((flat < -log2).float().mean())
        diag["opd_frac_ratio_above_2"] = float((flat > log2).float().mean())
        diag["opd_frac_ratio_extreme"] = float(((flat < -log10) | (flat > log10)).float().mean())

        # Cumulative window log-ratio and the ESS an importance correction would retain,
        # both as a function of depth. Log-space throughout (rule-4 discipline applies here
        # too: exp of a 63-step sum overflows long before the statistics are interesting).
        cum = (lr * m.float()).cumsum(dim=1)  # [B, L-1]; index d -> depth j = d+1
        for j in self._OPD_DEPTHS:
            d = min(j, L - 1) - 1
            at = lr[:, d][m[:, d]]
            if at.numel() > 0:
                diag[f"opd_logratio_mean_j{j}"] = float(at.mean())
            c = cum[:, d]
            diag[f"opd_cum_logratio_mean_j{j}"] = float(c.mean())
            # ESS/B = (sum w)^2 / (B * sum w^2), w = exp(cum): 1 = uniform, 1/B = collapse.
            ess = (2.0 * torch.logsumexp(c, 0) - torch.logsumexp(2.0 * c, 0)).exp() / B
            diag[f"opd_ess_frac_j{j}"] = float(ess)

        # Does mismatch concentrate on the windows that carry the cost signal?
        cmag = cum[:, -1].abs()
        ret = returns[:, -1]
        if float(cmag.std()) > 1e-8 and float(ret.std()) > 1e-8:
            xy = torch.stack([cmag, ret])
            diag["opd_corr_cummismatch_vs_windowcost"] = float(torch.corrcoef(xy)[0, 1])

        # Mismatch bucketed by replay age (batch quartiles): the step-3 question --
        # is drift specifically a stale-data problem?
        if age is not None:
            qs = age.quantile(torch.tensor([0.25, 0.5, 0.75], device=age.device))
            lo = torch.cat([torch.zeros(1, device=age.device), qs])
            hi = torch.cat([qs, torch.full((1,), float("inf"), device=age.device)])
            mean_abs_lr = (lr.abs() * m.float()).sum(1) / m.float().sum(1).clamp_min(1.0)  # [B]
            for q, (a, b) in enumerate(zip(lo, hi), start=1):
                sel = (age >= a) & (age < b)
                if bool(sel.any()):
                    diag[f"opd_absl_logratio_ageq{q}"] = float(mean_abs_lr[sel].mean())

        vers = window.get("cost_window_version")
        if vers is not None:
            diag["opd_version_gap_mean"] = float((vers[:, 0].max() - vers[:, 0]).mean())

        self._last_opd_diag = diag

    # -- E-step ------------------------------------------------------------------------------

    @property
    def fh_kappa(self) -> float:
        """Current point on the ``kappa`` ramp."""
        return self._fh_kappa

    def _estep_cost_readout_name(self) -> str:
        """The readout the exponent is *currently* built from -- not the configured mode.

        During the kappa warmup a ``cvar`` arm is reading the plain mean, so reporting the config
        value would mislabel the first few thousand updates.
        """
        return "mean" if self._fh_kappa == 0.0 else self.fh_risk_mode

    def _estep_cost(self, critic_obs: torch.Tensor, actions: torch.Tensor, target: bool) -> torch.Tensor:
        """The conservatism statistic the E-step constrains. Shape ``[N*B, 1]``.

        Read off the quantile locations directly: for a QR head the forward output *is* the
        distribution, so no projection or Gaussian fit stands between the critic and the tail.
        """
        critics = self.policy.cost_critic_targets if target else self.policy.cost_critics
        normalizer = self.policy.critic_obs_normalizer
        # Normalize WITHOUT updating the statistics: `critic_obs` is the [N*B] expanded tensor, so
        # updating here would count every state `sample_action_num` times.
        obs_n = normalizer.normalize(critic_obs) if hasattr(normalizer, "normalize") else normalizer(critic_obs)

        vals = []
        vals_ref = []  # kappa = 0 reference, for the dose diagnostic only
        for c in critics:
            theta = c(obs_n, actions)
            if self.fh_risk_mode == "evt":
                val, info = evt_conservatism_statistic(theta, self.fh_evt_mu, self.fh_evt_nu, self._fh_kappa)
                self._last_evt_info = info
            else:
                val = conservatism_statistic(theta, self.fh_alpha, self._fh_kappa)
            vals.append(val)
            vals_ref.append(val if self._fh_kappa == 0.0 else theta.mean(dim=-1))
        out = torch.stack(vals, dim=0).mean(dim=0).unsqueeze(-1)
        # At kappa = 0 both statistics ARE the mean (`conservatism_statistic` returns it directly,
        # and the EVT offset is scaled by kappa), so alias the tensor rather than recomputing it:
        # the dose is then exactly 1.0, with no float drift to make the mean arm look perturbed.
        self._last_estep_cost_ref = (
            out if self._fh_kappa == 0.0 else (torch.stack(vals_ref, dim=0).mean(dim=0).unsqueeze(-1))
        )
        return out

    def _estep_weights(self, q: torch.Tensor, actions: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Advance the ``kappa`` ramp, then run CVPO's constrained E-step unchanged.

        ``kappa`` is advanced *before* the parent reads the cost signal, so the value recorded in
        the diagnostics is the one the weights were actually built with.
        """
        self._fh_kappa = kappa_at(self._fh_updates, self.fh_kappa_target, self.fh_kappa_warmup, self.fh_kappa_ramp)
        self._fh_updates += 1
        weights = super()._estep_weights(q, actions, critic_obs)
        # The parent's `eqc` is already in episodic units here (qc_scale == 1), so it is directly
        # comparable with the runner's realized J_c -- which is the whole point of the change.
        self._last_estep_info["fh_kappa"] = self._fh_kappa
        self._last_estep_info["fh_rho_over_limit"] = self._eqc / max(float(self.cost_limits[0]), 1e-12)
        return weights

    def get_penalty_info(self) -> dict[str, Any]:
        info = super().get_penalty_info()
        info.update(
            {
                "fh_kappa": self._fh_kappa,
                "fh_alpha": self.fh_alpha,
                "fh_updates": float(self._fh_updates),
                # The bound's predicted violation rate at the current tail level. Logged next to
                # the realized exceedance rate so the gap -- which IS the sum of the bound's error
                # terms -- is visible without post-hoc analysis.
                "fh_predicted_violation_rate": (1.0 - self.fh_alpha) if self._fh_kappa >= 1.0 else float("nan"),
            }
        )
        info.update(self._last_evt_info)
        info.update(self._last_opd_diag)
        info.update(self._last_boot_diag)
        return info

    # -- Diagnostics -------------------------------------------------------------------------

    def cost_units_report(self) -> dict[str, float]:
        """Assert-able summary that the cost units really are undiscounted episodic.

        Every number here must hold for the constraint to mean what the config says. Checked in
        ``tests/test_fhdcmpo.py`` rather than left to a reviewer's eye, because each one is a
        silent-failure path: a stray ``qc_scale``, a discounted bootstrap, or a discounted n-step
        window all produce a plausible-looking run that enforces the wrong budget.
        """
        nstep = self._cost_bootstrap_discount(torch.tensor([10, 3], dtype=torch.long))
        # A time-limit truncation: done=1, bootstrap=1. The reward channel keeps bootstrapping
        # (mask 1); the cost channel must not (mask 0).
        trunc = self._cost_bootstrap_mask(torch.ones(1, 1), torch.ones(1, 1))
        return {
            "truncation_mask": float(trunc.max()),
            "truncation_mask_reward": float(self._bootstrap_mask(torch.ones(1, 1), torch.ones(1, 1)).max()),
            "qc_thres": float(self.qc_thres),
            "cost_limit": float(self.cost_limits[0]),
            "qc_scale": float(self._qc_scale),
            "cost_gamma": float(self.cost_gamma),
            "bootstrap_discount_1step": float(self._cost_bootstrap_discount(None)),
            # A tensor here would mean gamma_c ** n crept back in; a scalar 1.0 is the fixed point.
            "bootstrap_discount_nstep_max": float(torch.as_tensor(nstep).max()),
        }
