from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.distributions import Normal, kl_divergence
from typing import Any

from scipy.optimize import minimize

from safe_rl.algorithms.mpo import effective_sample_size, nonparametric_kl_from_weights
from safe_rl.common.cost_scaling import make_thresholds, measured_qc_scale
from safe_rl.algorithms.safe_sac import SafeSAC
from safe_rl.modules.safe_sac_actor_critic import SafeSACActorCritic


class CVPO(SafeSAC):
    """Constrained Variational Policy Optimization (Liu et al., ICML 2022).

    https://arxiv.org/abs/2201.11927

    An off-policy, Expectation-Maximization safe-RL algorithm. It reuses the twin
    reward critics, cost critic and replay buffer from :class:`SafeSAC` verbatim and
    replaces *only* the actor update with the CVPO E-step / M-step:

    * **E-step** — for each state sample ``sample_action_num`` candidate actions from
      the (target) policy, evaluate ``Q_r`` and ``Q_c`` for each, and solve a small
      2-variable convex dual (over temperature ``eta`` and cost multiplier ``lambda``)
      in closed form to obtain a non-parametric variational distribution
      ``q(a|s) ∝ exp((Q_r - lambda·Q_c)/eta)`` that is high-reward, low-cost and stays
      within a KL trust region of the current policy.
    * **M-step** — fit the parametric Gaussian policy to the weighted samples by
      weighted maximum likelihood, subject to *decoupled* mean / covariance KL trust
      regions to the old policy (MPO-style, with Lagrange multipliers on each KL).

    Unlike PID-Lagrangian methods (SafeSAC/PPOL_PID), the cost multiplier ``lambda`` is
    re-solved *per batch* from the convex dual rather than integrated by a slow outer
    controller — this is what avoids the "wait for the multiplier to catch up"
    oscillation that pins penalty methods on the reward/cost frontier.

    Only single-constraint problems (``num_costs == 1``) are supported by the E-step
    dual solve.
    """

    policy: SafeSACActorCritic

    def __init__(
        self,
        policy: SafeSACActorCritic,
        # --- CVPO-specific ---
        sample_action_num: int = 64,
        dual_constraint: float = 0.1,  # E-step KL bound (eps in the dual)
        kl_mean_constraint: float = 0.01,  # M-step mean-KL trust region
        kl_var_constraint: float = 1e-4,  # M-step covariance-KL trust region
        alpha_mean_scale: float = 1.0,  # dual ascent step for the mean-KL multiplier
        alpha_var_scale: float = 100.0,  # dual ascent step for the var-KL multiplier
        # Overflow guards only: a low cap pins the multiplier and un-enforces the M-step
        # trust region.
        alpha_mean_max: float = 10.0,
        alpha_var_max: float = 1000.0,
        mstep_iteration_num: int = 5,
        per_dim_constraining: bool = False,
        decoupled_mstep: bool = False,
        estep_use_target_critic: bool = False,
        estep_q_reduction: str = "min",  # "min" | "mean"
        target_actor_update: str = "polyak",  # "polyak" | "hard"
        target_actor_period: int = 100,
        cost_horizon: int = 1000,  # episode length used to scale the episodic cost limit -> Q-space
        qc_thres: float | None = None,  # override the auto-computed cost-Q threshold
        lambda_mode: str = "grad",  # "grad" (graded projected ascent) or "dual" (per-batch joint SLSQP)
        lambda_lr: float = 0.03,  # step size for the graded-lambda controller
        lambda_max: float = 100.0,  # cap on lambda (also the dual upper bound)
        cost_constraint_mode: str = "mean",  # "mean" = E[Z_c] (CVPO) | "cvar" = tail mean (WCSAC-style)
        cvar_alpha: float = 0.9,  # tail level for cvar mode: constrain the worst 1-alpha fraction
        cost_critic_passive: bool = False,  # train Q_c but never let it influence the policy
        use_measured_qc_scale: bool = False,  # estimate qc_scale from rollouts, then freeze
        qc_scale_estimate_episodes: int = 50,  # completed episodes to estimate it from
        qc_scale_source: str = "analytic",  # "analytic" | "measured" — how episodic -> Q-space is scaled
        qc_scale_measured: float | None = None,  # required when qc_scale_source == "measured"
        qc_scale_probe: str | None = None,  # provenance: which probe run the measured value came from
        qc_thres_adapt: bool = False,  # calibrate qc_thres from realized episodic cost
        qc_thres_lr: float = 2e-3,  # integral gain of that outer loop
        qc_thres_min_frac: float = 0.05,  # floor, as a fraction of the analytic qc_thres
        qc_ema: float = 0.05,  # EMA weight for the realized-cost estimate
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

        self.sample_action_num = int(sample_action_num)
        self.eps_dual = float(dual_constraint)
        self.eps_kl_mean = float(kl_mean_constraint)
        self.eps_kl_var = float(kl_var_constraint)
        self.alpha_mean_scale = float(alpha_mean_scale)
        self.alpha_var_scale = float(alpha_var_scale)
        self.alpha_mean_max = float(alpha_mean_max)
        self.alpha_var_max = float(alpha_var_max)
        self.mstep_iteration_num = int(mstep_iteration_num)
        self.per_dim_constraining = bool(per_dim_constraining)
        self.decoupled_mstep = bool(decoupled_mstep)
        self.estep_use_target_critic = bool(estep_use_target_critic)
        if estep_q_reduction not in ("min", "mean"):
            raise ValueError(f"estep_q_reduction must be 'min' or 'mean', got {estep_q_reduction!r}.")
        self.estep_q_reduction = estep_q_reduction
        if target_actor_update not in ("polyak", "hard"):
            raise ValueError(f"target_actor_update must be 'polyak' or 'hard', got {target_actor_update!r}.")
        self.target_actor_update = target_actor_update
        self.target_actor_period = int(target_actor_period)
        self._actor_update_count = 0
        if lambda_mode not in ("grad", "dual"):
            raise ValueError(f"lambda_mode must be 'grad' or 'dual', got {lambda_mode!r}.")
        self.lambda_mode = lambda_mode
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
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
        if cost_constraint_mode not in ("mean", "cvar"):
            raise ValueError(f"cost_constraint_mode must be 'mean' or 'cvar', got {cost_constraint_mode!r}.")
        self.cost_constraint_mode = cost_constraint_mode
        if not 0.0 <= float(cvar_alpha) < 1.0:
            raise ValueError(f"cvar_alpha must be in [0, 1), got {cvar_alpha}.")
        self.cvar_alpha = float(cvar_alpha)
        self.cost_critic_passive = bool(cost_critic_passive)
        self.lam = 0.0 if self.cost_critic_passive else 1.0  # E-step cost multiplier
        self.alpha_mean = np.zeros(dual_dim)  # M-step mean-KL multiplier(s)
        self.alpha_var = np.zeros(dual_dim)  # M-step var-KL multiplier(s)
        self._solver_status = -1.0
        self._solver_iters = 0.0
        self._last_actor_info: dict[str, float] = {}

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
        if not self.qc_thres_adapt or not current_costs:
            return
        realized = float(current_costs[0])
        if self._realized_cost_ema is None:
            self._realized_cost_ema = realized
        else:
            self._realized_cost_ema += self.qc_ema * (realized - self._realized_cost_ema)

        limit = float(self.cost_limits[0])
        step = self.qc_thres_lr * (self._realized_cost_ema - limit) * self._qc_scale
        floor = self.qc_thres_min_frac * self._qc_thres_initial
        self.qc_thres = float(min(max(self.qc_thres - step, floor), self._qc_thres_initial))

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
        thres = self.qc_thres

        if self.lambda_mode == "grad":
            lam = self.lam  # fixed this batch; updated by the controller after the E-step

            def dual_eta(x: np.ndarray) -> float:
                eta = x[0]
                z = (q_np - lam * qc_np) / eta
                zmax = z.max(axis=0, keepdims=True)
                lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
                return eta * eps + eta * float(np.mean(lse))  # lam*thres is constant in eta

            try:
                res = minimize(dual_eta, np.array([max(self.eta, 1e-3)]), method="SLSQP", bounds=[(1e-6, 1e6)])
                eta = float(res.x[0])
                self._solver_status = float(res.status)
                self._solver_iters = float(res.nit)
                if not np.isfinite(eta):
                    raise ValueError("non-finite eta")
            except Exception as exc:  # pragma: no cover - numerical fallback
                print(f"CVPO eta solve failed ({exc}); keeping previous eta.")
                eta = self.eta
                self._solver_status = -1.0
                self._solver_iters = 0.0
            return max(eta, 1e-6), max(lam, 0.0)

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

        if not getattr(self.policy, "is_distributional_cost_critic", False):
            raise RuntimeError(
                "cost_constraint_mode='cvar' requires a distributional cost critic "
                "(policy cost_critic_type='distributional')."
            )
        critics = self.policy.cost_critic_targets if target else self.policy.cost_critics
        obs_n = self.policy.critic_obs_normalizer(critic_obs)
        vals = [c.get_cvar(c.get_dist(c(obs_n, actions)), self.cvar_alpha) for c in critics]
        return torch.stack(vals, dim=0).mean(dim=0).unsqueeze(-1)

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> tuple[float, float]:
        """CVPO E-step + M-step in place of the SafeSAC Lagrangian actor update.

        ``obs`` / ``critic_obs`` arrive already normalised from :meth:`SafeSAC.update`.
        """
        critic_obs = obs if critic_obs is None else critic_obs
        batch_size = obs.shape[0]
        n = self.sample_action_num
        act_b = self.policy.actor.action_b
        act_c = self.policy.actor.action_c

        # The E/M steps call the actor networks directly rather than through
        # ``policy.act`` / ``policy.sample``, which apply this normalizer internally — so
        # apply it once here, or the policy would be trained on a different input scale
        # than it acts on. No-op (Identity) unless ``actor_obs_normalization`` is set.
        actor_obs = self.policy.actor_obs_normalizer(obs)

        # ----- E-step (no gradients) -----
        with torch.no_grad():
            mean_old, log_std_old = self.actor_target(actor_obs)  # [B, A]
            std_old = log_std_old.exp()
            dist_old = Normal(mean_old, std_old)

            x = dist_old.sample((n,))  # [N, B, A] pre-tanh
            actions = act_b + act_c * torch.tanh(x)  # squashed into env bounds

            cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
            act_flat = actions.reshape(n * batch_size, -1)
            if self.estep_use_target_critic:
                q1, q2 = self.policy.evaluate_q_target(cobs_exp, act_flat)
                qc_flat = self._estep_cost(cobs_exp, act_flat, target=True)
            else:
                q1, q2 = self.policy.evaluate_q(cobs_exp, act_flat)
                qc_flat = self._estep_cost(cobs_exp, act_flat, target=False)
            q_pair = torch.min(q1, q2) if self.estep_q_reduction == "min" else 0.5 * (q1 + q2)
            q = q_pair.reshape(n, batch_size)  # [N, B]
            qc = qc_flat[:, 0].reshape(n, batch_size)

            eta, lam = self._solve_dual(q.cpu().numpy().astype(np.float64), qc.cpu().numpy().astype(np.float64))
            self.eta, self.lam = eta, lam

            # Non-parametric variational weights q(a|s): softmax over the N samples.
            logits = (q - lam * qc) / eta  # [N, B]
            weights = torch.softmax(logits, dim=0)  # [N, B], columns sum to 1

            # Graded-lambda controller: integrate the constraint violation E_q[Q_c] - qc_thres
            # so a single slack batch can't collapse lambda to the floor (the bang-bang failure
            # of the joint per-batch solve). Projected onto [0, lambda_max]; warm-started.
            if self.lambda_mode == "grad":
                eqc = (weights * qc).sum(dim=0).mean().item()  # E_q[Q_c] over states
                if not self.cost_critic_passive:
                    self.lam = float(
                        np.clip(self.lam + self.lambda_lr * (eqc - self.qc_thres), 0.0, self.lambda_max)
                    )
                self._eqc = eqc

            # E-step health: kl_q is the KL the dual was supposed to hold at eps_dual, so
            # eps_dual - kl_q is the dual residual dg/deta and should sit at ~0. ess collapsing
            # toward 1 means the M-step is regressing onto a single sampled action per state.
            kl_q = nonparametric_kl_from_weights(weights)
            ess = effective_sample_size(weights)

        mean_old = mean_old.detach()
        std_old = std_old.detach()

        # ----- M-step: weighted MLE under decoupled mean/covariance KL trust regions -----
        dist_old_ref = Normal(mean_old, std_old)
        for _ in range(self.mstep_iteration_num):
            mean, log_std = self.policy.actor(actor_obs)  # [B, A]
            std = log_std.exp()

            # Decoupled KL (old || new): mean uses old covariance, covariance uses old mean.
            dist_mean = Normal(mean, std_old)  # vary mean, hold std
            dist_var = Normal(mean_old, std)  # hold mean, vary std

            if self.decoupled_mstep:
                # Mean term at the old std, std term at the old mean (split like the KL).
                mle = (weights * dist_mean.log_prob(x).sum(dim=-1)).sum(dim=0).mean() + (
                    weights * dist_var.log_prob(x).sum(dim=-1)
                ).sum(dim=0).mean()
            else:
                log_prob = Normal(mean, std).log_prob(x).sum(dim=-1)  # [N, B]
                mle = (weights * log_prob).sum(dim=0).mean()

            # Per-action-dim KLs, averaged over states. Summed -> one joint budget;
            # kept as a vector -> one budget and multiplier per dimension.
            kl_mean_dims = kl_divergence(dist_old_ref, dist_mean).mean(dim=0)  # [A]
            kl_var_dims = kl_divergence(dist_old_ref, dist_var).mean(dim=0)  # [A]
            if self.per_dim_constraining:
                kl_mean_vec, kl_var_vec = kl_mean_dims, kl_var_dims
            else:
                kl_mean_vec = kl_mean_dims.sum().reshape(1)
                kl_var_vec = kl_var_dims.sum().reshape(1)

            self.alpha_mean = np.clip(
                self.alpha_mean + self.alpha_mean_scale * (kl_mean_vec.detach().cpu().numpy() - self.eps_kl_mean),
                0.0,
                self.alpha_mean_max,
            )
            self.alpha_var = np.clip(
                self.alpha_var + self.alpha_var_scale * (kl_var_vec.detach().cpu().numpy() - self.eps_kl_var),
                0.0,
                self.alpha_var_max,
            )
            alpha_mean_t = torch.as_tensor(self.alpha_mean, dtype=kl_mean_vec.dtype, device=kl_mean_vec.device)
            alpha_var_t = torch.as_tensor(self.alpha_var, dtype=kl_var_vec.dtype, device=kl_var_vec.device)

            actor_loss = -(
                mle
                + (alpha_mean_t * (self.eps_kl_mean - kl_mean_vec)).sum()
                + (alpha_var_t * (self.eps_kl_var - kl_var_vec)).sum()
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        # Post-loop diagnostics from the final inner iteration.
        actor_loss_val = actor_loss.item()
        # Report the joint (summed-over-dims) KL in both modes so the number is comparable.
        kl_mean_val = float(kl_mean_dims.sum().item())
        kl_var_val = float(kl_var_dims.sum().item())
        with torch.no_grad():
            std_d = std.detach()
            std_min = float(std_d.min().item())
            std_max = float(std_d.max().item())
            std_cond = float((std_d.max(dim=-1).values / std_d.min(dim=-1).values.clamp_min(1e-12)).mean().item())
            # tanh saturates past |x| ~ 2.5, where Q is flat in the pre-tanh mean.
            mean_absmax = float(mean.detach().abs().max().item())
            frac_saturated = float((mean.detach().abs() > 2.5).float().mean().item())

        # Target actor update: Polyak EMA, or Acme's hard periodic copy (fixed anchor).
        self._actor_update_count += 1
        with torch.no_grad():
            if self.target_actor_update == "hard":
                if self._actor_update_count % self.target_actor_period == 0:
                    for p, tp in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                        tp.data.copy_(p.data)
            else:
                for p, tp in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        self._last_actor_info = {
            "eta": eta,
            "lambda": self.lam,
            "eqc": getattr(self, "_eqc", float("nan")),
            # Eqc expressed back in episodic-cost units, for direct comparison with the
            # runner's measured episode cost (equal only if costs are uniform in time).
            "eqc_as_episodic_cost": getattr(self, "_eqc", float("nan")) / max(self._qc_scale, 1e-12),
            "kl_mean": kl_mean_val,
            "kl_var": kl_var_val,
            # Ratios are budget-usage, not raw KL / per-dim eps: in per-dimension mode the
            # total budget is eps * num_actions, so the two trust-region modes stay comparable.
            "kl_mean_rel": kl_mean_val / max(self.eps_kl_mean * len(self.alpha_mean), 1e-12),
            "kl_var_rel": kl_var_val / max(self.eps_kl_var * len(self.alpha_var), 1e-12),
            "alpha_mean": float(self.alpha_mean.mean()),
            "alpha_var": float(self.alpha_var.mean()),
            "kl_q": float(kl_q.mean().item()),
            "kl_q_rel": float(kl_q.mean().item()) / max(self.eps_dual, 1e-12),
            "dual_residual_eta": self.eps_dual - float(kl_q.mean().item()),
            "ess": float(ess.mean().item()),
            "ess_min": float(ess.min().item()),
            "pi_std_min": std_min,
            "pi_std_max": std_max,
            "pi_std_cond": std_cond,
            "pretanh_mean_absmax": mean_absmax,
            "frac_saturated": frac_saturated,
            "solver_status": self._solver_status,
            "solver_iters": self._solver_iters,
        }
        # Second return slot is the SAC alpha loss (unused by CVPO).
        return actor_loss_val, 0.0

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
        }
        info.update(self._last_actor_info)
        # Hazard-stratified replay composition (empty unless hazard_fraction > 0).
        info.update(self._last_replay_info)
        return info
