from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.distributions import Normal, kl_divergence
from typing import Any

from scipy.optimize import minimize

from safe_rl.algorithms.mpo import effective_sample_size, nonparametric_kl_from_weights
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
        # Overflow guards only — a low cap pins the multiplier and un-enforces the M-step
        # trust region (measured across three envs; see codex/mpo-vs-acme-reference.md).
        alpha_mean_max: float = 10.0,
        alpha_var_max: float = 1000.0,
        mstep_iteration_num: int = 5,
        per_dim_constraining: bool = False,  # Acme default is True; ours stays scalar for back-compat
        decoupled_mstep: bool = False,  # Acme splits the weighted MLE into mean/std halves
        estep_use_target_critic: bool = False,
        estep_q_reduction: str = "min",  # "min" (SAC-style pessimism) or "mean" (Acme-style)
        target_actor_update: str = "polyak",  # "polyak" (tau EMA) or "hard" (Acme: periodic copy)
        target_actor_period: int = 100,  # hard mode: actor updates between copies
        cost_horizon: int = 1000,  # episode length used to scale the episodic cost limit -> Q-space
        qc_thres: float | None = None,  # override the auto-computed cost-Q threshold
        lambda_mode: str = "grad",  # "grad" (graded projected ascent) or "dual" (per-batch joint SLSQP)
        lambda_lr: float = 0.03,  # step size for the graded-lambda controller
        lambda_max: float = 100.0,  # cap on lambda (also the dual upper bound)
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

        # Convert the episodic cost limit (e.g. 25) into the discounted cost-Q scale the
        # cost critic actually predicts: Q_c ~ c_bar * (1 - gamma^H) / (1 - gamma).
        if qc_thres is not None:
            self.qc_thres = float(qc_thres)
        else:
            g, h = self.gamma, int(cost_horizon)
            scale = (1.0 - g**h) / (1.0 - g) / max(h, 1)
            self.qc_thres = float(self.cost_limits[0]) * scale
        print(f"CVPO qc_thres (cost-Q threshold) = {self.qc_thres:.4f}  (episodic limit {self.cost_limits[0]})")

        # Frozen target actor: the E-step samples from it and the M-step KL is measured
        # against it. Polyak-averaged toward the online actor after each actor update.
        self.actor_target = deepcopy(self.policy.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # M-step KL Lagrange multipliers (dual-ascent, warm-started across batches). They are
        # arrays so the scalar and per-dimension trust regions share one code path:
        # shape [1] vs [num_actions].
        dual_dim = self.policy.actor.num_actions if self.per_dim_constraining else 1
        self.eta = 1.0  # E-step temperature (warm-start for SLSQP)
        self.lam = 1.0  # E-step cost multiplier (warm-start for SLSQP)
        self.alpha_mean = np.zeros(dual_dim)  # M-step mean-KL multiplier(s)
        self.alpha_var = np.zeros(dual_dim)  # M-step var-KL multiplier(s)
        self._solver_status = -1.0
        self._solver_iters = 0.0
        self._last_actor_info: dict[str, float] = {}

    # PID-Lagrangian is unused; CVPO re-solves the multiplier per batch in the E-step.
    def update_lagrangian_multipliers(self, current_costs: list[float]) -> None:  # noqa: D401
        return

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
                qc_flat = self.policy.evaluate_cost_q_target(cobs_exp, act_flat)
            else:
                q1, q2 = self.policy.evaluate_q(cobs_exp, act_flat)
                qc_flat = self.policy.evaluate_cost_q(cobs_exp, act_flat)
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
                self.lam = float(np.clip(self.lam + self.lambda_lr * (eqc - self.qc_thres), 0.0, self.lambda_max))
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
                # Acme splits the weighted MLE the same way it splits the KL, so the mean
                # gradient is scaled by the old 1/sigma^2 instead of the shrinking new one.
                mle = (weights * dist_mean.log_prob(x).sum(dim=-1)).sum(dim=0).mean() + (
                    weights * dist_var.log_prob(x).sum(dim=-1)
                ).sum(dim=0).mean()
            else:
                log_prob = Normal(mean, std).log_prob(x).sum(dim=-1)  # [N, B]
                mle = (weights * log_prob).sum(dim=0).mean()

            # Per-action-dim KLs, averaged over states. Summing to a scalar constrains the
            # whole action vector jointly; keeping the vector gives each dim its own budget
            # and its own multiplier (Acme's per_dim_constraining).
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
            # tanh saturates past |x| ~ 2.5, where Q is flat in the pre-tanh mean and the
            # M-step stops pushing back — the failure mode Acme's out-of-bound action
            # penalization guards against in its unsquashed parameterization.
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
            "eta": self.eta,
        }
        info.update(self._last_actor_info)
        return info
