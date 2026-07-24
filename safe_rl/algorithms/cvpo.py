from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.distributions import Normal, kl_divergence

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
        dual_constraint: float = 0.1,        # E-step KL bound (eps in the dual)
        kl_mean_constraint: float = 0.01,    # M-step mean-KL trust region
        kl_var_constraint: float = 1e-4,     # M-step covariance-KL trust region
        alpha_mean_scale: float = 1.0,       # dual ascent step for the mean-KL multiplier
        alpha_var_scale: float = 100.0,      # dual ascent step for the var-KL multiplier
        alpha_mean_max: float = 0.1,
        alpha_var_max: float = 10.0,
        mstep_iteration_num: int = 5,
        cost_horizon: int = 1000,            # episode length used to scale the episodic cost limit -> Q-space
        qc_thres: float | None = None,       # override the auto-computed cost-Q threshold
        lambda_mode: str = "grad",           # "grad" (graded projected ascent) or "dual" (per-batch joint SLSQP)
        lambda_lr: float = 0.03,             # step size for the graded-lambda controller
        lambda_max: float = 100.0,           # cap on lambda (also the dual upper bound)
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
            scale = (1.0 - g ** h) / (1.0 - g) / max(h, 1)
            self.qc_thres = float(self.cost_limits[0]) * scale
        print(f"CVPO qc_thres (cost-Q threshold) = {self.qc_thres:.4f}  (episodic limit {self.cost_limits[0]})")

        # Frozen target actor: the E-step samples from it and the M-step KL is measured
        # against it. Polyak-averaged toward the online actor after each actor update.
        self.actor_target = deepcopy(self.policy.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # M-step KL Lagrange multipliers (dual-ascent, warm-started across batches).
        self.eta = 1.0            # E-step temperature (warm-start for SLSQP)
        self.lam = 1.0            # E-step cost multiplier (warm-start for SLSQP)
        self.alpha_mean = 0.0     # M-step mean-KL multiplier
        self.alpha_var = 0.0      # M-step var-KL multiplier
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
                if not np.isfinite(eta):
                    raise ValueError("non-finite eta")
            except Exception as exc:  # pragma: no cover - numerical fallback
                print(f"CVPO eta solve failed ({exc}); keeping previous eta.")
                eta = self.eta
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
            if not np.isfinite(eta) or not np.isfinite(lam):
                raise ValueError("non-finite dual solution")
        except Exception as exc:  # pragma: no cover - numerical fallback
            print(f"CVPO dual solve failed ({exc}); keeping previous (eta, lambda).")
            eta, lam = self.eta, self.lam
        return max(eta, 1e-6), max(lam, 1e-6)

    def _update_actor_and_alpha(
        self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None
    ) -> tuple[float, float]:
        """CVPO E-step + M-step in place of the SafeSAC Lagrangian actor update.

        ``obs`` / ``critic_obs`` arrive already normalised from :meth:`SafeSAC.update`.
        """
        critic_obs = obs if critic_obs is None else critic_obs
        batch_size = obs.shape[0]
        n = self.sample_action_num
        act_b = self.policy.actor.action_b
        act_c = self.policy.actor.action_c

        # ----- E-step (no gradients) -----
        with torch.no_grad():
            mean_old, log_std_old = self.actor_target(obs)          # [B, A]
            std_old = log_std_old.exp()
            dist_old = Normal(mean_old, std_old)

            x = dist_old.sample((n,))                               # [N, B, A] pre-tanh
            actions = act_b + act_c * torch.tanh(x)                 # squashed into env bounds

            cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
            act_flat = actions.reshape(n * batch_size, -1)
            q1, q2 = self.policy.evaluate_q(cobs_exp, act_flat)
            q = torch.min(q1, q2).reshape(n, batch_size)            # [N, B]
            qc = self.policy.evaluate_cost_q(cobs_exp, act_flat)[:, 0].reshape(n, batch_size)

            eta, lam = self._solve_dual(q.cpu().numpy().astype(np.float64), qc.cpu().numpy().astype(np.float64))
            self.eta, self.lam = eta, lam

            # Non-parametric variational weights q(a|s): softmax over the N samples.
            logits = (q - lam * qc) / eta                          # [N, B]
            weights = torch.softmax(logits, dim=0)                 # [N, B], columns sum to 1

            # Graded-lambda controller: integrate the constraint violation E_q[Q_c] - qc_thres
            # so a single slack batch can't collapse lambda to the floor (the bang-bang failure
            # of the joint per-batch solve). Projected onto [0, lambda_max]; warm-started.
            if self.lambda_mode == "grad":
                eqc = (weights * qc).sum(dim=0).mean().item()      # E_q[Q_c] over states
                self.lam = float(np.clip(self.lam + self.lambda_lr * (eqc - self.qc_thres),
                                         0.0, self.lambda_max))
                self._eqc = eqc

        mean_old = mean_old.detach()
        std_old = std_old.detach()

        # ----- M-step: weighted MLE under decoupled mean/covariance KL trust regions -----
        actor_loss_val = 0.0
        kl_mean_val = 0.0
        kl_var_val = 0.0
        for _ in range(self.mstep_iteration_num):
            mean, log_std = self.policy.actor(obs)                  # [B, A]
            std = log_std.exp()
            dist = Normal(mean, std)

            log_prob = dist.log_prob(x).sum(dim=-1)                # [N, B]
            mle = (weights * log_prob).sum(dim=0).mean()

            # Decoupled KL (old || new): mean uses old covariance, covariance uses old mean.
            dist_mean = Normal(mean, std_old)                      # vary mean, hold std
            dist_var = Normal(mean_old, std)                       # hold mean, vary std
            kl_mean = kl_divergence(Normal(mean_old, std_old), dist_mean).sum(dim=-1).mean()
            kl_var = kl_divergence(Normal(mean_old, std_old), dist_var).sum(dim=-1).mean()

            self.alpha_mean = float(
                np.clip(self.alpha_mean + self.alpha_mean_scale * (kl_mean.item() - self.eps_kl_mean),
                        0.0, self.alpha_mean_max)
            )
            self.alpha_var = float(
                np.clip(self.alpha_var + self.alpha_var_scale * (kl_var.item() - self.eps_kl_var),
                        0.0, self.alpha_var_max)
            )

            actor_loss = -(mle
                           + self.alpha_mean * (self.eps_kl_mean - kl_mean)
                           + self.alpha_var * (self.eps_kl_var - kl_var))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()
            kl_mean_val = kl_mean.item()
            kl_var_val = kl_var.item()

        with torch.no_grad():
            for p, tp in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        self._last_actor_info = {
            "eta": eta,
            "lambda": self.lam,
            "eqc": getattr(self, "_eqc", float("nan")),
            "kl_mean": kl_mean_val,
            "kl_var": kl_var_val,
            "alpha_mean": self.alpha_mean,
            "alpha_var": self.alpha_var,
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
