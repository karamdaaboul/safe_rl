from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.distributions import Normal, kl_divergence

from safe_rl.algorithms.sac import SAC
from safe_rl.modules.sac_actor_critic import SACActorCritic


class MPO(SAC):
    """Maximum a Posteriori Policy Optimization (Abdolmaleki et al., ICLR 2018).

    https://arxiv.org/abs/1806.06920

    An off-policy, Expectation-Maximization actor-critic. It reuses the twin reward
    critics and replay buffer from :class:`SAC` verbatim and replaces *only* the actor
    update with the MPO E-step / M-step (the unconstrained sibling of :class:`CVPO`):

    * **E-step** — for each state sample ``sample_action_num`` candidate actions from
      the (target) policy, evaluate ``Q`` for each, and solve a small 1-variable convex
      dual (over the temperature ``eta``) to obtain a non-parametric variational
      distribution ``q(a|s) ∝ exp(Q/eta)`` that improves on the policy while staying
      within a KL trust region of it.
    * **M-step** — fit the parametric Gaussian policy to the weighted samples by
      weighted maximum likelihood, subject to *decoupled* mean / covariance KL trust
      regions to the old policy (with dual-ascent Lagrange multipliers on each KL).

    Exploration comes from the E-step KL trust region rather than SAC entropy, so the
    entropy temperature is disabled. Candidate actions are sampled in pre-tanh Gaussian
    space and squashed into the env bounds only for critic evaluation, so no
    out-of-bound action penalty is needed (unlike implementations that sample the raw
    Gaussian, e.g. Acme / RL-X).
    """

    policy: SACActorCritic

    def __init__(
        self,
        policy: SACActorCritic,
        # --- MPO-specific ---
        sample_action_num: int = 64,
        dual_constraint: float = 0.1,        # E-step KL bound (eps in the dual)
        kl_mean_constraint: float = 0.01,    # M-step mean-KL trust region
        kl_var_constraint: float = 1e-4,     # M-step covariance-KL trust region
        alpha_mean_scale: float = 1.0,       # dual ascent step for the mean-KL multiplier
        alpha_var_scale: float = 100.0,      # dual ascent step for the var-KL multiplier
        alpha_mean_max: float = 0.1,
        alpha_var_max: float = 10.0,
        mstep_iteration_num: int = 5,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        # MPO handles exploration through the E-step KL trust region, not SAC entropy.
        # Disable the SAC entropy temperature (alpha ~ 0) and its auto-tuning.
        kwargs.pop("auto_entropy_tuning", None)
        kwargs.pop("alpha", None)
        super().__init__(policy, device=device, auto_entropy_tuning=False, alpha=1e-8, **kwargs)

        self.sample_action_num = int(sample_action_num)
        self.eps_dual = float(dual_constraint)
        self.eps_kl_mean = float(kl_mean_constraint)
        self.eps_kl_var = float(kl_var_constraint)
        self.alpha_mean_scale = float(alpha_mean_scale)
        self.alpha_var_scale = float(alpha_var_scale)
        self.alpha_mean_max = float(alpha_mean_max)
        self.alpha_var_max = float(alpha_var_max)
        self.mstep_iteration_num = int(mstep_iteration_num)

        # Frozen target actor: the E-step samples from it and the M-step KL is measured
        # against it. Polyak-averaged toward the online actor after each actor update.
        self.actor_target = deepcopy(self.policy.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # Dual variables (warm-started across batches).
        self.eta = 1.0            # E-step temperature (warm-start for SLSQP)
        self.alpha_mean = 0.0     # M-step mean-KL multiplier
        self.alpha_var = 0.0      # M-step var-KL multiplier
        self._last_actor_info: dict[str, float] = {}

    def _solve_eta(self, q_np: np.ndarray) -> float:
        """Solve the MPO E-step dual for the temperature ``eta``.

        q_np : [N, B] numpy array (N sampled actions per state, B states).

        Minimizes ``g(eta) = eta * eps + eta * E_s[log mean_a exp(Q(s,a)/eta)]``
        (Abdolmaleki et al. 2018, eq. 9), warm-started from the previous solution.

        Returns the eta used to form this batch's variational weights, > 0.
        """
        eps = self.eps_dual

        def dual_eta(x: np.ndarray) -> float:
            eta = x[0]
            z = q_np / eta
            zmax = z.max(axis=0, keepdims=True)
            lse = zmax.squeeze(0) + np.log(np.mean(np.exp(z - zmax), axis=0))
            return eta * eps + eta * float(np.mean(lse))

        try:
            res = minimize(dual_eta, np.array([max(self.eta, 1e-3)]), method="SLSQP", bounds=[(1e-6, 1e6)])
            eta = float(res.x[0])
            if not np.isfinite(eta):
                raise ValueError("non-finite eta")
        except Exception as exc:  # pragma: no cover - numerical fallback
            print(f"MPO eta solve failed ({exc}); keeping previous eta.")
            eta = self.eta
        return max(eta, 1e-6)

    def _update_actor_and_alpha(
        self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None
    ) -> tuple[float, float]:
        """MPO E-step + M-step in place of the SAC entropy-regularized actor update.

        ``obs`` / ``critic_obs`` arrive already normalised from :meth:`SAC.update`.
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

            eta = self._solve_eta(q.cpu().numpy().astype(np.float64))
            self.eta = eta

            # Non-parametric variational weights q(a|s): softmax over the N samples.
            weights = torch.softmax(q / eta, dim=0)                # [N, B], columns sum to 1

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
            "kl_mean": kl_mean_val,
            "kl_var": kl_var_val,
            "alpha_mean": self.alpha_mean,
            "alpha_var": self.alpha_var,
        }
        # Second return slot is the SAC alpha loss (unused by MPO).
        return actor_loss_val, 0.0

    def get_penalty_info(self) -> dict[str, Any]:
        """MPO diagnostics: the E-step temperature and M-step KL / multiplier state."""
        info = {"eta": self.eta}
        info.update(self._last_actor_info)
        return info
