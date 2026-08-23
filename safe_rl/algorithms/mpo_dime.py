"""MPO with a DIME/TruDi diffusion actor: path-space weighted-MLE M-step.

MPO's E-step is policy-class-agnostic — it needs K action samples per state and
their critic values. The policy class enters only in the M-step's weighted
maximum likelihood, which for a diffusion actor is intractable in action space:
the action marginal ``pi_theta(a|s)`` integrates over the whole denoising chain,
and the DIME sampler defines no likelihood for an externally given action.

The tractable substitute (chapter §7-8, path-space variant): the E-step samples
FULL SDE trajectories ``x^{0:T}`` from the frozen target actor and the M-step
maximizes the weighted PATH log-likelihood

    max_theta  sum_ij w_ij * sum_k log N(x_{k+1}; mu_theta_k(x_k, s_i), sigma_k^2)

subject to the path-space KL(pi_old ‖ pi_theta) <= beta — the sum of per-step
Gaussian KLs along the same old trajectories (Lemma 8.1's upper bound on the
marginal KL; same forward direction as MPO's Gaussian M-step KL). The
constraint is enforced exactly like MPO's: a scalar multiplier updated by
projected dual ascent. This is a weighted-ELBO projection with the old chain as
inference distribution, so Prop. 7.2's bias analysis applies unchanged.

Deliberately shared with MPO (inherited, not copied): the SAC critic tower,
``_solve_eta`` (SLSQP temperature dual), ``_estep_weights``,
``_sync_target_actor``, and the ``actor_target`` deepcopy — one frozen copy of
pi_old, owned by the algorithm.

Prior and tanh-Jacobian terms of the path log-likelihood are theta-independent
(``learn_prior`` is rejected at init) and dropped, mirroring MPO's deliberate
omission of the tanh Jacobian. With ``learn_dt``/``learn_friction`` enabled the
transition noise scale becomes theta-dependent: the MLE handles that correctly
(full Gaussian log-prob), but the simplified KL does not constrain it — pair
learnable friction with ``kl_form="full"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from safe_rl.algorithms.mpo import MPO, effective_sample_size, nonparametric_kl_from_weights
from safe_rl.modules.mpo_dime_actor_critic import MPODIMEActorCritic
from safe_rl.networks.dime.path_mle import path_logprob_and_kl, sde_rollout_with_traj


class MPODIME(MPO):
    """MPO whose policy is a DIME diffusion sampler instead of a Gaussian."""

    policy: MPODIMEActorCritic

    def __init__(
        self,
        policy: MPODIMEActorCritic,
        kl_path_constraint: float = 0.05,
        alpha_path_scale: float = 1.0,
        alpha_path_max: float = 100.0,
        kl_form: str = "simplified",
        estep_q_reduction: str = "min",  # "min" | "mean" over the twin critics
        **kwargs: Any,
    ) -> None:
        if estep_q_reduction not in ("min", "mean"):
            raise ValueError(f"estep_q_reduction must be 'min' or 'mean', got {estep_q_reduction!r}.")
        self.estep_q_reduction = estep_q_reduction
        if not hasattr(getattr(policy, "actor", None), "diffusion_model"):
            raise TypeError(
                "MPODIME requires a diffusion policy (MPODIMEActorCritic); "
                f"got {type(policy).__name__}. Use MPO for Gaussian policies."
            )
        if kl_form not in ("simplified", "full"):
            raise ValueError(f"kl_form must be 'simplified' or 'full', got {kl_form!r}")
        if policy.actor.diffusion_model.learn_prior:
            raise ValueError(
                "MPODIME requires learn_prior=false: the path-MLE drops the prior term "
                "as theta-independent, which a learnable prior would violate."
            )
        if policy.actor.diffusion_model.learn_friction and kl_form != "full":
            print(
                "MPODIME WARNING: learn_friction=true with kl_form='simplified' leaves the "
                "transition-noise scale unconstrained by the trust region (noise-collapse "
                "risk). Use kl_form='full'."
            )
        super().__init__(policy, **kwargs)

        # Path-space trust region: one scalar budget beta replaces the Gaussian
        # mean/var pair; the multiplier follows MPO's projected-dual-ascent pattern.
        self.eps_kl_path = float(kl_path_constraint)
        self.alpha_path_scale = float(alpha_path_scale)
        self.alpha_path_max = float(alpha_path_max)
        self.alpha_path = 0.0
        self.kl_form = kl_form

    # ------------------------------------------------------------------
    # E-step: sample trajectories from the frozen target actor
    # ------------------------------------------------------------------

    def _estep_sample_paths(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor):
        """Roll ``sample_action_num`` SDE chains per state and score the final actions.

        Returns ``(obs_tiled [N*B, obs], traj [T+1, N*B, A], old_means [T, N*B, A],
        old_scales [T, A], actions [N*B, A], q [N, B])``. Runs under the caller's
        ``torch.no_grad()``; everything returned is gradient-free.
        """
        batch_size = actor_obs.shape[0]
        n = self.sample_action_num

        # [N*B, obs]: sample j of state i sits at row j*B + i, matching the
        # reshape(n, batch_size) used for q and the weights below.
        obs_tiled = actor_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
        traj, old_means, old_scales = sde_rollout_with_traj(self.actor_target.diffusion_model, obs_tiled)

        actions = self.policy.actor.action_scale * torch.tanh(traj[-1])

        cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
        if self.estep_use_target_critic:
            q1, q2 = self.policy.evaluate_q_target(cobs_exp, actions)
        else:
            q1, q2 = self.policy.evaluate_q(cobs_exp, actions)
        q_pair = torch.min(q1, q2) if self.estep_q_reduction == "min" else 0.5 * (q1 + q2)
        return obs_tiled, traj, old_means, old_scales, actions, q_pair.reshape(n, batch_size)

    # ------------------------------------------------------------------
    # M-step: weighted path-MLE under the path-space KL trust region
    # ------------------------------------------------------------------

    def _mstep_dime(
        self,
        obs_tiled: torch.Tensor,
        traj: torch.Tensor,
        old_means: torch.Tensor,
        old_scales: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        """Mirror of MPO's ``_mstep`` with path-space MLE and KL."""
        n, batch_size = weights.shape
        for _ in range(self.mstep_iteration_num):
            log_prob, kl = path_logprob_and_kl(
                self.policy.actor.diffusion_model,
                obs_tiled,
                traj,
                old_means,
                old_scales,
                kl_form=self.kl_form,
            )
            mle = (weights * log_prob.view(n, batch_size)).sum(dim=0).mean()
            kl_path = kl.view(n, batch_size).mean()

            self.alpha_path = float(
                np.clip(
                    self.alpha_path + self.alpha_path_scale * (kl_path.item() - self.eps_kl_path),
                    0.0,
                    self.alpha_path_max,
                )
            )
            alpha_path_t = torch.as_tensor(self.alpha_path, dtype=kl_path.dtype, device=kl_path.device)

            actor_loss = -(mle + alpha_path_t * (self.eps_kl_path - kl_path))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        with torch.no_grad():
            pretanh = traj[-1]
            dm = self.policy.actor.diffusion_model
            friction = torch.nn.functional.softplus(dm.friction)
            info = {
                "kl_path": float(kl_path.item()),
                "kl_path_rel": float(kl_path.item()) / max(self.eps_kl_path, 1e-12),
                "alpha_path": self.alpha_path,
                "path_mle": float(mle.item()),
                # tanh saturates past |x| ~ 2.5, where Q is flat in the pre-tanh action.
                "pretanh_absmax": float(pretanh.abs().max().item()),
                "frac_saturated": float((pretanh.abs() > 2.5).float().mean().item()),
                "dime_friction": float(friction.mean().item()),
                "dime_dt": float(torch.nn.functional.softplus(dm.dt).mean().item()),
                "dime_noise_scale": float(self.policy.action_std.mean().item()),
            }
        self._sync_target_actor()
        return actor_loss.item(), info

    # ------------------------------------------------------------------
    # Actor update: E-step + M-step (replaces MPO's Gaussian pair)
    # ------------------------------------------------------------------

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> tuple[float, float]:
        """Same skeleton as MPO's, with trajectories in place of pre-tanh samples."""
        critic_obs = obs if critic_obs is None else critic_obs
        # The E/M steps call the actor directly, bypassing policy.act/sample which apply
        # this normalizer internally; apply it once here so training and acting agree.
        actor_obs = self.policy.actor_obs_normalizer(obs)

        with torch.no_grad():
            obs_tiled, traj, old_means, old_scales, actions, q = self._estep_sample_paths(actor_obs, critic_obs)
            weights = self._estep_weights(q, actions, critic_obs)
            kl_q = nonparametric_kl_from_weights(weights)
            ess = effective_sample_size(weights)

        actor_loss_val, mstep_info = self._mstep_dime(obs_tiled, traj, old_means, old_scales, weights)

        self._last_actor_info = {
            "eta": self.eta,
            **mstep_info,
            "kl_q": float(kl_q.mean().item()),
            "kl_q_rel": float(kl_q.mean().item()) / max(self.eps_dual, 1e-12),
            "dual_residual_eta": self.eps_dual - float(kl_q.mean().item()),
            "ess": float(ess.mean().item()),
            "ess_min": float(ess.min().item()),
            "solver_status": self._solver_status,
            "solver_iters": self._solver_iters,
            **self._estep_extra_info(),
        }
        # Second return slot is the SAC alpha loss (unused by MPO).
        return actor_loss_val, 0.0
