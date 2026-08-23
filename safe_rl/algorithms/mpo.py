from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Normal, kl_divergence
from typing import Any

from scipy.optimize import minimize

from safe_rl.algorithms.sac import SAC
from safe_rl.modules.sac_actor_critic import SACActorCritic


def nonparametric_kl_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """Per-state ``KL(q* || pi_old)`` actualised by the E-step weights.

    ``weights`` is ``[N, B]`` and normalised over the sample axis, so under the sampling
    distribution ``q*(a_i) = w_i`` and ``pi_old(a_i) = 1/N``, giving ``sum_i w_i log(N w_i)``.
    This is the E-step dual residual's other half: at the dual optimum it equals ``eps_dual``.
    """
    n = weights.shape[0]
    return (weights * torch.log(n * weights + 1e-8)).sum(dim=0)


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Per-state ESS ``(sum w)^2 / sum w^2``; the weights are normalised, so the numerator is 1.

    Ranges from ``N`` (uniform weights, the E-step did nothing) down to 1 (all mass on a single
    sampled action, so the M-step regresses onto one point and its gradient is pure noise).
    """
    return 1.0 / weights.pow(2).sum(dim=0).clamp_min(1e-12)


class MPO(SAC):
    """Maximum a Posteriori Policy Optimization (Abdolmaleki et al., ICLR 2018).

    https://arxiv.org/abs/1806.06920

    Off-policy EM actor-critic. Reuses SAC's twin critics and replay buffer and replaces
    only the actor update:

    * **E-step** — sample ``sample_action_num`` actions per state from the target policy,
      evaluate Q, and solve the 1-D convex dual over the temperature ``eta`` to get the
      non-parametric ``q(a|s) proportional to exp(Q/eta)`` inside a KL trust region.
    * **M-step** — fit the Gaussian policy to the weighted samples by weighted maximum
      likelihood under decoupled mean / covariance KL trust regions.

    Exploration comes from the E-step trust region, so SAC's entropy temperature is off.
    Actions are sampled pre-tanh and squashed only for critic evaluation, so out-of-bound
    actions cannot occur and need no penalty. The M-step log-prob and KLs omit the tanh
    Jacobian deliberately: it is theta-independent, and KL is invariant under a bijection
    applied to both arguments, so the pre-tanh KLs equal the action-space ones.

    Acme-parity options (all default-off; measured comparison in
    codex/mpo-vs-acme-reference.md): ``per_dim_constraining`` (per-action-dim KL budgets
    and multipliers), ``decoupled_mstep`` (split weighted MLE), ``estep_use_target_critic``
    and ``target_actor_update``.
    """

    policy: SACActorCritic

    def __init__(
        self,
        policy: SACActorCritic,
        # --- MPO-specific ---
        sample_action_num: int = 64,
        estep_sample_std_scale: float = 1.0,
        dual_constraint: float = 0.1,  # E-step KL bound (eps in the dual)
        kl_mean_constraint: float = 0.01,  # M-step mean-KL trust region
        kl_var_constraint: float = 1e-4,  # M-step covariance-KL trust region
        alpha_mean_scale: float = 1.0,  # dual ascent step for the mean-KL multiplier
        alpha_var_scale: float = 100.0,  # dual ascent step for the var-KL multiplier
        # Overflow guards only: the multiplier update is an integral controller and a low
        # cap opens the loop, leaving the M-step trust region unenforced.
        alpha_mean_max: float = 10.0,
        alpha_var_max: float = 1000.0,
        mstep_iteration_num: int = 5,
        per_dim_constraining: bool = False,
        decoupled_mstep: bool = False,
        estep_use_target_critic: bool = False,
        target_actor_update: str = "polyak",  # "polyak" | "hard"
        target_actor_period: int = 100,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        # MPO handles exploration through the E-step KL trust region, not SAC entropy.
        # Disable the SAC entropy temperature (alpha ~ 0) and its auto-tuning.
        kwargs.pop("auto_entropy_tuning", None)
        kwargs.pop("alpha", None)
        super().__init__(policy, device=device, auto_entropy_tuning=False, alpha=1e-8, **kwargs)

        self.sample_action_num = int(sample_action_num)
        # Widen the std used ONLY to draw E-step candidates. 1.0 = unchanged (every existing arm).
        # The E-step can only choose among the actions it is shown; if all N candidates are equally
        # safe there is nothing to reweight toward, whatever lambda does. Measured on FH-DCMPO:
        # std_qc = 0.065 against a cost level of ~11, i.e. the 64 candidates differ by 0.6%.
        # Note this makes the proposal wider than pi_old, so the KL that `eps` bounds is measured
        # against a distribution the candidates were NOT drawn from -- watch `kl_q`.
        self.estep_sample_std_scale = float(estep_sample_std_scale)
        if self.estep_sample_std_scale <= 0.0:
            raise ValueError(f"estep_sample_std_scale must be > 0, got {estep_sample_std_scale}")
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
        if target_actor_update not in ("polyak", "hard"):
            raise ValueError(f"target_actor_update must be 'polyak' or 'hard', got {target_actor_update!r}.")
        self.target_actor_update = target_actor_update
        self.target_actor_period = int(target_actor_period)
        self._actor_update_count = 0

        # Frozen target actor: the E-step samples from it and the M-step KL is measured
        # against it. Polyak-averaged toward the online actor after each actor update.
        self.actor_target = deepcopy(self.policy.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # Dual variables (warm-started across batches). The M-step multipliers are arrays so the
        # scalar and per-dimension trust regions share one code path: shape [1] vs [num_actions].
        dual_dim = self.policy.num_actions if self.per_dim_constraining else 1
        self.eta = 1.0  # E-step temperature (warm-start for SLSQP)
        self.alpha_mean = np.zeros(dual_dim)  # M-step mean-KL multiplier(s)
        self.alpha_var = np.zeros(dual_dim)  # M-step var-KL multiplier(s)
        self._solver_status = -1.0
        self._solver_iters = 0.0
        self._last_actor_info: dict[str, float] = {}

        # De-duplicated: num_reward_critics=1 aliases critic_2 onto critic_1.
        self._critic_params = list(self.policy.critic_1.parameters())
        if self.policy.critic_2 is not self.policy.critic_1:
            self._critic_params += list(self.policy.critic_2.parameters())
        group = self.critic_optimizer.param_groups[0]
        self.critic_optimizer = type(self.critic_optimizer)(
            self._critic_params,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            betas=group["betas"],
        )

    def _update_critic_distributional(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        bootstrap: torch.Tensor | None = None,
        effective_n_steps: torch.Tensor | None = None,
        survival_discount: torch.Tensor | None = None,
    ) -> float:
        """Categorical policy evaluation of Q^pi.

        No entropy bonus: MPO explores via the E-step KL trust region. The double-Q min
        stays -- removing it collapsed reward 26.45 -> 3.74 (PointGoal1, seed 2, 60k).
        """
        rewards = rewards.squeeze(-1)
        bootstrap_mask = self._bootstrap_mask(dones, bootstrap).squeeze(-1)
        # Must be gamma**n, not gamma: a bare gamma at n_step=10 cost 14.3 -> 2.9 reward.
        discount = self._bootstrap_discount(effective_n_steps, survival_discount)
        if isinstance(discount, torch.Tensor):
            discount = discount.reshape(-1)
        c1, c2 = self.policy.critic_1, self.policy.critic_2
        t1, t2 = self.policy.critic_1_target, self.policy.critic_2_target

        with torch.no_grad():
            # log_prob unused; the call is kept so the RNG stream matches earlier runs.
            next_actions, _ = self.policy.sample_with_log_prob(next_obs)
            next_obs_norm = self.policy.critic_obs_normalizer(next_critic_obs)

            d1 = t1.get_dist(t1(next_obs_norm, next_actions))
            d2 = t2.get_dist(t2(next_obs_norm, next_actions))
            # Whole lower-mean distribution, not a per-atom min (which neither critic represents).
            use_1 = (t1.get_value(d1) < t2.get_value(d2)).unsqueeze(-1)
            next_dist = torch.where(use_1, d1, d2)

            target_dist = t1.project(
                next_dist=next_dist, rewards=rewards, bootstrap=bootstrap_mask, discount=discount
            )

        obs_normalized = self.policy.critic_obs_normalizer(critic_obs)
        critic_loss = -torch.sum(target_dist * F.log_softmax(c1(obs_normalized, actions), dim=-1), dim=-1).mean()
        if c2 is not c1:
            critic_loss = critic_loss - torch.sum(
                target_dist * F.log_softmax(c2(obs_normalized, actions), dim=-1), dim=-1
            ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
        self.critic_optimizer.step()

        return critic_loss.item()

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
            self._solver_status = float(res.status)
            self._solver_iters = float(res.nit)
            if not np.isfinite(eta):
                raise ValueError("non-finite eta")
        except Exception as exc:  # pragma: no cover - numerical fallback
            print(f"MPO eta solve failed ({exc}); keeping previous eta.")
            eta = self.eta
            self._solver_status = -1.0
            self._solver_iters = 0.0
        return max(eta, 1e-6)

    def _estep_sample(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor):
        """Draw ``sample_action_num`` actions from pi_old and score them under Q.

        Returns ``(x, actions, q, mean_old, std_old)``: ``x`` pre-tanh [N, B, A], ``actions``
        squashed into env bounds, ``q`` the reduced twin-critic value [N, B].
        """
        batch_size = actor_obs.shape[0]
        n = self.sample_action_num
        mean_old, log_std_old = self.actor_target(actor_obs)
        std_old = log_std_old.exp()

        # `std_old` still parameterises pi_old for the M-step KL; only the PROPOSAL is widened.
        sample_std = std_old * self.estep_sample_std_scale
        x = Normal(mean_old, sample_std).sample((n,))  # [N, B, A] pre-tanh
        actions = self.policy.actor.action_b + self.policy.actor.action_c * torch.tanh(x)

        cobs_exp = critic_obs.unsqueeze(0).expand(n, -1, -1).reshape(n * batch_size, -1)
        act_flat = actions.reshape(n * batch_size, -1)
        evaluate = self.policy.evaluate_q_target if self.estep_use_target_critic else self.policy.evaluate_q
        q1, q2 = evaluate(cobs_exp, act_flat)
        q = torch.min(q1, q2)  # same pessimism as the critic target; identical tensors if one critic
        return x, actions, q.reshape(n, batch_size), mean_old, std_old

    def _estep_weights(self, q: torch.Tensor, actions: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Non-parametric weights q(a|s) proportional to exp(Q/eta), softmaxed over samples.

        CVPO overrides this to subtract the cost term; everything else is shared.
        """
        self.eta = self._solve_eta(q.cpu().numpy().astype(np.float64))
        return torch.softmax(q / self.eta, dim=0)  # [N, B], columns sum to 1

    def _mstep(self, actor_obs, x, weights, mean_old, std_old) -> tuple[float, dict[str, float]]:
        """Weighted MLE under decoupled mean / covariance KL trust regions.

        Shared verbatim with CVPO: only the weights differ between the two.
        """
        dist_old_ref = Normal(mean_old, std_old)
        for _ in range(self.mstep_iteration_num):
            mean, log_std = self.policy.actor(actor_obs)  # [B, A]
            std = log_std.exp()

            # Decoupled KL (old || new): mean uses old covariance, covariance uses old mean.
            dist_mean = Normal(mean, std_old)
            dist_var = Normal(mean_old, std)

            if self.decoupled_mstep:
                mle = (weights * dist_mean.log_prob(x).sum(dim=-1)).sum(dim=0).mean() + (
                    weights * dist_var.log_prob(x).sum(dim=-1)
                ).sum(dim=0).mean()
            else:
                log_prob = Normal(mean, std).log_prob(x).sum(dim=-1)  # [N, B]
                mle = (weights * log_prob).sum(dim=0).mean()

            # Summed -> one joint budget; kept as a vector -> one budget per action dim.
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

        kl_mean_val = float(kl_mean_dims.sum().item())
        kl_var_val = float(kl_var_dims.sum().item())
        with torch.no_grad():
            std_d = std.detach()
            std_cond = float((std_d.max(dim=-1).values / std_d.min(dim=-1).values.clamp_min(1e-12)).mean().item())
            # tanh saturates past |x| ~ 2.5, where Q is flat in the pre-tanh mean.
            info = {
                "kl_mean": kl_mean_val,
                "kl_var": kl_var_val,
                # Budget usage, not raw KL / per-dim eps: in per-dim mode the total budget
                # is eps * num_actions, so the two trust-region modes stay comparable.
                "kl_mean_rel": kl_mean_val / max(self.eps_kl_mean * len(self.alpha_mean), 1e-12),
                "kl_var_rel": kl_var_val / max(self.eps_kl_var * len(self.alpha_var), 1e-12),
                "alpha_mean": float(self.alpha_mean.mean()),
                "alpha_var": float(self.alpha_var.mean()),
                "pi_std_min": float(std_d.min().item()),
                "pi_std_max": float(std_d.max().item()),
                "pi_std_cond": std_cond,
                "pretanh_mean_absmax": float(mean.detach().abs().max().item()),
                "frac_saturated": float((mean.detach().abs() > 2.5).float().mean().item()),
            }
        self._sync_target_actor()
        return actor_loss.item(), info

    def _sync_target_actor(self) -> None:
        """Polyak drifts the trust-region anchor every update; "hard" keeps pi_old fixed."""
        self._actor_update_count += 1
        with torch.no_grad():
            if self.target_actor_update == "hard":
                if self._actor_update_count % self.target_actor_period == 0:
                    for p, tp in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                        tp.data.copy_(p.data)
            else:
                for p, tp in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _update_actor_and_alpha(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> tuple[float, float]:
        """MPO E-step + M-step in place of the SAC entropy-regularized actor update.

        ``obs`` / ``critic_obs`` arrive already normalised from :meth:`SAC.update`.
        """
        critic_obs = obs if critic_obs is None else critic_obs
        # The E/M steps call the actor directly, bypassing policy.act/sample which apply
        # this normalizer internally; apply it once here so training and acting agree.
        actor_obs = self.policy.actor_obs_normalizer(obs)

        with torch.no_grad():
            x, actions, q, mean_old, std_old = self._estep_sample(actor_obs, critic_obs)
            weights = self._estep_weights(q, actions, critic_obs)
            # eps_dual - kl_q is the dual residual dg/deta and should sit at ~0; ess
            # collapsing toward 1 means the M-step regresses onto one action per state.
            kl_q = nonparametric_kl_from_weights(weights)
            ess = effective_sample_size(weights)

        actor_loss_val, mstep_info = self._mstep(actor_obs, x, weights, mean_old.detach(), std_old.detach())

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

    def _estep_extra_info(self) -> dict[str, Any]:
        """Diagnostics contributed by a subclass's E-step. Empty for MPO."""
        return {}

    def get_penalty_info(self) -> dict[str, Any]:
        """MPO diagnostics: the E-step temperature and M-step KL / multiplier state."""
        info = {"eta": self.eta}
        info.update(self._last_actor_info)
        return info
