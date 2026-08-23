"""REPPO with a DIME diffusion actor (TruDi's `reppo_dime` — our port).

The TruDi reference (`trudi/src/torchrl/reppo_dime.py`) is a copy-paste fork of
its `reppo.py` — the exact file `safe_rl.algorithms.reppo.REPPO` is calibrated
against — with ONLY the actor swapped for a DIME denoising-diffusion sampler.
This class expresses that same diff as a subclass, so everything the fork kept
byte-identical (collection bookkeeping, soft-lambda returns, HL-Gauss critic +
aux loss, dual algebra, minibatch loop) stays literally shared, and only the
Gaussian seams are overridden:

  * `act()`            — no `(mu, sigma)` snapshot; the storage buffers get
                         zeros/ones and the old policy lives in the module's
                         frozen `old_actor` instead.
  * `_update_actor()`  — pseudo-log-prob = ELBO cost sum; entropy = -run_cost
                         (reference drops the terminal prior term here);
                         KL = path-space KL against `old_actor`.
  * `update()`         — appends the hard old-actor sync (polyak=1.0, after
                         all epochs — reference sync location).

`process_env_step` / `compute_returns` are NOT overridden: they reach the
policy only through `sample_with_log_prob`, which `DIMEActorCritic` provides
with the pseudo-log-prob in the log-prob slot — making the soft-return entropy
bonus exactly the reference's `r - gamma * temperature * (run+sto+term)`.

Trust-region variants (reference ships both):
  * `dime_kl_mode="forward"` (default, their headline config): forward
    path-KL(old‖new) via a SECOND full denoising rollout per minibatch
    (× `kl_action_rep`).
  * `dime_kl_mode="reverse"`: reverse KL(new‖old) fused into the sampling
    rollout — one rollout, ~half the actor-update cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from safe_rl.algorithms.reppo import REPPO


class REPPODIME(REPPO):
    """REPPO update with a diffusion policy and a frozen-old-actor trust region."""

    def __init__(
        self,
        policy,
        kl_action_rep: int = 1,
        dime_kl_mode: str = "forward",
        log_full_kl: bool = True,
        trudi_wandb_schema: bool = False,
        **kwargs,
    ) -> None:
        for method in ("sample_pi", "kl_forward", "sample_pi_with_kl", "sync_old_actor"):
            if not hasattr(policy, method):
                raise TypeError(
                    f"REPPODIME needs a diffusion policy exposing `{method}` "
                    f"(e.g. DIMEActorCritic); got {type(policy).__name__}"
                )
        if dime_kl_mode not in ("forward", "reverse"):
            raise ValueError(f"dime_kl_mode must be 'forward' or 'reverse'; got {dime_kl_mode!r}")
        if kwargs.get("target_entropy_final") is not None:
            raise ValueError(
                "target_entropy_final is Gaussian-calibrated (differential entropy); the DIME "
                "pseudo-entropy (-run_cost) lives on a different scale — no anneal is supported."
            )
        self.kl_action_rep = int(kl_action_rep)
        self.dime_kl_mode = dime_kl_mode
        # Diagnostic only (forward mode): also accumulate the full closed-form KL
        # alongside the reference's simplified one. Same rollout, same RNG, same
        # trust-region value used by the loss — a few extra elementwise ops.
        self.log_full_kl = bool(log_full_kl) and dime_kl_mode == "forward"
        # Mirror our metrics under the TruDi reference's wandb key names so our runs
        # and the authors' runs overlay on the same charts (see
        # safe_rl/utils/trudi_wandb_schema.py). Additive — our own keys are kept.
        # Passed THROUGH to REPPO rather than set here: the base __init__ assigns
        # this attribute too, so setting it before super() would be overwritten.
        kwargs["trudi_wandb_schema"] = bool(trudi_wandb_schema)
        # Extra per-minibatch diagnostics accumulated by _update_actor and
        # merged into update()'s metric dict (the parent's update() only
        # aggregates its fixed key list, so extras ride their own accumulator).
        self._extra_sums: dict[str, float] = {}
        self._extra_count = 0
        super().__init__(policy, **kwargs)

    # ------------------------------------------------------------------
    # Collection — no (mu, sigma) snapshot; old policy is the frozen old_actor
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        # Same normalize-once discipline as the parent (see REPPO.act).
        norm_obs = self._normalize(self.policy.actor_obs_normalizer, obs, update_stats=True)
        norm_critic_obs = self._normalize(
            self.policy.critic_obs_normalizer, critic_obs, update_stats=True
        )

        with torch.no_grad():
            action, run, sto, term = self.policy.sample_pi(norm_obs, normalized=True)
            value = self.policy.evaluate_q(norm_critic_obs, action, normalized=True)
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        self.transition.actions = action
        self.transition.values = value
        self.transition.actions_log_prob = run + sto + term
        # Dead buffers: a diffusion policy has no (mu, sigma) summary. The KL
        # reference is the frozen old_actor on the policy module; storage keeps
        # its layout so RolloutStorage stays untouched.
        self.transition.action_mean = torch.zeros_like(action)
        self.transition.action_sigma = torch.ones_like(action)
        self.transition.observations = norm_obs
        self.transition.privileged_observations = norm_critic_obs
        return action

    # ------------------------------------------------------------------
    # Actor update — ELBO pseudo-log-prob + path-space KL vs old_actor
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
    ) -> dict[str, float]:
        # old_mu / old_sigma are the dead storage buffers — ignored.
        del old_mu, old_sigma

        kl_full = None
        if self.dime_kl_mode == "forward":
            # Reference main variant: sampling rollout + a second full rollout
            # for the forward path-KL (kl_div walks the OLD policy's chain).
            action_pi, run_cost, sto_cost, terminal_cost = self.policy.sample_pi(obs, normalized=True)
            if self.log_full_kl:
                _, kl, kl_full = self.policy.kl_forward(obs, self.kl_action_rep, with_full=True)
            else:
                _, kl = self.policy.kl_forward(obs, self.kl_action_rep)
        else:
            # rev_kl variant: one fused rollout under the NEW policy.
            action_pi, run_cost, sto_cost, terminal_cost, kl = self.policy.sample_pi_with_kl(obs)
        log_prob_pi = run_cost + sto_cost + terminal_cost

        # Pathwise Q — identical to the parent (and to the reference, which
        # also differentiates straight through the denoising chain).
        self._set_critic_grad(requires_grad=False)
        q_pi = self.policy.evaluate_q(critic_obs, action_pi, normalized=True).squeeze(-1)

        primary = self.alpha_temp.detach() * log_prob_pi - q_pi

        if self.kl_clip_mode == "clipped":
            actor_loss = torch.where(
                kl < self.desired_kl, primary, self.alpha_kl.detach() * kl
            ).mean()
        else:
            actor_loss = (primary + self.alpha_kl.detach() * kl).mean()

        # Reference (reppo_dime.py): entropy = -run_cost — the terminal prior
        # log-prob is excluded from the temperature target, while the FULL
        # pseudo-log-prob feeds the actor loss and the reward-side bonus. That
        # asymmetry is the reference's; preserve it.
        entropy = -run_cost

        # Mode-vs-sample gap under the critic — the multimodality diagnostic
        # this integration exists to move (parent's version is Gaussian-mode
        # based; here the mode analog is the ODE chain).
        with torch.no_grad():
            mode_action, *_ = self.policy.actor.ode_sample(obs)
            q_mode = self.policy.evaluate_q(critic_obs, mode_action, normalized=True).squeeze(-1)
            deployment_gap = (q_pi.detach() - q_mode).mean()

        # Extended per-minibatch diagnostics (merged into update()'s dict).
        # sto_cost is identically zero in the reference and is not logged.
        with torch.no_grad():
            gate_active = (
                (kl >= self.desired_kl).float().mean() if self.kl_clip_mode == "clipped"
                else torch.zeros(())
            )
            extras = {
                "dime_run_cost": run_cost.mean().item(),
                "dime_terminal_cost": terminal_cost.mean().item(),
                # Identically zero by construction in the vendored sampler
                # (`stochastic_costs = zeros_like(running_cost)`), but the reference
                # logs it, so log it too — a silently-absent metric reads as a
                # divergence when overlaying our run against theirs.
                "dime_sto_cost": sto_cost.mean().item(),
                "dime_log_prob": log_prob_pi.mean().item(),
                "kl_max": kl.max().item(),
                "kl_std": kl.std().item(),
                "kl_frac_over_bound": gate_active.item(),
                "q_mode": q_mode.mean().item(),
                # tanh saturation — the Gaussian failure mode on grasp tasks.
                "dime_action_sat_frac": (action_pi.detach().abs() > 0.99).float().mean().item(),
                "dime_action_abs_mean": action_pi.detach().abs().mean().item(),
                # SDE-sample vs ODE-mode spread in action space (independent
                # prior draws, so this measures policy spread, not chain noise).
                "dime_sde_ode_action_gap": (action_pi.detach() - mode_action).abs().mean().item(),
                "entropy_full_pseudo": (-log_prob_pi).mean().item(),
            }
            if kl_full is not None:
                # The reference's per-step KL drops the log-variance terms, so it is
                # exact only while old/new friction agree. This measures what the
                # trust region is actually giving away on each update.
                extras["kl_full_closed_form"] = kl_full.mean().item()
                extras["kl_full_minus_simplified"] = (kl_full - kl).mean().item()
                extras["kl_full_frac_over_bound"] = (
                    (kl_full >= self.desired_kl).float().mean().item()
                )
        for key, value in extras.items():
            self._extra_sums[key] = self._extra_sums.get(key, 0.0) + value
        self._extra_count += 1

        # Dual losses — identical algebra to the parent (and, with
        # dual_optim_mode="actor" + target_entropy = -ent_target_mult * n_act,
        # algebraically identical to the reference's entropy/lagrangian losses).
        alpha_temp_loss = self.alpha_temp * (entropy.mean().detach() - self.target_entropy)
        alpha_kl_loss = self.alpha_kl * (self.desired_kl - kl.mean().detach())

        self.optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        (actor_loss + alpha_temp_loss + alpha_kl_loss).backward()
        clip_params: list[nn.Parameter] = list(self.policy.actor.parameters())
        if self.dual_optim_mode == "actor":
            clip_params += [self.log_alpha_temp, self.log_alpha_kl]
        actor_grad_norm = nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
        self.optimizer.step()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        self._set_critic_grad(requires_grad=True)

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "kl": kl.mean().item(),
            "q_value": q_pi.mean().item(),
            "alpha_temp_loss": alpha_temp_loss.item(),
            "alpha_kl_loss": alpha_kl_loss.item(),
            "deployment_gap": deployment_gap.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": getattr(self, "_last_critic_grad_norm", 0.0),
        }

    # ------------------------------------------------------------------
    # Update — parent loop + hard old-actor sync (reference location: after
    # ALL epochs, polyak = 1.0)
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        metrics = super().update()
        self.policy.sync_old_actor()

        # Merge the per-minibatch extras (means over the update's minibatches).
        n = max(self._extra_count, 1)
        for key, total in self._extra_sums.items():
            metrics[key] = total / n
        self._extra_sums.clear()
        self._extra_count = 0

        # Per-iteration SDE parameter state (these move slowly; end-of-update
        # values suffice).
        dm = self.policy.actor.diffusion_model
        with torch.no_grad():
            friction = torch.nn.functional.softplus(dm.friction)
            metrics["dime_friction"] = friction.mean().item()
            metrics["dime_friction_min"] = friction.min().item()
            metrics["dime_friction_max"] = friction.max().item()
            metrics["dime_dt"] = torch.nn.functional.softplus(dm.dt).mean().item()
            # Per-step transition-noise scale summary (the action_std analog).
            metrics["dime_noise_scale"] = self.policy.action_std.mean().item()

        if self.trudi_wandb_schema:
            from safe_rl.utils.trudi_wandb_schema import add_trudi_aliases

            metrics = add_trudi_aliases(metrics)
        return metrics
