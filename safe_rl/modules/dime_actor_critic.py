"""DIME diffusion actor + REPPO critic, for the REPPODIME algorithm.

Mirrors the TruDi reference's REPPO-DIME `TrainState`: a `DIMEActor` (denoising
diffusion sampler), a frozen `old_actor` copy that is the trust-region KL
reference (hard-synced after each update, `polyak=1.0`), a single critic, and
two `EmpiricalNormalization` instances.

Deliberately NOT a subclass of `StochasticActorCriticBase` — that base is a
Gaussian contract (`_build_distribution -> Normal`, `.distribution`,
`squashed()`), none of which exists for a diffusion policy. The critic
construction and `evaluate_q*` surface are duplicated from `REPPOActorCritic`
instead of refactored out of it, to keep that shared, in-flight file untouched.

The Gaussian `(mu, sigma)` snapshot has no diffusion counterpart, so unlike
`REPPOActorCritic` (which stores per-step `(mu, sigma)` instead of an
`old_actor` — see its docstring) this class DOES carry the frozen copy; it is
the entire memory of pi_old.

Everything the policy returns as a "log prob" is the DIME ELBO pseudo-log-prob
`run_cost + sto_cost + terminal_cost` (an upper bound on log pi(a|s)); the
reference uses it wherever the Gaussian log-prob used to be, including the
soft-return entropy bonus, and so do we.
"""

from __future__ import annotations

import copy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.critic import DistributionalCritic, ReferenceREPPOCritic, StandardCritic
from safe_rl.modules.normalizer import EmpiricalNormalization
from safe_rl.networks.dime import (
    DT_SCHEDULES,
    ControlNetwork,
    DiffusionModel,
    DIMEActor,
    logratio,
    ode_integrator,
    sde_integrator,
    sde_integrator_with_kl,
)


class DIMEActorCritic(nn.Module):
    """DIME diffusion actor + frozen old actor + a single Q(s,a) critic."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        critic_type: str = "reference",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        diffusion: dict[str, Any] | None = None,
        ode_coef: float = 1.0,
        action_scale: float = 1.0,
        compile_score_net: bool = False,
        compile_mode: str = "default",
        critic_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "DIMEActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )
        super().__init__()

        self.num_actions = num_actions
        # Read by REPPO.__init__ to shift target_entropy by +num_actions*log(scale)
        # (reppo.py `_entropy_shift`), so a widened range does not silently demand a
        # log(s)-per-dim sharper policy. Must live on the POLICY, not just the actor.
        self.action_scale = float(action_scale)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        # Score-scaling coefficient `c` for the probability-flow ODE used at
        # deployment (TruDi paper eq.: a^{n-1} = a^n + d(b a^n + c*2 eta^2 b grad log pi)).
        # Their evaluation ablation sweeps this and reports ODE > SDE > best-of-K;
        # c = 1.0 is the reference default (`ode_coefs: [1.0]`).
        self.ode_coef = float(ode_coef)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs, eps_mode="add_var")
            if actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs, eps_mode="add_var")
            if critic_obs_normalization
            else nn.Identity()
        )

        # --------------------------------------------------------------
        # Diffusion actor (reference reppo_dime_torch.yaml defaults)
        # --------------------------------------------------------------
        diffusion = dict(diffusion) if diffusion else {}
        score_cfg = dict(diffusion.pop("score_model", {}))
        sched_cfg = dict(diffusion.pop("dt_schedule", {"type": "cosine", "min": 0.001, "s": 0.008, "pow": 2}))
        diff_steps = int(diffusion.pop("diff_steps", 8))

        sched_type = sched_cfg.pop("type", "cosine")
        if sched_type not in DT_SCHEDULES:
            raise ValueError(f"Unknown dt_schedule type: {sched_type!r}. Must be one of {sorted(DT_SCHEDULES)}")
        if sched_type == "constant":
            dt_schedule = DT_SCHEDULES[sched_type]()
        else:
            dt_schedule = DT_SCHEDULES[sched_type](total_steps=diff_steps, **sched_cfg)

        learn_forward = diffusion.pop("learn_forward", True)
        learn_backward = diffusion.pop("learn_backward", False)
        fwd_model = (
            ControlNetwork(action_dim=num_actions, observation_dim=num_actor_obs, **score_cfg)
            if learn_forward
            else None
        )
        bwd_model = (
            ControlNetwork(action_dim=num_actions, observation_dim=num_actor_obs, **score_cfg)
            if learn_backward
            else None
        )

        diffusion_model = DiffusionModel(
            action_dim=num_actions,
            observation_dim=num_actor_obs,
            fwd_model=fwd_model,
            bwd_model=bwd_model,
            diff_steps=diff_steps,
            dt_schedule=dt_schedule,
            **diffusion,
        )
        self.actor = DIMEActor(
            action_dim=num_actions,
            observation_dim=num_actor_obs,
            diffusion_model=diffusion_model,
            sde_integrator=sde_integrator,
            sde_integrator_with_kl=sde_integrator_with_kl,
            ode_integrator=ode_integrator,
            logratio=logratio,
            # (-action_scale, +action_scale) squashed range; 1.0 == the reference.
            # REPPO's `_entropy_shift` adds the matching +num_actions*log(scale) to
            # target_entropy, so the YAML target keeps its meaning when this widens.
            action_scale=action_scale,
        )
        print(f"REPPO-DIME Actor: {self.actor}")

        # Frozen KL reference — hard-synced by sync_old_actor() after each
        # update (reference: polyak=1.0, requires_grad=False). Lives here so it
        # is covered by state_dict()/to(device) and excluded from the actor
        # optimizer (which enumerates policy.actor.parameters() only).
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)

        # Optional speedup. The chain is `diff_steps` sequential forwards of a SMALL
        # control net — kernel-launch-latency bound, which is exactly what compile
        # (and CUDA-graph capture via mode="reduce-overhead") targets. Uncompiled we
        # pay 4.1x the Gaussian's wall-clock where the paper reports 1.8x.
        #
        # Use nn.Module.compile() (the METHOD, torch>=2.2), NOT torch.compile(mod):
        # the method compiles in place and leaves state_dict keys untouched, whereas
        # the function returns an OptimizedModule whose keys gain an `_orig_mod.`
        # prefix — that would break every existing checkpoint and the old_actor
        # deepcopy/load_state_dict sync. Verified: keys preserved, deepcopy works,
        # and compiled<->uncompiled state_dicts load both ways.
        # Applied AFTER the old_actor deepcopy so the copy is made from a clean module.
        self.compile_score_net = bool(compile_score_net)
        self.compile_mode = str(compile_mode)
        if self.compile_score_net:
            for actor in (self.actor, self.old_actor):
                dm = actor.diffusion_model
                for net in (dm.fwd_model, dm.bwd_model):
                    if net is not None:
                        net.compile(mode=self.compile_mode)

        # --------------------------------------------------------------
        # Critic (identical to REPPOActorCritic)
        # --------------------------------------------------------------
        critic_kwargs = dict(critic_kwargs) if critic_kwargs else {}
        self.critic_type = critic_type
        if critic_type == "standard":
            self.is_distributional_critic = False
            critic_kwargs.setdefault("hidden_dims", [256, 256])
            critic_kwargs.setdefault("activation", "relu")
            self.critic = StandardCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critic = DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        elif critic_type == "reference":
            self.is_distributional_critic = True
            self.critic = ReferenceREPPOCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
        else:
            raise ValueError(
                f"Unknown critic_type: {critic_type}. Must be 'standard', 'distributional' or 'reference'."
            )
        print(f"REPPO-DIME Critic: {self.critic}")

        self._last_log_prob: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Runner compatibility
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_std(self) -> torch.Tensor:
        """Per-dim SDE step-noise scale, averaged over denoising steps.

        The runner logs `policy.action_std.mean()` unconditionally. There is no
        Gaussian sigma here; the closest analog is the per-step transition-noise
        scale sqrt(2 * dt * sched(k) / friction), whose learnable part
        (friction, optionally dt) tracks how much injected noise survives
        training.
        """
        dm = self.actor.diffusion_model
        with torch.no_grad():
            scales = []
            for step in torch.arange(0, dm.diff_steps, dtype=torch.float32):
                eta = dm.delta_t_fn(step) / dm.friction_fn(step)
                scales.append(torch.sqrt(2 * eta).expand(self.num_actions))
            return torch.stack(scales).mean(dim=0)

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(critic_obs)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, deterministic: bool = False, normalized: bool = False) -> torch.Tensor:
        """Sample an action via the stochastic denoising chain.

        `normalized=True` means the caller already applied `actor_obs_normalizer`
        (REPPO normalizes once at collection and stores the result).
        """
        if not normalized:
            obs = self.actor_obs_normalizer(obs)
        if deterministic:
            action, run, sto, term = self.actor.ode_sample(obs, ode_coef=self.ode_coef)
        else:
            action, run, sto, term = self.actor.sde_sample(obs)
        self._last_log_prob = (run + sto + term).detach()
        return action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deployment action via the deterministic (ODE) chain.

        Not fully deterministic: the chain start is still a prior draw, so this
        samples a MODE of the (potentially multimodal) policy conditioned on
        that draw — the desired deployment behavior for a multimodal actor.
        """
        obs = self.actor_obs_normalizer(obs)
        action, *_ = self.actor.ode_sample(obs, ode_coef=self.ode_coef)
        return action

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Pseudo-log-prob of the LAST `act()` sample (cached; `actions` ignored).

        A diffusion policy has no tractable log-prob of an arbitrary action;
        REPPO only ever calls this with the action `act()` just returned, for
        logging/storage.
        """
        if self._last_log_prob is None:
            raise RuntimeError("get_actions_log_prob called before act()")
        return self._last_log_prob

    def sample_with_log_prob(
        self, obs: torch.Tensor, normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample -> (action, pseudo_log_prob, action, ones).

        Keeps the 4-tuple contract of the Gaussian modules so
        `REPPO.process_env_step` / `compute_returns` work unchanged; the (mu,
        sigma) slots are meaningless for a diffusion policy and both call sites
        discard them.
        """
        action, run, sto, term = self.sample_pi(obs, normalized=normalized)
        return action, run + sto + term, action, torch.ones_like(action)

    def sample_pi(
        self, obs: torch.Tensor, normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gradient-carrying sample with the DIME cost decomposition."""
        if not normalized:
            obs = self.actor_obs_normalizer(obs)
        return self.actor.sde_sample(obs)

    # ------------------------------------------------------------------
    # Trust region vs the frozen old actor
    # ------------------------------------------------------------------

    def kl_forward(
        self, obs: torch.Tensor, n_samples: int = 1, with_full: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Path-space KL(pi_old ‖ pi_new): trajectory under old, drift-diff charged to new.

        Second full denoising rollout (× n_samples). `stop_grad=False` — the
        gradient reaches the new model through its per-step forward means only;
        the old-policy trajectory is detached inside the integrator.

        `with_full=True` additionally returns the full closed-form KL (keeping the
        log-variance terms the reference drops) as a third element, for diagnostics
        only — the returned trust-region value is unchanged either way.
        """
        if with_full:
            return self.actor.kl_div_with_full(obs, self.old_actor, n_samples, stop_grad=False)
        return self.actor.kl_div(obs, self.old_actor, n_samples, stop_grad=False)

    def sample_pi_with_kl(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused sample + reverse KL(pi_new ‖ pi_old) — one rollout (`rev_kl` variant)."""
        return self.actor.sde_sample_and_kl(obs, self.old_actor, stop_grad=False)

    def sync_old_actor(self) -> None:
        """Hard-copy actor -> old_actor (reference: polyak=1.0, once per update)."""
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.requires_grad_(False)

    # ------------------------------------------------------------------
    # Q evaluation (identical to REPPOActorCritic)
    # ------------------------------------------------------------------

    def evaluate_q(
        self, obs: torch.Tensor, actions: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        """Scalar Q(s, a). Actions are passed through as-is (already squashed)."""
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        if self.is_distributional_critic:
            logits = self.critic(obs, actions)
            return self.critic.get_value(self.critic.get_dist(logits)).unsqueeze(-1)
        return self.critic(obs, actions)

    def evaluate_q_dist(
        self, obs: torch.Tensor, actions: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        """Raw categorical logits, for the HL-Gauss cross-entropy critic loss."""
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_dist requires a distributional critic")
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        return self.critic(obs, actions)

    def evaluate_q_features(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        normalized: bool = False,
        predict: bool = False,
    ) -> torch.Tensor:
        """Critic trunk features for the self-predictive aux loss."""
        if not hasattr(self.critic, "features"):
            raise RuntimeError(f"{type(self.critic).__name__} exposes no `features` for the aux loss")
        if not normalized:
            obs = self.critic_obs_normalizer(obs)
        features = self.critic.features(obs, actions)
        if predict:
            features = self.critic.predict_features(features)
        return features
