"""DIME diffusion actor + SAC twin critics, for the MPODIME algorithm.

MPO runs on SAC's critic machinery, which `DIMEActorCritic` cannot serve: that
module carries a single REPPO-style critic (scalar `evaluate_q`, no targets, no
`soft_update_targets`) and a 4-tuple `sample_with_log_prob`. This class pairs
the DIME actor block with `SACActorCritic`'s twin-critic block instead. Both
blocks are duplicated rather than refactored out of their homes, following the
precedent set by `DIMEActorCritic` itself ("duplicated from REPPOActorCritic …
to keep that shared, in-flight file untouched").

Differences from `DIMEActorCritic`:
* twin critics + frozen targets, tuple-returning `evaluate_q*`,
  `soft_update_targets` — the SAC/MPO contract (`sac.py`).
* `sample_with_log_prob` returns the 2-tuple ``(action [B, A], log_prob
  [B, 1])`` that `SAC._update_critic*` unpacks. The "log prob" is the DIME
  ELBO pseudo-log-prob; under MPO the entropy coefficient is pinned to 1e-8,
  so it only has to be finite and correctly shaped.
* NO `old_actor`: MPO's algorithm-owned `actor_target` deepcopy is the single
  frozen copy of pi_old (`mpo.py`), Polyak-averaged by `_sync_target_actor`.
"""

from __future__ import annotations

import copy
from typing import Any, NoReturn

import torch
import torch.nn as nn

from safe_rl.modules.critic import DistributionalCritic, StandardCritic
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


class MPODIMEActorCritic(nn.Module):
    """DIME diffusion actor + SAC twin Q(s,a) critics with frozen targets."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        critic_type: str = "standard",
        num_critics: int = 2,
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
                "MPODIMEActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )
        super().__init__()

        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.action_scale = float(action_scale)
        self.actor_type = "dime"
        self.critic_type = critic_type
        self.num_critics = num_critics
        # Score-scaling coefficient for the deployment-time probability-flow ODE
        # (see DIMEActorCritic for the reference ablation; 1.0 is the default).
        self.ode_coef = float(ode_coef)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs, eps_mode="add_var")
            if actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        # --------------------------------------------------------------
        # Diffusion actor (construction mirrors DIMEActorCritic)
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
            action_scale=action_scale,
        )
        print(f"MPO-DIME Actor: {self.actor}")

        # Optional speedup — see DIMEActorCritic for why the compile() METHOD is
        # used (state_dict keys stay unprefixed, deepcopy for MPO's actor_target
        # keeps working).
        self.compile_score_net = bool(compile_score_net)
        self.compile_mode = str(compile_mode)
        if self.compile_score_net:
            dm = self.actor.diffusion_model
            for net in (dm.fwd_model, dm.bwd_model):
                if net is not None:
                    net.compile(mode=self.compile_mode)

        # --------------------------------------------------------------
        # Twin critics + frozen targets (identical to SACActorCritic)
        # --------------------------------------------------------------
        critic_kwargs = dict(critic_kwargs) if critic_kwargs else {}
        if critic_type == "standard":
            self.is_distributional_critic = False
            critic_kwargs.setdefault("hidden_dims", [256, 256])
            critic_kwargs.setdefault("activation", "relu")
            self.critics = nn.ModuleList(
                StandardCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
                for _ in range(num_critics)
            )
        elif critic_type == "distributional":
            self.is_distributional_critic = True
            self.critics = nn.ModuleList(
                DistributionalCritic(num_obs=num_critic_obs, num_actions=num_actions, **critic_kwargs)
                for _ in range(num_critics)
            )
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'standard' or 'distributional'.")
        print(f"MPO-DIME Critic: {self.critics[0]}")

        self.critic_targets = nn.ModuleList(copy.deepcopy(critic) for critic in self.critics)
        for target in self.critic_targets:
            for param in target.parameters():
                param.requires_grad = False

        self.critic_1 = self.critics[0]
        self.critic_2 = self.critics[1] if num_critics > 1 else self.critics[0]
        self.critic_1_target = self.critic_targets[0]
        self.critic_2_target = self.critic_targets[1] if num_critics > 1 else self.critic_targets[0]

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
        """Per-dim SDE step-noise scale, averaged over denoising steps (see DIMEActorCritic)."""
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

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action via the denoising chain (ODE when deterministic)."""
        obs = self.actor_obs_normalizer(obs)
        if deterministic:
            action, run, sto, term = self.actor.ode_sample(obs, ode_coef=self.ode_coef)
        else:
            action, run, sto, term = self.actor.sde_sample(obs)
        self._last_log_prob = (run + sto + term).detach()
        return action

    def act_with_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(obs, deterministic=False)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deployment action via the deterministic (ODE) chain — mode sampling."""
        return self.act(obs, deterministic=True)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Pseudo-log-prob of the LAST `act()` sample (cached; `actions` ignored)."""
        if self._last_log_prob is None:
            raise RuntimeError("get_actions_log_prob called before act()")
        return self._last_log_prob

    def sample_with_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample -> ``(action [B, A], pseudo_log_prob [B, 1])``.

        The 2-tuple contract of `SACActorCritic` — `SAC._update_critic*` unpacks
        exactly two values and multiplies the log-prob by alpha (pinned to 1e-8
        under MPO, so the ELBO pseudo-log-prob only has to be finite).
        """
        obs = self.actor_obs_normalizer(obs)
        action, run, sto, term = self.actor.sde_sample(obs)
        return action, (run + sto + term).unsqueeze(-1)

    def sample_random_action(self, num_envs: int) -> torch.Tensor:
        """Uniform warmup actions in the squashed range (-action_scale, +action_scale)."""
        device = next(self.parameters()).device
        unit = torch.rand(num_envs, self.num_actions, device=device) * 2 - 1
        return self.action_scale * unit

    def as_onnx(self, obs_normalizer: nn.Module | None = None, verbose: bool = False) -> nn.Module:
        raise NotImplementedError(
            "MPODIMEActorCritic: the diffusion actor has no single-forward ONNX export; "
            "use torch checkpoints for deployment."
        )

    # ------------------------------------------------------------------
    # Q evaluation (identical to SACActorCritic)
    # ------------------------------------------------------------------

    @property
    def logits(self) -> torch.Tensor:
        if not self.is_distributional_critic:
            raise RuntimeError("logits only available for distributional critic")
        return self._logits

    @property
    def value_dist(self) -> torch.Tensor:
        if not self.is_distributional_critic:
            raise RuntimeError("value_dist only available for distributional critic")
        return self._value_dist

    def evaluate_q(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(obs)
        if self.is_distributional_critic:
            logits_1 = self.critic_1(obs, actions)
            logits_2 = self.critic_2(obs, actions)
            dist_1 = self.critic_1.get_dist(logits_1)
            dist_2 = self.critic_2.get_dist(logits_2)
            q1 = self.critic_1.get_value(dist_1).unsqueeze(-1)
            q2 = self.critic_2.get_value(dist_2).unsqueeze(-1)
            self._logits = torch.stack([logits_1, logits_2], dim=1)
            self._value_dist = dist_1
        else:
            q1 = self.critic_1(obs, actions)
            q2 = self.critic_2(obs, actions)
        return q1, q2

    def evaluate_q_dist(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_dist only available for distributional critics")
        obs = self.critic_obs_normalizer(obs)
        return self.critic_1(obs, actions), self.critic_2(obs, actions)

    def evaluate_q_target(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(obs)
        if self.is_distributional_critic:
            logits_1 = self.critic_1_target(obs, actions)
            logits_2 = self.critic_2_target(obs, actions)
            dist_1 = self.critic_1_target.get_dist(logits_1)
            dist_2 = self.critic_2_target.get_dist(logits_2)
            q1_target = self.critic_1_target.get_value(dist_1).unsqueeze(-1)
            q2_target = self.critic_2_target.get_value(dist_2).unsqueeze(-1)
        else:
            q1_target = self.critic_1_target(obs, actions)
            q2_target = self.critic_2_target(obs, actions)
        return q1_target, q2_target

    def evaluate_q_target_dist(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_distributional_critic:
            raise RuntimeError("evaluate_q_target_dist only available for distributional critics")
        obs = self.critic_obs_normalizer(obs)
        return self.critic_1_target(obs, actions), self.critic_2_target(obs, actions)

    def soft_update_targets(self, tau: float) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
