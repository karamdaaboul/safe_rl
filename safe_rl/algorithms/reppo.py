from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from safe_rl.modules import REPPOActorCritic, RewardNormalization
from safe_rl.storage import RolloutStorage


class REPPO:
    """Relative Entropy Pathwise Policy Optimization — Q(s,a) variant.

    Faithful to https://arxiv.org/abs/2507.11019: the actor is updated by
    pathwise (reparameterized) gradients through a single Q(s,a) critic, so
    ∂Q/∂a · ∂a/∂θ carries the reward signal directly. The critic is trained
    on a soft-Q λ-target computed once per update from the online nets:

        a'_t ~ π_target(s'_t)
        soft_V(s'_t) = Q(s'_t, a'_t) − α · log_prob(a'_t)
        target_q[t]  = λ-blended return on (r_t − α · log_prob_t) bootstrapped
                       on soft_V(s'_t), masking γ-bootstrap by (1 − done | truncated)

    Actor loss (pathwise):
        a_π = π(s).rsample();  q_π = Q(s, a_π)
        L_actor = E[α · log π(a_π|s) − q_π]
    REPPO gate: replace L_actor with α_kl · KL(π_old ‖ π) when KL exceeds the
    desired bound. KL is closed-form on the raw Gaussian (μ, σ).

    Single-GPU only — multi-GPU sync removed.
    """

    policy: REPPOActorCritic

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 8,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 3e-4,
        critic_learning_rate: float | None = None,
        alpha_lr: float = 3e-4,
        optimizer_class: str = "adamw",
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.1,
        target_entropy: float = -1.0,
        target_entropy_final: float | None = None,
        target_entropy_anneal_start: int = 0,
        target_entropy_anneal_end: int = 1,
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        kl_clip_mode: str = "full",
        dual_optim_mode: str = "separate",
        force_last_step_truncated: bool = False,
        critic_loss_denominator: str = "mask",
        aux_loss_mult: float = 0.0,
        aux_reward_pred: bool = False,
        reward_scale: float = 1.0,
        trudi_wandb_schema: bool = False,
        reward_normalization: bool = False,
        reward_norm_g_max: float = 10.0,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if multi_gpu_cfg is not None:
            raise NotImplementedError("REPPO is single-GPU only.")
        self.device = device
        self.is_multi_gpu = False
        self.gpu_global_rank = 0
        self.gpu_world_size = 1

        # Runner-compat stubs
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None
        self.symmetry = None

        self.policy = policy
        self.policy.to(self.device)

        # KL is computed closed-form on raw (μ, σ) against the rollout-time
        # distribution stored per step (old_mu/old_sigma), which equals the
        # policy at the start of the update for on-policy collection — so no
        # separate frozen old-policy snapshot is needed.

        # Algorithm-level learnable scalars.
        self.log_alpha_temp = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha_temp), device=self.device))
        )
        self.log_alpha_kl = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha_kl), device=self.device))
        )

        # Optimizers — actor on `self.optimizer` so the runner's checkpoint code finds it.
        # Defaults keep the historical AdamW(wd=1e-3, betas=(0.9, 0.95)); the reference
        # REPPO uses plain Adam with default betas and NO weight decay, so both are
        # reachable from config for the ref-match ablation.
        if optimizer_class.lower() == "adamw":
            opt_cls: type[optim.Optimizer] = optim.AdamW
        elif optimizer_class.lower() == "adam":
            opt_cls = optim.Adam
        else:
            raise ValueError(f"optimizer_class must be 'adam' or 'adamw'; got {optimizer_class!r}")
        opt_kwargs = {"betas": tuple(betas)}
        if opt_cls is optim.AdamW:
            opt_kwargs["weight_decay"] = weight_decay
        # Where the duals live. In the reference, `log_temp` / `log_lagrange` are
        # nn.Parameters INSIDE the Actor, so they (a) ride the single actor optimizer
        # at `lr` and (b) are covered by clip_grad_norm_(actor.parameters(), ...) —
        # when the actor grad norm exceeds max_grad_norm the dual gradients are
        # scaled down by the same factor. Our historical "separate" mode gives them
        # their own optimizer at `alpha_lr` and never clips them, so the duals move
        # on a different trajectory. Both reachable from config.
        if dual_optim_mode not in ("separate", "actor"):
            raise ValueError(f"dual_optim_mode must be 'separate' or 'actor'; got {dual_optim_mode!r}")
        self.dual_optim_mode = dual_optim_mode

        actor_params = list(policy.actor.parameters())
        if self.dual_optim_mode == "actor":
            actor_params = actor_params + [self.log_alpha_temp, self.log_alpha_kl]
        self.optimizer = opt_cls(actor_params, lr=learning_rate, **opt_kwargs)
        critic_lr = critic_learning_rate if critic_learning_rate is not None else learning_rate
        self.critic_optimizer = opt_cls(policy.critic.parameters(), lr=critic_lr, **opt_kwargs)
        if self.dual_optim_mode == "separate":
            self.alpha_optimizer: optim.Optimizer | None = opt_cls(
                [self.log_alpha_temp, self.log_alpha_kl], lr=alpha_lr, **opt_kwargs
            )
        else:
            self.alpha_optimizer = None

        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # Per-step bootstrap quantities computed during collection (reference
        # collect_fn semantics); stacked in compute_returns, cleared in update().
        self._collect_next_values: list[torch.Tensor] = []
        self._collect_ent_bonus: list[torch.Tensor] = []
        self._collect_aux_targets: list[torch.Tensor] = []

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        # Reference REPPO has NO target networks — the bootstrap uses the online
        # critic. Retained as an option; the single critic is not optional.
        # Reference actor_kl_clip_mode: "clipped" (author default) hard-gates each
        # sample — when its KL exceeds the bound the reward loss is REPLACED by
        # alpha_kl*KL, structurally stopping oversized steps. "full" is the soft
        # penalty variant (also a supported reference mode).
        if kl_clip_mode not in ("full", "clipped"):
            raise ValueError(f"kl_clip_mode must be 'full' or 'clipped'; got {kl_clip_mode!r}")
        self.kl_clip_mode = kl_clip_mode
        # Reference compute_gve does `truncated[-1] = 1.0` IN PLACE, and the same
        # tensor is what the critic update reads. So the last rollout step both
        # bootstraps 1-step (no lambda blend, no termination mask) and is dropped
        # from the critic loss, for every env. We keep the real flag by default.
        self.force_last_step_truncated = bool(force_last_step_truncated)
        # Reference normalizes the truncation-masked critic/aux losses by the FULL
        # batch ((mask * ce).mean()); we divide by mask.sum(). Identical when
        # nothing is masked, ~1% apart at typical truncation rates — and the gap
        # widens with force_last_step_truncated, which masks a further 1/T.
        if critic_loss_denominator not in ("mask", "batch"):
            raise ValueError(
                f"critic_loss_denominator must be 'mask' or 'batch'; got {critic_loss_denominator!r}"
            )
        self.critic_loss_denominator = critic_loss_denominator
        # Self-predictive aux loss weight (reference: aux_loss_mult * MSE between
        # critic features of (s,a) and sg[features of (s',a')]).
        self.aux_loss_mult = float(aux_loss_mult)
        self._aux_targets: torch.Tensor | None = None
        # Reference (JAX) aux loss also regresses a one-step reward prediction and
        # averages it TOGETHER with the per-feature errors over D+1 slots, masked by
        # (1 - done):
        #   aux = mean((1-done) * concat[(pred_f - next_emb)^2, (pred_r - r)^2], -1)
        # (`jaxrl/reppo.py:493-501`). Off by default -- the torch reference this port
        # was originally written against has the embedding term only, so enabling it
        # unconditionally would change every existing REPPO config.
        self.aux_reward_pred = bool(aux_reward_pred)
        self._aux_rewards: torch.Tensor | None = None
        self._aux_dones: torch.Tensor | None = None
        # Reward scaling (reference REPPO env.reward_scaling): dt-scaled sim
        # rewards make per-step reward ~0.01, so Q-differences are tiny next to
        # the fixed-scale KL trust-region cost and the actor barely moves.
        # Scaling inside the algorithm keeps runner/wandb reward logging raw.
        self.reward_scale = float(reward_scale)
        # Emit our metrics ALSO under the TruDi reference's wandb key names, so a
        # run of ours and a run of theirs can be overlaid on one chart. See
        # safe_rl/utils/trudi_wandb_schema.py. Off by default; set true in the
        # ManiSkill comparison configs.
        self.trudi_wandb_schema = bool(trudi_wandb_schema)
        # Adaptive alternative (reference RewardNormalizer): divide rewards by
        # max(std(G), max|G|/g_max) of the running discounted return — removes
        # the per-task reward_scale tuning and auto-fits the critic support to
        # roughly [-g_max, g_max]. Applied before reward_scale.
        if reward_normalization:
            self.reward_normalizer: RewardNormalization | None = RewardNormalization(
                gamma=gamma, g_max=reward_norm_g_max
            ).to(self.device)
        else:
            self.reward_normalizer = None
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate

        # Resolve the action dimension for the entropy target. Prefer the
        # wrapper policy's num_actions (always set on REPPOActorCritic) — the
        # actor-attribute fallback silently returned 1 for StochasticActor
        # (which lacked .num_actions), scaling the target to -0.5 instead of
        # -0.5*n_act and turning the temperature dual into a sigma-pump that
        # pinned entropy at -0.5 across v10-v12.
        num_actions = getattr(
            policy, "num_actions", getattr(getattr(policy, "actor", None), "num_actions", 1)
        )
        self.target_entropy = target_entropy * num_actions
        # Widening the squashed action range to (-s, +s) adds an exact
        # +num_actions*log(s) to the differential entropy of every policy. Shift the
        # target by the same constant so `target_entropy: -0.5` keeps meaning "this
        # sharp relative to the action range" instead of silently demanding a
        # log(s)-per-dim sharper policy the moment the range is widened.
        action_scale = float(getattr(policy, "action_scale", 1.0))
        self._entropy_shift = num_actions * math.log(action_scale) if action_scale != 1.0 else 0.0
        self.target_entropy += self._entropy_shift

        # Optional target-entropy anneal. The target is held at its initial value
        # while the critic is still forming (annealing a temperature against a
        # critic that does not yet rank actions just sharpens toward noise), then
        # moves linearly to `target_entropy_final` between the two iteration marks.
        # Both endpoints are PER ACTION DIMENSION, like `target_entropy`, and both
        # get the same action-scale shift so the schedule means the same thing at
        # any action range.
        self._num_actions = num_actions
        self._target_entropy_start = self.target_entropy
        self._target_entropy_final = (
            target_entropy_final * num_actions + self._entropy_shift
            if target_entropy_final is not None
            else None
        )
        self._anneal_start = int(target_entropy_anneal_start)
        self._anneal_end = int(target_entropy_anneal_end)
        if self._target_entropy_final is not None and self._anneal_end <= self._anneal_start:
            raise ValueError(
                "target_entropy_anneal_end must exceed target_entropy_anneal_start; got "
                f"{self._anneal_start} -> {self._anneal_end}"
            )
        self._iteration = 0

        self._final_obs_warned = False

    @property
    def alpha_temp(self) -> torch.Tensor:
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        return self.log_alpha_kl.exp()

    def extra_state_dict(self) -> dict:
        """Algorithm state the runner's checkpoint does not otherwise capture.

        `policy.state_dict()` covers the networks and obs normalizers, and the
        runner saves `self.optimizer`. That leaves the duals (which live here, not
        on the policy) and the critic/alpha optimizer moments. Without these a
        "resume" silently restarts the dual dynamics: alpha_temp jumps from its
        converged value back to `init_alpha_temp`, re-injecting entropy bonus into
        the critic targets.
        """
        state = {
            "log_alpha_temp": self.log_alpha_temp.detach().clone(),
            "log_alpha_kl": self.log_alpha_kl.detach().clone(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "iteration": self._iteration,
        }
        if self.alpha_optimizer is not None:
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def load_extra_state(self, state: dict) -> None:
        """Restore what `extra_state_dict` saved. Tolerates older checkpoints."""
        if not state:
            return
        with torch.no_grad():
            if "log_alpha_temp" in state:
                self.log_alpha_temp.copy_(state["log_alpha_temp"].to(self.device))
            if "log_alpha_kl" in state:
                self.log_alpha_kl.copy_(state["log_alpha_kl"].to(self.device))
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "alpha_optimizer" in state and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self._iteration = int(state.get("iteration", self._iteration))

    def _advance_target_entropy(self) -> None:
        """Linear target-entropy anneal, held flat until `target_entropy_anneal_start`.

        No-op unless `target_entropy_final` is configured, so every existing config
        keeps a constant target.
        """
        self._iteration += 1
        if self._target_entropy_final is None:
            return
        if self._iteration <= self._anneal_start:
            self.target_entropy = self._target_entropy_start
            return
        frac = (self._iteration - self._anneal_start) / (self._anneal_end - self._anneal_start)
        frac = min(max(frac, 0.0), 1.0)
        self.target_entropy = (
            self._target_entropy_start
            + frac * (self._target_entropy_final - self._target_entropy_start)
        )

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
            store_next_obs=True,
        )

    def test_mode(self) -> None:
        self.policy.eval()

    def train_mode(self) -> None:
        self.policy.train()

    @staticmethod
    def _normalize(normalizer: nn.Module, x: torch.Tensor, update_stats: bool) -> torch.Tensor:
        """Apply an observation normalizer, optionally without updating its statistics."""
        if update_stats or not normalizer.training:
            return normalizer(x)
        normalizer.eval()
        try:
            return normalizer(x)
        finally:
            normalizer.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        # Normalize ONCE here and store the normalized tensors (reference REPPO stores
        # normalized obs in its rollout buffer). Everything downstream — the bootstrap in
        # compute_returns and every minibatch pass in update() — then runs on exactly the
        # inputs the collecting policy saw, so KL(pi_old || pi_new) measures the policy
        # change and nothing else. Storing raw obs and re-normalizing later charged the
        # trust region for observation-statistic drift across the rollout.
        norm_obs = self._normalize(self.policy.actor_obs_normalizer, obs, update_stats=True)
        norm_critic_obs = self._normalize(
            self.policy.critic_obs_normalizer, critic_obs, update_stats=True
        )

        action = self.policy.act(norm_obs, normalized=True).detach()
        # Stored value is the reduced Q(s, a) — for logging only; not used in updates.
        value = self.policy.evaluate_q(norm_critic_obs, action, normalized=True).detach()
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        self.transition.actions = action
        self.transition.values = value
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(action).detach()
        # Per-state (μ, σ) from the actual sampling distribution — with a
        # state-dependent-σ actor the batch-averaged policy.action_std property
        # would corrupt the closed-form KL, so read the distribution directly.
        dist = self.policy.distribution
        self.transition.action_mean = dist.mean.detach()
        self.transition.action_sigma = dist.stddev.detach().expand_as(action)
        self.transition.observations = norm_obs
        self.transition.privileged_observations = norm_critic_obs
        return action

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_obs: torch.Tensor | None = None,
        next_critic_obs: torch.Tensor | None = None,
    ) -> None:
        scaled_rewards = rewards.clone()
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(rewards.view(-1), dones.view(-1).float())
            scaled_rewards = self.reward_normalizer(scaled_rewards)
        self.transition.rewards = scaled_rewards * self.reward_scale
        self.transition.dones = dones

        time_outs = infos.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.float32)
        else:
            time_outs = time_outs.to(self.device).float()

        # Truth on truncation: use info["final_observation"] if surfaced by the
        # env wrapper. Otherwise fall back to the post-step obs (which is the
        # auto-reset obs for truncated envs — wrong for Q-bootstrap, but masked
        # to zero anyway when the env wrapper doesn't truncate).
        if next_obs is not None:
            next_actor_obs = next_obs.clone()
            next_priv_obs = next_critic_obs.clone() if next_critic_obs is not None else next_obs.clone()
            final_obs = infos.get("final_observation")
            if final_obs is not None and time_outs.any():
                final_actor = final_obs.get("actor", final_obs) if isinstance(final_obs, dict) else final_obs
                final_actor = final_actor.to(self.device)
                mask = time_outs.bool().view(-1)
                next_actor_obs[mask] = final_actor[mask]
                if isinstance(final_obs, dict) and "critic" in final_obs:
                    final_critic = final_obs["critic"].to(self.device)
                    next_priv_obs[mask] = final_critic[mask]
                else:
                    next_priv_obs[mask] = final_actor[mask]
            elif final_obs is None and time_outs.any() and not self._final_obs_warned:
                print(
                    "[REPPO] WARNING: env truncated some episodes but did not provide "
                    "infos['final_observation']; Q-bootstrap on truncation will use the "
                    "auto-reset observation (wrong). Add final_observation forwarding to the "
                    "env wrapper."
                )
                self._final_obs_warned = True
            # Reference collect_fn, verbatim semantics: next_obs is normalized in
            # TRAIN mode (stats update on next_obs too — the reference normalizer
            # sees every observation twice per step), and ALL bootstrap
            # quantities are computed HERE, per step, with the collection-time
            # normalizer statistics:
            #   a' ~ pi(s'), logp', V' = Q(s', a'), aux features(s', a'),
            #   soft reward r - gamma * alpha * logp'.
            # Previously these were computed once at update start with
            # end-of-rollout statistics — equal in expectation, but a different
            # early-training trajectory than the reference.
            norm_next_actor = self._normalize(
                self.policy.actor_obs_normalizer, next_actor_obs, update_stats=True
            )
            norm_next_priv = self._normalize(
                self.policy.critic_obs_normalizer, next_priv_obs, update_stats=True
            )
            self.transition.next_observations = norm_next_actor
            self.transition.next_privileged_observations = norm_next_priv

            with torch.no_grad():
                # Reference REPPO bootstraps from the ONLINE actor and critic; the
                # per-iteration freezing of these values at collection is the target
                # mechanism, so no polyak copy is involved.
                next_a, next_logp, _, _ = self.policy.sample_with_log_prob(
                    norm_next_actor, normalized=True
                )
                next_q = self.policy.evaluate_q(norm_next_priv, next_a, normalized=True)
                next_v = next_q.view(-1, 1)
                alpha = self.alpha_temp.detach()
                ent_bonus = (-self.gamma * alpha * next_logp).view(-1, 1)
                self._collect_next_values.append(next_v)
                self._collect_ent_bonus.append(ent_bonus)
                if self.aux_loss_mult > 0.0:
                    self._collect_aux_targets.append(
                        self.policy.evaluate_q_features(norm_next_priv, next_a, normalized=True)
                    )

        self.transition.truncated = time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    # ------------------------------------------------------------------
    # Soft-Q λ targets — computed once per update from frozen target actor
    # ------------------------------------------------------------------

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Soft-Q λ-target.

        Per stored step t (reference torchrl form, exactly):
          a'_t ~ π(s'_t)
          r'_t        = r_t − γ · α · log_prob(a'_t | s'_t)   (entropy in the REWARD)
          target_q[t] = r'_t + γ · m_t · ((1−λ) · Q(s'_t, a'_t) + λ · target_q[t+1])
        where m_t = (1 − dones_t) | truncated_t (truncations keep bootstrap).

        The entropy bonus is counted once per step at FULL weight via r'_t.
        (Two wrong variants seen before: subtracting α·logπ(a_t|s_t) as well
        double-counts entropy; putting −α·logp' inside the (1−λ)-blended
        bootstrap instead undercounts the future entropy chain by (1−λ).)
        On truncation the λ-trace is cut (blend → Q' only) so the next
        episode's return does not leak backwards (matches reference compute_gve).
        """
        alpha = self.alpha_temp.detach().item()
        T = self.storage.num_transitions_per_env

        # Reference compute_gve mutates `truncated[-1] = 1.0` before the recursion,
        # and the critic update reads that same buffer. Done here (in place, on the
        # storage buffer) so both the λ recursion below and the critic's truncation
        # mask see it, exactly as in the reference.
        if self.force_last_step_truncated:
            self.storage.truncated[-1] = 1.0

        # Observations in storage are already normalized (see act()), so nothing on this
        # path touches the normalizers — no freeze/thaw dance is needed and the KL is
        # measured on exactly the inputs the collecting policy saw.

        # Bootstrap quantities: preferred path uses the per-step values computed
        # DURING collection (reference collect_fn semantics — collection-time
        # normalizer statistics and per-step temperature). Fallback recomputes at
        # update start for callers that do not pass next_obs through
        # process_env_step (kept for tests / non-runner uses).
        with torch.no_grad():
            if len(self._collect_next_values) == T:
                soft_v = torch.stack(self._collect_next_values)  # [T, N, 1]
                ent_bonus = torch.stack(self._collect_ent_bonus)  # [T, N, 1]
                if self.aux_loss_mult > 0.0 and len(self._collect_aux_targets) == T:
                    self._aux_targets = torch.stack(self._collect_aux_targets).flatten(0, 1).detach()
                else:
                    self._aux_targets = None
            else:
                next_priv = self.storage.next_privileged_observations  # [T, N, *]
                B = next_priv.shape[1] * T
                flat_next = next_priv.reshape(B, -1)
                flat_next_actor = self.storage.next_observations.reshape(B, -1)
                target_a, target_logp, _, _ = self.policy.sample_with_log_prob(
                    flat_next_actor, normalized=True
                )
                next_q = self.policy.evaluate_q(flat_next, target_a, normalized=True)
                # Reference-exact soft-return decomposition (torchrl collect_fn):
                # the entropy bonus enters the REWARD at full weight,
                #   r'_t = r_t - gamma * alpha * logpi(a'_t | s'_t),
                # and the lambda-blend uses the PLAIN Q'.
                q_next = next_q.squeeze(-1)  # plain Q'  [B]
                soft_v = q_next.view(T, -1, 1)  # [T, N, 1]
                ent_bonus = (-self.gamma * alpha * target_logp).view(T, -1, 1)  # [T, N, 1]
                if self.aux_loss_mult > 0.0:
                    self._aux_targets = self.policy.evaluate_q_features(
                        flat_next, target_a, normalized=True
                    ).detach()
                else:
                    self._aux_targets = None

            # mask: 1 if we should bootstrap from next state, 0 if pure termination.
            done = self.storage.dones.float()
            trunc = self.storage.truncated

            # Per-sample targets for the reference reward-prediction aux term. Flattened
            # the same [T*N] way as `_aux_targets`, so the minibatch `idx` indexes all
            # three consistently. `storage.rewards` is already reward_scale-multiplied
            # (line 405), matching the reference, which scales in the env wrapper.
            if self.aux_reward_pred and self.aux_loss_mult > 0.0:
                self._aux_rewards = self.storage.rewards.flatten(0, 1).detach()
                self._aux_dones = done.flatten(0, 1).detach()
            else:
                self._aux_rewards = self._aux_dones = None
            not_terminal = (1.0 - done).clamp_min(0.0)
            bootstrap_mask = torch.maximum(not_terminal, trunc)  # truncated → bootstrap

            recurr = soft_v[-1]  # init from last step's bootstrap target
            for step in reversed(range(T)):
                # Entropy folded into the reward at full weight (reference form).
                soft_r = self.storage.rewards[step] + ent_bonus[step]
                next_v = soft_v[step]
                m = bootstrap_mask[step]
                # Reference compute_gve cuts the λ-trace on truncation: a timeout
                # ends the episode, so the buffered step t+1 is a NEW episode and
                # blending λ·recurr[t+1] would leak its return backwards. On
                # truncation use the pure next-state bootstrap (next_v); otherwise
                # the standard (1−λ)·next_v + λ·recurr blend.
                trunc_step = trunc[step].bool()
                blend = torch.where(
                    trunc_step, next_v, (1.0 - self.lam) * next_v + self.lam * recurr
                )
                recurr = soft_r + self.gamma * m * blend
                self.storage.returns[step] = recurr

            # No advantages — pathwise actor doesn't need them.
            self.storage.advantages = self.storage.returns - self.storage.values

            # Diagnostics (surface in runner logs + wandb via update()'s dict):
            #  q_bias  — E[Q(s,a) − λ-return target] on the same rollout pairs;
            #            persistently positive = critic overestimation spiral.
            #  frac_targets_clipped — targets outside the HL-Gauss support get
            #            clamped to the edge; high values self-confirm optimism.
            ret = self.storage.returns
            self._diagnostics = {
                "returns_mean": ret.mean().item(),
                "returns_max": ret.max().item(),
                "q_bias": (self.storage.values - ret).mean().item(),
            }
            if self.policy.is_distributional_critic:
                c = self.policy.critic
                self._diagnostics["frac_targets_clipped"] = (
                    ((ret > c.v_max) | (ret < c.v_min)).float().mean().item()
                )

        # last_critic_obs is unused (we already have stored next_obs for every step,
        # including the last). Keep the runner signature compatible.
        _ = last_critic_obs

    # ------------------------------------------------------------------
    # Update — interleaved critic+actor per minibatch (matches paper).
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        self._advance_target_entropy()
        critic_loss_sum = 0.0
        actor_loss_sum = entropy_sum = kl_sum = q_value_sum = alpha_temp_loss_sum = alpha_kl_loss_sum = 0.0
        actor_gn_sum = critic_gn_sum = dep_gap_sum = 0.0
        # Last-minibatch values, for like-for-like comparison with the reference's logging
        # (which reports the final minibatch, not a mean). See metrics_out below.
        critic_loss_last = q_value_last = entropy_last = kl_last = 0.0
        critic_gn_last = actor_gn_last = 0.0
        n = 0
        for batch in self._minibatch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs_b, critic_obs_b, actions_b, returns_b, old_mu_b, old_sigma_b, truncated_b, idx_b = batch

            aux_target_b = self._aux_targets[idx_b] if self._aux_targets is not None else None
            aux_reward_b = self._aux_rewards[idx_b] if self._aux_rewards is not None else None
            aux_done_b = self._aux_dones[idx_b] if self._aux_dones is not None else None
            critic_loss = self._update_critic(
                critic_obs_b, actions_b, returns_b, truncated_b, aux_target_b, aux_reward_b, aux_done_b
            )
            critic_loss_sum += critic_loss

            metrics = self._update_actor(obs_b, critic_obs_b, old_mu_b, old_sigma_b)
            actor_loss_sum += metrics["actor_loss"]
            entropy_sum += metrics["entropy"]
            kl_sum += metrics["kl"]
            q_value_sum += metrics["q_value"]
            alpha_temp_loss_sum += metrics["alpha_temp_loss"]
            alpha_kl_loss_sum += metrics["alpha_kl_loss"]
            actor_gn_sum += metrics["actor_grad_norm"]
            critic_gn_sum += metrics["critic_grad_norm"]
            dep_gap_sum += metrics["deployment_gap"]

            critic_loss_last = critic_loss
            q_value_last = metrics["q_value"]
            entropy_last = metrics["entropy"]
            kl_last = metrics["kl"]
            critic_gn_last = metrics["critic_grad_norm"]
            actor_gn_last = metrics["actor_grad_norm"]

            n += 1

        self.storage.clear()
        self._collect_next_values.clear()
        self._collect_ent_bonus.clear()
        self._collect_aux_targets.clear()
        self._aux_rewards = self._aux_dones = None

        metrics_out = {
            "value_function": critic_loss_sum / max(n, 1),
            "surrogate": actor_loss_sum / max(n, 1),
            "entropy": entropy_sum / max(n, 1),
            "kl": kl_sum / max(n, 1),
            "q_value": q_value_sum / max(n, 1),
            "alpha_temp": self.alpha_temp.item(),
            "alpha_kl": self.alpha_kl.item(),
            "alpha_temp_loss": alpha_temp_loss_sum / max(n, 1),
            "alpha_kl_loss": alpha_kl_loss_sum / max(n, 1),
            # Pre-clip gradient norms (reference logs both). Reference Go2 finals:
            # actor 0.043, critic 0.67 — its actor clip never binds.
            "target_entropy": self.target_entropy,
            "deployment_gap": dep_gap_sum / max(n, 1),
            "actor_grad_norm": actor_gn_sum / max(n, 1),
            "critic_grad_norm": critic_gn_sum / max(n, 1),
            # LAST-minibatch values alongside the means above. The reference assigns its
            # log dict INSIDE the minibatch loop, so every per-iteration scalar it reports
            # is the last minibatch of the last epoch — not a mean. Comparing our means
            # against their last-minibatch values is not like-for-like, and it produced a
            # spurious "their critic is driven 9x harder" reading (their 7.01 vs our 0.77)
            # that was pure aggregation difference. Log both so either comparison is valid.
            "value_function_last": critic_loss_last,
            "q_value_last": q_value_last,
            "entropy_last": entropy_last,
            "kl_last": kl_last,
            "critic_grad_norm_last": critic_gn_last,
            "actor_grad_norm_last": actor_gn_last,
            **getattr(self, "_diagnostics", {}),
        }
        if getattr(self, "trudi_wandb_schema", False):
            # Mirror our metrics under the TruDi reference's wandb key names
            # (actor/kl, critic/qf_mean, ...) so our runs and the authors' runs
            # overlay on the SAME wandb charts. Additive: our own keys are kept,
            # and the runner logs any key containing "/" verbatim.
            from safe_rl.utils.trudi_wandb_schema import add_trudi_aliases

            metrics_out = add_trudi_aliases(metrics_out)
        return metrics_out

    def _minibatch_generator(self, num_mini_batches: int, num_epochs: int):
        """REPPO-local shuffled minibatch generator.

        Mirrors ``RolloutStorage.mini_batch_generator`` but additionally yields
        the per-sample ``truncated`` flag (aligned to the shuffled indices) so the
        critic loss can mask timeout steps — the shared generator does not expose
        it. Yields (obs, critic_obs, actions, returns, old_mu, old_sigma, truncated).
        """
        st = self.storage
        batch_size = st.num_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        obs = st.observations.flatten(0, 1)
        critic_obs = (
            st.privileged_observations.flatten(0, 1)
            if st.privileged_observations is not None
            else obs
        )
        actions = st.actions.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        old_mu = st.mu.flatten(0, 1)
        old_sigma = st.sigma.flatten(0, 1)
        truncated = st.truncated.flatten(0, 1)

        for _ in range(num_epochs):
            # Re-permute every epoch (reference re-draws the partition each epoch);
            # drawing once meant all epochs replayed the same minibatch grouping.
            indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    obs[idx],
                    critic_obs[idx],
                    actions[idx],
                    returns[idx],
                    old_mu[idx],
                    old_sigma[idx],
                    truncated[idx],
                    idx,
                )

    # ------------------------------------------------------------------
    # Critic update — HL-Gauss CE (distributional) or MSE (standard)
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        truncated: torch.Tensor,
        aux_target: torch.Tensor | None = None,
        aux_reward: torch.Tensor | None = None,
        aux_done: torch.Tensor | None = None,
    ) -> float:
        # Mask timeout (truncated) steps out of the critic loss: their bootstrap
        # target is only valid when the env surfaced final_observation, and the
        # references (torchrl truncation_mask, rsl_rl (1 - truncations)) drop them
        # unconditionally. Truncations are rare, so the lost signal is negligible.
        mask = (1.0 - truncated.view(-1)).clamp_(0.0, 1.0)
        # "mask": mean over the surviving samples. "batch": reference convention —
        # masked samples still count in the denominator, so dropping a step also
        # shrinks the gradient rather than just removing the sample.
        denom = mask.sum().clamp_min(1.0) if self.critic_loss_denominator == "mask" else float(mask.numel())

        if self.policy.is_distributional_critic:
            logits = self.policy.evaluate_q_dist(critic_obs, actions, normalized=True)
            c = self.policy.critic
            soft_targets = self._hlgauss_embed(
                returns.view(-1), c.v_min, c.v_max, c.num_atoms
            ).detach()
            ce = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(-1)
            critic_loss = (mask * ce).sum() / denom
        else:
            q = self.policy.evaluate_q(critic_obs, actions, normalized=True)
            se = (returns - q).pow(2).view(-1)
            critic_loss = (mask * se).sum() / denom

        # Self-predictive aux loss (reference): the prediction head applied to critic
        # features of (s,a) regresses to sg[features of (s',a')], truncation-masked like
        # the value loss. Without the head this degenerates into pulling the critic's own
        # representation toward its next-state features, which smooths dQ/da.
        if self.aux_loss_mult > 0.0 and aux_target is not None:
            if self.aux_reward_pred and aux_reward is not None:
                # Reference form (jaxrl/reppo.py:493-501): the reward error is
                # CONCATENATED onto the D per-feature errors and the mean is taken over
                # all D+1 slots — so the reward term carries weight 1/(D+1), not 1/2 —
                # and the whole thing is masked by (1 - done) before the outer
                # (1 - truncated) mask below.
                pred, pred_rew = self.policy.evaluate_q_features_reward(
                    critic_obs, actions, normalized=True
                )
                se = torch.cat(
                    [(pred - aux_target).pow(2), (pred_rew - aux_reward.view(-1, 1)).pow(2)], dim=-1
                )
                not_done = (1.0 - aux_done.view(-1, 1)).clamp_(0.0, 1.0) if aux_done is not None else 1.0
                aux_per_sample = (not_done * se).mean(dim=-1)
            else:
                pred = self.policy.evaluate_q_features(
                    critic_obs, actions, normalized=True, predict=True
                )
                aux_per_sample = (pred - aux_target).pow(2).mean(dim=-1)
            critic_loss = critic_loss + self.aux_loss_mult * (mask * aux_per_sample).sum() / denom

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_ returns the PRE-clip total norm — log it: the reference
        # reports critic_grad_norm ~0.67 and actor_grad_norm ~0.043 on Go2, i.e. its
        # actor clip never binds. If ours sits above max_grad_norm the two runs are
        # not in the same optimization regime, whatever the losses look like.
        self._last_critic_grad_norm = float(
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        )
        self.critic_optimizer.step()
        return critic_loss.item()

    @staticmethod
    def _hlgauss_embed(
        targets: torch.Tensor, v_min: float, v_max: float, num_atoms: int
    ) -> torch.Tensor:
        """Embed scalar targets as soft categorical distributions (HL-Gauss).

        Bin centers at linspace(v_min, v_max, num_atoms); edges extend half a
        bin-width beyond endpoints so boundary targets are smoothed
        symmetrically. σ = 0.75 × bin_width (reference default).
        """
        delta_z = (v_max - v_min) / (num_atoms - 1)
        sigma_sqrt2 = 0.75 * delta_z * math.sqrt(2.0)
        edges = torch.linspace(
            v_min - delta_z / 2.0,
            v_max + delta_z / 2.0,
            num_atoms + 1,
            device=targets.device,
        )
        targets = targets.clamp(v_min, v_max)
        cdf = torch.erf((edges - targets.unsqueeze(-1)) / sigma_sqrt2)
        probs = cdf[..., 1:] - cdf[..., :-1]
        z = (cdf[..., -1:] - cdf[..., :1]).clamp_min(1e-8)
        return probs / z

    # ------------------------------------------------------------------
    # Actor update — pathwise Q gradient + closed-form Gaussian KL
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
    ) -> dict[str, float]:
        action_pi, log_prob_pi, mu_new, sigma_new = self.policy.sample_with_log_prob(
            obs, normalized=True
        )

        # Pathwise Q: gradient flows through ∂Q/∂a · ∂a/∂θ. Keep critic params'
        # requires_grad off through the backward so the actor pass does not
        # populate critic .grad buffers (re-enabled after optimizer.step()).
        self._set_critic_grad(requires_grad=False)
        q_pi = self.policy.evaluate_q(critic_obs, action_pi, normalized=True).squeeze(-1)

        primary = (self.alpha_temp.detach() * log_prob_pi - q_pi)

        if getattr(self.policy, "squash", "none") == "tanh":
            # Tanh-squashed policy: no closed-form KL — use the reference's MC
            # estimator (16 samples from the OLD squashed policy):
            #   KL(old‖new) ≈ E_{a~old}[log old(a) − log new(a)]
            # (μ, σ) stored per step are the BASE Normal params.
            new_td = self.policy.squashed(Normal(mu_new, sigma_new.clamp_min(1e-8)))
            with torch.no_grad():
                old_td_ng = self.policy.squashed(Normal(old_mu, old_sigma.clamp_min(1e-8)))
                old_a = old_td_ng.sample((16,))
                old_a = self.policy._clamp_squashed(old_a)
                old_lp = old_td_ng.log_prob(old_a).sum(dim=-1)  # [16, B]
            new_lp = new_td.log_prob(old_a).sum(dim=-1)  # [16, B], grads -> new params
            kl = (old_lp - new_lp).mean(dim=0)  # [B]
        else:
            # Closed-form Gaussian KL on raw (μ, σ): reference π_old ‖ π_new.
            # KL(N(μ_o, σ_o) ‖ N(μ_n, σ_n)) per dim:
            #   log(σ_n/σ_o) + (σ_o^2 + (μ_o − μ_n)^2)/(2σ_n^2) − 1/2
            var_old = old_sigma.pow(2)
            var_new = sigma_new.pow(2).clamp_min(1e-8)
            kl_per_dim = (
                torch.log(sigma_new.clamp_min(1e-8) / old_sigma.clamp_min(1e-8))
                + (var_old + (old_mu - mu_new).pow(2)) / (2.0 * var_new)
                - 0.5
            )
            kl = kl_per_dim.sum(dim=-1)

        # REPPO actor loss — two reference-supported modes:
        #   "clipped" (author default): per-sample hard gate — when KL exceeds
        #     the bound, the reward loss is REPLACED by α_kl·KL for that sample
        #     (torch.where in reference make_actor_update_fn). Structural brake
        #     on oversized steps; the primary loss resumes once KL re-enters.
        #   "full": smooth Lagrangian penalty L = primary + α_kl·KL — the
        #     reward term stays active and α_kl (dual below) regulates step size.
        if self.kl_clip_mode == "clipped":
            actor_loss = torch.where(
                kl < self.desired_kl, primary, self.alpha_kl.detach() * kl
            ).mean()
        else:
            actor_loss = (primary + self.alpha_kl.detach() * kl).mean()

        # Entropy for logging + dual-temperature target. Squashed: MC estimate
        # −log π(a_π) of the current sample (reference: entropy = -log_probs);
        # raw Gaussian: analytic.
        if getattr(self.policy, "squash", "none") == "tanh":
            entropy = -log_prob_pi
        else:
            entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma_new.clamp_min(1e-8))).sum(-1)

        # Critic-measured deployment gap: how much value the critic thinks is lost
        # by deploying the mode instead of sampling,
        #     Delta_dep = E_s[ E_a~pi[Q(s,a)] - Q(s, tanh(mu(s))) ].
        # Positive => the critic rates typical samples ABOVE the deterministic
        # action, i.e. the max-ent policy's mode is not what the critic was
        # trained to value. This is the quantity the entropy experiments move.
        with torch.no_grad():
            if getattr(self.policy, "squash", "none") == "tanh":
                mode_action = getattr(self.policy, "action_scale", 1.0) * torch.tanh(mu_new)
            else:
                mode_action = mu_new
            q_mode = self.policy.evaluate_q(critic_obs, mode_action, normalized=True).squeeze(-1)
            deployment_gap = (q_pi.detach() - q_mode).mean()

        # Dual updates — gradients only flow into log_alpha_*; entropy/kl detached.
        # α_temp: push policy entropy toward target_entropy (paper convention).
        alpha_temp_loss = self.alpha_temp * (entropy.mean().detach() - self.target_entropy)
        # α_kl: increase α_kl when KL exceeds the bound.
        alpha_kl_loss = self.alpha_kl * (self.desired_kl - kl.mean().detach())

        self.optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        (actor_loss + alpha_temp_loss + alpha_kl_loss).backward()
        clip_params: list[nn.Parameter] = list(self.policy.actor.parameters())
        if self.dual_optim_mode == "actor":
            # Reference: the duals are actor parameters, so they are inside the norm.
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
    # Helpers
    # ------------------------------------------------------------------

    def _set_critic_grad(self, requires_grad: bool) -> None:
        for param in self.policy.critic.parameters():
            param.requires_grad = requires_grad
