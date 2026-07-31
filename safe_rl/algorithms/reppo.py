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
    pathwise (reparameterized) gradients through a twin Q(s,a) critic, so
    ∂Q/∂a · ∂a/∂θ carries the reward signal directly. The critic is trained
    on a soft-Q λ-target computed once per update from a frozen target actor:

        a'_t ~ π_target(s'_t)
        soft_V(s'_t) = min(Q1, Q2)_target(s'_t, a'_t) − α · log_prob_target(a'_t)
        target_q[t]  = λ-blended return on (r_t − α · log_prob_t) bootstrapped
                       on soft_V(s'_t), masking γ-bootstrap by (1 − done | truncated)

    Actor loss (pathwise):
        a_π = π(s).rsample();  q_π = min_i Q_i(s, a_π)
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
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        tau: float = 0.005,
        use_target_networks: bool = True,
        actor_q_reduction: str = "min",
        kl_clip_mode: str = "full",
        alpha_kl_min: float = 0.0,
        aux_loss_mult: float = 0.0,
        reward_scale: float = 1.0,
        reward_normalization: bool = False,
        reward_norm_g_max: float = 10.0,
        normalize_advantage_per_mini_batch: bool = False,
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
        self.optimizer = opt_cls(policy.actor.parameters(), lr=learning_rate, **opt_kwargs)
        critic_lr = critic_learning_rate if critic_learning_rate is not None else learning_rate
        self.critic_optimizer = opt_cls(policy.critics.parameters(), lr=critic_lr, **opt_kwargs)
        self.alpha_optimizer = opt_cls(
            [self.log_alpha_temp, self.log_alpha_kl], lr=alpha_lr, **opt_kwargs
        )

        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        # Reference REPPO has NO target networks (online bootstrap) and a single
        # critic (no twin-min pessimism); both retained here as options.
        self.use_target_networks = use_target_networks
        if actor_q_reduction not in ("min", "mean", "q1"):
            raise ValueError(f"actor_q_reduction must be 'min', 'mean' or 'q1'; got {actor_q_reduction!r}")
        self.actor_q_reduction = actor_q_reduction
        # Reference actor_kl_clip_mode: "clipped" (author default) hard-gates each
        # sample — when its KL exceeds the bound the reward loss is REPLACED by
        # alpha_kl*KL, structurally stopping oversized steps. "full" is the soft
        # penalty variant (also a supported reference mode).
        if kl_clip_mode not in ("full", "clipped"):
            raise ValueError(f"kl_clip_mode must be 'full' or 'clipped'; got {kl_clip_mode!r}")
        self.kl_clip_mode = kl_clip_mode
        # Floor on the KL dual. Without it the exponential parameterization decays
        # alpha_kl to ~0 during any stretch where KL < bound (loss alpha*(bound-kl)
        # with positive slack), leaving the clipped gate with NO restoring force by
        # the time the policy reaches the bound — observed: init 0.5 -> 0.002
        # within minutes, KL then pinned at the bound for the whole run (v10/v11).
        self.alpha_kl_min = float(alpha_kl_min)
        # Self-predictive aux loss weight (reference: aux_loss_mult * MSE between
        # critic features of (s,a) and sg[features of (s',a')]).
        self.aux_loss_mult = float(aux_loss_mult)
        self._aux_targets: torch.Tensor | None = None
        # Reward scaling (reference REPPO env.reward_scaling): dt-scaled sim
        # rewards make per-step reward ~0.01, so Q-differences are tiny next to
        # the fixed-scale KL trust-region cost and the actor barely moves.
        # Scaling inside the algorithm keeps runner/wandb reward logging raw.
        self.reward_scale = float(reward_scale)
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
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

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

        self._final_obs_warned = False

    @property
    def alpha_temp(self) -> torch.Tensor:
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        return self.log_alpha_kl.exp()

    def _q_reduce(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Combine twin Q estimates per `actor_q_reduction` (reference: single Q)."""
        if self.actor_q_reduction == "min":
            return torch.minimum(q1, q2)
        if self.actor_q_reduction == "mean":
            return 0.5 * (q1 + q2)
        return q1

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
        q1, q2 = self.policy.evaluate_q(norm_critic_obs, action, normalized=True)
        value = self._q_reduce(q1, q2).detach()
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
            # Normalize with the CURRENT statistics but do not update them: this same
            # observation arrives again as `obs` on the next act() call, which is where it
            # is counted. Stored normalized (see act()) so the bootstrap in
            # compute_returns runs on collection-time inputs.
            self.transition.next_observations = self._normalize(
                self.policy.actor_obs_normalizer, next_actor_obs, update_stats=False
            )
            self.transition.next_privileged_observations = self._normalize(
                self.policy.critic_obs_normalizer, next_priv_obs, update_stats=False
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

        # Observations in storage are already normalized (see act()), so nothing on this
        # path touches the normalizers — no freeze/thaw dance is needed and the KL is
        # measured on exactly the inputs the collecting policy saw.

        # Bootstrap value for the last step uses the runner-supplied last_critic_obs.
        with torch.no_grad():
            next_priv = self.storage.next_privileged_observations  # [T, N, *]
            B = next_priv.shape[1] * T
            flat_next = next_priv.reshape(B, -1)
            flat_next_actor = self.storage.next_observations.reshape(B, -1)
            if self.use_target_networks:
                target_a, target_logp = self.policy.target_sample_with_log_prob(
                    flat_next_actor, normalized=True
                )
                q1, q2 = self.policy.evaluate_q_target(flat_next, target_a, normalized=True)
            else:
                # Reference REPPO: bootstrap from the ONLINE actor and critic
                # (no target networks; value estimates track the newest params).
                target_a, target_logp, _, _ = self.policy.sample_with_log_prob(
                    flat_next_actor, normalized=True
                )
                q1, q2 = self.policy.evaluate_q(flat_next, target_a, normalized=True)
            # Reference-exact soft-return decomposition (torchrl collect_fn):
            # the entropy bonus enters the REWARD at full weight,
            #   r'_t = r_t - gamma * alpha * logpi(a'_t | s'_t),
            # and the lambda-blend uses the PLAIN Q'. Putting -alpha*logp' inside
            # soft_v under the (1-lambda) blend instead undercounts the future
            # entropy chain by a factor (1-lambda) (~5% weight at lambda=0.95).
            q_next = self._q_reduce(q1, q2).squeeze(-1)  # plain Q'  [B]
            soft_v = q_next.view(T, -1, 1)  # [T, N, 1]
            ent_bonus = (-self.gamma * alpha * target_logp).view(T, -1, 1)  # [T, N, 1]

            # Self-predictive aux targets: critic features of (s', a'), fixed for
            # the whole update (reference computes them at collection time).
            if self.aux_loss_mult > 0.0:
                # Targets are RAW trunk features of (s', a') — the prediction head is
                # applied only on the online side (reference: pred_module(f(s,a)) -> sg[f(s',a')]).
                self._aux_targets = self.policy.evaluate_q_features(
                    flat_next, target_a, normalized=True
                ).detach()
            else:
                self._aux_targets = None

            # mask: 1 if we should bootstrap from next state, 0 if pure termination.
            done = self.storage.dones.float()
            trunc = self.storage.truncated
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
                c = self.policy.critic_1
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
        critic_loss_sum = 0.0
        actor_loss_sum = entropy_sum = kl_sum = q_value_sum = alpha_temp_loss_sum = alpha_kl_loss_sum = 0.0
        n = 0
        for batch in self._minibatch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs_b, critic_obs_b, actions_b, returns_b, old_mu_b, old_sigma_b, truncated_b, idx_b = batch

            aux_target_b = self._aux_targets[idx_b] if self._aux_targets is not None else None
            critic_loss = self._update_critic(critic_obs_b, actions_b, returns_b, truncated_b, aux_target_b)
            critic_loss_sum += critic_loss

            metrics = self._update_actor(obs_b, critic_obs_b, old_mu_b, old_sigma_b)
            actor_loss_sum += metrics["actor_loss"]
            entropy_sum += metrics["entropy"]
            kl_sum += metrics["kl"]
            q_value_sum += metrics["q_value"]
            alpha_temp_loss_sum += metrics["alpha_temp_loss"]
            alpha_kl_loss_sum += metrics["alpha_kl_loss"]

            if self.use_target_networks:
                self.policy.soft_update_targets(self.tau)
                self.policy.soft_update_actor_target(self.tau)
            n += 1

        self.storage.clear()

        return {
            "value_function": critic_loss_sum / max(n, 1),
            "surrogate": actor_loss_sum / max(n, 1),
            "entropy": entropy_sum / max(n, 1),
            "kl": kl_sum / max(n, 1),
            "q_value": q_value_sum / max(n, 1),
            "alpha_temp": self.alpha_temp.item(),
            "alpha_kl": self.alpha_kl.item(),
            "alpha_temp_loss": alpha_temp_loss_sum / max(n, 1),
            "alpha_kl_loss": alpha_kl_loss_sum / max(n, 1),
            **getattr(self, "_diagnostics", {}),
        }

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
    ) -> float:
        # Mask timeout (truncated) steps out of the critic loss: their bootstrap
        # target is only valid when the env surfaced final_observation, and the
        # references (torchrl truncation_mask, rsl_rl (1 - truncations)) drop them
        # unconditionally. Truncations are rare, so the lost signal is negligible.
        mask = (1.0 - truncated.view(-1)).clamp_(0.0, 1.0)
        denom = mask.sum().clamp_min(1.0)

        # With num_critics=1, critic_2 aliases critic_1 — summing both terms
        # would silently double the gradient (effective 2x critic lr).
        twin = self.policy.critic_2 is not self.policy.critic_1

        if self.policy.is_distributional_critic:
            logits_1, logits_2 = self.policy.evaluate_q_dist(critic_obs, actions, normalized=True)
            c = self.policy.critic_1
            soft_targets = self._hlgauss_embed(
                returns.view(-1), c.v_min, c.v_max, c.num_atoms
            ).detach()
            ce_1 = -(soft_targets * F.log_softmax(logits_1, dim=-1)).sum(-1)
            critic_loss = (mask * ce_1).sum() / denom
            if twin:
                ce_2 = -(soft_targets * F.log_softmax(logits_2, dim=-1)).sum(-1)
                critic_loss = critic_loss + (mask * ce_2).sum() / denom
        else:
            q1, q2 = self.policy.evaluate_q(critic_obs, actions, normalized=True)
            se_1 = (returns - q1).pow(2).view(-1)
            critic_loss = (mask * se_1).sum() / denom
            if twin:
                se_2 = (returns - q2).pow(2).view(-1)
                critic_loss = critic_loss + (mask * se_2).sum() / denom

        # Self-predictive aux loss (reference): the prediction head applied to critic
        # features of (s,a) regresses to sg[features of (s',a')], truncation-masked like
        # the value loss. Without the head this degenerates into pulling the critic's own
        # representation toward its next-state features, which smooths dQ/da.
        if self.aux_loss_mult > 0.0 and aux_target is not None:
            pred = self.policy.evaluate_q_features(
                critic_obs, actions, normalized=True, predict=True
            )
            aux_per_sample = (pred - aux_target).pow(2).mean(dim=-1)
            critic_loss = critic_loss + self.aux_loss_mult * (mask * aux_per_sample).sum() / denom

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.critics.parameters(), self.max_grad_norm)
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
        q1, q2 = self.policy.evaluate_q(critic_obs, action_pi, normalized=True)
        q_pi = self._q_reduce(q1, q2).squeeze(-1)

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

        # Dual updates — gradients only flow into log_alpha_*; entropy/kl detached.
        # α_temp: push policy entropy toward target_entropy (paper convention).
        alpha_temp_loss = self.alpha_temp * (entropy.mean().detach() - self.target_entropy)
        # α_kl: increase α_kl when KL exceeds the bound.
        alpha_kl_loss = self.alpha_kl * (self.desired_kl - kl.mean().detach())

        self.optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        (actor_loss + alpha_temp_loss + alpha_kl_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.alpha_optimizer.step()
        if self.alpha_kl_min > 0.0:
            with torch.no_grad():
                self.log_alpha_kl.clamp_(min=math.log(self.alpha_kl_min))
        self._set_critic_grad(requires_grad=True)

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "kl": kl.mean().item(),
            "q_value": q_pi.mean().item(),
            "alpha_temp_loss": alpha_temp_loss.item(),
            "alpha_kl_loss": alpha_kl_loss.item(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_critic_grad(self, requires_grad: bool) -> None:
        for param in self.policy.critics.parameters():
            param.requires_grad = requires_grad
