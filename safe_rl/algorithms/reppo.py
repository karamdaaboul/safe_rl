from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.modules import REPPOActorCritic
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
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.1,
        target_entropy: float = -1.0,
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        tau: float = 0.005,
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

        # Old-policy snapshot for closed-form Gaussian KL on raw (μ, σ).
        self.old_policy = copy.deepcopy(policy)
        self.old_policy.to(self.device)
        self.old_policy.eval()

        # Algorithm-level learnable scalars.
        self.log_alpha_temp = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha_temp), device=self.device))
        )
        self.log_alpha_kl = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha_kl), device=self.device))
        )

        # Optimizers — actor on `self.optimizer` so the runner's checkpoint code finds it.
        self.optimizer = optim.AdamW(
            policy.actor.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.95)
        )
        critic_lr = critic_learning_rate if critic_learning_rate is not None else learning_rate
        self.critic_optimizer = optim.AdamW(
            policy.critics.parameters(), lr=critic_lr, weight_decay=1e-3, betas=(0.9, 0.95)
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha_temp, self.log_alpha_kl], lr=alpha_lr
        )

        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        num_actions = getattr(getattr(policy, "actor", None), "num_actions", 1)
        self.target_entropy = target_entropy * num_actions

        self._final_obs_warned = False

    @property
    def alpha_temp(self) -> torch.Tensor:
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        return self.log_alpha_kl.exp()

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

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        action = self.policy.act(obs).detach()
        # Stored value is min(Q1, Q2)(s, a) — for logging only; not used in updates.
        q1, q2 = self.policy.evaluate_q(critic_obs, action)
        value = torch.minimum(q1, q2).detach()
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        self.transition.actions = action
        self.transition.values = value
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(action).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach().expand_as(action)
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return action

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_obs: torch.Tensor | None = None,
        next_critic_obs: torch.Tensor | None = None,
    ) -> None:
        self.transition.rewards = rewards.clone()
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
            self.transition.next_observations = next_actor_obs
            self.transition.next_privileged_observations = next_priv_obs

        self.transition.truncated = time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    # ------------------------------------------------------------------
    # Soft-Q λ targets — computed once per update from frozen target actor
    # ------------------------------------------------------------------

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Soft-Q λ-target.

        Per stored step t:
          a'_t ~ π_target(s'_t)
          soft_V(s'_t) = min(Q1, Q2)_target(s'_t, a'_t) − α · log_prob_target(a'_t)
          r'_t = r_t − α · log_prob_t
          target_q[t] = r'_t + γ · m_t · ((1−λ) · soft_V(s'_t) + λ · target_q[t+1])
        where m_t = (1 − dones_t) | truncated_t (truncations keep bootstrap).
        """
        alpha = self.alpha_temp.detach().item()
        T = self.storage.num_transitions_per_env

        # Bootstrap value for the last step uses the runner-supplied last_critic_obs.
        with torch.no_grad():
            next_priv = self.storage.next_privileged_observations  # [T, N, *]
            B = next_priv.shape[1] * T
            flat_next = next_priv.reshape(B, -1)
            flat_next_actor = self.storage.next_observations.reshape(B, -1)
            target_a, target_logp = self.policy.target_sample_with_log_prob(flat_next_actor)
            q1, q2 = self.policy.evaluate_q_target(flat_next, target_a)
            soft_v_flat = torch.minimum(q1, q2).squeeze(-1) - alpha * target_logp  # [B]
            soft_v = soft_v_flat.view(T, -1, 1)  # [T, N, 1]

            # mask: 1 if we should bootstrap from next state, 0 if pure termination.
            done = self.storage.dones.float()
            trunc = self.storage.truncated
            not_terminal = (1.0 - done).clamp_min(0.0)
            bootstrap_mask = torch.maximum(not_terminal, trunc)  # truncated → bootstrap

            recurr = soft_v[-1]  # init from last step's bootstrap target
            for step in reversed(range(T)):
                soft_r = self.storage.rewards[step] - alpha * self.storage.actions_log_prob[step]
                next_v = soft_v[step]
                m = bootstrap_mask[step]
                recurr = soft_r + self.gamma * m * (
                    (1.0 - self.lam) * next_v + self.lam * recurr
                )
                self.storage.returns[step] = recurr

            # No advantages — pathwise actor doesn't need them.
            self.storage.advantages = self.storage.returns - self.storage.values

        # last_critic_obs is unused (we already have stored next_obs for every step,
        # including the last). Keep the runner signature compatible.
        _ = last_critic_obs

    # ------------------------------------------------------------------
    # Update — interleaved critic+actor per minibatch (matches paper).
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        with torch.no_grad():
            self.old_policy.load_state_dict(self.policy.state_dict())

        gen_fn = self.storage.mini_batch_generator

        critic_loss_sum = 0.0
        actor_loss_sum = entropy_sum = kl_sum = q_value_sum = alpha_temp_loss_sum = alpha_kl_loss_sum = 0.0
        n = 0
        for batch in gen_fn(self.num_mini_batches, self.num_learning_epochs):
            obs_b, critic_obs_b, actions_b, _, _, returns_b, _, old_mu_b, old_sigma_b, _, _, _ = batch

            critic_loss = self._update_critic(critic_obs_b, actions_b, returns_b)
            critic_loss_sum += critic_loss

            metrics = self._update_actor(obs_b, critic_obs_b, old_mu_b, old_sigma_b)
            actor_loss_sum += metrics["actor_loss"]
            entropy_sum += metrics["entropy"]
            kl_sum += metrics["kl"]
            q_value_sum += metrics["q_value"]
            alpha_temp_loss_sum += metrics["alpha_temp_loss"]
            alpha_kl_loss_sum += metrics["alpha_kl_loss"]

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
        }

    # ------------------------------------------------------------------
    # Critic update — HL-Gauss CE (distributional) or MSE (standard)
    # ------------------------------------------------------------------

    def _update_critic(
        self, critic_obs: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> float:
        if self.policy.is_distributional_critic:
            logits_1, logits_2 = self.policy.evaluate_q_dist(critic_obs, actions)
            c = self.policy.critic_1
            soft_targets = self._hlgauss_embed(
                returns.view(-1), c.v_min, c.v_max, c.num_atoms
            ).detach()
            loss_1 = -(soft_targets * F.log_softmax(logits_1, dim=-1)).sum(-1).mean()
            loss_2 = -(soft_targets * F.log_softmax(logits_2, dim=-1)).sum(-1).mean()
            critic_loss = loss_1 + loss_2
        else:
            q1, q2 = self.policy.evaluate_q(critic_obs, actions)
            critic_loss = (returns - q1).pow(2).mean() + (returns - q2).pow(2).mean()

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
        action_pi, log_prob_pi, mu_new, sigma_new = self.policy.sample_with_log_prob(obs)

        # Pathwise Q: gradient flows through ∂Q/∂a · ∂a/∂θ.
        self._set_critic_grad(requires_grad=False)
        q1, q2 = self.policy.evaluate_q(critic_obs, action_pi)
        q_pi = torch.minimum(q1, q2).squeeze(-1)
        self._set_critic_grad(requires_grad=True)

        primary = (self.alpha_temp.detach() * log_prob_pi - q_pi)

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

        # REPPO gate: primary loss inside the KL ball, KL penalty outside.
        actor_loss = torch.where(
            (kl < self.desired_kl).detach(),
            primary,
            self.alpha_kl.detach() * kl,
        ).mean()

        # Gaussian entropy (analytic) for logging + dual-temperature target.
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
