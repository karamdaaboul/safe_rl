from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from safe_rl.algorithms.ppol_pid import PPOL_PID
from safe_rl.storage.reach_rollout_storage import RolloutStorageReach


class RCPPO(PPOL_PID):
    """Reachability-Constrained PPO (RCRL-style safe RL).

    Based on "Reachability Constrained Reinforcement Learning" (Yu et al., ICML 2022,
    https://arxiv.org/abs/2205.07536) and RESPO (NeurIPS 2023). Instead of constraining
    the discounted *sum* of costs like PPOL_PID/P3O, the cost critic is retrained as a
    Hamilton-Jacobi-style reachability value function

        V_h(s) = (1 - gamma_h) * h(s) + gamma_h * max(h(s), V_h(s'))

    — the worst future constraint violation from s — whose zero-sublevel set approximates
    the largest feasible (control-invariant) set. The inherited PID Lagrangian then
    regulates the *state-wise* safety level E[V_h] toward the feasibility threshold
    ``cost_limits`` (epsilon, typically ~0.1) instead of an episodic cost budget, and the
    inherited normalized surrogate (adv_r - lambda * adv_h) / (1 + lambda) applies as-is.

    Everything else — PID machinery, cost-critic regression loop, logging interface — is
    inherited from PPOL_PID unchanged; only the targets and the constraint feedback differ.

    Optionally (``train_reach_q``, requires ``ActorCriticReachQ``) an action-conditioned
    head Q_h(s, a) is regressed on the same targets, enabling the runtime
    ``ReachabilitySafetyFilter`` to rank candidate actions at rollout/eval time.
    """

    def __init__(
        self,
        policy,
        gamma_h: float = 0.99,
        cost_margin: float = 0.0,
        signed_margin: bool = False,
        scale_cost_advantage: bool = False,
        train_reach_q: bool = True,
        reach_q_lr: Optional[float] = None,
        reach_q_loss_coef: float = 1.0,
        **kwargs: Any,
    ):
        if kwargs.pop("use_clipped_cost_loss", False):
            print("[RCPPO] use_clipped_cost_loss does not apply to max-backup targets; forcing False.")
        super().__init__(policy, use_clipped_cost_loss=False, **kwargs)

        if getattr(self.policy, "is_distributional_cost_critic", False):
            raise ValueError(
                "RCPPO supports only the MSE cost critic (cost_critic_kwargs.loss_type: mse); "
                "distributional (HL-Gauss/categorical) heads are not compatible with max-backup targets."
            )

        self.gamma_h = gamma_h
        self.cost_margin = cost_margin
        self.signed_margin = signed_margin
        self.scale_cost_advantage = scale_cost_advantage
        self.train_reach_q = train_reach_q
        self.reach_q_loss_coef = reach_q_loss_coef
        self._reach_q_loss_accum = 0.0
        self._reach_q_updates = 0
        self._last_reach_level: List[float] = [0.0] * self.num_costs

        if train_reach_q:
            if not hasattr(self.policy, "reach_critic"):
                raise ValueError(
                    "RCPPO with train_reach_q=True needs a policy with a reach_critic head; "
                    "set policy.class_name: ActorCriticReachQ in the config (or train_reach_q: false)."
                )
            # Rebuild the main optimizer without the Q_h head and give the head its own
            # optimizer, so the regression neither perturbs the KL-adaptive actor LR nor
            # is throttled by it. Its Adam state is not checkpointed (the runner saves
            # only self.optimizer), which is acceptable for a pure regression head.
            reach_params = list(self.policy.reach_critic.parameters())
            reach_param_ids = {id(p) for p in reach_params}
            main_params = [p for p in self.policy.parameters() if id(p) not in reach_param_ids]
            self.optimizer = optim.Adam(main_params, lr=self.learning_rate)
            self.reach_q_lr = reach_q_lr if reach_q_lr is not None else self.learning_rate
            self.reach_q_optimizer = optim.Adam(reach_params, lr=self.reach_q_lr)
        else:
            self.reach_q_optimizer = None

        # Cost-aware transition with the extra timeout/bootstrap fields.
        self.transition = RolloutStorageReach.Transition()

        print(f"RCPPO: gamma_h={gamma_h}, feasibility threshold epsilon={self.cost_limits}, "
              f"cost_margin={cost_margin}, signed_margin={signed_margin}, "
              f"scale_cost_advantage={scale_cost_advantage}, train_reach_q={train_reach_q}")

    def init_storage(self, training_type: str, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: Tuple, critic_obs_shape: Tuple, action_shape: Tuple) -> None:
        # Always pass an explicit cost_shape so the (num_envs, num_costs) paths are
        # exercised uniformly (PPOL_PID drops to the legacy single-cost layout otherwise).
        self.storage = RolloutStorageReach(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
            training_type=training_type, cost_shape=(self.num_costs,), device=self.device,
        )

    def process_env_step(self, rewards: torch.Tensor, costs: torch.Tensor,
                         dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """Store the step; unlike PPOL_PID, the timeout bootstrap is NOT folded into the
        cost additively (the max-backup is nonlinear) — it enters the recursion as the
        critic's V_h at the true final observation instead."""
        self.transition.rewards = rewards.clone()
        # h(s) = cost - margin. Default: clamped so h >= 0 keeps the "0 = safe"
        # semantics of nonnegative env costs. With signed_margin the env already
        # emits a signed distance margin (GeometricMarginWrapper: negative = safe
        # with a real gradient toward safety) and the clamp would erase it.
        margins = self._format_costs_tensor(costs) - self.cost_margin
        self.transition.costs = margins if self.signed_margin else margins.clamp(min=0.0)
        self.transition.dones = dones

        if "time_outs" in infos:
            time_outs = infos["time_outs"].view(-1, 1).to(self.device)
            # Reward bootstrap on truncation (same as PPO/PPOL_PID).
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.time_outs = time_outs
            if bool(time_outs.any()):
                # V_h at the true terminal observation. Note: `final_observation` is raw
                # (un-normalized) and equals the critic obs only when actor/critic obs
                # coincide — both hold for Safety-Gymnasium with empirical_normalization
                # off. Fall back to V_h(s_t) as a proxy when the env doesn't forward it.
                final_obs = infos.get("final_observation")
                if final_obs is not None:
                    bootstrap = self.policy.evaluate_cost(final_obs.to(self.device)).detach()
                else:
                    bootstrap = self.transition.cost_values
                self.transition.reach_bootstrap = bootstrap * time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_cost_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute reachability (max-backup) targets instead of GAE cost returns."""
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_reachability_returns(last_cost_values, self.gamma_h)

    def update(
        self,
        current_costs: Optional[List[torch.Tensor]] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, float]:
        # The runner feeds episodic cost *sums*; the reachability constraint lives on the
        # state-wise level E[V_h] instead, which the storage now reports as mean cost
        # return. Feed that to the inherited PID so lambda regulates E[V_h] <= epsilon.
        reach_level = self.storage.get_mean_episode_costs()
        self._last_reach_level = [float(reach_level[i].item()) for i in range(self.num_costs)]
        self._reach_q_loss_accum = 0.0
        self._reach_q_updates = 0

        # Scale-only normalization of the cost advantages (no mean-centering: the sign
        # carries the safe/unsafe semantics). Reward advantages are already normalized
        # to std 1 at rollout level, while raw A_c inherits the V_h critic-residual
        # scale (~sqrt(cost MSE), e.g. 0.08) — so lambda is only "operative" from
        # ~1/std(A_c) (~12) upward and PID gains don't transfer across tasks. Dividing
        # A_c by its std makes lambda dimensionless with operative scale ~1.
        cost_adv_std = None
        if self.scale_cost_advantage and self.storage.cost_advantages is not None:
            flat = self.storage.cost_advantages.flatten(0, 1)
            cost_adv_std = flat.std(dim=0).clamp_min(1e-8)
            self.storage.cost_advantages = self.storage.cost_advantages / cost_adv_std

        loss_dict = super().update(current_costs=list(reach_level), iteration=iteration)

        if cost_adv_std is not None:
            loss_dict["cost_adv_std"] = float(cost_adv_std.mean().item())

        loss_dict["reach_level_mean"] = sum(self._last_reach_level) / len(self._last_reach_level)
        if self.train_reach_q and self._reach_q_updates > 0:
            loss_dict["reach_q"] = self._reach_q_loss_accum / self._reach_q_updates
        return loss_dict

    def _update_policy(
        self, batch: Tuple, current_costs: Optional[List[torch.Tensor]] = None
    ) -> Tuple[float, float, float]:
        # Full PPO + Lagrangian + V_h-critic step, then the Q_h regression on the same
        # max-backup targets (evaluated at the taken actions).
        value_loss, cost_loss, surrogate_loss = super()._update_policy(batch, current_costs)

        if self.train_reach_q:
            (obs_batch, critic_obs_batch, actions_batch, _target_values_batch,
             _advantages_batch, _returns_batch, _target_cost_values_batch, _cost_advantages_batch,
             returns_cost_batch, *_rest) = batch
            q_pred = self.policy.evaluate_reach_q(critic_obs_batch, actions_batch)
            reach_q_loss = self.reach_q_loss_coef * (q_pred - returns_cost_batch).pow(2).mean()
            self.reach_q_optimizer.zero_grad()
            reach_q_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.reach_critic.parameters(), self.max_grad_norm)
            self.reach_q_optimizer.step()
            self._reach_q_loss_accum += reach_q_loss.item()
            self._reach_q_updates += 1

        return value_loss, cost_loss, surrogate_loss
