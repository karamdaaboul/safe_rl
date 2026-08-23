from __future__ import annotations

import torch

from safe_rl.storage.cost_rollout_storage import RolloutStorageCMDP


class RolloutStorageReach(RolloutStorageCMDP):
    """CMDP rollout storage whose cost targets are *reachability* values.

    Instead of GAE on discounted cost sums, ``compute_reachability_returns`` regresses
    the cost critic toward the RCRL/HJ safety value function (Yu et al., ICML 2022)

        V_h(s) = (1 - gamma_h) * h(s) + gamma_h * max(h(s), V_h(s'))

    i.e. an estimate of the *worst future* constraint violation from s. Targets are
    written into the inherited ``cost_returns`` / ``cost_advantages`` tensors so every
    downstream consumer keeps working unchanged — in particular
    ``get_mean_episode_costs()`` then reports the mean on-policy safety level E[V_h],
    which is exactly the feedback signal a state-wise Lagrangian needs.

    Timeout handling needs two extra per-step buffers: ``time_outs`` marks truncations
    (where the recursion bootstraps from ``reach_bootstrap``, the critic value at the
    true final observation) while real terminations ground the recursion at V_h = h(s).
    The bootstrap cannot be folded into the cost additively (as the sum-of-costs
    algorithms do) because the max-backup is nonlinear.
    """

    class Transition(RolloutStorageCMDP.Transition):
        def __init__(self):
            super().__init__()
            self.time_outs = None
            self.reach_bootstrap = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_outs = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.reach_bootstrap = torch.zeros(
            self.num_transitions_per_env, self.num_envs, self.num_costs, device=self.device
        )

    def add_transitions(self, transition: Transition):
        step = self.step
        super().add_transitions(transition)
        if getattr(transition, "time_outs", None) is not None:
            self.time_outs[step].copy_(transition.time_outs.view(-1, 1))
        else:
            self.time_outs[step].zero_()
        if getattr(transition, "reach_bootstrap", None) is not None:
            self.reach_bootstrap[step].copy_(transition.reach_bootstrap.view(self.num_envs, self.num_costs))
        else:
            self.reach_bootstrap[step].zero_()

    def compute_reachability_returns(self, last_cost_values: torch.Tensor, gamma_h: float) -> None:
        """Backward max-backup recursion over the rollout.

        Args:
            last_cost_values: V_h estimate at the observation after the final stored step,
                shape (num_envs, num_costs).
            gamma_h: reachability discount (close to 1; contracts the operator).
        """
        if self.training_type not in ["rl", "saferl"]:
            raise ValueError("Reachability returns can only be computed for reinforcement learning training.")

        next_target = last_cost_values.view(self.num_envs, self.num_costs)
        for step in reversed(range(self.num_transitions_per_env)):
            costs = self.costs[step]
            done = self.dones[step].float().expand(-1, self.num_costs)
            timeout = self.time_outs[step].expand(-1, self.num_costs)
            true_done = done * (1.0 - timeout)
            # Successor value: next target inside a trajectory, critic bootstrap across
            # truncations, nothing after a true terminal (there, V_h(s) = h(s)).
            v_next = (1.0 - done) * next_target + timeout * self.reach_bootstrap[step]
            has_next = ((1.0 - done) + timeout).clamp(max=1.0)
            backup = (1.0 - gamma_h) * costs + gamma_h * torch.max(costs, v_next)
            target = has_next * backup + true_done * costs
            self.cost_returns[step] = target
            next_target = target

        # The sign of A_h = V_h_target - V_h carries the safe/unsafe direction used by the
        # Lagrangian surrogate; never mean-center (cf. P3O's normalize_cost_advantage=False).
        self.cost_advantages = self.cost_returns - self.cost_values
