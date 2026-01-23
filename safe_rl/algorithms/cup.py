from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional

from safe_rl.modules import ActorCriticCost
from safe_rl.storage import RolloutStorageCMDP


class CUP:
    """
    Constrained Update Projection (CUP) for Safe Reinforcement Learning
    
    Based on the paper: "Constrained Update Projection Approach to Safe Policy Optimization"
    https://arxiv.org/pdf/2209.07089.pdf
    """
    policy: ActorCriticCost

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # CUP specific parameters
        cost_limits: Optional[List[float]] = None,
        cost_loss_coef=1.0,
        use_clipped_cost_loss=True,
        # CUP Lagrange multiplier parameters
        nu_lr=0.01,                    # Learning rate for Lagrange multiplier
        nu_max=100.0,                  # Maximum value for nu
        delta=0.01,                    # KL divergence threshold for early stopping
        c_gamma=0.99,                  # Discount factor for cost GAE
        c_gae_lam=0.95,               # GAE lambda for costs
        # Backward compatibility
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Handle multiple costs
        if cost_limits is not None:
            self.cost_limits = cost_limits
            self.num_costs = len(cost_limits)
        else:
            self.cost_limits = [0.]  # Default single constraint
            self.num_costs = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Validate and fix cost critic output dimensions
        self._validate_and_fix_cost_critic()

        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorageCMDP.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # CUP parameters
        self.cost_loss_coef = cost_loss_coef
        self.use_clipped_cost_loss = use_clipped_cost_loss

        # Lagrange multiplier parameters
        self.nu_lr = nu_lr
        self.nu_max = nu_max
        self.delta = delta # KL threshold
        self.c_gamma = c_gamma
        self.c_gae_lam = c_gae_lam

        # Initialize Lagrange multipliers for each cost constraint
        self.nu = [1.0] * self.num_costs

        # Cost history tracking
        self.cost_history = [[] for _ in range(self.num_costs)]

        # RND compatibility (CUP doesn't use RND, but runner checks for it)
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None

        # Multi-GPU support (disabled by default)
        self.is_multi_gpu = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0

        # Symmetry augmentation (disabled by default)
        self.symmetry = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # Cost shape should be (num_costs,) for multiple cost constraints
        cost_shape = (self.num_costs,) if hasattr(self, 'num_costs') and self.num_costs > 1 else None
        self.storage = RolloutStorageCMDP(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, 
            training_type=training_type, cost_shape=cost_shape, device=self.device
        )

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions
    
    def _adjust_learning_rate(self, kl_mean):
        """Adjust learning rate based on KL divergence"""
        if kl_mean > self.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2 and kl_mean > 0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def _compute_kl_divergence(self, old_mu, old_sigma, new_mu, new_sigma):
        """Compute KL divergence between two Gaussian policies"""
        kl = torch.sum(
            torch.log(new_sigma / old_sigma + 1.0e-5)
            + (torch.square(old_sigma) + torch.square(old_mu - new_mu))
            / (2.0 * torch.square(new_sigma))
            - 0.5,
            axis=-1,
        )
        return kl.mean()
    
    def _ppo_update(self):  # ppo update function
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        # Note: RolloutStorageCMDP yields 15 elements including cost data
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            target_cost_values_batch,  # Not used in Phase 1
            cost_advantages_batch,      # Not used in Phase 1
            returns_cost_batch,         # Not used in Phase 1
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # NOTE: Storage is NOT cleared here - it's needed for Phase 2 constraint projection
        # Storage will be cleared at the end of update() after both phases complete

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
    
    def _update_lagrange_multipliers(self, current_costs):
        """
        Update Lagrange multipliers based on constraint violation.

        Paper reference (Eq. 18, Section 4.2):
            ν_{k+1} = {ν_k + η * (Ĵ^C - b)}+

        where:
            - ν_k is the current Lagrange multiplier
            - η is the learning rate (nu_lr)
            - Ĵ^C is the estimated cost (current_costs)
            - b is the cost limit (cost_limits)
            - {x}+ = max(0, x) is the projection to non-negative
        """
        for cost_idx in range(self.num_costs):
            if current_costs is not None:
                # Compute constraint violation: (current_cost - cost_limit)
                cost_violation = current_costs[cost_idx] - self.cost_limits[cost_idx]

                # Update multiplier with gradient ascent and clamp to [0, nu_max]
                self.nu[cost_idx] = max(0.0, min(
                    self.nu_max,
                    self.nu[cost_idx] + self.nu_lr * cost_violation
                ))

            # Track cost history for logging
            self.cost_history[cost_idx].append(
                current_costs[cost_idx] if current_costs is not None else 0.0
            )

    def _constraint_projection(self):
        """
        Phase 2: Constraint projection using Lagrange multipliers.

        This method projects the policy back towards the constraint-satisfying region
        by minimizing the KL divergence from Phase 1 policy plus a weighted cost penalty.

        Paper reference (Algorithm 1, Projection step):
            θ_{k+1} = argmin { KL(π_{θ_{k+1/2}}, π_θ) + ν * [(1-γλ)/(1-γ)] * (π_θ/π_θk) * Â^C }
        """
        mean_constraint_loss = 0
        mean_kl = 0
        num_updates = 0
        early_stop_epoch = self.num_learning_epochs

        # Cost loss coefficient from paper: (1-γλ)/(1-γ)
        c_loss_coef = (1 - self.c_gamma * self.c_gae_lam) / (1 - self.c_gamma)

        # Phase 2 update loop with early stopping based on KL divergence
        for epoch in range(self.num_learning_epochs):
            # Generator for mini batches (same as Phase 1, includes cost data)
            if self.policy.is_recurrent:
                generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, 1)
            else:
                generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)

            epoch_kl = 0
            epoch_constraint_loss = 0
            batch_count = 0

            for batch in generator:
                (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                 advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch,
                 returns_cost_batch, old_actions_log_prob_batch,
                 old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch) = batch

                # Normalize cost advantages per mini batch if configured
                if self.normalize_advantage_per_mini_batch:
                    for cost_idx in range(self.num_costs):
                        cost_advantages_batch[:, cost_idx] = (
                            (cost_advantages_batch[:, cost_idx] - cost_advantages_batch[:, cost_idx].mean()) /
                            (cost_advantages_batch[:, cost_idx].std() + 1e-8)
                        )

                # Forward pass to get current policy distribution
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std

                # Compute KL divergence from Phase 1 policy (old_mu, old_sigma from storage)
                kl_divergence = self._compute_kl_divergence(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
                epoch_kl += kl_divergence.item()

                # Importance sampling ratio (w.r.t. behavior policy from data collection)
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

                # Constraint projection loss: sum over all cost constraints
                total_constraint_loss = torch.tensor(0.0, device=self.device)
                for cost_idx in range(self.num_costs):
                    if self.nu[cost_idx] > 0:  # Only apply constraint if multiplier is positive
                        constraint_loss = self.nu[cost_idx] * c_loss_coef * ratio * cost_advantages_batch[:, cost_idx]
                        total_constraint_loss = total_constraint_loss + constraint_loss.mean()

                epoch_constraint_loss += total_constraint_loss.item()

                # Total Phase 2 loss: KL penalty + constraint projection
                total_loss = kl_divergence + total_constraint_loss

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()

                # Collect gradients from all GPUs (before applying gradients)
                if self.is_multi_gpu:
                    self.reduce_parameters()

                # Apply the gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                batch_count += 1
                num_updates += 1

            # Check KL early stopping after each epoch
            avg_epoch_kl = epoch_kl / batch_count if batch_count > 0 else 0
            if avg_epoch_kl > self.delta:
                early_stop_epoch = epoch + 1
                break

            mean_kl += epoch_kl
            mean_constraint_loss += epoch_constraint_loss

        # Average the losses
        if num_updates > 0:
            mean_constraint_loss /= num_updates
            mean_kl /= num_updates

        return mean_constraint_loss, mean_kl, early_stop_epoch

    def update(self, current_costs=None):
        """
        Main CUP update function implementing the two-phase algorithm.

        Paper reference (Algorithm 1):
            Phase 1: Policy Improvement via PPO
                θ_{k+1/2} = argmax { (1/T) Σ (π_θ/π_θk) * Â_t - α√(D̂_KL) }

            Lagrange Update:
                ν_{k+1} = {ν_k + η(Ĵ^C - b)}+

            Phase 2: Projection
                θ_{k+1} = argmin { KL(π_{θ_{k+1/2}}, π_θ) + ν * [(1-γλ)/(1-γ)] * (π_θ/π_θk) * Â^C }

        Args:
            current_costs: List of current cost values for each constraint, used to update
                          Lagrange multipliers. If None, multipliers are not updated.

        Returns:
            dict: Dictionary containing all loss values for logging.
        """
        # ========== Phase 1: Reward-focused PPO update ==========
        # This updates the policy to maximize rewards while training the value function
        phase1_losses = self._ppo_update()

        # ========== Update Lagrange multipliers ==========
        # Update based on constraint violations: ν_{k+1} = {ν_k + η(Ĵ^C - b)}+
        if current_costs is not None:
            self._update_lagrange_multipliers(current_costs)

        # ========== Phase 2: Constraint projection ==========
        # Project policy back towards constraint-satisfying region
        constraint_loss, kl_phase2, early_stop_epoch = self._constraint_projection()

        # Combine all losses for logging
        loss_dict = {
            **phase1_losses,
            "constraint_projection": constraint_loss,
            "phase2_kl": kl_phase2,
            "phase2_early_stop_epoch": early_stop_epoch,
            "lagrange_nu_mean": sum(self.nu) / len(self.nu),
        }

        # Add individual Lagrange multipliers
        for i, nu_val in enumerate(self.nu):
            loss_dict[f"lagrange_nu_{i}"] = nu_val

        # Clear storage after both phases complete
        self.storage.clear()

        return loss_dict
    

    def process_env_step(self, rewards, costs, dones, infos):
        """Process environment step with multiple costs"""
        self.transition.rewards = rewards.clone()
        
        if isinstance(costs, list):
            self.transition.costs = torch.stack([cost.clone() for cost in costs], dim=1)
        else:
            if costs.dim() == 1 and self.num_costs == 1:
                self.transition.costs = costs.unsqueeze(1).clone()
            else:
                self.transition.costs = costs.clone()
        
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            timeout_mask = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * timeout_mask, 1
            )
            # Cost bootstrapping - expand timeout mask to match cost dimensions
            timeout_mask_expanded = timeout_mask.expand(-1, self.num_costs)
            cost_bootstrap = self.gamma * self.transition.cost_values * timeout_mask_expanded
            self.transition.costs += cost_bootstrap

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute GAE returns for rewards"""
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs):
        """Compute GAE returns for costs using CUP-specific discount factors"""
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values, self.c_gamma, self.c_gae_lam,
            normalize_cost_advantage=not self.normalize_advantage_per_mini_batch
        )

    def get_lagrange_info(self):
        """Get Lagrange multiplier information for logging"""
        return {
            "nu_mean": torch.tensor(self.nu).mean().item() if isinstance(self.nu, list) else self.nu,
            "nu_list": self.nu,
            "cost_limits": self.cost_limits,
            "nu_lr": self.nu_lr,
            "nu_max": self.nu_max,
            "delta": self.delta,
            "recent_costs": [
                self.cost_history[i][-5:] if len(self.cost_history[i]) >= 5 else self.cost_history[i]
                for i in range(self.num_costs)
            ]
        }

    def get_penalty_info(self):
        """
        Get penalty information for logging (compatible with OnPolicyRunner).

        Returns a dict with 'kappa', 'kappa_list', and 'cost_limits'
        to match the interface expected by the runner.
        """
        return {
            "kappa": sum(self.nu) / len(self.nu),  # Average for scalar logging
            "kappa_list": self.nu,  # List for per-constraint logging
            "cost_limits": self.cost_limits,
            "nu_lr": self.nu_lr,
            "delta": self.delta,
        }

    def reduce_parameters(self):
        """
        Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize
        the gradients across all GPUs in multi-GPU training.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Unpack the gradients back to the parameters
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                offset += numel

    def _validate_and_fix_cost_critic(self):
        """
        Validate that the cost critic outputs the correct number of cost values.
        If not, recreate the cost critic with the correct output dimension.
        """
        # Check if the policy has a cost_critic attribute
        if not hasattr(self.policy, 'cost_critic'):
            raise ValueError("ActorCritic must have a cost_critic attribute for CUP algorithm")
        
        # Get the last layer of the cost critic to check output dimension
        last_layer = None
        for module in reversed(list(self.policy.cost_critic.modules())):
            if isinstance(module, torch.nn.Linear):
                last_layer = module
                break
        
        if last_layer is None:
            raise ValueError("Could not find linear layer in cost critic")
        
        current_outputs = last_layer.out_features
        
        if current_outputs != self.num_costs:
            print(f"WARNING: Cost critic outputs {current_outputs} values but CUP expects {self.num_costs}.")
            print("This mismatch will cause runtime errors. Please configure ActorCriticCost with num_costs parameter.")
            print(f"Example: ActorCriticCost(..., num_costs={self.num_costs})")