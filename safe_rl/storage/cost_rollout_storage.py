# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from safe_rl.utils import split_and_pad_trajectories


class RolloutStorageCMDP:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.costs = None  # Cost vector for multiple constraints
            self.dones = None
            self.values = None
            self.cost_values = None  # Single cost critic output with multiple neurons
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        training_type="rl",
        cost_shape=None,  # Shape for cost vector (num_costs,)
        rnd_state_shape=None,
        device="cpu",
    ):
        # Store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape
        self.cost_shape = cost_shape

        # Determine number of costs
        if cost_shape is not None:
            self.num_costs = cost_shape[0] if len(cost_shape) > 0 else 1
        else:
            self.num_costs = 1  # Default single cost for backward compatibility

        # Core storage tensors
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
            
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        
        # Cost storage for multiple constraints
        if cost_shape is not None:
            self.costs = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        else:
            # Backward compatibility: single cost
            self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()


        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type in ["rl", "saferl"]:
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            
            # SINGLE cost critic with multiple output neurons
            if cost_shape is not None:
                self.cost_values = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
            else:
                # Backward compatibility
                self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            
            # Returns and advantages
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            
            # Cost returns and advantages for multiple cost outputs
            if cost_shape is not None:
                self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
                self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
            else:
                # Backward compatibility
                self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        else:
            # For non-RL training types, set cost attributes to None
            self.cost_values = None
            self.cost_returns = None
            self.cost_advantages = None

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        
        # Handle cost vectors
        if transition.costs is not None:
            if self.cost_shape is not None:
                # Multiple costs case
                if transition.costs.dim() == 1:
                    # Single environment case
                    self.costs[self.step].copy_(transition.costs.view(1, -1))
                else:
                    # Multiple environments case
                    self.costs[self.step].copy_(transition.costs)
            else:
                # Backward compatibility: single cost
                if isinstance(transition.costs, list):
                    # Convert list to tensor and take first cost for backward compatibility
                    cost_tensor = torch.stack(transition.costs) if len(transition.costs) > 1 else transition.costs[0]
                    self.costs[self.step].copy_(cost_tensor.view(-1, 1))
                else:
                    self.costs[self.step].copy_(transition.costs.view(-1, 1))
                    
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type in ["rl", "saferl"]:
            self.values[self.step].copy_(transition.values.view(-1, 1))
            
            # Handle single cost critic with multiple outputs
            if transition.cost_values is not None:
                if self.cost_shape is not None:
                    # Single cost critic with multiple output neurons
                    if transition.cost_values.dim() == 1:
                        self.cost_values[self.step].copy_(transition.cost_values.view(1, -1))
                    else:
                        self.cost_values[self.step].copy_(transition.cost_values)
                else:
                    # Backward compatibility: single cost output
                    self.cost_values[self.step].copy_(transition.cost_values.view(-1, 1))
                        
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None and transition.rnd_state is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden state to match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states, tuple) else (hidden_states,)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # Copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def get_mean_episode_costs(self) -> torch.Tensor:
        """
        Get the mean episode costs using GAE cost returns.
        Returns tensor of shape (num_costs,) with mean costs for each constraint.
        Uses cost_returns which represent bias-corrected estimates of total episode costs.
        """
        if self.training_type not in ["rl", "saferl"]:
            raise ValueError("Episode costs are only available for reinforcement learning training.")
        
        # Use GAE cost returns for constraint violation detection
        if self.cost_shape is not None:
            # Multiple costs case - use cost returns (GAE estimates of episode costs)
            cost_returns_flattened = self.cost_returns.flatten(0, 1)  # Shape: (total_steps, num_costs)
            return cost_returns_flattened.mean(dim=0)  # Shape: (num_costs,)
        else:
            # Single cost case
            cost_returns_flattened = self.cost_returns.flatten(0, 1)  # Shape: (total_steps, 1)
            return cost_returns_flattened.mean(dim=0)  # Shape: (1,)

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_cost_returns(self, last_cost_values, gamma, lam, normalize_cost_advantage: bool = True):
        """
        Compute GAE cost returns for single cost critic with multiple output neurons.
        last_cost_values: tensor of shape (num_envs, num_costs) from single cost critic
        """
        if self.training_type not in ["rl", "saferl"]:
            raise ValueError("Cost returns can only be computed for reinforcement learning training.")
        
        if self.cost_shape is not None:
            # Single cost critic with multiple output neurons - vectorized computation
            cost_advantage = torch.zeros_like(self.cost_advantages[0])  # Shape: (num_envs, num_costs)
            
            for step in reversed(range(self.num_transitions_per_env)):
                # If we are at the last step, bootstrap the cost return value
                if step == self.num_transitions_per_env - 1:
                    next_cost_values = last_cost_values  # Shape: (num_envs, num_costs)
                else:
                    next_cost_values = self.cost_values[step + 1]
                    
                # 1 if we are not in a terminal state, 0 otherwise
                next_is_not_terminal = 1.0 - self.dones[step].float()  # Shape: (num_envs, 1)
                
                # Broadcast for multiple costs
                next_is_not_terminal_broadcast = next_is_not_terminal.expand(-1, self.num_costs)
                
                # TD error: c_t + gamma * V_c(s_{t+1}) - V_c(s_t) for each cost output
                cost_delta = (self.costs[step] + 
                             next_is_not_terminal_broadcast * gamma * next_cost_values - 
                             self.cost_values[step])
                
                # Cost Advantage: A_c(s_t, a_t) = cost_delta_t + gamma * lambda * A_c(s_{t+1}, a_{t+1})
                cost_advantage = cost_delta + next_is_not_terminal_broadcast * gamma * lam * cost_advantage
                
                # Cost Return: R_c_t = A_c(s_t, a_t) + V_c(s_t)
                self.cost_returns[step] = cost_advantage + self.cost_values[step]
                
        else:
            # Single cost case - backward compatibility
            cost_advantage = 0
            
            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    next_cost_values = last_cost_values
                else:
                    next_cost_values = self.cost_values[step + 1]
                    
                next_is_not_terminal = 1.0 - self.dones[step].float()
                
                cost_delta = self.costs[step] + next_is_not_terminal * gamma * next_cost_values - self.cost_values[step]
                cost_advantage = cost_delta + next_is_not_terminal * gamma * lam * cost_advantage
                self.cost_returns[step] = cost_advantage + self.cost_values[step]

        # Compute the cost advantages
        self.cost_advantages = self.cost_returns - self.cost_values
        
        # Normalize cost advantages per cost output neuron
        if normalize_cost_advantage:
            if self.cost_shape is not None:
                # Normalize each cost output separately
                for cost_idx in range(self.num_costs):
                    cost_adv_flat = self.cost_advantages[:, :, cost_idx].flatten()
                    cost_adv_normalized = (cost_adv_flat - cost_adv_flat.mean()) / (cost_adv_flat.std() + 1e-8)
                    self.cost_advantages[:, :, cost_idx] = cost_adv_normalized.view(self.num_transitions_per_env, self.num_envs)
            else:
                # Single cost case
                self.cost_advantages = (self.cost_advantages - self.cost_advantages.mean()) / (self.cost_advantages.std() + 1e-8)

    # For distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[i], self.dones[i]

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type not in ["rl", "saferl"]:
            raise ValueError("This function is only available for reinforcement learning training.")
            
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For SafeRL with single cost critic having multiple outputs
        if self.cost_values is not None:
            cost_values = self.cost_values.flatten(0, 1)
            cost_returns = self.cost_returns.flatten(0, 1)
            cost_advantages = self.cost_advantages.flatten(0, 1)
        else:
            cost_values = None
            cost_returns = None
            cost_advantages = None

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For SafeRL with single cost critic having multiple outputs
                if cost_values is not None:
                    target_cost_values_batch = cost_values[batch_idx]
                    cost_advantages_batch = cost_advantages[batch_idx]
                    returns_cost_batch = cost_returns[batch_idx]
                else:
                    target_cost_values_batch = None
                    cost_advantages_batch = None
                    returns_cost_batch = None

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # Yield the mini-batch
                yield (obs_batch, privileged_observations_batch, actions_batch, target_values_batch, 
                       advantages_batch, returns_batch, target_cost_values_batch, cost_advantages_batch, 
                       returns_cost_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
                       (None, None), None, rnd_state_batch)

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type not in ["rl", "saferl"]:
            raise ValueError("This function is only available for reinforcement learning training.")
            
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                
                # Handle single cost critic with multiple outputs in recurrent case
                if self.cost_values is not None:
                    cost_values_batch = self.cost_values[:, start:stop]
                    cost_advantages_batch = self.cost_advantages[:, start:stop]
                    cost_returns_batch = self.cost_returns[:, start:stop]
                else:
                    cost_values_batch = None
                    cost_advantages_batch = None
                    cost_returns_batch = None
                    
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # Reshape hidden states for recurrent networks
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # Remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch if len(hid_c_batch) == 1 else hid_c_batch

                # Yield the mini-batch
                yield (obs_batch, privileged_obs_batch, actions_batch, values_batch, 
                       advantages_batch, returns_batch, cost_values_batch, cost_advantages_batch, 
                       cost_returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
                       (hid_a_batch, hid_c_batch), masks_batch, rnd_state_batch)

                first_traj = last_traj
