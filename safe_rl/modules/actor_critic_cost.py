from __future__ import annotations

import torch.nn as nn

from safe_rl.modules.actor import GaussianActor
from safe_rl.modules.critic import StandardCritic


class ActorCriticCost(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        num_costs=1,  # Number of cost constraints
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticCost.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor = GaussianActor(
            num_obs=num_actor_obs,
            num_actions=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        self.critic = StandardCritic(
            num_obs=num_critic_obs,
            num_actions=0,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        self.cost_critic = StandardCritic(
            num_obs=num_critic_obs,
            num_actions=0,
            output_dim=num_costs,
            hidden_dims=cost_critic_hidden_dims,
            activation=activation,
        )

        # Action distribution (populated in update_distribution)
        self.distribution = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        self.distribution = self.actor.distribution(observations)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def evaluate_cost(self, critic_observations, **kwargs):
        cost_value = self.cost_critic(critic_observations)
        return cost_value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
