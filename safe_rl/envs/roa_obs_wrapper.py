"""Compose the actor observation for Regularized Online Adaptation (ROA).

The env publishes two extra observation groups next to ``actor`` and ``critic``:

* ``priv``: the randomized physical parameters, ``[num_envs, num_priv]`` (simulation only);
* ``actor_history``: the last ``hist_len`` proprioceptive steps, ``[num_envs, hist_len, hist_dim]``,
  oldest first.

This wrapper returns ``[obs | priv | history]`` as the actor observation, the layout
:class:`~safe_rl.modules.ActorCriticROA` splits. It is applied by the runner only for ROA
policies, so a baseline on the same task never sees ``priv``.
"""

from __future__ import annotations

import torch
from typing import Any

from safe_rl.envs.vec_env import VecEnv

PRIV_KEY = "priv"
HISTORY_KEY = "actor_history"


class ROAObsWrapper(VecEnv):
    def __init__(self, env: VecEnv) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg

        obs, extras = env.get_observations()
        groups = extras.get("observations", {})
        missing = [key for key in (PRIV_KEY, HISTORY_KEY) if key not in groups]
        if missing:
            raise ValueError(
                f"ROA needs the observation groups {missing} from the env; "
                f"got {sorted(groups)}. Use a task that defines them (e.g. Unitree-G1-Flat-ROA)."
            )
        history = groups[HISTORY_KEY]
        if history.ndim != 3:
            raise ValueError(
                f"'{HISTORY_KEY}' must be [num_envs, hist_len, hist_dim] (flatten_history_dim=False); "
                f"got shape {tuple(history.shape)}"
            )
        self.roa_obs_dims = {
            "num_prop": int(obs.shape[1]),
            "num_priv": int(groups[PRIV_KEY].shape[1]),
            "hist_len": int(history.shape[1]),
            "hist_dim": int(history.shape[2]),
        }
        self._last: tuple[torch.Tensor, dict] | None = self._compose(obs, extras)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def _compose(self, obs: torch.Tensor, extras: dict) -> tuple[torch.Tensor, dict]:
        groups = extras["observations"]
        actor_obs = torch.cat([obs, groups[PRIV_KEY], groups[HISTORY_KEY].flatten(1)], dim=-1)
        return actor_obs, extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last is None:
            return self.reset()
        return self._last

    def reset(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict]:
        self._last = self._compose(*self.env.reset(*args, **kwargs))
        return self._last

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        self._last = self._compose(obs, extras)
        return self._last[0], rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["env"], name)
