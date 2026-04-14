from __future__ import annotations

from typing import Any

import torch

from .vec_env import VecEnv


class MjlabVecEnv(VecEnv):
    """Adapter that exposes an mjlab RL environment through the safe_rl VecEnv API."""

    def __init__(self, env: Any, clip_actions: float | None = None, cost_limits: list[float] | None = None) -> None:
        self.env = env
        self.device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
        self.num_envs = int(env.num_envs)
        self.num_actions = self._resolve_num_actions(env)
        self.max_episode_length = int(
            getattr(env, "max_episode_length", round(getattr(env, "max_episode_length_s") / env.step_dt))
        )
        self.episode_length_buf = env.episode_length_buf
        self.step_dt = float(env.step_dt)
        self.cfg = env.cfg
        self.clip_actions = clip_actions
        self.cost_limits = cost_limits
        self._last_obs: torch.Tensor | None = None
        self._last_extras: dict[str, Any] | None = None

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last_obs is None or self._last_extras is None:
            return self.reset()
        return self._last_obs, self._last_extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        obs_tensor, extras = self._convert_observations(obs, info)
        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(self.device)
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Expected mjlab env.step() to return (obs, rewards, terminated, truncated, info).")

        obs, rewards, terminated, truncated, info = result
        obs_tensor, extras = self._convert_observations(obs, info)

        rewards_tensor = self._as_float_tensor(rewards)
        terminated_tensor = self._as_float_tensor(terminated)
        truncated_tensor = self._as_float_tensor(truncated)
        dones = torch.logical_or(terminated_tensor > 0, truncated_tensor > 0).to(torch.float32)
        extras["time_outs"] = truncated_tensor

        costs = info.get("costs", info.get("cost"))
        if costs is not None:
            extras["costs"] = self._as_float_tensor(costs)

        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, rewards_tensor, dones, extras

    def close(self) -> None:
        self.env.close()

    def _convert_observations(self, obs: Any, info: dict[str, Any] | None) -> tuple[torch.Tensor, dict[str, Any]]:
        obs_groups: dict[str, Any] = {}
        if isinstance(obs, dict):
            obs_groups.update(obs)

        info = info or {}
        info_observations = info.get("observations")
        if isinstance(info_observations, dict):
            for key, value in info_observations.items():
                obs_groups.setdefault(key, value)

        actor_obs = obs_groups.get("actor", obs_groups.get("policy", obs))
        actor_tensor = self._as_float_tensor(actor_obs)

        extras: dict[str, Any] = {"observations": {}}
        for key in ("critic", "rnd_state", "teacher"):
            if key in obs_groups:
                extras["observations"][key] = self._as_float_tensor(obs_groups[key])
        if "episode" in info:
            extras["episode"] = info["episode"]
        if "log" in info:
            extras["log"] = info["log"]

        return actor_tensor, extras

    def _as_float_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    @staticmethod
    def _resolve_num_actions(env: Any) -> int:
        if hasattr(env, "num_actions"):
            return int(env.num_actions)
        action_manager = getattr(env, "action_manager", None)
        if action_manager is not None and hasattr(action_manager, "total_action_dim"):
            return int(action_manager.total_action_dim)
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            return int(env.action_space.shape[-1])
        raise AttributeError("Could not resolve action dimension from mjlab environment.")
