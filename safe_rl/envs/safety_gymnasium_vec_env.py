from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

import gymnasium as gym
import safety_gymnasium

from .vec_env import VecEnv


class SafetyGymnasiumVecEnv(VecEnv):
    """VecEnv wrapper for Safety-Gymnasium vector environments."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        render_mode: str | None = None,
        cost_limits: list[float] | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        camera_name: str | None = None,
    ) -> None:
        make_kwargs: Dict[str, Any] = {"render_mode": render_mode}
        if width is not None:
            make_kwargs["width"] = width
        if height is not None:
            make_kwargs["height"] = height
        if camera_name is not None:
            make_kwargs["camera_name"] = camera_name
        self.env = safety_gymnasium.vector.make(env_id, num_envs=num_envs, **make_kwargs)
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_actions = int(self.env.single_action_space.shape[0])
        self.max_episode_length = self._resolve_max_episode_length()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cost_limits = cost_limits if cost_limits is not None else [1.0]
        self.cfg: Dict[str, Any] = {"env_id": env_id, "num_envs": num_envs, "render_mode": render_mode}
        self._last_obs: torch.Tensor | None = None
        self._last_extras: Dict[str, Any] | None = None
        self.step_dt = 1.0  # used for RND scaling in the runner
        self._seed = seed

    @property
    def unwrapped(self) -> "SafetyGymnasiumVecEnv":
        return self

    def _resolve_max_episode_length(self) -> int:
        max_steps = None
        if hasattr(self.env, "spec") and self.env.spec is not None:
            max_steps = self.env.spec.max_episode_steps
        if max_steps is None and hasattr(self.env, "single_env") and self.env.single_env.spec is not None:
            max_steps = self.env.single_env.spec.max_episode_steps
        if max_steps is None and hasattr(self.env, "envs") and self.env.envs:
            spec = getattr(self.env.envs[0], "spec", None)
            if spec is not None:
                max_steps = spec.max_episode_steps
        return int(max_steps) if max_steps is not None else 1000

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._last_obs is None or self._last_extras is None:
            return self.reset()
        return self._last_obs, self._last_extras

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(seed=self._seed)
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        self.episode_length_buf.zero_()
        extras = self._build_extras(info=info)
        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, extras

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, costs, terminated, truncated, info = self.env.step(actions_np)
        dones = terminated | truncated

        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        rewards_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        costs_tensor = torch.as_tensor(costs, device=self.device, dtype=torch.float32)
        dones_tensor = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        time_outs = torch.as_tensor(truncated, device=self.device, dtype=torch.float32)

        self.episode_length_buf += 1
        if dones_tensor.any():
            done_ids = (dones_tensor > 0).nonzero(as_tuple=False).squeeze(-1)
            self.episode_length_buf[done_ids] = 0

        extras = self._build_extras(info=info, costs=costs_tensor, time_outs=time_outs)
        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, rewards_tensor, dones_tensor, extras

    def close(self) -> None:
        self.env.close()

    def render(self) -> Any:
        # safety-gymnasium's AsyncVectorEnv worker has no 'render' command handler;
        # its native render() kills the worker. Route through call("render") instead.
        if hasattr(self.env, "call"):
            frames = self.env.call("render")
            if frames is None:
                return None
            if self.num_envs == 1:
                return frames[0]
            import numpy as np
            return np.stack(frames, axis=0)
        return self.env.render()

    def _build_extras(
        self,
        info: Dict[str, Any],
        costs: torch.Tensor | None = None,
        time_outs: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        extras: Dict[str, Any] = {"observations": {}}
        if costs is not None:
            extras["costs"] = costs
        if time_outs is not None:
            extras["time_outs"] = time_outs
        if "episode" in info:
            extras["episode"] = info["episode"]
        if "log" in info:
            extras["log"] = info["log"]
        return extras





