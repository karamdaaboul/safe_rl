"""Brax VecEnv wrapper: run google/brax GPU-parallel envs behind the safe_rl VecEnv API.

Env ids use the ``Brax`` prefix, e.g. ``BraxAnt`` -> brax ``ant``,
``BraxHalfcheetah`` -> ``halfcheetah``. Physics steps stay on the GPU in JAX;
observations/rewards cross into torch via dlpack (zero-copy on GPU).

Notes:
- brax's ``envs.create(batch_size=..., auto_reset=True)`` supplies the episode
  wrapper, vmap batching, and auto-reset; ``state.info["truncation"]`` marks
  time-limit terminations (surfaced as ``extras["time_outs"]``).
- Actions are expected in [-1, 1] (brax convention) and are clipped here.
- Extra Safety-Gymnasium-specific kwargs from ``make_env`` are accepted and
  ignored (with a notice) so the standard train script works unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from .vec_env import VecEnv


def _jax_to_torch(x) -> torch.Tensor:
    return torch.from_dlpack(x)


class BraxVecEnv(VecEnv):
    def __init__(
        self,
        env_id: str,
        num_envs: int,
        device: str = "cuda:0",
        seed: int | None = 0,
        episode_length: int = 1000,
        backend: str = "generalized",
        reward_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        ignored = {k: v for k, v in kwargs.items() if v not in (None, False)}
        if ignored:
            print(f"BraxVecEnv: ignoring non-applicable kwargs: {sorted(ignored)}")

        import jax
        import jax.numpy as jnp
        from brax import envs as brax_envs

        self._jax = jax
        self._jnp = jnp

        assert env_id.startswith("Brax"), env_id
        self.env_id = env_id
        env_name = env_id[len("Brax"):].lower()

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.reward_scale = float(reward_scale)
        self.max_episode_length = int(episode_length)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {"env_id": env_id, "backend": backend, "episode_length": episode_length}
        self.cost_limits = None

        self._env = brax_envs.create(
            env_name,
            episode_length=episode_length,
            action_repeat=1,
            auto_reset=True,
            batch_size=self.num_envs,
            backend=backend,
        )
        self.num_actions = int(self._env.action_size)
        self.num_obs = int(self._env.observation_size)
        self.num_privileged_obs = 0

        self._step_fn = jax.jit(self._env.step)
        self._reset_fn = jax.jit(self._env.reset)
        self._key = jax.random.PRNGKey(0 if seed is None else int(seed))
        self._state = None
        self._last_obs: torch.Tensor | None = None

    # ------------------------------------------------------------------

    def _obs_to_torch(self, obs) -> torch.Tensor:
        t = _jax_to_torch(obs)
        if t.device != self.device:
            t = t.to(self.device)
        return t.float()

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._key, sub = self._jax.random.split(self._key)
        self._state = self._reset_fn(sub)
        self.episode_length_buf.zero_()
        obs = self._obs_to_torch(self._state.obs)
        self._last_obs = obs
        return obs, {"observations": {}}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last_obs is None:
            return self.reset()
        return self._last_obs, {"observations": {}}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._state is None:
            self.reset()
        actions = actions.detach().clamp(-1.0, 1.0).contiguous()
        jax_actions = self._jnp.from_dlpack(actions)
        self._state = self._step_fn(self._state, jax_actions)

        obs = self._obs_to_torch(self._state.obs)
        rewards = self._obs_to_torch(self._state.reward).view(-1) * self.reward_scale
        dones = self._obs_to_torch(self._state.done).view(-1) > 0.5
        truncation = self._obs_to_torch(self._state.info["truncation"]).view(-1) > 0.5

        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0

        extras: dict = {"observations": {}, "time_outs": truncation}
        self._last_obs = obs
        return obs, rewards, dones, extras

    def close(self) -> None:
        pass
