from __future__ import annotations

import inspect
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
        self.step_dt = float(env.step_dt)
        self.cfg = env.cfg
        self.clip_actions = clip_actions
        # cost_limits are owned by the env's cost manager (env.cost_limits). Read
        # them from there; the explicit kwarg only acts as an override/fallback.
        if cost_limits is None:
            unwrapped = getattr(env, "unwrapped", env)
            cost_limits = getattr(unwrapped, "cost_limits", None)
        self.cost_limits = cost_limits
        self._last_obs: torch.Tensor | None = None
        self._last_extras: dict[str, Any] | None = None
        self._require_final_observations = False
        self._snapshot_groups: tuple[str, ...] = ()
        self._final_observations: dict[str, torch.Tensor] | None = None

    def enable_final_observations(self) -> None:
        """Require real timeout successors for off-policy replay.

        Native final observations are forwarded when available. Older mjlab
        releases need a pre-reset snapshot. That fallback is limited to stateless,
        uncorrupted observations without commands or step/interval events (e.g.
        Ant-Flat): evaluating history, delay or stateful terms twice would change
        the observation process. Other tasks must publish native final observations.
        """
        self._require_final_observations = True
        self._snapshot_groups = ()
        env = self.unwrapped
        manager = getattr(env, "observation_manager", None)
        if manager is None or not callable(getattr(env, "_reset_idx", None)):
            return
        if getattr(env.cfg, "commands", {}):
            return
        events = getattr(env.cfg, "events", {})
        if any(getattr(event, "mode", None) in ("step", "interval") for event in events.values()):
            return
        groups = tuple(key for key in ("actor", "policy", "critic") if key in manager.active_terms)
        if not any(key in groups for key in ("actor", "policy")):
            return
        for group in groups:
            cfg = manager.cfg[group]
            if cfg.history_length or cfg.enable_corruption or not cfg.concatenate_terms:
                return
            for term in manager.active_terms[group]:
                cfg = manager.get_term_cfg(group, term)
                if cfg.history_length or cfg.delay_max_lag or not inspect.isfunction(cfg.func):
                    return
        self._snapshot_groups = groups

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # Must reach the UNDERLYING env: the runner's init_at_random_ep_len
        # staggering assigns a fresh tensor to this attribute, and a plain
        # attribute copy on the wrapper would swallow it silently — all envs
        # then run phase-locked and truncate simultaneously, which correlates
        # minibatches and mass-cuts the lambda-trace (observed: REPPO tracking
        # plateau on Go2 that the reference pipeline, which staggers for real,
        # does not have).
        self.env.episode_length_buf = value

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last_obs is None or self._last_extras is None:
            return self.reset()
        return self._last_obs, self._last_extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._final_observations = None
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

        result = self._step_with_final_observations(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Expected mjlab env.step() to return (obs, rewards, terminated, truncated, info).")

        obs, rewards, terminated, truncated, info = result
        obs_tensor, extras = self._convert_observations(obs, info)

        rewards_tensor = self._as_float_tensor(rewards)
        terminated_tensor = self._as_float_tensor(terminated)
        # A true termination takes precedence when both flags are set.
        truncated_tensor = self._as_float_tensor(truncated) * (terminated_tensor == 0)
        dones = torch.logical_or(terminated_tensor > 0, truncated_tensor > 0).to(torch.float32)
        extras["time_outs"] = truncated_tensor
        self._forward_final_observations(info, obs_tensor, extras, truncated_tensor)

        costs = info.get("costs", info.get("cost"))
        if costs is not None:
            extras["costs"] = self._as_float_tensor(costs)

        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, rewards_tensor, dones, extras

    def _step_with_final_observations(self, actions: torch.Tensor):
        self._final_observations = None
        if not self._snapshot_groups:
            return self.env.step(actions)

        env = self.unwrapped
        reset_idx = env._reset_idx

        def capture_then_reset(env_ids):
            timeouts = env.reset_time_outs.bool() & ~env.reset_terminated.bool()
            if timeouts[env_ids].any():
                # mj_step leaves derived quantities one physics substep behind.
                # Refresh them before observing the terminal state. compute_group
                # bypasses compute()'s previous-step cache without updating history.
                env.sim.forward()
                env.sim.sense()
                self._final_observations = {
                    group: env.observation_manager.compute_group(group, update_history=False).clone()
                    for group in self._snapshot_groups
                }
            return reset_idx(env_ids)

        # Scope the hook to this step; manual resets and other env users keep the
        # original method, including when simulation or observation code raises.
        missing = object()
        previous_override = vars(env).get("_reset_idx", missing)
        env._reset_idx = capture_then_reset
        try:
            return self.env.step(actions)
        finally:
            if previous_override is missing:
                del env._reset_idx
            else:
                env._reset_idx = previous_override

    def _forward_final_observations(
        self, info: dict[str, Any], obs: torch.Tensor, extras: dict[str, Any], timeouts: torch.Tensor
    ) -> None:
        final = info.get("final_observation")
        if final is None:
            final = info.get("terminal_observation")
        if final is None:
            final = self._final_observations
        if final is None:
            if self._require_final_observations and timeouts.any():
                raise RuntimeError(
                    "mjlab truncated without final_observation; cannot bootstrap from an auto-reset observation. "
                    "Native final observations are required for tasks with history, delay, corruption, "
                    "stateful observation terms, commands or step/interval events."
                )
            return

        final_actor, final_extras = self._convert_observations(final, {})
        if final_actor.shape != obs.shape:
            raise ValueError(f"Final actor observation shape {final_actor.shape} does not match {obs.shape}.")
        extras["final_observation"] = final_actor.clone()
        final_critic = info.get("final_critic_observation", final_extras["observations"].get("critic"))
        critic = extras["observations"].get("critic")
        if final_critic is not None and critic is not None:
            final_critic = self._as_float_tensor(final_critic)
            if final_critic.shape != critic.shape:
                raise ValueError(f"Final critic observation shape {final_critic.shape} does not match {critic.shape}.")
            extras["final_critic_observation"] = final_critic.clone()
        elif self._require_final_observations and timeouts.any() and critic is not None:
            raise RuntimeError("mjlab truncated without a final critic observation for the separate critic input.")

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
