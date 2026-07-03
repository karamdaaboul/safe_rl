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
        hidden_goal: bool = False,
        hidden_goal_continue: bool = False,
        task_seeds: list[int] | None = None,
        cbf_state: bool = False,
    ) -> None:
        make_kwargs: Dict[str, Any] = {"render_mode": render_mode}
        if width is not None:
            make_kwargs["width"] = width
        if height is not None:
            make_kwargs["height"] = height
        if camera_name is not None:
            make_kwargs["camera_name"] = camera_name

        # Build a chain of per-sub-env wrappers. Each wrapper callable takes the
        # raw gym env and returns a wrapped env; we compose them left-to-right.
        wrapper_chain: list = []
        if hidden_goal:
            from functools import partial
            from .hidden_goal_wrapper import HiddenGoalWrapper
            wrapper_chain.append(partial(HiddenGoalWrapper, continue_goal=hidden_goal_continue))
        if cbf_state:
            from safe_rl.cbf.sg_state_wrapper import SGCBFStateWrapper
            wrapper_chain.append(SGCBFStateWrapper)

        if wrapper_chain:
            if len(wrapper_chain) == 1:
                make_kwargs["wrappers"] = wrapper_chain[0]
            else:
                # compose: outermost wrapper applied last
                def _compose(env, _chain=wrapper_chain):
                    for w in _chain:
                        env = w(env)
                    return env
                make_kwargs["wrappers"] = _compose

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
        # Fixed multi-task set: a list of seeds spread round-robin across the
        # parallel envs so a single (non-adaptive) policy trains jointly on N
        # hidden-goal tasks. With HiddenGoalWrapper(fix_task=True) each sub-env
        # keeps its own goal across auto-resets, so the N goals stay constant.
        self._task_seeds = [int(s) for s in task_seeds] if task_seeds else None
        # Count goals reached within each (per-env) episode. `goal_met` fires once
        # per goal; with continue_goal=True a single episode chains several, so the
        # per-episode total measures how many hidden goals a policy reaches.
        self._goals_in_episode = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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

    def set_task(self, seed: int | None) -> None:
        """Select the task for the next reset.

        In Safety-Gymnasium a "task" is the obstacle/goal layout, which is fully
        determined by the reset seed (the cost function itself is task-independent).
        Meta-RL runners call this to switch tasks between inner-loop adaptations.
        The new seed takes effect on the next ``reset()``.
        """
        self._seed = seed

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        # Build the explicit per-env seed list.
        # - task_seeds set: spread the N task seeds round-robin across the envs
        #   (env i -> task_seeds[i % N]) so one policy trains jointly on N fixed
        #   hidden-goal tasks. Use num_envs a multiple of N for an even split.
        # - else a single seed: tile it so all sub-envs share the SAME task
        #   (gymnasium would otherwise spread a bare int as seed+i = N layouts);
        #   this is what per-task (MAML) inner-loop adaptation needs.
        if self._task_seeds is not None:
            n = len(self._task_seeds)
            seeds = [self._task_seeds[i % n] for i in range(self.num_envs)]
        elif self._seed is not None:
            seeds = [self._seed] * self.num_envs
        else:
            seeds = None
        obs, info = self.env.reset(seed=seeds)
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        self.episode_length_buf.zero_()
        self._goals_in_episode.zero_()
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

        # Tally goals reached this step. `info['goal_met']` is a per-env bool array,
        # present only on steps where at least one env reached its goal.
        goal_met = info.get("goal_met")
        if goal_met is not None:
            import numpy as np
            self._goals_in_episode += torch.as_tensor(
                np.asarray(goal_met, dtype=np.float32), device=self.device
            )

        self.episode_length_buf += 1
        extras = self._build_extras(info=info, costs=costs_tensor, time_outs=time_outs)
        if dones_tensor.any():
            done_ids = (dones_tensor > 0).nonzero(as_tuple=False).squeeze(-1)
            # Per-episode goal totals for the finished envs -> logged by the runner
            # as Episode/goals_reached; then reset those counters.
            extras.setdefault("log", {})["goals_reached"] = self._goals_in_episode[done_ids].clone()
            self.episode_length_buf[done_ids] = 0
            self._goals_in_episode[done_ids] = 0.0

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

        # Forward the true terminal observations on truncation so Q-bootstrap is
        # correct (gymnasium puts the new-episode reset obs in `obs` after auto-
        # reset). `final_observation` is an object array of shape [num_envs] with
        # the real terminal obs for terminated/truncated envs and None elsewhere.
        final_obs = info.get("final_observation")
        if final_obs is not None:
            import numpy as np
            stacked = np.zeros((self.num_envs, *self._last_obs.shape[1:]), dtype=np.float32) \
                if self._last_obs is not None else None
            if stacked is not None:
                for i, fo in enumerate(final_obs):
                    if fo is not None:
                        stacked[i] = np.asarray(fo, dtype=np.float32)
                extras["final_observation"] = torch.as_tensor(stacked, device=self.device)
        return extras





