from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

import gymnasium as gym
import safety_gymnasium

from .vec_env import VecEnv


class _SafetyEnvFactory:
    """Top-level, picklable env-builder for spawn/forkserver workers.

    ``safety_gymnasium.vector.make`` builds its per-env callables as nested
    closures, which cannot be pickled — so it only works with the default
    ``fork`` start method. For the vision envs, fork + MuJoCo EGL rendering
    deadlocks (a worker's render exception holds an EGL ctypes context that
    cannot be pickled back through the error queue, hanging the whole run).
    A ``spawn`` context gives each worker a clean interpreter and sidesteps
    that, but needs a picklable factory — this class.
    """

    def __init__(self, env_id, make_kwargs, disable_env_checker, wrappers=None):
        self.env_id = env_id
        self.make_kwargs = make_kwargs
        self.disable_env_checker = disable_env_checker
        self.wrappers = wrappers or []

    def __call__(self):
        env = safety_gymnasium.make(
            self.env_id, disable_env_checker=self.disable_env_checker, **self.make_kwargs
        )
        for wrapper in self.wrappers:
            env = wrapper(env)
        return env


def _require_fcsrl_wrapper(name: str):
    """Fetch a wrapper that exists only in the local safety-gymnasium fork."""
    try:
        return getattr(safety_gymnasium.wrappers, name)
    except AttributeError:
        raise ImportError(
            f"safety_gymnasium.wrappers.{name} is missing. It ships only with the local fork "
            "(github.com/karamdaaboul/safety-gymnasium); reinstall safety-gymnasium from there, "
            "or drop the action_repeat / goal_pseudo_terminal options."
        ) from None


def _fcsrl_wrapper_chain(action_repeat: int, goal_pseudo_terminal: bool) -> list:
    """FCSRL-style treatments, both off by default (codex/fcsrl-harness-tricks.md).

    The goal wrapper goes inside the action repeat so a goal reached on any repeated
    step is OR-ed into that agent step's ``pseudo_terminated``.
    """
    if action_repeat < 1:
        raise ValueError(f"action_repeat must be a positive integer, got {action_repeat}.")

    chain: list = []
    if goal_pseudo_terminal:
        chain.append(_require_fcsrl_wrapper("SafeGoalMetTerminal"))
    if action_repeat > 1:
        from functools import partial

        chain.append(partial(_require_fcsrl_wrapper("SafeActionRepeat"), n_repeat=action_repeat))
    return chain


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
        geom_margin: bool = False,
        geom_margin_d_safe: float = 0.4,
        geom_margin_min: float | None = None,
        action_repeat: int = 1,
        goal_pseudo_terminal: bool = False,
        cost_limit_curriculum: dict | None = None,
        risk_modes: int = 0,
        risk_fixed_level: float | None = None,
        cbf_state: bool = False,
        vision: bool = False,
        vision_size: int = 64,
        asynchronous: bool = True,
        mp_context: str | None = None,
    ) -> None:
        make_kwargs: Dict[str, Any] = {"render_mode": render_mode}
        if width is not None:
            make_kwargs["width"] = width
        if height is not None:
            make_kwargs["height"] = height
        if camera_name is not None:
            make_kwargs["camera_name"] = camera_name
        if vision:
            # Render the vision observation directly at the target size (the
            # registered default is 256x256); merged into the task config by
            # safety_gymnasium's `make` and parsed as a dotted key.
            make_kwargs["config"] = {"vision_env_conf.vision_size": (vision_size, vision_size)}

        wrapper_chain = self._build_wrapper_chain(
            cbf_state=cbf_state,
            geom_margin=geom_margin,
            geom_margin_d_safe=geom_margin_d_safe,
            geom_margin_min=geom_margin_min,
            cost_limit_curriculum=cost_limit_curriculum,
            action_repeat=action_repeat,
            goal_pseudo_terminal=goal_pseudo_terminal,
        )
        self.action_repeat = int(action_repeat)
        self.goal_pseudo_terminal = bool(goal_pseudo_terminal)
        self.cost_limit_curriculum = cost_limit_curriculum or None

        # Use explicit, picklable factories when a non-default start method is
        # requested (e.g. spawn for the vision envs) or when running the
        # single-process synchronous vector env; otherwise fall back to
        # safety_gymnasium.vector.make (default fork async), unchanged.
        if not asynchronous:
            # safety_gymnasium's SafetySyncVectorEnv inherits gymnasium's base
            # step(), which unpacks the standard 5-tuple and cannot handle the
            # safety envs' 6-tuple (obs, reward, cost, terminated, truncated,
            # info) — so a single-process sync vector env is not usable here.
            # Use an async context instead (spawn for the vision envs).
            raise NotImplementedError(
                "SafetyGymnasiumVecEnv does not support asynchronous=False: "
                "safety_gymnasium's synchronous vector env drops the cost channel. "
                "Use mp_context='spawn' for the vision envs instead."
            )
        if mp_context is not None:
            from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv

            env_fns = [
                _SafetyEnvFactory(
                    env_id,
                    make_kwargs,
                    disable_env_checker=(i > 0),
                    wrappers=wrapper_chain,
                )
                for i in range(num_envs)
            ]
            self.env = SafetyAsyncVectorEnv(env_fns, context=mp_context)
        else:
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

        # Dict observation spaces (the `*Vision-v0` envs register with
        # observation_flatten=False) are split here: all non-image keys are
        # concatenated into the flat state tensor that stays the primary obs,
        # while the uint8 image batch rides along in extras["observations"].
        obs_space = self.env.single_observation_space
        self._dict_obs = isinstance(obs_space, gym.spaces.Dict)
        self._vision_key = "vision"
        if self._dict_obs:
            self._state_keys = [k for k in obs_space.spaces if k != self._vision_key]
            self.obs_key_slices: Dict[str, slice] = {}
            offset = 0
            for key in self._state_keys:
                n = int(np.prod(obs_space.spaces[key].shape))
                self.obs_key_slices[key] = slice(offset, offset + n)
                offset += n
            self.num_state_obs = offset
        self.max_episode_length = self._resolve_max_episode_length()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cost_limits = cost_limits if cost_limits is not None else [1.0]
        self.cfg: Dict[str, Any] = {"env_id": env_id, "num_envs": num_envs, "render_mode": render_mode}
        self._last_obs: torch.Tensor | None = None
        self._last_extras: Dict[str, Any] | None = None
        self.step_dt = 1.0  # used for RND scaling in the runner
        self._seed = seed
        # Count goals reached within each (per-env) episode. `goal_met` fires once
        # per goal; with continue_goal=True a single episode chains several.
        self._goals_in_episode = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._reward_in_episode = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._cost_in_episode = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Risk-conditioned policy: ONE scalar risk level is appended to the observation, so the
        # actor, both critics and the replay buffer carry it with no schema change. Scalar
        # rather than one-hot so the level is ordered -- 0 = risk-seeking, 1 = risk-averse --
        # which lets a trained policy interpolate to levels never sampled during training.
        # Drawn once per episode, not per step: Q_c is the cost-to-go of the policy actually
        # being run, and switching mid-episode makes the bootstrap a mixture over modes.
        self.num_risk_modes = int(risk_modes)
        # Evaluation pins the level instead of sampling it -- that IS the deliverable: one
        # checkpoint, a safety level chosen at deployment. Kept as a float so a level between
        # trained modes can be queried, which the scalar conditioning supports.
        self.risk_fixed_level = None if risk_fixed_level is None else float(risk_fixed_level)
        self._risk_idx = self._draw_risk(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    @property
    def risk_obs_dim(self) -> int:
        """Extra observation columns the risk conditioning adds (0 or 1)."""
        return 1 if self.num_risk_modes else 0

    def _risk_name(self, mode: int) -> str:
        """Readable name for a mode. `cost_risk0` needs the config to decode; `cost_seeking`
        does not, and these keys are read on a dashboard, not in code."""
        if self.num_risk_modes == 3:
            return ("seeking", "neutral", "averse")[mode]
        if self.num_risk_modes == 2:
            return ("seeking", "averse")[mode]
        return f"mode{mode}"

    def _draw_risk(self, mask: torch.Tensor, current: torch.Tensor | None = None) -> torch.Tensor:
        """Resample risk modes where ``mask``; keep the rest."""
        if not self.num_risk_modes:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        drawn = torch.randint(0, self.num_risk_modes, (self.num_envs,), device=self.device)
        return drawn if current is None else torch.where(mask, drawn, current)

    def _append_risk(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate the scalar risk level, mode index mapped onto [0, 1]."""
        if not self.num_risk_modes:
            return obs
        if self.risk_fixed_level is not None:
            level = torch.full((obs.shape[0], 1), self.risk_fixed_level, device=obs.device, dtype=obs.dtype)
        else:
            denom = max(self.num_risk_modes - 1, 1)
            level = (self._risk_idx.to(obs.dtype) / denom).unsqueeze(-1)
        return torch.cat([obs, level], dim=-1)

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

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        # Tile a single seed so all sub-envs share the SAME task; gymnasium would
        # otherwise spread a bare int as seed+i, giving N different layouts.
        seeds = [self._seed] * self.num_envs if self._seed is not None else None
        obs, info = self.env.reset(seed=seeds)
        obs_tensor, vision_tensor = self._convert_obs(obs)
        self.episode_length_buf.zero_()
        self._goals_in_episode.zero_()
        self._reward_in_episode.zero_()
        self._cost_in_episode.zero_()
        self._risk_idx = self._draw_risk(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        obs_tensor = self._append_risk(obs_tensor)
        extras = self._build_extras(info=info, obs=obs_tensor, vision=vision_tensor)
        self._last_obs, self._last_extras = obs_tensor, extras
        return obs_tensor, extras

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, costs, terminated, truncated, info = self.env.step(actions_np)
        dones = terminated | truncated

        obs_tensor, vision_tensor = self._convert_obs(obs)
        rewards_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        costs_tensor = torch.as_tensor(costs, device=self.device, dtype=torch.float32)
        dones_tensor = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        time_outs = torch.as_tensor(truncated, device=self.device, dtype=torch.float32)

        # Tally goals reached this step. `info['goal_met']` is a per-env bool array,
        # present only on steps where at least one env reached its goal.
        goal_met = info.get("goal_met")
        if goal_met is not None:
            self._goals_in_episode += torch.as_tensor(
                np.asarray(goal_met, dtype=np.float32), device=self.device
            )

        self.episode_length_buf += 1
        self._reward_in_episode += rewards_tensor
        # costs is (num_envs,) for a single constraint, (num_envs, m) for several.
        self._cost_in_episode += costs_tensor if costs_tensor.dim() == 1 else costs_tensor.sum(dim=-1)
        extras = self._build_extras(
            info=info, costs=costs_tensor, time_outs=time_outs, obs=obs_tensor, vision=vision_tensor
        )
        if dones_tensor.any():
            done_ids = (dones_tensor > 0).nonzero(as_tuple=False).squeeze(-1)
            # Per-episode totals -> Episode/{goals_reached,reward,cost}; then reset.
            episode_log = extras.setdefault("log", {})
            goals, reward, cost = (
                self._goals_in_episode[done_ids].clone(),
                self._reward_in_episode[done_ids].clone(),
                self._cost_in_episode[done_ids].clone(),
            )
            episode_log["goals_reached"] = goals
            episode_log["reward"] = reward
            episode_log["cost"] = cost
            if self.num_risk_modes:
                # Split by the mode the finished episodes actually ran under -- read BEFORE
                # the resample below. The aggregate keys average the risk-seeking and
                # risk-averse agents together, which hides the very spread being trained for.
                modes = self._risk_idx[done_ids]
                for m in range(self.num_risk_modes):
                    sel = modes == m
                    if bool(sel.any()):
                        name = self._risk_name(m)
                        episode_log[f"reward_{name}"] = reward[sel]
                        episode_log[f"cost_{name}"] = cost[sel]
                # The env has auto-reset, so `obs_tensor` belongs to the next episode --
                # draw its risk mode before the level is attached below.
                self._risk_idx = self._draw_risk(dones_tensor > 0, self._risk_idx)
            self.episode_length_buf[done_ids] = 0
            self._goals_in_episode[done_ids] = 0.0
            self._reward_in_episode[done_ids] = 0.0
            self._cost_in_episode[done_ids] = 0.0

        obs_tensor = self._append_risk(obs_tensor)
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

    def _convert_obs(self, obs: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Convert batched numpy obs to tensors.

        Flat spaces pass through as a single float32 tensor. Dict spaces are
        split: state keys are concatenated (one vectorized concat, no per-env
        loop) and the image batch is moved to the device as uint8 — it stays
        uint8 until inside a vision encoder.
        """
        if not self._dict_obs:
            return torch.as_tensor(obs, device=self.device, dtype=torch.float32), None
        state = np.concatenate(
            [np.asarray(obs[k]).reshape(self.num_envs, -1) for k in self._state_keys], axis=1
        )
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        vision_tensor = None
        if self._vision_key in obs:
            vision_tensor = torch.from_numpy(np.ascontiguousarray(obs[self._vision_key])).to(self.device)
        return state_tensor, vision_tensor

    @staticmethod
    def _build_wrapper_chain(
        *,
        cbf_state,
        geom_margin,
        geom_margin_d_safe,
        geom_margin_min,
        cost_limit_curriculum,
        action_repeat,
        goal_pseudo_terminal,
    ) -> list:
        """Per-sub-env wrappers, composed left-to-right (outermost applied last)."""
        from functools import partial

        chain: list = []
        if cbf_state:
            from safe_rl.cbf.sg_state_wrapper import SGCBFStateWrapper

            chain.append(SGCBFStateWrapper)
        if geom_margin:
            from .geom_margin_wrapper import GeometricMarginWrapper

            chain.append(partial(GeometricMarginWrapper, d_safe=geom_margin_d_safe, margin_min=geom_margin_min))
        if cost_limit_curriculum:
            # Innermost, so it sees every simulator step's goal_met.
            chain.insert(0, partial(_require_fcsrl_wrapper("SafeCostLimitCurriculum"), **cost_limit_curriculum))
        chain.extend(_fcsrl_wrapper_chain(action_repeat, goal_pseudo_terminal))
        return chain

    def _forward_fcsrl_info(self, extras: Dict[str, Any], info: Dict[str, Any]) -> None:
        """Surface the fork wrappers' info keys; absent unless a wrapper is enabled."""
        pseudo_terminated = info.get("pseudo_terminated")
        if pseudo_terminated is not None:
            extras["pseudo_terminated"] = torch.as_tensor(
                np.asarray(pseudo_terminated, dtype=np.float32), device=self.device
            )
        sim_steps = info.get("sim_steps")
        if sim_steps is not None:
            extras["sim_steps"] = float(np.sum(np.asarray(sim_steps, dtype=np.float64)))

        # Per-sub-env wrapper, so budgets diverge; the constraint is global, hence min.
        cost_limit = info.get("cost_limit")
        if cost_limit is not None:
            limits = np.asarray(cost_limit, dtype=np.float64)
            extras["cost_limit"] = float(np.min(limits))
            extras["log"] = extras.get("log", {})
            extras["log"]["cost_limit"] = float(np.mean(limits))

    def _build_extras(
        self,
        info: Dict[str, Any],
        costs: torch.Tensor | None = None,
        time_outs: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        vision: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        extras: Dict[str, Any] = {"observations": {}}
        if vision is not None:
            # uint8 (num_envs, H, W, 3) image batch for a vision wrapper/encoder.
            extras["observations"]["vision"] = vision
            # Full ground-truth state (sensors + lidar) as privileged critic obs,
            # so reward/cost critics can train asymmetrically while the actor
            # sees pixels.
            extras["observations"]["critic"] = obs
        if costs is not None:
            extras["costs"] = costs
        if time_outs is not None:
            extras["time_outs"] = time_outs

        self._forward_fcsrl_info(extras, info)
        if "episode" in info:
            extras["episode"] = info["episode"]
        if "log" in info:
            extras["log"] = info["log"]

        # GeometricMarginWrapper reports the true (unwrapped) per-episode cost sum
        # on the terminal step; the async worker moves that step's info into
        # `final_info`. Forward it so runs with the margin cost channel still log
        # a comparable Episode/true_episode_cost safety metric.
        final_info = info.get("final_info")
        if final_info is not None:
            true_costs: list | np.ndarray = []
            if isinstance(final_info, dict):
                # Vector-env aggregation recursed into the per-env dicts: values come
                # as one array plus a `_key` presence mask.
                vals = final_info.get("true_episode_cost")
                if vals is not None:
                    mask = final_info.get("_true_episode_cost")
                    mask = np.ones(len(vals), dtype=bool) if mask is None else np.asarray(mask)
                    true_costs = np.asarray(vals)[mask]
            else:  # object array of per-env dicts (None where not done)
                true_costs = [
                    fi["true_episode_cost"]
                    for fi in final_info
                    if isinstance(fi, dict) and "true_episode_cost" in fi
                ]
            if len(true_costs):
                extras.setdefault("log", {})["true_episode_cost"] = torch.as_tensor(
                    true_costs, device=self.device, dtype=torch.float32
                )

        # Forward the true terminal observations on truncation so Q-bootstrap is
        # correct (gymnasium puts the new-episode reset obs in `obs` after auto-
        # reset). `final_observation` is an object array of shape [num_envs] with
        # the real terminal obs for terminated/truncated envs and None elsewhere.
        final_obs = info.get("final_observation")
        if final_obs is not None:
            if self._dict_obs:
                # State keys only, same concat order as _convert_obs. The image is
                # deliberately skipped: no on-policy algorithm here bootstraps from
                # final_observation (only REPPO does, which is unsupported with
                # vision), and encoding per-truncated-env frames would serialize
                # the encoder.
                stacked = np.zeros((self.num_envs, self.num_state_obs), dtype=np.float32)
                for i, fo in enumerate(final_obs):
                    if isinstance(fo, dict):
                        stacked[i] = np.concatenate(
                            [np.asarray(fo[k], dtype=np.float32).reshape(-1) for k in self._state_keys]
                        )
                extras["final_observation"] = self._append_risk(torch.as_tensor(stacked, device=self.device))
            elif self._last_obs is not None:
                # `_last_obs` carries the risk one-hot but `fo` does not, so size the buffer
                # from the raw width and re-attach the one-hot below.
                raw_dim = self._last_obs.shape[1] - self.risk_obs_dim
                stacked = np.zeros((self.num_envs, raw_dim), dtype=np.float32)
                for i, fo in enumerate(final_obs):
                    if fo is not None:
                        stacked[i] = np.asarray(fo, dtype=np.float32)
                # Called before the episode-boundary resample, so `_risk_idx` is still the
                # mode the finished episode ran under -- which is what the bootstrap needs.
                extras["final_observation"] = self._append_risk(torch.as_tensor(stacked, device=self.device))
        return extras





