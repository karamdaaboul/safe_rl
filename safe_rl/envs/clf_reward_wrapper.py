"""CLF-RL reward-shaping wrapper for mjlab locomotion environments.

Wraps a :class:`~safe_rl.envs.vec_env.VecEnv` (typically
:class:`~safe_rl.envs.mjlab_vec_env.MjlabVecEnv`) and adds a Control-Lyapunov-
Function shaped reward on top of the environment's own reward, following Li,
Olkin et al., "CLF-RL: Control Lyapunov Function Guided Reinforcement Learning"
(RA-L 2025), Sec. III-B.

The wrapper is transparent: it delegates the whole VecEnv API to the inner env
and only overrides ``step()`` to (1) read the robot state, (2) query an analytic
H-LIP reference, (3) form the tracking error ``eta = [yd - y; yd_dot - y_dot]``,
(4) evaluate the CLF ``V = eta^T P eta`` plus its finite-difference decrease
condition, and (5) add the resulting reward terms. Each component is logged
under ``extras["log"]["clf/..."]`` so it shows up in TensorBoard / W&B.

No edits to the external mjlab package are required: all robot state is read
through ``env.unwrapped.scene["robot"].data`` and ``env.unwrapped.command_manager``.
"""

from __future__ import annotations

from typing import Any

import torch

from .reference.hlip import DEFAULT_OUTPUTS, HLIPReference
from .vec_env import VecEnv
from ..utils.clf import build_clf_P, clf_sigma


class CLFRewardWrapper(VecEnv):
    """Add a CLF-RL shaped reward to an mjlab locomotion VecEnv.

    Args:
        env: The wrapped VecEnv (must expose ``.unwrapped`` with an mjlab
            ``ManagerBasedRlEnv`` carrying ``scene`` and ``command_manager``).
        clf_cfg: Configuration dict. Recognized keys (all optional):

            - ``outputs``: ordered list of reference output kinds (see
              :mod:`safe_rl.envs.reference.hlip`). Default velocity outputs.
            - ``w_v``, ``w_vdot``: CLF tracking / decay reward weights.
            - ``lam``: desired CLF decrease rate.
            - ``eta_max``, ``eta_dot_max``: normalization bounds for sigma.
            - ``q_pos``, ``q_vel``, ``r``: CARE weights for ``P``.
            - ``base_scale``: multiplier on the wrapped env's reward.
            - ``step_period``, ``com_height``, ``gravity``, ``swing_height``:
              H-LIP reference parameters.
            - ``command_name``: command-manager term name (default ``"twist"``).
            - ``robot_name``: scene entity name (default ``"robot"``).
            - ``left_foot_pattern`` / ``right_foot_pattern``: body name regexes,
              required only when ``swing_foot_z`` is among ``outputs``.
    """

    def __init__(self, env: VecEnv, clf_cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(clf_cfg or {})
        self.env = env
        self.device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.cfg = env.cfg
        self.step_dt = float(getattr(env, "step_dt", cfg.get("step_dt", 0.02)))

        self.w_v = float(cfg.get("w_v", 10.0))
        self.w_vdot = float(cfg.get("w_vdot", 2.0))
        self.lam = float(cfg.get("lam", 1.0))
        self.base_scale = float(cfg.get("base_scale", 1.0))
        self.command_name = str(cfg.get("command_name", "twist"))
        self.robot_name = str(cfg.get("robot_name", "robot"))

        outputs = tuple(cfg.get("outputs", DEFAULT_OUTPUTS))
        self.reference = HLIPReference(
            num_envs=self.num_envs,
            device=self.device,
            outputs=outputs,
            step_period=float(cfg.get("step_period", 0.4)),
            com_height=float(cfg.get("com_height", 0.74)),
            gravity=float(cfg.get("gravity", 9.81)),
            swing_height=float(cfg.get("swing_height", 0.08)),
        )
        self.outputs = self.reference.outputs
        self.n_out = self.reference.n_out

        self.P = build_clf_P(
            self.n_out,
            q_pos=float(cfg.get("q_pos", 1.0)),
            q_vel=float(cfg.get("q_vel", 1.0)),
            r=float(cfg.get("r", 1.0)),
            device=self.device,
        )
        eta_max = float(cfg.get("eta_max", 1.0))
        eta_dot_max = cfg.get("eta_dot_max", None)
        eta_dot_max = float(eta_dot_max) if eta_dot_max is not None else None
        self.sigma_v, self.sigma_vdot = clf_sigma(self.P, eta_max, eta_dot_max, self.lam)

        # Per-env memory for finite differences (actual y_dot and CLF decay).
        self._prev_y = torch.zeros(self.num_envs, self.n_out, device=self.device)
        self._prev_V = torch.zeros(self.num_envs, device=self.device)
        self._initialized = False

        # Resolve swing-foot body indices lazily if needed.
        self._foot_ids: tuple[int, int] | None = None
        self._left_pat = cfg.get("left_foot_pattern")
        self._right_pat = cfg.get("right_foot_pattern")

    # -- VecEnv API: delegate everything except step() --------------------------------

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.env.get_observations()

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = self.env.reset()
        self._prev_y.zero_()
        self._prev_V.zero_()
        self._initialized = False
        return obs, extras

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        # Fall back to the wrapped env for anything not defined here. __getattr__
        # is only called when normal attribute lookup fails, so it won't shadow
        # the attributes set in __init__.
        return getattr(self.__dict__["env"], name)

    # -- Core: shaped step -----------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, base_reward, dones, extras = self.env.step(actions)
        dones = dones.to(self.device)
        done_mask = dones > 0

        command = self._get_command()
        phase = self._get_phase()
        yd, yd_dot = self.reference.compute(command, phase)

        y = self._actual_outputs()
        # Actual output rate via finite difference; invalid right after a reset.
        y_dot = (y - self._prev_y) / self.step_dt
        if not self._initialized:
            y_dot = torch.zeros_like(y_dot)
        y_dot = torch.where(done_mask.unsqueeze(1), torch.zeros_like(y_dot), y_dot)

        eta = torch.cat([yd - y, yd_dot - y_dot], dim=1)  # [pos block; vel block]
        # V = eta^T P eta, batched.
        V = torch.einsum("bi,ij,bj->b", eta, self.P, eta)

        # CLF tracking reward.
        r_v = self.w_v * torch.exp(-V / self.sigma_v)

        # CLF decrease condition V_dot <= -lam V, via finite difference.
        V_dot = (V - self._prev_V) / self.step_dt
        decay_violation = V_dot + self.lam * V
        decay_norm = torch.clamp(decay_violation / self.sigma_vdot, 0.0, 1.0)
        r_vdot = -self.w_vdot * decay_norm
        if not self._initialized:
            r_vdot = torch.zeros_like(r_vdot)
        r_vdot = torch.where(done_mask, torch.zeros_like(r_vdot), r_vdot)

        shaped = self.base_scale * base_reward.to(self.device) + r_v + r_vdot

        # Update memory. prev_y carries the current outputs forward (for done envs
        # this keeps the next y_dot near zero rather than crossing the episode
        # boundary); prev_V is zeroed for done envs so the next decay term is masked.
        self._prev_y = y.detach().clone()
        self._prev_V = torch.where(done_mask, torch.zeros_like(V), V.detach())
        self._initialized = True

        log = extras.setdefault("log", {})
        log["clf/V"] = V.mean()
        log["clf/V_dot"] = V_dot.mean()
        log["clf/reward_tracking"] = r_v.mean()
        log["clf/reward_decay"] = r_vdot.mean()
        log["clf/reward_base"] = base_reward.mean()

        return obs, shaped, dones, extras

    # -- State extraction ------------------------------------------------------------

    def _robot(self) -> Any:
        return self.unwrapped.scene[self.robot_name]

    def _get_command(self) -> torch.Tensor:
        cmd = self.unwrapped.command_manager.get_command(self.command_name)
        if cmd is None:
            raise RuntimeError(f"Command '{self.command_name}' not found on the env's command manager.")
        return cmd.to(self.device)

    def _episode_time(self) -> torch.Tensor:
        # Read live from the env each step in case the buffer is reassigned.
        buf = getattr(self.unwrapped, "episode_length_buf", self.episode_length_buf)
        return buf.to(self.device, dtype=torch.float32) * self.step_dt

    def _get_phase(self) -> torch.Tensor:
        within_step = torch.remainder(self._episode_time(), self.reference.step_period)
        return within_step / self.reference.step_period

    def _swing_foot_z(self) -> torch.Tensor:
        if self._foot_ids is None:
            if not (self._left_pat and self._right_pat):
                raise ValueError("swing_foot_z output requires left_foot_pattern and right_foot_pattern in clf cfg.")
            robot = self._robot()
            left_ids, _ = robot.find_bodies(self._left_pat)
            right_ids, _ = robot.find_bodies(self._right_pat)
            self._foot_ids = (int(left_ids[0]), int(right_ids[0]))
        left_id, right_id = self._foot_ids
        foot_z = self._robot().data.body_link_pos_w[:, [left_id, right_id], 2]  # (E, 2)
        # Swing foot alternates by step index parity within the full gait cycle.
        t = self._episode_time()
        step_index = torch.floor(torch.remainder(t, 2.0 * self.reference.step_period) / self.reference.step_period)
        swing_is_right = step_index > 0.5
        return torch.where(swing_is_right, foot_z[:, 1], foot_z[:, 0])

    def _actual_outputs(self) -> torch.Tensor:
        data = self._robot().data
        lin = data.root_link_lin_vel_b  # (E, 3), body frame, matches "twist" command
        ang = data.root_link_ang_vel_b  # (E, 3)
        cols: list[torch.Tensor] = []
        for kind in self.outputs:
            if kind == "lin_vel_x":
                cols.append(lin[:, 0])
            elif kind == "lin_vel_y":
                cols.append(lin[:, 1])
            elif kind == "lin_vel_z":
                cols.append(lin[:, 2])
            elif kind == "ang_vel_z":
                cols.append(ang[:, 2])
            elif kind == "swing_foot_z":
                cols.append(self._swing_foot_z())
            else:  # pragma: no cover - guarded by HLIPReference
                raise ValueError(f"Unsupported output kind for extraction: {kind}")
        return torch.stack(cols, dim=1).to(self.device)
