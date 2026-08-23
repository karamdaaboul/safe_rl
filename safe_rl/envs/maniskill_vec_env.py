"""ManiSkill3 VecEnv wrapper: GPU-parallel manipulation tasks behind the safe_rl VecEnv API.

Env ids use the ``ManiSkill`` prefix, e.g. ``ManiSkillPickCube-v1`` -> the
ManiSkill task ``PickCube-v1``.

Why this exists: TruDi (arXiv 2606.15260) reports its diffusion policy solving
ManiSkill manipulation tasks that *require an actual grasp* (PickCube-v1,
StackCube-v1, PullCubeTool-v1; Fig. 4/11). Our REPPO-DIME port fails to grasp on
mjlab `Lift-Cube-Yam-Grasp` while PPO solves it, and the parity test only proves
the DIME **actor** is bit-identical to the reference — it does not cover the
REPPO collection/critic path in combination with a diffusion actor. Running the
paper's own grasping benchmark is therefore the control that separates
"the mjlab env defeats REPPO's critic bootstrap" from "our integration has a
defect". See codex/reppo-dime-integration.md.

Truncation semantics (the subtle bit): ManiSkill bootstraps on ``terminated``
where mujoco-playground bootstraps on ``truncated``. The reference's own wrapper
unifies this by folding ``terminated`` into ``truncated`` and reporting
``done=False`` when ``partial_reset`` is on, so success-terminations keep their
bootstrap instead of being treated as absorbing. We reproduce that exactly —
getting it wrong silently changes the return target on every successful episode.
"""

from __future__ import annotations

from typing import Any

import torch

from .vec_env import VecEnv


class ManiSkillVecEnv(VecEnv):
    """ManiSkill3 GPU envs exposed through the safe_rl VecEnv contract."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        device: str = "cuda:0",
        seed: int | None = 0,
        obs_mode: str = "state",
        control_mode: str | None = None,
        sim_backend: str = "physx_cuda",
        max_episode_steps: int | None = None,
        partial_reset: bool = True,
        reward_mode: str = "normalized_dense",
        reward_scale: float = 1.0,
        render_mode: str = "rgb_array",
        reconfiguration_freq: int | None = None,
        num_eval_envs: int | None = None,
        eval_reconfiguration_freq: int = 1,
        **kwargs: Any,
    ) -> None:
        ignored = {k: v for k, v in kwargs.items() if v not in (None, False)}
        if ignored:
            print(f"ManiSkillVecEnv: ignoring non-applicable kwargs: {sorted(ignored)}")

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers the tasks)
        from mani_skill.utils import gym_utils
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        assert env_id.startswith("ManiSkill"), env_id
        self.env_id = env_id
        task_id = env_id[len("ManiSkill"):]

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.reward_scale = float(reward_scale)
        self.partial_reset = bool(partial_reset)

        env_kwargs: dict[str, Any] = {
            "num_envs": self.num_envs,
            "obs_mode": obs_mode,
            "reward_mode": reward_mode,
            "sim_backend": sim_backend,
            "render_mode": render_mode,
        }
        if control_mode is not None:
            env_kwargs["control_mode"] = control_mode
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = max_episode_steps

        # reconfiguration_freq=None on the TRAIN env, matching the reference
        # (trudi/src/torchrl/envs.py); reconfiguring every reset is expensive and is
        # only used for their eval envs — see the `eval_env` property below.
        base = gym.make(task_id, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
        if isinstance(base.action_space, gym.spaces.Dict):
            base = FlattenActionSpaceWrapper(base)

        # `ignore_terminations = NOT partial_reset` — the reference's polarity.
        # With partial_reset=True the env DOES terminate on success (and auto-resets);
        # our step() then re-labels that termination as a truncation so the value
        # bootstrap survives. Setting this to `partial_reset` instead (as an earlier
        # version did) suppresses success terminations entirely: episodes run the full
        # horizon, the agent never sees an episode boundary at success, and PickCube
        # never learns. That bug produced success_rate ~0.005 over a full 50M-step run.
        self._env = ManiSkillVectorEnv(
            base,
            self.num_envs,
            auto_reset=True,
            ignore_terminations=not self.partial_reset,
            record_metrics=True,  # populates info["episode"]/success metrics
        )

        derived = gym_utils.find_max_episode_steps_value(base)
        self.max_episode_length = int(max_episode_steps or derived or 50)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_actions = int(self._env.single_action_space.shape[0])
        self.num_obs = int(self._env.single_observation_space.shape[0])
        self.cfg = {"env_id": env_id, "obs_mode": obs_mode, "reward_mode": reward_mode}
        # The reference DERIVES the discount from the episode length rather than
        # using a locomotion-style 0.99:  gamma = 1 - 10/max_episode_steps
        # (trudi/src/torchrl/envs.py). PickCube's 50-step episodes give gamma = 0.8;
        # 0.99 would give a ~100-step effective horizon on a 50-step task. Exposed
        # here so configs//callers can assert against it.
        self.suggested_gamma = 1.0 - 10.0 / max(self.max_episode_length, 1)
        self.cost_limits = None
        self._seed = seed
        self._last_obs: torch.Tensor | None = None
        # Rolling success, so the runner's log shows the metric that actually matters
        # on these tasks (reward is dense/shaped; success is the task).
        self._success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Everything needed to build the eval twin lazily (see `eval_env`).
        self._num_eval_envs = int(num_eval_envs) if num_eval_envs else None
        self._eval_reconfiguration_freq = int(eval_reconfiguration_freq)
        self._eval_env: ManiSkillVecEnv | None = None
        self._eval_twin_kwargs = {
            "env_id": env_id,
            "device": device,
            "seed": seed,
            "obs_mode": obs_mode,
            "control_mode": control_mode,
            "sim_backend": sim_backend,
            "max_episode_steps": max_episode_steps,
            "reward_mode": reward_mode,
            "reward_scale": reward_scale,
            "render_mode": render_mode,
        }

    # ------------------------------------------------------------------

    @property
    def eval_env(self) -> ManiSkillVecEnv | None:
        """A second env instance matching the reference's EVALUATION env, or None.

        Two differences from the training env, both taken from the reference
        (`trudi/src/torchrl/envs.py`), and both of which change the reported number:

        * ``reconfiguration_freq=1`` — assets and layout are resampled on every reset.
          The training env uses ``None`` (reconfiguring is expensive and pointless
          when the same scene is reused for millions of steps). Evaluating on the
          training env therefore measures the policy on the *one* scene instantiation
          it trained against, which is exactly the generalization that
          ``PickSingleYCB-v1`` and ``PokeCube-v1`` exist to test. Scoring those on a
          non-reconfiguring env inflates success against a reference that reconfigured
          — i.e. it manufactures a "we beat the paper" result.
        * ``partial_reset=False`` -> ``ignore_terminations=True`` — episodes run the
          full horizon instead of ending at the first success, so ``success_once``
          accumulates over the whole episode, which is the quantity their CSVs report.

        Returns None unless ``num_eval_envs`` was set, so existing runs are unaffected
        and no second simulator is ever allocated for them. Built once, on first use.
        """
        if self._num_eval_envs is None:
            return None
        if self._eval_env is None:
            self._eval_env = ManiSkillVecEnv(
                num_envs=self._num_eval_envs,
                partial_reset=False,
                reconfiguration_freq=self._eval_reconfiguration_freq,
                num_eval_envs=None,  # the twin never builds a twin of its own
                **self._eval_twin_kwargs,
            )
        return self._eval_env

    # ------------------------------------------------------------------

    def _to_torch(self, x) -> torch.Tensor:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        return t.to(self.device).float()

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None  # only seed once; later resets continue the stream
        self.episode_length_buf.zero_()
        obs = self._to_torch(obs)
        self._last_obs = obs
        return obs, {"observations": {}}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if self._last_obs is None:
            return self.reset()
        return self._last_obs, {"observations": {}}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._last_obs is None:
            self.reset()
        actions = actions.detach().clamp(-1.0, 1.0).contiguous()
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        obs = self._to_torch(obs)
        rewards = self._to_torch(rewards).view(-1) * self.reward_scale
        terminated = self._to_torch(terminated).view(-1) > 0.5
        truncated = self._to_torch(truncated).view(-1) > 0.5

        if self.partial_reset:
            # Reference semantics: ManiSkill bootstraps on `terminated`, so fold it
            # into `truncated` and report no hard done -- otherwise every successful
            # episode would be treated as absorbing and lose its bootstrap.
            truncated = torch.logical_or(terminated, truncated)
            dones = torch.zeros_like(truncated)
        else:
            dones = torch.logical_or(terminated, truncated)
            truncated = torch.zeros_like(dones)

        self.episode_length_buf += 1
        self.episode_length_buf[truncated | dones] = 0

        extras: dict = {"observations": {}, "time_outs": truncated}

        # CRITICAL for short-horizon tasks: on truncation the env has already
        # auto-reset, so `obs` is the FIRST observation of a NEW episode. Bootstrapping
        # Q(s',a') from it is wrong. ManiSkill returns the true terminal observation in
        # info["final_observation"] (the reference gates on `has_final_obs: true` and
        # uses it in collect_fn). Surface it under the key REPPO.process_env_step looks
        # for; it substitutes it only for the truncated envs.
        # Omitting this corrupts ~1 transition in 50 on PickCube (50-step episodes) vs
        # ~1 in 1000 on mjlab Ant/Humanoid -- which is why mjlab runs were unaffected
        # and PickCube flatlined.
        final_obs = info.get("final_observation")
        if final_obs is None:
            final_obs = info.get("final_obs")
        if final_obs is not None:
            if isinstance(final_obs, dict):
                final_obs = final_obs.get("state", next(iter(final_obs.values())))
            extras["final_observation"] = self._to_torch(final_obs)

        # Success MUST be read from info["final_info"], never from info["success"].
        #
        # ManiSkillVectorEnv.step replaces the ENTIRE top-level info dict with the
        # POST-RESET info as soon as *any* env is done
        # (mani_skill/vector/wrappers/gymnasium.py):
        #     final_info = torch_clone_dict(infos)
        #     obs, infos = self.reset(options=dict(env_idx=env_idx))
        #     infos["final_info"] = final_info
        # so info["success"] describes freshly-reset states. On PickCube the only
        # early termination IS success, which means the env that just succeeded is
        # exactly the env that was just reset: success was structurally unobservable
        # through info["success"] and both the train success_rate and eval/success
        # read ~0.002 forever -- while the policy was in fact solving the task
        # (episode length had already fallen 50 -> 25, matching the reference's
        # 50 -> 15 signature). The reference reads
        # final_info["episode"]["success_once"]; so do we.
        final_info = info.get("final_info")
        final_mask = info.get("_final_info")
        if isinstance(final_info, dict) and final_mask is not None:
            episode = final_info.get("episode") or {}
            succ = episode.get("success_once")
            if succ is None:  # record_metrics off -> fall back to terminal success
                succ = final_info.get("success")
            if succ is not None:
                succ = self._to_torch(succ).view(-1).float()
                fmask = self._to_torch(final_mask).view(-1).bool()
                self._success = torch.where(fmask, succ, self._success)
                # True only for envs that ENDED this step having succeeded at some
                # point during the episode (ManiSkill's `success_once`). The
                # evaluator ORs this across the episode, which stays correct with a
                # boundary-only signal.
                extras["success_flag"] = fmask & succ.bool()
        extras["log"] = {"success_rate": self._success.mean().item()}

        self._last_obs = obs
        return obs, rewards, dones, extras

    def close(self) -> None:
        for env in (self._eval_env, self):
            if env is None:
                continue
            try:
                env._env.close()
            except Exception:  # noqa: BLE001
                pass
