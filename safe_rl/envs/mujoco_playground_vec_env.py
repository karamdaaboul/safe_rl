"""MuJoCo Playground (MJX) VecEnv wrapper — the REPPO paper's headline benchmark suite.

Env ids use the ``Mjx`` prefix, e.g. ``MjxCheetahRun`` -> the playground task
``CheetahRun``. ``Mjx`` matches the reference's own naming for the suite
(``config/env/mjx_dmc.yaml``, ``type: mjx``) and does not collide with ``Mjlab``.

Why this is thin: MuJoCo Playground already ships a torch bridge,
``mujoco_playground.wrapper_torch.RSLRLBraxWrapper``, which subclasses rsl_rl's own
``VecEnv``, performs the jax->torch dlpack conversion, jits reset/step, and returns
``(obs, reward, done, {"time_outs": ..., "observations": {...}, "log": {...}})`` — very
nearly our contract already. It is also the exact wrapper the reference's torch path
uses (``reppo_original/src/env_utils/torch_wrappers/mujoco_playground_env.py``), so
composing it rather than re-porting it buys env-side parity for free.

What this adds on top, and why each piece is load-bearing:

* **A non-resetting ``get_observations()``.** ``RSLRLBraxWrapper.get_observations()``
  calls ``reset()``. Our runner calls ``get_observations`` at learn start *and* after
  every periodic eval, so delegating would silently reset the training env ~20 times
  per run and quietly discard partial episodes.
* **Stripping ``observations["critic"]`` when it is ``None``.** The wrapper always emits
  the key and sets it to ``None`` for symmetric-observation tasks (all 23 DMC tasks).
  ``OnPolicyRunner`` treats the key's *presence* as "privileged observations exist" and
  then reads ``.shape[1]`` off it.
* **``episode_length_buf``**, which the wrapper does not maintain and the runner needs.
* **Cloning the tensors out of jax-owned memory.** ``from_dlpack`` aliases buffers that
  jax is free to donate or free on the next ``step``; the runner and REPPO write into
  observation buffers in place (``RolloutStorage``, ``EmpiricalNormalization``). This is
  the same class of bug that already required ``.clone()`` on the ManiSkill path.

Deliberately NOT ported, for parity:

* ``RandomizeInitialWrapper`` (desynchronises episode phase across envs). The reference
  torch path uses it; the **JAX** path that produced the published DMC curves does not.
  Our runner has the same capability natively via ``init_at_random_ep_len``; leave it off.
* ``final_observation`` forwarding. ``BraxAutoResetWrapper`` does expose the true
  terminal observation as ``info['raw_obs']``, and the reference does not consume it.
  Fixing that on our side only would confound the comparison on a 1-in-1000 transition,
  so it stays behind a default-off flag.
"""

from __future__ import annotations

from typing import Any

import torch

from .vec_env import VecEnv


class MujocoPlaygroundVecEnv(VecEnv):
    """MuJoCo Playground MJX envs behind the safe_rl VecEnv contract."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        device: str = "cuda:0",
        seed: int | None = 0,
        episode_length: int | None = None,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        num_eval_envs: int | None = None,
        forward_final_observation: bool = False,
        **kwargs: Any,
    ) -> None:
        ignored = {k: v for k, v in kwargs.items() if v not in (None, False)}
        if ignored:
            print(f"MujocoPlaygroundVecEnv: ignoring non-applicable kwargs: {sorted(ignored)}")

        from mujoco_playground import registry, wrapper_torch

        assert env_id.startswith("Mjx"), env_id
        self.env_id = env_id
        task = env_id[len("Mjx"):]

        env_cfg = registry.get_default_config(task)
        base = registry.load(task, config=env_cfg)
        self.max_episode_length = int(episode_length or env_cfg.episode_length)

        self._inner = wrapper_torch.RSLRLBraxWrapper(
            base,
            num_actors=int(num_envs),
            seed=0 if seed is None else int(seed),
            episode_length=self.max_episode_length,
            action_repeat=int(action_repeat),
        )

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.reward_scale = float(reward_scale)
        self.num_actions = int(self._inner.num_actions)
        self.num_obs = int(self._inner.num_obs)
        self.asymmetric_obs = bool(self._inner.asymmetric_obs)
        self.num_privileged_obs = (
            int(self._inner.num_privileged_obs) if self._inner.num_privileged_obs else None
        )
        self.forward_final_observation = bool(forward_final_observation)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cost_limits = None
        self.cfg = {
            "env_id": env_id,
            "task": task,
            "episode_length": self.max_episode_length,
            "action_repeat": int(action_repeat),
            "asymmetric_obs": self.asymmetric_obs,
        }

        self._last_obs: torch.Tensor | None = None
        self._last_extras: dict | None = None

        # Eval twin, built lazily. Without it the periodic deterministic eval runs on the
        # training env, resetting it at every eval point and spending eval env-steps that
        # are not counted against the training budget.
        self._num_eval_envs = int(num_eval_envs) if num_eval_envs else None
        self._eval_env: MujocoPlaygroundVecEnv | None = None
        self._eval_twin_kwargs = {
            "env_id": env_id,
            "device": device,
            # A different seed so the eval initial states are not the training ones.
            "seed": (0 if seed is None else int(seed)) + 10_000,
            "episode_length": self.max_episode_length,
            "action_repeat": int(action_repeat),
            "reward_scale": reward_scale,
        }

    # ------------------------------------------------------------------

    @property
    def eval_env(self) -> MujocoPlaygroundVecEnv | None:
        """A second env instance for evaluation, or None if ``num_eval_envs`` is unset."""
        if self._num_eval_envs is None:
            return None
        if self._eval_env is None:
            self._eval_env = MujocoPlaygroundVecEnv(
                num_envs=self._num_eval_envs,
                num_eval_envs=None,  # the twin never builds a twin of its own
                **self._eval_twin_kwargs,
            )
        return self._eval_env

    # ------------------------------------------------------------------

    def _clean_extras(self, info: dict) -> dict:
        """Map the wrapper's info onto our contract.

        Drops observation entries that are ``None`` — the wrapper always emits a
        ``critic`` key and sets it to ``None`` for symmetric-obs tasks, and the runner
        keys off presence, not value. Also drops ``raw``, which carries the true terminal
        observation we deliberately do not consume (see the module docstring).
        """
        raw = (info.get("observations") or {}).get("raw")
        observations = {}
        for key, value in (info.get("observations") or {}).items():
            # ``raw`` is a dict, never a tensor; it is surfaced separately as
            # ``final_observation`` below rather than left in ``observations``, where the
            # runner would key off its presence and try to read ``.shape`` from a dict.
            if key == "raw":
                continue
            if isinstance(value, torch.Tensor):
                observations[key] = value.clone().to(self.device)

        extras: dict[str, Any] = {"observations": observations}
        time_outs = info.get("time_outs")
        if time_outs is not None:
            extras["time_outs"] = time_outs.clone().to(self.device).view(-1)

        # The TRUE terminal observation on a timeout. ``BraxAutoResetWrapper`` records it
        # as ``info['raw_obs']`` before swapping in the reset state, and
        # ``RSLRLBraxWrapper`` forwards it as ``observations['raw']``. Without this,
        # Q-bootstrap at a truncation reads a freshly-reset state — which REPPO itself
        # warns about — and on a sparse task where the bootstrap is the only signal, those
        # corrupted targets are most of what the critic learns from.
        if self.forward_final_observation and isinstance(raw, dict):
            actor_obs = raw.get("obs")
            if isinstance(actor_obs, torch.Tensor):
                final: dict[str, torch.Tensor] = {"actor": actor_obs.clone().to(self.device)}
                critic_obs = raw.get("critic_obs")
                if isinstance(critic_obs, torch.Tensor):
                    final["critic"] = critic_obs.clone().to(self.device)
                extras["final_observation"] = final
        if info.get("log"):
            extras["log"] = dict(info["log"])
        return extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs = self._inner.reset()
        obs = obs.clone().to(self.device)
        self.episode_length_buf.zero_()
        self._last_obs = obs
        self._last_extras = {"observations": {}}
        return obs, self._last_extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Current observations WITHOUT stepping or resetting.

        Deliberately does not delegate to ``RSLRLBraxWrapper.get_observations()``, which
        resets. See the module docstring.
        """
        if self._last_obs is None:
            return self.reset()
        return self._last_obs, (self._last_extras or {"observations": {}})

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._last_obs is None:
            self.reset()

        obs, rewards, dones, info = self._inner.step(actions)

        # Copy out of jax-owned memory before anything can write into these.
        obs = obs.clone().to(self.device)
        rewards = rewards.clone().to(self.device).view(-1).float() * self.reward_scale
        dones = dones.clone().to(self.device).view(-1) > 0.5

        extras = self._clean_extras(info)
        time_outs = extras.get("time_outs")

        self.episode_length_buf += 1
        finished = dones if time_outs is None else torch.logical_or(dones, time_outs > 0.5)
        self.episode_length_buf[finished] = 0

        self._last_obs = obs
        self._last_extras = extras
        return obs, rewards, dones, extras

    def close(self) -> None:
        self._eval_env = None
