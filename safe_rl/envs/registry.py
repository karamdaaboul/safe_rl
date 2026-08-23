from __future__ import annotations

from typing import Any

from .vec_env import VecEnv


def make_env(env_id: str, num_envs: int | None = None, **kwargs: Any) -> VecEnv:
    """Construct a VecEnv wrapper based on the ``env_id`` prefix.

    Safety-Gymnasium ids (``Safety*``) are built directly by the wrapper.
    mjlab ids (``Mjlab*``/``Unitree*``, and other registered mjlab tasks such
    as ``Ant-*``) expect a pre-built mjlab env passed as ``env=`` in ``kwargs``
    — this mirrors the current ``MjlabVecEnv`` contract. Any id that arrives
    with a pre-built ``env`` is therefore routed to ``MjlabVecEnv``.
    """
    if env_id.startswith("Safety"):
        from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

        if SafetyGymnasiumVecEnv is None:
            raise ImportError("safety_gymnasium is not installed.")

        # Finite-horizon observation augmentation (FH-DCMPO). Routed here rather than made a
        # constructor flag on the base class so the base env keeps exactly its current behaviour.
        # Keys present but falsy are stripped, so `horizon_feature: false` in a config means "plain
        # env" rather than "unexpected keyword argument".
        fh_keys = ("horizon_feature", "budget_feature", "budget_limit")
        if kwargs.get("horizon_feature") or kwargs.get("budget_feature"):
            from .horizon_augmented_vec_env import HorizonAugmentedVecEnv

            return HorizonAugmentedVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)
        for key in fh_keys:
            kwargs.pop(key, None)
        return SafetyGymnasiumVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)

    if env_id.startswith("Brax"):
        from .brax_vec_env import BraxVecEnv

        return BraxVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)

    if env_id.startswith("ManiSkill"):
        from .maniskill_vec_env import ManiSkillVecEnv

        return ManiSkillVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)

    # MuJoCo Playground (MJX). Must precede the mjlab branch: ``Mjx`` and ``Mjlab``
    # share a prefix only up to two characters, but the mjlab branch also catches any
    # id carrying a pre-built ``env`` kwarg.
    if env_id.startswith("Mjx"):
        from .mujoco_playground_vec_env import MujocoPlaygroundVecEnv

        return MujocoPlaygroundVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)

    # mjlab tasks: routed by prefix, or by the presence of a pre-built ``env``
    # (the MjlabVecEnv contract) so non-prefixed task ids like ``Ant-Flat`` work.
    if env_id.startswith("Mjlab") or env_id.startswith("Unitree") or "env" in kwargs:
        from .mjlab_vec_env import MjlabVecEnv

        return MjlabVecEnv(**kwargs)

    raise ValueError(f"Unknown env_id prefix: {env_id!r}")
