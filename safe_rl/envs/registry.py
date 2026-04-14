from __future__ import annotations

from typing import Any

from .vec_env import VecEnv


def make_env(env_id: str, num_envs: int | None = None, **kwargs: Any) -> VecEnv:
    """Construct a VecEnv wrapper based on the ``env_id`` prefix.

    Safety-Gymnasium ids (``Safety*``) are built directly by the wrapper.
    mjlab/Unitree ids expect a pre-built mjlab env to be passed as ``env=``
    in ``kwargs`` — this mirrors the current ``MjlabVecEnv`` contract.
    """
    if env_id.startswith("Safety"):
        from .safety_gymnasium_vec_env import SafetyGymnasiumVecEnv

        if SafetyGymnasiumVecEnv is None:
            raise ImportError("safety_gymnasium is not installed.")
        return SafetyGymnasiumVecEnv(env_id=env_id, num_envs=num_envs, **kwargs)

    if env_id.startswith("Mjlab") or env_id.startswith("Unitree"):
        from .mjlab_vec_env import MjlabVecEnv

        return MjlabVecEnv(**kwargs)

    raise ValueError(f"Unknown env_id prefix: {env_id!r}")
