"""safe_rl-owned mjlab task variants.

Importing this package registers them into mjlab's task registry, so they become
available to ``mjlab.tasks.registry.list_tasks()`` and therefore to
``scripts/train/unitree_mjlab.py --env_id ...`` exactly like a stock task.

Registration is best-effort: mjlab is an optional dependency of this repo, and a
failure here must not take down a run that does not use these tasks.
"""

from __future__ import annotations

import sys


def register_all() -> list[str]:
    """Register every safe_rl mjlab task variant. Returns the ids now available."""
    registered: list[str] = []
    try:
        from safe_rl.envs.mjlab_tasks import lift_cube_grasp

        lift_cube_grasp.register()
        registered.append(lift_cube_grasp.TASK_ID)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] safe_rl mjlab task registration failed ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
    return registered


__all__ = ["register_all"]
