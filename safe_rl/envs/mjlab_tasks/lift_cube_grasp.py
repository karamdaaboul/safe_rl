"""``Mjlab-Lift-Cube-Yam-Grasp`` — lift-cube plus the grasp signals ManiSkill has.

Why this variant exists
-----------------------
On stock ``Mjlab-Lift-Cube-Yam``, REPPO (both safe_rl's port AND the authors' own
trainer, run side by side) plateaus at exactly 0.0000 success while the task's
registered PPO reaches ~0.75-0.83. See ``codex/reppo-v24-manipulation-analysis.md``.

The cause is the reward, not the algorithm. Stock lift-cube's two positive terms
are functions of POSITIONS ONLY::

    staged_position_reward = reaching * (1 + bringing)   # ee<->cube, cube<->goal
    bring_object_reward    = exp(-||cube - goal||^2 / 0.05^2)

Nothing rewards closing the gripper. Until the cube physically moves, the gripper
action dimension has identically zero gradient — which is fatal for REPPO
specifically, because its actor learns PATHWISE (maximise ``Q(s, a_theta(s))`` by
differentiating through the critic) and therefore consumes ``dQ/da`` and nothing
else. PPO survives the same plateau because a rare lucky pinch still produces a
realized advantage that its likelihood-ratio surrogate multiplies in directly.

REPPO does solve manipulation on ManiSkill (PegInsertionSide 0.98,
UnitreeG1TransportBox 0.999, PickSingleYCB 0.80 — with essentially DEFAULT
hyperparameters), and ManiSkill's ``PickSingleYCB`` dense reward is the tell::

    reaching_reward = 1 - tanh(5 * tcp_to_obj_dist)
    reward  = reaching_reward
    reward += is_grasped                      # <-- explicit +1, contact-based
    reward += place_reward * is_grasped
    reward += is_obj_placed * is_grasped

``is_grasped`` is ``agent.is_grasping(obj)``: a contact-force test on BOTH fingers
(>= 0.5 N). It pays out the instant the gripper closes, whether or not the object
has moved — so ``dQ/d(gripper)`` is non-zero from the first update.

This module reproduces that signal on mjlab.

What is added (all ADDITIVE — stock terms are untouched, so a run on this task is
still directly comparable to one on the stock task term-by-term):

* two contact sensors, left/right fingertip geoms vs the cube;
* ``grasp`` reward, weight +1.0: 1.0 when BOTH fingers contact the cube above
  ``force_threshold``. Mirrors ManiSkill's ``reward += is_grasped``;
* ``grasp_lift`` reward, weight +1.0: ``bringing * is_grasped``, mirroring
  ManiSkill's ``place_reward * is_grasped`` — this is what pays for carrying a
  HELD cube toward the goal, and is zero unless the cube is actually grasped;
* ``is_grasped`` appended to the actor and critic observation groups, so the
  policy can condition on whether it currently holds the cube (ManiSkill's state
  obs mode exposes the same information).

Magnitudes are deliberately in line with the stock terms (``lift`` in [0,2],
``lift_precise`` in [0,1], both weight 1.0), so the existing value-support
calibration still applies: mjlab dt-scales every reward term, and the ceiling
rises from 20*(2+1)=60 to 20*(2+1+1+1)=80 scaled — still inside the
``[-20, 150]`` support measured in ``config/mjlab_liftcube_reppo_v24.yaml``.

NOTE: adding reward terms CHANGES THE TASK. The PPO baseline must be re-run on
this task id for any comparison to be fair; do not compare against the stock-task
PPO numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Fingertip collision geoms, same regex the stock cfg uses for friction
# randomisation (`[lr]f_down(6|7|8|9|10|11)_collision`), split per finger.
_LEFT_FINGER_GEOMS = r"lf_down(6|7|8|9|10|11)_collision"
_RIGHT_FINGER_GEOMS = r"rf_down(6|7|8|9|10|11)_collision"

LEFT_GRASP_SENSOR = "left_finger_cube_contact"
RIGHT_GRASP_SENSOR = "right_finger_cube_contact"

TASK_ID = "Mjlab-Lift-Cube-Yam-Grasp"


def _finger_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Peak contact-force magnitude for one fingertip sensor, shape [B]."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3] -> strongest contact over slots and history.
        return torch.norm(data.force_history, dim=-1).amax(dim=-1).amax(dim=-1)
    if data.force is not None:
        # force: [B, N, 3]
        return torch.norm(data.force, dim=-1).amax(dim=-1)
    assert data.found is not None
    return data.found.any(dim=-1).float()


def is_grasped(
    env: ManagerBasedRlEnv,
    force_threshold: float = 0.5,
    left_sensor: str = LEFT_GRASP_SENSOR,
    right_sensor: str = RIGHT_GRASP_SENSOR,
) -> torch.Tensor:
    """1.0 where BOTH fingertips press the cube above ``force_threshold``, else 0.0.

    Direct analogue of ManiSkill's ``agent.is_grasping(obj)`` (both fingers,
    default 0.5 N).
    """
    left = _finger_contact_force(env, left_sensor)
    right = _finger_contact_force(env, right_sensor)
    return ((left > force_threshold) & (right > force_threshold)).float()


def grasp_reward(
    env: ManagerBasedRlEnv,
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """ManiSkill's ``reward += is_grasped``. The signal stock lift-cube lacks."""
    return is_grasped(env, force_threshold=force_threshold)


def grasp_and_bring_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "lift_height",
    object_name: str = "cube",
    std: float = 0.3,
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """ManiSkill's ``reward += place_reward * is_grasped``.

    Gaussian kernel on cube-to-goal distance, gated on actually holding the cube,
    so it cannot be farmed by nudging the cube around on the table. ``std``
    matches the stock ``bringing_std`` of 0.3.
    """
    from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

    obj = env.scene[object_name]
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, LiftingCommand)
    position_error = torch.sum(
        torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
    )
    bringing = torch.exp(-position_error / std**2)
    return bringing * is_grasped(env, force_threshold=force_threshold)


def is_grasped_obs(
    env: ManagerBasedRlEnv,
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """``is_grasped`` as a [B, 1] observation term."""
    return is_grasped(env, force_threshold=force_threshold).unsqueeze(-1)


def yam_lift_cube_grasp_env_cfg(play: bool = False):
    """Stock ``yam_lift_cube_env_cfg`` plus the grasp sensors, rewards and obs."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.manipulation.config.yam.env_cfgs import yam_lift_cube_env_cfg

    cfg = yam_lift_cube_env_cfg(play=play)

    # -- contact sensors, one per fingertip group, against the cube --------------
    grasp_sensors = tuple(
        ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="geom", pattern=pattern, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=0,
        )
        for name, pattern in (
            (LEFT_GRASP_SENSOR, _LEFT_FINGER_GEOMS),
            (RIGHT_GRASP_SENSOR, _RIGHT_FINGER_GEOMS),
        )
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + grasp_sensors

    # -- rewards (additive; stock terms untouched) -------------------------------
    cfg.rewards["grasp"] = RewardTermCfg(
        func=grasp_reward,
        weight=1.0,
        params={"force_threshold": 0.5},
    )
    cfg.rewards["grasp_lift"] = RewardTermCfg(
        func=grasp_and_bring_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "object_name": "cube",
            "std": 0.3,
            "force_threshold": 0.5,
        },
    )

    # -- observation: let the policy know whether it is holding the cube ---------
    for group in ("actor", "critic"):
        cfg.observations[group].terms["is_grasped"] = ObservationTermCfg(
            func=is_grasped_obs,
            params={"force_threshold": 0.5},
            # Binary contact flag: corrupting it with additive noise would be
            # meaningless, so it stays noise-free in both groups.
        )

    return cfg


def register() -> bool:
    """Register ``Mjlab-Lift-Cube-Yam-Grasp``. Idempotent; returns True if added."""
    from mjlab.tasks.manipulation.config.yam.rl_cfg import yam_lift_cube_ppo_runner_cfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if TASK_ID in list_tasks():
        return False

    rl_cfg = yam_lift_cube_ppo_runner_cfg()
    rl_cfg.experiment_name = "yam_lift_cube_grasp"
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=yam_lift_cube_grasp_env_cfg(),
        play_env_cfg=yam_lift_cube_grasp_env_cfg(play=True),
        rl_cfg=rl_cfg,
    )
    return True
