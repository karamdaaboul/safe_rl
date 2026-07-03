from __future__ import annotations

import gymnasium


class HiddenGoalWrapper(gymnasium.Wrapper):
    """Turn a Safety-Gymnasium *Goal* task into a hidden-goal meta-RL task.

    A standard ``SafetyXGoal`` task is *goal-conditioned* RL, not meta-RL: the
    goal direction is exposed through the ``goal_lidar`` sensor, so a single
    reactive policy solves any layout zero-shot and there is no latent task to
    infer. This wrapper makes the goal a hidden variable so adaptation actually
    matters:

      1. **Hides the goal lidar at the source.** Sets the goal obstacle's
         ``is_lidar_observed = False`` and rebuilds the observation space, which
         removes ``goal_lidar`` from both the observation *and* the rendered lidar
         halo (the renderer gates the halo on the same flag, see
         ``underlying.py``). So the agent can no longer sense the goal and must
         explore to find it; the distance-shaped reward still fires, so the goal
         is a latent inferable only from reward — the defining property of meta-RL.
      2. **Controls goal respawning via ``continue_goal``.** With the default
         ``continue_goal=False`` the goal is placed once at episode reset and
         reaching it *terminates* the episode (one hidden goal per episode — the
         meta-RL setting). With ``continue_goal=True`` reaching a goal instead
         draws a *new* hidden goal and the episode continues (``builder.py`` step
         handling), so a single episode can chain many goals — used to measure
         how many hidden goals a reward-only policy can reach per episode.

    The goal still exists in the simulation (it drives the reward and the agent's
    relative position, and renders as the green goal area) — only its *lidar* is
    removed, which is what the agent would otherwise use to sense it.

    ``fix_task``: a "task" is one fixed hidden goal that the inner loop adapts to
    over several episodes. The async vector env auto-resets a sub-env *without* a
    seed, which would draw a new random goal each episode (a memoryless POMDP, not
    a task you can adapt to). With ``fix_task`` we remember the last explicit task
    seed and reuse it on every seedless (auto-)reset, so the goal stays constant
    until the runner selects a new task via a seeded reset.
    """

    def __init__(self, env: gymnasium.Env, fix_task: bool = True, continue_goal: bool = False) -> None:
        super().__init__(env)
        task = env.unwrapped.task

        # Hide the goal from the lidar (removed from obs AND render), then rebuild
        # the observation space so it no longer contains goal_lidar.
        task.goal.is_lidar_observed = False
        task.build_observation_space()
        # continue_goal=False -> one goal per episode, reaching it terminates.
        # continue_goal=True  -> reaching a goal respawns a new one, episode runs on.
        self.continue_goal = continue_goal
        task.mechanism_conf.continue_goal = continue_goal

        # Adopt the rebuilt (goal-lidar-free) observation space.
        self.observation_space = env.observation_space

        self.fix_task = fix_task
        self._task_seed: int | None = None

    def _reassert(self) -> None:
        # Persist our task settings in case the task rebuilds on reset; harmless
        # if already set.
        task = self.env.unwrapped.task
        task.goal.is_lidar_observed = False
        task.mechanism_conf.continue_goal = self.continue_goal

    def reset(self, *, seed: int | None = None, options=None):
        self._reassert()
        if seed is not None:
            self._task_seed = seed  # a new task was selected
        eff_seed = self._task_seed if (self.fix_task and seed is None) else seed
        return self.env.reset(seed=eff_seed, options=options)

    def step(self, action):
        # Safety-Gymnasium returns a 6-tuple (obs, reward, COST, terminated,
        # truncated, info) — pass it through unchanged (cost channel preserved).
        return self.env.step(action)
