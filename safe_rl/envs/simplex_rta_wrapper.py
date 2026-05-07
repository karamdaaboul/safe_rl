import gymnasium as gym


class SimplexRTAWrapper(gym.Wrapper):
    def __init__(self, env, backup_controller, pitch_limit=0.5):
        super().__init__(env)
        self.backup_controller = backup_controller
        self.pitch_limit = pitch_limit

        self.activation_threshold = self.pitch_limit - 0.1

        self.rta_active = False
        self.recovery_steps = 0
        self.min_recovery_duration = 10

    def is_state_unsafe(self, obs):
        """In HalfCheetah-v4, obs[1] is the pitch angle of the torso."""
        pitch = obs[1]
        return abs(pitch) > self.activation_threshold

    def step(self, action):
        current_obs = self.unwrapped._get_obs()

        if self.rta_active:
            self.recovery_steps += 1
            if self.recovery_steps >= self.min_recovery_duration and not self.is_state_unsafe(current_obs):
                self.rta_active = False
                self.recovery_steps = 0
        elif self.is_state_unsafe(current_obs):
            self.rta_active = True
            self.recovery_steps = 0

        if self.rta_active:
            actual_action = self.backup_controller.get_action(current_obs)
            info_flag = {"rta_active": True, "rta_interventions": 1}
        else:
            actual_action = action
            info_flag = {"rta_active": False, "rta_interventions": 0}

        next_obs, reward, terminated, truncated, info = self.env.step(actual_action)

        if self.rta_active:
            reward -= 5.0

        info.update(info_flag)

        return next_obs, reward, terminated, truncated, info
