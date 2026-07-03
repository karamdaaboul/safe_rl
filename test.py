from safe_rl.envs import SimplexRTAWrapper
import gymnasium as gym
import numpy as np

# ==============================================================================
# 2. THE BACKUP CONTROLLER (Safety Filter)
# ==============================================================================
class SafeRecoveryController:
    def __init__(self, action_space):
        self.action_space = action_space
        
        # Target neutral joint positions for HalfCheetah (bthigh, bshin, bfoot, fthigh, fshin, ffoot)
        self.target_qpos = np.array([0.0, -0.2, 0.0, 0.0, -0.2, 0.0])
        self.kp = 1.5  # Proportional gain
        self.kd = 0.1  # Derivative gain

    def get_action(self, obs):
        """
        A simple PD controller that tries to return the cheetah to a stable, neutral posture.
        """
        # In HalfCheetah-v4:
        # obs[2:8] are the joint angles
        # obs[11:17] are the joint velocities
        current_qpos = obs[2:8]
        current_qvel = obs[11:17]
        
        # Calculate PD control to force joints back to a stable position
        error = self.target_qpos - current_qpos
        action = (self.kp * error) - (self.kd * current_qvel)
        
        # Clip action to valid environment bounds
        return np.clip(action, self.action_space.low, self.action_space.high)

# ==============================================================================
# 3. THE PRIMARY CONTROLLER (Dummy RL Agent)
# ==============================================================================
class RecklessAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        """
        This agent is intentionally reckless to trigger the RTA.
        It continuously applies maximum torque in one direction to flip the cheetah.
        """
        # Force the cheetah to pitch backward aggressively
        return np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0])

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    # Initialize the base environment
    # Use render_mode="human" to actually see it working on your screen
    base_env = gym.make("HalfCheetah-v4", render_mode="human")
    
    # Initialize controllers
    backup_ctrl = SafeRecoveryController(base_env.action_space)
    primary_agent = RecklessAgent(base_env.action_space)
    
    # Wrap the environment with our Simplex RTA
    env = SimplexRTAWrapper(base_env, backup_controller=backup_ctrl, pitch_limit=0.6)
    
    obs, info = env.reset()
    
    print("\nStarting Simulation...")
    print("Primary agent will try to flip the Cheetah. RTA will prevent it.\n")
    
    for step in range(500):
        # 1. Primary agent computes its desired (reckless) action
        desired_action = primary_agent.get_action(obs)
        
        # 2. Step the wrapped environment (RTA will intervene if necessary)
        obs, reward, terminated, truncated, info = env.step(desired_action)
        
        # Print status when RTA takes over
        if info.get("rta_active"):
            print(f"Step {step:03d} | WARNING: Pitch {obs[1]:.2f} rad. RTA ACTIVE! Overriding action.")
        
        if terminated or truncated:
            print("\nEpisode finished. Resetting...")
            obs, info = env.reset()

    env.close()
    print("\nSimulation Complete.")