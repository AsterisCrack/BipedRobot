import numpy as np
from envs.rewards.base_reward import BaseReward

class TargetReward(BaseReward):
    def __init__(self, env, weights=None):
        default_weights = {
            "velocity": 0.15,
            "step": 0.001,
            "height": 0.05,
            "torque": 0.02,
            "action_diff": 0.02,
            "acceleration": 0.1,
            "yaw": 0.02,
            "pitch_roll": 0.02,
            "target": 1.0,
            "looking_at_target": 0.1,
            "termination": -0.1
        }
        if weights:
            default_weights.update(weights)
        super().__init__(env, default_weights)
        
        self.height_command = 0.23
        self.desired_velocity = 0.6

    def compute(self, action):
        env = self.env
        data = env.data
        model = env.model
        
        # Direction to target
        direction_vector = np.array([np.cos(env.target), np.sin(env.target), 0])
        velocity_command = direction_vector * self.desired_velocity
        
        # 1. Velocity reward
        v_reward = self._velocity_reward(data.qvel, velocity_command)
        
        # 2. Height reward
        height_reward = self._height_reward(data.qpos[2], self.height_command)
        
        # 3. Action diff
        action_diff_reward = self._action_diff_penalty(action, env.prev_actions)
        
        # 4. Torque
        torque_reward = self._torque_penalty(data.actuator_force, model.actuator_forcerange[:, 1])
        
        # 5. Acceleration
        base_accel_reward = self._acceleration_penalty(data.qacc[0:3])
        
        # 6. Orientation
        yaw_difference = np.abs(data.qpos[3] - env.target)
        yaw_orient = np.exp(-30 * (yaw_difference / np.pi)) # Specific logic for target tracking
        
        orientation_cmd = np.array([env.target, 0, 0])
        pitch_roll_orient = self._orientation_reward(data.qpos[3:7], orientation_cmd, axis="pitch_roll")

        # 7. Target reaching & Looking
        looking_at_target = 1.0 if yaw_difference < 0.1 else 0.0
        reached_target = 1.0 if env._check_reached_target() else 0.0
        
        # 8. Termination
        termination_penalty = self._termination_penalty()

        rewards = {
            "velocity": v_reward,
            "step": 1.0,
            "height": height_reward,
            "torque": torque_reward,
            "action_diff": action_diff_reward,
            "acceleration": base_accel_reward,
            "yaw": yaw_orient,
            "pitch_roll": pitch_roll_orient,
            "target": reached_target,
            "looking_at_target": looking_at_target,
            "termination": termination_penalty
        }
        
        total_reward = sum(rewards[k] * self.weights.get(k, 0.0) for k in rewards)
        self.reward_dict = {f"reward_{k}": v * self.weights.get(k, 0.0) for k, v in rewards.items()}
        
        return total_reward
