import numpy as np

class BaseReward:
    def __init__(self, env, weights=None):
        self.env = env
        self.weights = weights or {}
        self.reward_dict = {}

    def compute(self, action):
        """Compute the total reward and update reward_dict for logging."""
        raise NotImplementedError

    def reset(self):
        """Reset internal reward state if any."""
        self.reward_dict = {}

    def get_reward_info(self):
        """Return the current reward breakdown for logging."""
        return self.reward_dict

    # --- Helper methods for common reward terms ---

    def _velocity_reward(self, current_vel, target_vel, sigma=5.0):
        """Gaussian reward for velocity matching."""
        vel_diff = np.linalg.norm(current_vel[:2] - target_vel[:2])
        return np.exp(-sigma * np.square(vel_diff))

    def _height_reward(self, current_height, target_height, sigma=20.0):
        """Gaussian reward for maintaining height."""
        return np.exp(-sigma * np.square(current_height - target_height))

    def _torque_penalty(self, torques, max_torques, sigma=0.02):
        """Exponential penalty for high torques."""
        return np.exp(-sigma * np.sum(np.abs(torques) / max_torques) / len(torques))

    def _action_diff_penalty(self, action, prev_action, sigma=0.02):
        """Exponential penalty for large action changes (smoothness)."""
        diff = np.sum(np.abs(action - prev_action))
        return np.exp(-sigma * diff)

    def _acceleration_penalty(self, base_accel, sigma=0.01):
        """Exponential penalty for base acceleration."""
        return np.exp(-sigma * np.sum(np.abs(base_accel)))

    def _orientation_reward(self, q1, q2, axis="yaw", sigma=30.0):
        """Gaussian reward for orientation matching."""
        dist = self.env._quaternion_distance(q1, q2, axis=axis)
        return np.exp(-sigma * dist)

    def _torso_centering_reward(self, com_pos, l_foot_pos, r_foot_pos, sigma=20.0):
        """Reward for keeping COM between feet."""
        feet_midpoint = 0.5 * (l_foot_pos + r_foot_pos)
        horizontal_offset = np.linalg.norm(com_pos[:2] - feet_midpoint[:2])
        return np.exp(-sigma * horizontal_offset**2)

    def _termination_penalty(self):
        """Penalty for falling or other termination conditions."""
        return 1.0 if self.env._is_terminated() else 0.0
