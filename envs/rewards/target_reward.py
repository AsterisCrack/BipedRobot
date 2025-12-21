import numpy as np
from envs.rewards.base_reward import BaseReward

class TargetReward(BaseReward):
    def __init__(self, env, weights=None):
        default_weights = {
            "velocity_tracking": 1.0,
            "step": 0.001,
            "height": 0.05,
            "torque": 0.02,
            "action_diff": 0.02,
            "acceleration": 0.1,
            "pitch_roll": 0.02,
            "termination": -0.1
        }
        if weights:
            default_weights.update(weights)
        super().__init__(env, default_weights)
        
        self.height_command = 0.23

    def compute(self, action):
        env = self.env
        data = env.data
        model = env.model
        
        # Get target velocity from environment (in robot frame)
        target_vel = env.target  # [x_vel, y_vel, w_vel]
        
        # Get robot velocity in robot frame
        # Linear velocity: need to transform from world to robot frame
        quat = data.qpos[3:7]  # Robot orientation quaternion
        world_lin_vel = data.qvel[0:3]  # Linear velocity in world frame
        
        # Convert quaternion to rotation matrix and transform velocity
        # Simplified: use inverse rotation to get velocity in robot frame
        # For proper implementation, we'd use full quaternion rotation
        # For now, approximate using yaw-only rotation
        yaw = data.qpos[3]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        robot_lin_vel = np.array([
            cos_yaw * world_lin_vel[0] + sin_yaw * world_lin_vel[1],  # x in robot frame
            -sin_yaw * world_lin_vel[0] + cos_yaw * world_lin_vel[1],  # y in robot frame
            world_lin_vel[2]  # z (not used in target but included)
        ])
        
        # Angular velocity is already in appropriate frame
        w_vel = data.qvel[5]  # Yaw rate
        
        # Compute velocity tracking error
        vel_error = np.array([
            target_vel[0] - robot_lin_vel[0],  # x velocity error
            target_vel[1] - robot_lin_vel[1],  # y velocity error
            target_vel[2] - w_vel  # angular velocity error
        ])
        
        # Velocity tracking reward (negative MSE)
        velocity_tracking_reward = -np.sum(vel_error ** 2)
        
        # 1. Height reward
        height_reward = self._height_reward(data.qpos[2], self.height_command)
        
        # 2. Action diff
        action_diff_reward = self._action_diff_penalty(action, env.prev_actions)
        
        # 3. Torque
        torque_reward = self._torque_penalty(data.actuator_force, model.actuator_forcerange[:, 1])
        
        # 4. Acceleration
        base_accel_reward = self._acceleration_penalty(data.qacc[0:3])
        
        # 5. Orientation (pitch and roll should stay near zero)
        orientation_cmd = np.array([0, 0, 0])  # Upright orientation
        pitch_roll_orient = self._orientation_reward(data.qpos[3:7], orientation_cmd, axis="pitch_roll")
        
        # 6. Termination
        termination_penalty = self._termination_penalty()

        rewards = {
            "velocity_tracking": velocity_tracking_reward,
            "step": 1.0,
            "height": height_reward,
            "torque": torque_reward,
            "action_diff": action_diff_reward,
            "acceleration": base_accel_reward,
            "pitch_roll": pitch_roll_orient,
            "termination": termination_penalty
        }
        
        total_reward = sum(rewards[k] * self.weights.get(k, 0.0) for k in rewards)
        self.reward_dict = {f"reward_{k}": v * self.weights.get(k, 0.0) for k, v in rewards.items()}
        
        return total_reward
