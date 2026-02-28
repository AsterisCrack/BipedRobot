import numpy as np
from envs.rewards.base_reward import BaseReward

class WalkReward(BaseReward):
    def __init__(self, env, weights=None):
        # Default weights for walking if not provided
        default_weights = {
            "velocity": 0.15,
            "step": 0.001,
            "height": 0.0,
            "torque": 0.01,
            "action_diff": 0.01,
            "acceleration": 0.05,
            "yaw": 0.02,
            "pitch_roll": 0.04,
            "feet_orient": 0.3,
            "torso_centering": 0.1,
            "contact": 1.5,
            "termination": -0.05
        }
        if weights:
            default_weights.update(weights)
        super().__init__(env, default_weights)
        
        self.height_command = 0.23
        self.velocity_command = np.array([0.5, 0, 0])
        self.min_velocity = 0.1

    def compute(self, action):
        env = self.env
        data = env.data
        model = env.model
        
        # 1. Velocity reward
        v_reward = self._velocity_reward(data.qvel, self.velocity_command)
        if np.linalg.norm(data.qvel[0:2]) < self.min_velocity or data.qvel[0] < 0:
            v_reward = -10.0
            
        # 2. Height reward
        height_reward = self._height_reward(data.qpos[2], self.height_command)
        
        # 3. Step reward
        step_reward = 1.0
        
        # 4. Action difference reward
        action_diff_reward = self._action_diff_penalty(action, env.prev_actions)
        
        # 5. Torque reward
        torque_reward = self._torque_penalty(data.actuator_force, model.actuator_forcerange[:, 1])
        
        # 6. Acceleration reward
        base_accel_reward = self._acceleration_penalty(data.qacc[0:3])
        
        # 7. Orientation rewards
        yaw_orient = self._orientation_reward(data.qpos[3:7], np.zeros(3), axis="yaw")
        pitch_roll_orient = self._orientation_reward(data.qpos[3:7], np.zeros(3), axis="pitch_roll")
        
        # 8. Feet orientation
        l_foot_orient = env._get_body_orientation(env.l_feet_body)
        r_foot_orient = env._get_body_orientation(env.r_feet_body)
        l_foot_diff = env._quaternion_distance(l_foot_orient, np.zeros(3), axis="yaw_pitch_roll")
        r_foot_diff = env._quaternion_distance(r_foot_orient, np.zeros(3), axis="yaw_pitch_roll")
        feet_orient_reward = np.exp(-30 * (l_foot_diff + r_foot_diff)) # Keep this here or could be _orientation_reward(..., axis="all")
        
        # 9. Torso centering
        l_foot_pos = data.geom_xpos[env.l_feet_geom][:3]
        r_foot_pos = data.geom_xpos[env.r_feet_geom][:3]
        torso_centering_reward = self._torso_centering_reward(data.qpos[0:3], l_foot_pos, r_foot_pos)
        
        # 10. Contact reward (Phase-based) - Logic remains here as it's specific to stepping
        contact_reward = 0.0
        left_contact = env._feet_in_contact(env.l_feet_geom)
        right_contact = env._feet_in_contact(env.r_feet_geom)
        
        if left_contact and not env.prev_left_contact:
            if data.geom_xpos[env.l_feet_geom][0] > data.geom_xpos[env.r_feet_geom][0]:
                contact_reward = data.time - env.last_step_time
                env.last_step_time = data.time
        if right_contact and not env.prev_right_contact:
            if data.geom_xpos[env.r_feet_geom][0] > data.geom_xpos[env.l_feet_geom][0]:
                contact_reward = data.time - env.last_step_time
                env.last_step_time = data.time
                
        # 11. Termination penalty
        termination_penalty = self._termination_penalty()
        
        # Update state for next step
        env.prev_left_contact = left_contact
        env.prev_right_contact = right_contact

        # Aggregate
        rewards = {
            "velocity": v_reward,
            "step": step_reward,
            "height": height_reward,
            "torque": torque_reward,
            "action_diff": action_diff_reward,
            "acceleration": base_accel_reward,
            "yaw": yaw_orient,
            "pitch_roll": pitch_roll_orient,
            "feet_orient": feet_orient_reward,
            "torso_centering": torso_centering_reward,
            "contact": contact_reward,
            "termination": termination_penalty
        }
        
        total_reward = sum(rewards[k] * self.weights.get(k, 0.0) for k in rewards)
        self.reward_dict = {f"reward_{k}": v * self.weights.get(k, 0.0) for k, v in rewards.items()}
        
        return total_reward
