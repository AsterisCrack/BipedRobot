from __future__ import annotations

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from envs.isaaclab.basic_biped_env_cfg import BasicBipedEnvCfg
from envs.isaaclab.rewards import rewards

class BasicBipedEnv(DirectRLEnv):
    cfg: BasicBipedEnvCfg

    def __init__(self, cfg: BasicBipedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get feet indices
        self.feet_indices, self.feet_names = self.robot.find_bodies(".*_foot")
        feet_info = sorted(zip(self.feet_indices, self.feet_names), key=lambda x: x[1])
        self.feet_indices = [x[0] for x in feet_info]
        
        # Buffers
        self.base_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self.joint_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self.joint_efforts = torch.zeros(self.num_envs, 12, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Contact buffers
        self.last_step_time = torch.zeros(self.num_envs, device=self.device)
        self.prev_feet_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        
        # Joint limits
        self.joint_pos_limits = self.robot.data.soft_joint_pos_limits[0, :, :]
        self.joint_vel_limits = self.robot.data.soft_joint_vel_limits[0, :]
        self.joint_effort_limits = self.robot.data.joint_effort_limits[0, :]
        self.default_joint_pos = self.robot.data.default_joint_pos[0, :]

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        self.contact_sensor = self.scene.sensors["contact_forces"]
        self.contact_sensor_torso = self.scene.sensors["contact_forces_torso"]

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # Apply action scaling (if any)
        # BasicEnv does not scale actions, but clips them. 
        # Here we assume actions are [-1, 1] from agent.
        # If action_scale is 1.0, we send [-1, 1] to the robot.
        scaled_actions = self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(scaled_actions)

    def _apply_action(self) -> None:
        # Already handled in _pre_physics_step for DirectRLEnv?
        # No, DirectRLEnv calls _pre_physics_step then steps sim.
        pass

    def _get_observations(self) -> dict:
        # Update buffers
        self.base_lin_vel_b = self.robot.data.root_lin_vel_b
        self.base_ang_vel_b = self.robot.data.root_ang_vel_b
        self.base_quat = self.robot.data.root_quat_w
        self.base_pos = self.robot.data.root_pos_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_efforts = self.robot.data.applied_torque
        self.projected_gravity_b = self.robot.data.projected_gravity_b

        # BasicEnv Observation Construction (37 dims)
        # qpos (19): [pos (3), quat (4), joint_pos (12)]
        # qvel (18): [lin_vel (3), ang_vel (3), joint_vel (12)]
        
        # Note: BasicEnv uses global position (x, y, z).
        # Note: BasicEnv uses qvel. For free joint, MuJoCo qvel is usually local-frame angular, world-frame linear?
        # Or local-frame for both?
        # Standard MuJoCo: qvel[0:3] is linear velocity in world frame? No, usually local.
        # Let's stick to body frame velocities as they are more standard for RL.
        # If BasicEnv uses self.data.qvel directly, and it's a free joint...
        # We will use body frame velocities for now.
        
        obs = torch.cat([
            self.base_pos,          # 3
            self.base_quat,         # 4
            self.joint_pos,         # 12
            self.base_lin_vel_b,    # 3
            self.base_ang_vel_b,    # 3
            self.joint_vel,         # 12
        ], dim=-1)
        
        return {"policy": obs, "critic": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Get feet state
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]
        feet_quat = self.robot.data.body_quat_w[:, self.feet_indices]

        # Contact Reward
        current_time = self.episode_length_buf * self.step_dt

        # Compute individual rewards matching BasicEnv
        
        # 1. Velocity Reward (Gaussian around 0.5 m/s)
        r_forward_vel = rewards.forward_velocity_reward(self.base_lin_vel_b, target_velocity=0.5)
        
        # 2. Height Reward (Gaussian around 0.23 m)
        r_height = rewards.base_height_reward(self.base_pos, target_height=0.23)
        
        # 3. Torque Penalty
        r_torque = rewards.torque_reward(self.joint_efforts, self.joint_effort_limits)
        
        # 4. Action Diff Penalty
        r_action_diff = rewards.action_diff_reward(self.actions, self.previous_actions)
        
        # 5. Acceleration Penalty
        r_acceleration = rewards.acceleration_reward(self.robot.data.joint_acc)
        
        # 6. Orientation Penalties
        r_yaw_orient = rewards.yaw_orientation_penalty(self.base_quat)
        r_orient = rewards.orientation_penalty(self.projected_gravity_b) # Pitch/Roll
        
        # 7. Feet Orientation Penalty
        r_feet_orient = rewards.feet_orientation_penalty(feet_quat)
        
        # 8. Torso Centering
        r_torso_centering = rewards.torso_centering_reward(self.base_pos, feet_pos)
        
        # 9. Step Contact Reward
        r_contact_step, self.last_step_time, self.prev_feet_contact = rewards.step_contact_reward(
            self.contact_sensor.data.net_forces_w,
            feet_pos,
            self.prev_feet_contact,
            self.last_step_time,
            current_time
        )
        
        # 10. Termination Penalty
        torso_contact = torch.max(torch.max(torch.norm(self.contact_sensor_torso.data.force_matrix_w, dim=-1), dim=-1)[0], dim=-1)[0] > 1.0
        r_termination = rewards.termination_penalty(torso_contact)

        # Weighted sum
        total_reward += self.cfg.rewards["survived"]
        total_reward += r_forward_vel * self.cfg.rewards["velocity"]
        total_reward += self.cfg.rewards["step"] # Constant step reward? BasicEnv has step_reward=0.001
        total_reward += r_height * self.cfg.rewards["height"]
        total_reward += r_torque * self.cfg.rewards["torque"]
        total_reward += r_action_diff * self.cfg.rewards["action_diff"]
        total_reward += r_acceleration * self.cfg.rewards["acceleration"]
        total_reward += r_yaw_orient * self.cfg.rewards["yaw"]
        total_reward += r_orient * self.cfg.rewards["pitch_roll"]
        total_reward += r_feet_orient * self.cfg.rewards["feet_orient"]
        total_reward += r_torso_centering * self.cfg.rewards["torso_centering"]
        total_reward += r_contact_step * self.cfg.rewards["contact"]
        total_reward += r_termination * self.cfg.rewards["termination"]
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Terminate if torso contacts the ground
        torso_contact = torch.max(torch.max(torch.norm(self.contact_sensor_torso.data.force_matrix_w, dim=-1), dim=-1)[0], dim=-1)[0] > 1.0
        
        # Terminate if height is too low (BasicEnv: < 0.2)
        height_termination = self.base_pos[:, 2] < 0.2
        
        died = torso_contact | height_termination
        
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        # Reset robot state (BasicEnv resets to initial state + noise)
        # We use default root state + noise if configured, but here we disabled randomization
        # So just reset to default
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset buffers
        self.previous_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self.last_step_time[env_ids] = 0.0 
        self.prev_feet_contact[env_ids] = False 
        
        # Set fixed commands
        self.commands[env_ids, 0] = 0.5 # Fixed 0.5 m/s x-velocity
