from __future__ import annotations

import logging
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, quat_apply, quat_from_euler_xyz
from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material, push_by_setting_velocity, reset_joints_by_offset, reset_root_state_uniform
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .biped_env_cfg_v2 import BipedEnvCfg
from .rewards import rewards_v3 as rewards
from .mdp.commands import UniformVelocityCommand
from envs.assets.robot.biped_robot import JOINT_LIMITS


class BipedEnv(DirectRLEnv):
    cfg: BipedEnvCfg

    def __init__(self, cfg: BipedEnvCfg, render_mode: str | None = None, **kwargs):
        # Set logging level
        if hasattr(cfg, "logging_level"):
            logging.getLogger().setLevel(getattr(logging, cfg.logging_level.upper()))

        # Calculate Observation Dimensions
        # Proprio: 3+3+3+3+12+12+12 = 48
        self.obs_proprio_dim = 48
        # Privileged: 3 (lin_vel_w) + 1 (height) + 6 (contacts: 2 feet * 3 forces) = 10
        self.obs_priv_dim = 10
        
        self.num_obs_policy = self.obs_proprio_dim
        if cfg.policy_has_privileged_info:
            self.num_obs_policy += self.obs_priv_dim
            
        self.num_obs_critic = self.obs_proprio_dim
        if cfg.critic_has_privileged_info:
            self.num_obs_critic += self.obs_priv_dim
            
        # Apply history
        if cfg.history_size > 0 and cfg.use_history:
            self.num_obs_policy *= cfg.history_size
            self.num_obs_critic *= cfg.history_size
            
        # Update config dimensions
        cfg.observation_space_dim["policy"] = self.num_obs_policy
        cfg.observation_space_dim["critic"] = self.num_obs_critic
        
        # Set observation_space and action_space for DirectRLEnvCfg validation
        cfg.observation_space = self.num_obs_policy
        cfg.action_space = 12
        cfg.state_space = self.num_obs_critic
        
        super().__init__(cfg, render_mode, **kwargs)

        # Find feet indices
        self.feet_indices, _ = self.robot.find_bodies(".*_foot")
        self.right_foot_idx = self.feet_indices[0]
        self.left_foot_idx = self.feet_indices[1]

        # Specific joints for penalty (user specified 2, 3 for abduction/sumo pose)
        self.abduction_joint_indices = torch.tensor([2, 3], device=self.device, dtype=torch.long)

        # Joint limits

        self.joint_pos_limits = self.robot.data.soft_joint_pos_limits[0, :, :]
        self.joint_vel_limits = self.robot.data.soft_joint_vel_limits[0, :]
        self.joint_effort_limits = self.robot.data.joint_effort_limits[0, :]
        
        # Defaults
        self.default_joint_pos = self.robot.data.default_joint_pos[0, :]
        self.default_joint_vel = self.robot.data.default_joint_vel[0, :]

        # Buffers
        self.num_actions = self.cfg.action_space
        self.num_joints = self.robot.num_joints
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.previous_previous_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.joint_efforts = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.targets = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        
        # Commands
        self.command_generator = UniformVelocityCommand(self.cfg, self.num_envs, self.device, self.step_dt)
        self.commands = self.command_generator.commands
        
        # Base state
        self.base_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)

        # IMU data (body frame)
        self.imu_lin_acc_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.imu_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Logging
        self.episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.rewards.keys()
        }
        
        # History
        self.history_size = self.cfg.history_size if self.cfg.use_history else 0
        self.policy_history_buf = torch.zeros(self.num_envs, self.history_size, self.num_obs_policy // self.history_size, device=self.device) if self.history_size > 0 else None
        self.critic_history_buf = torch.zeros(self.num_envs, self.history_size, self.num_obs_critic // self.history_size, device=self.device) if self.history_size > 0 else None
        
        # Track last foot contact for alternating gait reward
        # 0: Right, 1: Left. Initialize to 1 so Right foot (0) can start.
        self.last_feet_indices = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        
        # Step length buffers
        self.feet_pos_liftoff = torch.zeros(self.num_envs, 2, 3, device=self.device) # 2 feet
        self.feet_in_contact_prev = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        # Randomization
        self.push_interval_steps = int(self.cfg.push_interval_s / self.step_dt)
        
        # Observation Noise
        self._observation_noise_model = None
        if self.cfg.observation_noise_model:
            self._observation_noise_model = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )

        # Get specific body indices
        self._base_id, _ = self.contact_sensor.find_bodies("torso_link")
        self._feet_ids, _ = self.contact_sensor.find_bodies(".*foot")
        # Undesired contact bodies are all but feet
        self._undesired_contact_body_ids = [i for i in range(self.robot.num_bodies) if i not in self._feet_ids]
        
        # Specific joint indices for rewards
        self.hip_indices, _ = self.robot.find_joints([".*_hip_z", ".*_hip_x"])
        self.ankle_roll_indices, _ = self.robot.find_joints([".*_ankle_x"])

        # Custom joint limit scaling
        self.joint_limits_min = torch.tensor([x[0] for x in JOINT_LIMITS], device=self.device)
        self.joint_limits_max = torch.tensor([x[1] for x in JOINT_LIMITS], device=self.device)
        self.joint_range = self.joint_limits_max - self.joint_limits_min
        
    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        
        self.contact_sensor = ContactSensor(self.scene.sensors["contact_forces"].cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.imu_sensor = Imu(self.scene.sensors["imu"].cfg)
        self.scene.sensors["imu_sensor"] = self.imu_sensor
        
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Visualization Markers
        if self.sim.has_gui():
            # Target Velocity (Green)
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/CommandVelocity",
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self.vis_cmd = VisualizationMarkers(marker_cfg)
            
            # Actual Velocity (Blue)
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/BaseVelocity",
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
            self.vis_vel = VisualizationMarkers(marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        
        # --- Command Updates ---
        # Delegate to Command Generator
        self.command_generator.compute(self.base_quat)

        
        self.targets = self.actions * self.cfg.action_scale + self.default_joint_pos
        self.targets = torch.clamp(self.targets, self.joint_limits_min, self.joint_limits_max)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, info = super().step(actions)
        self.previous_previous_actions = self.previous_actions.clone()
        self.previous_actions = self.actions.clone()

        if self.sim.has_gui():
            self._update_visualization()

        return obs, rew, terminated, truncated, info

    def _update_visualization(self):
        self.command_generator.visualize(
            self.vis_cmd, 
            self.vis_vel, 
            self.robot.data.root_pos_w, 
            self.robot.data.root_quat_w, 
            self.base_lin_vel_b
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.base_pos = self.robot.data.root_pos_w
        self.base_quat = self.robot.data.root_quat_w
        self.base_lin_vel_b = self.robot.data.root_lin_vel_b
        self.base_ang_vel_b = self.robot.data.root_ang_vel_b

        imu_data = self.imu_sensor.data
        self.imu_lin_acc_b = imu_data.lin_acc_b
        self.imu_ang_vel_b = imu_data.ang_vel_b
        self.projected_gravity_b = imu_data.projected_gravity_b
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_efforts = self.robot.data.applied_torque
        
        # Proprioceptive observations (Standard Policy)
        # 3 (lin acc) + 3 (ang vel) + 3 (gravity) + 3 (commands) + 12 (joint pos) + 12 (joint vel) + 12 (prev_action) = 48
        # Removed scaling to match MuJoCo
        obs_proprio = torch.cat([
            self.imu_lin_acc_b,
            self.imu_ang_vel_b,
            self.projected_gravity_b,
            self.commands,
            (self.joint_pos - self.default_joint_pos),
            self.joint_vel,
            self.previous_actions,
        ], dim=-1)
        
        # Privileged observations (Standard Critic)
        feet_contact_forces = self.contact_sensor.data.force_matrix_w_history[:, :, self._feet_ids]
        
        # feet_contact_forces shape: [N, History, Feet, SensorDat, 3]
        # We process this to a single scalar [N] representing max contact force intensity
        
        # 1. Norm of the 3D force vector -> [N, H, F, S]
        contact_norm = torch.norm(feet_contact_forces, dim=-1)
        
        # 2. Max over sensor points (S) -> [N, H, F]
        max_over_sensors = torch.max(contact_norm, dim=-1)[0]
        
        # 3. Max over feet (F) -> [N, H]
        max_over_feet = torch.max(max_over_sensors, dim=-1)[0]
        
        # 4. Max over history (H) -> [N]
        forces = torch.max(max_over_feet, dim=-1)[0]
        
        obs_priv = torch.cat([
            forces.unsqueeze(-1),
        ], dim=-1)

        # Construct Policy Observations
        if self.cfg.policy_has_privileged_info:
            obs_policy = torch.cat([obs_proprio, obs_priv], dim=-1)
        else:
            obs_policy = obs_proprio.clone()
            # Add observation noise to policy observations
            if self._observation_noise_model:
                obs_policy = self._observation_noise_model(obs_policy)

        # Construct Critic Observations
        if self.cfg.critic_has_privileged_info:
            obs_critic = torch.cat([obs_proprio, obs_priv], dim=-1)
        else:
            obs_critic = obs_proprio.clone()

        # History
        if self.history_size > 0:
            # Policy History
            self.policy_history_buf = torch.roll(self.policy_history_buf, shifts=-1, dims=1)
            self.policy_history_buf[:, -1] = obs_policy
            obs_policy = self.policy_history_buf.view(self.num_envs, -1)
            
            # Critic History
            self.critic_history_buf = torch.roll(self.critic_history_buf, shifts=-1, dims=1)
            self.critic_history_buf[:, -1] = obs_critic
            obs_critic = self.critic_history_buf.view(self.num_envs, -1)

        return {"policy": obs_policy, "critic": obs_critic}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # --- Prepare Data ---
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]
        
        # Feet Contact Mask (for sliding and stride)
        # Check current contact force > threshold
        feet_contact_forces = self.contact_sensor.data.net_forces_w[:, self._feet_ids]
        feet_in_contact = torch.norm(feet_contact_forces, dim=-1) > 1.0

        # --- Update stride state ---
        # Detect Liftoff: Contact -> Air
        liftoff = self.feet_in_contact_prev & ~feet_in_contact
        self.feet_pos_liftoff[liftoff] = feet_pos[liftoff]
        
        # Detect Touchdown: Air -> Contact
        touchdown = ~self.feet_in_contact_prev & feet_in_contact

        # Update last feet indices
        hit_right = touchdown[:, 0]
        hit_left = touchdown[:, 1]
        self.last_feet_indices[hit_right] = 0
        self.last_feet_indices[hit_left] = 1
        
        # Calculate stride length: distance from liftoff to current (touchdown) pos
        # We calculate it for all (vectorized), but only use it where touchdown is True
        stride_dist = torch.norm(feet_pos - self.feet_pos_liftoff, dim=-1)
        
        # Update prev contact
        self.feet_in_contact_prev[:] = feet_in_contact

        # 1. track_lin_vel_xy_exp (w=2.0)
        r_track_lin_vel_xy = rewards.track_lin_vel_xy_exp(self.commands, self.base_lin_vel_b, std=0.25)
        
        # 2. track_ang_vel_z_exp (w=1.0)
        r_track_ang_vel_z = rewards.track_ang_vel_z_exp(self.commands, self.base_ang_vel_b, std=0.25)
        
        # 3. termination_penalty (w=-10.0)
        limit_angle = 0.78 # From TerminationsCfg
        died = torch.acos(-self.projected_gravity_b[:, 2]).abs() > limit_angle
        r_termination = died.float()

        # 4. lin_vel_z_l2 (w=-0.1)
        r_lin_vel_z = rewards.lin_vel_z_l2(self.base_lin_vel_b)
        
        # 5. ang_vel_xy_l2 (w=-0.05)
        r_ang_vel_xy = rewards.ang_vel_xy_l2(self.base_ang_vel_b)
        
        # 6. flat_orientation_l2 (w=-2.0)
        r_flat_orientation = rewards.flat_orientation_l2(self.projected_gravity_b)
        
        # 7. action_rate_l2 (w=-0.01)
        r_action_rate = rewards.action_rate_l2(self.actions, self.previous_actions)
        
        # 8. dof_torques_l2 (w=-2.0e-3)
        r_dof_torques = rewards.joint_torques_l2(self.joint_efforts)
        
        # 9. dof_acc_l2 (w=-1.0e-6)
        r_dof_acc = torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
        
        # 10. dof_pos_limits (w=-1.0)
        r_dof_pos_limits = rewards.joint_pos_limits(self.joint_pos, self.joint_limits_min, self.joint_limits_max)
        
        # 11. feet_air_time (w=1.0) ~ threshold 0.4
        current_air_time = self.contact_sensor.data.current_air_time[:, self._feet_ids]
        current_contact_time = self.contact_sensor.data.current_contact_time[:, self._feet_ids]
        r_feet_air_time = rewards.feet_air_time_positive_biped(current_air_time, current_contact_time, self.commands, threshold=0.5, min_speed_command_threshold=0.05)
        
        # Reward only when alternating feet
        target_air_foot = (self.last_feet_indices + 1) % 2
        feet_in_contact_target = torch.gather(feet_in_contact, 1, target_air_foot.unsqueeze(-1)).squeeze(-1)
        r_feet_air_time *= (~feet_in_contact_target).float()

        # 12. feet_slide (w=-0.1)
        # Pass raw history and velocity (sliced to feet) for internal computation
        r_feet_slide = rewards.feet_slide(
             self.contact_sensor.data.force_matrix_w_history[:, :, self._feet_ids, :], 
             self.robot.data.body_lin_vel_w[:, self.feet_indices]
        )
        
        # 13. undesired_contacts (w=-1.0) ~ threshold 1.0
        # Contact forces for Undesired Contacts (hips, knees, base)
        net_contact_forces_undesired = self.contact_sensor.data.force_matrix_w_history[:, :, self._undesired_contact_body_ids]
        # Max over history -> [N, History, Bodies, 3] -> Norm -> [N, History, Bodies] -> Max History -> [N, Bodies]
        undesired_forces_norm = torch.norm(net_contact_forces_undesired, dim=-1) # [N, T, B]
        undesired_forces_max_per_body = torch.max(torch.max(undesired_forces_norm, dim=1)[0], dim=-1)[0]
        r_undesired_contacts = rewards.undesired_contacts(undesired_forces_max_per_body, threshold=1.0)
        
        # 14. joint_deviation_hip (w=-0.2)
        r_joint_deviation_hip = rewards.joint_deviation_l1(
            self.joint_pos[:, self.hip_indices], 
            self.default_joint_pos[self.hip_indices]
        )
        
        # 15. joint_deviation_ankle_roll (w=-0.2)
        r_joint_deviation_ankle = rewards.joint_deviation_l1(
            self.joint_pos[:, self.ankle_roll_indices],
            self.default_joint_pos[self.ankle_roll_indices]
        )
        
        # 16. step_length (w=0.0 default, set in config)
        r_step_length = rewards.step_length(touchdown, stride_dist, self.commands, min_speed_command_threshold=0.1)

        # 17. swing_foot_height (w=0.0 default)
        r_swing_foot_height = rewards.swing_foot_height(feet_pos, feet_in_contact, min_height=0.05, max_height=0.15)
        
        # 18: Torso centering
        r_torso_centering = rewards.torso_centering_reward(self.robot.data.root_pos_w, feet_pos)
        
        # Term dict
        reward_terms = {
            "track_lin_vel_xy_exp": r_track_lin_vel_xy * self.cfg.rewards.get("track_lin_vel_xy_exp", 0.0),
            "track_ang_vel_z_exp": r_track_ang_vel_z * self.cfg.rewards.get("track_ang_vel_z_exp", 0.0),
            "termination_penalty": r_termination * self.cfg.rewards.get("termination_penalty", 0.0),
            "lin_vel_z_l2": r_lin_vel_z * self.cfg.rewards.get("lin_vel_z_l2", 0.0),
            "ang_vel_xy_l2": r_ang_vel_xy * self.cfg.rewards.get("ang_vel_xy_l2", 0.0),
            "flat_orientation_l2": r_flat_orientation * self.cfg.rewards.get("flat_orientation_l2", 0.0),
            "action_rate_l2": r_action_rate * self.cfg.rewards.get("action_rate_l2", 0.0),
            "dof_torques_l2": r_dof_torques * self.cfg.rewards.get("dof_torques_l2", 0.0),
            "dof_acc_l2": r_dof_acc * self.cfg.rewards.get("dof_acc_l2", 0.0),
            "dof_pos_limits": r_dof_pos_limits * self.cfg.rewards.get("dof_pos_limits", 0.0),
            "feet_air_time": r_feet_air_time * self.cfg.rewards.get("feet_air_time", 0.0),
            "feet_slide": r_feet_slide * self.cfg.rewards.get("feet_slide", 0.0),
            "undesired_contacts": r_undesired_contacts * self.cfg.rewards.get("undesired_contacts", 0.0),
            "joint_deviation_hip": r_joint_deviation_hip * self.cfg.rewards.get("joint_deviation_hip", 0.0),
            "joint_deviation_ankle_roll": r_joint_deviation_ankle * self.cfg.rewards.get("joint_deviation_ankle_roll", 0.0),
            "step_length": r_step_length * self.cfg.rewards.get("step_length", 0.0),
            "swing_foot_height": r_swing_foot_height * self.cfg.rewards.get("swing_foot_height", 0.0),
            "torso_centering": r_torso_centering * self.cfg.rewards.get("torso_centering", 0.0),
        }
        
        # Weighted sum
        total_reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
        
        # Apply global scaling
        if hasattr(self.cfg, "reward_scale"):
           total_reward *= self.cfg.reward_scale
        
        # Logging
        for key, value in reward_terms.items():
            if key in self.episode_sums:
                self.episode_sums[key] += value
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        limit_angle = 0.784  # ~45 degrees
        died = torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs() > limit_angle
        
        # Terminate if both feet are airborne (after 1s of settling time)
        feet_contact_forces = self.contact_sensor.data.net_forces_w[:, self._feet_ids]
        feet_in_contact = torch.norm(feet_contact_forces, dim=-1) > 1.0
        both_airborne = torch.all(~feet_in_contact, dim=-1)
        
        min_time_steps = int(1.0 / self.step_dt)
        died = died | (both_airborne & (self.episode_length_buf > min_time_steps))
        return died, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        # Reset robot state
        if "reset_robot_joints" in self.cfg.events:
            params = self.cfg.events["reset_robot_joints"].params.copy()
            reset_joints_by_offset(self, env_ids, **params)
        else:
            raise ValueError("Missing 'reset_robot_joints' event in config for resetting robot joints.")
        
        # Reset base state
        if "reset_base" in self.cfg.events:
            params = self.cfg.events["reset_base"].params.copy()
            reset_root_state_uniform(self, env_ids, **params)
        else:
            # Fallback to default reset
            self.robot.set_root_state(
                pos=self.robot.data.default_root_pos[0, :].unsqueeze(0).repeat(len(env_ids), 1),
                quat=self.robot.data.default_root_quat[0, :].unsqueeze(0).repeat(len(env_ids), 1),
                lin_vel=torch.zeros(len(env_ids), 3, device=self.device),
                ang_vel=torch.zeros(len(env_ids), 3, device=self.device),
                env_ids=env_ids,
            )
        
        # Logging
        extras = dict()
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            extras["Reward_Terms/" + key] = episodic_sum_avg / (self.max_episode_length * self.step_dt)
            self.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras_term = dict()
        extras_term["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras_term["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras_term)
        
        # Console Logging
        if len(self.extras["log"]) > 0:
            logging.info("Episode Reward:")
            for key, value in self.extras["log"].items():
                logging.info(f"  {key}: {value}")

        # Reset buffers
        self.previous_actions[env_ids] = 0.0
        self.previous_previous_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        if self.policy_history_buf is not None:
            self.policy_history_buf[env_ids] = 0.0
        if self.critic_history_buf is not None:
            self.critic_history_buf[env_ids] = 0.0
        
        # Reset feet contact history to True.
        self.feet_in_contact_prev[env_ids] = True

        # Reset last feet indices to 1 (Left) so Right foot (0) is valid first
        self.last_feet_indices[env_ids] = 1

        # Resample commands
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        self.command_generator.reset(env_ids)

