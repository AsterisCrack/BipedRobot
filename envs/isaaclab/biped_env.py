from __future__ import annotations

import logging
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, quat_apply, quat_from_euler_xyz
from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material, push_by_setting_velocity, reset_joints_by_offset, reset_root_state_uniform
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .biped_env_cfg import BipedEnvCfg
from .rewards import rewards_v2 as rewards

class BipedEnv(DirectRLEnv):
    cfg: BipedEnvCfg

    def __init__(self, cfg: BipedEnvCfg, render_mode: str | None = None, **kwargs):
        # Set logging level
        if hasattr(cfg, "logging_level"):
            logging.getLogger().setLevel(getattr(logging, cfg.logging_level.upper()))

        # Calculate Observation Dimensions
        # Proprio: 3+3+4+3+12+12+12 = 48
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
        if cfg.history_size > 0:
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
        self.joint_efforts = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.targets = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        
        # Commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Base state
        self.base_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Logging
        self.episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.rewards.keys()
        }
        
        # History
        self.history_size = self.cfg.history_size
        self.policy_history_buf = torch.zeros(self.num_envs, self.history_size, self.num_obs_policy // self.history_size, device=self.device) if self.history_size > 0 else None
        self.critic_history_buf = torch.zeros(self.num_envs, self.history_size, self.num_obs_critic // self.history_size, device=self.device) if self.history_size > 0 else None
        
        # Randomization
        self.push_interval_steps = int(self.cfg.push_interval_s / self.step_dt)
        
        # Observation Noise
        self._observation_noise_model = None
        if self.cfg.observation_noise_model:
            self._observation_noise_model = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )
        
        # Physics Randomizers
        self.mass_randomizer = None
        self.friction_randomizer = None
        if self.cfg.enable_physics_randomization:
            if "randomize_mass" in self.cfg.events:
                self.mass_randomizer = randomize_rigid_body_mass(self.cfg.events["randomize_mass"], self)
            else:
                logging.warning("Physics randomization is enabled but 'randomize_mass' event is missing in config.")

            if "randomize_friction" in self.cfg.events:
                self.friction_randomizer = randomize_rigid_body_material(self.cfg.events["randomize_friction"], self)
            else:
                logging.warning("Physics randomization is enabled but 'randomize_friction' event is missing in config.")

        # Get specific body indices
        self._base_id, _ = self.contact_sensor.find_bodies("torso_link")
        self._feet_ids, _ = self.contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self.contact_sensor.find_bodies("(?!.*_foot).*")
        
    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        
        self.contact_sensor = ContactSensor(self.scene.sensors["contact_forces"].cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        
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
        self.actions = actions.clone() # Action noise is handled by DirectRLEnv
        
        # Clip actions
        self.actions = torch.clamp(self.actions, *self.cfg.action_space_limits)
        self.targets = self.robot.data.default_joint_pos + self.actions * self.cfg.action_scale
        
        # Random pushes
        if self.cfg.enable_perturbations and self.cfg.push_interval_s > 0:
            push_indices = (self.episode_length_buf % self.push_interval_steps == 0) & (self.episode_length_buf > 0)
            if torch.any(push_indices):
                # Use the event function
                if "push_robot" in self.cfg.events:
                    params = self.cfg.events["push_robot"].params.copy()
                    push_by_setting_velocity(self, push_indices.nonzero(as_tuple=True)[0], **params)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, info = super().step(actions)
        self.previous_actions = self.actions.clone()

        if self.sim.has_gui():
            self._update_visualization()

        return obs, rew, terminated, truncated, info

    def _update_arrow(self, marker: VisualizationMarkers, velocity_w: torch.Tensor, offset: list[float], scale: list[float]):
        # Orientation
        yaw = torch.atan2(velocity_w[:, 1], velocity_w[:, 0])
        orientations = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        
        # Scale
        scales = torch.tensor(scale, device=self.device).repeat(self.num_envs, 1)
        
        marker.visualize(translations=self.robot.data.root_pos_w + torch.tensor(offset, device=self.device),
                         orientations=orientations,
                         scales=scales)

    def _update_visualization(self):
        # 1. Command Velocity (Green)
        cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_vel_b[:, 0] = self.commands[:, 0]
        cmd_vel_b[:, 1] = self.commands[:, 1]
        cmd_vel_w = quat_apply(self.robot.data.root_quat_w, cmd_vel_b)
        self._update_arrow(self.vis_cmd, cmd_vel_w, [0, 0, 0.5], [0.2, 0.2, 0.2])
                               
        # 2. Actual Velocity (Blue)
        vel_w = quat_apply(self.robot.data.root_quat_w, self.base_lin_vel_b.clone())
        self._update_arrow(self.vis_vel, vel_w, [0, 0, 0.6], [0.5, 0.1, 0.1])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.base_pos = self.robot.data.root_pos_w
        self.base_quat = self.robot.data.root_quat_w
        self.base_lin_vel_b = self.robot.data.root_lin_vel_b
        self.base_ang_vel_b = self.robot.data.root_ang_vel_b
        self.projected_gravity_b = self.robot.data.projected_gravity_b
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_efforts = self.robot.data.applied_torque
        
        # Proprioceptive observations (Standard Policy)
        # 3 (lin vel) + 3 (ang vel) + 3 (Gravity) + 3 (commands) + 12 (joint pos) + 12 (joint vel) + 12 (prev_action) = 48
        # Removed scaling to match MuJoCo
        obs_proprio = torch.cat([
            self.base_lin_vel_b, 
            self.base_ang_vel_b,
            self.projected_gravity_b,
            self.commands,
            (self.joint_pos - self.default_joint_pos),
            self.joint_vel,
            self.previous_actions,
        ], dim=-1)
        
        # Privileged observations (Standard Critic)
        feet_contact_forces = self.contact_sensor.data.force_matrix_w_history[:, :, self._feet_ids]
        forces = torch.max(torch.max(torch.norm(feet_contact_forces, dim=-1), dim=-1)[0], dim=-1)[0]
        
        obs_priv = torch.cat([
            forces.unsqueeze(-1),
        ], dim=-1)
        
        # Mirroring
        if self.cfg.enable_mirroring:
            # TODO: Implement mirroring logic
            pass

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
        
        # Get feet state
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]
        feet_quat = self.robot.data.body_quat_w[:, self.feet_indices]

        # Compute individual rewards
        r_vel_tracking = rewards.velocity_tracking_reward(self.commands, self.base_lin_vel_b)
        r_ang_vel_tracking = rewards.angular_velocity_tracking_reward(self.commands, self.base_ang_vel_b)
        r_height_vel_tracking = rewards.height_velocity_tracking_reward(self.base_lin_vel_b)
        
        r_height = rewards.base_height_reward(self.robot.data.root_pos_w)
        r_stall = rewards.stall_penalty(self.base_lin_vel_b, self.commands)
        r_base_stability = rewards.base_stability_reward(self.base_ang_vel_b)
        
        r_torque = rewards.torque_reward(self.joint_efforts)
        r_action_diff = rewards.action_diff_reward(self.actions, self.previous_actions)
        r_acceleration = rewards.acceleration_reward(self.robot.data.joint_acc)
        r_flat_orient = rewards.flat_orientation_reward(self.projected_gravity_b)
        r_feet_flat = rewards.feet_flat_reward(feet_quat)
        r_torso_centering = rewards.torso_centering_reward(self.robot.data.root_pos_w, feet_pos)
        
        # Airtime reward
        first_contact = self.contact_sensor.compute_first_contact(self.step_dt)
        last_air_time = self.contact_sensor.data.last_air_time
        r_feet_airtime = rewards.feet_airtime_reward(first_contact, last_air_time, self.commands)
        
        # Termination check for reward
        net_contact_forces = self.contact_sensor.data.force_matrix_w_history
        died = torch.any(torch.max(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=-1)[0], dim=-1)[0] > 1.0, dim=1)
        r_termination = rewards.termination_penalty(died)
        
        # Weighted sum
        total_reward += self.cfg.rewards["survived"]
        total_reward += r_vel_tracking * self.cfg.rewards.get("velocity", 0.0)
        total_reward += r_ang_vel_tracking * self.cfg.rewards.get("ang_vel_tracking", 0.0)
        total_reward += r_height_vel_tracking * self.cfg.rewards.get("height_vel_tracking", 0.0)
        
        total_reward += r_height * self.cfg.rewards.get("height", 0.0)
        total_reward += r_stall * self.cfg.rewards.get("stall", 0.0)
        total_reward += r_base_stability * self.cfg.rewards.get("base_stability", 0.0)
        
        total_reward += r_torque * self.cfg.rewards["torque"]
        total_reward += r_action_diff * self.cfg.rewards["action_diff"]
        total_reward += r_acceleration * self.cfg.rewards["acceleration"]
        total_reward += r_flat_orient * self.cfg.rewards.get("flat_orientation", 0.0)
        total_reward += r_feet_flat * self.cfg.rewards.get("feet_flat", 0.0)
        total_reward += r_torso_centering * self.cfg.rewards.get("torso_centering", 0.0)
        total_reward += r_feet_airtime * self.cfg.rewards.get("feet_airtime", 0.0)
        total_reward += r_termination * self.cfg.rewards.get("termination", 0.0)
        
        # Update logging
        self.episode_sums["velocity"] = self.episode_sums.get("velocity", 0.0) + r_vel_tracking
        self.episode_sums["ang_vel_tracking"] = self.episode_sums.get("ang_vel_tracking", 0.0) + r_ang_vel_tracking
        self.episode_sums["height_vel_tracking"] = self.episode_sums.get("height_vel_tracking", 0.0) + r_height_vel_tracking
        self.episode_sums["height"] = self.episode_sums.get("height", 0.0) + r_height
        self.episode_sums["stall"] = self.episode_sums.get("stall", 0.0) + r_stall
        self.episode_sums["base_stability"] = self.episode_sums.get("base_stability", 0.0) + r_base_stability
        
        self.episode_sums["feet_flat"] = self.episode_sums.get("feet_flat", 0.0) + r_feet_flat
        self.episode_sums["torso_centering"] = self.episode_sums.get("torso_centering", 0.0) + r_torso_centering
        self.episode_sums["feet_airtime"] = self.episode_sums.get("feet_airtime", 0.0) + r_feet_airtime
        self.episode_sums["termination"] = self.episode_sums.get("termination", 0.0) + r_termination
        self.episode_sums["torque"] += r_torque
        self.episode_sums["action_diff"] += r_action_diff
        self.episode_sums["acceleration"] += r_acceleration
        self.episode_sums["flat_orientation"] = self.episode_sums.get("flat_orientation", 0.0) + r_flat_orient
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self.contact_sensor.data.force_matrix_w_history
        died = torch.any(torch.max(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=-1)[0], dim=-1)[0] > 1.0, dim=1)
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
        
        # Physics Randomization
        if self.cfg.enable_physics_randomization:
            if self.mass_randomizer:
                params = self.cfg.events["randomize_mass"].params.copy()
                self.mass_randomizer(self, env_ids, **params)
                
            if self.friction_randomizer:
                params = self.cfg.events["randomize_friction"].params.copy()
                self.friction_randomizer(self, env_ids, **params)
        
        # Reset buffers
        self.previous_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        if self.policy_history_buf is not None:
            self.policy_history_buf[env_ids] = 0.0
        if self.critic_history_buf is not None:
            self.critic_history_buf[env_ids] = 0.0
        
        # Resample commands
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        ranges = self.cfg.commands["base_velocity"]["ranges"]
        r_x = ranges["lin_vel_x"]
        r_y = ranges["lin_vel_y"]
        r_z = ranges["ang_vel_z"]
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r_x)
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r_y)
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r_z)
