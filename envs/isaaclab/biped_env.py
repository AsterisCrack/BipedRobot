from __future__ import annotations

import logging
import os
import torch
import math
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, Imu, TiledCamera
from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material, push_by_setting_velocity, reset_joints_by_offset, reset_root_state_uniform
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .biped_env_cfg import BipedEnvCfg
from . import rewards
from .mdp.commands import UniformVelocityCommand
from utils.motion_reference import MotionReference, MotionReferenceConfig


class BipedEnv(DirectRLEnv):
    cfg: BipedEnvCfg

    def __init__(self, cfg: BipedEnvCfg, render_mode: str | None = None, **kwargs):
        # Set logging level
        if hasattr(cfg, "logging_level"):
            logging.getLogger().setLevel(getattr(logging, cfg.logging_level.upper()))

        # Calculate Observation Dimensions
        # Proprio: 3+3+3+3+12+12+12+2 = 50
        # (+2 gait phase clock)
        self.obs_proprio_dim = 50
        # Privileged: 3 (lin_vel_w) + 1 (height) + 6 (contacts: 2 feet * 3 forces) = 10
        self.obs_priv_dim = 10
        
        self.num_obs_policy = self.obs_proprio_dim
        if cfg.policy_has_privileged_info:
            self.num_obs_policy += self.obs_priv_dim
            
        self.num_obs_critic = self.obs_proprio_dim
        if cfg.critic_has_privileged_info:
            self.num_obs_critic += self.obs_priv_dim
            
        # History stacking for policy only (critic has privileged ground-truth, no history needed)
        if cfg.history_size > 0 and cfg.use_history:
            self.num_obs_policy *= cfg.history_size
            # num_obs_critic stays at proprio+priv dim — no multiplication
            
        # Update config dimensions
        cfg.observation_space_dim["policy"] = self.num_obs_policy
        cfg.observation_space_dim["critic"] = self.num_obs_critic
        
        # Set observation_space and action_space for DirectRLEnvCfg validation
        cfg.observation_space = self.num_obs_policy
        cfg.action_space = 12
        cfg.state_space = self.num_obs_critic
        
        super().__init__(cfg, render_mode, **kwargs)

        # Find feet indices (right then left)
        self.feet_indices, _ = self.robot.find_bodies(
            [self.cfg.right_foot_body_name, self.cfg.left_foot_body_name]
        )
        self.right_foot_idx = self.feet_indices[0]
        self.left_foot_idx = self.feet_indices[1]

        # Specific joints for penalty (optional; keep empty if unused)
        self.abduction_joint_indices = torch.tensor([], device=self.device, dtype=torch.long)

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

        # Gait phase clock: scalar phase per env, advanced proportional to commanded speed
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.gait_clock_freq = getattr(cfg, 'gait_clock_base_freq', 1.5)

        # Action smoothing filter (EMA): target = α*raw + (1-α)*prev
        self.filtered_targets = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.action_filter_alpha = getattr(cfg, 'action_filter_alpha', 1.0)

        # Actuator delay buffer: stores last max_delay+1 raw targets per env
        delay_range = getattr(cfg, 'action_delay_steps_range', [0, 0])
        self.max_action_delay = int(delay_range[1])
        self.action_delay_buffer = torch.zeros(self.num_envs, self.max_action_delay + 1, self.num_joints, device=self.device)
        self.action_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
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
        single_step_dim = self.num_obs_policy // self.history_size if self.history_size > 0 else self.num_obs_policy
        self.policy_history_buf = torch.zeros(self.num_envs, self.history_size, single_step_dim, device=self.device) if self.history_size > 0 else None
        
        # Track last foot contact for alternating gait reward
        # 0: Right, 1: Left. Initialize to 1 so Right foot (0) can start.
        self.last_feet_indices = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        
        # Step length buffers
        self.feet_pos_liftoff = torch.zeros(self.num_envs, 2, 3, device=self.device) # 2 feet
        self.feet_in_contact_prev = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.knee_bend_max = torch.zeros(self.num_envs, 2, device=self.device)

        # Pre-compute mirror transform tensors for symmetry augmentation
        self._build_mirror_transform()

        # Terrain curriculum notification: rolling window of episode lengths (in seconds).
        # Fires a one-time log message when mean exceeds the configured threshold.
        self._ep_len_window: list[float] = []
        self._ep_len_window_size = 500        # ~500 episodes across all envs
        self._curriculum_notified = False

        # DR curriculum: track last applied scale to avoid redundant event-param mutations
        self._curriculum_last_dr_scale: float = -1.0
        self._curriculum_last_cmd_scale: float = -1.0
        self._curriculum_last_init_scale: float = -1.0

        # Randomization
        self.push_interval_steps = int(self.cfg.push_interval_s / self.step_dt)
        
        # Observation Noise
        self._observation_noise_model = None
        if self.cfg.observation_noise_model:
            self._observation_noise_model = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )

        # Get specific body indices
        self._base_id, _ = self.contact_sensor.find_bodies(self.cfg.base_body_name)
        self._feet_ids, _ = self.contact_sensor.find_bodies(
            [self.cfg.right_foot_body_name, self.cfg.left_foot_body_name]
        )
        # Undesired contact bodies are all but feet
        self._undesired_contact_body_ids = [i for i in range(self.robot.num_bodies) if i not in self._feet_ids]
        
        # Specific joint indices for rewards
        self.hip_indices, _ = self.robot.find_joints(self.cfg.hip_joint_names)
        self.ankle_roll_indices, _ = self.robot.find_joints(self.cfg.ankle_roll_joint_names)
        _pitch_names = getattr(self.cfg, "ankle_pitch_joint_names", [])
        _pitch_idx, _ = self.robot.find_joints(_pitch_names) if _pitch_names else ([], [])
        self.ankle_pitch_indices = torch.tensor(_pitch_idx, device=self.device, dtype=torch.long)
        knee_indices, _ = self.robot.find_joints(self.cfg.knee_joint_names)
        self.knee_indices = torch.tensor(knee_indices, device=self.device, dtype=torch.long)

        # Custom joint limit scaling
        self.joint_limits_min = torch.tensor([x[0] for x in self.cfg.joint_limits], device=self.device)
        self.joint_limits_max = torch.tensor([x[1] for x in self.cfg.joint_limits], device=self.device)
        self.joint_range = self.joint_limits_max - self.joint_limits_min

        # Motion reference (imitation)
        self.motion_ref = None
        self.motion_time = None
        if self.cfg.animation_npz_path:
            motion_cfg = MotionReferenceConfig(
                npz_path=self.cfg.animation_npz_path,
                loop=self.cfg.animation_loop,
                speed=self.cfg.animation_speed,
                root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            )
            self.motion_ref = MotionReference(motion_cfg, self.device)
            self.motion_ref.bind_to_robot(self.robot.joint_names)
            self.motion_time = torch.zeros(self.num_envs, device=self.device)
            if self.cfg.animation_random_start:
                self._randomize_motion_time(torch.arange(self.num_envs, device=self.device))
        
    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        
        self.contact_sensor = ContactSensor(self.scene.sensors["contact_forces"].cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.imu_sensor = Imu(self.scene.sensors["imu"].cfg)
        self.scene.sensors["imu_sensor"] = self.imu_sensor

        # Video camera — InteractiveScene already instantiated it; just keep a reference.
        self.video_camera: TiledCamera | None = None
        if getattr(self.cfg, "enable_video_camera", False) and "camera" in self.scene.sensors:
            self.video_camera = self.scene.sensors["camera"]
        
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
        self.command_generator.compute(self.base_quat)

        # Map actions in [-1, 1] to joint position targets
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        normalized_action_01 = (self.actions + 1.0) * 0.5
        raw_targets = self.joint_limits_min + normalized_action_01 * self.joint_range

        # Actuator delay simulation: push new target into buffer, read delayed target
        if self.max_action_delay > 0:
            self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
            self.action_delay_buffer[:, 0] = raw_targets
            self.targets = self.action_delay_buffer[
                torch.arange(self.num_envs, device=self.device), self.action_delay
            ]
        else:
            self.targets = raw_targets

    def _apply_action(self):
        # Action smoothing filter: EMA on PD position targets
        self.filtered_targets = (
            self.action_filter_alpha * self.targets
            + (1.0 - self.action_filter_alpha) * self.filtered_targets
        )
        self.robot.set_joint_position_target(self.filtered_targets)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array" or self.video_camera is None:
            return super().render()
        # sim.render() must be called to populate TiledCamera annotator buffers.
        # RecordVideo calls render() immediately after reset() before any step has run,
        # so the camera is uninitialised unless we trigger the GPU render explicitly.
        self.sim.render()
        try:
            # TiledCamera output shape: [N, H, W, C]; return env-0 frame as [H, W, 3] RGB
            rgb = self.video_camera.data.output["rgb"]
            return rgb[0].cpu().numpy().astype(np.uint8)
        except Exception:
            cfg = self.cfg.scene.camera
            return np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)

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
        
        # Gait phase clock: advance proportional to commanded XY speed
        v_cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        self.gait_phase = (
            self.gait_phase + self.step_dt * 2.0 * math.pi * self.gait_clock_freq * v_cmd_norm
        ) % (2.0 * math.pi)
        phase_obs = torch.stack([torch.sin(self.gait_phase), torch.cos(self.gait_phase)], dim=-1)

        # Proprioceptive observations (48 base + 2 phase = 50)
        obs_proprio = torch.cat([
            self.imu_lin_acc_b,
            self.imu_ang_vel_b,
            self.projected_gravity_b,
            self.commands,
            (self.joint_pos - self.default_joint_pos),
            self.joint_vel,
            self.previous_actions,
            phase_obs,
        ], dim=-1)
        
        # Privileged observations (Standard Critic) — 10 dims total
        # These are unavailable on a real robot but help the value function learn faster.
        # [base_lin_vel_b (3)] [height (1)] [feet_contact_forces_flat (6)]

        # Ground-truth body-frame linear velocity — not in policy obs (which uses IMU accel)
        priv_lin_vel = self.base_lin_vel_b  # [N, 3]

        # Height above ground
        priv_height = self.robot.data.root_pos_w[:, 2:3]  # [N, 1]

        # Net contact force at each foot, flattened — [N, 2, 3] → [N, 6]
        feet_net_forces = self.contact_sensor.data.net_forces_w[:, self._feet_ids]  # [N, 2, 3]
        priv_feet_forces = feet_net_forces.reshape(self.num_envs, -1)  # [N, 6]

        obs_priv = torch.cat([
            priv_lin_vel,
            priv_height,
            priv_feet_forces,
        ], dim=-1)  # [N, 10]

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
        self.knee_bend_max[liftoff] = 0.0
        
        # Detect Touchdown: Air -> Contact
        touchdown = ~self.feet_in_contact_prev & feet_in_contact

        # Update last feet indices
        hit_right = touchdown[:, 0]
        hit_left = touchdown[:, 1]
        self.last_feet_indices[hit_right] = 0
        self.last_feet_indices[hit_left] = 1
        
        # Calculate stride length: distance from liftoff to current (touchdown) pos
        # We calculate it for all (vectorized), but only use it where touchdown is True
        stride_dist = torch.clamp(torch.norm(feet_pos - self.feet_pos_liftoff, dim=-1), max=0.15)
        
        # Update prev contact
        self.feet_in_contact_prev[:] = feet_in_contact

        # Track max knee bend during swing (per foot)
        knee_pos = self.joint_pos[:, self.knee_indices]
        knee_default = self.default_joint_pos[self.knee_indices]
        knee_bend = torch.abs(knee_pos - knee_default)
        knee_airborne = ~feet_in_contact
        self.knee_bend_max = torch.where(knee_airborne, torch.maximum(self.knee_bend_max, knee_bend), self.knee_bend_max)

        # 1. track_lin_vel_xy_exp (w=2.0)
        r_track_lin_vel_xy = rewards.track_lin_vel_xy_exp(self.commands, self.base_lin_vel_b, std=0.15)
        
        # 2. track_ang_vel_z_exp (w=1.0)
        r_track_ang_vel_z = rewards.track_ang_vel_z_exp(self.commands, self.base_ang_vel_b, std=0.15)
        
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
        r_swing_foot_height = rewards.swing_foot_height(feet_pos, feet_in_contact, min_height=0.02, max_height=0.06)
        
        # 18: Torso centering
        r_torso_centering = rewards.torso_centering_reward(self.robot.data.root_pos_w, feet_pos)

        # 19: Knee bend during swing on touchdown
        r_knee_bend_touchdown = rewards.knee_bend_on_touchdown(touchdown, self.knee_bend_max, min_bend=0.2, max_bend=1.0)
        self.knee_bend_max[touchdown] = 0.0

        # 20: Ankle pitch torque penalty — discourages toe-only stance (plantarflexion bias)
        if len(self.ankle_pitch_indices) > 0:
            r_ankle_torques = rewards.joint_torques_l2(self.joint_efforts[:, self.ankle_pitch_indices])
        else:
            r_ankle_torques = torch.zeros(self.num_envs, device=self.device)

        # 21: Animation tracking (optional)
        if self.motion_ref is not None:
            ref_pos, ref_vel = self.motion_ref.sample(self.motion_time)
            joint_pos_ref = self.joint_pos[:, self.motion_ref.robot_joint_indices]
            joint_vel_ref = self.joint_vel[:, self.motion_ref.robot_joint_indices]
            r_track_joint_pos = rewards.track_joint_pos_exp(joint_pos_ref, ref_pos, std=self.cfg.animation_pos_std)
            r_track_joint_vel = rewards.track_joint_vel_exp(joint_vel_ref, ref_vel, std=self.cfg.animation_vel_std)
        else:
            r_track_joint_pos = torch.zeros(self.num_envs, device=self.device)
            r_track_joint_vel = torch.zeros(self.num_envs, device=self.device)
        
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
            "knee_bend_touchdown": r_knee_bend_touchdown * self.cfg.rewards.get("knee_bend_touchdown", 0.0),
            "track_joint_pos_exp": r_track_joint_pos * self.cfg.rewards.get("track_joint_pos_exp", 0.0),
            "track_joint_vel_exp": r_track_joint_vel * self.cfg.rewards.get("track_joint_vel_exp", 0.0),
            "ankle_torques_l2": r_ankle_torques * self.cfg.rewards.get("ankle_torques_l2", 0.0),
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

        if self.motion_time is not None:
            self.motion_time += self.step_dt
        
        return total_reward

    def _randomize_motion_time(self, env_ids: torch.Tensor) -> None:
        if self.motion_ref is None or self.motion_time is None:
            return
        self.motion_time[env_ids] = torch.rand(len(env_ids), device=self.device) * self.motion_ref.duration

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
        self._apply_curriculum()
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

        # Reset episode sums for all reset environments
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.0

        if self.motion_time is not None:
            if self.cfg.animation_random_start:
                self._randomize_motion_time(env_ids)
            else:
                self.motion_time[env_ids] = 0.0
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
        self.knee_bend_max[env_ids] = 0.0

        # Reset gait phase to random uniform [0, 2π]
        self.gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * math.pi

        # Reset action filter to current joint positions to avoid a large transient
        self.filtered_targets[env_ids] = self.robot.data.joint_pos[env_ids]

        # Reset delay buffer and sample new per-env delay
        self.action_delay_buffer[env_ids] = 0.0
        if self.max_action_delay > 0:
            delay_range = getattr(self.cfg, 'action_delay_steps_range', [0, 0])
            self.action_delay[env_ids] = torch.randint(
                delay_range[0], delay_range[1] + 1, (len(env_ids),), device=self.device
            )
        
        # Reset feet contact history to True.
        self.feet_in_contact_prev[env_ids] = True

        # Reset last feet indices to 1 (Left) so Right foot (0) is valid first
        self.last_feet_indices[env_ids] = 1

        # Track rolling mean episode length for curriculum notification.
        if not self._curriculum_notified:
            ep_lens_s = (self.episode_length_buf[env_ids].float() * self.step_dt).tolist()
            self._ep_len_window.extend(ep_lens_s)
            if len(self._ep_len_window) > self._ep_len_window_size:
                self._ep_len_window = self._ep_len_window[-self._ep_len_window_size:]
            if (len(self._ep_len_window) >= self._ep_len_window_size
                    and not self.cfg.use_rough_terrain):
                mean_ep_len = sum(self._ep_len_window) / len(self._ep_len_window)
                threshold = getattr(self.cfg, 'curriculum_episode_len_threshold', 15.0)
                if mean_ep_len >= threshold:
                    logging.warning(
                        f"[Curriculum] Mean episode length {mean_ep_len:.1f}s >= {threshold:.1f}s threshold. "
                        "Robot is ready for rough terrain. Set use_rough_terrain=True and "
                        "use_terrain_curriculum=True in your config to advance to the next stage."
                    )
                    self._curriculum_notified = True

        # Terrain curriculum: advance per-env difficulty based on episode outcome.
        # Only active when rough terrain generator is used AND curriculum is enabled.
        if (getattr(self.cfg, 'use_terrain_curriculum', False)
                and self.cfg.use_rough_terrain
                and hasattr(self.scene, 'terrain')
                and hasattr(self.scene.terrain, 'update_env_origins')):
            succeeded = self.reset_time_outs[env_ids]   # timeout = episode completed
            failed    = self.reset_terminated[env_ids]  # early termination = fell
            self.scene.terrain.update_env_origins(env_ids, move_up=succeeded, move_down=failed)

        # Resample commands
        self._resample_commands(env_ids)

    def _get_curriculum_scales(self) -> dict[str, float]:
        step = float(self.common_step_counter)
        dr_start = float(self.cfg.curriculum_dr_start_steps)
        dr_full  = float(self.cfg.curriculum_dr_full_steps)
        dr_scale = max(0.0, min(1.0, (step - dr_start) / max(1.0, dr_full - dr_start)))
        cmd_scale  = min(1.0, step / max(1.0, float(self.cfg.curriculum_cmd_ramp_steps)))
        init_scale = min(1.0, step / max(1.0, float(self.cfg.curriculum_init_ramp_steps)))
        return {"dr": dr_scale, "cmd": cmd_scale, "init": init_scale}

    def _apply_curriculum(self) -> None:
        if not getattr(self.cfg, "curriculum_enabled", False):
            return
        scales = self._get_curriculum_scales()
        dr   = scales["dr"]
        cmd  = scales["cmd"]
        init = scales["init"]

        # curriculum_dr_events: if non-empty, only those event names are scaled;
        # empty list (default) means scale ALL DR events (backward compat).
        _cur_ev = getattr(self.cfg, "curriculum_dr_events", [])
        def _dr_applies(name: str) -> bool:
            return not _cur_ev or name in _cur_ev

        # Only rewrite event params when scale actually changed (saves dict allocations)
        if dr != self._curriculum_last_dr_scale:
            self._curriculum_last_dr_scale = dr

            if "push_robot" in self.cfg.events and _dr_applies("push_robot"):
                px = self.cfg.curriculum_dr_max_push_x * dr
                py = self.cfg.curriculum_dr_max_push_y * dr
                self.cfg.events["push_robot"].params["velocity_range"] = {
                    "x": (-px, px), "y": (-py, py)
                }

            if "randomize_mass" in self.cfg.events and _dr_applies("randomize_mass"):
                lo, hi = self.cfg.curriculum_dr_mass_range
                self.cfg.events["randomize_mass"].params["mass_distribution_params"] = (lo * dr, hi * dr)

            if "randomize_actuator_gains" in self.cfg.events and _dr_applies("randomize_actuator_gains"):
                lo, hi = self.cfg.curriculum_dr_gains_range
                scaled_lo = 1.0 - (1.0 - lo) * dr
                scaled_hi = 1.0 + (hi - 1.0) * dr
                self.cfg.events["randomize_actuator_gains"].params["stiffness_distribution_params"] = (scaled_lo, scaled_hi)
                self.cfg.events["randomize_actuator_gains"].params["damping_distribution_params"]   = (scaled_lo, scaled_hi)

            if "randomize_friction" in self.cfg.events and _dr_applies("randomize_friction"):
                lo, hi = self.cfg.curriculum_dr_friction_range
                scaled_lo = 1.0 - (1.0 - lo) * dr
                scaled_hi = 1.0 + (hi - 1.0) * dr
                self.cfg.events["randomize_friction"].params["static_friction_range"]  = (scaled_lo, scaled_hi)
                self.cfg.events["randomize_friction"].params["dynamic_friction_range"] = (scaled_lo, scaled_hi)

            if "randomize_com" in self.cfg.events and _dr_applies("randomize_com"):
                c = self.cfg.curriculum_dr_com_range * dr
                self.cfg.events["randomize_com"].params["com_range"] = {
                    "x": (-c, c), "y": (-c, c), "z": (-c, c)
                }

            if "randomize_payload" in self.cfg.events and _dr_applies("randomize_payload"):
                self.cfg.events["randomize_payload"].params["mass_distribution_params"] = (
                    0.0, self.cfg.curriculum_dr_payload_max * dr
                )

        if cmd != self._curriculum_last_cmd_scale:
            self._curriculum_last_cmd_scale = cmd
            start_lo, start_hi = self.cfg.curriculum_cmd_start_lin_vel_x
            full_lo,  full_hi  = self.cfg.curriculum_cmd_full_lin_vel_x
            new_vx = (
                start_lo + (full_lo - start_lo) * cmd,
                start_hi + (full_hi - start_hi) * cmd,
            )
            lat = 0.3 * cmd
            yaw = 0.3 * cmd
            self.command_generator.update_ranges({
                "lin_vel_x": new_vx,
                "lin_vel_y": (-lat, lat),
                "ang_vel_z": (-yaw, yaw),
            })

        if init != self._curriculum_last_init_scale:
            self._curriculum_last_init_scale = init
            r_min = self.cfg.curriculum_init_range_min
            r_max = self.cfg.curriculum_init_range_max
            r = r_min + (r_max - r_min) * init
            if "reset_robot_joints" in self.cfg.events:
                self.cfg.events["reset_robot_joints"].params["position_range"] = (-r, r)
                self.cfg.events["reset_robot_joints"].params["velocity_range"] = (-r, r)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Curriculum/dr_scale"]   = dr
        self.extras["log"]["Curriculum/cmd_scale"]  = cmd
        self.extras["log"]["Curriculum/init_scale"] = init

    def _build_mirror_transform(self):
        """Pre-compute permutation/sign tensors for left-right bilateral symmetry.
        Joint perm and signs are read from cfg.mirror_joint_perm / cfg.mirror_joint_signs
        so each robot class can define its own layout without touching this method.
        """
        d = self.device
        j_perm  = torch.tensor(self.cfg.mirror_joint_perm,  device=d, dtype=torch.long)
        j_signs = torch.tensor(self.cfg.mirror_joint_signs, device=d)

        # --- 50-dim actor obs mirror ---
        p = torch.arange(50, device=d, dtype=torch.long)
        s = torch.ones(50, device=d)
        s[1]  = -1.               # imu_lin_acc y
        s[3]  = -1.; s[5] = -1.  # imu_ang_vel x, z
        s[7]  = -1.               # gravity y
        s[10] = -1.; s[11] = -1. # commands: lin_vel_y, ang_vel_z
        p[12:24] = 12 + j_perm;  s[12:24] = j_signs  # joint_pos
        p[24:36] = 24 + j_perm;  s[24:36] = j_signs  # joint_vel
        p[36:48] = 36 + j_perm;  s[36:48] = j_signs  # prev_actions
        # phase_clock [48:50]: symmetric — no change
        self._mirror_obs_perm  = p
        self._mirror_obs_signs = s

        # --- 12-dim action mirror ---
        self._mirror_action_perm  = j_perm
        self._mirror_action_signs = j_signs

        # --- 60-dim critic obs mirror (50 actor + 10 privileged) ---
        # Privileged layout: [50:53]=lin_vel_b, [53]=height, [54:60]=feet_forces
        cp = torch.arange(60, device=d, dtype=torch.long)
        cs = torch.ones(60, device=d)
        cp[:50] = p;  cs[:50] = s
        cs[51]  = -1.  # lin_vel_b y  (offset 50 + 1)
        # feet_forces [54:60] = [r_fx, r_fy, r_fz, l_fx, l_fy, l_fz]
        # mirror   → [l_fx, -l_fy, l_fz, r_fx, -r_fy, r_fz]
        cp[54:60] = torch.tensor([57, 58, 59, 54, 55, 56], device=d, dtype=torch.long)
        cs[54:60] = torch.tensor([ 1., -1.,  1.,  1., -1.,  1.], device=d)
        self._mirror_critic_obs_perm  = cp
        self._mirror_critic_obs_signs = cs

    def mirror_obs(self, obs_dict: dict) -> dict:
        """Return a left-right mirrored copy of an observation dict.

        Accepts both ``"actor"``/``"critic"`` (replay-buffer convention) and
        ``"policy"``/``"critic"`` (env step convention) key names.
        Output keys mirror the input keys exactly.
        """
        result = {}
        for key in ("actor", "policy"):
            if key in obs_dict:
                a = obs_dict[key]
                if a.shape[-1] > self.obs_proprio_dim:
                    n, total = a.shape
                    chunks = a.view(n, total // self.obs_proprio_dim, self.obs_proprio_dim)
                    result[key] = (chunks[:, :, self._mirror_obs_perm] * self._mirror_obs_signs).view(n, total)
                else:
                    result[key] = a[:, self._mirror_obs_perm] * self._mirror_obs_signs
        if "critic" in obs_dict:
            c = obs_dict["critic"]
            result["critic"] = c[:, self._mirror_critic_obs_perm] * self._mirror_critic_obs_signs
        return result

    def mirror_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return a left-right mirrored copy of actions."""
        return actions[:, self._mirror_action_perm] * self._mirror_action_signs

    def _resample_commands(self, env_ids: torch.Tensor):
        self.command_generator.reset(env_ids)

