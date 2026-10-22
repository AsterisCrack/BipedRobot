from __future__ import annotations

import logging
import os
import re
import torch
import math
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, Imu, TiledCamera
from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material, push_by_setting_velocity, reset_joints_by_offset, reset_root_state_uniform
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .biped_env_cfg import BipedEnvCfg
from . import rewards
from .mdp.commands import UniformVelocityCommand
from utils.motion_reference import GaitTableReference, MotionReference, MotionReferenceConfig


class StepAveragedImu(Imu):
    """Imu whose linear/angular acceleration is the mean over the time since its last recompute.

    Isaac Lab's Imu divides the velocity change since its LAST RECOMPUTE by the LAST physics dt.
    With lazy sensor updates (the scene default) the buffers are only recomputed when read, i.e.
    once per control step, so the motion part of lin_acc_b came out `decimation` times too large
    (4x at 200 Hz physics / 50 Hz control) and the policy trained on an accelerometer that neither
    MuJoCo nor the robot's BNO055 reproduces. Dividing by the real elapsed time gives the average
    over the control step, which is also closer to the BNO055's internally filtered output.
    """

    def _update_buffers_impl(self, env_ids):
        elapsed = (self._timestamp - self._timestamp_last_update)[env_ids]
        dt = self._dt
        self._dt = elapsed.clamp(min=self._sim_physics_dt).unsqueeze(-1)
        try:
            super()._update_buffers_impl(env_ids)
        finally:
            self._dt = dt


class BipedEnv(DirectRLEnv):
    cfg: BipedEnvCfg

    def __init__(self, cfg: BipedEnvCfg, render_mode: str | None = None, **kwargs):
        # Set logging level
        if hasattr(cfg, "logging_level"):
            logging.getLogger().setLevel(getattr(logging, cfg.logging_level.upper()))

        # Sign guard: a penalty term with a positive weight is a bonus (foot_flat_l2 = +1.0 rewarded
        # edge standing). Raises ValueError listing every wrong-sign weight; see
        # rewards.REWARD_WEIGHT_SIGN. Valid configs are untouched.
        rewards.validate_reward_weight_signs(
            cfg.rewards,
            schedule={k: getattr(cfg, k, None) for k in rewards.SCHEDULED_WEIGHT_SIGN},
            where="BipedEnv cfg.rewards",
        )

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
        # preserve_order=True is required: without it find_bodies returns ascending body-index
        # order, and V2's left chain precedes the right in the asset, so [right, left] silently
        # came back as [left, right] — inverting right_foot_idx/left_foot_idx and the leg
        # convention in gait_phase_contact.
        self.feet_indices, _ = self.robot.find_bodies(
            [self.cfg.right_foot_body_name, self.cfg.left_foot_body_name], preserve_order=True
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
        if cfg.gait_clock_mode not in ("speed_proportional", "fixed_period"):
            raise ValueError(f"gait_clock_mode {cfg.gait_clock_mode!r}: expected speed_proportional | fixed_period")
        if cfg.gait_clock_mode == "fixed_period" and not (0.2 < cfg.gait_clock_period_s < 5.0):
            raise ValueError(f"gait_clock_period_s {cfg.gait_clock_period_s!r} s is implausible (expected 0.2-5 s)")
        self._fixed_period_clock = cfg.gait_clock_mode == "fixed_period" and not cfg.reference_npz_path
        if cfg.feet_air_time_mode not in ("per_step", "touchdown"):
            raise ValueError(f"feet_air_time_mode {cfg.feet_air_time_mode!r}: expected per_step | touchdown")
        if not (0.0 <= cfg.feet_air_time_min_s < cfg.feet_air_time_max_s):
            raise ValueError("feet_air_time_min_s must be >= 0 and < feet_air_time_max_s")
        if cfg.imitation_foot_agg not in ("mean", "min"):
            raise ValueError(f"imitation_foot_agg {cfg.imitation_foot_agg!r}: expected mean | min")
        if cfg.contact_height_mode not in ("hard", "soft"):
            raise ValueError(f"contact_height_mode {cfg.contact_height_mode!r}: expected hard | soft")
        rewards.validate_track_lin_target(
            cfg.track_lin_target, has_reference=bool(cfg.reference_npz_path), std_rel=cfg.track_lin_std_rel)
        rewards.validate_stand_pose_l2(cfg.rewards.get("stand_pose_l2", 0.0), has_reference=bool(cfg.reference_npz_path))
        if cfg.swing_max_time_s <= 0.0:
            raise ValueError("swing_max_time_s must be > 0")
        if min(cfg.track_lin_std_rel, cfg.track_ang_std_rel) < 0.0 or min(cfg.track_lin_std_min, cfg.track_ang_std_min) <= 0.0:
            raise ValueError("track_*_std_rel must be >= 0 and track_*_std_min > 0")

        # Action smoothing filter (EMA): target = α*raw + (1-α)*prev
        self.filtered_targets = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.action_filter_alpha = getattr(cfg, 'action_filter_alpha', 1.0)

        # Actuator delay buffer: stores last max_delay+1 raw targets per env
        delay_range = getattr(cfg, 'action_delay_steps_range', [0, 0])
        self.max_action_delay = int(delay_range[1])
        self.action_delay_buffer = torch.zeros(self.num_envs, self.max_action_delay + 1, self.num_joints, device=self.device)
        self.action_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Sensor lag buffers. Same ring-buffer convention as the actuator delay above:
        # newest sample written at index 0, index == age in control steps.
        # Servo encoders (joint pos+vel) lag a fixed number of steps for every env.
        self.servo_obs_delay = int(getattr(cfg, 'servo_obs_delay_steps', 0))
        self.servo_obs_buffer = torch.zeros(
            self.num_envs, self.servo_obs_delay + 1, 2 * self.num_joints, device=self.device
        )
        # IMU (accel + gyro + projected gravity, all from the same sensor) lags a per-env
        # amount drawn once at reset, so the policy must be robust to an unknown fixed lag.
        imu_delay_range = getattr(cfg, 'imu_obs_delay_steps_range', [0, 0])
        self.imu_obs_delay_range = [int(imu_delay_range[0]), int(imu_delay_range[1])]
        self.max_imu_obs_delay = self.imu_obs_delay_range[1]
        self.imu_obs_buffer = torch.zeros(
            self.num_envs, self.max_imu_obs_delay + 1, 9, device=self.device
        )
        self.imu_obs_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Per-episode constant sensor biases (see _get_observations for why constant matters).
        self.imu_accel_bias_range = float(getattr(cfg, 'imu_accel_bias', 0.0))
        self.imu_gyro_bias_range = float(getattr(cfg, 'imu_gyro_bias', 0.0))
        self.imu_accel_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.imu_gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-robot assembly errors, drawn ONCE here, never per episode. A mis-zeroed servo
        # horn or a crooked IMU board does not change between resets.
        #
        # Mount offset d, per joint, applied on BOTH paths. The servo closes its loop on its
        # own (offset) encoder, so the true joint settles displaced rather than the sensor
        # merely misreporting:
        #   observation  r = q + d                       (_get_observations)
        #   command      sim target = t - d              (_apply_action)
        #   next step    r = (t - d) + d = t             self-consistent in encoder space
        # The old encoder_offset was observation-only. That models a sensor lying to a PD
        # that still tracks the true angle, which a real serial servo cannot do.
        self.mount_offset_range = float(getattr(cfg, 'mount_offset_rad', 0.0))
        self.mount_offset = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        if self.mount_offset_range > 0.0:
            self.mount_offset.uniform_(-self.mount_offset_range, self.mount_offset_range)

        # IMU mounting misalignment: a constant rotation of the sensor frame, random axis,
        # angle uniform in [0, max]. This trains tolerance to residual misalignment. It can't
        # learn a SYSTEMATIC mount error, which is why the runtime corrects the known 180 deg
        # board flip explicitly (BipedRobotJetson hardware/imu/bno055.py).
        self.imu_misalign_max_rad = math.radians(float(getattr(cfg, 'imu_misalignment_deg', 0.0)))
        self.imu_rot = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        if self.imu_misalign_max_rad > 0.0:
            axis = torch.nn.functional.normalize(torch.randn(self.num_envs, 3, device=self.device), dim=-1)
            angle = torch.rand(self.num_envs, device=self.device) * self.imu_misalign_max_rad
            self.imu_rot = math_utils.matrix_from_quat(math_utils.quat_from_angle_axis(angle, axis))

        # Commands
        self.command_generator = UniformVelocityCommand(self.cfg, self.num_envs, self.device, self.step_dt)
        self.commands = self.command_generator.commands
        # The configured (full) command ranges. The command curriculum ramps TOWARD these;
        # it used to hard-code lat/yaw at +-0.3 and silently override config.yaml.
        self._cmd_full_ranges = {k: tuple(v) for k, v in self.command_generator.ranges.items()}
        
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

        # Torque headroom accumulators. The STS3215's continuous rating is 0.98 N-m at 12 V and
        # the DCMotor envelope clamps there, so we need to see whether the policy is living at
        # the ceiling -- a gait that draws near-continuous torque will not survive on hardware.
        # Under BAM there is no fixed torque ceiling (it is set by duty, back-EMF and vin), so
        # "saturated" means the firmware duty is pinned at its limit; see _get_rewards.
        self.actuator_effort_limit_ref = 0.98
        self.torque_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self.torque_sat_sum = torch.zeros(self.num_envs, device=self.device)
        self.tau_ext_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self.friction_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_step_count = torch.zeros(self.num_envs, device=self.device)

        # Nominal per-body COM and mass, captured before any randomization event has run (the
        # first _reset_idx happens after __init__). randomize_com caches onto the same
        # attribute, so both read one source of truth and the logged delta is a true drift.
        if not hasattr(self.robot, "_nominal_coms"):
            self.robot._nominal_coms = self.robot.root_physx_view.get_coms().clone()
        self._nominal_masses = self.robot.root_physx_view.get_masses().clone()

        # Raw body-frame lateral foot separation, in metres. foot_separation_penalty clamps at
        # min_sep, so once the stance is wider than min_sep the reward term reads a flat zero
        # and tells us nothing -- which is exactly what happened when it moved to body frame.
        # Log the unclamped value so min_sep can be calibrated against the real distribution.
        self.foot_sep_sum = torch.zeros(self.num_envs, device=self.device)
        # Same for the signed SOLE-CENTRE separation (foot_separation_mode: sole_band's quantity).
        self.foot_sole_sep_sum = torch.zeros(self.num_envs, device=self.device)
        
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

        # Observation noise, one std per proprio dim. Applied to the current 50-dim frame
        # before it enters the history buffer -- applying it to the flattened 250-dim stack
        # instead would re-randomise all 5 frames every step and destroy the history's
        # temporal coherence, which is the only thing carrying velocity information.
        _noise = self.cfg.observation_noise_std
        self.obs_noise_std = torch.cat([
            torch.full((n,), float(_noise[k]), device=self.device) for k, n in [
                ("imu_acc", 3), ("gyro", 3), ("projected_gravity", 3), ("commands", 3),
                ("joint_pos", self.num_joints), ("joint_vel", self.num_joints),
                ("previous_actions", self.num_joints), ("gait_clock", 2),
            ]
        ])
        assert self.obs_noise_std.shape[0] == self.obs_proprio_dim, (
            f"noise vector is {self.obs_noise_std.shape[0]} dims, "
            f"proprio is {self.obs_proprio_dim}"
        )

        # STS3215 magnetic encoder: 4096 counts/rev on position, and the speed register
        # reports in counts/s, so both channels share the same quantum.
        self.quantize_joint_obs = bool(getattr(self.cfg, 'quantize_joint_obs', False))
        self.encoder_step_rad = 2.0 * math.pi / 4096.0

        # Get specific body indices
        self._base_id, _ = self.contact_sensor.find_bodies(self.cfg.base_body_name)
        self._feet_ids, _ = self.contact_sensor.find_bodies(
            [self.cfg.right_foot_body_name, self.cfg.left_foot_body_name], preserve_order=True
        )
        # Must be [right, left] — gait_phase_contact indexes feet_in_contact[:, 0] as right.
        print(f"[biped_env] _feet_ids order: {[self.contact_sensor.body_names[i] for i in self._feet_ids]}")
        print(f"[biped_env] robot feet order: {[self.robot.body_names[i] for i in self.feet_indices]}")
        # Undesired contact bodies are all but feet
        self._undesired_contact_body_ids = [i for i in range(self.robot.num_bodies) if i not in self._feet_ids]
        # leg_contact_l2: every non-foot body of either leg (hip/thigh/calf/ankle links; base_link
        # and imu_link are excluded by the l_/r_ prefix). Indexed in contact-sensor body order.
        _leg_re = re.compile(r"^(l|r|left|right)_")
        self._leg_body_ids = [i for i, n in enumerate(self.contact_sensor.body_names)
                              if i not in self._feet_ids and _leg_re.match(n)]
        print(f"[biped_env] leg_contact bodies: {[self.contact_sensor.body_names[i] for i in self._leg_body_ids]}")
        _sole = [float(v) for v in self.cfg.foot_sole_offset_m]
        if len(_sole) != 3:
            raise ValueError(f"foot_sole_offset_m {_sole}: expected [x, lateral_magnitude, z]")
        self._foot_sole_offset = tuple(_sole)
        # foot_proximity (E21): footprint = sole rectangle centred on foot_sole_offset_m, see rewards.foot_footprint_gap
        _half = [float(v) for v in self.cfg.foot_sole_half_extents_m]
        if len(_half) != 2 or min(_half) <= 0.0:
            raise ValueError(f"foot_sole_half_extents_m {_half}: expected two positive values [x, y]")
        self._foot_sole_half = tuple(_half)
        self._foot_prox_margin = float(self.cfg.foot_proximity_margin_m)
        if self._foot_prox_margin <= 0.0:
            raise ValueError(f"foot_proximity_margin_m {self._foot_prox_margin}: must be > 0")
        _ov_s = float(self.cfg.foot_overlap_termination_s)
        if _ov_s < 0.0:
            raise ValueError(f"foot_overlap_termination_s {_ov_s}: must be >= 0 (0 = off)")
        # 0 = termination off. Otherwise: consecutive control steps of footprint overlap that end the episode.
        self._foot_overlap_steps = int(math.ceil(_ov_s / self.step_dt - 1e-6)) if _ov_s > 0.0 else 0
        # Consecutive-overlap counter. Updated ONCE per step, in _get_dones (which DirectRLEnv.step runs before
        # _get_rewards), zeroed in _reset_idx. Only advanced while the termination is enabled.
        self.foot_overlap_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Envs whose last _get_dones ended the episode ONLY because of the overlap (for Episode_Termination/foot_overlap)
        self._foot_overlap_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Foot-link-origin height above the ground at a flat stance = the sole centre's depth below
        # the link origin. Seeds the swing datum (foot_ground_z) of reset envs on FLAT terrain.
        self._flat_foot_origin_z = abs(_sole[2])
        if self.cfg.foot_separation_mode not in ("origin_one_sided", "sole_band"):
            raise ValueError(f"foot_separation_mode {self.cfg.foot_separation_mode!r}: "
                             "expected origin_one_sided | sole_band")
        
        # Specific joint indices for rewards
        self.hip_indices, _ = self.robot.find_joints(self.cfg.hip_joint_names)
        self.ankle_roll_indices, _ = self.robot.find_joints(self.cfg.ankle_roll_joint_names)
        _pitch_names = getattr(self.cfg, "ankle_pitch_joint_names", [])
        _pitch_idx, _ = self.robot.find_joints(_pitch_names) if _pitch_names else ([], [])
        self.ankle_pitch_indices = torch.tensor(_pitch_idx, device=self.device, dtype=torch.long)
        knee_indices, _ = self.robot.find_joints(self.cfg.knee_joint_names)
        self.knee_indices = torch.tensor(knee_indices, device=self.device, dtype=torch.long)
        _hpitch_names = getattr(self.cfg, "hip_pitch_joint_names", [])
        _hpitch_idx, _ = self.robot.find_joints(_hpitch_names) if _hpitch_names else ([], [])
        self.hip_pitch_indices = torch.tensor(_hpitch_idx, device=self.device, dtype=torch.long)

        # Custom joint limit scaling.
        # NOTE: this binds row i of cfg.joint_limits to DOF i by POSITION, with no reference to
        # robot.joint_names. mirror_joint_perm rests on the same unverified assumption. We
        # already found find_bodies silently returning the feet in the wrong order, so print the
        # real order and check it against the documented one in biped_env_cfg.
        self.joint_limits_min = torch.tensor([x[0] for x in self.cfg.joint_limits], device=self.device)
        self.joint_limits_max = torch.tensor([x[1] for x in self.cfg.joint_limits], device=self.device)
        self.joint_range = self.joint_limits_max - self.joint_limits_min
        print(f"[biped_env] joint order ({len(self.robot.joint_names)}): {list(self.robot.joint_names)}")
        print(f"[biped_env] joint_limits rows (deg): "
              f"{[[round(math.degrees(v), 1) for v in lim] for lim in self.cfg.joint_limits]}")

        # Action map. "centered": a = 0 is the default (standing) pose and a = +-1 reaches each
        # joint limit, piecewise linear on either side. "absolute" is the legacy map that
        # spreads [-1, 1] across the whole limit box, which puts a = 0 at the box MIDPOINT
        # (knee -57.5 deg) and forces the policy to hold a ~ +0.92 on both knees just to
        # stand, close to tanh saturation. A plain residual (default + scale*a) was rejected:
        # with the knee box at [-120, +5] deg it either wastes most of one side or clips
        # flexion. The map is identical for L and R, so the mirror table is unaffected.
        self.action_map = getattr(self.cfg, "action_map", "absolute")
        if self.action_map not in ("centered", "absolute", "residual"):
            raise ValueError(f"action_map must be 'centered', 'absolute' or 'residual', got {self.action_map!r}")
        self.action_span_pos = self.joint_limits_max - self.default_joint_pos
        self.action_span_neg = self.default_joint_pos - self.joint_limits_min
        if self.action_map == "centered" and (
                (self.action_span_pos <= 0).any() or (self.action_span_neg <= 0).any()):
            raise ValueError("centered action map needs every default joint pos strictly inside its limits")
        if self.action_map == "residual":
            self._init_residual_map()

        # BAM servo actuators need the articulation for the mass matrix and joint_acc, and
        # Isaac Lab gives an actuator no handle to it. Duck-typed, so the V1 robot (DCMotor)
        # is unaffected.
        self.bam_actuators = [a for a in self.robot.actuators.values() if hasattr(a, "attach")]
        for act in self.bam_actuators:
            act.attach(self.robot)

        # swing_foot_height datum: each foot's OWN last touchdown height (world Z of the
        # foot-link origin while in contact). Clearance is measured from there, so it is
        # terrain-local and per foot. A foot that is unloaded but still on the ground reads
        # ~0, which keeps the standing exploit closed.
        # (The previous datum was a single EMA of in-contact foot height relative to the
        # env origin. On rough terrain it absorbed the patch elevation, measuring 46.5 mm
        # against a true ~34 mm, which raised the effective dead zone to ~22 mm and
        # collapsed the reward.)
        self.foot_ground_z = torch.zeros(self.num_envs, 2, device=self.device)
        self.foot_ground_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Gait-quality diagnostics (per-episode means, logged at reset). They turn what you
        # see on screen into numbers: body ahead of the feet, torso lean, toe-in, foot lift,
        # limping. Signs are verified in src/isaaclab/check_actuator.py.
        _hy_l, _ = self.robot.find_joints(["(l|left)_hip_yaw.*"])
        _hy_r, _ = self.robot.find_joints(["(r|right)_hip_yaw.*"])
        self.hip_yaw_lr = torch.tensor(_hy_l + _hy_r, device=self.device, dtype=torch.long)
        self.diag_keys = ("com_ahead_m", "com_lateral_abs_m", "com_outside_mm", "lean_fwd_deg",
                          "toe_in_deg", "vel_err_moving_ms", "yaw_err_moving_rads",
                          "foot_gap_mm", "foot_overlap_frac")
        self.diag_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self.diag_keys}
        # Separate count per key: the *_moving_* keys are nan on envs the command tells to
        # stand, so they must be averaged over the steps they were actually valid.
        self.diag_counts = {k: torch.zeros(self.num_envs, device=self.device) for k in self.diag_keys}
        self.swing_peak = torch.zeros(self.num_envs, 2, device=self.device)
        self.swing_peak_sum = torch.zeros(self.num_envs, device=self.device)
        self.swing_count = torch.zeros(self.num_envs, device=self.device)
        self.contact_steps = torch.zeros(self.num_envs, 2, device=self.device)
        # Diagnostics/dragged_foot_frac: foot-steps with |F| <= 1 N, clearance < thresh and the foot
        # moving horizontally (the exploit contact_height_thresh_m closes). Count per env, per episode.
        self.dragged_foot_steps = torch.zeros(self.num_envs, device=self.device)

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

        # Gait-table imitation (Gait Studio). With a table the gait clock is fixed-period
        # (the table's) and holds while standing, instead of speed-proportional.
        self.gait_ref = None
        self.ref_frame = None   # last sampled reference, for debugging / visualisation
        if self.cfg.reference_npz_path:
            self.gait_ref = GaitTableReference(
                self.cfg.reference_npz_path, self.device,
                root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            )
            self.gait_ref.bind_to_robot(self.robot.joint_names)
            self._ref_std = dict(self.cfg.reference_std)
            g = self.gait_ref
            print(f"[biped_env] gait table: grid vx{tuple(round(v, 3) for v in (g.grid[0][0].item(), g.grid[0][-1].item()))} "
                  f"vy{tuple(round(v, 3) for v in (g.grid[1][0].item(), g.grid[1][-1].item()))} "
                  f"wz{tuple(round(v, 3) for v in (g.grid[2][0].item(), g.grid[2][-1].item()))}, "
                  f"{g.n_frames} frames, period {1.0 / g.freq.max().item():.3f}-{1.0 / g.freq.min().item():.3f}s, "
                  f"style {g.style_digest}")
            for axis, key in zip(g.grid, ("lin_vel_x", "lin_vel_y", "ang_vel_z")):
                lo, hi = self._cmd_full_ranges.get(key, (0.0, 0.0))
                if lo < axis[0].item() - 1e-6 or hi > axis[-1].item() + 1e-6:
                    print(f"[biped_env] WARNING: command range {key}=({lo}, {hi}) exceeds the table grid "
                          f"({axis[0].item():.3f}, {axis[-1].item():.3f}); commands outside get the edge reference")

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        
        self.contact_sensor = ContactSensor(self.scene.sensors["contact_forces"].cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # imu_step_averaged=False reproduces the old 4x accelerometer, only for evaluating
        # checkpoints trained before the fix.
        imu_cls = StepAveragedImu if getattr(self.cfg, "imu_step_averaged", True) else Imu
        self.imu_sensor = imu_cls(self.scene.sensors["imu"].cfg)
        self.scene.sensors["imu_sensor"] = self.imu_sensor

        # Video camera — InteractiveScene already instantiated it; just keep a reference.
        self.video_camera: TiledCamera | None = None
        if getattr(self.cfg, "enable_video_camera", False) and "camera" in self.scene.sensors:
            self.video_camera = self.scene.sensors["camera"]
        
        # add ground plane / rough terrain
        print(f"[Terrain] type={self.cfg.terrain.terrain_type}  generator={self.cfg.terrain.terrain_generator is not None}")
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

        # Map actions in [-1, 1] to joint position targets (see action_map in __init__).
        # Targets are in ENCODER space -- what the servo believes its angle is.
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        raw_targets = self.actions_to_targets(self.actions)

        # Actuator delay simulation: push new target into buffer, read delayed target
        if self.max_action_delay > 0:
            self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
            self.action_delay_buffer[:, 0] = raw_targets
            self.targets = self.action_delay_buffer[
                torch.arange(self.num_envs, device=self.device), self.action_delay
            ]
        else:
            self.targets = raw_targets

    def _init_residual_map(self) -> None:
        """residual: target = clamp(home + scale*a, limits). home == default_joint_pos, which
        env_config wrote into the articulation init_state from env_config.action_home_deg, so
        every 'relative to default' term (obs, deviation rewards, dof_pos_l2, knee bend) is
        measured from it. Check that it really arrived, by joint NAME order, and is mirror-safe."""
        from utils.residual_action import JOINT_TYPES, check_residual_map, home_rad, scale_rad
        for i, name in enumerate(self.robot.joint_names):
            want = ("l_" if i % 2 == 0 else "r_") + JOINT_TYPES[i // 2]
            if not name.startswith(want):
                raise ValueError(f"residual map: joint {i} is {name!r}, expected {want}* "
                                 f"(policy order is L,R interleaved per joint type)")
        if not (self.cfg.action_home_deg and self.cfg.action_scale_rad):
            raise ValueError("action_map 'residual' needs cfg.action_home_deg and cfg.action_scale_rad")
        home = home_rad(self.cfg.action_home_deg)
        scale = scale_rad(self.cfg.action_scale_rad)
        check_residual_map(home, scale, self.joint_limits_min.cpu().numpy(), self.joint_limits_max.cpu().numpy(),
                           self.cfg.mirror_joint_perm, self.cfg.mirror_joint_signs)
        home_t = torch.as_tensor(home, dtype=self.default_joint_pos.dtype, device=self.device)
        err = (self.default_joint_pos - home_t).abs().max().item()
        if err > 1e-4:
            raise ValueError(f"residual map: articulation default_joint_pos differs from action_home_deg "
                             f"by {err:.2e} rad; env_config.apply_env_config must set init_state.joint_pos")
        self.action_scale = torch.as_tensor(scale, dtype=self.default_joint_pos.dtype, device=self.device)

    def actions_to_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Policy action in [-1, 1] -> joint target in encoder space. Mirrored by export_onnx."""
        if self.action_map == "residual":
            return torch.clamp(self.default_joint_pos + self.action_scale * torch.clamp(actions, -1.0, 1.0),
                               self.joint_limits_min, self.joint_limits_max)
        if self.action_map == "centered":
            span = torch.where(actions >= 0.0, self.action_span_pos, self.action_span_neg)
            return self.default_joint_pos + actions * span
        return self.joint_limits_min + (actions + 1.0) * 0.5 * self.joint_range

    def targets_to_actions(self, targets: torch.Tensor) -> torch.Tensor:
        """Inverse of actions_to_targets (encoder-space joint target -> action in [-1, 1])."""
        if self.action_map == "residual":
            return torch.clamp((targets - self.default_joint_pos) / self.action_scale, -1.0, 1.0)
        if self.action_map == "centered":
            span = torch.where(targets >= self.default_joint_pos, self.action_span_pos, self.action_span_neg)
            return torch.clamp((targets - self.default_joint_pos) / span, -1.0, 1.0)
        return torch.clamp(2.0 * (targets - self.joint_limits_min) / self.joint_range - 1.0, -1.0, 1.0)

    def _apply_action(self):
        # Action smoothing filter: EMA on PD position targets
        self.filtered_targets = (
            self.action_filter_alpha * self.targets
            + (1.0 - self.action_filter_alpha) * self.filtered_targets
        )
        # The filtered target is in encoder space. A servo with mount offset d drives its
        # encoder there, so the TRUE joint goes to target - d (see mount_offset in __init__).
        self.robot.set_joint_position_target(self.filtered_targets - self.mount_offset)

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
        # previous_actions is advanced at the end of _get_rewards, inside super().step(), so the
        # observation built afterwards carries this step's action (see there).
        obs, rew, terminated, truncated, info = super().step(actions)

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
        
        # Sensor lag. The self.* attributes above stay ground truth -- the reward terms and
        # _get_dones read them -- so the delayed copies live in locals used only for the
        # observation. Overwriting the attributes would silently compute rewards on stale data.
        joint_pos_obs, joint_vel_obs = self.joint_pos, self.joint_vel
        if self.servo_obs_delay > 0:
            self.servo_obs_buffer = torch.roll(self.servo_obs_buffer, shifts=1, dims=1)
            self.servo_obs_buffer[:, 0] = torch.cat([self.joint_pos, self.joint_vel], dim=-1)
            joint_pos_obs, joint_vel_obs = self.servo_obs_buffer[:, self.servo_obs_delay].split(
                self.num_joints, dim=-1
            )

        imu_acc_obs = self.imu_lin_acc_b
        imu_gyro_obs = self.imu_ang_vel_b
        gravity_obs = self.projected_gravity_b
        if self.max_imu_obs_delay > 0:
            self.imu_obs_buffer = torch.roll(self.imu_obs_buffer, shifts=1, dims=1)
            self.imu_obs_buffer[:, 0] = torch.cat(
                [self.imu_lin_acc_b, self.imu_ang_vel_b, self.projected_gravity_b], dim=-1
            )
            imu_delayed = self.imu_obs_buffer[
                torch.arange(self.num_envs, device=self.device), self.imu_obs_delay
            ]
            imu_acc_obs, imu_gyro_obs, gravity_obs = imu_delayed.split(3, dim=-1)

        # Per-episode sensor biases. Unlike the observation noise model (resampled every step,
        # so it averages out over the history stack), these are CONSTANT within an episode --
        # which is what makes them hard, and what real hardware actually does: an IMU has a
        # run-to-run accel/gyro null offset. (Per-servo zero error is the mount offset below,
        # a per-robot constant applied to the command path too.)
        # Applied to the observation copies only; self.* stay ground truth for the rewards.
        # A misaligned board measures every vector in its own rotated frame. The bias is in
        # that same sensor frame, so it goes on after the rotation.
        if self.imu_misalign_max_rad > 0.0:
            _rot = lambda v: torch.bmm(self.imu_rot, v.unsqueeze(-1)).squeeze(-1)
            imu_acc_obs, imu_gyro_obs, gravity_obs = _rot(imu_acc_obs), _rot(imu_gyro_obs), _rot(gravity_obs)
        if self.imu_accel_bias_range > 0.0:
            imu_acc_obs = imu_acc_obs + self.imu_accel_bias
        if self.imu_gyro_bias_range > 0.0:
            imu_gyro_obs = imu_gyro_obs + self.imu_gyro_bias
        # Encoder reads the true angle plus the mount offset (same d the command path removes).
        joint_pos_obs = joint_pos_obs + self.mount_offset

        # Encoder quantisation, applied after the offset because the servo digitises the
        # angle it actually sees, offset included.
        if self.quantize_joint_obs:
            joint_pos_obs = torch.round(joint_pos_obs / self.encoder_step_rad) * self.encoder_step_rad
            joint_vel_obs = torch.round(joint_vel_obs / self.encoder_step_rad) * self.encoder_step_rad

        # Gait phase clock: advance proportional to commanded XY speed. With a gait table the
        # clock is advanced in _get_rewards instead (fixed period), so the reward and the next
        # observation see the same phase. gait_clock_mode=fixed_period (no table) does the same.
        if self.gait_ref is None and not self._fixed_period_clock:
            v_cmd_norm = torch.norm(self.commands[:, :2], dim=1)
            self.gait_phase = (
                self.gait_phase + self.step_dt * 2.0 * math.pi * self.gait_clock_freq * v_cmd_norm
            ) % (2.0 * math.pi)
        phase_obs = torch.stack([torch.sin(self.gait_phase), torch.cos(self.gait_phase)], dim=-1)

        # Proprioceptive observations (48 base + 2 phase = 50)
        obs_proprio = torch.cat([
            imu_acc_obs,
            imu_gyro_obs,
            gravity_obs,
            self.commands,
            (joint_pos_obs - self.default_joint_pos),
            joint_vel_obs,
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

        # Net contact force at each foot, rotated into the body frame and flattened.
        # Body frame is required for two reasons: it makes the feature heading-independent,
        # and the symmetry mirror flips the y-component, which is only a valid sagittal
        # reflection in the body frame (in world frame it would only be correct at yaw 0).
        feet_net_forces = self.contact_sensor.data.net_forces_w[:, self._feet_ids]  # [N, 2, 3]
        quat_expanded = self.base_quat.unsqueeze(1).expand(-1, feet_net_forces.shape[1], -1)
        feet_net_forces_b = math_utils.quat_apply_inverse(
            quat_expanded.reshape(-1, 4), feet_net_forces.reshape(-1, 3)
        ).reshape(self.num_envs, -1)  # [N, 6]
        priv_feet_forces = feet_net_forces_b

        obs_priv = torch.cat([
            priv_lin_vel,
            priv_height,
            priv_feet_forces,
        ], dim=-1)  # [N, 10]

        # Construct Policy Observations
        if self.cfg.policy_has_privileged_info:
            obs_policy = torch.cat([obs_proprio, obs_priv], dim=-1)
        else:
            obs_policy = obs_proprio + torch.randn_like(obs_proprio) * self.obs_noise_std

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

        # With a gait table, advance the clock first: the reward then compares the post-step
        # state with the reference at the post-step phase, and the next observation shows it.
        # The reference frame at that (command, phase) is sampled ONCE here and shared by the
        # linear-velocity tracking term (track_lin_target: reference) and the imitation terms.
        ref = None
        if self.gait_ref is not None:
            self._advance_gait_phase()
            ref = self._sample_reference()
        elif self._fixed_period_clock:
            self._advance_fixed_period_phase()

        # --- Prepare Data ---
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]
        
        # Feet Contact Mask (for sliding and stride)
        # Check current contact force > threshold
        feet_contact_forces = self.contact_sensor.data.net_forces_w[:, self._feet_ids]
        feet_in_contact = torch.norm(feet_contact_forces, dim=-1) > 1.0

        # Foot clearance above the foot's own last in-contact height (the swing_foot_height datum;
        # see section 17). Computed here, before feet_slide, so the reward-side contact flag below
        # can use it. The datum update is driven by the FORCE-based feet_in_contact only, never by
        # the height-aware flag, so the datum cannot freeze on a dragged foot.
        foot_z = feet_pos[:, :, 2]
        fresh = ~self.foot_ground_init
        if fresh.any():                       # first step after reset: seed the datum
            if self.cfg.use_rough_terrain:
                # Unknown local height under the feet: both feet = their current z (a foot that
                # starts mid-swing reads clearance ~0 until it lands).
                self.foot_ground_z[fresh] = foot_z[fresh]
            else:
                # Flat terrain: the ground is at env_origins z, and a foot resting flat has its link
                # origin _flat_foot_origin_z above it. Seeds airborne feet (RSI mid-swing) with
                # their true clearance; a foot in contact is overwritten with its z just below.
                self.foot_ground_z[fresh] = self.scene.env_origins[fresh][:, 2:3] + self._flat_foot_origin_z
            self.foot_ground_init[fresh] = True
        self.foot_ground_z = torch.where(feet_in_contact, foot_z, self.foot_ground_z)
        clearance = foot_z - self.foot_ground_z

        # Reward-side contact flag (imitation_contact, feet_slide only): |F| > 1 N OR clearance <
        # contact_height_thresh_m. A foot dragged with < 1 N load counts as in contact, not airborne.
        # None / 0 = old force-only behaviour. Sensor consumers (stride, air time, COM support,
        # gait_phase_contact, swing_foot_height, foot_flat) keep the force flag.
        # contact_height_mode: hard (default) = the bool flag above; soft = a score in [0, 1],
        # c = max(force_flag, clamp(1 - clearance / thresh, 0, 1)), used as the feet_slide weight
        # on the planar speed and as the imitation_contact value (match = 1 - |c - table|). It
        # removes the cliff at the threshold (3.9 mm = full penalty, 4.1 mm = none).
        _ch_thresh = float(self.cfg.contact_height_thresh_m or 0.0)
        _ch_soft = self.cfg.contact_height_mode == "soft"
        if _ch_soft:
            clear_contact = rewards.clearance_contact_soft(clearance, _ch_thresh)       # float [0, 1]
            feet_contact_rw = torch.maximum(feet_in_contact.float(), clear_contact)
        else:
            clear_contact = rewards.clearance_contact(clearance, _ch_thresh)
            feet_contact_rw = feet_in_contact | clear_contact
        # dragged_foot_frac diagnostic: unloaded, near the ground, sliding. Its definition is
        # unchanged by contact_height_mode: it always uses the HARD threshold (clearance <
        # contact_height_thresh_m). Uses 2 mm when the flag is off so a baseline run logs the same quantity.
        _feet_vel_xy = torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_indices, :2], dim=-1)
        _dragged = (~feet_in_contact) & (clearance < (_ch_thresh if _ch_thresh > 0.0 else 0.002)) & (_feet_vel_xy > 0.05)
        self.dragged_foot_steps += _dragged.float().sum(dim=1)

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
        # track_*_std_rel > 0: kernel width scales with the command (rewards.py *_rel); 0 = fixed std.
        # track_lin_target reference: the kernel is centred on the table's base velocity at (command,
        # phase), the same frame the imitation terms use, and measured on the FRESH robot.data velocity
        # (self.base_lin_vel_b is refreshed in _get_observations, so it is one step stale here). The
        # width still comes from the command. command (default): unchanged, below.
        if self.cfg.track_lin_target == "reference":
            r_track_lin_vel_xy = rewards.track_lin_vel_xy_exp_rel_target(
                self.gait_ref.field(ref, "base_lin_vel_b")[:, :2], self.commands, self.robot.data.root_lin_vel_b,
                std_rel=float(self.cfg.track_lin_std_rel), std_min=float(self.cfg.track_lin_std_min))
        elif self.cfg.track_lin_std_rel > 0.0:
            r_track_lin_vel_xy = rewards.track_lin_vel_xy_exp_rel(
                self.commands, self.base_lin_vel_b,
                std_rel=float(self.cfg.track_lin_std_rel), std_min=float(self.cfg.track_lin_std_min))
        else:
            r_track_lin_vel_xy = rewards.track_lin_vel_xy_exp(self.commands, self.base_lin_vel_b, std=0.2)
        
        # 2. track_ang_vel_z_exp (w=1.5)
        if self.cfg.track_ang_std_rel > 0.0:
            r_track_ang_vel_z = rewards.track_ang_vel_z_exp_rel(
                self.commands, self.base_ang_vel_b,
                std_rel=float(self.cfg.track_ang_std_rel), std_min=float(self.cfg.track_ang_std_min))
        else:
            r_track_ang_vel_z = rewards.track_ang_vel_z_exp(self.commands, self.base_ang_vel_b, std=0.5)
        
        # 3. termination_penalty (w=-10.0)
        # Uses the same helper as _get_dones. Previously this recomputed the test with a
        # different threshold (0.78 vs 0.784) against a different, one-step-stale source
        # (self.projected_gravity_b, the IMU copy refreshed only in _get_observations), so it
        # almost never fired on the step that actually terminated -- the logged episodic value
        # was ~1% of a single firing.
        r_termination = self._tilt_exceeded().float()

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
        if self.cfg.feet_air_time_mode == "touchdown":
            # Paid once, on the step the foot lands. `touchdown` (env force threshold, same 1 N and
            # same net_forces_w as the sensor, evaluated once per control step above) marks the
            # landing; the sensor's last_air_time holds that swing's duration on exactly that
            # step (Isaac Lab updates it at first contact, before current_air_time resets).
            # Other foot "was down for the swing" <=> its contact time covers the whole swing
            # (half-step tolerance for sensor time quantisation). Feet order is [right, left].
            last_air_time = self.contact_sensor.data.last_air_time[:, self._feet_ids]
            other_contact_time = current_contact_time[:, [1, 0]]
            other_in_contact = (other_contact_time > 0.0) & (other_contact_time + 0.5 * self.step_dt >= last_air_time)
            r_feet_air_time = rewards.feet_air_time_touchdown(
                touchdown, last_air_time, other_in_contact, self.commands,
                air_time_max=float(self.cfg.feet_air_time_max_s),
                air_time_min=float(self.cfg.feet_air_time_min_s), min_speed_command_threshold=0.05)
        else:
            r_feet_air_time = rewards.feet_air_time_positive_biped(current_air_time, current_contact_time, self.commands, threshold=0.3, min_speed_command_threshold=0.05)

        # 12. feet_slide
        # force_matrix_w_history is only populated when filter_prim_paths_expr is non-empty
        # (flat terrain). With rough terrain the filter is cleared, so fall back to net_forces_w.
        if self.contact_sensor.data.force_matrix_w_history is not None:
            r_feet_slide = rewards.feet_slide(
                self.contact_sensor.data.force_matrix_w_history[:, :, self._feet_ids, :],
                self.robot.data.body_lin_vel_w[:, self.feet_indices],
                clear_contact if _ch_thresh > 0.0 else None,
            )
        else:
            feet_contact = torch.maximum(
                (torch.norm(self.contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float(),
                clear_contact.float())
            feet_vel_xy  = torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_indices, :2], dim=-1)
            r_feet_slide = rewards.feet_slide_with_vel(feet_contact, feet_vel_xy)

        # 13. undesired_contacts
        if self.contact_sensor.data.force_matrix_w_history is not None:
            net_contact_forces_undesired = self.contact_sensor.data.force_matrix_w_history[:, :, self._undesired_contact_body_ids]
            undesired_forces_norm = torch.norm(net_contact_forces_undesired, dim=-1)  # [N, T, B]
            undesired_forces_max_per_body = torch.max(torch.max(undesired_forces_norm, dim=1)[0], dim=-1)[0]

            # Foot-foot contact. The sensor filter only reports body-vs-ground in force_matrix_w,
            # while net_forces_w contains every contact. So (net - ground) on a foot is whatever
            # else is touching it -- i.e. the other foot (feet are the only bodies excluded above,
            # and legs/hips are already penalised on their own).
            feet_net = self.contact_sensor.data.net_forces_w_history[:, :, self._feet_ids]  # [N, T, F, 3]
            feet_ground = self.contact_sensor.data.force_matrix_w_history[:, :, self._feet_ids].sum(dim=3)  # [N, T, F, 3]
            feet_other_norm = torch.norm(feet_net - feet_ground, dim=-1).max(dim=1)[0]  # [N, F]
            undesired_forces_max_per_body = torch.cat([undesired_forces_max_per_body, feet_other_norm], dim=-1)
        else:
            # Without filter history, use current net forces directly.
            # Keep shape [N, num_undesired_bodies] so undesired_contacts() can apply
            # torch.any(dim=-1) per-environment — do NOT collapse the body dim here.
            undesired_forces_max_per_body = torch.norm(
                self.contact_sensor.data.net_forces_w[:, self._undesired_contact_body_ids], dim=-1
            )  # [N, num_undesired]
        r_undesired_contacts = rewards.undesired_contacts(
            undesired_forces_max_per_body,
            threshold=float(getattr(self.cfg, "undesired_contacts_thresh_n", 1.0)),
            saturation=float(getattr(self.cfg, "undesired_contacts_sat_n", 20.0)),
        )

        # 13b. leg_contact_l2: legs bracing on each other (see _leg_contact_forces for coverage).
        if float(self.cfg.rewards.get("leg_contact_l2", 0.0)) != 0.0:
            r_leg_contact = rewards.leg_contact_l2(
                self._leg_contact_forces(),
                thresh_n=float(self.cfg.leg_contact_thresh_n),
                sat_n=float(self.cfg.leg_contact_sat_n),
            )
        else:
            r_leg_contact = torch.zeros(self.num_envs, device=self.device)
        
        # 13c. stand_pose_l2: distance to the table's stand-frame pose while the reference stands. Reuses the
        # frame sampled once above (ref), fresh robot.data.joint_pos (self.joint_pos is one step stale here).
        if ref is not None and float(self.cfg.rewards.get("stand_pose_l2", 0.0)) != 0.0:
            r_stand_pose = rewards.stand_pose_l2(
                self.robot.data.joint_pos,
                self.gait_ref.joint_field(ref, "joint_pos"),
                self._reference_standing().float(),
            )
        else:
            r_stand_pose = torch.zeros(self.num_envs, device=self.device)

        # 14. joint_deviation_hip (w=-0.2)
        r_joint_deviation_hip = rewards.joint_deviation_l1(
            self.joint_pos[:, self.hip_indices], 
            self.default_joint_pos[self.hip_indices]
        )
        
        # 15. joint_deviation_ankle_roll
        r_joint_deviation_ankle = rewards.joint_deviation_l1(
            self.joint_pos[:, self.ankle_roll_indices],
            self.default_joint_pos[self.ankle_roll_indices]
        )

        # 15b. joint_deviation_ankle_pitch — penalise bent-ankle posture (toe-only stance)
        if len(self.ankle_pitch_indices) > 0:
            r_joint_deviation_ankle_pitch = rewards.joint_deviation_l1(
                self.joint_pos[:, self.ankle_pitch_indices],
                self.default_joint_pos[self.ankle_pitch_indices]
            )
        else:
            r_joint_deviation_ankle_pitch = torch.zeros(self.num_envs, device=self.device)

        # 16. knee deviation — penalise bent knees to encourage upright stance
        r_joint_deviation_knee = rewards.joint_deviation_l1(
            self.joint_pos[:, self.knee_indices],
            self.default_joint_pos[self.knee_indices]
        )

        # 17. step_length (w=0.0 default, set in config)
        r_step_length = rewards.step_length(touchdown, stride_dist, self.commands, min_speed_command_threshold=0.1)

        # 17. swing_foot_height: clearance above the foot's own resting height, command-gated.
        # Datum: the foot-link origin sits ~3.6 cm above the sole, so measuring raw world Z
        # paid a flat-but-unloaded foot ~0.025/step to do nothing (the standing jitter).
        # The resting height is learned from in-contact feet, and the terrain origin is
        # subtracted so patch elevation cannot be farmed either.
        # (foot_ground_z / clearance are updated near the top of _get_rewards, where feet_in_contact
        # is computed: feet_slide and imitation_contact need the clearance too.)
        r_swing_foot_height = rewards.swing_foot_height(
            clearance, feet_in_contact, self.commands,
            min_clearance=float(getattr(self.cfg, "swing_min_clearance", 0.01)),
            max_clearance=float(getattr(self.cfg, "swing_max_clearance", 0.07)),
            min_speed_command_threshold=0.05,
            # anti-farming: no credit once this foot's current swing exceeds swing_max_time_s
            swing_valid=rewards.swing_time_mask(current_air_time, float(self.cfg.swing_max_time_s)),
        )
        # diagnostics: peak clearance per completed swing, contact share per foot
        swinging = ~feet_in_contact
        self.swing_peak = torch.where(swinging, torch.maximum(self.swing_peak, clearance), self.swing_peak)
        self.swing_peak_sum += (self.swing_peak * touchdown.float()).sum(dim=1)
        self.swing_count += touchdown.float().sum(dim=1)
        self.swing_peak = torch.where(touchdown, torch.zeros_like(self.swing_peak), self.swing_peak)
        self.contact_steps += feet_in_contact.float()
        
        # 18: Gait phase contact — reward feet in sync with clock to enforce alternation
        r_gait_phase_contact = rewards.gait_phase_contact(
            self.gait_phase, feet_in_contact, self.commands, min_speed=0.05
        )

        # 19: Torso centering (base pos proxy — kept for backward compat, set weight to 0 to disable)
        r_torso_centering = rewards.torso_centering_reward(self.robot.data.root_pos_w, feet_pos)

        # 19b: COM support centering (true mass-weighted COM vs. contacting feet centroid)
        com_offset = rewards.com_support_offset(
            self.robot.data.body_com_pose_w,
            self.robot.data.default_mass.to(self.device),
            feet_pos,
            feet_in_contact,
        )
        com_dist = rewards.com_support_distance(
            self.robot.data.body_com_pose_w,
            self.robot.data.default_mass.to(self.device),
            feet_pos,
            feet_in_contact,
            foot_margin=float(getattr(self.cfg, "com_foot_margin", 0.03)),
        )
        r_com_support_centering = rewards.com_support_centering_reward(
            com_dist, std=max(float(getattr(self.cfg, "com_centering_std", 0.04)), 1e-3))
        diag = self.gait_diagnostics(com_offset, com_dist)
        # Footprint gap of the two feet from the FRESH articulation state (same source as _get_dones); one
        # evaluation per step, shared by the penalty and the diagnostics.
        foot_gap = self._foot_footprint_gap()
        diag["foot_gap_mm"] = foot_gap * 1000.0
        diag["foot_overlap_frac"] = (foot_gap <= 0.0).float()
        for k, v in diag.items():
            valid = ~torch.isnan(v)
            self.diag_sums[k] += torch.where(valid, v, torch.zeros_like(v))
            self.diag_counts[k] += valid.float()

        # 19: Knee bend during swing on touchdown
        r_knee_bend_touchdown = rewards.knee_bend_on_touchdown(touchdown, self.knee_bend_max, min_bend=0.2, max_bend=1.0)
        self.knee_bend_max[touchdown] = 0.0

        # 20: Ankle pitch torque penalty — discourages toe-only stance (plantarflexion bias)
        if len(self.ankle_pitch_indices) > 0:
            r_ankle_torques = rewards.joint_torques_l2(self.joint_efforts[:, self.ankle_pitch_indices])
        else:
            r_ankle_torques = torch.zeros(self.num_envs, device=self.device)

        # upright: positive reward for staying vertical (complements flat_orientation_l2)
        r_upright = rewards.upright(self.projected_gravity_b, std=0.3)

        # angular_momentum: mass-weighted body angular momentum to discourage limb wobble
        r_angular_momentum = rewards.angular_momentum_l2(
            self.robot.data.body_ang_vel_w,
            self.robot.data.default_mass.to(self.device),
        )

        # 22: action_l2 — NOTE: kept for backward compat with config keys, but weight should be 0.0
        # (see plan: action_l2 is structurally broken due to asymmetric joint limits)
        r_action_l2 = rewards.action_l2(self.actions)

        # 24: dof_pos_l2 — correct all-joint pose regulariser (joint-space, not action-space)
        r_dof_pos_l2 = rewards.dof_pos_l2(self.joint_pos, self.default_joint_pos)

        # 25: base_height_l2 — penalise crouching (below target) and tiptoe (above). The target
        # is cfg.target_base_height (0.29 m; follows spawn_height_m when env_config sets it).
        r_base_height = torch.square(
            self.robot.data.root_pos_w[:, 2] - getattr(self.cfg, "target_base_height", 0.29)
        )

        # 26: foot_separation. Measured along body +Y (URDF-verified lateral axis) so it is
        # heading-independent; the previous world-Y version read ~0 whenever the robot faced
        # world +-Y.
        # origin_one_sided (old): the hip joints are +-21.7 mm apart (the 0.0433 m the policy once
        # settled at is 2 * 0.021672, i.e. hip_roll driven to zero; it is NOT a foot-origin
        # distance). The measured point is the foot LINK ORIGIN (ankle_pitch joint, 31.6 mm above
        # the sole, 2 mm inside the medial edge), 64.7 mm apart at the home pose, and min_sep
        # 0.09 is clamp-one-sided, so it only ever pushes feet apart and barely sees ankle roll.
        # sole_band: two-sided band on the SOLE-CENTRE separation (centres are 106.9 mm apart at
        # home vs the 105 mm hip spacing); see rewards.foot_separation_sole_band_penalty.
        feet_quat = self.robot.data.body_quat_w[:, self.feet_indices]
        if self.cfg.foot_separation_mode == "sole_band":
            r_foot_separation = rewards.foot_separation_sole_band_penalty(
                feet_pos, feet_quat, self.robot.data.root_quat_w,
                target=float(self.cfg.foot_sep_target_m), tol=float(self.cfg.foot_sep_tol_m),
                scale=float(self.cfg.foot_sep_scale_m),
                sole_x=self._foot_sole_offset[0], sole_lat=self._foot_sole_offset[1],
                sole_z=self._foot_sole_offset[2],
            )
        else:
            r_foot_separation = rewards.foot_separation_penalty(
                feet_pos, self.robot.data.root_quat_w, min_sep=0.09
            )

        # 26b: foot_proximity (E21) -- clamp((margin - gap)/margin, 0, 2)^2 on the horizontal-plane gap between the
        # sole rectangles. Terrain-independent: needs no contact sensor, so it also covers the feet on rough terrain.
        r_foot_proximity = rewards.foot_proximity_penalty(foot_gap, self._foot_prox_margin)

        # 27: foot_flat_l2 -- roll tilt of each foot in contact (sin^2 of the sole normal's
        # lateral component). No term saw sole flatness before, so the policy stood on the
        # inner/outer foot edges with ankle_roll saturated.
        r_foot_flat = rewards.foot_flat_roll_l2(
            feet_quat, self.robot.data.root_quat_w, feet_in_contact
        )

        # 23: hip pitch deviation — mild penalty allows walking but discourages lean/crouch
        if len(self.hip_pitch_indices) > 0:
            r_joint_deviation_hip_pitch = rewards.joint_deviation_l1(
                self.joint_pos[:, self.hip_pitch_indices],
                self.default_joint_pos[self.hip_pitch_indices]
            )
        else:
            r_joint_deviation_hip_pitch = torch.zeros(self.num_envs, device=self.device)

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
            "joint_deviation_ankle_pitch": r_joint_deviation_ankle_pitch * self.cfg.rewards.get("joint_deviation_ankle_pitch", 0.0),
            "joint_deviation_knee": r_joint_deviation_knee * self.cfg.rewards.get("joint_deviation_knee", 0.0),
            "step_length": r_step_length * self.cfg.rewards.get("step_length", 0.0),
            "swing_foot_height": r_swing_foot_height * self.cfg.rewards.get("swing_foot_height", 0.0),
            "torso_centering": r_torso_centering * self.cfg.rewards.get("torso_centering", 0.0),
            "com_support_centering": r_com_support_centering * self.cfg.rewards.get("com_support_centering", 0.0),
            "gait_phase_contact": r_gait_phase_contact * self.cfg.rewards.get("gait_phase_contact", 0.0),
            "knee_bend_touchdown": r_knee_bend_touchdown * self.cfg.rewards.get("knee_bend_touchdown", 0.0),
            "track_joint_pos_exp": r_track_joint_pos * self.cfg.rewards.get("track_joint_pos_exp", 0.0),
            "track_joint_vel_exp": r_track_joint_vel * self.cfg.rewards.get("track_joint_vel_exp", 0.0),
            "ankle_torques_l2": r_ankle_torques * self.cfg.rewards.get("ankle_torques_l2", 0.0),
            "action_l2": r_action_l2 * self.cfg.rewards.get("action_l2", 0.0),
            "joint_deviation_hip_pitch": r_joint_deviation_hip_pitch * self.cfg.rewards.get("joint_deviation_hip_pitch", 0.0),
            "dof_pos_l2": r_dof_pos_l2 * self.cfg.rewards.get("dof_pos_l2", 0.0),
            "base_height_l2": r_base_height * self.cfg.rewards.get("base_height_l2", 0.0),
            "foot_separation": r_foot_separation * self.cfg.rewards.get("foot_separation", 0.0),
            "foot_proximity": r_foot_proximity * self.cfg.rewards.get("foot_proximity", 0.0),
            "foot_flat_l2": r_foot_flat * self.cfg.rewards.get("foot_flat_l2", 0.0),
            "leg_contact_l2": r_leg_contact * self.cfg.rewards.get("leg_contact_l2", 0.0),
            "stand_pose_l2": r_stand_pose * self.cfg.rewards.get("stand_pose_l2", 0.0),
            "upright": r_upright * self.cfg.rewards.get("upright", 0.0),
            "angular_momentum": r_angular_momentum * self.cfg.rewards.get("angular_momentum", 0.0),
        }
        
        if self.gait_ref is not None:
            for name, term in self._imitation_terms(feet_pos, feet_contact_rw, clearance, ref).items():
                reward_terms[name] = term * self.cfg.rewards.get(name, 0.0)

        # Weighted sum
        total_reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
        
        # Apply global scaling
        if hasattr(self.cfg, "reward_scale"):
           total_reward *= self.cfg.reward_scale
        
        # Logging
        for key, value in reward_terms.items():
            if key in self.episode_sums:
                self.episode_sums[key] += value

        # Torque headroom. Read applied_torque fresh rather than self.joint_efforts, which is
        # assigned in _get_observations and therefore one step stale at this point.
        _abs_tau = self.robot.data.applied_torque.abs()
        self.torque_abs_sum += _abs_tau.mean(dim=1)
        if self.bam_actuators:
            _act = self.bam_actuators[0]
            self.torque_sat_sum += _act.duty_saturated.float().mean(dim=1)
            self.tau_ext_abs_sum += _act.tau_ext.abs().mean(dim=1)
            self.friction_abs_sum += _act.servo.tau_friction.abs().mean(dim=1)
        else:
            self.torque_sat_sum += (_abs_tau > 0.95 * self.actuator_effort_limit_ref).float().mean(dim=1)
        self.episode_step_count += 1.0

        # Unclamped body-frame lateral foot separation, for calibrating min_sep.
        _q = self.robot.data.root_quat_w
        _body_y = torch.stack([
            2.0 * (_q[:, 1] * _q[:, 2] - _q[:, 0] * _q[:, 3]),
            1.0 - 2.0 * (_q[:, 1] * _q[:, 1] + _q[:, 3] * _q[:, 3]),
            2.0 * (_q[:, 2] * _q[:, 3] + _q[:, 0] * _q[:, 1]),
        ], dim=-1)
        self.foot_sep_sum += torch.abs(
            torch.sum((feet_pos[:, 0, :] - feet_pos[:, 1, :]) * _body_y, dim=-1)
        )
        self.foot_sole_sep_sum += rewards.foot_sole_centre_lateral_sep(
            feet_pos, feet_quat, self.robot.data.root_quat_w,
            self._foot_sole_offset[0], self._foot_sole_offset[1], self._foot_sole_offset[2],
        )

        if self.motion_time is not None:
            self.motion_time += self.step_dt

        # Advance the action history HERE, after action_rate_l2 has used a_{t-1} and before
        # _reset_idx / _get_observations run. The next observation then carries a_t, which is
        # what the Jetson feeds and what next step's action_rate penalty is measured against.
        # (It used to be advanced in step(), after the observation was built, so the policy saw
        # a_{t-2} and could never observe the difference it was penalised for.) Reset envs are
        # re-seeded by _reset_idx, which runs after this.
        self.previous_previous_actions = self.previous_actions.clone()
        self.previous_actions = self.actions.clone()

        return total_reward

    # ------------------------------------------------------------------ gait-table imitation
    def _reference_standing(self) -> torch.Tensor:
        """Envs whose command is below the table's stand threshold: stand frame, clock held."""
        thr = self.cfg.reference_stand_threshold
        return (torch.norm(self.commands[:, :2], dim=1) < thr) & (self.commands[:, 2].abs() < thr)

    def _advance_fixed_period_phase(self) -> None:
        """gait_clock_mode=fixed_period without a table: constant rate, held while standing.
        Same stand rule as the table clock (|v_xy| and |wz| both below the threshold)."""
        thr = self.cfg.gait_clock_stand_threshold
        standing = (torch.norm(self.commands[:, :2], dim=1) < thr) & (self.commands[:, 2].abs() < thr)
        self.gait_phase = (
            self.gait_phase + (~standing).float() * self.step_dt * 2.0 * math.pi / self.cfg.gait_clock_period_s
        ) % (2.0 * math.pi)

    def _advance_gait_phase(self) -> None:
        """Table clock: 2*pi*freq(command) per second, held while standing. A v2 table's cadence
        follows the command (Gait Studio's cadence law); freq is trilinear in the command like
        the table itself, so the clock and the reference always describe the same cycle."""
        moving = (~self._reference_standing()).float()
        freq = self.gait_ref.frequency(self.commands)
        self.gait_phase = (self.gait_phase + moving * self.step_dt * 2.0 * math.pi * freq) % (2.0 * math.pi)

    def _leg_contact_forces(self) -> torch.Tensor:
        """(N, B) peak |force| [N] over the sensor history on the bodies leg_contact_l2 watches.

        Contact sensor: prim_path Robot/Robot/.* (every link), so net_forces_w holds the sum of
        ALL contacts on a body, floor and other robot links alike (self-collisions are enabled in
        the asset). filter_prim_paths_expr is the ground plane only, so force_matrix_w is the
        floor part alone, which is why the old undesired_contacts term (it reads force_matrix_w
        for non-foot bodies) cannot see leg-leg contact.
          * non-foot leg bodies (hip/thigh/calf/ankle): the net force. The floor never touches
            them while standing, so any force is the other leg (or a fall) -- always undesired.
          * feet (flat terrain only, i.e. when force_matrix_w exists): net - floor = whatever else
            touches the foot (the other foot), the construction undesired_contacts already used.
            With rough terrain the filter is cleared (biped_env_cfg.__post_init__), the floor
            force cannot be removed from a foot's net force, and the feet are left OUT.
        """
        cs = self.contact_sensor.data
        net_hist = cs.net_forces_w_history                                      # [N, T, B, 3]
        forces = torch.norm(net_hist[:, :, self._leg_body_ids], dim=-1).max(dim=1)[0]
        if cs.force_matrix_w_history is not None:
            feet_net = net_hist[:, :, self._feet_ids]
            feet_ground = cs.force_matrix_w_history[:, :, self._feet_ids].sum(dim=3)
            feet_other = torch.norm(feet_net - feet_ground, dim=-1).max(dim=1)[0]
            forces = torch.cat([forces, feet_other], dim=-1)
        return forces

    def _sample_reference(self) -> torch.Tensor:
        """Table frame [N, D] at the current (command, phase); stored in self.ref_frame."""
        ref = self.gait_ref.sample(self.commands, self.gait_phase / (2.0 * math.pi), self._reference_standing())
        self.ref_frame = ref
        return ref

    def _imitation_terms(self, feet_pos: torch.Tensor, feet_in_contact: torch.Tensor,
                         clearance: torch.Tensor, ref: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Eight imitation terms against the table at (command, phase). Unweighted, in [0, 1].

        Reads robot.data directly: the self.joint_pos etc. caches are refreshed in
        _get_observations, so they are one step stale here. Env foot order is [right, left]; the
        table's is [left, right].

        feet_in_contact is the reward-side flag (|F| > 1 N or clearance < contact_height_thresh_m;
        contact_height_mode soft: the float score in [0, 1]). imitation_foot_agg picks mean | min
        over the two feet for imitation_foot_height and imitation_contact.
        clearance (N, 2, env order) is the swing_foot_height datum quantity; the table's
        foot_height is the sole height above the ground (stance baseline 0.01-0.06 mm, far below
        the 4 mm std, so it is used as is).
        ref: the frame _get_rewards already sampled for this step (shared with the tracking term);
        None samples it here."""
        g, d, std = self.gait_ref, self.robot.data, self._ref_std
        _agg_min = self.cfg.imitation_foot_agg == "min"
        _soft = self.cfg.contact_height_mode == "soft"
        if ref is None:
            ref = self._sample_reference()

        rel = (feet_pos - d.root_pos_w.unsqueeze(1))[:, [1, 0]]                      # [N, 2, 3] world, [left, right]
        quat = d.root_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        feet_b = math_utils.quat_apply_inverse(quat, rel.reshape(-1, 3)).reshape(self.num_envs, 6)

        return {
            "imitation_joint_pos": rewards.imitation_mse_exp(d.joint_pos, g.joint_field(ref, "joint_pos"), std["joint_pos"]),
            "imitation_joint_vel": rewards.imitation_mse_exp(d.joint_vel, g.joint_field(ref, "joint_vel"), std["joint_vel"]),
            "imitation_foot_pos": rewards.imitation_mse_exp(feet_b, g.field(ref, "foot_pos_b"), std["foot_pos"]),
            "imitation_contact": rewards.imitation_contact_match(feet_in_contact[:, [1, 0]], g.field(ref, "contact"), _soft, _agg_min),
            "imitation_orientation": rewards.imitation_mse_exp(d.projected_gravity_b, g.field(ref, "projected_gravity"), std["orientation"]),
            "imitation_lin_vel": rewards.imitation_mse_exp(d.root_lin_vel_b, g.field(ref, "base_lin_vel_b"), std["lin_vel"]),
            "imitation_ang_vel": rewards.imitation_mse_exp(d.root_ang_vel_b, g.field(ref, "base_ang_vel_b"), std["ang_vel"]),
            # Sole height above the ground vs the table's. imitation_foot_pos only sees the foot
            # origin in the base frame (an 11 mm swing-height miss costs ~1% of it). env order is
            # flipped to the table's [left, right] inside the function.
            "imitation_foot_height": rewards.imitation_foot_height(clearance, g.field(ref, "foot_height"), std["foot_height"], _agg_min),
        }

    def _reference_state_init(self, env_ids: torch.Tensor) -> None:
        """Reference state initialisation: start a fraction of resets mid-gait on the table.

        Must run after _resample_commands (needs the new command and the random phase drawn
        in _reset_idx). Keeps the position and yaw the reset event drew; takes joints, base
        height, roll/pitch and velocities from the reference.

        BAM servo state is NOT touched here: its reset is deferred and the first motor() call
        seeds q_smooth, the command-delay ring and the back-EMF state from the pose and
        velocity written below, so joints must be written before that call (they are).
        """
        prob = float(self.cfg.reference_rsi_prob)
        if prob <= 0.0:
            return
        ids = env_ids[torch.rand(len(env_ids), device=self.device) < prob]
        n = len(ids)
        if n == 0:
            return
        g = self.gait_ref
        ref = g.sample(self.commands[ids], self.gait_phase[ids] / (2.0 * math.pi), self._reference_standing()[ids])

        noise = (torch.rand(n, self.num_joints, device=self.device) * 2.0 - 1.0) * self.cfg.reference_rsi_joint_noise
        q = torch.clamp(g.joint_field(ref, "joint_pos") + noise, self.joint_limits_min, self.joint_limits_max)
        qd = g.joint_field(ref, "joint_vel")
        self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)

        # Base: keep xy/yaw from the reset event; roll/pitch from the reference gravity vector.
        # For R = Rz(yaw) Ry(pitch) Rx(roll), R^T (0, 0, -1) = (sin p, -sin r cos p, -cos r cos p).
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w[ids])
        gb = g.field(ref, "projected_gravity")
        pitch = torch.asin(gb[:, 0].clamp(-1.0, 1.0))
        roll = torch.atan2(-gb[:, 1], -gb[:, 2])
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        pos = self.robot.data.root_pos_w[ids].clone()
        pos[:, 2] = self.scene.env_origins[ids, 2] + g.field(ref, "base_height")[:, 0] + 0.005
        self.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
        lin_w = math_utils.quat_apply(quat, g.field(ref, "base_lin_vel_b"))
        ang_w = math_utils.quat_apply(quat, g.field(ref, "base_ang_vel_b"))
        self.robot.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1), env_ids=ids)

        # Re-seed what _reset_idx derived from the pre-RSI pose. q / qd are what was written.
        # Encoder space: the first commanded target must equal the pose or the slew limiter
        # walks the joints away from it.
        self.filtered_targets[ids] = q + self.mount_offset[ids]
        self.action_delay_buffer[ids] = self.filtered_targets[ids].unsqueeze(1)
        self.servo_obs_buffer[ids] = torch.cat([q, qd], dim=-1).unsqueeze(1)
        a_ref = self.targets_to_actions(self.filtered_targets[ids])
        self.previous_actions[ids] = a_ref
        self.previous_previous_actions[ids] = a_ref
        self.feet_in_contact_prev[ids] = g.field(ref, "contact")[:, [1, 0]] > 0.5

    def _randomize_motion_time(self, env_ids: torch.Tensor) -> None:
        if self.motion_ref is None or self.motion_time is None:
            return
        self.motion_time[env_ids] = torch.rand(len(env_ids), device=self.device) * self.motion_ref.duration

    def gait_diagnostics(self, com_offset_xy: torch.Tensor,
                         com_dist: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Per-env gait-quality numbers for this step (see the table in __init__).

        com_ahead_m      CoM minus support centre, along the yaw-only heading (+ = ahead)
        com_lateral_abs_m  |same, across the heading|
        com_outside_mm   distance the CoM lies OUTSIDE the support polygon (0 = inside);
                         this is what com_support_centering actually scores
        lean_fwd_deg     base pitch, + = nose down. For pitch theta about body +Y (REP-103,
                         + is nose-down), body X in world is (cos, 0, -sin), so projected
                         gravity's x = x_b . (0,0,-1) = sin(theta).
        toe_in_deg       mean of -q_hip_yaw over both legs. l_hip_yaw turns about +z, so +q
                         swings the left toes toward +y (outward); r_hip_yaw turns about -z,
                         so +q swings the right toes toward -y (outward). Toe-in is -q for
                         both, which is also why the mirror sign is +1.

        vel_err_moving_ms / yaw_err_moving_rads are the HONEST command-following numbers:
        raw error in physical units, over commanded-to-move envs only. Reward_Terms
        /track_*_exp cannot answer "does it follow commands": it is an exp kernel averaged
        over all envs, and a standing env scores ~1.0 for free. At rel_standing_envs 0.25
        that inflated the headline ~22% -- run C scored ABOVE the walking DC run on
        track_lin_vel_xy_exp while actually tracking slightly worse.
        """
        q = self.robot.data.root_quat_w
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        g = self.robot.data.projected_gravity_b

        # Honest tracking error, moving envs only. nan for standing envs so the episodic
        # mean ignores them entirely rather than averaging in a free 0.
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.05
        turning = self.commands[:, 2].abs() > 0.05
        vel_err = torch.norm(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_err = (self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]).abs()
        nan = torch.full_like(vel_err, float("nan"))
        if com_dist is None:                      # callers that only want the angles
            com_dist = torch.zeros_like(vel_err)
        return {
            "com_ahead_m": (com_offset_xy * fwd).sum(dim=-1),
            "com_lateral_abs_m": (com_offset_xy * left).sum(dim=-1).abs(),
            "com_outside_mm": com_dist * 1000.0,
            "lean_fwd_deg": torch.rad2deg(torch.asin(g[:, 0].clamp(-1.0, 1.0))),
            "toe_in_deg": torch.rad2deg(-self.robot.data.joint_pos[:, self.hip_yaw_lr].mean(dim=1)),
            "vel_err_moving_ms": torch.where(moving, vel_err, nan),
            "yaw_err_moving_rads": torch.where(turning, yaw_err, nan),
        }

    def _tilt_exceeded(self) -> torch.Tensor:
        """Shared tilt-past-limit test for the termination AND its reward penalty.

        Both must read the SAME source at the SAME freshness. _get_rewards runs before
        _get_dones within a step, and self.projected_gravity_b is only refreshed in
        _get_observations (which runs last), so reading that attribute here would be one step
        stale -- and would therefore miss the exact step on which the episode terminates,
        which is precisely when the penalty needs to fire. Read the articulation directly.

        The clamp guards acos: projected_gravity_b can land a hair outside [-1, 1] on
        normalisation round-off, and acos would return NaN, which silently compares False.
        """
        limit_angle = 0.784  # ~45 degrees
        cos_tilt = (-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
        return torch.acos(cos_tilt).abs() > limit_angle

    def _foot_footprint_gap(self) -> torch.Tensor:
        """(N,) signed horizontal-plane gap [m] between the two foot sole rectangles (negative = overlapping).

        Reads robot.data.body_pos_w / body_quat_w directly (fresh after the physics step, like _tilt_exceeded), so
        the penalty, the diagnostics and the overlap termination all see the same state. self.feet_indices is
        [right, left]; rewards.foot_footprint_gap takes (left, right)."""
        pos = self.robot.data.body_pos_w[:, self.feet_indices]
        quat = self.robot.data.body_quat_w[:, self.feet_indices]
        return rewards.foot_footprint_gap(
            pos[:, 1], quat[:, 1], pos[:, 0], quat[:, 0],
            list(self._foot_sole_offset), list(self._foot_sole_half))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self._tilt_exceeded()
        # Terminate on NaN physics — prevents replay buffer corruption
        nan_state = (
            torch.isnan(self.robot.data.joint_pos).any(dim=-1) |
            torch.isnan(self.robot.data.joint_vel).any(dim=-1) |
            torch.isnan(self.robot.data.root_pos_w).any(dim=-1)
        )
        died = died | nan_state
        # Optional foot-overlap termination (E21, off by default). DirectRLEnv.step runs _get_dones BEFORE
        # _get_rewards, both after the physics step, so this is the one place the counter advances.
        if self._foot_overlap_steps > 0:
            overlap = self._foot_footprint_gap() <= 0.0
            self.foot_overlap_count = torch.where(
                overlap, self.foot_overlap_count + 1, torch.zeros_like(self.foot_overlap_count))
            overlap_done = self.foot_overlap_count >= self._foot_overlap_steps
            self._foot_overlap_term = overlap_done & ~died
            died = died | overlap_done
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
        # _apply_curriculum runs at the TOP of this method and writes Curriculum/* into
        # extras["log"], so the unconditional wipe below used to destroy them before they were
        # ever logged. Carry them across explicitly.
        _curriculum_log = {
            k: v for k, v in self.extras.get("log", {}).items() if k.startswith("Curriculum/")
        }
        self.extras["log"] = dict()
        self.extras["log"].update(_curriculum_log)
        self.extras["log"].update(extras)
        extras_term = dict()
        _ov_term = self._foot_overlap_term[env_ids] & self.reset_terminated[env_ids]
        extras_term["Episode_Termination/base_contact"] = (
            torch.count_nonzero(self.reset_terminated[env_ids]) - torch.count_nonzero(_ov_term)).item()
        extras_term["Episode_Termination/foot_overlap"] = torch.count_nonzero(_ov_term).item()
        extras_term["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # Torque headroom. Normalised by ACTUAL episode length (unlike Reward_Terms, which
        # divides by max length), so these read directly against the servo's continuous rating.
        _steps = self.episode_step_count[env_ids].clamp(min=1.0)
        extras_term["Torque/mean_abs"] = torch.mean(self.torque_abs_sum[env_ids] / _steps).item()
        extras_term["Torque/frac_saturated"] = torch.mean(self.torque_sat_sum[env_ids] / _steps).item()
        if self.bam_actuators:
            extras_term["Actuator/mean_abs_tau_ext"] = torch.mean(self.tau_ext_abs_sum[env_ids] / _steps).item()
            extras_term["Actuator/mean_abs_friction"] = torch.mean(self.friction_abs_sum[env_ids] / _steps).item()
        extras_term["Diagnostics/foot_sep_body_m"] = torch.mean(self.foot_sep_sum[env_ids] / _steps).item()
        extras_term["Diagnostics/foot_sole_sep_body_m"] = torch.mean(self.foot_sole_sep_sum[env_ids] / _steps).item()
        for k, v in self.diag_sums.items():
            # Average over the steps this key was valid, not over every step: the
            # *_moving_* keys are undefined while an env is commanded to stand.
            cnt = self.diag_counts[k][env_ids]
            per_env = v[env_ids] / cnt.clamp(min=1.0)
            if cnt.sum() > 0:
                extras_term[f"Diagnostics/{k}"] = per_env[cnt > 0].mean().item()
            v[env_ids] = 0.0
            self.diag_counts[k][env_ids] = 0.0
        _swings = self.swing_count[env_ids]
        if _swings.sum() > 0:
            extras_term["Diagnostics/swing_clearance_mm"] = (
                1000.0 * self.swing_peak_sum[env_ids].sum() / _swings.sum()).item()
        _cf = self.contact_steps[env_ids] / _steps.unsqueeze(-1)
        extras_term["Diagnostics/contact_asym"] = torch.mean((_cf[:, 0] - _cf[:, 1]).abs()).item()
        # fraction of foot-steps with |F| <= 1 N, clearance < thresh, moving > 0.05 m/s
        extras_term["Diagnostics/dragged_foot_frac"] = torch.mean(
            self.dragged_foot_steps[env_ids] / (2.0 * _steps)).item()
        self.dragged_foot_steps[env_ids] = 0.0
        self.swing_peak[env_ids] = 0.0
        self.swing_peak_sum[env_ids] = 0.0
        self.swing_count[env_ids] = 0.0
        self.contact_steps[env_ids] = 0.0

        # Whole-run scalars for mass / COM drift. com_drift_max_m is the one that matters: with
        # a correct (non-accumulating) randomizer it should sit flat near the configured range;
        # if it climbs steadily, something is compounding again.
        _masses_now = self.robot.root_physx_view.get_masses().to(self.device)
        _coms_now = self.robot.root_physx_view.get_coms().to(self.device)
        _coms_nom = self.robot._nominal_coms.to(self.device)
        _com_drift = torch.norm(_coms_now[..., :3] - _coms_nom[..., :3], dim=-1)
        extras_term["Diagnostics/total_mass_mean_kg"] = _masses_now.sum(dim=1).mean().item()
        extras_term["Diagnostics/com_drift_max_m"] = _com_drift.max().item()
        extras_term["Diagnostics/com_drift_mean_m"] = _com_drift.mean().item()

        self.torque_abs_sum[env_ids] = 0.0
        self.torque_sat_sum[env_ids] = 0.0
        self.tau_ext_abs_sum[env_ids] = 0.0
        self.friction_abs_sum[env_ids] = 0.0
        self.foot_sep_sum[env_ids] = 0.0
        self.foot_sole_sep_sum[env_ids] = 0.0
        self.foot_overlap_count[env_ids] = 0
        self._foot_overlap_term[env_ids] = False
        self.episode_step_count[env_ids] = 0.0

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

        # Reset action filter to the current ENCODER reading (filtered_targets live in encoder
        # space; _apply_action subtracts mount_offset), so the commanded true target equals
        # the true joint position and there is no reset transient.
        self.filtered_targets[env_ids] = self.robot.data.joint_pos[env_ids] + self.mount_offset[env_ids]

        # Reset delay buffer and sample new per-env delay
        self.action_delay_buffer[env_ids] = 0.0
        if self.max_action_delay > 0:
            lo, hi = getattr(self.cfg, 'action_delay_steps_range', [0, 0])
            hi_now = int(lo) + int(round((int(hi) - int(lo)) * self._dr_ramp("action_delay")))
            self.action_delay[env_ids] = torch.randint(
                int(lo), hi_now + 1, (len(env_ids),), device=self.device
            )

        # Servo-stiffness DR: per-episode firmware P IN SIM. The stall cap is applied by raising
        # R, which halves the model's small-signal stiffness (17 vs 37 N*m/rad), and the bench
        # validated the uncapped fit, so the real servo is likely stiffer than the capped model.
        # One value per robot (the model error is systematic). The robot still runs P = kp_fw.
        # Ramped by the curriculum ("servo_stiffness"): drawn from scratch at full width, this and
        # the action delay together made walking too hard to discover (2026-10-12 run stood still).
        if self.bam_actuators and getattr(self.cfg, "enable_physics_randomization", False):
            for act in self.bam_actuators:
                rng = getattr(act.cfg, "kp_fw_range", None)
                if not rng:
                    continue
                k = act.servo.kernel
                lo, hi = float(rng[0]), float(rng[1])
                hi_now = lo + (hi - lo) * self._dr_ramp("servo_stiffness")
                k.kp[env_ids] = torch.empty(len(env_ids), 1, device=k.kp.device, dtype=k.kp.dtype).uniform_(
                    lo, hi_now).expand(len(env_ids), k.num_joints)

        # Seed the servo lag buffer with the post-reset encoder reading in every slot, so a
        # freshly reset env doesn't observe pre-reset joint state for its first few steps.
        self.servo_obs_buffer[env_ids] = torch.cat(
            [self.robot.data.joint_pos[env_ids], self.robot.data.joint_vel[env_ids]], dim=-1
        ).unsqueeze(1)
        # The Imu sensor is lazily updated and may still hold the pre-reset step here, so zero
        # rather than seeding from a stale read.
        self.imu_obs_buffer[env_ids] = 0.0
        if self.max_imu_obs_delay > 0:
            self.imu_obs_delay[env_ids] = torch.randint(
                self.imu_obs_delay_range[0], self.imu_obs_delay_range[1] + 1,
                (len(env_ids),), device=self.device
            )

        # Redraw the per-episode sensor biases. Held constant for the whole episode.
        _n = len(env_ids)
        if self.imu_accel_bias_range > 0.0:
            self.imu_accel_bias[env_ids] = (
                torch.empty(_n, 3, device=self.device).uniform_(-1.0, 1.0) * self.imu_accel_bias_range
            )
        if self.imu_gyro_bias_range > 0.0:
            self.imu_gyro_bias[env_ids] = (
                torch.empty(_n, 3, device=self.device).uniform_(-1.0, 1.0) * self.imu_gyro_bias_range
            )
        # (mount_offset is a per-robot constant drawn once in __init__, not redrawn here.)

        # Reset feet contact history to True.
        self.feet_in_contact_prev[env_ids] = True
        # Swing datum re-seeds from the post-reset foot height on the next reward step.
        self.foot_ground_init[env_ids] = False

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

        # Last: RSI needs the new commands and phase, and re-seeds the buffers written above.
        if self.gait_ref is not None:
            self._reference_state_init(env_ids)

    def _get_curriculum_scales(self) -> dict[str, float]:
        step = float(self.common_step_counter)
        dr_start = float(self.cfg.curriculum_dr_start_steps)
        dr_full  = float(self.cfg.curriculum_dr_full_steps)
        dr_scale = max(0.0, min(1.0, (step - dr_start) / max(1.0, dr_full - dr_start)))
        # A ramp of 0 means "full from the start". step / max(1, 0) is 0 at step 0, so the first
        # reset used to get the curriculum START range (vx 0-0.2, no vy/wz) instead.
        _ramp = lambda n: 1.0 if float(n) <= 0.0 else min(1.0, step / float(n))
        cmd_scale  = _ramp(self.cfg.curriculum_cmd_ramp_steps)
        init_scale = _ramp(self.cfg.curriculum_init_ramp_steps)
        return {"dr": dr_scale, "cmd": cmd_scale, "init": init_scale}

    def _dr_ramp(self, name: str) -> float:
        """Curriculum scale (0..1) for a DR term ramped in code, e.g. "action_delay".

        Same rule as the event-based terms: ramped when the curriculum is on and `name` is in
        curriculum_dr_events (or that list is empty); otherwise 1.0, i.e. full width from step 0.
        """
        if not getattr(self.cfg, "curriculum_enabled", False):
            return 1.0
        listed = getattr(self.cfg, "curriculum_dr_events", [])
        if listed and name not in listed:
            return 1.0
        return self._get_curriculum_scales()["dr"]

    def _apply_schedules(self) -> None:
        """Linear ramps that are independent of the DR curriculum switch.

        Both are in common_step_counter units (control steps), like the DR curriculum. A ramp
        of 0 jumps straight to the end value.
          * stand-still fraction: no stillness reward is used; standing is taught purely by
            how often a zero command is drawn (reference: 2% -> 25%).
          * action_rate_l2 weight: the main sim2real smoothness regulariser, started small so
            it doesn't freeze early exploration.
        """
        step = float(self.common_step_counter)
        if getattr(self.cfg, "standing_envs_end", None) is not None:
            f = min(1.0, step / max(1.0, float(self.cfg.standing_envs_ramp_steps)))
            rel = self.cfg.standing_envs_start + (self.cfg.standing_envs_end - self.cfg.standing_envs_start) * f
            self.command_generator.rel_standing_envs = rel
            self.extras.setdefault("log", {})["Curriculum/rel_standing_envs"] = rel
        if getattr(self.cfg, "action_rate_weight_end", None) is not None:
            f = min(1.0, step / max(1.0, float(self.cfg.action_rate_ramp_steps)))
            w = self.cfg.action_rate_weight_start + (self.cfg.action_rate_weight_end - self.cfg.action_rate_weight_start) * f
            self.cfg.rewards["action_rate_l2"] = w
            self.extras.setdefault("log", {})["Curriculum/action_rate_weight"] = w

    def _apply_curriculum(self) -> None:
        self._apply_schedules()
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
                # randomize_mass uses operation="scale", so the neutral point is 1.0, NOT 0.
                # Multiplying the bounds by dr (correct for an "add" term) would drive the
                # scale factors toward ZERO at low dr and produce massless links -- exactly the
                # bug this event was just fixed for. Interpolate toward 1.0 instead, the same
                # way the gains and friction terms below do.
                scaled_lo = 1.0 - (1.0 - lo) * dr
                scaled_hi = 1.0 + (hi - 1.0) * dr
                self.cfg.events["randomize_mass"].params["mass_distribution_params"] = (scaled_lo, scaled_hi)

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
            full = self._cmd_full_ranges
            full_lo, full_hi = full.get("lin_vel_x", self.cfg.curriculum_cmd_full_lin_vel_x)
            new_vx = (
                start_lo + (full_lo - start_lo) * cmd,
                start_hi + (full_hi - start_hi) * cmd,
            )
            vy_lo, vy_hi = full.get("lin_vel_y", (-0.3, 0.3))
            wz_lo, wz_hi = full.get("ang_vel_z", (-0.3, 0.3))
            self.command_generator.update_ranges({
                "lin_vel_x": new_vx,
                "lin_vel_y": (vy_lo * cmd, vy_hi * cmd),
                "ang_vel_z": (wz_lo * cmd, wz_hi * cmd),
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
        lo, hi = getattr(self.cfg, "action_delay_steps_range", [0, 0])
        self.extras["log"]["Curriculum/action_delay_max_steps"] = (
            int(lo) + int(round((int(hi) - int(lo)) * self._dr_ramp("action_delay"))))
        if self.bam_actuators and getattr(self.bam_actuators[0].cfg, "kp_fw_range", None):
            klo, khi = self.bam_actuators[0].cfg.kp_fw_range
            self.extras["log"]["Curriculum/servo_kp_max"] = klo + (khi - klo) * self._dr_ramp("servo_stiffness")
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
        # phase_clock [48:50] is ANTIsymmetric, not symmetric: gait_phase_contact encodes
        # "sin(phi) > 0 -> right foot stance", so an L<->R swap means phi -> phi + pi, which
        # negates both sin and cos.
        s[48] = -1.; s[49] = -1.
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

    def actor_mirror_tables(self):
        """(obs_perm, obs_sign, act_perm, act_sign) as CPU tensors for the ACTOR input exactly as the
        actor receives it, i.e. the per-frame 50-dim tables tiled over the history stack (the same
        chunk-wise mirror as mirror_obs). mirror x := x[..., perm] * sign. Used by the equivariant actor
        wrapper (train.equivariant_actor), which stores them as checkpoint buffers."""
        if self.cfg.policy_has_privileged_info:
            raise ValueError("actor_mirror_tables: the actor obs carries privileged dims, which mirror_obs "
                             "does not mirror (its per-frame layout is the 50-dim proprio frame only)")
        n_frames, rest = divmod(self.num_obs_policy, self.obs_proprio_dim)
        if rest or n_frames < 1:
            raise ValueError(f"actor obs dim {self.num_obs_policy} is not a multiple of the {self.obs_proprio_dim}-dim frame")
        perm = torch.cat([self._mirror_obs_perm.cpu() + i * self.obs_proprio_dim for i in range(n_frames)])
        sign = self._mirror_obs_signs.cpu().repeat(n_frames)
        return perm, sign, self._mirror_action_perm.cpu(), self._mirror_action_signs.cpu()

    def _resample_commands(self, env_ids: torch.Tensor):
        self.command_generator.reset(env_ids)
        # Heading-mode yaw from the post-reset heading now, not at the next step: the first
        # observation and the RSI reference are built from this command.
        self.command_generator.apply_heading(self.robot.data.root_quat_w)

