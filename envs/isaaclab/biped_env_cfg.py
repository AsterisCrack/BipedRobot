from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .mdp.events import randomize_bam_actuator, randomize_rigid_body_com_absolute
from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG as BIPED_ROBOT_V1_CFG
from envs.assets.robot.biped_robot import JOINT_LIMITS as JOINT_LIMITS_V1
from envs.assets.robotV2.biped_robot import BIPED_ROBOT_CFG as BIPED_ROBOT_V2_CFG
from envs.assets.robotV2.biped_robot import JOINT_LIMITS as JOINT_LIMITS_V2

# Rough terrain scaled for the 29 cm BipedRobotV2.
# Purpose: ground perturbation training, NOT an obstacle course.
# The robot is BLIND (no height scanner), so anything it cannot absorb reactively is just
# noise in the reward. Worst feature is ~12 mm; see the per-terrain notes below for scaling.
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # num_rows=20 is critical: terrain generator centers the mesh at world origin, so
    # num_rows*size must cover the env grid (64x64 @ 2.5m = ±78.75m). With num_rows=20,
    # terrain spans ±80m — enough to cover all 4096 env origins without robot pileup.
    # (num_rows=3 → only 3 x-positions for 4096 robots → 68 robots per 1m² → physics chaos)
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,  # 10 cm/pixel — coarser mesh reduces edge-contact artifacts
    # 5 mm was too coarse: a 10 mm feature had only TWO height levels, so "rough" ground was
    # really a blocky two-tone staircase. 1 mm gives smooth relief at these amplitudes and
    # costs nothing (it is Z quantisation, not extra vertices).
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    # curriculum is deliberately left False. With True, difficulty is set by ROW index, which
    # only helps if envs also start at row 0 (max_init_terrain_level=0, i.e.
    # use_terrain_curriculum: true). But _compute_env_origins_curriculum assigns each env to a
    # PATCH ORIGIN, so that would pack all 4096 envs onto the 20 row-0 patches (~205 per patch)
    # instead of spreading them over 400. With curriculum False each patch instead draws
    # difficulty ~ U(0, 1), which already gives the full easy->hard spread as randomization --
    # just without progression. See TODO for what enabling progression would require.
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # Amplitudes are sized for a 29 cm robot with ~5 cm feet, ~20 cm legs, no height
        # scanner, and (post swing_foot_height fix) roughly 5-7 cm of foot clearance.
        # Rule of thumb: keep the worst feature under ~20-25% of swing clearance.
        # Scaling to a 1.7 m human is ~5.9x, so 12 mm here feels like ~7 cm of undulation --
        # rough grass or gravel, walked blind. The previous 20-25 mm was ~12-15 cm at human
        # scale, i.e. curbs, which is why it was too much.

        # Flat baseline. Keeps a safe fraction of envs for stable policy improvement.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),

        # Fine ground texture. NOTE: random_uniform_terrain IGNORES the difficulty parameter
        # (documented in hf_terrains.py), so this is ALWAYS at full range -- hence the small
        # ceiling. downsampled_scale > horizontal_scale samples coarsely and interpolates,
        # giving rolling texture instead of per-pixel noise the foot can catch on.
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.002, 0.008), noise_step=0.001,
            downsampled_scale=0.2, border_width=0.25,
        ),

        # Smooth long-wavelength undulation. Respects difficulty (amplitude scales with it).
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.20, amplitude_range=(0.0, 0.012), num_waves=4, border_width=0.25,
        ),

        # Gentle grades, up and down. 0.08 rad = 4.6 deg at max difficulty.
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.08), platform_width=1.5, border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.08), platform_width=1.5, border_width=0.25,
        ),

        # Discrete height patches. grid_width 0.25 m >> 5 cm foot, so the foot lands squarely
        # on one tile rather than straddling an edge -- tests stepping onto a slightly
        # different level, not knife-edge contact.
        # grid_width must NOT divide size (8.0 m) evenly: random_grid_terrain derives its
        # border as the remainder, `size - int(size/grid_width) * grid_width`, and raises if
        # that is 0. 0.25 gives exactly 32 tiles and no border. 0.3 gives 26 tiles + 0.2 m.
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.3, grid_height_range=(0.0, 0.010),
            platform_width=1.5,
        ),
    },
)

@configclass
class BipedRobotSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BIPED_ROBOT_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/torso_link",
    )


@configclass
class BipedRobotV2SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BIPED_ROBOT_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/base_link",
        offset=ImuCfg.OffsetCfg(pos=(-0.023673, 0.0, 0.025133)),
    )


@configclass
class BipedEnvCfg(DirectRLEnvCfg):
    # Switches
    use_rough_terrain: bool = False

    # Terrain curriculum: per-env difficulty progression when rough terrain is active.
    # Each env moves to a harder sub-terrain on timeout (success) and easier on termination (failure).
    use_terrain_curriculum: bool = False
    # Rolling-mean episode-length threshold for notifying that rough terrain is ready to enable.
    curriculum_episode_len_threshold: float = 15.0

    # scene
    scene: BipedRobotSceneCfg = BipedRobotSceneCfg(
        num_envs=4096,
        env_spacing=0.5,
    )

    # robot-specific mappings
    joint_limits = JOINT_LIMITS_V1
    right_foot_body_name = "r_foot"
    left_foot_body_name = "l_foot"
    base_body_name = "torso_link"
    hip_joint_names = ["r_hip_z", "l_hip_z", "r_hip_x", "l_hip_x"]
    ankle_roll_joint_names = ["r_ankle_x", "l_ankle_x"]
    ankle_pitch_joint_names: list = []  # override in subclass if robot has ankle pitch joints
    knee_joint_names = ["r_knee", "l_knee"]

    # Left-right symmetry transform for joints.
    # V1 joint order (R,L interleaved): r_hip_z=0, l_hip_z=1, r_hip_x=2, l_hip_x=3,
    #   r_hip_y=4, l_hip_y=5, r_knee=6, l_knee=7, r_ankle_y=8, l_ankle_y=9,
    #   r_ankle_x=10, l_ankle_x=11
    # perm: which joint index maps to each position after L↔R swap
    mirror_joint_perm: list = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    # signs: -1 for joints whose direction inverts under L↔R reflection
    # yaw (0,1), roll (2,3), ankle_roll (10,11) flip; pitch and knee do not
    mirror_joint_signs: list = [-1., -1., -1., -1., 1., 1., 1., 1., 1., 1., -1., -1.]
    
    # env
    episode_length_s = 20.0
    decimation = 4
    # DEAD: never read. _pre_physics_step maps action [-1,1] affinely onto the full joint-limit
    # box, so action 0 lands on the limit MIDPOINT (knee -57.5 deg, hip_pitch -30 deg) rather
    # than the all-zeros default pose. Switching to a residual map
    # (target = default_pos + action_scale * action) would make this live and put action 0 at
    # the neutral pose
    action_scale = 0.7
    action_space = 12
    observation_space = 48
    state_space = 59 # Observations + privileged info
    action_space_limits = (-1.0, 1.0) # Normalized action space
    history_size = 0 # Default history size
    use_history = False # Whether to use history in observations
    logging_level = "INFO" # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    reward_scale = 1.0 # Scale all rewards by this factor

    # Animation (imitation)
    animation_npz_path: str | None = None
    animation_loop: bool = True
    animation_random_start: bool = True
    animation_speed: float = 1.0
    animation_pos_std: float = 0.5
    animation_vel_std: float = 1.0

    # Gait-table imitation (Gait Studio npz). None = off, and the speed-proportional gait clock
    # below stays in charge. With a table the clock is fixed-period (the table's) and holds
    # while the command is below reference_stand_threshold.
    reference_npz_path: str | None = None
    reference_stand_threshold: float = 0.05
    reference_rsi_prob: float = 0.8          # fraction of resets that start on the reference
    reference_rsi_joint_noise: float = 0.05  # rad, uniform, clamped to the action limits
    reference_std: dict = {
        "joint_pos": 0.15, "joint_vel": 1.5, "foot_pos": 0.02,
        "orientation": 0.08, "lin_vel": 0.1, "ang_vel": 0.3,
        "foot_height": 0.015,   # imitation_foot_height: per-foot sole clearance vs the table, [m]
    }

    # Observations
    # IMU acceleration = mean over the control step (see biped_env.StepAveragedImu). False is the
    # old Isaac Lab behaviour (motion part 4x too large); only for evaluating old checkpoints.
    imu_step_averaged: bool = True
    observation_type = "normal" # Options: "normal", "basic"
    policy_has_privileged_info = False
    critic_has_privileged_info = True

    # Gait clock
    # Phase advances as dphi = dt * 2*pi * freq * v_cmd, so a full cycle takes
    # T = 1 / (freq * v_cmd) and the implied step length is v_cmd * T / 2 = 1 / (2 * freq).
    # The frequency therefore sets STEP LENGTH directly and is independent of speed.
    #   freq 1.5 -> 0.33 m stride (impossible on 0.2 m legs; the old value)
    #   freq 5.0 -> 0.10 m stride (sane for a 30 cm biped)
    # A miscalibrated clock makes gait_phase_contact unsatisfiable regardless of policy, which
    # is one of the reasons the earlier attempt to enable that reward failed.
    gait_clock_base_freq: float = 5.0  # => 1/(2*freq) = 0.10 m step length
    # Clock mode when NO gait table is set (a table always uses its own fixed period):
    #   speed_proportional: the rate above (gait frequency = freq * |v_xy| Hz).
    #   fixed_period: constant rate 2*pi/gait_clock_period_s while moving, HELD while the command
    #     is below gait_clock_stand_threshold in both |v_xy| and |wz| (same rule as the table clock).
    gait_clock_mode: str = "speed_proportional"
    gait_clock_period_s: float = 0.7
    # cadence_law (no table): period from max (slow, just above the stand threshold) to min (fastest command),
    # linear in the elliptic command magnitude (utils/cadence_law.py).
    gait_clock_period_min_s: float = 0.6
    gait_clock_period_max_s: float = 0.9
    gait_clock_stand_threshold: float = 0.05

    # Action smoothing filter (EMA): target = alpha*raw + (1-alpha)*prev
    action_filter_alpha: float = 0.4

    # Actuator delay simulation: per-env delay sampled from [min, max] steps at reset
    action_delay_steps_range: list = [0, 0]
    # Sensor lag (observation side), in control steps. Servo delay is fixed for all envs;
    # IMU delay is sampled per-env at reset from [min, max].
    servo_obs_delay_steps: int = 0
    imu_obs_delay_steps_range: list = [0, 0]
    # Per-episode CONSTANT sensor biases, drawn uniformly in +-range at reset. Distinct from
    # observation_noise_std, which resamples every step and therefore averages out across the
    # history stack. Real hardware has a fixed null offset per power-cycle, which is harder.
    imu_accel_bias: float = 0.0      # m/s^2
    imu_gyro_bias: float = 0.0       # rad/s
    # Per-ROBOT constants, drawn once at startup (see BipedEnv.__init__):
    mount_offset_rad: float = 0.0       # servo zero error, +-range, applied to obs AND command
    imu_misalignment_deg: float = 0.0   # IMU board tilt, random axis, angle in [0, max]

    # Action map: "centered" (a=0 -> default pose, a=+-1 -> limits), "absolute" (legacy) or
    # "residual" (target = clamp(home + scale*a, limits)). "residual" takes its home pose
    # (deg) and per-joint-type scale (rad) from these dicts, set by env_config.apply_env_config;
    # the home is also written into the articulation's init_state, so it IS default_joint_pos.
    # spawn_height_m: init base z (None = the asset's 0.29 m; residual defaults to 0.286). env_config may say "auto" for
    # action_home_deg / action_scale_rad / spawn_height_m: train.py freezes them into numbers before they reach this cfg
    # (utils/residual_home.py), and env_config._apply_action_home resolves them itself, with a warning, if they arrive unfrozen.
    action_map: str = "absolute"
    action_home_deg: dict | None = None
    action_scale_rad: dict | None = None   # per type: a number, or {pos: x, neg: y} (asymmetric; utils/residual_action.py)
    spawn_height_m: float | None = None
    target_base_height: float = 0.29     # base_height_l2 target; follows spawn_height_m

    # Schedules, in common_step_counter units; None leaves the value alone.
    standing_envs_start: float = 0.02
    standing_envs_end: float | None = None
    standing_envs_ramp_steps: int = 0
    action_rate_weight_start: float = -0.01
    action_rate_weight_end: float | None = None
    action_rate_ramp_steps: int = 0

    # swing_foot_height window, as clearance ABOVE the measured resting foot height [m]
    swing_min_clearance: float = 0.01
    swing_max_clearance: float = 0.07
    # A foot whose CURRENT swing has lasted longer than this earns no swing_foot_height
    # (stops the "hold one leg up" exploit). Uses the contact sensor's current_air_time.
    swing_max_time_s: float = 0.5

    # Height-aware contact flag for the REWARD terms imitation_contact and feet_slide only (the
    # contact sensor, the clearance datum and every other consumer stay force-based): a foot
    # counts as in contact if |F| > 1 N OR its clearance (the swing_foot_height datum quantity)
    # is below this many metres. Closes the "drag the foot with < 1 N load" exploit. None / 0 =
    # old behaviour (force only).
    contact_height_thresh_m: float | None = None
    # hard (default) = the bool flag above. soft = score c = max(force_flag, clamp(1 - clearance /
    # contact_height_thresh_m, 0, 1)) in [0, 1]: feet_slide weights the planar speed by c and
    # imitation_contact scores 1 - |c - table_contact| per foot (no cliff at the threshold). Needs
    # contact_height_thresh_m > 0. (Diagnostics/dragged_foot_frac keeps the hard-threshold definition.)
    contact_height_mode: str = "hard"

    # imitation_foot_height and imitation_contact: per-foot values combined over the two feet by
    # "mean" (default, old) or "min" (the worse foot decides: a limp stops keeping half the reward).
    imitation_foot_agg: str = "mean"

    # Velocity-tracking kernel width. *_std_rel = 0: fixed std (0.2 linear, 0.5 yaw), the old
    # behaviour. > 0: per-env std = clamp(rel * |cmd|, min=*_std_min) (see rewards.py).
    track_lin_std_rel: float = 0.0
    track_lin_std_min: float = 0.07
    track_ang_std_rel: float = 0.0
    track_ang_std_min: float = 0.15
    # Centre of the linear tracking kernel. command (default, old): the commanded velocity. reference: the
    # gait table's base_lin_vel_b at the current (command, phase) (needs the table and track_lin_std_rel > 0),
    # measured against the fresh robot.data velocity. The width stays clamp(rel * |cmd|, min) either way. The
    # table's own sway (lateral std ~0.065 m/s) otherwise makes a perfect copy score 0.38 at vx 0.1 (standing 0.13).
    track_lin_target: str = "command"

    # feet_air_time: per_step = old (feet_air_time_positive_biped, paid every step while a foot
    # is up in single stance); touchdown = paid once at landing =
    # clamp(swing - feet_air_time_min_s, 0, feet_air_time_max_s - feet_air_time_min_s),
    # single stance only, command-gated. The min is a dead zone for short flicker flights.
    feet_air_time_mode: str = "per_step"
    feet_air_time_min_s: float = 0.15
    feet_air_time_max_s: float = 0.4

    # com_support_centering: reward = exp(-0.5 (d/std)^2) where d is the distance the CoM
    # lies OUTSIDE the support polygon (0 when inside). com_foot_margin is the foot's own
    # half-extent, so a CoM over any part of a planted foot reads 0.
    com_centering_std: float = 0.04
    com_foot_margin: float = 0.03

    # foot_separation: origin_one_sided = old (penalise the LINK-ORIGIN lateral separation below
    # min_sep 0.09, one-sided); sole_band = two-sided band on the SOLE-CENTRE lateral separation,
    # penalty clamp((|sep - foot_sep_target_m| - foot_sep_tol_m) / foot_sep_scale_m, 0, 3)^2.
    # foot_sole_offset_m = (x, lateral magnitude, z) of the sole centre in the foot link frame
    # (ankle_pitch joint origin); the lateral sign (+left / -right) is applied by the reward.
    foot_separation_mode: str = "origin_one_sided"
    foot_sep_target_m: float = 0.095
    foot_sep_tol_m: float = 0.025
    foot_sep_scale_m: float = 0.02
    foot_sole_offset_m: list = [0.0163, 0.0211, -0.0316]

    # undesired_contacts: per-body clamp((|F| - thresh) / sat, 0, 1). Defaults = old behaviour.
    undesired_contacts_sat_n: float = 20.0
    undesired_contacts_thresh_n: float = 1.0

    # leg_contact_l2 (weight in rewards, default 0): per-body clamp((|F| - thresh) / (sat - thresh), 0, 1)^2
    # over the non-foot leg bodies (net force, includes the other leg) and the feet (net force
    # minus the ground-plane filter force). See BipedEnv._leg_contact_forces.
    leg_contact_thresh_n: float = 0.5
    leg_contact_sat_n: float = 5.0

    # foot_proximity (E21): TERRAIN-INDEPENDENT foot-to-foot gap. With use_rough_terrain the contact sensor has no
    # filter prims (force_matrix_w is None), so undesired_contacts / leg_contact_l2 cannot tell floor force from
    # other-foot force and leave the feet out: feet pressing together cost nothing (run gaitref_2026-10-22_07-18-30:
    # 100% of steps at command 0, 13.8 N, -1.2 mm). This term is purely geometric: the horizontal-plane gap between the
    # two foot SOLE rectangles (rewards.foot_footprint_gap), penalty clamp((margin - gap)/margin, 0, 2)^2.
    # The footprint rectangle is centred on foot_sole_offset_m (above) with these half extents [x, y] (metres). Derived from
    # the collision mesh r/l_foot_link_1.stl (robot_v2.xml, foot link frame): the whole mesh projects to x -28.7 .. +61.3 mm
    # (90.0 mm) and |y| 2.0 .. 39.3 mm (37.3 mm) -> half extents (45.0, 18.6) mm. (The sole face alone is |y| 3.0 .. 39.2 mm,
    # half 18.1; the foot is 1 mm wider per side above the sole chamfer, and 18.6 splits that so the gap matches MuJoCo
    # within ~1 mm for both a flat foot, +0.9 mm, and a rolled one pressing on its sole edge, -1.0 mm.)
    # tests/test_foot_proximity.py re-derives them from the MJCF and asserts this matches within 1 mm.
    foot_sole_half_extents_m: list = [0.0450, 0.0186]
    # Gap [m] below which the penalty starts. 6 mm: the gait table's own footprint gap over every cell the training command
    # ranges select (the whole 16x12x16 grid) has min 6.4 mm / p1 11.4 mm / p5 17.1 mm (CHANGES.md, E21), so the reference
    # gait is never penalised. Largest whole mm below the table minimum; tests/test_foot_proximity.py recomputes it.
    foot_proximity_margin_m: float = 0.006
    # foot_drag (E25): a foot that is NOT loaded (|F| <= 1 N) and lower than this clearance [m] (lowest sole corner above its
    # height at the foot's last loaded step; NOT the link-origin swing clearance, which a pitched foot biases) and moving is
    # scuffing; the penalty fades linearly to 0 at this height. See rewards.foot_drag / rewards.foot_sole_slip_speed (foot_slip
    # uses the same sole corners and foot_sole_half_extents_m). The gait-reference yaml sets 0.002.
    foot_drag_height_m: float = 0.005
    # Optional termination (counts as `died`, so termination_penalty applies): end the episode once the footprint gap has
    # been <= 0 for this many seconds in a row. 0 = off. 0.25 is the suggested value if the penalty alone does not stop it.
    foot_overlap_termination_s: float = 0.0
    # TODO: Changes made: action max delay 2 -> 0, commented new randomizations, removed history, lowered mlp size, removed symmetry

    # Randomization
    enable_perturbations = True

    # Physics Randomization
    enable_physics_randomization = True

    # Video recording — set to True (via train.py) when --video is passed
    enable_video_camera: bool = False

    # Curriculum learning — all DR and command ranges scale from 0→full over training
    curriculum_enabled: bool = False
    curriculum_dr_start_steps: int = 3_000_000   # env steps with zero DR (robot learns basic walking)
    curriculum_dr_full_steps: int  = 20_000_000  # env steps when DR reaches 100%
    curriculum_cmd_ramp_steps: int = 5_000_000   # env steps to reach full command velocity range
    curriculum_init_ramp_steps: int = 5_000_000  # env steps to reach full joint init range
    # Which DR events are curriculum-scaled. Empty list = scale ALL events (backward compat).
    curriculum_dr_events: list = []

    # Target (full-curriculum) DR values — curriculum scales toward these
    curriculum_dr_max_push_x: float = 0.3
    curriculum_dr_max_push_y: float = 0.3
    # Mass is now a SCALE factor, not an additive kg offset, so this is centred on 1.0.
    # The old (-0.2, 0.4) was in kg and drove sub-0.2 kg leg links to zero mass.
    curriculum_dr_mass_range: tuple = (0.9, 1.1)
    curriculum_dr_gains_range: tuple = (0.7, 1.3)
    curriculum_dr_friction_range: tuple = (0.3, 1.4)
    # NOTE: com and payload are the ONLY two names in curriculum_dr_events, so these two values
    # are what actually reach the sim -- _apply_curriculum overwrites the event params from
    # them every reset. Editing the EventTerm params for com/payload does nothing; edit here.
    # (The reverse holds for mass/gains/friction/push: those are not in the list, so their
    # EventTerm params are live and these values are only documentation of the intended target.)
    curriculum_dr_com_range: float = 0.025  
    curriculum_dr_payload_max: float = 0.1    # 2026-10-24 from 0.1: the robot carries no payload; 3% of the 1.661 kg body
    # Command range limits
    curriculum_cmd_start_lin_vel_x: tuple = (0.0, 0.2)   # slow forward-only at start
    curriculum_cmd_full_lin_vel_x: tuple  = (-0.3, 0.5)  # full range at end
    # Joint init range limits
    curriculum_init_range_min: float = 0.01
    curriculum_init_range_max: float = 0.10
    
    # Noise
    # Action noise
    action_noise_model: NoiseModelCfg = NoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01)
    )
    
    # Observation noise, per proprio channel. This replaces DirectRLEnvCfg's
    # observation_noise_model, which is deliberately left at its inherited None so
    # DirectRLEnv.step() does not also noise the flattened history stack.
    #
    # A single scalar across all 50 dims cannot be
    # right: the same number is either negligible for the IMU or enormous for the encoders,
    # and three of the channels are not sensors at all (commands and previous actions are
    # the policy's own inputs; the gait clock is generated). Those get exactly zero.
    observation_noise_std: dict = {
        "imu_acc": 0.15,        # m/s^2
        "gyro": 0.02,           # rad/s
        "projected_gravity": 0.01,
        "commands": 0.0,
        "joint_pos": 0.0015,    # rad, ~1 encoder LSB of jitter on top of the quantisation
        # rad/s. The reference robot uses +-0.25 here against +-0.001 on joint_pos: the
        # firmware's velocity estimate is the one genuinely bad servo channel. Refine with
        # tools/speed_register_check.py once real data exists.
        "joint_vel": 0.25,
        "previous_actions": 0.0,
        "gait_clock": 0.0,
    }

    # Encoder quantisation. STS3215 is 4096 counts/rev on both position and the speed
    # register, so the policy sees the same resolution floor in sim as on hardware.
    quantize_joint_obs: bool = True
    # Band-limited sensor 'vibration' (2026-10-26 hardware traces): walking on the robot carries 3-4x the sim's
    # 3-10 Hz content on the gyro (+0.4-0.9 rad/s), accelerometer (+0.5-1.3 m/s^2) and knee/hip joint velocity
    # (+0.3-0.8 rad/s) -- structure/gear play the encoders and the sim don't have. White per-step noise largely
    # averages out over the history stack; this does not. Per env and per dim: a resonator (pole radius 0.9,
    # ~1.6 Hz bandwidth) at a centre frequency drawn in observation_vibration_hz, unit-variance normalised, times
    # std[channel] times a per-env amplitude U(0, 1). Redrawn at reset. Keys: imu_acc, gyro, joint_vel. {} = off.
    observation_vibration_std: dict = {}
    observation_vibration_hz: list = [3.0, 10.0]
    observation_vibration_amp_range: list = [0.0, 1.0]   # per-env amplitude multiplier, drawn uniformly at reset

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
            solver_type=1,
        )
    )
    

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    
    # events
    events: dict[str, EventTerm] = {
        "reset_base": EventTerm(
            func="isaaclab.envs.mdp:reset_root_state_uniform",
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-0.01, 0.01),
                    "pitch": (-0.01, 0.01),
                    "yaw": (-0.01, 0.01),
                },
            },
        ),
        "reset_robot_joints": EventTerm(
            func="isaaclab.envs.mdp:reset_joints_by_scale",
            mode="reset",
            params={
                # DEAD VALUES. _apply_curriculum overwrites both with
                # curriculum_init_range_max (0.10) whenever curriculum_enabled is true, and
                # curriculum_init_ramp_steps: 0 makes init_scale 1.0 from the very first reset.
                # Edit curriculum_init_range_min/max above to change the actual init range.
                # (Original note: ±0.1 created an escape-barrier for dof_pos_l2 at weight -0.5.)
                "position_range": (-0.02, 0.02),
                "velocity_range": (-0.02, 0.02),
            },
        ),
        # Random perturbations (push)
        "push_robot": EventTerm(
            func="isaaclab.envs.mdp:push_by_setting_velocity",
            mode="interval",
            interval_range_s=(3.0, 6.0),
            # Added yaw: every disturbance was previously a pure linear shove. A real knock --
            # a hand, a cable snag, an uneven foot strike -- almost always has a rotational
            # component, and recovering yaw mid-stride is a different skill from recovering a
            # lateral shove. x/y also widened 0.3/0.2 -> 0.4/0.3.
            # NOTE: push_robot is NOT in curriculum_dr_events, so these params are live from
            # step 0 and curriculum_dr_max_push_x/y never reach the sim.
            params={"velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.6, 0.6),
            }},
        ),
        # Physics Randomization
        # NOTE: Isaac Lab draws an INDEPENDENT delta per body (events.py, sample shape is
        # (num_envs, num_bodies)) -- not one value broadcast across bodies.
        #
        # The previous config was add (-0.2, +0.4) kg on body_names=".*". V2's nominal total is
        # 1.661 kg and every one of the 12 leg links is LIGHTER than 0.2 kg, so the negative
        # end of that range drove links to min_mass=1e-6 -- silently, no error. Result: the
        # ankle/hip_lower links were massless on ~30% of episodes and the feet on ~17%, with
        # recompute_inertia scaling their inertia by ~4.7e-5. Mean total mass was 1.86x nominal
        # over a 0.38x-3.98x range, i.e. the real robot sat at ~the 1st percentile of training.
        #
        # scale is dimensionally correct, centred on 1.0, and cannot produce a zero or negative
        # mass. body_names=".*_link_1" fullmatches exactly the 12 leg links and excludes
        # base_link, which is owned by randomize_payload -- that term resets its target to
        # default before applying, so any mass term on base_link would be silently discarded.
        "randomize_mass": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_mass",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_1"),
                "mass_distribution_params": (0.92, 1.08),  # 2026-10-24 tightened from (0.85, 1.15): CAD / weighed links, gait-insensitive
                "operation": "scale",
            },
        ),
        # Servo population for the BAM actuator: per-robot supply voltage, per-servo friction
        # scale and backlash. Replaces randomize_actuator_gains and randomize_effort_limit --
        # under BAM there are no PD gains to scale and the torque ceiling follows from vin and
        # duty. The params here are overwritten from config.yaml's `actuator`/`backlash`
        # blocks by src/isaaclab/env_config.py.
        "randomize_bam_actuator": EventTerm(
            func=randomize_bam_actuator,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "vin_range": (11.1, 12.6),
                "friction_scale_range": (0.9, 1.1),
                "lash_range_rad": None,
            },
        ),
        # Joint friction (stiction) and armature (reflected rotor inertia).
        # Previously fixed constants: friction=0.2, armature=0.04 for every joint on every
        # robot. Stiction is the largest unmodelled unit-to-unit variation on hobby servos --
        # nominally-identical STS3215s differ substantially, and it directly sets how much
        # torque is lost before the joint moves at all.
        #
        # mode="startup", NOT "reset", deliberately. These are manufacturing properties: a
        # given physical servo's stiction does not change between episodes, so re-drawing it
        # every reset models something that cannot happen. Drawing once per env gives 4096
        # distinct robots, and the shared policy must be robust across that population. It also
        # avoids the per-reset CPU round-trip this term warns about in its own docstring.
        #
        # friction band is wider than armature's on purpose: armature is set by the gearbox
        # ratio and is well determined, stiction is not.
        "randomize_joint_params": EventTerm(
            func="isaaclab.envs.mdp:randomize_joint_parameters",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                # No friction here: the BAM kernel owns all servo friction and forces the PhysX
                # joint friction to 0 (scaling a 0 is a no-op anyway). Friction spread is
                # randomize_bam_actuator's friction_scale_range.
                "armature_distribution_params": (0.85, 1.15),  # about BAM's identified armature
                "operation": "scale",
            },
        ),
        # Randomize friction
        "randomize_friction": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_material",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                # Effective foot-ground mu equals this range: sim.physics_material uses
                # friction_combine_mode="multiply" and the terrain material is exactly 1.0, so
                # the robot's draw passes through unchanged. If terrain friction ever changes
                # from 1.0, this whole range scales silently.
                # Live range below: static (0.6, 1.1), dynamic (0.5, 1.0) (an older note said (0.3, 1.4)). Wet tile
                # or laminate is mu ~0.2-0.3, and that is exactly where a flat-footed biped
                # slips. Training only down to 0.5 never sees a genuinely slippery surface.
                "static_friction_range": (0.6, 1.1),
                "dynamic_friction_range": (0.5, 1.0),
                # Was pinned at zero. Carpet/rubber/tile differ in bounce, and a hard zero also
                # means foot-strike never has any rebound to reject.
                "restitution_range": (0.0, 0.3),
                "num_buckets": 64,
            },
        ),
        # COM offset randomization: curriculum-scaled via curriculum_dr_events.
        #
        # Deliberately NOT isaaclab.envs.mdp:randomize_rigid_body_com. That implementation does
        # `coms += sample` against the CURRENT com (events.py:426-429) and, unlike
        # randomize_rigid_body_mass, never resets to a nominal. Under mode="reset" the offset
        # therefore compounds every episode into an unbounded random walk: sigma is 5.77 mm per
        # reset for a +-1 cm range, growing as sqrt(N) -- ~3 cm by 29 resets, ~6 cm by 100,
        # ~12 cm by 400. On a 30 cm robot that eventually displaces the whole-body com by more
        # than half a foot length, per-env and invisible to the policy. This was the confirmed
        # cause of the robot refusing to walk whenever COM randomization was on; the magnitude
        # was never the problem, the accumulation was.
        "randomize_com": EventTerm(
            func=randomize_rigid_body_com_absolute,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)},
            },
        ),
        # Payload: curriculum-scaled via curriculum_dr_events
        "randomize_payload": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_mass",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (0.0, 0.1),
                "operation": "add",
            },
        ),
    }

    # observation groups
    # We define the keys here, but the actual construction happens in the env class
    observation_space_dim = {
        "policy": None, 
        "critic": None, 
    }
    
    # rewards
    rewards = {}
    
    # commands
    # We can define command ranges here
    commands = {
        "base_velocity": {
            "resampling_time_range": (10.0, 10.0),
            "heading_control_stiffness": 0.3,
            "rel_standing_envs": 0.02,
            "rel_heading_envs": 1.0,
            # "box" = per-axis ranges only; "ellipse_xy" also limits (vx, vy) to an ellipse (utils/command_limit.py).
            "limit_shape": "box",
            "ranges": {
                "lin_vel_x": (-0.3, 0.4),
                "lin_vel_y": (-0.3, 0.3),
                "ang_vel_z": (-0.2, 0.2),
                "heading": (-3.14159, 3.14159),
            },
        }
    }

    def __post_init__(self):
        super().__post_init__()

        # These gates were commented out while train.py/common.py kept assigning the flags,
        # which default to False in schema.py. Net effect: the config claimed DR was OFF while
        # every randomization ran at full strength, so anyone disabling DR to debug saw no
        # change and wrongly cleared it as a cause. config.yaml now sets both to true
        # explicitly, so re-enabling the gate preserves behaviour AND makes the flag usable.
        if not self.enable_perturbations:
            self.events.pop("push_robot", None)
        if not self.enable_physics_randomization:
            for key in ["randomize_mass", "randomize_friction", "randomize_com",
                        "randomize_payload", "randomize_joint_params", "randomize_bam_actuator"]:
                self.events.pop(key, None)

        if self.use_rough_terrain:
            self.terrain.terrain_type = "generator"
            self.terrain.terrain_generator = ROUGH_TERRAINS_CFG
            self.terrain.max_init_terrain_level = 0 if self.use_terrain_curriculum else None
            self.terrain.collision_group = 0
            self.scene.contact_forces.filter_prim_paths_expr = []
            
@configclass
class BipedRoughEnvCfg(BipedEnvCfg):
    def __post_init__(self):
        self.use_rough_terrain = True
        super().__post_init__()


@configclass
class BipedRobotV2EnvCfg(BipedEnvCfg):
    # scene
    scene: BipedRobotV2SceneCfg = BipedRobotV2SceneCfg(
        num_envs=4096,
        env_spacing=0.5,
    )

    # robot-specific mappings
    joint_limits = JOINT_LIMITS_V2
    right_foot_body_name = "(r|right)_foot_link(_\d+)?"
    left_foot_body_name  = "(l|left)_foot_link(_\d+)?"
    base_body_name       = "base_link"

    hip_joint_names = [
        "(l|left)_hip_yaw.*",
        "(r|right)_hip_yaw.*",
        "(l|left)_hip_roll.*",
        "(r|right)_hip_roll.*",
        # hip_pitch deliberately excluded: it swings the leg forward/backward and is required for walking
    ]

    ankle_roll_joint_names = [
        "(l|left)_ankle_roll.*",
        "(r|right)_ankle_roll.*",
    ]

    ankle_pitch_joint_names = [
        "(l|left)_ankle_pitch.*",
        "(r|right)_ankle_pitch.*",
    ]

    hip_pitch_joint_names = [
        "(l|left)_hip_pitch.*",
        "(r|right)_hip_pitch.*",
    ]

    knee_joint_names = [
        "(l|left)_knee.*",
        "(r|right)_knee.*",
    ]

    # Left-right symmetry transform for V2 joints.
    # V2 joint order (L,R interleaved): l_hip_yaw=0, r_hip_yaw=1, l_hip_roll=2, r_hip_roll=3,
    #   l_hip_pitch=4, r_hip_pitch=5, l_knee=6, r_knee=7,
    #   l_ankle_roll=8, r_ankle_roll=9, l_ankle_pitch=10, r_ankle_pitch=11
    # All signs are +1 and the mirror is a PURE PERMUTATION. The V2 URDF is a mirrored CAD
    # model, so the sagittal reflection is already baked into the joint axis definitions:
    # a_R = -M*a_L for all six pairs, which gives q_R = +q_L, i.e. no sign flip.
    # Do NOT copy V1's sign vector here — V1 is not a mirrored model and its indices differ
    # (V1: 8,9 = ankle_pitch, 10,11 = ankle_roll; V2: 8,9 = ankle_roll, 10,11 = ankle_pitch).
    # Setting these to -1 collapses training.
    mirror_joint_perm: list = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    mirror_joint_signs: list = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
