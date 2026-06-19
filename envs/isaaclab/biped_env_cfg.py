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

from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG as BIPED_ROBOT_V1_CFG
from envs.assets.robot.biped_robot import JOINT_LIMITS as JOINT_LIMITS_V1
from envs.assets.robotV2.biped_robot import BIPED_ROBOT_CFG as BIPED_ROBOT_V2_CFG
from envs.assets.robotV2.biped_robot import JOINT_LIMITS as JOINT_LIMITS_V2

# Rough terrain scaled for the 30 cm BipedRobotV2.
# Purpose: ground perturbation training, NOT an obstacle course.
# The robot has no height scanner, so stairs/slopes are impossible to anticipate.
# Keep terrain gentle (max 8mm bumps, 12mm waves) so the flat-terrain policy adapts quickly.
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
    vertical_scale=0.005,  # 5 mm/unit
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 35% flat — still enough safe terrain for policy stability
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.35),
        # Random bumps 0–20 mm at max difficulty
        # noise_step must be >= vertical_scale (0.005) so int(noise_step/vertical_scale) >= 1
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.020), noise_step=0.005, border_width=0.25
        ),
        # Smooth waves 0–25 mm amplitude
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.25, amplitude_range=(0.0, 0.025), num_waves=3, border_width=0.25
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
        env_spacing=2.5,
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
    
    # Observations
    observation_type = "normal" # Options: "normal", "basic"
    policy_has_privileged_info = False
    critic_has_privileged_info = True

    # Gait clock
    gait_clock_base_freq: float = 1.5  # Hz, advances proportional to commanded speed

    # Action smoothing filter (EMA): target = alpha*raw + (1-alpha)*prev
    action_filter_alpha: float = 0.4

    # Actuator delay simulation: per-env delay sampled from [min, max] steps at reset
    action_delay_steps_range: list = [0, 0]
    # TODO: Changes made: action max delay 2 -> 0, commented new randomizations, removed history, lowered mlp size, removed symmetry

    # Randomization
    enable_perturbations = True
    push_interval_s = 15.0
    push_vel_range = 1.0

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
    curriculum_dr_max_push_y: float = 0.2
    curriculum_dr_mass_range: tuple = (-0.2, 0.4)
    curriculum_dr_gains_range: tuple = (0.8, 1.2)
    curriculum_dr_friction_range: tuple = (0.8, 1.1)
    curriculum_dr_com_range: float = 0.01
    curriculum_dr_payload_max: float = 0.15
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
    
    # Observation noise
    observation_noise_model: NoiseModelCfg = NoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01)
    )

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
                "position_range": (-0.02, 0.02),  # tight: start near neutral; ±0.1 created escape-barrier for dof_pos_l2
                "velocity_range": (-0.02, 0.02),
            },
        ),
        # Random perturbations (push)
        "push_robot": EventTerm(
            func="isaaclab.envs.mdp:push_by_setting_velocity",
            mode="interval",
            interval_range_s=(1.0, 5.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}},  # stronger push (was ±0.05)
        ),
        # Physics Randomization
        "randomize_mass": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_mass",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (-0.2, 0.4),
                "operation": "add",
            },
        ),
        # Randomize actuator gains
        "randomize_actuator_gains": EventTerm(
            func="isaaclab.envs.mdp:randomize_actuator_gains",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        ),
        # Randomize friction
        "randomize_friction": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_material",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 1.1),
                "dynamic_friction_range": (0.8, 1.1),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        ),
        # COM offset randomization: curriculum-scaled via curriculum_dr_events
        #"randomize_com": EventTerm(
        #    func="isaaclab.envs.mdp:randomize_rigid_body_com",
        #    mode="reset",
        #    params={
        #        "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        #        "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        #    },
        #),
        # Payload: curriculum-scaled via curriculum_dr_events
        "randomize_payload": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_mass",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (0.0, 0.15),
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

        # if not self.enable_perturbations:
        #     self.events.pop("push_robot", None)
        # if not self.enable_physics_randomization:
        #     for key in ["randomize_mass", "randomize_actuator_gains",
        #                 "randomize_friction", "randomize_com", "randomize_payload"]:
        #         self.events.pop(key, None)

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
        env_spacing=2.5,
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
    # yaw (0,1), roll (2,3), ankle_roll (8,9) flip; pitch and knee do not
    mirror_joint_perm: list = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    mirror_joint_signs: list = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
