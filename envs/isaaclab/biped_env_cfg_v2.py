from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.05), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.03), noise_step=0.01, border_width=0.25
        ),
    },
)

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BIPED_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"]  # Only track contacts with the ground terrain
    )
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/torso_link",
    )
    

@configclass
class BipedEnvCfg(DirectRLEnvCfg):
    # Switches
    use_rough_terrain: bool = False

    # scene
    scene: BipedSceneCfg = BipedSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    
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
    
    # Observations
    observation_type = "normal" # Options: "normal", "basic"
    policy_has_privileged_info = False
    critic_has_privileged_info = True

    # Randomization
    enable_perturbations = True
    push_interval_s = 15.0
    push_vel_range = 1.0
    
    # Physics Randomization
    enable_physics_randomization = True
    
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
                "position_range": (-0.01, 0.01),
                "velocity_range": (-0.01, 0.01),
            },
        ),
        # Random perturbations (push)
        "push_robot": EventTerm(
            func="isaaclab.envs.mdp:push_by_setting_velocity",
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
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
        
        if self.use_rough_terrain:
            # Update Terrain
            self.terrain.terrain_type = "generator"
            self.terrain.terrain_generator = ROUGH_TERRAINS_CFG
            self.terrain.max_init_terrain_level = 5
            self.terrain.collision_group = 0
            
@configclass
class BipedRoughEnvCfg(BipedEnvCfg):
    def __post_init__(self):
        self.use_rough_terrain = True
        super().__post_init__()

