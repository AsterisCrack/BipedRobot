from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg

from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG

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

@configclass
class BipedEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_scale = 1.0 
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
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # scene
    scene: BipedSceneCfg = BipedSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # termination
    termination_height = 0.2

    # events
    events: dict[str, EventTerm] = {
        "reset_base": EventTerm(
            func="isaaclab.envs.mdp:reset_root_state_uniform",
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        ),
        "reset_robot_joints": EventTerm(
            func="isaaclab.envs.mdp:reset_joints_by_offset",
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        ),
        # Random perturbations (push)
        "push_robot": EventTerm(
            func="isaaclab.envs.mdp:push_by_setting_velocity",
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
        # Physics Randomization
        "randomize_mass": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_mass",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (-1.0, 1.0),
                "operation": "add",
            },
        ),
        "randomize_friction": EventTerm(
            func="isaaclab.envs.mdp:randomize_rigid_body_material",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 1.5),
                "dynamic_friction_range": (0.5, 1.5),
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
    rewards = {
        "survived": 0.001,
        "velocity": 0.15,
        "ang_vel_tracking": 0.05,
        "height_vel_tracking": 0.05,
        "torque": 0.01,
        "action_diff": 0.01,
        "acceleration": 0.05,
        "flat_orientation": 0.04,
        "feet_flat": 0.3,
        "height": 0.1,
        "stall": 0.1,
        "base_stability": 0.05,
        "feet_airtime": 0.1,
        "torso_centering": 0.1,
        "joint_deviation": 0.05,
        "termination": -0.05,
    }
    
    # commands
    # We can define command ranges here
    commands = {
        "base_velocity": {
            "ranges": {
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0),
            },
        }
    }