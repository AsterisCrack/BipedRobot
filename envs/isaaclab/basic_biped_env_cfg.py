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
class BasicBipedSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BIPED_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"]  # Only track contacts with the ground terrain
    )

@configclass
class BasicBipedEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 1 # Match MuJoCo's 100Hz (assuming sim_frequency=100)
    action_scale = 1.0  # Match MuJoCo's unscaled action space
    action_space = 12
    observation_space = 37 # Match BasicEnv exactly (19 qpos + 18 qvel)
    state_space = 37 # Same as obs
    action_space_limits = (-1.0, 1.0) # Normalized action space
    history_size = 0 # Default history size
    logging_level = "INFO" # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Observations
    observation_type = "basic" 
    policy_has_privileged_info = False
    critic_has_privileged_info = False # BasicEnv critic uses same obs as actor
    
    # observation groups
    observation_space_dim = {
        "policy": 37, 
        "critic": 37, 
    }

    # Randomization
    enable_perturbations = False # Disable for basic env
    push_interval_s = 15.0
    push_vel_range = 1.0
    
    # Physics Randomization
    enable_physics_randomization = False # Disable for basic env
    
    # Noise
    # Action noise
    action_noise_model: NoiseModelCfg = None # Disable noise
    
    # Observation noise
    observation_noise_model: NoiseModelCfg = None # Disable noise

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        use_fabric=True,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
    )
    
    # scene
    scene: BasicBipedSceneCfg = BasicBipedSceneCfg(
        num_envs=4096, # Match replay ratio
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
    )

    # events
    events: dict = {}

    # rewards
    rewards: dict = {
        "survived": 0.001,
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
        "termination": -0.05,
    }
    
    # Commands (Target velocities)
    commands: dict = {
        "base_velocity": {
            "ranges": {
                "lin_vel_x": (0.5, 0.5), # Fixed target 0.5 m/s
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            }
        }
    }
