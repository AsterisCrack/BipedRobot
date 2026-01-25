"""
Isaac Sim Configuration for Biped Robot.

This configuration mirrors the MuJoCo configuration found in `Robot_description/urdf/robot_mujoco.xml`.
It defines the robot asset, initial state, and actuator properties.
"""

import math
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "robot.usd")

# Joint limits in degrees [min, max]
_JOINT_LIMITS_DEG = [
    [-90, 45],  # r_hip_z
    [-45, 90],  # r_hip_x
    [-45, 45],  # r_hip_y
    [-45, 45],  # r_knee
    [-90, 90],  # r_ankle_y
    [-90, 90],  # r_ankle_x
    [-90, 90],  # l_hip_z
    [-90, 90],  # l_hip_x
    [-90, 90],  # l_hip_y
    [-90, 90],  # l_knee
    [-90, 30],  # l_ankle_y
    [-30, 90],  # l_ankle_x
]

# Convert to radians
JOINT_LIMITS = [[math.radians(lim) for lim in joint] for joint in _JOINT_LIMITS_DEG]

BIPED_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.022936, 0.0, 0.29416), # Offset due to root offset in URDF
        joint_pos={
            "r_hip_z": 0.0,
            "r_hip_x": 0.0,
            "r_hip_y": 0.0,
            "r_knee": 0.0,
            "r_ankle_y": 0.0,
            "r_ankle_x": 0.0,
            "l_hip_z": 0.0,
            "l_hip_x": 0.0,
            "l_hip_y": 0.0,
            "l_knee": 0.0,
            "l_ankle_y": 0.0,
            "l_ankle_x": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # actuators={
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=100.0,
    #         damping=10.0,
    #         armature=0.045,
    #         friction=0.03,
    #         effort_limit_sim=10.0,
    #         velocity_limit_sim=100.0, ),
    #     },
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=200,
            damping=20,
            armature=0.01,
            friction=0.0,
            effort_limit_sim=30.0,
            velocity_limit_sim=500.0,
        ),
    },
)
