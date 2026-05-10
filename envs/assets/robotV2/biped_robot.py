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
# Tuned for ~30cm biped: maps the full [-1,1] action range to physically useful positions.
# Using tight limits ensures the dof_pos_limits penalty is effective and the policy
# does not waste capacity on unreachable regions outside the physical URDF limits.
_JOINT_LIMITS_DEG = [
    [-45, 45],  # left_hip_yaw   - rotation about vertical axis (turning)
    [-45, 45],  # right_hip_yaw
    [-25, 45],  # left_hip_roll  - lateral lean / frontal-plane balance
    [-25, 45],  # right_hip_roll
    [-90, 30],  # left_hip_pitch - forward/backward leg swing
    [-90, 30],  # right_hip_pitch
    [-120, 5],  # left_knee      - bends in one direction only (0 = straight)
    [-120, 5],  # right_knee
    [-60, 60],  # left_ankle_roll  - lateral foot tilt
    [-60, 60],  # right_ankle_roll
    [-35, 50],  # left_ankle_pitch - dorsi/plantarflexion
    [-35, 50],  # right_ankle_pitch
]

# Convert to radians
JOINT_LIMITS = [[math.radians(lim) for lim in joint] for joint in _JOINT_LIMITS_DEG]

BIPED_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.2, # Increased damping for stability (Note: applies to whole robot)
            angular_damping=0.5, 
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.29), # Offset due to root offset in URDF
        joint_pos={
            "l_hip_yaw.*": 0.0,
            "r_hip_yaw.*": 0.0,
            "l_hip_roll.*": 0.0,
            "r_hip_roll.*": 0.0,
            "l_hip_pitch.*": 0.0,
            "r_hip_pitch.*": 0.0,
            "l_knee.*": 0.0,
            "r_knee.*": 0.0,
            "l_ankle_roll.*": 0.0,
            "r_ankle_roll.*": 0.0,
            "l_ankle_pitch.*": 0.0,
            "r_ankle_pitch.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=80,
            damping=8,
            armature=0.04,
            friction=0.2,
            effort_limit=3.0,
            velocity_limit=5.0,
        ),
    }
)
