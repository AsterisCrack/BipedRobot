import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Get joint count
n_joints = model.nu  # exclude free joint


def mirror_observation(obs, n_joints):
    """
    Mirrors a MuJoCo observation over the Y-axis.
    
    Parameters:
    - obs: np.array, shape=(7 + n_joints + 6 + n_joints,)
    - n_joints: int, number of joints (must be even to mirror properly)
    
    Returns:
    - mirrored_obs: np.array, mirrored observation
    """
    obs = obs.copy()
    
    # Indices
    pos_start = 0
    quat_start = 3
    joint_start = 7
    vel_start = 7 + n_joints
    joint_vel_start = vel_start + 6
    
    # 1. Mirror base position (negate y)
    obs[1] *= -1
    
    # 2. Mirror orientation quaternion over Y-axis (XZ plane)
    qx, qy, qz, qw = obs[quat_start:quat_start+4]
    obs[quat_start:quat_start+4] = [-qx, qy, -qz, qw]
    
    # 3. Swap joint angles (first half <-> second half)
    joint_angles = obs[joint_start:vel_start]
    half = n_joints // 2
    obs[joint_start:vel_start] = np.concatenate([joint_angles[half:], joint_angles[:half]])*-1
    
    # Invert joints on y-axis
    obs[9] *= -1  # r_hip_x
    obs[10] *= -1  # r_knee
    obs[11] *= -1  # r_ankle_y
    obs[15] *= -1  # l_hip_x
    obs[16] *= -1  # l_knee
    obs[17] *= -1  # l_ankle_y
    
    # 4. Mirror linear velocity (negate vy)
    obs[vel_start+1] *= -1  # vy
    
    # 5. Mirror angular velocity (negate wx, wz)
    obs[vel_start+3] *= -1  # wx
    obs[vel_start+5] *= -1  # wz
    
    # 6. Swap joint velocities (first half <-> second half)
    joint_vels = obs[joint_vel_start:]
    obs[joint_vel_start:] = np.concatenate([joint_vels[half:], joint_vels[:half]])
    
    # Invert joint velocities on y-axis
    obs[joint_vel_start+1] *= -1  # r_hip_x
    obs[joint_vel_start+2] *= -1  # r_knee
    obs[joint_vel_start+3] *= -1  # r_ankle_y
    obs[joint_vel_start+7] *= -1  # l_hip_x
    obs[joint_vel_start+8] *= -1  # l_knee
    obs[joint_vel_start+9] *= -1  # l_ankle_y
    
    return obs

# Viewer setup
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Initial timestamp
    last_mirror_time = time.time()

    # Run simulation
    while viewer.is_running():
        now = time.time()

        # Randomize qpos at start
        data.qpos[0:13] = np.random.uniform(low=-0.5, high=0.5, size=13)
        data.qpos[13:] = np.random.uniform(low=0, high=0.5, size=n_joints//2)

        # Reset qvel to zero
        data.qvel[:] = 0

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)  # Show initial posture for 1s

        # Then loop: mirror every 1s
        while viewer.is_running():
            if time.time() - last_mirror_time > 1:
                # Get current state as observation
                obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
                mirrored_obs = mirror_observation(obs, n_joints)

                # Update state
                data.qpos[:] = mirrored_obs[:model.nq]
                data.qvel[:] = mirrored_obs[model.nq:]
                mujoco.mj_forward(model, data)

                last_mirror_time = time.time()

            viewer.sync()
            time.sleep(0.01)

