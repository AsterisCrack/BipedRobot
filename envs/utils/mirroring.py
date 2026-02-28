import numpy as np

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
    obs[joint_start:vel_start] = np.concatenate([joint_angles[half:], joint_angles[:half]])
    
    # Invert joints on y-axis
    obs[joint_start+0] *= -1  # r_hip_x
    obs[joint_start+2] *= -1  # r_knee
    obs[joint_start+5] *= -1  # r_ankle_y
    obs[joint_start+6] *= -1  # l_hip_x
    obs[joint_start+8] *= -1  # l_knee
    obs[joint_start+11] *= -1  # l_ankle_y
    
    # 4. Mirror linear velocity (negate vy)
    obs[vel_start+1] *= -1  # vy
    
    # 5. Mirror angular velocity (negate wx, wz)
    obs[vel_start+4] *= -1  # wx
    obs[vel_start+5] *= -1  # wz
    
    # 6. Swap joint velocities (first half <-> second half)
    joint_vels = obs[joint_vel_start:]
    obs[joint_vel_start:] = np.concatenate([joint_vels[half:], joint_vels[:half]])
    
    # Invert joint velocities on y-axis
    obs[joint_vel_start+0] *= -1  # r_hip_x
    obs[joint_vel_start+2] *= -1  # r_knee
    obs[joint_vel_start+5] *= -1  # r_ankle_y
    obs[joint_vel_start+6] *= -1  # l_hip_x
    obs[joint_vel_start+8] *= -1  # l_knee
    obs[joint_vel_start+11] *= -1  # l_ankle_y
    
    return obs

def mirror_action(action, n_joints):
    """
    Mirrors a MuJoCo action over the Y-axis.
    
    Parameters:
    - action: np.array, shape=(n_joints,)
    - n_joints: int, number of joints (must be even to mirror properly)
    
    Returns:
    - mirrored_action: np.array, mirrored action
    """
    action = action.copy()
    
    # 1. Swap joint angles (first half <-> second half)
    half = n_joints // 2
    action = np.concatenate([action[half:], action[:half]])
    # Mirror joint angles on y-axis
    action[0] *= -1  # r_hip_x
    action[2] *= -1  # r_knee
    action[5] *= -1  # r_ankle_y
    action[6] *= -1  # l_hip_x
    action[8] *= -1  # l_knee
    action[11] *= -1  # l_ankle_y
    
    return action
