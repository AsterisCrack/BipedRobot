import torch
from typing import Tuple
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

@torch.jit.script
def velocity_tracking_reward(commands: torch.Tensor, base_lin_vel_b: torch.Tensor, sigma: float = 5.0):
    """
    Reward for tracking the target velocity (Negative MSE).
    Matches MuJoCo TargetReward implementation.
    """
    # Linear velocity error (x, y)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-sigma * lin_vel_error)

@torch.jit.script
def angular_velocity_tracking_reward(commands: torch.Tensor, base_ang_vel_b: torch.Tensor, sigma: float = 5.0):
    """
    Reward for tracking the target angular velocity (Negative MSE).
    """
    # Angular velocity error (yaw rate)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_b[:, 2])
    return torch.exp(-sigma * ang_vel_error)

@torch.jit.script
def height_velocity_tracking_reward(base_lin_vel_b: torch.Tensor, target_height_vel: float = 0.0, sigma: float = 10.0):
    """
    Reward for maintaining a specific height velocity.
    """
    height_vel = base_lin_vel_b[:, 2]
    error = torch.square(target_height_vel - height_vel)
    return torch.exp(-sigma * error)

@torch.jit.script
def base_height_reward(base_pos: torch.Tensor, target_height: float = 0.23, sigma: float = 20.0):
    """
    Reward for maintaining a specific base height.
    """
    base_height = base_pos[:, 2]
    error = torch.square(target_height - base_height)
    return torch.exp(-sigma * error)

@torch.jit.script
def base_height_threshold_penalty(base_pos: torch.Tensor, min_height: float = 0.1):
    """
    Penalty for maintaining a base height below a threshold.
    Returns 1.0 if height <= min_height
    """
    base_height = base_pos[:, 2]
    return torch.where(base_height <= min_height, 1.0, 0.0)

@torch.jit.script
def feet_contact_reward(contact_forces: torch.Tensor, threshold: float = 1.0):
    """
    Reward 1.0 if any foot is in contact.
    """
    forces_norm = torch.norm(contact_forces, dim=-1)
    in_contact = (forces_norm > threshold).float()
    return 1.0 * torch.any(in_contact, dim=1).float()

@torch.jit.script
def no_motion_penalty(base_lin_vel_b: torch.Tensor, epsilon: float = 0.01):
    """
    Penalty for not moving.
    Returns 1.0 if velocity is below epsilon.
    """
    vel_norm = torch.norm(base_lin_vel_b, dim=1)
    return torch.where(vel_norm < epsilon, 1.0, 0.0)

@torch.jit.script
def torque_reward(joint_efforts: torch.Tensor, joint_effort_limits: torch.Tensor, sigma: float = 0.02):
    """
    Reward for minimizing torque (Gaussian).
    Matches MuJoCo BaseReward._torque_penalty implementation (which is actually a reward).
    """
    # Normalized torque sum
    # MuJoCo: np.sum(np.abs(torques) / max_torques) / len(torques)
    normalized_torques = torch.abs(joint_efforts) / joint_effort_limits
    mean_normalized_torque = torch.mean(normalized_torques, dim=-1)
    return torch.exp(-sigma * mean_normalized_torque)

@torch.jit.script
def action_diff_reward(actions: torch.Tensor, previous_actions: torch.Tensor, sigma: float = 0.02):
    """
    Reward for smooth actions (Gaussian).
    Matches MuJoCo BaseReward._action_diff_penalty implementation.
    """
    diff = torch.sum(torch.abs(actions - previous_actions), dim=-1)
    return torch.exp(-sigma * diff)

@torch.jit.script
def acceleration_reward(joint_acc: torch.Tensor, sigma: float = 0.01):
    """
    Reward for minimizing acceleration (Gaussian).
    Matches MuJoCo BaseReward._acceleration_penalty implementation.
    """
    acc_sum = torch.sum(torch.abs(joint_acc), dim=-1)
    return torch.exp(-sigma * acc_sum)

@torch.jit.script
def orientation_penalty(projected_gravity_b: torch.Tensor, sigma: float = 5.0):
    """
    Penalty for non-flat base orientation (pitch/roll).
    """
    return torch.exp(-sigma * torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1))

@torch.jit.script
def feet_orientation_penalty(feet_orientations: torch.Tensor, sigma: float = 30.0):
    """
    Penalty for feet not being flat and facing forward (identity orientation).
    feet_orientations: (num_envs, num_feet, 4) (w, x, y, z)
    """
    # We want q to be close to (1, 0, 0, 0) or (-1, 0, 0, 0).
    # Error = 1 - q_w^2 = q_x^2 + q_y^2 + q_z^2
    # This is approximately theta^2/4 for small angles.
    
    # Extract x, y, z components
    q_vec = feet_orientations[..., 1:] # (num_envs, num_feet, 3)
    error = torch.sum(torch.square(q_vec), dim=-1) # (num_envs, num_feet)
    
    # Sum over feet
    total_error = torch.sum(error, dim=1)
    return torch.exp(-sigma * total_error)

@torch.jit.script
def torso_centering_reward(base_pos: torch.Tensor, feet_pos: torch.Tensor, sigma: float = 20.0):
    """
    Reward for keeping torso centered between feet.
    base_pos: (num_envs, 3)
    feet_pos: (num_envs, num_feet, 3)
    """
    # Compute feet midpoint
    feet_midpoint = torch.mean(feet_pos, dim=1) # (num_envs, 3)
    
    # Compute horizontal distance
    dist = torch.norm(base_pos[:, :2] - feet_midpoint[:, :2], dim=1)
    
    return torch.exp(-sigma * dist)

@torch.jit.script
def step_contact_reward(
    contact_forces: torch.Tensor,
    feet_pos: torch.Tensor,
    prev_feet_contact: torch.Tensor,
    last_step_time: torch.Tensor,
    current_time: torch.Tensor,
    contact_threshold: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the step contact reward and updates step timing.
    Returns:
        reward: The computed reward.
        new_last_step_time: Updated last step time tensor.
        new_prev_feet_contact: Updated previous contact state tensor.
    """
    # contact_forces: (num_envs, 2, 3)
    contact_forces_norm = torch.norm(contact_forces, dim=-1)
    current_contact = contact_forces_norm > contact_threshold
    
    # Detect new contacts
    just_touched = current_contact & ~prev_feet_contact
    
    # Check forward condition (assuming index 0 is Right, 1 is Left)
    # Right foot (0) forward of Left foot (1)
    right_forward = feet_pos[:, 0, 0] > feet_pos[:, 1, 0]
    left_forward = feet_pos[:, 1, 0] > feet_pos[:, 0, 0]
    
    reward = torch.zeros_like(last_step_time)
    new_last_step_time = last_step_time.clone()
    
    # Right foot step
    valid_right_step = just_touched[:, 0] & right_forward
    if torch.any(valid_right_step):
        reward[valid_right_step] = current_time[valid_right_step] - last_step_time[valid_right_step]
        new_last_step_time[valid_right_step] = current_time[valid_right_step]
        
    # Left foot step
    valid_left_step = just_touched[:, 1] & left_forward
    if torch.any(valid_left_step):
        reward[valid_left_step] = current_time[valid_left_step] - last_step_time[valid_left_step]
        new_last_step_time[valid_left_step] = current_time[valid_left_step]
        
    return reward, new_last_step_time, current_contact

@torch.jit.script
def yaw_orientation_penalty(base_quat: torch.Tensor, sigma: float = 5.0):
    """
    Penalty for non-zero yaw orientation.
    """
    # Extract yaw from quaternion
    # q = [w, x, y, z]
    w, x, y, z = base_quat.unbind(-1)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.exp(-sigma * torch.square(yaw))

@torch.jit.script
def termination_penalty(terminated: torch.Tensor):
    """
    Penalty for termination (e.g. falling).
    Returns 1.0 if terminated.
    """
    return terminated.float()

