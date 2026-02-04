import torch
from isaaclab.utils.math import quat_apply

@torch.jit.script
def track_lin_vel_xy_exp(commands: torch.Tensor, base_lin_vel_b: torch.Tensor, std: float = 0.5):
    """
    Tracking of linear velocity commands (xy axes) using exponential kernel.
    """
    # Compute error
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

@torch.jit.script
def track_ang_vel_z_exp(commands: torch.Tensor, base_ang_vel_b: torch.Tensor, std: float = 0.5):
    """
    Tracking of angular velocity commands (yaw) using exponential kernel.
    """
    # Compute error
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

@torch.jit.script
def lin_vel_z_l2(base_lin_vel_b: torch.Tensor):
    """
    Penalty for vertical linear velocity (L2 norm).
    """
    return torch.square(base_lin_vel_b[:, 2])

@torch.jit.script
def ang_vel_xy_l2(base_ang_vel_b: torch.Tensor):
    """
    Penalty for angular velocity on xy axes (roll and pitch).
    """
    return torch.sum(torch.square(base_ang_vel_b[:, :2]), dim=1)

@torch.jit.script
def joint_torques_l2(joint_efforts: torch.Tensor):
    """
    Penalty for joint torques (L2 norm).
    """
    return torch.sum(torch.square(joint_efforts), dim=1)

@torch.jit.script
def action_rate_l2(actions: torch.Tensor, last_actions: torch.Tensor):
    """
    Penalty for rate of change of actions (L2 norm).
    """
    return torch.sum(torch.square(actions - last_actions), dim=1)

@torch.jit.script
def feet_air_time_positive_biped(current_air_time: torch.Tensor, current_contact_time: torch.Tensor, commands: torch.Tensor, threshold: float = 0.4, min_speed_command_threshold: float = 0.1):
    """
    Reward long steps taken by the feet for bipeds.
    Rewards the agent for taking steps up to a specified threshold and also keeping one foot at a time in the air.
    """
    # Check if commands are above threshold (speed > min_speed)
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    mask = (cmd_norm > min_speed_command_threshold)
    
    in_contact = current_contact_time > 0.0
    in_mode_time = torch.where(in_contact, current_contact_time, current_air_time)
    
    # Check for single stance (exactly one foot in contact)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1)[0]
    
    # Clamp to threshold
    reward = torch.clamp(reward, max=threshold)
    
    return reward * mask

@torch.jit.script
def feet_air_time(first_contact: torch.Tensor, last_air_time: torch.Tensor, commands: torch.Tensor, min_airtime_threshold: float = 0.2, min_speed_command_threshold: float = 0.1):
    """
    Reward for airtime. Directly proportional to airtime when foot makes contact. But only if airtime more than a threshold.
    Only active if speed command is above a threshold because low speed walking may not require as much foot lift.
    
    first_contact: (num_envs, num_feet) bool tensor indicating if foot just contacted
    last_air_time: (num_envs, num_feet) float tensor with last air time duration
    """
    # Check if commands are above threshold (speed > min_speed)
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    mask = (cmd_norm > min_speed_command_threshold)
    
    # Calculate reward for each foot that just contacted
    # Reward is (air_time - threshold)
    reward = torch.sum((last_air_time - min_airtime_threshold) * first_contact, dim=1)
    
    # Apply mask
    return reward * mask

@torch.jit.script
def feet_slide(net_forces_w_history: torch.Tensor, body_lin_vel_w: torch.Tensor):
    """
    Penalize feet sliding.

    net_forces_w_history: (num_envs, history_len, num_feet, 3)
    body_lin_vel_w: (num_envs, num_feet, 3)
    """
    # Calculate contacts from history
    # norm(dim=-1) -> [N, T, F]
    # max(dim=1)[0] -> [N, F]
    contacts = torch.max(torch.max(torch.norm(net_forces_w_history, dim=-1), dim=1)[0], dim=-1)[0] > 1.0
    
    # Calculate velocity norm (xy plane)
    feet_vel_xy = torch.norm(body_lin_vel_w[:, :, :2], dim=-1)
    
    # Sum of velocity * contact
    return torch.sum(feet_vel_xy * contacts, dim=1)

@torch.jit.script
def feet_slide_with_vel(feet_contact: torch.Tensor, feet_vel_xy_norm: torch.Tensor):
    """
    Penalty for feet sliding (velocity when in contact).
    feet_contact: (num_envs, num_feet) bool/float
    feet_vel_xy_norm: (num_envs, num_feet) float of planar velocity norm
    """
    return torch.sum(feet_contact * feet_vel_xy_norm, dim=1)

@torch.jit.script
def undesired_contacts(net_contact_forces: torch.Tensor, threshold: float = 1.0):
    """
    Penalty for contacts on undesired bodies (hips, thighs, etc).
    net_contact_forces: (num_envs, num_bodies) magnitude 
                        OR (num_envs, num_bodies, 3) vector
    """
    # If input is vector (num_envs, num_bodies, 3), take norm
    if net_contact_forces.dim() == 3:
        forces = torch.norm(net_contact_forces, dim=-1)
    else:
        forces = net_contact_forces
        
    # Check if any force exceeds threshold
    return torch.any(forces > threshold, dim=-1).float()

@torch.jit.script
def joint_deviation_l1(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor):
    """
    Penalty for deviation from default joint positions (L1 norm).
    """
    return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)

@torch.jit.script
def flat_orientation_l2(projected_gravity_b: torch.Tensor):
    """
    Penalty for non-flat orientation (L2 norm of projected gravity xy).
    """
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

@torch.jit.script
def joint_pos_limits(joint_pos: torch.Tensor, limits_min: torch.Tensor, limits_max: torch.Tensor):
    """
    Penalize joint positions if they cross the soft limits.
    """
    # compute out of limits constraints
    out_of_limits = -(joint_pos - limits_min).clip(max=0.0)
    out_of_limits += (joint_pos - limits_max).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
