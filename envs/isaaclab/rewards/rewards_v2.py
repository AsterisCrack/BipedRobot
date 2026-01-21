import torch
from typing import Tuple
from isaaclab.managers import SceneEntityCfg

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
def base_height_reward(base_pos: torch.Tensor, min_height: float = 0.2):
    """
    Reward for maintaining a specific base height.
    """
    base_height = base_pos[:, 2]
    return (base_height >= min_height).float()

@torch.jit.script
def base_stability_reward(base_ang_vel_b: torch.Tensor, sigma: float = 5.0):
    """
    Reward for minimizing angular velocity (xy components).
    Renamed from angular_velocity_penalty.
    """
    ang_vel_error = torch.sum(torch.square(base_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-sigma * ang_vel_error)

@torch.jit.script
def torque_reward(joint_efforts: torch.Tensor, sigma: float = 1.0):
    """
    Reward for minimizing torque
    """
    return torch.exp(-sigma * torch.sum(torch.abs(joint_efforts), dim=-1))

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
def feet_airtime_reward(first_contact: torch.Tensor, last_air_time: torch.Tensor, commands: torch.Tensor, min_airtime_threshold: float = 0.5, min_speed_command_threshold: float = 0.1):
    """
    Reward for airtime. Directly proportional to airtime when foot makes contact. But only if airtime more than a threshold.
    Onlu active if speed command is above a threshold because low speed walking may not require as much foot lift.
    first_contact: (num_envs, num_feet) bool tensor indicating if foot just contacted
    last_air_time: (num_envs, num_feet) float tensor with last air time duration
    """
    air_time = torch.sum((last_air_time - min_airtime_threshold) * first_contact, dim=1) * (
        torch.norm(commands[:, :2], dim=1) > min_speed_command_threshold
    )
    return air_time

@torch.jit.script
def flat_orientation_reward(projected_gravity_b: torch.Tensor, sigma: float = 5.0):
    """
    Reward for flat base orientation (pitch/roll).
    Renamed from orientation_penalty.
    """
    return torch.exp(-sigma * torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1))

@torch.jit.script
def stall_penalty(base_lin_vel_b: torch.Tensor, commands: torch.Tensor, epsilon: float = 0.01):
    """
    Penalty for not moving when commanded to move.
    Returns -1.0 if velocity is below epsilon, only for non-zero commands.
    Renamed from no_motion_penalty.
    """
    vel_norm = torch.norm(base_lin_vel_b, dim=1)
    command_norm = torch.norm(commands[:, :2], dim=1)
    no_motion = (vel_norm < epsilon) & (command_norm > epsilon)
    return -1.0 * no_motion.float()


@torch.jit.script
def feet_flat_reward(feet_orientations: torch.Tensor, sigma: float = 30.0):
    """
    Reward for feet not being flat and facing forward (identity orientation).
    feet_orientations: (num_envs, num_feet, 4) (w, x, y, z)
    Renamed from feet_orientation_penalty.
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
def termination_penalty(terminated: torch.Tensor):
    """
    Penalty for termination (e.g. falling).
    Returns 1.0 if terminated.
    """
    return terminated.float()