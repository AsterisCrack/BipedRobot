import torch

@torch.jit.script
def swing_foot_height(
    feet_pos_w: torch.Tensor,
    feet_contact: torch.Tensor,
    min_height: float = 0.01,
    max_height: float = 0.05,
) -> torch.Tensor:
    # feet_pos_w: (N, F, 3)
    foot_height = feet_pos_w[:, :, 2]

    # only reward when foot is NOT in contact
    swing = (~feet_contact).float()

    # clamp reward window
    height_error = torch.clamp(foot_height - min_height, min=0.0)
    height_reward = torch.clamp(height_error, max=max_height - min_height)

    return torch.sum(height_reward * swing, dim=1)

@torch.jit.script
def track_lin_vel_xy_exp(commands: torch.Tensor, base_lin_vel_b: torch.Tensor, std: float = 0.25):
    """
    Tracking of linear velocity commands (xy axes) using exponential kernel.
    """
    # Compute error
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

@torch.jit.script
def track_ang_vel_z_exp(commands: torch.Tensor, base_ang_vel_b: torch.Tensor, std: float = 0.25):
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
def feet_air_time(first_contact: torch.Tensor, last_air_time: torch.Tensor, commands: torch.Tensor, min_airtime_threshold: float = 0.5, min_speed_command_threshold: float = 0.05):
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
def torso_centering_reward(base_pos: torch.Tensor, feet_pos: torch.Tensor, sigma: float = 1.0):
    """
    Reward for keeping torso centered between feet.
    base_pos: (num_envs, 3)
    feet_pos: (num_envs, num_feet, 3)
    """
    # Compute feet midpoint
    feet_midpoint = torch.mean(feet_pos, dim=1) # (num_envs, 3)
    
    # Compute horizontal distance
    dist_sq = torch.sum(torch.square(base_pos[:, :2] - feet_midpoint[:, :2]), dim=1)
    
    return torch.exp(-sigma * dist_sq)

@torch.jit.script
def com_support_centering_reward(
    body_com_pose_w: torch.Tensor,
    body_mass: torch.Tensor,
    feet_pos: torch.Tensor,
    feet_in_contact: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Reward for keeping the true whole-body COM centered over the support polygon.
    body_com_pose_w: (num_envs, num_bodies, 7) — world-frame COM pose per body
    body_mass:       (num_envs, num_bodies)     — mass of each body
    feet_pos:        (num_envs, 2, 3)
    feet_in_contact: (num_envs, 2) bool
    """
    # True whole-body COM (mass-weighted centroid of all body COMs)
    com_pos = body_com_pose_w[:, :, :3]
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com = (com_pos * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass  # (num_envs, 3)

    # Support center: contact-weighted average of feet; fall back to midpoint when airborne
    contact_f = feet_in_contact.float()
    contact_sum = contact_f.sum(dim=1, keepdim=True)
    no_contact = contact_sum.squeeze(1) < 0.5
    midpoint = feet_pos[:, :, :2].mean(dim=1)
    weighted = (feet_pos[:, :, :2] * contact_f.unsqueeze(-1)).sum(dim=1) / contact_sum.clamp(min=1.0)
    support_center = torch.where(no_contact.unsqueeze(-1), midpoint, weighted)

    dist_sq = torch.sum(torch.square(com[:, :2] - support_center), dim=1)
    return torch.exp(-sigma * dist_sq)

@torch.jit.script
def joint_pos_limits(joint_pos: torch.Tensor, limits_min: torch.Tensor, limits_max: torch.Tensor):
    """
    Penalize joint positions if they cross the soft limits.
    """
    # compute out of limits constraints
    out_of_limits = -(joint_pos - limits_min).clip(max=0.0)
    out_of_limits += (joint_pos - limits_max).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

@torch.jit.script
def step_length(touchdown: torch.Tensor, stride_length: torch.Tensor, commands: torch.Tensor, min_speed_command_threshold: float = 0.05):
    """
    Reward for step length on touchdown.
    touchdown: (num_envs, num_feet) bool
    stride_length: (num_envs, num_feet) float
    """
    # Mask by command speed
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    mask = (cmd_norm > min_speed_command_threshold)
    
    # Reward stride length at the moment of touchdown
    reward = torch.sum(torch.square(stride_length) * touchdown.float(), dim=1)
    
    return reward * mask

@torch.jit.script
def knee_bend_on_touchdown(
    touchdown: torch.Tensor,
    max_knee_bend: torch.Tensor,
    min_bend: float = 0.1,
    max_bend: float = 1.0,
):
    """
    Reward knee bend during swing when the foot touches down.

    touchdown: (num_envs, num_feet) bool
    max_knee_bend: (num_envs, num_feet) float, max |knee_pos - default| during swing
    """
    bend = torch.clamp(max_knee_bend - min_bend, min=0.0)
    bend = torch.clamp(bend, max=max_bend - min_bend)
    return torch.sum(bend * touchdown.float(), dim=1)

@torch.jit.script
def track_joint_pos_exp(joint_pos: torch.Tensor, ref_joint_pos: torch.Tensor, std: float = 0.5):
    """
    Reward for tracking reference joint positions using exponential kernel.
    """
    error = torch.sum(torch.square(joint_pos - ref_joint_pos), dim=1)
    return torch.exp(-error / (std**2))

@torch.jit.script
def track_joint_vel_exp(joint_vel: torch.Tensor, ref_joint_vel: torch.Tensor, std: float = 1.0):
    """
    Reward for tracking reference joint velocities using exponential kernel.
    """
    error = torch.sum(torch.square(joint_vel - ref_joint_vel), dim=1)
    return torch.exp(-error / (std**2))

@torch.jit.script
def gait_phase_contact(
    gait_phase: torch.Tensor,
    feet_in_contact: torch.Tensor,
    commands: torch.Tensor,
    min_speed: float = 0.05,
) -> torch.Tensor:
    """Reward feet contacting in sync with the gait phase clock.

    Convention: sin(phase) > 0 → right foot (index 0) in stance;
                sin(phase) < 0 → left  foot (index 1) in stance.
    Returns [0, 1] — 1 = perfect phase match, 0.5 = one foot wrong, 0 = both wrong.
    Only active when commanded planar speed > min_speed.
    """
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    speed_mask = (cmd_norm > min_speed).float()

    sin_p = torch.sin(gait_phase)
    desired_right = (sin_p + 1.0) * 0.5    # 1 when phase=π/2, 0 when phase=3π/2
    desired_left  = (-sin_p + 1.0) * 0.5   # complementary

    right_c = feet_in_contact[:, 0].float()
    left_c  = feet_in_contact[:, 1].float()

    reward = 1.0 - 0.5 * (torch.abs(desired_right - right_c) + torch.abs(desired_left - left_c))
    return reward * speed_mask

@torch.jit.script
def action_l2(actions: torch.Tensor) -> torch.Tensor:
    """Penalise large action magnitudes — pulls joints toward neutral action-space position."""
    return torch.sum(torch.square(actions), dim=1)

@torch.jit.script
def foot_separation_penalty(
    feet_pos_w: torch.Tensor,
    min_sep: float = 0.07,
) -> torch.Tensor:
    """Penalise lateral (Y-axis) foot separation below min_sep metres.
    V2 hip joints are only ±21.7 mm apart, giving 4.3 cm natural spacing at hip_roll=0;
    min_sep=0.07 m requires ~0.15 rad hip abduction for a stable wider stance."""
    sep = torch.abs(feet_pos_w[:, 0, 1] - feet_pos_w[:, 1, 1])
    return torch.clamp(min_sep - sep, min=0.0)

@torch.jit.script
def dof_pos_l2(joint_pos: torch.Tensor, default_pos: torch.Tensor) -> torch.Tensor:
    """Penalise all joint positions deviating from their defaults — uniform pose regulariser.
    Unlike action_l2, this operates in joint-space so it correctly targets the neutral pose
    regardless of whether joint limits are symmetric around zero."""
    return torch.sum(torch.square(joint_pos - default_pos), dim=1)
