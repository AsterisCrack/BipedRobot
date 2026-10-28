import torch
from typing import List, Optional

@torch.jit.script
def swing_foot_height(
    foot_clearance: torch.Tensor,
    feet_contact: torch.Tensor,
    commands: torch.Tensor,
    min_clearance: float = 0.01,
    max_clearance: float = 0.07,
    min_speed_command_threshold: float = 0.05,
    swing_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reward swing-foot clearance, only while the robot is commanded to move.

    swing_valid (N, F) bool, optional: feet where it is False earn nothing even if airborne
    (the env passes swing_time_mask so a leg held up beyond swing_max_time_s stops paying).
    None = old behaviour.

    foot_clearance (N, F) is the foot height ABOVE ITS OWN RESTING STANCE HEIGHT, which the
    env measures from feet in contact. It is not the link origin's world Z. The old version
    used world Z with min_height = 0.02, but the V2 foot-link origin rests ~0.036 m above
    the ground, so an UNLOADED foot sitting flat on the floor already earned
    1.5 * (0.036 - 0.02) ~= 0.025 per step. Rocking side to side to alternately unweight
    each foot farmed it, which was the standing jitter. It also had no command gate
    (every other gait term has one), so it paid out while the robot was told to stand
    still. On rough terrain world Z added the patch elevation on top.

    min_clearance is a dead zone, so scuffing the floor earns nothing. The reward is
    linear up to max_clearance, the cap that controls how high the foot is lifted (and,
    through the leg geometry, how much the knee bends).
    """
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    moving = (cmd_norm > min_speed_command_threshold).float()
    swing = (~feet_contact).float()
    if swing_valid is not None:
        swing = swing * swing_valid.float()
    height_error = torch.clamp(foot_clearance - min_clearance, min=0.0)
    height_reward = torch.clamp(height_error, max=max_clearance - min_clearance)
    return torch.sum(height_reward * swing, dim=1) * moving

@torch.jit.script
def track_lin_vel_xy_exp(commands: torch.Tensor, base_lin_vel_b: torch.Tensor, std: float = 0.35):
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
def track_lin_vel_xy_exp_rel(commands: torch.Tensor, base_lin_vel_b: torch.Tensor,
                             std_rel: float = 0.7, std_min: float = 0.07):
    """Linear-velocity tracking with a kernel width that scales with the command.

    The fixed-std kernel pays too much for ignoring a small command: at |cmd| = 0.1 and std 0.2,
    standing still scores exp(-0.01/0.04) = 0.78. Here std_i = clamp(std_rel * |cmd_xy|_i, min=std_min)
    per env, so the same error is judged relative to what was asked (std_rel 0.7, std_min 0.07):
      standing at any |cmd| >= 0.1        -> exp(-1/std_rel^2) ~ 0.13
      walking at half the commanded speed -> exp(-0.25/std_rel^2) ~ 0.6
      zero command, 0.05 m/s drift        -> exp(-(0.05/std_min)^2) ~ 0.6
    std_min keeps the kernel from collapsing to a delta at zero command (sensor noise floor).
    """
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    std = torch.clamp(std_rel * cmd_norm, min=std_min)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / torch.square(std))

@torch.jit.script
def track_lin_vel_xy_exp_rel_target(target_xy: torch.Tensor, commands: torch.Tensor, base_lin_vel_b: torch.Tensor,
                                    std_rel: float = 0.7, std_min: float = 0.07):
    """track_lin_vel_xy_exp_rel with the kernel CENTRED on target_xy [N, >=2] instead of the command
    (track_lin_target: reference: target_xy = the gait table's base_lin_vel_b at the current
    (command, phase)). The WIDTH is still std_i = clamp(std_rel * |cmd_xy|_i, min=std_min), from the
    command, not from the target.

    Why: walking has natural body sway, and the table has it too. Per cycle its lateral velocity has
    std 0.062-0.072 m/s and its forward velocity 0.03-0.05 m/s, while the cycle MEAN equals the
    command exactly. Against the constant command, a PERFECT copy of the table scores only
      0.38 at vx = 0.1   (standing still: 0.13)
      0.28 at vx = 0.05  (standing still: 0.60, so standing beats perfect walking).
    Centred on the table's own velocity a perfect copy scores 1.0 at every command, while standing
    still (measured 0) scores the cycle mean of exp(-|target|^2 / std^2): 0.08-0.33 at the grid
    commands tested, so the walking-minus-standing gap is >= 0.67 (tests/test_track_lin_reference_target.py).
    """
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    std = torch.clamp(std_rel * cmd_norm, min=std_min)
    lin_vel_error = torch.sum(torch.square(target_xy[:, :2] - base_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / torch.square(std))

def validate_track_lin_target(mode: str, has_reference: bool, std_rel: float) -> None:
    """env_config.track_lin_target: command (default) | reference. reference needs the gait table
    (its base_lin_vel_b is the target) and a command-scaled kernel width (track_lin_std_rel > 0)."""
    if mode not in ("command", "reference"):
        raise ValueError(f"track_lin_target {mode!r}: expected command | reference")
    if mode == "reference":
        if not has_reference:
            raise ValueError("track_lin_target: reference needs the gait-table reference "
                             "(env_config.reference.npz_path); it tracks the table's base velocity")
        if not std_rel > 0.0:
            raise ValueError("track_lin_target: reference needs track_lin_std_rel > 0 (kernel width from the command)")

@torch.jit.script
def track_ang_vel_z_exp_rel(commands: torch.Tensor, base_ang_vel_b: torch.Tensor,
                            std_rel: float = 0.7, std_min: float = 0.15):
    """Yaw-rate tracking with std_i = clamp(std_rel * |wz_cmd|_i, min=std_min). See the linear
    version. With the defaults, ignoring a yaw command >= 0.3 rad/s scores exp(-1/0.49) ~ 0.13
    (the 0.15 floor makes smaller commands somewhat more lenient: 0.2 -> 0.17)."""
    std = torch.clamp(std_rel * torch.abs(commands[:, 2]), min=std_min)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / torch.square(std))

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
def feet_air_time_touchdown(first_contact: torch.Tensor, last_air_time: torch.Tensor,
                            other_in_contact: torch.Tensor, commands: torch.Tensor,
                            air_time_max: float = 0.4, air_time_min: float = 0.15,
                            min_speed_command_threshold: float = 0.05):
    """Touchdown-paid air time (feet_air_time_mode: touchdown).

    Pays clamp(last_air_time - air_time_min, 0, air_time_max - air_time_min) ONCE, on the step a
    foot lands (first_contact), and only if the other foot was on the ground for that swing
    (single stance, so a two-foot hop pays nothing) and the command is above the speed gate.
    Zero on every other step, so a foot held in the air forever earns nothing (it never lands;
    the per-step version paid every step). air_time_min is a dead zone: 70-130 ms flights (the
    ~7 touchdowns/s vibration) pay exactly 0. With 0.15 / 0.4: 0.3 s swing -> 0.15, 0.7 s -> 0.25.

    first_contact, other_in_contact: (N, F) bool. last_air_time: (N, F) s. Returns (N,).
    """
    cmd_norm = torch.norm(commands[:, :2], dim=1)
    moving = (cmd_norm > min_speed_command_threshold).float()
    paid = (first_contact & other_in_contact).float()
    air = torch.clamp(last_air_time - air_time_min, 0.0, air_time_max - air_time_min)
    return torch.sum(air * paid, dim=1) * moving

@torch.jit.script
def swing_time_mask(current_air_time: torch.Tensor, max_time: float = 0.5):
    """(N, F) bool: True while the foot's CURRENT swing has lasted <= max_time seconds. A foot
    held up longer than a real step stops earning swing_foot_height (anti-farming)."""
    return current_air_time <= max_time

@torch.jit.script
def feet_slide(net_forces_w_history: torch.Tensor, body_lin_vel_w: torch.Tensor,
               extra_contact: Optional[torch.Tensor] = None):
    """
    Penalize feet sliding.

    net_forces_w_history: (num_envs, history_len, num_feet, 3)
    body_lin_vel_w: (num_envs, num_feet, 3)
    extra_contact: optional (num_envs, num_feet), combined with the force-based contact flag
        by max. bool (clearance_contact: a foot dragged on the floor with < 1 N load still counts
        as in contact) = logical OR; float in [0, 1] (clearance_contact_soft) = the soft contact
        weight c = max(force_flag, extra) on the planar speed. None = the old force-only behaviour.
    """
    # Calculate contacts from history
    # norm(dim=-1) -> [N, T, F]
    # max(dim=1)[0] -> [N, F]
    contacts = (torch.max(torch.max(torch.norm(net_forces_w_history, dim=-1), dim=1)[0], dim=-1)[0] > 1.0).to(torch.float32)
    if extra_contact is not None:
        contacts = torch.maximum(contacts, extra_contact.to(torch.float32))

    # Calculate velocity norm (xy plane)
    feet_vel_xy = torch.norm(body_lin_vel_w[:, :, :2], dim=-1)
    
    # Sum of velocity * contact
    return torch.sum(feet_vel_xy * contacts, dim=1)

@torch.jit.script
def feet_slide_with_vel(feet_contact: torch.Tensor, feet_vel_xy_norm: torch.Tensor):
    """
    Penalty for feet sliding (velocity when in contact).
    feet_contact: (num_envs, num_feet) bool/float (a float in [0, 1] weights the speed: soft contact)
    feet_vel_xy_norm: (num_envs, num_feet) float of planar velocity norm
    """
    return torch.sum(feet_contact * feet_vel_xy_norm, dim=1)

@torch.jit.script
def undesired_contacts(
    net_contact_forces: torch.Tensor,
    threshold: float = 1.0,
    saturation: float = 20.0,
):
    """Graded penalty for contacts on undesired bodies (hips, thighs, etc).

    net_contact_forces: (num_envs, num_bodies) magnitude
                        OR (num_envs, num_bodies, 3) vector

    Was `any(force > threshold)`, i.e. binary. That is a cliff: a 2 N brush cost exactly as
    much as a 40 N impact, and the gradient was zero on both sides of the threshold so the
    policy got no signal about how to reduce it. It matters more now the gait is dynamic --
    V2's hips are only 4.3 cm apart, so the legs legitimately pass close during swing and a
    light graze should be nearly free.

    Now each body contributes `clamp((force - threshold) / saturation, 0, 1)`. A body at or
    above `threshold + saturation` still costs 1.0, exactly as before, so hard impacts are
    deterred identically and existing weights keep their meaning -- but a light touch now costs
    proportionally little and there is a usable gradient in between.
    """
    if net_contact_forces.dim() == 3:
        forces = torch.norm(net_contact_forces, dim=-1)
    else:
        forces = net_contact_forces

    over = (forces - threshold) / saturation
    return torch.sum(torch.clamp(over, min=0.0, max=1.0), dim=-1)

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
def com_support_offset(
    body_com_pose_w: torch.Tensor,
    body_mass: torch.Tensor,
    feet_pos: torch.Tensor,
    feet_in_contact: torch.Tensor,
) -> torch.Tensor:
    """World-frame XY vector from the support centre to the whole-body CoM, (num_envs, 2).

    The support centre is the contact-weighted mean of the feet, falling back to the feet's
    midpoint when airborne. In single support it is the stance foot.
    """
    com_pos = body_com_pose_w[:, :, :3]
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com = (com_pos * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass  # (num_envs, 3)

    contact_f = feet_in_contact.float()
    contact_sum = contact_f.sum(dim=1, keepdim=True)
    no_contact = contact_sum.squeeze(1) < 0.5
    midpoint = feet_pos[:, :, :2].mean(dim=1)
    weighted = (feet_pos[:, :, :2] * contact_f.unsqueeze(-1)).sum(dim=1) / contact_sum.clamp(min=1.0)
    support_center = torch.where(no_contact.unsqueeze(-1), midpoint, weighted)
    return com[:, :2] - support_center


def com_support_distance(
    body_com_pose_w: torch.Tensor,
    body_mass: torch.Tensor,
    feet_pos: torch.Tensor,
    feet_in_contact: torch.Tensor,
    foot_margin: float = 0.03,
) -> torch.Tensor:
    """Horizontal distance from the whole-body CoM to the SUPPORT POLYGON, 0 when inside.

    With two feet the polygon is the segment between them, so this is the point-to-segment
    distance; in single support the segment degenerates to the stance foot. foot_margin is
    the foot's own half-extent, so a CoM over any part of a planted foot reads 0.

    Why not distance to the support CENTRE (the previous two attempts): lifting a foot
    REQUIRES first shifting the CoM over the other foot, which is a ~half-stance-width
    excursion from the centre but is perfectly stable -- the CoM is directly over a planted
    foot. Penalising it taxed the weight transfer that begins every single step. At std_lat
    0.04 that tax was ~0.09/step against a swing reward of ~0.045/step, so stepping stayed
    a net loss and foot lift never moved off ~6 mm across two rounds of retuning.

    This form is direction-agnostic and needs no fore/aft split: a CoM between the feet is
    free whatever the stance, while a genuine fall (CoM outside the polygon, in any
    direction) grows the distance immediately.
    """
    com_pos = body_com_pose_w[:, :, :3]
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com = ((com_pos * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass)[:, :2]

    a, b = feet_pos[:, 0, :2], feet_pos[:, 1, :2]
    c0 = feet_in_contact[:, 0:1].float()
    c1 = feet_in_contact[:, 1:2].float()
    # Collapse the segment onto whichever feet are planted. Airborne (neither in contact)
    # keeps the full segment, matching com_support_offset's midpoint fallback.
    both = ((c0 + c1) > 1.5).float()
    none = ((c0 + c1) < 0.5).float()
    keep_seg = torch.clamp(both + none, max=1.0)
    a_eff = torch.where(keep_seg > 0.5, a, c0 * a + c1 * b)
    b_eff = torch.where(keep_seg > 0.5, b, c0 * a + c1 * b)

    ab = b_eff - a_eff
    denom = torch.sum(ab * ab, dim=1, keepdim=True).clamp(min=1e-12)
    t = (torch.sum((com - a_eff) * ab, dim=1, keepdim=True) / denom).clamp(0.0, 1.0)
    closest = a_eff + t * ab
    dist = torch.norm(com - closest, dim=1)
    return torch.clamp(dist - foot_margin, min=0.0)


@torch.jit.script
def com_support_centering_reward(com_dist: torch.Tensor, std: float = 0.04) -> torch.Tensor:
    """exp(-0.5 (d/std)^2) on the distance OUTSIDE the support polygon (see above)."""
    return torch.exp(-0.5 * torch.square(com_dist / std))


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
    root_quat_w: torch.Tensor,
    min_sep: float = 0.07,
) -> torch.Tensor:
    """Penalise lateral foot separation below min_sep metres, in the BODY frame.

    The lateral axis is body +Y, verified from the URDF: l_hip_yaw sits at
    origin y = +0.021672 and r_hip_yaw at y = -0.021672 relative to base_link, i.e. 4.33 cm
    apart along Y (matching the "+-21.7 mm" figure quoted elsewhere). +Y is left. Standard
    REP-103: X forward, Y left, Z up.

    This must be body frame, not world. Reset yaw is uniform over +-pi and heading commands
    keep rotating the robot, so a world-Y measurement collapses to ~0 whenever the robot faces
    world +-Y -- charging the full penalty for a perfectly good stance roughly half the time.
    At weight -3.0 that is a large spurious signal.

    (An earlier body-frame attempt was reverted after appearing to read ~0. That run also
    carried three other changes and a collapsing policy, and Reward_Terms are deflated by
    actual/max episode length -- so a short episode alone shrinks the logged value ~10x. The
    formula was fine; the diagnosis was not.)

    V2's natural spacing is only 4.3 cm at hip_roll=0, so min_sep=0.07 m asks for roughly
    0.15 rad of hip abduction.
    """
    # Body +Y expressed in world = second column of R(q), for quaternion [w, x, y, z].
    qw = root_quat_w[:, 0]; qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]; qz = root_quat_w[:, 3]
    body_y_w = torch.stack([
        2.0 * (qx * qy - qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz + qw * qx),
    ], dim=-1)
    feet_vec = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    sep = torch.abs(torch.sum(feet_vec * body_y_w, dim=-1))
    return torch.clamp(min_sep - sep, min=0.0)

@torch.jit.script
def _body_y_w(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Body +Y expressed in world = second column of R(q), quaternion [w, x, y, z]. (N, 3)."""
    qw = root_quat_w[:, 0]; qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]; qz = root_quat_w[:, 3]
    return torch.stack([
        2.0 * (qx * qy - qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz + qw * qx),
    ], dim=-1)


@torch.jit.script
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v (..., 3) by quaternion q (..., 4) [w, x, y, z] (body -> world)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


@torch.jit.script
def foot_flat_roll_l2(
    foot_quat_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    in_contact: torch.Tensor,
) -> torch.Tensor:
    """Roll tilt of the foot soles, squared, summed over feet that are in contact. Returns (N,).

    foot_quat_w (N, F, 4) [w, x, y, z]; root_quat_w (N, 4); in_contact (N, F) bool.

    The foot link frame is the ankle_pitch joint frame and the sole is the plane link z = -31.6 mm
    (mesh z spans -31.6 .. +15.8 mm; every joint origin in the URDF has rpy = 0, so link z is
    world up at the zero pose). The link +z axis is therefore the sole normal, pointing up out of
    the foot. n_w = R_foot @ e_z is the third column of R(q).

    Roll tilt = component of n_w along the robot's lateral direction, i.e. sin(roll). The lateral
    direction is body +Y (as in foot_separation_penalty) PROJECTED ONTO THE HORIZONTAL PLANE and
    renormalised: the plain 3D body Y of a rolled torso is not horizontal, and a foot lying flat
    on the floor under a torso rolled by 5 degrees would otherwise read sin(5 deg) of "foot
    tilt". Projected, a flat foot reads exactly 0 whatever the torso does. The result is
    sin(roll)^2: a 14 degree roll costs 0.058 per foot in contact. Airborne feet cost 0.
    """
    qw = foot_quat_w[..., 0]; qx = foot_quat_w[..., 1]
    qy = foot_quat_w[..., 2]; qz = foot_quat_w[..., 3]
    n_w = torch.stack([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ], dim=-1)                                              # (N, F, 3)
    by = _body_y_w(root_quat_w)                             # (N, 3)
    lat = torch.stack([by[:, 0], by[:, 1]], dim=-1)
    lat = lat / torch.norm(lat, dim=-1, keepdim=True).clamp(min=1e-6)   # (N, 2) horizontal
    tilt = n_w[..., 0] * lat[:, 0:1] + n_w[..., 1] * lat[:, 1:2]        # (N, F) = sin(roll)
    return torch.sum(torch.square(tilt) * in_contact.float(), dim=1)


@torch.jit.script
def foot_flat_roll_l1(
    foot_quat_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    in_contact: torch.Tensor,
) -> torch.Tensor:
    """|sin(sole roll)| summed over feet in contact (same geometry as foot_flat_roll_l2). Returns (N,).

    Why L1 (2026-10-25): sin^2 has no gradient near flat, so at weight -5 a 5 deg outer-edge stance cost only
    ~0.04/step and the policy kept riding the edge (the ankle axis sits over the sole's inner edge, so the outer
    edge is a rigid line support in sim; the real EVA-foam sole edge creeps and the robot tipped). |sin| pulls with
    the same force at 1 deg as at 20 deg, all the way to flat."""
    qw = foot_quat_w[..., 0]; qx = foot_quat_w[..., 1]
    qy = foot_quat_w[..., 2]; qz = foot_quat_w[..., 3]
    n_x = 2.0 * (qx * qz + qw * qy)
    n_y = 2.0 * (qy * qz - qw * qx)
    by = _body_y_w(root_quat_w)                             # (N, 3)
    lat = torch.stack([by[:, 0], by[:, 1]], dim=-1)
    lat = lat / torch.norm(lat, dim=-1, keepdim=True).clamp(min=1e-6)
    tilt = n_x * lat[:, 0:1] + n_y * lat[:, 1:2]            # (N, F) = sin(roll)
    return torch.sum(torch.abs(tilt) * in_contact.float(), dim=1)


@torch.jit.script
def foot_sole_centre_lateral_sep(
    feet_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    sole_x: float = 0.0163,
    sole_lat: float = 0.0211,
    sole_z: float = -0.0316,
) -> torch.Tensor:
    """SIGNED lateral (body +Y) distance, left sole centre minus right sole centre, metres. (N,).

    feet_pos_w (N, 2, 3) is [right, left] link ORIGINS (the ankle_pitch joint), foot_quat_w
    (N, 2, 4). The sole centre sits at (+sole_x, +-sole_lat, sole_z) in the foot link frame.
    The lateral sign differs per foot, derived from Robot.urdf / the foot STLs (link frame =
    mesh * 0.001 + visual origin): the LEFT foot mesh spans y = +2.0 .. +39.3 mm, sole-face
    centre (16.3, +21.1, -31.6); the RIGHT foot spans y = -39.3 .. -2.0 mm, centre
    (16.3, -21.1, -31.6). So the sole centre lies OUTBOARD of the origin: sign +1 for left
    (index 1), -1 for right (index 0). Positive result = left foot on the left, as expected;
    crossed feet read negative.
    """
    # Built with full_like/stack: a torch.tensor(...) constant or a scripted `off[...] = float` each
    # copies a CPU value to the device, which syncs (this runs every step). Same values: right -lat, left +lat.
    one = feet_pos_w[:, 0, 0]
    lat = torch.stack([torch.full_like(one, -sole_lat), torch.full_like(one, sole_lat)], dim=1)   # (N, 2)
    off = torch.stack([torch.full_like(lat, sole_x), lat, torch.full_like(lat, sole_z)], dim=-1)  # (N, 2, 3)
    centres = feet_pos_w + _quat_rotate(foot_quat_w, off)
    vec = centres[:, 1, :] - centres[:, 0, :]
    return torch.sum(vec * _body_y_w(root_quat_w), dim=-1)


@torch.jit.script
def foot_separation_sole_band_penalty(
    feet_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    target: float = 0.095,
    tol: float = 0.025,
    scale: float = 0.02,
    sole_x: float = 0.0163,
    sole_lat: float = 0.0211,
    sole_z: float = -0.0316,
) -> torch.Tensor:
    """foot_separation_mode: sole_band. Two-sided band on the SOLE-CENTRE lateral separation.

    penalty = clamp((|sep - target| - tol) / scale, 0, 3)^2, so it is 0 inside
    [target - tol, target + tol] (0.07 .. 0.12 m by default), then grows quadratically both for
    too narrow (feet braced against each other, 0.04 m -> 2.25) and too wide (0.16 m -> 4.0).
    Max 9. Unlike foot_separation_penalty this measures where the SOLES are, not the ankle_pitch
    link origin (which sits 31.6 mm above the sole and 2 mm inside the medial edge, so it moves
    only 1.3 mm per degree of ankle roll while the sole moves ~7x more).
    """
    sep = foot_sole_centre_lateral_sep(feet_pos_w, foot_quat_w, root_quat_w,
                                       sole_x, sole_lat, sole_z)
    return torch.square(torch.clamp((torch.abs(sep - target) - tol) / scale, min=0.0, max=3.0))


@torch.jit.script
def _planar_obb(pos_w: torch.Tensor, quat_w: torch.Tensor, centre_offset: List[float]):
    """Horizontal-plane footprint frame of one foot. pos_w (N, 3), quat_w (N, 4) [w, x, y, z].

    Returns (centre (N, 2), ax (N, 2), ay (N, 2)): the sole centre pos + R @ centre_offset
    projected to xy, and the unit foot-forward / foot-left axes in the world horizontal plane.
    The axes come from the foot's YAW (atan2 of the link +X axis projected to the floor), so a
    pitched or rolled foot still yields an orthonormal planar frame (atan2(0, 0) = 0 is the
    harmless fallback for a foot standing on its toe)."""
    one = pos_w[:, 0]   # full_like/stack, not `off[:, k] = float` (a scripted CPU->device copy = a sync)
    off = torch.stack([torch.full_like(one, centre_offset[0]), torch.full_like(one, centre_offset[1]),
                       torch.full_like(one, centre_offset[2])], dim=-1)
    centre = (pos_w + _quat_rotate(quat_w, off))[:, 0:2]
    w = quat_w[:, 0]
    x = quat_w[:, 1]
    y = quat_w[:, 2]
    z = quat_w[:, 3]
    # link +X axis in world = first column of R(q)
    yaw = torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    ax = torch.stack([c, s], dim=-1)
    ay = torch.stack([-s, c], dim=-1)
    return centre, ax, ay


@torch.jit.script
def _obb_corners(centre: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    """(N, 4, 2) corners of an oriented rectangle."""
    ex = ax * hx
    ey = ay * hy
    return torch.stack([centre + ex + ey, centre + ex - ey, centre - ex - ey, centre - ex + ey], dim=1)


@torch.jit.script
def _point_obb_dist(p: torch.Tensor, centre: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor,
                    hx: float, hy: float) -> torch.Tensor:
    """Euclidean distance (N, 4) from points p (N, 4, 2) to the (filled) rectangle; 0 inside."""
    d = p - centre.unsqueeze(1)
    qx = torch.abs(torch.sum(d * ax.unsqueeze(1), dim=-1)) - hx
    qy = torch.abs(torch.sum(d * ay.unsqueeze(1), dim=-1)) - hy
    return torch.sqrt(torch.square(torch.clamp(qx, min=0.0)) + torch.square(torch.clamp(qy, min=0.0)))


@torch.jit.script
def foot_footprint_gap(
    pos_l: torch.Tensor,
    quat_l: torch.Tensor,
    pos_r: torch.Tensor,
    quat_r: torch.Tensor,
    centre_offset: List[float],
    half_extents: List[float],
) -> torch.Tensor:
    """SIGNED horizontal-plane gap [m] between the two foot footprints (N,). Terrain-independent.

    pos_* (N, 3) foot LINK origins, quat_* (N, 4) [w, x, y, z] (world), centre_offset = [x, lateral
    magnitude, z] of the sole centre in the foot link frame (cfg.foot_sole_offset_m; the lateral
    sign is +y for the left foot, -y for the right, as in foot_sole_centre_lateral_sep),
    half_extents = [hx, hy] of the footprint rectangle about that centre (cfg.foot_sole_half_extents_m).

    Each foot is an oriented rectangle in the world horizontal plane (see _planar_obb). Apart:
    the exact minimum 2D distance between the rectangles (smallest vertex-to-rectangle distance
    over both rectangles' corners). Overlapping: minus the penetration depth, i.e. the SAT
    minimum-translation distance (max over the 4 edge normals of |d.n| - (r_a + r_b), which is <= 0
    exactly when they overlap). The two branches meet at 0 when touching.

    It is 2D: height is ignored, so a swing foot passing OVER the other foot's footprint reads <= 0
    although the 3D geometry is apart. The stance-width band keeps that rare (see CHANGES.md).
    """
    hx = half_extents[0]
    hy = half_extents[1]
    off_l = [centre_offset[0], centre_offset[1], centre_offset[2]]
    off_r = [centre_offset[0], -centre_offset[1], centre_offset[2]]
    ca, axa, aya = _planar_obb(pos_l, quat_l, off_l)
    cb, axb, ayb = _planar_obb(pos_r, quat_r, off_r)
    d = cb - ca
    # SAT: separation along each of the four edge normals
    sep = torch.full_like(d[:, 0], -1.0e9)
    for ax_ in (axa, aya, axb, ayb):
        proj = torch.abs(torch.sum(d * ax_, dim=-1))
        ra = hx * torch.abs(torch.sum(axa * ax_, dim=-1)) + hy * torch.abs(torch.sum(aya * ax_, dim=-1))
        rb = hx * torch.abs(torch.sum(axb * ax_, dim=-1)) + hy * torch.abs(torch.sum(ayb * ax_, dim=-1))
        sep = torch.maximum(sep, proj - ra - rb)
    corners_a = _obb_corners(ca, axa, aya, hx, hy)
    corners_b = _obb_corners(cb, axb, ayb, hx, hy)
    dist = torch.minimum(
        _point_obb_dist(corners_a, cb, axb, ayb, hx, hy).amin(dim=1),
        _point_obb_dist(corners_b, ca, axa, aya, hx, hy).amin(dim=1),
    )
    return torch.where(sep > 0.0, dist, sep)


@torch.jit.script
def foot_proximity_penalty(gap: torch.Tensor, margin_m: float) -> torch.Tensor:
    """clamp((margin - gap) / margin, 0, 2)^2 (N,): 0 for gap >= margin, 1 when touching, up to 4
    when overlapping by one margin or more. Meant for a NEGATIVE weight (foot_proximity)."""
    return torch.square(torch.clamp((margin_m - gap) / max(margin_m, 1e-6), min=0.0, max=2.0))

@torch.jit.script
def feet_loaded_flag(force_history: torch.Tensor) -> torch.Tensor:
    """(N, F) bool: foot carries load, |F| > 1 N at any history step and any filter body. Same
    quantity as the contact flag inside feet_slide. force_history = contact sensor
    force_matrix_w_history[:, :, feet_ids, :], (N, T, F, M, 3) (flat terrain). On rough terrain
    (no filter prims) the env uses the net force instead: |net_forces_w| > 1 N."""
    return torch.max(torch.max(torch.norm(force_history, dim=-1), dim=1)[0], dim=-1)[0] > 1.0

@torch.jit.script
def foot_sole_slip_speed(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
    centre_offset: List[float],
    half_extents: List[float],
    tau_m: float = 0.002,
):
    """(slip_speed (N, F), sole_low_z (N, F)): planar speed [m/s] of each foot's SOLE at the contact, rolling
    excluded, and the world height [m] of its lowest sole corner.

    pos_w, lin_vel_w, ang_vel_w (N, F, 3) and quat_w (N, F, 4) [w, x, y, z]: world pose and
    velocity of the foot LINK ORIGIN (the ankle-pitch frame, 31.6 mm above the sole), feet ordered
    [right, left] (F = 2). centre_offset = [x, lateral magnitude, z] of the sole centre in the foot
    frame (cfg.foot_sole_offset_m; lateral sign -1 right / +1 left, as foot_footprint_gap and
    foot_sole_centre_lateral_sep), half_extents = [hx, hy] of the sole rectangle (cfg.foot_sole_half_extents_m).

    The 4 sole corners have world velocity v_c = v_origin + w x (p_c - p_origin). They are weighted
    toward the lowest ones, w_c = exp(-(z_c - min_c z_c) / tau_m), and
    slip = sum_c w_c |v_c,xy| / sum_c w_c. A flat foot weighs all 4 equally (= the sole-centre
    speed); a foot rolling on its heel or toe edge is represented by that edge's corners, which
    stay ~still when it rolls without slipping (a corner 1 mm higher still counts 61 %, 5 mm 8 %).
    sole_low_z = min_c z_c is the pitch-independent foot height (a foot on its toe edge reads 0 above the floor,
    while the link origin rises 18 mm, 31.6 -> 49.7 mm at the table's vx 0.2 toe-off): the env keeps its last
    loaded value as a datum and uses the difference as the clearance of foot_drag.

    Why not the link origin (feet_slide): with the gait table's heel-toe rocker the origin moves at
    omega * 31.6 mm even when the sole rolls without slipping. MuJoCo, policy
    gaitref_2026-10-22_18-48-09: true contact-point slip (force-weighted) ~15 mm/s mean at vx 0.1-0.2 and
    20 mm/s at vy 0.1, origin speed 25-30 mm/s: about half of the old "slip" was rolling. (Isaac's feet_slide
    measured 76-79 mm/s: robot.data.body_lin_vel_w is the foot COM frame, and the soft contact score also counts
    unloaded feet within a few mm of the floor.) Calibration table: CHANGES.md, "Feet sliding" (2026-10-23).
    """
    assert pos_w.shape[1] == 2
    one = pos_w[:, :, 0]                                                  # (N, F)
    sign = torch.stack([-torch.ones_like(one[:, 0]), torch.ones_like(one[:, 0])], dim=1)   # right -, left +
    ox = centre_offset[0]
    oy = centre_offset[1]
    oz = centre_offset[2]
    hx = half_extents[0]
    hy = half_extents[1]
    cx = torch.stack([torch.full_like(one, ox + hx), torch.full_like(one, ox + hx),
                      torch.full_like(one, ox - hx), torch.full_like(one, ox - hx)], dim=-1)   # (N, F, 4)
    cy = torch.stack([sign * oy + hy, sign * oy - hy, sign * oy - hy, sign * oy + hy], dim=-1)
    cz = torch.full_like(cx, oz)
    local = torch.stack([cx, cy, cz], dim=-1)                                                # (N, F, 4, 3)
    q = quat_w.unsqueeze(2).expand(-1, -1, 4, -1)
    r = _quat_rotate(q, local)                                                               # corner - origin, world
    v = lin_vel_w.unsqueeze(2) + torch.cross(ang_vel_w.unsqueeze(2).expand(-1, -1, 4, -1), r, dim=-1)
    speed = torch.norm(v[..., :2], dim=-1)                                                   # (N, F, 4)
    z = pos_w[:, :, 2:3] + r[..., 2]                                                         # corner heights
    z_low = z.amin(dim=-1, keepdim=True)
    w = torch.exp(-(z - z_low) / max(tau_m, 1e-6))
    return torch.sum(w * speed, dim=-1) / torch.sum(w, dim=-1), z_low[..., 0]

@torch.jit.script
def foot_slip(slip_speed: torch.Tensor, loaded: torch.Tensor) -> torch.Tensor:
    """foot_slip penalty (N,) [m/s] = sum over feet of sole slip speed x load flag.
    slip_speed (N, F) from foot_sole_slip_speed, loaded (N, F) bool/float. Negative weight."""
    return torch.sum(slip_speed * loaded.to(slip_speed.dtype), dim=1)

@torch.jit.script
def foot_drag(slip_speed: torch.Tensor, clearance: torch.Tensor, loaded: torch.Tensor,
              h_drag_m: float) -> torch.Tensor:
    """foot_drag penalty (N,) [m/s]: scuffing = a foot that is NOT loaded, low and moving.
    sum over feet of (1 - clamp(clearance / h_drag, 0, 1)) x slip_speed x (1 - loaded).
    clearance (N, F): lowest sole corner above its height at the foot's last loaded step (foot_sole_slip_speed's
    sole_low_z minus the env's datum); factor 1 at or below the floor, linear to 0 at h_drag_m. slip_speed is the
    lowest-corner-weighted sole speed (foot_sole_slip_speed). Negative weight.

    Why not the env's swing clearance (link origin above its last in-contact height, as swing_foot_height /
    imitation_foot_height): the gait table's toe-off pitches the foot, the origin is 49.7 mm high at the last loaded
    step (31.6 flat) and then reads -5..-11 mm for most of the swing, i.e. permanently "low": measured on the table
    kinematics (vx 0.2) that gave 270-350 mm/s of drag on the reference itself. The sole-corner height has no such bias."""
    low = 1.0 - torch.clamp(clearance / max(h_drag_m, 1e-6), min=0.0, max=1.0)
    return torch.sum(low * slip_speed * (1.0 - loaded.to(slip_speed.dtype)), dim=1)


@torch.jit.script
def leg_contact_l2(
    body_forces: torch.Tensor,
    thresh_n: float = 0.5,
    sat_n: float = 5.0,
) -> torch.Tensor:
    """Inter-leg contact penalty. body_forces (N, B) force magnitudes [N] on the leg bodies
    (non-floor part). Per body clamp((F - thresh_n) / (sat_n - thresh_n), 0, 1)^2, summed.
    Free below thresh_n (sensor noise), saturating at sat_n, so bracing on the other leg costs
    the same whether it is 5 N or 50 N but a graze costs next to nothing."""
    x = torch.clamp((body_forces - thresh_n) / max(sat_n - thresh_n, 1e-6), min=0.0, max=1.0)
    return torch.sum(torch.square(x), dim=-1)

@torch.jit.script
def stand_pose_l2(joint_pos: torch.Tensor, ref_joint_pos: torch.Tensor,
                  standing_mask: torch.Tensor) -> torch.Tensor:
    """Non-saturating stand-pose anchor: sum_j (q_j - q_ref_j)^2 [rad^2] while the reference is standing, else 0.

    joint_pos, ref_joint_pos: (N, J) in the same (robot) joint order; standing_mask: (N,) float (1 = the
    gait reference is on its stand frame, 0 = walking). Meant for a NEGATIVE weight (a penalty).

    Why: at command 0 the run gaitref_2026-10-21_07-17-11 (98.3M steps) left the stand pose for a braced
    stance (per-joint deviation [-28 -18 -4 -13 33 15 26 24 -22 20 2 19] deg, knees on their +5 deg stop,
    legs pressing each other with 8-15 N, sole gap 50 mm). Every stand anchor is flat there:
    imitation_joint_pos = exp(-mean(e^2)/0.15^2) ~ 0.003, imitation_foot_pos ~ 0, leg_contact_l2 and
    undesired_contacts saturate at 5 N, so nothing leads back. A quadratic never saturates:
    at the observed brace sum(e^2) ~ 1.5 rad^2 (1.56 from the rounded numbers above; weight -1.0 costs ~1.5/step), 2 deg on every joint
    costs ~0.015, walking envs 0 (the mask), so the walking gait is untouched."""
    return torch.sum(torch.square(joint_pos - ref_joint_pos), dim=-1) * standing_mask


def validate_stand_pose_l2(weight: float, has_reference: bool) -> None:
    """stand_pose_l2 compares with the gait table's stand frame, so a nonzero weight needs the table."""
    if float(weight) != 0.0 and not has_reference:
        raise ValueError("stand_pose_l2 != 0 needs the gait-table reference (env_config.reference.npz_path): "
                         "it penalises the distance to the table's stand-frame joint pose")


@torch.jit.script
def angular_momentum_l2(body_ang_vel_w: torch.Tensor, body_mass: torch.Tensor) -> torch.Tensor:
    """Penalize mass-weighted body angular momentum to reduce limb wobble.
    body_ang_vel_w: (N, num_bodies, 3) — world-frame angular velocity per body
    body_mass: (N, num_bodies) or (1, num_bodies) broadcast
    """
    ang_vel_sq = torch.sum(torch.square(body_ang_vel_w), dim=-1)  # (N, num_bodies)
    return torch.sum(ang_vel_sq * body_mass, dim=-1)

def flight(feet_contact: torch.Tensor) -> torch.Tensor:
    """1 while neither foot touches the ground (product of the per-foot no-contact scores), else 0.

    feet_contact: (N, F) bool or float in [0, 1] (the reward-side contact flag / soft score).
    The gait table has no flight phase, so walking pays nothing. Targets the hop-and-twist push
    recovery (both feet off for 4-8 steps, whole body yawing 5-6 rad/s) that lands in sim and
    fell on hardware 2026-10-27.
    """
    return torch.prod(1.0 - feet_contact.float(), dim=1)

@torch.jit.script
def upright(projected_gravity_b: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    """Reward uprightness: 1.0 when perfectly vertical, decays with tilt.
    std=0.3 → reward≈0.37 at ~17° tilt."""
    tilt_sq = torch.sum(projected_gravity_b[:, :2] ** 2, dim=1)
    return torch.exp(-tilt_sq / (std ** 2))

@torch.jit.script
def dof_pos_l2(joint_pos: torch.Tensor, default_pos: torch.Tensor) -> torch.Tensor:
    """Penalise all joint positions deviating from their defaults — uniform pose regulariser.
    Unlike action_l2, this operates in joint-space so it correctly targets the neutral pose
    regardless of whether joint limits are symmetric around zero."""
    return torch.sum(torch.square(joint_pos - default_pos), dim=1)


@torch.jit.script
def imitation_mse_exp(value: torch.Tensor, ref: torch.Tensor, std: float) -> torch.Tensor:
    """exp(-mean((value - ref)^2) / std^2) over the last dim: 1.0 on the reference, decays with error."""
    return torch.exp(-torch.mean(torch.square(value - ref), dim=-1) / (std ** 2))


@torch.jit.script
def imitation_contact_match(feet_in_contact: torch.Tensor, ref_contact: torch.Tensor,
                            soft: bool = False, agg_min: bool = False) -> torch.Tensor:
    """Per-foot contact match against the reference, combined over the feet by mean (default) or min.

    soft=False (default, old behaviour): binary, the foot's contact flag > 0.5 equals
    (ref_contact > 0.5) -> 1 else 0.
    soft=True: feet_in_contact is the soft score c in [0, 1] (soft_contact_score) and the match is
    1 - |c - ref_contact|, which has a gradient through the 0-6 mm lift instead of a 4 mm cliff.
    agg_min=True: the worse foot decides (a limp, one foot matched and one not, scores 0, not 0.5)."""
    if soft:
        match = 1.0 - torch.abs(torch.clamp(feet_in_contact.to(torch.float32), 0.0, 1.0) - ref_contact)
    else:
        in_contact = feet_in_contact.to(torch.float32) > 0.5
        match = (in_contact == (ref_contact > 0.5)).to(torch.float32)
    if agg_min:
        return torch.min(match, dim=-1)[0]
    return torch.mean(match, dim=-1)


@torch.jit.script
def clearance_contact(clearance: torch.Tensor, thresh_m: float) -> torch.Tensor:
    """(N, F) bool: foot sole is within thresh_m of the ground it last stood on.

    clearance: (N, F) [m], foot-origin height above the foot's own last in-contact height (the
    swing_foot_height datum; ~0 for a foot resting on the floor even when unloaded). thresh_m <= 0
    disables the test (all False), which is the old force-only behaviour."""
    if thresh_m <= 0.0:
        return torch.zeros_like(clearance, dtype=torch.bool)
    return clearance < thresh_m


@torch.jit.script
def clearance_contact_soft(clearance: torch.Tensor, thresh_m: float) -> torch.Tensor:
    """(N, F) float in [0, 1]: clamp(1 - clearance / thresh_m, 0, 1). 1 on the ground (or below
    it), 0.5 at thresh_m / 2, 0 from thresh_m up -- the soft version of clearance_contact, with no
    cliff at the threshold. thresh_m <= 0 disables it (all 0, the force-only behaviour)."""
    if thresh_m <= 0.0:
        return torch.zeros_like(clearance)
    return torch.clamp(1.0 - clearance / thresh_m, 0.0, 1.0)


@torch.jit.script
def soft_contact_score(force_contact: torch.Tensor, clearance: torch.Tensor, thresh_m: float) -> torch.Tensor:
    """Soft contact score for REWARD terms (contact_height_mode: soft): c = max(force_flag,
    clamp(1 - clearance / thresh_m, 0, 1)) in [0, 1]. A loaded foot is 1; an unloaded one decays
    linearly to 0 over thresh_m of lift (6 mm: a 3 mm lift counts half). thresh_m <= 0 returns the
    force flag as float."""
    return torch.maximum(force_contact.to(torch.float32), clearance_contact_soft(clearance, thresh_m))


@torch.jit.script
def feet_contact_with_clearance(force_contact: torch.Tensor, clearance: torch.Tensor,
                                thresh_m: float) -> torch.Tensor:
    """Contact flag for REWARD terms: |F| > 1 N (force_contact) OR clearance < thresh_m.

    A foot dragged along the floor with < 1 N load has force_contact False but clearance ~0, so it
    no longer counts as airborne (it used to earn imitation_contact in the reference swing and
    escape feet_slide). The sensor flag itself, and the datum that produces `clearance`, stay
    force-based, so there is no feedback loop. thresh_m <= 0 returns force_contact unchanged."""
    return force_contact | clearance_contact(clearance, thresh_m)


@torch.jit.script
def imitation_foot_height(clearance_env: torch.Tensor, ref_height: torch.Tensor, std: float,
                          agg_min: bool = False) -> torch.Tensor:
    """Per-foot kernel exp(-((h - h_ref) / std)^2), averaged over the two feet (agg_min=True: the
    worse foot, so a limp that lifts one foot and drags the other no longer keeps half the reward).

    clearance_env: (N, 2) robot sole clearance [m] in ENV foot order [right, left], clamped >= 0.
    ref_height: (N, 2) the gait table's `foot_height` field, TABLE order [left, right].
    The env order is flipped to the table's here. 1.0 on the reference; 4 mm off with std 4 mm
    -> e^-1 = 0.37; an 11 mm miss (foot dragged instead of lifted) -> 0.0005."""
    h = torch.flip(torch.clamp(clearance_env, min=0.0), [1])
    k = torch.exp(-torch.square((h - ref_height) / std))
    if agg_min:
        return torch.min(k, dim=-1)[0]
    return torch.mean(k, dim=-1)


# --------------------------------------------------------------------------- weight sign guard
# Expected sign of each reward weight, from the sign of the term the weight multiplies (the
# RETURNED quantity of the function in BipedEnv._get_rewards / BipedEnvV2._compute_reward, not
# the name). Shared by both envs. The total reward is sum(weight * term), so:
#   -1: the term is >= 0 and is meant to be subtracted -> weight must be <= 0. A positive
#       weight turns the penalty into a bonus (foot_flat_l2 = +1.0 paid the policy for standing
#       on its foot edges in run gaitref_2026-10-19_20-20-42).
#   +1: the term is >= 0 (an exp kernel, a clamped gap, a count) and is meant to be earned
#       -> weight must be >= 0.
# 0 is valid for both (term disabled).
REWARD_WEIGHT_SIGN: dict[str, int] = {
    # penalties (term >= 0)
    "termination_penalty": -1,          # 0/1 flag
    "lin_vel_z_l2": -1,
    "ang_vel_xy_l2": -1,
    "flat_orientation_l2": -1,
    "action_rate_l2": -1,
    "action_l2": -1,
    "dof_torques_l2": -1,
    "dof_acc_l2": -1,
    "dof_pos_limits": -1,               # amount outside the soft limits, >= 0
    "dof_pos_l2": -1,
    "ankle_torques_l2": -1,
    "angular_momentum": -1,
    "flight": -1,                       # 0/1 (or soft): both feet off the ground
    "base_height_l2": -1,              # squared height error
    "feet_slide": -1,                   # contact * foot speed
    "foot_slip": -1,                    # sum over loaded feet of the planar sole slip speed [m/s]
    "foot_drag": -1,                    # sum over unloaded low feet of the planar sole speed, clearance-weighted
    "undesired_contacts": -1,           # clamp(., 0, 1) per body
    "leg_contact_l2": -1,
    "stand_pose_l2": -1,                # sum of squared joint error to the table stand frame, only while standing
    "foot_flat_l2": -1,                 # sin^2(sole roll) over feet in contact
    "foot_flat_l1": -1,                 # |sin(sole roll)| over feet in contact (no dead zone near flat)
    "foot_separation": -1,              # both modes: origin_one_sided clamp(min_sep - sep, 0), sole_band squared
    "foot_proximity": -1,               # clamp((margin - footprint gap)/margin, 0, 2)^2: feet pressing together
    "joint_deviation_hip": -1,          # L1 |q - q_default|
    "joint_deviation_hip_pitch": -1,
    "joint_deviation_ankle_roll": -1,
    "joint_deviation_ankle_pitch": -1,
    "joint_deviation_knee": -1,
    # rewards (term >= 0)
    "track_lin_vel_xy_exp": 1,
    "track_ang_vel_z_exp": 1,
    "track_joint_pos_exp": 1,
    "track_joint_vel_exp": 1,
    "upright": 1,
    "feet_air_time": 1,                 # positive_biped and touchdown modes are both >= 0
    "swing_foot_height": 1,
    "gait_phase_contact": 1,            # [0, 1]
    "knee_bend_touchdown": 1,
    "step_length": 1,                   # sum of stride^2 at touchdown: a reward despite no "reward" in the name
    "torso_centering": 1,
    "com_support_centering": 1,
    "imitation_joint_pos": 1,
    "imitation_joint_vel": 1,
    "imitation_foot_pos": 1,
    "imitation_contact": 1,
    "imitation_orientation": 1,
    "imitation_lin_vel": 1,
    "imitation_ang_vel": 1,
    "imitation_foot_height": 1,
}

# Scheduled weights (cfg.action_rate_weight_start / _end drive rewards["action_rate_l2"]).
SCHEDULED_WEIGHT_SIGN: dict[str, int] = {
    "action_rate_weight_start": -1,
    "action_rate_weight_end": -1,
}


def reward_weight_sign_errors(weights, schedule=None, keys=None) -> list[str]:
    """One message per weight whose sign contradicts REWARD_WEIGHT_SIGN. [] when all are valid.

    weights: mapping key -> number (cfg.rewards / env_config.reward_weights). Keys that are not
        in REWARD_WEIGHT_SIGN are ignored (unknown / future terms are not this guard's business).
    schedule: optional mapping of SCHEDULED_WEIGHT_SIGN keys -> number (None values skipped).
    keys: optional iterable restricting the check to those weight keys (the MuJoCo env only
        implements a subset of the terms).
    """
    errors: list[str] = []
    only = None if keys is None else set(keys)
    for src, table, restrict in ((weights, REWARD_WEIGHT_SIGN, True),
                                 (schedule, SCHEDULED_WEIGHT_SIGN, False)):
        for key, val in dict(src or {}).items():
            sign = table.get(key)
            if sign is None or val is None or (restrict and only is not None and key not in only):
                continue
            v = float(val)
            if (sign < 0 and v > 0.0) or (sign > 0 and v < 0.0):
                want = "<= 0 (penalty)" if sign < 0 else ">= 0 (reward)"
                errors.append(f"{key}: {v:g}, expected {want}")
    return errors


def validate_reward_weight_signs(weights, schedule=None, keys=None, where: str = "reward weights") -> None:
    """Raise ValueError listing ALL wrong-sign weights (see REWARD_WEIGHT_SIGN)."""
    errors = reward_weight_sign_errors(weights, schedule, keys)
    if errors:
        raise ValueError(
            f"{where}: {len(errors)} reward weight(s) with the wrong sign. The total reward is "
            "sum(weight * term) and every listed term is >= 0, so a wrong sign rewards what it "
            "was meant to penalise (or the reverse):\n  " + "\n  ".join(errors))
