import torch
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, quat_apply
from isaaclab.markers import VisualizationMarkers

class UniformVelocityCommand:
    def __init__(self, cfg, num_envs, device, step_dt):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        
        # Command buffers: [v_x, v_y, w_z]
        self.commands = torch.zeros(num_envs, 3, device=device)
        
        # Internal state
        self.time_left = torch.zeros(num_envs, device=device)
        self.heading_target = torch.zeros(num_envs, device=device)
        self.in_heading_mode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Parsing Configuration
        self.resampling_time_range = (10.0, 10.0)
        self.heading_control_stiffness = 0.5
        self.rel_standing_envs = 0.02
        self.rel_heading_envs = 1.0
        # Turn-in-place bucket: vx = vy = 0 with a large |wz|. Under independent uniform
        # sampling, "lin ~ 0, |ang| large" is ~2% of the data, so spinning on the spot is
        # effectively never trained, and real-robot turning comes out slow and unstable.
        self.rel_turn_in_place_envs = 0.0
        self.turn_in_place_min_frac = 0.4   # |wz| drawn from [frac*max, max]
        # Yaw-rate ceiling for heading mode. None = the ang_vel_z range. Heading mode turns at
        # error * stiffness, so with a wide ang_vel_z range (+-1.2 for turn-in-place) most moving
        # envs would spend the first seconds of every command turning at the limit.
        self.heading_max_ang_vel = None
        self.ranges = {}

        if hasattr(cfg, "commands") and isinstance(cfg.commands, dict):
            base_vel_cfg = cfg.commands.get("base_velocity", {})
            self.resampling_time_range = base_vel_cfg.get("resampling_time_range", self.resampling_time_range)
            self.heading_control_stiffness = base_vel_cfg.get("heading_control_stiffness", self.heading_control_stiffness)
            self.rel_standing_envs = base_vel_cfg.get("rel_standing_envs", self.rel_standing_envs)
            self.rel_heading_envs = base_vel_cfg.get("rel_heading_envs", self.rel_heading_envs)
            self.rel_turn_in_place_envs = base_vel_cfg.get("rel_turn_in_place_envs",
                                                           self.rel_turn_in_place_envs)
            self.heading_max_ang_vel = base_vel_cfg.get("heading_max_ang_vel", self.heading_max_ang_vel)
            self.ranges = base_vel_cfg.get("ranges", {})
        
    def update_ranges(self, ranges: dict) -> None:
        """Update velocity sampling ranges in-place (used by curriculum)."""
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
            if key in ranges:
                self.ranges[key] = ranges[key]

    def reset(self, env_ids):
        """Reset commands for specific environments."""
        self.resample(env_ids)

    def resample(self, env_ids):
        if len(env_ids) == 0:
            return

        # 1. Update Timers
        t_min, t_max = self.resampling_time_range
        self.time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(t_min, t_max)
        
        # 2. Get Ranges
        try:
             r_x = self.ranges.get("lin_vel_x", (-0.5, 0.5))
             r_y = self.ranges.get("lin_vel_y", (-0.25, 0.25))
             r_z = self.ranges.get("ang_vel_z", (-1.0, 1.0))
             r_head = self.ranges.get("heading", (-3.14159, 3.14159))
        except (KeyError, TypeError, AttributeError) as e:
             r_x = (-0.5, 0.5)
             r_y = (-0.25, 0.25)
             r_z = (-1.0, 1.0)
             r_head = (-3.14159, 3.14159)

        # 3. Sample Velocities (x, y, ang_z)
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r_x)
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r_y)
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r_z)
        
        # 4. Heading Mode Logic
        is_standing = torch.rand(len(env_ids), device=self.device) < self.rel_standing_envs
        is_heading = torch.rand(len(env_ids), device=self.device) < self.rel_heading_envs
        
        # Standing envs have zero command
        self.commands[env_ids[is_standing], :] = 0.0

        # Turn-in-place envs, disjoint from standing: no translation, a yaw rate drawn from
        # the top of the range with a random sign.
        is_turn = (torch.rand(len(env_ids), device=self.device) < self.rel_turn_in_place_envs) & (~is_standing)
        if is_turn.any():
            turn_ids = env_ids[is_turn]
            wz_max = max(abs(float(r_z[0])), abs(float(r_z[1])))
            mag = torch.empty(len(turn_ids), device=self.device).uniform_(
                self.turn_in_place_min_frac * wz_max, wz_max)
            sign = torch.where(torch.rand(len(turn_ids), device=self.device) < 0.5, -1.0, 1.0)
            self.commands[turn_ids, 0] = 0.0
            self.commands[turn_ids, 1] = 0.0
            self.commands[turn_ids, 2] = mag * sign

        # Heading mode rewrites wz every step from the heading error, so it would silently
        # overwrite the turn bucket. Both standing and turning envs are excluded.
        self.in_heading_mode[env_ids] = is_heading & (~is_standing) & (~is_turn)
        
        # Sample target heading
        self.heading_target[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*r_head)

    def compute(self, base_quat):
        """Update commands based on time and heading control."""
        
        # 1. Update Timer
        self.time_left -= self.step_dt
        
        # 2. Resample expired
        resample_env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_env_ids) > 0:
            self.resample(resample_env_ids)
            
        # 3. Heading Control Logic
        self.apply_heading(base_quat)
        return self.commands

    def apply_heading(self, base_quat):
        """Overwrite wz of heading-mode envs from their heading error.

        Also call this right after resampling: resample() draws wz uniformly over the whole
        ang_vel_z range, and until the next compute() a freshly reset env (and the reference
        state it is initialised on) would see that draw instead of its heading-control rate.
        """
        if torch.any(self.in_heading_mode):
            q = base_quat
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            
            heading_error = wrap_to_pi(self.heading_target - current_yaw)
            # Clamp to the configured yaw range. Without this, a heading error near +-pi with
            # stiffness 0.5 commands +-1.57 rad/s -- over 5x the configured +-0.3 -- which is
            # unreachable, and track_ang_vel_z_exp (std=0.15) returns numerically zero there,
            # so those envs get no yaw reward and no gradient while still paying the stability
            # penalties. Upstream Isaac Lab clamps this the same way.
            r_z = self.ranges.get("ang_vel_z", (-1.0, 1.0))
            lo, hi = float(r_z[0]), float(r_z[1])
            if self.heading_max_ang_vel is not None:
                lo, hi = max(lo, -float(self.heading_max_ang_vel)), min(hi, float(self.heading_max_ang_vel))
            target_ang_vel_z = torch.clamp(heading_error * self.heading_control_stiffness, min=lo, max=hi)

            # Apply to commands
            self.commands[self.in_heading_mode, 2] = target_ang_vel_z[self.in_heading_mode]
            
        return self.commands

    def _update_arrow(self, marker: VisualizationMarkers, velocity_w: torch.Tensor, root_pos_w: torch.Tensor, offset: list[float], scale: list[float]):
        """Helper to visualize arrows."""
        # Orientation
        yaw = torch.atan2(velocity_w[:, 1], velocity_w[:, 0])
        orientations = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        
        # Scale
        scales = torch.tensor(scale, device=self.device).repeat(self.num_envs, 1)
        
        marker.visualize(translations=root_pos_w + torch.tensor(offset, device=self.device),
                         orientations=orientations,
                         scales=scales)

    def visualize(self, cmd_marker, vel_marker, root_pos_w, root_quat_w, base_lin_vel_b):
        """Visualize command and actual velocity."""
        
        # 1. Command Velocity (Green)
        cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_vel_b[:, 0] = self.commands[:, 0]
        cmd_vel_b[:, 1] = self.commands[:, 1]
        
        cmd_vel_w = quat_apply(root_quat_w, cmd_vel_b)
        self._update_arrow(cmd_marker, cmd_vel_w, root_pos_w, [0, 0, 0.5], [0.2, 0.2, 0.2])
                               
        # 2. Actual Velocity (Blue)
        vel_w = quat_apply(root_quat_w, base_lin_vel_b.clone())
        self._update_arrow(vel_marker, vel_w, root_pos_w, [0, 0, 0.6], [0.5, 0.1, 0.1])
