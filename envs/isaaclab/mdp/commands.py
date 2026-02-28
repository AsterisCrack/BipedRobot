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
        self.ranges = {}

        if hasattr(cfg, "commands") and isinstance(cfg.commands, dict):
            base_vel_cfg = cfg.commands.get("base_velocity", {})
            self.resampling_time_range = base_vel_cfg.get("resampling_time_range", self.resampling_time_range)
            self.heading_control_stiffness = base_vel_cfg.get("heading_control_stiffness", self.heading_control_stiffness)
            self.rel_standing_envs = base_vel_cfg.get("rel_standing_envs", self.rel_standing_envs)
            self.rel_heading_envs = base_vel_cfg.get("rel_heading_envs", self.rel_heading_envs)
            self.ranges = base_vel_cfg.get("ranges", {})
        
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
        
        # Update heading mode state
        self.in_heading_mode[env_ids] = is_heading & (~is_standing)
        
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
        if torch.any(self.in_heading_mode):
            q = base_quat
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            
            heading_error = wrap_to_pi(self.heading_target - current_yaw)
            target_ang_vel_z = heading_error * self.heading_control_stiffness
            
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
