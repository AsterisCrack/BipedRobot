import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from utils.command_limit import limit_command, sample_in_limit, validate_shape
from utils import teleop as T

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
        # "box" (per-axis ranges, the historical sampler) or "ellipse_xy": (vx, vy) uniform inside the
        # ellipse spanned by the ranges. The config is the source of truth (utils/command_limit.py).
        self.limit_shape = "box"
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
            self.limit_shape = validate_shape(base_vel_cfg.get("limit_shape", self.limit_shape))
            self.ranges = base_vel_cfg.get("ranges", {})

        # Manual command (play.py --teleop): [3] tensor or None. When set it replaces every sampled and
        # heading-mode command; None (always, in training) leaves the sampler exactly as it was.
        self.manual = None
        # Arrow / arc scale from the FULL configured ranges (the curriculum narrows self.ranges in place).
        self._vis_k_lin, self._vis_k_ang = T.vis_scales(self.ranges)

    def set_manual(self, cmd) -> None:
        """Hold (vx, vy, wz) on every env until clear_manual(). Give it an already limited command
        (utils.teleop.axes_to_command); it takes effect at the next compute() / reset."""
        self.manual = torch.as_tensor(cmd, dtype=self.commands.dtype, device=self.device).reshape(3)

    def clear_manual(self) -> None:
        self.manual = None
        
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
        if self.limit_shape == "box":
            self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r_x)
            self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r_y)
            self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r_z)
        else:
            # (vx, vy) uniform inside the ellipse by rejection from the box (no sync, bounded rounds);
            # wz independent and uniform in its range, as above. The CURRENT ranges (curriculum) are used.
            self.commands[env_ids] = sample_in_limit(
                len(env_ids), {"lin_vel_x": r_x, "lin_vel_y": r_y, "ang_vel_z": r_z},
                self.limit_shape, device=self.device)
        
        # 4. Heading Mode Logic
        is_standing = torch.rand(len(env_ids), device=self.device) < self.rel_standing_envs
        is_heading = torch.rand(len(env_ids), device=self.device) < self.rel_heading_envs
        
        # Standing envs have zero command (masked select: boolean indexing env_ids would sync)
        self.commands[env_ids] = torch.where(is_standing.unsqueeze(-1), 0.0, self.commands[env_ids])

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
        if self.manual is not None:
            self.commands[:] = self.manual      # in place: env.commands aliases this tensor
            return self.commands

        # Computed for every env and selected with the mask: `if torch.any(mask)` and boolean-mask
        # indexing each synced the GPU, every step. Non-heading envs keep their command exactly.
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
        self.commands[:, 2] = torch.where(self.in_heading_mode, target_ang_vel_z, self.commands[:, 2])

        if self.limit_shape != "box":
            # Guard: a no-op for (vx, vy) (sampled inside the limit, wz is not part of the ellipse),
            # kept so nothing that rewrites commands can leave the limit. Box: nothing to do, bit-identical.
            self.commands[:] = limit_command(self.commands, self.ranges, self.limit_shape)

        return self.commands

    # ------------------------------------------------------------------ visualisation
    # Command (green) and measured (blue) velocity on ONE scale (utils/teleop.py, as in the MuJoCo viewer):
    # a straight arrow for (vx, vy) -- a dot when the command does not translate -- and an arc around the
    # base for wz, from the robot's heading, counter-clockwise for +wz. Yaw-only frame: no tilt with pitch.
    # Per env and colour: [shaft, head, arc chord x (K-1), arc head] instances of one PointInstancer.

    _LIN_DZ = {"cmd": 0.30, "meas": 0.34}
    _ARC_DZ = 0.22
    _HEAD_LEN = 0.05
    _HIDE = 1e-4      # instance scale that hides it (the instancer has no per-instance visibility)

    def visualize(self, markers, root_pos_w, root_quat_w, lin_vel_w, ang_vel_w):
        """markers: (green, blue) from make_command_markers(). Velocities in the world frame."""
        q = root_quat_w
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
        c, s = torch.cos(yaw), torch.sin(yaw)
        cmd = self.commands
        v_cmd = torch.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], dim=-1)
        for marker, v, wz, key, r in ((markers[0], v_cmd, cmd[:, 2], "cmd", T.ARC_RADIUS),
                                      (markers[1], lin_vel_w[:, :2], ang_vel_w[:, 2], "meas",
                                       T.ARC_RADIUS + T.ARC_RADIUS_GAP)):
            lin = self._arrow_instances(root_pos_w, v, self._LIN_DZ[key], dot=(key == "cmd"))
            arc = self._arc_instances(root_pos_w, yaw, wz, r)
            pos, quat, scale, idx = (torch.cat([a, b], dim=1).flatten(0, 1) for a, b in zip(lin, arc))
            marker.visualize(translations=pos, orientations=quat, scales=scale, marker_indices=idx)

    def _arrow_instances(self, root_pos_w, v, dz, dot: bool):
        """(pos, quat, scale, proto) per env, 2 instances: shaft (or dot) and head. Total length = k * |v|."""
        n, dev = v.shape[0], v.device
        speed = torch.linalg.norm(v, dim=-1)
        L = speed * self._vis_k_lin
        d = v / speed.unsqueeze(-1).clamp(min=1e-9)
        h = self._HEAD_LEN
        shaft_len = (L - h).clamp(min=0.0)
        head_s = (L / h).clamp(max=1.0)                  # a short arrow shrinks its head instead
        p0 = root_pos_w + torch.tensor([0.0, 0.0, dz], device=dev)
        d3 = torch.cat([d, torch.zeros(n, 1, device=dev)], dim=-1)
        stopped = speed < T.STOP_THRESH
        ang = torch.atan2(d[:, 1], d[:, 0])
        quat = quat_from_euler_xyz(torch.zeros_like(ang), torch.zeros_like(ang), ang)

        pos = torch.stack([p0 + d3 * (shaft_len / 2).unsqueeze(-1),
                           p0 + d3 * (shaft_len + h * head_s / 2).unsqueeze(-1)], dim=1)
        one = torch.ones(n, device=dev)
        sc = torch.stack([torch.stack([shaft_len.clamp(min=self._HIDE), one, one], dim=-1),
                          head_s.clamp(min=self._HIDE).unsqueeze(-1).expand(n, 3)], dim=1)
        idx = torch.tensor([0, 1], device=dev).expand(n, 2).clone()
        hide = torch.full_like(sc, self._HIDE)
        if dot:   # stopped command: the shaft instance becomes the dot, the head disappears
            st = stopped.unsqueeze(-1)
            pos[:, 0] = torch.where(st, p0, pos[:, 0])
            idx[:, 0] = torch.where(stopped, 2, 0)
            sc[:, 0] = torch.where(st, torch.ones_like(sc[:, 0]), sc[:, 0])
            sc[:, 1] = torch.where(st, hide[:, 1], sc[:, 1])
        else:
            sc = torch.where(stopped.view(n, 1, 1), hide, sc)
        return pos, quat.unsqueeze(1).expand(n, 2, 4), sc, idx

    def _arc_instances(self, root_pos_w, yaw, wz, r):
        """(pos, quat, scale, proto) per env, K instances: K-1 chords and a cone head along the last one."""
        n, dev, K = yaw.shape[0], yaw.device, T.ARC_SEGMENTS
        span = wz * self._vis_k_ang
        t = yaw.unsqueeze(-1) + span.unsqueeze(-1) * torch.linspace(0.0, 1.0, K + 1, device=dev)
        z = (root_pos_w[:, 2] + self._ARC_DZ).unsqueeze(-1).expand(n, K + 1)
        p = torch.stack([root_pos_w[:, 0:1] + r * torch.cos(t), root_pos_w[:, 1:2] + r * torch.sin(t), z], dim=-1)
        a, b = p[:, :-1], p[:, 1:]                                   # K chords
        seg = b - a
        seg_len = torch.linalg.norm(seg, dim=-1)
        ang = torch.atan2(seg[..., 1], seg[..., 0])
        tang = seg[:, -1] / seg_len[:, -1:].clamp(min=1e-9)
        h = self._HEAD_LEN
        pos = torch.cat([(a[:, :-1] + b[:, :-1]) / 2, (a[:, -1] + tang * h / 2).unsqueeze(1)], dim=1)
        quat = quat_from_euler_xyz(torch.zeros_like(ang), torch.zeros_like(ang), ang)
        one = torch.ones(n, K - 1, device=dev)
        sc = torch.cat([torch.stack([seg_len[:, :-1].clamp(min=self._HIDE), one, one], dim=-1),
                        torch.ones(n, 1, 3, device=dev)], dim=1)
        sc = torch.where((wz.abs() < T.STOP_THRESH).view(n, 1, 1), torch.full_like(sc, self._HIDE), sc)
        idx = torch.cat([torch.full((n, K - 1), 3, device=dev, dtype=torch.long),
                         torch.ones(n, 1, device=dev, dtype=torch.long)], dim=1)
        return pos, quat, sc, idx


def make_command_markers():
    """(green, blue) VisualizationMarkers for UniformVelocityCommand.visualize. Prototypes:
    0 shaft (unit length along X, scaled to the arrow), 1 head (cone along X), 2 dot, 3 arc chord."""
    out = []
    for name, rgb in (("CommandVelocity", (0.1, 0.9, 0.1)), ("BaseVelocity", (0.15, 0.35, 1.0))):
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
        cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/{name}",
            markers={
                "shaft": sim_utils.CylinderCfg(radius=0.008, height=1.0, axis="X", visual_material=mat),
                "head": sim_utils.ConeCfg(radius=0.022, height=UniformVelocityCommand._HEAD_LEN, axis="X",
                                          visual_material=mat),
                "dot": sim_utils.SphereCfg(radius=0.025, visual_material=mat),
                "chord": sim_utils.CylinderCfg(radius=0.005, height=1.0, axis="X", visual_material=mat),
            },
        )
        out.append(VisualizationMarkers(cfg))
    return tuple(out)
