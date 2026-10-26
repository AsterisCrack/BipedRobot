"""Motion reference loader and sampler for imitation rewards."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
import torch

from utils.command_limit import limit_command, validate_shape


@dataclass
class MotionReferenceConfig:
    npz_path: str
    loop: bool = True
    speed: float = 1.0
    root_dir: str | None = None


class MotionReference:
    def __init__(self, cfg: MotionReferenceConfig, device: torch.device) -> None:
        self.device = device
        self.loop = cfg.loop
        self.speed = float(cfg.speed)

        path = cfg.npz_path
        if not os.path.isabs(path) and cfg.root_dir:
            path = os.path.abspath(os.path.join(cfg.root_dir, path))
        self.npz_path = path

        data = np.load(self.npz_path, allow_pickle=True)
        self.joint_names = list(data["joint_names"])
        self.positions = torch.tensor(data["positions"], device=self.device, dtype=torch.float32)
        if "velocities" in data:
            self.velocities = torch.tensor(data["velocities"], device=self.device, dtype=torch.float32)
        else:
            self.velocities = torch.zeros_like(self.positions)

        dt = float(data["dt"]) if "dt" in data else 0.0
        fps = float(data["fps"]) if "fps" in data else 0.0
        if dt <= 0.0 and fps > 0.0:
            dt = 1.0 / fps
        if dt <= 0.0:
            raise ValueError(f"Invalid motion dt/fps in: {self.npz_path}")
        self.frame_dt = dt
        self.num_frames = int(self.positions.shape[0])
        if self.num_frames < 2:
            raise ValueError("Motion reference must have at least 2 frames.")

        self._robot_joint_indices: torch.Tensor | None = None
        self._ref_joint_indices: torch.Tensor | None = None

    @property
    def duration(self) -> float:
        return self.num_frames * self.frame_dt

    @property
    def robot_joint_indices(self) -> torch.Tensor:
        if self._robot_joint_indices is None:
            raise RuntimeError("Motion reference not bound to robot joint names.")
        return self._robot_joint_indices

    def bind_to_robot(self, robot_joint_names: Iterable[str]) -> None:
        ref_index = {name: i for i, name in enumerate(self.joint_names)}
        robot_indices = []
        ref_indices = []
        for i, name in enumerate(robot_joint_names):
            if name in ref_index:
                robot_indices.append(i)
                ref_indices.append(ref_index[name])
        if not robot_indices:
            raise ValueError("No overlapping joints between motion reference and robot.")

        self._robot_joint_indices = torch.tensor(robot_indices, device=self.device, dtype=torch.long)
        self._ref_joint_indices = torch.tensor(ref_indices, device=self.device, dtype=torch.long)

    def sample(self, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ref_joint_indices is None:
            raise RuntimeError("Motion reference not bound to robot joint names.")

        frame = (time_s / self.frame_dt) * self.speed
        frame_floor = torch.floor(frame)
        idx0 = frame_floor.long()
        alpha = (frame - frame_floor).unsqueeze(-1)

        if self.loop:
            idx0 = torch.remainder(idx0, self.num_frames)
            idx1 = torch.remainder(idx0 + 1, self.num_frames)
        else:
            idx0 = torch.clamp(idx0, 0, self.num_frames - 1)
            idx1 = torch.clamp(idx0 + 1, 0, self.num_frames - 1)

        pos0 = self.positions[idx0][:, self._ref_joint_indices]
        pos1 = self.positions[idx1][:, self._ref_joint_indices]
        vel0 = self.velocities[idx0][:, self._ref_joint_indices]
        vel1 = self.velocities[idx1][:, self._ref_joint_indices]

        pos = pos0 * (1.0 - alpha) + pos1 * alpha
        vel = vel0 * (1.0 - alpha) + vel1 * alpha
        return pos, vel


class CadenceGrid:
    """Gait clock frequency [cycles/s] as a function of the command, numpy (MuJoCo, export).

    Table v2 cycles each have their own period (Gait Studio's cadence law), so the clock runs at
    freq(cmd): trilinear in (vx, vy, wz), each axis clamped to the grid. The same rule as
    ``GaitTableReference.frequency`` (Isaac) and the Jetson runtime's ``cadence_grid`` clock.
    A v1 table has one period; ``from_npz`` turns it into a constant grid.
    """

    def __init__(self, grid_vx, grid_vy, grid_wz, freq) -> None:
        self.axes = [np.asarray(a, dtype=np.float64) for a in (grid_vx, grid_vy, grid_wz)]
        self.freq = np.asarray(freq, dtype=np.float64).reshape(tuple(len(a) for a in self.axes))

    @classmethod
    def from_npz(cls, z) -> "CadenceGrid":
        axes = [np.asarray(z[k]) for k in ("grid_vx", "grid_vy", "grid_wz")]
        if "freq" in z:
            return cls(*axes, z["freq"])
        return cls(*axes, np.full(tuple(len(a) for a in axes), 1.0 / float(z["period"])))

    @staticmethod
    def _locate(grid: np.ndarray, x: float) -> tuple[int, float]:
        n = grid.size
        if n == 1:
            return 0, 0.0
        x = min(max(float(x), float(grid[0])), float(grid[-1]))
        i0 = int(np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n - 2))
        return i0, (x - float(grid[i0])) / float(grid[i0 + 1] - grid[i0])

    def frequency(self, cmd) -> float:
        idx = [self._locate(a, c) for a, c in zip(self.axes, cmd)]
        out = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = 1.0
                    cell = []
                    for d, (i0, a), n in zip((di, dj, dk), idx, self.freq.shape):
                        w *= a if d else 1.0 - a
                        cell.append(min(i0 + d, n - 1))
                    out += w * float(self.freq[tuple(cell)])
        return out

    @property
    def constant(self) -> bool:
        return bool(np.ptp(self.freq) < 1e-9)


def resolve_table_path(npz_path: str, root_dir: str | None = None) -> str:
    """Path of a baked gait table: a relative path is taken from root_dir (the BipedRobot root)."""
    path = npz_path
    if not os.path.isabs(path) and root_dir:
        path = os.path.abspath(os.path.join(root_dir, path))
    return path


class GaitTableReference:
    """Command-conditioned gait table baked by Gait Studio (robot_motion_reference/gait_studio).

    One steady-state cycle per (vx, vy, wz) grid point plus a standing frame, held on the GPU as a
    flat [(vx, vy, wz, frame), D] table. ``sample`` interpolates trilinearly in the command
    (clamped to the grid) and linearly in phase (wrapping). Phase convention: phase 0 = right-foot
    touchdown, [0, 0.5) right stance. Table foot order is [left, right], joint order is the
    table's own: always go through ``bind_to_robot``.

    The cycles are phase-normalised. In a v2 table each has its own period (cadence follows the
    command, as in humans) and ``frequency`` gives the gait clock rate for a command, trilinear
    like the table. A v1 table has a single ``period``.

    Table versions: 1 (one period), 2 (per-command ``freq``), 3 (= 2 + command-limit metadata:
    ``command_limit_shape``, ``command_ranges_x/y/z``, ``cmd_baked``; cells outside the limit are
    baked at the projected command). The table never defines the command limit: the TRAINING config
    does, and the env hands it over with ``set_command_limit``. ``check_command_limit`` compares the
    table's metadata with it.
    """

    _limit: tuple[dict, str] | None = None     # the config's limit, set by the env (class default: none)
    baked_limit: dict | None = None            # what a v3 table was baked under (None for v1/v2)

    def __init__(self, npz_path: str, device: torch.device, root_dir: str | None = None) -> None:
        path = resolve_table_path(npz_path, root_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"gait table not found: {path} (bake it with "
                                    f"`python -m gait_studio.bake` in robot_motion_reference/)")
        z = np.load(path, allow_pickle=False)
        self.device = device
        self.npz_path = path
        data = torch.as_tensor(z["data"], dtype=torch.float32, device=device)   # [nvx, nvy, nwz, F, D]
        self.shape = tuple(data.shape[:3])
        self.n_frames = int(data.shape[3])
        self.dim = int(data.shape[4])
        self.table = data.reshape(-1, self.dim)
        self.stand = torch.as_tensor(z["stand"], dtype=torch.float32, device=device)
        self.grid = [torch.as_tensor(z[k], dtype=torch.float32, device=device)
                     for k in ("grid_vx", "grid_vy", "grid_wz")]
        self.version = int(z["version"]) if "version" in z else 1
        self.period = float(z["period"])     # v2: the zero-command cycle, for display only
        if "freq" in z:
            self.freq = torch.as_tensor(z["freq"], dtype=torch.float32, device=device).reshape(-1)
        else:
            self.freq = torch.full((int(np.prod(self.shape)),), 1.0 / self.period, device=device)
        if self.freq.numel() != int(np.prod(self.shape)) or not bool((self.freq > 0).all()):
            raise ValueError(f"gait table {path}: bad freq grid (shape {tuple(self.freq.shape)}, grid {self.shape})")
        self.dt = float(z["dt"]) if "dt" in z else self.period / self.n_frames
        self.field_names = [str(s) for s in z["field_names"]]
        widths = [int(w) for w in z["field_widths"]]
        edges = np.concatenate([[0], np.cumsum(widths)])
        self.slices = {n: slice(int(edges[i]), int(edges[i + 1])) for i, n in enumerate(self.field_names)}
        self.joint_names = [str(s) for s in z["joint_names"]]
        self.style_digest = str(z["style_digest"])
        self.phase_convention = str(z["phase_convention"])
        # Command limit the table was baked under (v3). None for v1/v2: baked without a limit.
        self.baked_limit: dict | None = None
        if "command_limit_shape" in z:
            self.baked_limit = {"shape": str(z["command_limit_shape"]),
                                "lin_vel_x": tuple(float(v) for v in z["command_ranges_x"]),
                                "lin_vel_y": tuple(float(v) for v in z["command_ranges_y"]),
                                "ang_vel_z": tuple(float(v) for v in z["command_ranges_z"])}
        self._limit: tuple[dict, str] | None = None   # the config's limit, set by the env
        self._joint_idx: torch.Tensor | None = None
        self._offsets = torch.as_tensor(
            [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=device)  # [8, 3]
        # Built once here: a torch.as_tensor(..., device=cuda) per call is a CPU->GPU copy (a sync).
        self._top = torch.as_tensor([n - 1 for n in self.shape], device=device)
        self._offsets_hi = self._offsets.bool()                                  # [8, 3]

    def set_command_limit(self, ranges: dict, shape: str) -> None:
        """The TRAINING config's command limit. ``corners`` (hence ``sample`` and ``frequency``) applies it to
        the commands before locating the cell, so an over-limit command reads the projected one's frame.
        Box = per-axis ranges only = no change here (the grid clamp in ``_locate`` already bounds each axis)."""
        shape = validate_shape(shape)
        self._limit = None if shape == "box" else ({k: tuple(ranges[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}, shape)

    def check_command_limit(self, ranges: dict, shape: str, tol: float = 1e-6) -> list[str]:
        """Messages (empty = consistent) comparing the table's baked command limit with the config's.

        Never reads the limit from the table; this is only a consistency check. A v1/v2 table has no
        metadata: that is silent for a box config, and a one-line note otherwise (boundary cells are
        interpolated, which is fine)."""
        shape = validate_shape(shape)
        b = self.baked_limit
        if b is None:
            if shape == "box":
                return []
            return [f"note: this table (v{self.version}) was baked without a command limit; config limit_shape="
                    f"{shape!r} only narrows the commands, boundary cells are interpolated (fine)"]
        out = []
        if b["shape"] != shape:
            out.append(f"WARNING: table baked with command_limit_shape={b['shape']!r} but the training config has "
                       f"limit_shape={shape!r}; the table's boundary cells are not baked for this limit "
                       f"(rebake in Gait Studio with command_limit_shape: {shape})")
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            cfg_r = tuple(float(v) for v in ranges[key])
            if any(abs(a - c) > tol for a, c in zip(b[key], cfg_r)):
                out.append(f"WARNING: table baked with command range {key}={b[key]} but the training config has "
                           f"{key}={cfg_r}")
        return out

    def bind_to_robot(self, robot_joint_names: Iterable[str]) -> None:
        names = list(robot_joint_names)
        missing = [n for n in names if n not in self.joint_names]
        if missing:
            raise ValueError(f"gait table has no joint(s) {missing}; table joints: {self.joint_names}")
        self._joint_idx = torch.as_tensor([self.joint_names.index(n) for n in names],
                                          dtype=torch.long, device=self.device)

    def _locate(self, grid: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower cell index and fractional weight along one axis, clamped to the grid."""
        n = grid.numel()
        if n == 1:
            return torch.zeros_like(x, dtype=torch.long), torch.zeros_like(x)
        x = torch.minimum(torch.maximum(x, grid[0]), grid[-1])
        i0 = (torch.searchsorted(grid, x, right=True) - 1).clamp(0, n - 2)
        lo, hi = grid[i0], grid[i0 + 1]
        return i0, (x - lo) / (hi - lo)

    def corners(self, commands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The 8 trilinear corners of each command, clamped to the grid: (cell index [N, 8], weight [N, 8]).

        All 8 at once (one batch of ops, not a Python loop of ~10 small kernels per corner). Pass the
        result to frequency() / sample() when both are evaluated for the same commands."""
        if self._limit is not None:
            commands = limit_command(commands, *self._limit)
        cmd = commands.to(device=self.device, dtype=torch.float32)
        i0, wx = self._locate(self.grid[0], cmd[:, 0])
        j0, wy = self._locate(self.grid[1], cmd[:, 1])
        k0, wz = self._locate(self.grid[2], cmd[:, 2])
        base = torch.stack([i0, j0, k0], dim=-1)                                   # [N, 3]
        w_hi = torch.stack([wx, wy, wz], dim=-1)                                   # [N, 3]
        _, nvy, nwz = self.shape
        idx = torch.minimum(base.unsqueeze(1) + self._offsets, self._top)          # [N, 8, 3]
        w = torch.where(self._offsets_hi, w_hi.unsqueeze(1), 1.0 - w_hi.unsqueeze(1)).prod(dim=-1)  # [N, 8]
        return (idx[..., 0] * nvy + idx[..., 1]) * nwz + idx[..., 2], w

    def frequency(self, commands: torch.Tensor, corners: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Gait clock frequency [cycles/s] for commands [N, 3] -> [N]."""
        cell, w = corners if corners is not None else self.corners(commands)
        f = w * self.freq[cell]                                                    # [N, 8]
        out = f[:, 0]
        for k in range(1, 8):   # corner by corner, in the old order: bit-identical sums (RSI poses, clock)
            out = out + f[:, k]
        return out

    def sample(self, commands: torch.Tensor, phase: torch.Tensor, standing: torch.Tensor,
               corners: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """commands [N, 3] (vx, vy, wz), phase [N] in [0, 1), standing [N] bool -> [N, D]."""
        phase = phase.to(device=self.device, dtype=torch.float32)
        n = commands.shape[0]
        pos = (phase % 1.0) * self.n_frames
        f_lo = torch.floor(pos)
        a = (pos - f_lo).unsqueeze(-1)                                             # [N, 1]
        f0 = f_lo.long() % self.n_frames
        f1 = (f0 + 1) % self.n_frames

        cell, w = corners if corners is not None else self.corners(commands)      # [N, 8]
        row0 = self.table[cell * self.n_frames + f0.unsqueeze(-1)]                 # [N, 8, D]
        row1 = self.table[cell * self.n_frames + f1.unsqueeze(-1)]
        frame = w.unsqueeze(-1) * (row0 * (1.0 - a).unsqueeze(-1) + row1 * a.unsqueeze(-1))   # [N, 8, D]
        out = frame[:, 0]
        for k in range(1, 8):   # corner by corner, in the old order: bit-identical to the old per-corner loop
            out = out + frame[:, k]
        return torch.where(standing.to(self.device).unsqueeze(-1), self.stand.expand(n, -1), out)

    def field(self, ref: torch.Tensor, name: str) -> torch.Tensor:
        return ref[:, self.slices[name]]

    def joint_field(self, ref: torch.Tensor, name: str) -> torch.Tensor:
        """'joint_pos' or 'joint_vel' of a sampled frame, in the bound robot's joint order."""
        if self._joint_idx is None:
            raise RuntimeError("call bind_to_robot(joint_names) first")
        return self.field(ref, name)[:, self._joint_idx]
