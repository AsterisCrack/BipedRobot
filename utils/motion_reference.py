"""Motion reference loader and sampler for imitation rewards."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
import torch


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
