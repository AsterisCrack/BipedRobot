"""Command limit shared by training, the MuJoCo tools, the export and the Jetson runtime.

One definition of "which (vx, vy, wz) commands exist":

* per axis, every command is clamped to its range (lin_vel_x, lin_vel_y, ang_vel_z: [lo, hi]);
* shape ``box``        - that clamp is the whole limit (the historical behaviour);
* shape ``ellipse_xy`` - additionally (vx / vx_b)^2 + (vy / vy_b)^2 <= 1, with vx_b the forward bound
  (hi) for vx >= 0 and the backward bound (-lo) otherwise, likewise vy_b. An over-limit command is
  scaled down along its own direction in (vx, vy), so the direction is kept. wz is untouched.

The TRAINING config (``commands.base_velocity.limit_shape`` + ``ranges``) is the source of truth; the
run config and the exported bundle carry it, Gait Studio only mirrors it for its bakes.

Numpy and torch share the same maths. The numpy half of this file is copied into Gait Studio
(``robot_motion_reference/gait_studio/command_limit.py``, which cannot import this repo) and the Jetson
runtime; ``tests/data/command_limit_golden.json`` pins all copies to the same numbers.
"""
from __future__ import annotations

import numpy as np

try:  # the numpy API works without torch (Gait Studio, Jetson)
    import torch
except ImportError:  # pragma: no cover
    torch = None

SHAPES = ("box", "ellipse_xy")
# Used for a missing range key; the same fallbacks the Isaac command sampler always had.
DEFAULT_RANGES = {"lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-1.0, 1.0)}
_MAX_ROUNDS = 12  # rejection rounds in torch; (1 - pi/4)^12 ~ 1e-8 of draws left over, then projected


def validate_shape(shape) -> str:
    """``shape`` as a string, or ValueError. None -> 'box' (an old config without the key)."""
    if shape is None:
        return "box"
    if not isinstance(shape, str) or shape not in SHAPES:
        raise ValueError(f"unknown command limit shape {shape!r}; expected one of {SHAPES}")
    return shape


def _bounds(ranges) -> tuple[float, float, float, float, float, float]:
    out = []
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo, hi = ranges.get(key, DEFAULT_RANGES[key]) if ranges is not None else DEFAULT_RANGES[key]
        out += [float(lo), float(hi)]
    return tuple(out)


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _xy_norm_np(vx, vy, lo_x, hi_x, lo_y, hi_y):
    """s = sqrt((vx / vx_b)^2 + (vy / vy_b)^2), vx_b = hi_x (vx >= 0) or -lo_x; vx, vy inside the box."""
    bx = np.where(vx >= 0, hi_x, -lo_x)
    by = np.where(vy >= 0, hi_y, -lo_y)
    # A zero bound means the box clamp already forced that component to 0: it adds nothing to s.
    tx = np.where(bx > 0, vx / np.where(bx > 0, bx, 1.0), 0.0)
    ty = np.where(by > 0, vy / np.where(by > 0, by, 1.0), 0.0)
    return np.sqrt(tx * tx + ty * ty)


def _xy_norm_torch(vx, vy, lo_x, hi_x, lo_y, hi_y):
    """Same as _xy_norm_np on tensors (bounds built in the input dtype: no float32 rounding of 0.2)."""
    def term(v, lo, hi):
        b = torch.where(v >= 0, torch.full_like(v, hi), torch.full_like(v, -lo))
        return torch.where(b > 0, v / torch.where(b > 0, b, torch.ones_like(b)), torch.zeros_like(v))
    tx, ty = term(vx, lo_x, hi_x), term(vy, lo_y, hi_y)
    return torch.sqrt(tx * tx + ty * ty)


def limit_command(cmd, ranges, shape: str = "box"):
    """Clamp commands (..., 3) = (vx, vy, wz) to the limit. numpy in -> numpy out, torch in -> torch out."""
    shape = validate_shape(shape)
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _bounds(ranges)
    if _is_torch(cmd):
        vx = cmd[..., 0].clamp(lo_x, hi_x)
        vy = cmd[..., 1].clamp(lo_y, hi_y)
        wz = cmd[..., 2].clamp(lo_z, hi_z)
        if shape == "ellipse_xy":
            s = _xy_norm_torch(vx, vy, lo_x, hi_x, lo_y, hi_y).clamp(min=1.0)
            vx, vy = vx / s, vy / s
        return torch.stack([vx, vy, wz], dim=-1)
    arr = np.asarray(cmd)
    arr = arr.astype(np.float64) if arr.dtype.kind != "f" else arr
    vx = np.clip(arr[..., 0], lo_x, hi_x)
    vy = np.clip(arr[..., 1], lo_y, hi_y)
    wz = np.clip(arr[..., 2], lo_z, hi_z)
    if shape == "ellipse_xy":
        s = np.maximum(_xy_norm_np(vx, vy, lo_x, hi_x, lo_y, hi_y), 1.0)
        vx, vy = vx / s, vy / s
    return np.stack([vx, vy, wz], axis=-1)


def sample_in_limit(n: int, ranges, shape: str = "box", rng=None, device=None):
    """n commands (n, 3) drawn uniformly inside the limit.

    (vx, vy) is uniform over the box (``box``) or the ellipse (``ellipse_xy``, by rejection from the
    box: scaling would pile density on the boundary); wz is uniform in its range, independent.

    numpy by default (``rng``: np.random.Generator or None). torch when ``device`` is given or
    ``rng`` is a torch.Generator: everything stays on-device with no host sync, a fixed number of
    rejection rounds, and the rare leftover is projected with limit_command.
    """
    shape = validate_shape(shape)
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _bounds(ranges)
    if device is not None or (torch is not None and isinstance(rng, torch.Generator)):
        if device is None:
            device = rng.device
        sx, sy, sz = hi_x - lo_x, hi_y - lo_y, hi_z - lo_z

        def draw():   # python-scalar arithmetic: no host->device tensor, so no implicit sync
            r = torch.rand(n, 3, device=device, generator=rng)
            return torch.stack((lo_x + sx * r[:, 0], lo_y + sy * r[:, 1], lo_z + sz * r[:, 2]), dim=-1)

        out = draw()
        if shape == "ellipse_xy":
            for _ in range(_MAX_ROUNDS):
                s = _xy_norm_torch(out[:, 0], out[:, 1], lo_x, hi_x, lo_y, hi_y)
                out = torch.where((s > 1.0).unsqueeze(-1), draw(), out)
            out = limit_command(out, ranges, shape)   # leftovers (~1e-8 of draws): onto the boundary
        return out
    rng = np.random.default_rng() if rng is None else rng
    lo = np.array([lo_x, lo_y, lo_z])
    span = np.array([hi_x - lo_x, hi_y - lo_y, hi_z - lo_z])
    out = lo + span * rng.random((n, 3))
    if shape == "ellipse_xy":
        for _ in range(64):
            bad = _xy_norm_np(out[:, 0], out[:, 1], lo_x, hi_x, lo_y, hi_y) > 1.0
            if not bad.any():
                break
            out[bad] = lo + span * rng.random((int(bad.sum()), 3))
        out = limit_command(out, ranges, shape)
    return out
