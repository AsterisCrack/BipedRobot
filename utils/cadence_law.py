"""Command-dependent gait cadence for training WITHOUT a gait table (env_config.gait_clock_mode: cadence_law).

The table path gets its cadence from Gait Studio (a per-command freq grid, trilinear in the command). This is
the config-defined equivalent, so config.yaml (no reference) has a speed-dependent cadence too:

    s      = elliptic command magnitude in [0, 1]: sqrt((vx/vx_b)^2 + (vy/vy_b)^2 + (wz/wz_b)^2), clamped,
             vx_b = the range bound on vx's side (hi for vx >= 0, -lo below), same for vy / wz
    period = gait_clock_period_max_s + (gait_clock_period_min_s - gait_clock_period_max_s) * s

i.e. the slowest cycle just above the stand threshold, the fastest at the edge of the command ranges. It is
baked into the SAME cadence grid a v2 table carries (``CadenceGrid``: trilinear, each axis clamped), so
MuJoCo (``gait_clock_from_config``), the export (``gait_clock.mode: cadence_grid``, bundle v5) and the Jetson
runtime reuse the table machinery unchanged; Isaac interpolates the identical grid on the GPU
(``grid_frequency_torch``). The clock holds while |v_xy| and |wz| are both below gait_clock_stand_threshold.
The training config is the source of truth: the law reads its period bounds and the command ranges.
"""
from __future__ import annotations

import numpy as np

GRID_POINTS = 9       # per axis (plus 0 if the range straddles it); the law is smooth away from cmd 0
_KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def validate_periods(t_min: float, t_max: float) -> None:
    if not (0.2 < float(t_min) <= float(t_max) < 5.0):
        raise ValueError(f"cadence_law: need 0.2 < gait_clock_period_min_s ({t_min}) <= gait_clock_period_max_s "
                         f"({t_max}) < 5 s (the Jetson accepts 0.2-5 Hz)")


def _bound(v: float, lo: float, hi: float) -> float:
    b = hi if v >= 0.0 else -lo
    return b if b > 1e-9 else float("inf")


def period_s(cmd, ranges: dict, t_min: float, t_max: float) -> float:
    """Gait cycle [s] of the law at one command (vx, vy, wz)."""
    s2 = 0.0
    for v, k in zip(cmd, _KEYS):
        lo, hi = (float(x) for x in ranges[k])
        s2 += (float(v) / _bound(float(v), lo, hi)) ** 2
    s = min(1.0, s2 ** 0.5)
    return float(t_max) + (float(t_min) - float(t_max)) * s


def _axis(lo: float, hi: float) -> np.ndarray:
    a = np.linspace(float(lo), float(hi), GRID_POINTS) if hi > lo else np.array([float(lo)])
    if lo < 0.0 < hi:
        a = np.unique(np.append(a, 0.0))
    return a


def cadence_grid(ranges: dict, t_min: float, t_max: float):
    """The law sampled on a (vx, vy, wz) grid over the command ranges -> utils.motion_reference.CadenceGrid."""
    from utils.motion_reference import CadenceGrid
    validate_periods(t_min, t_max)
    axes = [_axis(*(float(x) for x in ranges[k])) for k in _KEYS]
    freq = np.empty(tuple(len(a) for a in axes))
    for i, vx in enumerate(axes[0]):
        for j, vy in enumerate(axes[1]):
            for k, wz in enumerate(axes[2]):
                freq[i, j, k] = 1.0 / period_s((vx, vy, wz), ranges, t_min, t_max)
    return CadenceGrid(*axes, freq)


def grid_frequency_torch(axes: list, freq, commands):
    """Trilinear lookup of a cadence grid for commands [N, 3] -> [N], each axis clamped (= CadenceGrid.frequency).

    axes: three sorted 1-D tensors; freq: [nx, ny, nz] tensor, all on the commands' device."""
    import torch
    idx, wts = [], []
    for d, g in enumerate(axes):
        x = commands[:, d]
        n = g.numel()
        if n == 1:
            idx.append(torch.zeros_like(x, dtype=torch.long)); wts.append(torch.zeros_like(x))
            continue
        x = torch.clamp(x, g[0], g[-1])
        i0 = torch.clamp(torch.searchsorted(g, x.contiguous(), right=True) - 1, 0, n - 2)
        idx.append(i0); wts.append((x - g[i0]) / (g[i0 + 1] - g[i0]))
    out = torch.zeros_like(commands[:, 0])
    shape = freq.shape
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((wts[0] if di else 1.0 - wts[0]) * (wts[1] if dj else 1.0 - wts[1])
                     * (wts[2] if dk else 1.0 - wts[2]))
                i = torch.clamp(idx[0] + di, max=shape[0] - 1)
                j = torch.clamp(idx[1] + dj, max=shape[1] - 1)
                k = torch.clamp(idx[2] + dk, max=shape[2] - 1)
                out = out + w * freq[i, j, k]
    return out
