"""Residual action map helpers (action_map: residual). numpy + math only: no Isaac, no torch.

    target = clamp(home + scale_j * clamp(a, -1, 1), joint_min, joint_max)
    a      = clamp((target - home) / scale_j, -1, 1)

`home` is the crouched, flat-footed pose that a = 0 means; it is also the articulation's
default_joint_pos, so every "relative to default" quantity (joint_pos_rel obs, joint_deviation_*,
dof_pos_l2, knee bend) is measured from it. Both are configured BY JOINT TYPE in config.yaml
(env_config.action_home_deg / action_scale_rad, same value for the left and right joint) and
expanded here to the 12 policy slots.

Policy joint order (Isaac, MuJoCo and the exported bundle all agree): L,R interleaved per type,
    0 l_hip_yaw  1 r_hip_yaw  2 l_hip_roll  3 r_hip_roll  4 l_hip_pitch  5 r_hip_pitch
    6 l_knee     7 r_knee     8 l_ankle_roll 9 r_ankle_roll 10 l_ankle_pitch 11 r_ankle_pitch
"""
from __future__ import annotations

import math

import numpy as np

JOINT_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll", "ankle_pitch")

# Spawn base height for the residual home: the V2 default of 0.29 m minus the 3.9 mm the
# crouch lowers the base (home hip -10 / knee -26 / ankle -16 deg).
RESIDUAL_SPAWN_HEIGHT_M = 0.286


def expand_by_type(values: dict, name: str) -> list[float]:
    """{type: value} -> 12 floats in policy order (each type repeated for l, r)."""
    values = dict(values or {})
    unknown = set(values) - set(JOINT_TYPES)
    missing = set(JOINT_TYPES) - set(values)
    if unknown or missing:
        raise ValueError(f"{name}: need exactly the keys {list(JOINT_TYPES)}; "
                         f"missing {sorted(missing)}, unknown {sorted(unknown)}")
    return [float(values[t]) for t in JOINT_TYPES for _ in range(2)]


def home_rad(action_home_deg: dict) -> np.ndarray:
    return np.radians(np.asarray(expand_by_type(action_home_deg, "action_home_deg"), dtype=np.float64))


def scale_rad(action_scale_rad: dict) -> np.ndarray:
    s = np.asarray(expand_by_type(action_scale_rad, "action_scale_rad"), dtype=np.float64)
    if not np.all(s > 0.0):
        raise ValueError(f"action_scale_rad must be > 0 for every joint type, got {action_scale_rad}")
    return s


def check_residual_map(home, scale, lo, hi, mirror_perm=None, mirror_signs=None) -> None:
    """Raise unless the map is usable: home strictly inside the limits, scale > 0, and home
    invariant under the left/right mirror (home[l] == home[r] for the all-+1 V2 signs)."""
    home, scale = np.asarray(home, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if home.shape != (12,) or scale.shape != (12,):
        raise ValueError(f"home/scale must have 12 entries, got {home.shape}/{scale.shape}")
    if np.any(scale <= 0.0):
        raise ValueError("action scale must be > 0")
    if np.any(home <= lo) or np.any(home >= hi):
        raise ValueError("residual home must be strictly inside the joint limits")
    perm = [i ^ 1 for i in range(12)] if mirror_perm is None else list(mirror_perm)  # (0,1),(2,3).. L<->R
    signs = np.ones(12) if mirror_signs is None else np.asarray(mirror_signs, dtype=np.float64)
    if not np.allclose(home[perm] * signs, home, atol=1e-9) or not np.allclose(scale[perm], scale, atol=1e-9):
        raise ValueError("residual home/scale must be left/right symmetric (mirror augmentation "
                         "and the exported mirror table assume it)")


def actions_to_targets_np(a, home, scale, lo, hi):
    return np.clip(home + scale * np.clip(a, -1.0, 1.0), lo, hi)


def targets_to_actions_np(t, home, scale):
    return np.clip((np.asarray(t) - home) / scale, -1.0, 1.0)
