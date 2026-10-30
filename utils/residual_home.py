"""Derive the residual action map from the gait table and the robot model (`auto`). numpy + mujoco only:
no Isaac, no torch. ONE resolver, used by train.py (which FREEZES the numbers into the run config before the
env is built and the config is saved) and, defensively, by every low-level consumer that still receives `auto`.

env_config keys (each may be `auto`, independently; explicit numbers pass through untouched):

  action_home_deg   auto -> the table's `stand` frame joint_pos, per joint type the mean of L and R rounded to 0.1 deg
                    (a warning if L != R). Needs env_config.reference.npz_path.
  action_scale_rad  auto -> per joint type the narrowest value on a 0.05 rad grid such that every baked cell inside
                    the training command ranges (both legs, table joint_pos clamped to the joint limits) satisfies
                        max|a| + rsi_joint_noise / scale <= scale_headroom,   a = (q - home) / scale.
                    Needs the table too and uses the resolved home.
                    action_scale_mode: asymmetric (default symmetric, = the rule above) sizes the two sides of every joint
                    type separately, each with the same rule on that side's own max deviation (table clamped to the
                    limits, both legs): s_pos from max(q - home, 0), s_neg from max(home - q, 0), so
                        max|a_side| + rsi_joint_noise / s_side <= scale_headroom   on each side.
                    The result is {type: {pos: s_pos, neg: s_neg}} ("pos" = a > 0 = joint angle above home).
  spawn_height_m    auto -> Isaac root z at which the LOWEST sole point of the robot standing at the home joints
                    (base frame upright) is spawn_clearance_m above the floor, by forward kinematics of
                    envs/assets/robotV2/robot_v2.xml (foot mesh vertices; the sole-rectangle corners agree to
                    ~0.03 mm). Independent of the table, so it also serves a hand-set crouch home.
  spawn_clearance_m (default 0.0054 m)    scale_headroom (default 0.96)    action_scale_mode (symmetric | asymmetric)

Frames: the MJCF base_link frame equals the Isaac root frame, so the value is an ISAAC root z. The MuJoCo env
adds `height - 0.29` to an XML root at 0.30 and therefore reads 10 mm higher; that convention is unchanged.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import warnings
from functools import lru_cache

import numpy as np

from utils.residual_action import (JOINT_TYPES, RESIDUAL_SPAWN_HEIGHT_M, SCALE_SIDES, check_residual_map,
                                   expand_by_type, home_rad, scale_pair_rad, split_scale)

AUTO = "auto"
DEFAULT_SPAWN_CLEARANCE_M = 0.0054
DEFAULT_SCALE_HEADROOM = 0.96
DEFAULT_RSI_JOINT_NOISE = 0.05          # BipedEnvCfg.reference_rsi_joint_noise
SCALE_STEP = 0.05                       # rad
HOME_DECIMALS = 1                       # deg
LR_WARN_DEG = 0.05                      # table stand L/R difference that triggers a warning
SPAWN_DECIMALS = 4                      # m (0.1 mm)
AUTO_KEYS = ("action_home_deg", "action_scale_rad", "spawn_height_m")
SCALE_MODES = ("symmetric", "asymmetric")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
XML_REL = os.path.join("envs", "assets", "robotV2", "robot_v2.xml")
MESHES_REL = os.path.join("robot_hardware", "robot_description", "meshes")   # BipedRobotHardware submodule
ROBOT_PY_REL = os.path.join("envs", "assets", "robotV2", "biped_robot.py")

# Policy joint order (== envs.mujoco.biped_env_v2.JOINT_NAMES, the table's joint names): L,R interleaved per type.
POLICY_JOINT_NAMES = ("l_hip_yaw", "r_hip_yaw", "l_hip_roll_joint", "r_hip_roll_joint",
                      "l_hip_pitch_joint", "r_hip_pitch_joint", "l_knee_joint", "r_knee_joint",
                      "l_ankle_roll_joint", "r_ankle_roll_joint", "l_ankle_pitch_joint", "r_ankle_pitch_joint")


def is_auto(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def any_auto(env_config) -> bool:
    return any(is_auto(_get(env_config, k)) for k in AUTO_KEYS)


# ----------------------------------------------------------------------------------------------- config access
def _plain(x):
    return x.model_dump() if hasattr(x, "model_dump") else x


def _get(env_config, key, default=None):
    """env_config may be a pydantic EnvConfig, a namespace or a plain dict."""
    if env_config is None:
        return default
    val = env_config.get(key) if isinstance(env_config, dict) else getattr(env_config, key, None)
    return default if val is None else val


def _type_dict(value, name: str) -> dict:
    d = {k: float(v) for k, v in dict(_plain(value)).items()}
    expand_by_type(d, name)                         # keys must be exactly the six joint types
    return d


def _scale_dict(value, name: str) -> dict:
    """{type: number | {pos, neg}} -> {type: float | {pos: float, neg: float}} (numbers stay plain floats)."""
    d = {}
    for k, v in dict(_plain(value)).items():
        v = _plain(v)
        if isinstance(v, dict):
            split_scale(v, f"{name}.{k}")           # exactly the keys pos / neg
            d[k] = {s: float(v[s]) for s in SCALE_SIDES}
        else:
            d[k] = float(v)
    expand_by_type({k: 0.0 for k in d}, name)       # keys must be exactly the six joint types
    return d


# ----------------------------------------------------------------------------------------------- robot limits
def joint_limits_deg(root_dir: str = PROJECT_ROOT) -> np.ndarray:
    """[12, 2] policy-order limits [deg] from envs/assets/robotV2/biped_robot.py _JOINT_LIMITS_DEG (read as a literal:
    the module itself needs Isaac). The same source export_onnx uses; tests/test_residual_action_map.py LIM_DEG agrees."""
    with open(os.path.join(root_dir, ROBOT_PY_REL), encoding="utf-8") as fh:
        for node in ast.parse(fh.read()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_JOINT_LIMITS_DEG" for t in node.targets):
                return np.asarray(ast.literal_eval(node.value), dtype=np.float64)
    raise RuntimeError(f"_JOINT_LIMITS_DEG not found in {ROBOT_PY_REL}")


# ----------------------------------------------------------------------------------------------- gait table
def select_grid_cells(grid, bound: float) -> np.ndarray:
    """Grid points a command with |c| <= bound can interpolate from: every |g| <= bound plus, when the bound is not itself
    a grid point, the first grid point beyond it (the table sampler clamps and blends the two neighbours trilinearly)."""
    grid = np.asarray(grid, dtype=np.float64)
    keep = np.abs(grid) <= bound + 1e-6
    beyond = np.abs(grid)[~keep]
    if beyond.size and not np.any(np.isclose(np.abs(grid), bound, atol=1e-6)):
        keep |= np.isclose(np.abs(grid), beyond.min(), atol=1e-6)
    return keep


def load_table(npz_path: str, root_dir: str = PROJECT_ROOT) -> dict:
    """Joint-position part of a baked gait table, in POLICY joint order, float64 radians:
    {path, digest, grids (vx, vy, wz), q [nvx, nvy, nwz, frames, 12], stand [12]}."""
    from utils.motion_reference import resolve_table_path
    path = resolve_table_path(npz_path, root_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"gait table not found: {path} (bake it with `python -m gait_studio.bake` in "
                                f"robot_motion_reference/)")
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["field_names"]]
        widths = [int(w) for w in z["field_widths"]]
        if names[0] != "joint_pos" or widths[0] != 12:
            raise ValueError(f"{path}: first table field must be joint_pos (12), got {names[0]} ({widths[0]})")
        table_joints = [str(s) for s in z["joint_names"]]
        missing = [n for n in POLICY_JOINT_NAMES if n not in table_joints]
        if missing:
            raise ValueError(f"{path}: table has no joint(s) {missing}; table joints {table_joints}")
        order = [table_joints.index(n) for n in POLICY_JOINT_NAMES]
        return {
            "path": path,
            "digest": str(z["style_digest"]) if "style_digest" in z else "",
            "grids": tuple(np.asarray(z[k], dtype=np.float64) for k in ("grid_vx", "grid_vy", "grid_wz")),
            "q": np.asarray(z["data"][..., :12], dtype=np.float64)[..., order],
            "stand": np.asarray(z["stand"][:12], dtype=np.float64)[order],
        }


def home_from_stand(stand_rad, *, warn: bool = True) -> tuple[dict, float]:
    """({joint type: deg rounded to 0.1}, largest L-R difference [deg]) from the table's stand joint_pos (policy order)."""
    deg = np.degrees(np.asarray(stand_rad, dtype=np.float64)).reshape(6, 2)
    asym = float(np.abs(deg[:, 0] - deg[:, 1]).max())
    if warn and asym > LR_WARN_DEG:
        warnings.warn(f"gait table stand frame is not L/R symmetric (max |L-R| = {asym:.3f} deg): the residual home "
                      f"uses the per-type mean {dict(zip(JOINT_TYPES, np.round(deg.mean(axis=1), 3)))}", stacklevel=2)
    home = {t: float(np.round(m, HOME_DECIMALS) + 0.0) for t, m in zip(JOINT_TYPES, deg.mean(axis=1))}   # +0.0: no -0.0
    return home, asym


def command_bounds(env_config, grids) -> tuple[list[float], list[str]]:
    """|vx|, |vy|, |wz| bounds of the training command ranges; an axis without a range falls back to the whole grid."""
    cmds = _plain(_get(env_config, "commands")) or {}
    ranges = _plain((_plain(cmds.get("base_velocity")) or {}).get("ranges")) or {}
    out, notes = [], []
    for key, grid in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), grids):
        if key in ranges and ranges[key] is not None:
            out.append(max(abs(float(v)) for v in ranges[key]))
        else:
            out.append(float(np.abs(grid).max()))
            notes.append(f"commands.base_velocity.ranges.{key} missing: whole grid used")
    return out, notes


def required_scales(table: dict, home_rad_12, bounds, noise: float, headroom: float, lim_rad: np.ndarray) -> dict:
    """Narrowest 0.05-grid scale per joint type (see module docstring) and the numbers behind it."""
    masks = [select_grid_cells(g, b) for g, b in zip(table["grids"], bounds)]
    if not all(m.any() for m in masks):
        raise ValueError("no baked grid point inside the training command ranges")
    q = table["q"][masks[0]][:, masks[1]][:, :, masks[2]].reshape(-1, 12)
    dev = np.abs(np.clip(q, lim_rad[:, 0], lim_rad[:, 1]) - np.asarray(home_rad_12)).max(axis=0)   # [12] rad
    dev_t = dev.reshape(6, 2).max(axis=1)                                                          # per type, both legs
    req = (dev_t + noise) / headroom
    scale = np.maximum(SCALE_STEP, np.round(np.ceil(req / SCALE_STEP - 1e-9) * SCALE_STEP, 2))
    return {"scale": scale, "required": req, "dev_rad": dev_t, "max_a": dev_t / scale,
            "need": dev_t / scale + noise / scale, "n_cells": int(q.shape[0] // table["q"].shape[3]),
            "cells_selected": [int(m.sum()) for m in masks]}


def required_scales_asym(table: dict, home_rad_12, bounds, noise: float, headroom: float, lim_rad: np.ndarray) -> dict:
    """action_scale_mode asymmetric: the rule of required_scales applied to each side of each joint type on its own
    deviation. pos = table above home (a > 0), neg = table below home (a < 0); both legs share a value (max of L, R).
    Returns scale_pos / scale_neg [6] and, per side, required / dev_rad / max_a / need (suffix _pos / _neg)."""
    masks = [select_grid_cells(g, b) for g, b in zip(table["grids"], bounds)]
    if not all(m.any() for m in masks):
        raise ValueError("no baked grid point inside the training command ranges")
    q = table["q"][masks[0]][:, masks[1]][:, :, masks[2]].reshape(-1, 12)
    d = np.clip(q, lim_rad[:, 0], lim_rad[:, 1]) - np.asarray(home_rad_12)
    out = {"n_cells": int(q.shape[0] // table["q"].shape[3]), "cells_selected": [int(m.sum()) for m in masks]}
    for side, dev in (("pos", np.maximum(d, 0.0).max(axis=0)), ("neg", np.maximum(-d, 0.0).max(axis=0))):
        dev_t = dev.reshape(6, 2).max(axis=1)
        req = (dev_t + noise) / headroom
        scale = np.maximum(SCALE_STEP, np.round(np.ceil(req / SCALE_STEP - 1e-9) * SCALE_STEP, 2))
        out.update({f"scale_{side}": scale, f"required_{side}": req, f"dev_rad_{side}": dev_t,
                    f"max_a_{side}": dev_t / scale, f"need_{side}": dev_t / scale + noise / scale})
    return out


# ----------------------------------------------------------------------------------------------- spawn height (FK)
@lru_cache(maxsize=4)
def _fk_model(root_dir: str):
    import mujoco
    xml = os.path.join(root_dir, XML_REL)
    if not os.path.exists(xml):
        raise FileNotFoundError(f"robot MJCF not found: {xml}")
    with open(xml, encoding="utf-8") as fh:
        text = fh.read()
    # from_xml_string has no base directory for the MJCF's relative mesh paths: make them absolute.
    meshes = os.path.join(root_dir, MESHES_REL).replace("\\", "/")
    text = re.sub(r'(file=")[^"]*/meshes/', lambda m: m.group(1) + meshes + "/", text)
    return mujoco.MjModel.from_xml_string(text)


def lowest_sole_z(home_deg_by_type: dict | None, root_dir: str = PROJECT_ROOT) -> dict:
    """Forward kinematics at the home joints (None = all zero), base frame upright at z = 0: the lowest point of the foot
    collision meshes and of the sole rectangles (foot_sole_offset_m / foot_sole_half_extents_m defaults), both relative to
    the base frame, so  spawn z = clearance - lowest_z."""
    import mujoco
    m = _fk_model(os.path.abspath(root_dir))
    d = mujoco.MjData(m)
    q12 = (np.zeros(12) if home_deg_by_type is None else np.radians(expand_by_type(home_deg_by_type, "home")))
    adr = lambda name: int(np.asarray(m.joint(name).qposadr).reshape(-1)[0])
    d.qpos[:] = 0.0
    root = adr("root")
    d.qpos[root + 3] = 1.0                                   # identity orientation, z = 0
    for name, v in zip(POLICY_JOINT_NAMES, q12):
        d.qpos[adr(name)] = v
    mujoco.mj_kinematics(m, d)
    mesh_z, rect_z = [], []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        body = m.body(f"{side}_foot_link_1").id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a = int(m.geom_dataid[g])
                v = m.mesh_vert[m.mesh_vertadr[a]: m.mesh_vertadr[a] + m.mesh_vertnum[a]]
                mesh_z.append(float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()))
        rot, pos = d.xmat[body].reshape(3, 3), d.xpos[body]
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                corner = np.array([0.0163 + sx * 0.0450, sign * 0.0211 + sy * 0.0186, -0.0316])
                rect_z.append(float((pos + rot @ corner)[2]))
    if not mesh_z:
        raise RuntimeError("no foot mesh geometry found in the MJCF")
    return {"mesh_z": min(mesh_z), "rect_z": min(rect_z)}


def spawn_height_for(home_deg_by_type: dict | None, clearance_m: float, root_dir: str = PROJECT_ROOT) -> tuple[float, dict]:
    """(Isaac root z rounded to 0.1 mm, FK detail incl. the clearance actually reached after rounding)."""
    fk = lowest_sole_z(home_deg_by_type, root_dir)
    z = round(clearance_m - fk["mesh_z"], SPAWN_DECIMALS)
    fk.update(spawn_unrounded=clearance_m - fk["mesh_z"], clearance_m=z + fk["mesh_z"],
              clearance_rect_m=z + fk["rect_z"])
    return float(z), fk


# ----------------------------------------------------------------------------------------------- the resolver
def resolve_residual_map(env_config, root_dir: str | None = None) -> tuple[dict | None, dict | None, float | None, dict]:
    """(home_deg {type: deg}, scale_rad {type: rad}, spawn_height_m, report) for an env_config whose action_home_deg /
    action_scale_rad / spawn_height_m may each be `auto`. Explicit values are returned as given (floats). Non-residual maps
    return (None, None, spawn or None, report); `auto` home / scale need action_map: residual.
    Raises ValueError / FileNotFoundError with a plain message when an `auto` key lacks its inputs."""
    root = os.path.abspath(root_dir or PROJECT_ROOT)
    amap = _get(env_config, "action_map", "absolute")
    home_cfg, scale_cfg = _get(env_config, "action_home_deg"), _get(env_config, "action_scale_rad")
    spawn_cfg = _get(env_config, "spawn_height_m")
    clearance = float(_get(env_config, "spawn_clearance_m", DEFAULT_SPAWN_CLEARANCE_M))
    headroom = float(_get(env_config, "scale_headroom", DEFAULT_SCALE_HEADROOM))
    mode = str(_get(env_config, "action_scale_mode", "symmetric")).strip().lower()
    if mode not in SCALE_MODES:
        raise ValueError(f"env_config.action_scale_mode must be one of {list(SCALE_MODES)}, got {mode!r}")
    if not 0.0 < headroom <= 1.0:
        raise ValueError(f"env_config.scale_headroom must be in (0, 1], got {headroom}")
    ref = _plain(_get(env_config, "reference")) or {}
    npz_path = ref.get("npz_path")
    noise = float(ref.get("rsi_joint_noise", DEFAULT_RSI_JOINT_NOISE))
    report: dict = {"sources": {k: ("auto" if is_auto(_get(env_config, k)) else "explicit") for k in AUTO_KEYS},
                    "clearance_m": clearance, "headroom": headroom, "rsi_joint_noise": noise, "notes": []}
    if mode != "symmetric":
        report["scale_mode"] = mode

    if amap != "residual":
        if is_auto(home_cfg) or is_auto(scale_cfg):
            raise ValueError(f"action_home_deg / action_scale_rad: auto only applies to action_map: residual "
                             f"(action_map is {amap!r})")
        spawn = None
        if is_auto(spawn_cfg):
            spawn, fk = spawn_height_for(None, clearance, root)
            report["spawn_fk"] = fk
        elif spawn_cfg is not None:
            spawn = float(spawn_cfg)
        return None, None, spawn, report

    table = None

    def need_table(key: str) -> dict:
        nonlocal table
        if not npz_path:
            raise ValueError(f"env_config.{key}: auto needs env_config.reference.npz_path (the gait table it is "
                             f"derived from); set the reference block or write explicit numbers")
        if table is None:
            table = load_table(npz_path, root)
            report["table"] = {"path": table["path"], "digest": table["digest"]}
        return table

    # ---- home
    if is_auto(home_cfg):
        tab = need_table("action_home_deg")
        home, asym = home_from_stand(tab["stand"])
        report["stand_lr_asym_deg"] = asym
    elif not home_cfg:
        raise ValueError("action_map: residual needs env_config.action_home_deg (dict by joint type, or auto)")
    else:
        home = _type_dict(home_cfg, "action_home_deg")
    home_12 = home_rad(home)

    # ---- scale
    lim = np.radians(joint_limits_deg(root))
    if is_auto(scale_cfg):
        tab = need_table("action_scale_rad")
        bounds, notes = command_bounds(env_config, tab["grids"])
        if mode == "asymmetric":
            rs = required_scales_asym(tab, home_12, bounds, noise, headroom, lim)
            scale = {t: {"pos": float(p), "neg": float(n)} for t, p, n in zip(JOINT_TYPES, rs["scale_pos"], rs["scale_neg"])}
            report["scale"] = {"bounds": bounds, **{k: v for k, v in rs.items() if not k.startswith("scale_")}}
        else:
            rs = required_scales(tab, home_12, bounds, noise, headroom, lim)
            scale = {t: float(s) for t, s in zip(JOINT_TYPES, rs["scale"])}
            report["scale"] = {"bounds": bounds, **{k: v for k, v in rs.items() if k != "scale"}}
        report["notes"] += notes
    elif not scale_cfg:
        raise ValueError("action_map: residual needs env_config.action_scale_rad (dict by joint type, or auto)")
    else:
        scale = _scale_dict(scale_cfg, "action_scale_rad")
        if mode == "asymmetric":
            report["notes"].append("action_scale_mode: asymmetric only applies to action_scale_rad: auto; explicit "
                                   "action_scale_rad is used as written")
    s_pos, s_neg = scale_pair_rad(scale)
    check_residual_map(home_12, s_pos, lim[:, 0], lim[:, 1], scale_neg=s_neg)

    # ---- spawn height (from the home, not from the table)
    if is_auto(spawn_cfg):
        spawn, fk = spawn_height_for(home, clearance, root)
        report["spawn_fk"] = fk
    elif spawn_cfg is None:
        spawn = RESIDUAL_SPAWN_HEIGHT_M
    else:
        spawn = float(spawn_cfg)
    report["home_deg"], report["scale_rad"], report["spawn_height_m"] = home, scale, spawn
    return home, scale, spawn, report


def summary(home, scale, spawn, report=None) -> str:
    """One line for the training log."""
    h = " ".join(f"{k}={home[k]:g}" for k in JOINT_TYPES) if home else "-"
    def _s(v):
        p, n = split_scale(v)
        return f"{p:g}" if p == n else f"+{p:g}/-{n:g}"
    s = " ".join(f"{k}={_s(scale[k])}" for k in JOINT_TYPES) if scale else "-"
    out = f"home_deg[{h}] scale_rad[{s}] spawn_height_m={spawn if spawn is None else format(spawn, 'g')}"
    fk = (report or {}).get("spawn_fk")
    if fk:
        out += f" (sole clearance {fk['clearance_m'] * 1e3:.2f} mm)"
    tab = (report or {}).get("table")
    if tab:
        out += f" from {os.path.basename(tab['path'])}" + (f" [{tab['digest']}]" if tab["digest"] else "")
    return out


# ----------------------------------------------------------------------------------------------- freeze / defensive
def freeze_residual_map(config, root_dir: str | None = None, *, warn_unfrozen: bool = False):
    """Replace every `auto` in config.train.env_config by its resolved number, IN PLACE. Call right after the Config is loaded
    and before the env is built and the config is saved (train.py): every later consumer (Isaac env, MuJoCo env, ONNX export
    from the run's config.yaml, policy gate, sim2real, play) then sees plain numbers, and a later re-bake cannot change the
    map of an existing run. Explicit keys are left exactly as they are. Returns the report, or None if nothing was `auto`."""
    ec = getattr(getattr(config, "train", None), "env_config", None)
    if ec is None or not any_auto(ec):
        return None
    home, scale, spawn, rep = resolve_residual_map(ec, root_dir)
    for key, val in zip(AUTO_KEYS, (home, scale, spawn)):
        if is_auto(_get(ec, key)):
            if isinstance(ec, dict):
                ec[key] = val
            else:
                setattr(ec, key, val)
    if warn_unfrozen:
        print("[residual-map] WARNING: this config still says `auto`; the values below come from the CURRENT gait table, so "
              "a re-bake changes them. A training run freezes them into its saved config.yaml; use that file here.",
              file=sys.stderr)
    print("[residual-map] " + ("resolved (not frozen): " if warn_unfrozen else "frozen into the run config: ")
          + summary(home, scale, spawn, rep))
    return rep


def resolve_if_auto(env_config, who: str, root_dir: str | None = None):
    """Low-level consumers: (home_deg, scale_rad, spawn_height_m) if any key is `auto` (loud warning: table-dependent, NOT
    frozen), else None."""
    if env_config is None or not any_auto(env_config):
        return None
    home, scale, spawn, rep = resolve_residual_map(env_config, root_dir)
    print(f"[residual-map] WARNING ({who}): received `auto` for "
          f"{[k for k in AUTO_KEYS if is_auto(_get(env_config, k))]}; resolved from the CURRENT gait table, so the values "
          f"depend on the bake and are NOT frozen. Resolve at the entry point (train.py does) or write numbers. "
          + summary(home, scale, spawn, rep), file=sys.stderr)
    return home, scale, spawn
