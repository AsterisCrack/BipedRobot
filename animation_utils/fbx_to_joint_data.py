"""
Convert an FBX/USD animation to robot joint positions and velocities.
"""
import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

parser = argparse.ArgumentParser(description="Extract joint trajectories from FBX/USD.")
parser.add_argument("--fbx", type=str, default=None, help="Path to FBX file.")
parser.add_argument("--usd", type=str, default=None, help="Path to USD file (skips FBX conversion).")
parser.add_argument("--usd-out", type=str, default=None, help="Optional USD output path if converting from FBX.")
parser.add_argument("--force", action="store_true", help="Force FBX to USD conversion.")
parser.add_argument("--joint-map", type=str, required=True, help="Path to joint mapping JSON.")
parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
parser.add_argument("--fps", type=float, default=None, help="Override FPS and resample time samples.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch Isaac Sim (needed for FBX conversion)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim.converters as converters
from pxr import Gf, Usd, UsdSkel


def _convert_fbx_to_usd(fbx_path: str, usd_out: str | None, force: bool) -> str:
    if usd_out is None:
        base, _ = os.path.splitext(fbx_path)
        usd_out = base + ".usd"
    usd_out = os.path.abspath(usd_out)

    cfg = converters.MeshConverterCfg(
        asset_path=fbx_path,
        usd_dir=os.path.dirname(usd_out),
        usd_file_name=os.path.basename(usd_out),
        force_usd_conversion=force,
        make_instanceable=False,
        collision_props=None,
        mesh_collision_props=None,
    )
    converter = converters.MeshConverter(cfg)
    return converter.usd_path


def _find_anim_prim(stage: Usd.Stage) -> UsdSkel.Animation | None:
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdSkel.Animation):
            return UsdSkel.Animation(prim)
    return None


def _quat_to_axis_angle(quat: Gf.Quatf | Gf.Quatd) -> tuple[np.ndarray, float]:
    w = quat.GetReal()
    v = quat.GetImaginary()
    v_np = np.array([v[0], v[1], v[2]], dtype=np.float64)
    norm_v = np.linalg.norm(v_np)
    if norm_v < 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    angle = 2.0 * math.atan2(norm_v, w)
    axis = v_np / norm_v
    return axis, angle


def _angle_about_axis(quat: Gf.Quatf | Gf.Quatd, axis: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rot_axis, angle = _quat_to_axis_angle(quat)
    sign = 1.0 if np.dot(rot_axis, axis) >= 0.0 else -1.0
    return angle * sign


def _load_joint_map(map_path: str) -> dict[str, Any]:
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_mapping(
    joint_map: dict[str, Any],
    fbx_joints: list[str],
) -> tuple[list[str], list[int], list[np.ndarray], list[float], list[float], list[float], list[str | None], list[str]]:
    fbx_index = {name: i for i, name in enumerate(fbx_joints)}
    robot_joint_names = []
    fbx_indices = []
    axes = []
    signs = []
    offsets = []
    scales = []
    euler_components = []
    euler_orders = []

    for entry in joint_map.get("joints", []):
        robot_joint = entry.get("robot_joint")
        fbx_joint = entry.get("fbx_joint")
        if not robot_joint:
            continue
        if not fbx_joint:
            raise ValueError(f"Missing fbx_joint for robot_joint '{robot_joint}'")
        if fbx_joint not in fbx_index:
            raise ValueError(f"FBX joint '{fbx_joint}' not found in animation")

        axis = np.array(entry.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        sign = float(entry.get("sign", 1.0))
        offset = float(entry.get("offset", 0.0))
        offset_deg = entry.get("offset_deg")
        if offset_deg is not None:
            offset += math.radians(float(offset_deg))
        scale = float(entry.get("scale", 1.0))
        euler_component = entry.get("euler_component")
        euler_order = entry.get("euler_order", "XYZ")
        if euler_component is not None:
            euler_component = str(euler_component).lower()
            if euler_component not in {"x", "y", "z"}:
                raise ValueError(f"Invalid euler_component '{euler_component}' for '{robot_joint}'")
            if str(euler_order).upper() != "XYZ":
                raise ValueError("Only euler_order 'XYZ' is supported currently")
            euler_order = "XYZ"

        robot_joint_names.append(robot_joint)
        fbx_indices.append(fbx_index[fbx_joint])
        axes.append(axis)
        signs.append(sign)
        offsets.append(offset)
        scales.append(scale)
        euler_components.append(euler_component)
        euler_orders.append(euler_order)

    return robot_joint_names, fbx_indices, axes, signs, offsets, scales, euler_components, euler_orders


def _quat_to_euler_xyz_deg(quat: Gf.Quatf | Gf.Quatd) -> tuple[float, float, float]:
    w = float(quat.GetReal())
    v = quat.GetImaginary()
    x = float(v[0])
    y = float(v[1])
    z = float(v[2])

    # XYZ Tait-Bryan angles (roll, pitch, yaw)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)


def _sample_times(rot_attr: Usd.Attribute, fps: float | None, time_code_rate: float) -> list[float]:
    if fps is None:
        return rot_attr.GetTimeSamples()
    time_samples = rot_attr.GetTimeSamples()
    if not time_samples:
        return []
    start = time_samples[0]
    end = time_samples[-1]
    dt = time_code_rate / fps
    times = []
    t = start
    while t <= end + 1e-8:
        times.append(t)
        t += dt
    return times


def main() -> None:
    if not args_cli.fbx and not args_cli.usd:
        raise ValueError("Provide --fbx or --usd")

    if args_cli.fbx:
        fbx_path = os.path.abspath(args_cli.fbx)
        if not os.path.exists(fbx_path):
            raise FileNotFoundError(f"FBX not found: {fbx_path}")
        usd_path = _convert_fbx_to_usd(fbx_path, args_cli.usd_out, args_cli.force)
    else:
        usd_path = os.path.abspath(args_cli.usd)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD not found: {usd_path}")
        fbx_path = None

    joint_map = _load_joint_map(os.path.abspath(args_cli.joint_map))

    stage = Usd.Stage.Open(usd_path)
    anim = _find_anim_prim(stage)
    if anim is None:
        raise RuntimeError("No UsdSkel.Animation prim found in USD. Conversion may not include animation.")

    fbx_joints = anim.GetJointsAttr().Get() or []
    rot_attr = anim.GetRotationsAttr()
    time_code_rate = float(stage.GetTimeCodesPerSecond() or stage.GetFramesPerSecond() or 30.0)
    time_samples = _sample_times(rot_attr, args_cli.fps, time_code_rate)

    if not time_samples:
        raise RuntimeError("No rotation time samples found in animation")

    (
        robot_joint_names,
        fbx_indices,
        axes,
        signs,
        offsets,
        scales,
        euler_components,
        euler_orders,
    ) = _build_mapping(joint_map, list(fbx_joints))

    num_frames = len(time_samples)
    num_joints = len(robot_joint_names)
    positions = np.zeros((num_frames, num_joints), dtype=np.float32)

    for t_idx, t in enumerate(time_samples):
        rots = rot_attr.Get(t)
        if rots is None:
            raise RuntimeError(f"Missing rotations at time {t}")
        for j_idx, fbx_idx in enumerate(fbx_indices):
            quat = rots[fbx_idx]
            if euler_components[j_idx] is not None:
                ex, ey, ez = _quat_to_euler_xyz_deg(quat)
                comp = euler_components[j_idx]
                if comp == "x":
                    angle = math.radians(ex)
                elif comp == "y":
                    angle = math.radians(ey)
                else:
                    angle = math.radians(ez)
            else:
                angle = _angle_about_axis(quat, axes[j_idx])
            angle = (angle * scales[j_idx]) + offsets[j_idx]
            angle *= signs[j_idx]
            positions[t_idx, j_idx] = angle

            # Unwrap to avoid +/-pi jumps in extracted angles
            positions = np.unwrap(positions, axis=0)

    if args_cli.fps is None:
        if len(time_samples) >= 2:
            dt_tc = float(time_samples[1] - time_samples[0])
            dt = dt_tc / time_code_rate if time_code_rate > 0.0 else 0.0
            fps = 1.0 / dt if dt > 0.0 else 0.0
        else:
            dt = 0.0
            fps = 0.0
    else:
        fps = args_cli.fps
        dt = 1.0 / fps

    velocities = np.zeros_like(positions)
    if num_frames >= 2 and dt > 0.0:
        velocities[:-1] = (positions[1:] - positions[:-1]) / dt
        velocities[-1] = velocities[-2]

    out_path = os.path.abspath(args_cli.out)
    np.savez(
        out_path,
        joint_names=np.array(robot_joint_names, dtype=object),
        positions=positions,
        velocities=velocities,
        fps=fps,
        dt=dt,
        source_usd=usd_path,
        source_fbx=fbx_path or "",
    )

    print(f"Wrote joint data: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
