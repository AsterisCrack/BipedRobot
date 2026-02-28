"""
Inspect an FBX/USD file for skeleton joints and animation range.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

parser = argparse.ArgumentParser(description="Inspect FBX/USD skeleton joints.")
parser.add_argument("--fbx", type=str, default=None, help="Path to FBX file.")
parser.add_argument("--usd", type=str, default=None, help="Path to USD file (skips FBX conversion).")
parser.add_argument(
    "--usd-out",
    type=str,
    default=None,
    help="Optional USD output path if converting from FBX.",
)
parser.add_argument("--force", action="store_true", help="Force FBX to USD conversion.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# Launch Isaac Sim (needed for FBX conversion)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim.converters as converters
from pxr import Usd, UsdSkel


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

    stage = Usd.Stage.Open(usd_path)
    anim = _find_anim_prim(stage)
    if anim is None:
        raise RuntimeError("No UsdSkel.Animation prim found in USD. Conversion may not include animation.")

    joints = anim.GetJointsAttr().Get() or []
    rot_attr = anim.GetRotationsAttr()
    time_samples = rot_attr.GetTimeSamples()

    print("-" * 80)
    print(f"USD: {usd_path}")
    print(f"Joint count: {len(joints)}")
    print("Joints:")
    for name in joints:
        print(f"  - {name}")
    if time_samples:
        print(f"Time samples: {len(time_samples)}")
        print(f"Start: {time_samples[0]}  End: {time_samples[-1]}")
    else:
        print("No rotation time samples found.")
    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
