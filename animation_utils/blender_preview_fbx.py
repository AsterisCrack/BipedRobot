"""
Blender script: import FBX, play animation, overlay joint names.

Usage (from OS terminal):
  blender --python animation_utils/blender_preview_fbx.py -- --fbx "path/to/file.fbx"

Optional:
  --scale 0.01   # scale import
  --fps 30       # override scene FPS
"""
import argparse
import os
import sys

import bpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender FBX preview with joint names.")
    parser.add_argument("--fbx", type=str, required=True, help="Path to FBX file.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform import scale.")
    parser.add_argument("--fps", type=int, default=None, help="Override scene FPS.")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    return args


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_fbx(path: str, scale: float) -> None:
    bpy.ops.import_scene.fbx(filepath=path, global_scale=scale)


def _find_armature() -> bpy.types.Object | None:
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def _create_label_empties(arm: bpy.types.Object) -> list[bpy.types.Object]:
    empties = []
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in arm.data.bones:
        empty = bpy.data.objects.new(f"label_{bone.name}", None)
        empty.empty_display_size = 0.01
        empty.empty_display_type = "SPHERE"
        bpy.context.collection.objects.link(empty)
        # constrain to bone head
        empty.parent = arm
        empty.parent_type = "BONE"
        empty.parent_bone = bone.name
        empty.location = (0.0, 0.0, 0.0)
        empties.append(empty)
    return empties


def _add_viewport_text(arm: bpy.types.Object) -> None:
    # Draw handler for joint names in the 3D view.
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin("2D_UNIFORM_COLOR")

    def draw_callback():
        region = bpy.context.region
        rv3d = bpy.context.region_data
        if region is None or rv3d is None:
            return
        for bone in arm.pose.bones:
            head_world = arm.matrix_world @ bone.head
            co2d = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, head_world)
            if co2d is None:
                continue
            x, y = co2d
            blf.position(0, x + 6, y + 6, 0)
            blf.size(0, 12)
            blf.draw(0, bone.name)

    import bpy_extras.view3d_utils

    bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), "WINDOW", "POST_PIXEL")


def _setup_timeline(fps: int | None) -> None:
    if fps is not None:
        bpy.context.scene.render.fps = fps

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_current = 1


def main() -> None:
    args = _parse_args()
    fbx_path = os.path.abspath(args.fbx)
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"FBX not found: {fbx_path}")

    _clear_scene()
    _import_fbx(fbx_path, args.scale)

    arm = _find_armature()
    if arm is None:
        raise RuntimeError("No armature found in FBX")

    _setup_timeline(args.fps)
    _create_label_empties(arm)
    _add_viewport_text(arm)

    # Start animation playback
    bpy.ops.screen.animation_play()


if __name__ == "__main__":
    main()
