"""
Preview a joint animation on the biped robot, fixed in place.
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

parser = argparse.ArgumentParser(description="Preview a joint animation on the biped robot.")
parser.add_argument("--npz", type=str, required=True, help="Path to NPZ produced by fbx_to_joint_data.py.")
parser.add_argument("--play", action="store_true", help="Play animation (default: paused).")
parser.add_argument("--loop", action="store_true", help="Loop animation.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base", help="Allow base to move.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "joint_names": list(data["joint_names"]),
        "positions": data["positions"],
        "velocities": data["velocities"],
        "dt": float(data["dt"]),
        "fps": float(data["fps"]),
    }


def design_scene() -> tuple[dict, torch.Tensor]:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0].tolist())

    robot_cfg = BIPED_ROBOT_CFG.copy()
    if args_cli.freeze_base:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    robot_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot}, origins


def _apply_fixed_base(robot: Articulation, origin: torch.Tensor) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin.to(device=robot.device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def _create_ui(
    joint_names: list[str],
    default_fps: float,
) -> tuple[tk.Tk, list[tk.DoubleVar], list[tk.BooleanVar], tk.DoubleVar, dict]:
    root = tk.Tk()
    root.title("Animation Offsets")
    root.geometry("420x720")

    main_frame = ttk.Frame(root, padding="8")
    main_frame.pack(fill=tk.BOTH, expand=True)

    fps_frame = ttk.LabelFrame(main_frame, text="Playback FPS", padding="6")
    fps_frame.pack(fill=tk.X, pady=6)

    fps_var = tk.DoubleVar(value=default_fps if default_fps > 0.0 else 30.0)
    fps_label = ttk.Label(fps_frame, text=f"{fps_var.get():.1f}")
    fps_label.pack(side=tk.RIGHT, padx=6)

    def _on_fps_change(*_):
        fps_label.configure(text=f"{fps_var.get():.1f}")

    fps_scale = ttk.Scale(fps_frame, from_=1.0, to=120.0, variable=fps_var, orient="horizontal")
    fps_scale.pack(fill=tk.X, padx=4)
    fps_var.trace_add("write", _on_fps_change)

    sliders_frame = ttk.LabelFrame(main_frame, text="Joint Offsets (deg)", padding="6")
    sliders_frame.pack(fill=tk.BOTH, expand=True)

    offset_vars: list[tk.DoubleVar] = []
    sign_vars: list[tk.BooleanVar] = []

    def _make_slider(row: int, name: str) -> tuple[tk.DoubleVar, tk.BooleanVar]:
        lbl = ttk.Label(sliders_frame, text=name, width=20)
        lbl.grid(row=row, column=0, padx=2, pady=2, sticky="w")

        var = tk.DoubleVar(value=0.0)
        scale = ttk.Scale(sliders_frame, from_=-180.0, to=180.0, variable=var, orient="horizontal", length=200)
        scale.grid(row=row, column=1, padx=2, pady=2)

        val_lbl = ttk.Label(sliders_frame, text="0.0", width=6)
        val_lbl.grid(row=row, column=2, padx=2, pady=2)

        sign_var = tk.BooleanVar(value=False)
        sign_chk = ttk.Checkbutton(sliders_frame, text="Flip", variable=sign_var)
        sign_chk.grid(row=row, column=3, padx=4, pady=2)

        def _on_change(*_):
            val_lbl.configure(text=f"{var.get():.1f}")

        var.trace_add("write", _on_change)
        return var, sign_var

    for i, name in enumerate(joint_names):
        offset_var, sign_var = _make_slider(i, name)
        offset_vars.append(offset_var)
        sign_vars.append(sign_var)

    ui_state = {"open": True}

    def _on_close():
        ui_state["open"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    return root, offset_vars, sign_vars, fps_var, ui_state


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, anim: dict) -> None:
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()

    joint_names = anim["joint_names"]
    positions = np.unwrap(anim["positions"].astype(np.float32), axis=0)
    velocities = anim["velocities"].astype(np.float32)
    frame_dt = anim["dt"] if anim["dt"] > 0.0 else sim_dt
    default_fps = anim["fps"] if anim["fps"] > 0.0 else (1.0 / frame_dt if frame_dt > 0.0 else 30.0)

    ui_root, offset_vars, sign_vars, fps_var, ui_state = _create_ui(joint_names, default_fps)

    name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    joint_ids = [name_to_idx[name] for name in joint_names]

    num_frames = positions.shape[0]
    play_time = 0.0

    while simulation_app.is_running():
        try:
            ui_root.update()
        except Exception:
            break
        if not ui_state.get("open", True):
            break

        fps_val = float(fps_var.get())
        if fps_val > 0.0:
            frame_dt = 1.0 / fps_val
        if args_cli.freeze_base:
            _apply_fixed_base(robot, origins)

        if args_cli.play:
            play_time += sim_dt * args_cli.speed

        frame_float = play_time / frame_dt if frame_dt > 0.0 else 0.0
        frame_idx = int(frame_float)
        alpha = frame_float - frame_idx
        if args_cli.loop:
            frame_idx = frame_idx % num_frames
            next_idx = (frame_idx + 1) % num_frames
        else:
            frame_idx = min(frame_idx, num_frames - 1)
            next_idx = min(frame_idx + 1, num_frames - 1)
            alpha = min(max(alpha, 0.0), 1.0)

        pos_blend = (1.0 - alpha) * positions[frame_idx] + alpha * positions[next_idx]
        if velocities.size > 0:
            vel_blend = (1.0 - alpha) * velocities[frame_idx] + alpha * velocities[next_idx]
        else:
            vel_blend = None

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        offset_deg = np.array([var.get() for var in offset_vars], dtype=np.float32)
        offset_rad = np.deg2rad(offset_deg)
        sign = np.array([(-1.0 if var.get() else 1.0) for var in sign_vars], dtype=np.float32)
        joint_pos[:, joint_ids] = torch.tensor((pos_blend + offset_rad) * sign, device=robot.device)
        if vel_blend is not None:
            joint_vel[:, joint_ids] = torch.tensor(vel_blend * sign, device=robot.device)

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_data_to_sim()
        sim.step()
        if args_cli.freeze_base:
            _apply_fixed_base(robot, origins)
        robot.update(sim_dt)


def main() -> None:
    npz_path = os.path.abspath(args_cli.npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.5])

    scene_entities, origins = design_scene()
    sim.reset()

    anim = _load_npz(npz_path)
    run_simulator(sim, scene_entities, origins, anim)


if __name__ == "__main__":
    main()
    simulation_app.close()
