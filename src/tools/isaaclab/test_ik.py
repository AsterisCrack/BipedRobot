"""
Test IK for biped feet with interactive target sliders.

Usage:
    python -m src.isaaclab.test_ik --task BipedRobotV2 --num_envs 1
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import torch

from isaaclab.app import AppLauncher

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

parser = argparse.ArgumentParser(description="Biped IK test with interactive targets.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--task",
    type=str,
    default="BipedRobotV2",
    choices=["BipedRobot", "BipedRobotV2"],
    help="Task name.",
)
parser.add_argument("--right-base-frame", type=str, default="r_hip_motor", help="Right leg base frame name.")
parser.add_argument("--left-base-frame", type=str, default="l_hip_motor", help="Left leg base frame name.")
parser.add_argument("--debug-ik", action="store_true", help="Print IK jacobian debug info.")
parser.add_argument("--debug-ik-delta", action="store_true", help="Print IK delta for hip_z joints.")
parser.add_argument("--direct-set", action="store_true", help="Write joint state directly (debug actuator).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from envs.assets.robot.biped_robot import BIPED_ROBOT_CFG
from utils.ik_utils import EndEffectorConfig, MultiEndEffectorIK


JOINT_NAMES_R = ["r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x"]
JOINT_NAMES_L = ["l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x"]
HIP_Z_WEIGHT = 3.0
JOINT_WEIGHTS_R = [HIP_Z_WEIGHT, 1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_WEIGHTS_L = [HIP_Z_WEIGHT, 1.0, 1.0, 1.0, 1.0, 1.0]


def _create_ui() -> tuple[tk.Tk, dict[str, tk.DoubleVar], dict[str, tk.DoubleVar], dict]:
    root = tk.Tk()
    root.title("IK Targets")
    root.geometry("520x640")

    main = ttk.Frame(root, padding="8")
    main.pack(fill=tk.BOTH, expand=True)

    def _make_group(title: str):
        frame = ttk.LabelFrame(main, text=title, padding="6")
        frame.pack(fill=tk.X, pady=6)
        return frame

    def _make_slider(frame, label, var, frm, to):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=8).pack(side=tk.LEFT)
        scale = ttk.Scale(row, from_=frm, to=to, variable=var, orient="horizontal", length=300)
        scale.pack(side=tk.LEFT, padx=4)
        val = ttk.Label(row, text=f"{var.get():.2f}", width=6)
        val.pack(side=tk.LEFT)

        def _update(*_):
            val.configure(text=f"{var.get():.2f}")

        var.trace_add("write", _update)
        return scale

    right_pos = {k: tk.DoubleVar(value=v) for k, v in {"x": 0.0, "y": 0.0, "z": -0.20}.items()}
    left_pos = {k: tk.DoubleVar(value=v) for k, v in {"x": 0.0, "y": 0.0, "z": -0.20}.items()}
    right_rpy = {k: tk.DoubleVar(value=0.0) for k in ["r", "p", "y"]}
    left_rpy = {k: tk.DoubleVar(value=0.0) for k in ["r", "p", "y"]}

    rf = _make_group("Right Foot Target (meters / degrees)")
    _make_slider(rf, "x", right_pos["x"], -0.5, 0.5)
    _make_slider(rf, "y", right_pos["y"], -0.5, 0.5)
    _make_slider(rf, "z", right_pos["z"], -0.5, 0.5)
    _make_slider(rf, "roll", right_rpy["r"], -180.0, 180.0)
    _make_slider(rf, "pitch", right_rpy["p"], -180.0, 180.0)
    _make_slider(rf, "yaw", right_rpy["y"], -180.0, 180.0)

    lf = _make_group("Left Foot Target (meters / degrees)")
    _make_slider(lf, "x", left_pos["x"], -0.5, 0.5)
    _make_slider(lf, "y", left_pos["y"], -0.5, 0.5)
    _make_slider(lf, "z", left_pos["z"], -0.5, 0.5)
    _make_slider(lf, "roll", left_rpy["r"], -180.0, 180.0)
    _make_slider(lf, "pitch", left_rpy["p"], -180.0, 180.0)
    _make_slider(lf, "yaw", left_rpy["y"], -180.0, 180.0)

    ui_state = {"open": True}

    def _on_close():
        ui_state["open"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    return root, {"pos": right_pos, "rpy": right_rpy}, {"pos": left_pos, "rpy": left_rpy}, ui_state


def _design_scene(num_envs: int):
    cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(cfg)
    sim.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.5])

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = BIPED_ROBOT_CFG.copy()
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/IKTargets")
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    markers = VisualizationMarkers(marker_cfg)

    return sim, robot, markers


def main():
    if args_cli.num_envs != 1:
        print("[WARN] IK test is intended for --num_envs 1. Forcing to 1.")

    sim, robot, markers = _design_scene(num_envs=1)
    sim.reset()

    # Warm-up to populate buffers
    for _ in range(5):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    ik = MultiEndEffectorIK(
        robot,
        end_effectors=[
            EndEffectorConfig(name="r_foot", joint_names=JOINT_NAMES_R, joint_weights=JOINT_WEIGHTS_R),
            EndEffectorConfig(name="l_foot", joint_names=JOINT_NAMES_L, joint_weights=JOINT_WEIGHTS_L),
        ],
        lambda_val=0.05,
    )

    if args_cli.debug_ik:
        print(f"[DEBUG] Robot joints: {list(robot.joint_names)}")
        print(f"[DEBUG] Right joints: {JOINT_NAMES_R}")
        print(f"[DEBUG] Left joints: {JOINT_NAMES_L}")

    try:
        r_foot_id = robot.body_names.index("r_foot")
        l_foot_id = robot.body_names.index("l_foot")
    except ValueError as exc:
        raise ValueError("Foot body names not found. Check robot body names.") from exc

    try:
        r_base_id = robot.body_names.index(args_cli.right_base_frame)
        l_base_id = robot.body_names.index(args_cli.left_base_frame)
    except ValueError as exc:
        raise ValueError(
            f"Base frame name not found. Available bodies: {robot.body_names}"
        ) from exc

    ui_root, right_ctrl, left_ctrl, ui_state = _create_ui()

    step_count = 0
    while simulation_app.is_running():
        try:
            ui_root.update()
        except Exception:
            break
        if not ui_state.get("open", True):
            break

        rp = right_ctrl["pos"]
        rr = right_ctrl["rpy"]
        lp = left_ctrl["pos"]
        lr = left_ctrl["rpy"]

        delta_r_pos = torch.tensor([[rp["x"].get(), rp["y"].get(), rp["z"].get()]], device=robot.device)
        delta_l_pos = torch.tensor([[lp["x"].get(), lp["y"].get(), lp["z"].get()]], device=robot.device)

        delta_r_quat = quat_from_euler_xyz(
            torch.tensor([rr["r"].get()], device=robot.device) * torch.pi / 180.0,
            torch.tensor([rr["p"].get()], device=robot.device) * torch.pi / 180.0,
            torch.tensor([rr["y"].get()], device=robot.device) * torch.pi / 180.0,
        )
        delta_l_quat = quat_from_euler_xyz(
            torch.tensor([lr["r"].get()], device=robot.device) * torch.pi / 180.0,
            torch.tensor([lr["p"].get()], device=robot.device) * torch.pi / 180.0,
            torch.tensor([lr["y"].get()], device=robot.device) * torch.pi / 180.0,
        )

        r_base_pos_w = robot.data.body_pos_w[:, r_base_id]
        r_base_quat_w = robot.data.body_quat_w[:, r_base_id]
        l_base_pos_w = robot.data.body_pos_w[:, l_base_id]
        l_base_quat_w = robot.data.body_quat_w[:, l_base_id]

        right_pos = r_base_pos_w + quat_apply(r_base_quat_w, delta_r_pos)
        left_pos = l_base_pos_w + quat_apply(l_base_quat_w, delta_l_pos)
        right_quat = quat_mul(r_base_quat_w, delta_r_quat)
        left_quat = quat_mul(l_base_quat_w, delta_l_quat)

        target_pos_w = {"r_foot": right_pos, "l_foot": left_pos}
        target_quat_w = {"r_foot": right_quat, "l_foot": left_quat}

        joint_targets = ik.compute(target_pos_w, target_quat_w)
        if args_cli.direct_set:
            joint_vel = torch.zeros_like(robot.data.joint_vel)
            robot.write_joint_state_to_sim(joint_targets, joint_vel)
        else:
            robot.set_joint_position_target(joint_targets)

        markers.visualize(
            torch.cat([right_pos, left_pos], dim=0),
            torch.cat([right_quat, left_quat], dim=0),
        )

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if args_cli.debug_ik and step_count % 50 == 0:
            jacobians = robot.root_physx_view.get_jacobians()
            # Jacobian index is body_id - 1
            r_jac = jacobians[:, r_foot_id - 1, :, :]
            l_jac = jacobians[:, l_foot_id - 1, :, :]

            r_ids = [robot.joint_names.index(name) for name in JOINT_NAMES_R]
            l_ids = [robot.joint_names.index(name) for name in JOINT_NAMES_L]

            r_col = JOINT_NAMES_R.index("r_hip_z")
            l_col = JOINT_NAMES_L.index("l_hip_z")

            r_ang_norm = torch.norm(r_jac[:, 3:, r_ids[r_col]], dim=1).mean().item()
            l_ang_norm = torch.norm(l_jac[:, 3:, l_ids[l_col]], dim=1).mean().item()

            print(f"[DEBUG] r_hip_z ang jacobian norm: {r_ang_norm:.6f}")
            print(f"[DEBUG] l_hip_z ang jacobian norm: {l_ang_norm:.6f}")

        if args_cli.debug_ik_delta and step_count % 50 == 0:
            r_hip_idx = robot.joint_names.index("r_hip_z")
            l_hip_idx = robot.joint_names.index("l_hip_z")
            r_cur = robot.data.joint_pos[0, r_hip_idx].item()
            l_cur = robot.data.joint_pos[0, l_hip_idx].item()
            r_tgt = joint_targets[0, r_hip_idx].item()
            l_tgt = joint_targets[0, l_hip_idx].item()
            print(f"[DEBUG] r_hip_z cur {r_cur:.4f} tgt {r_tgt:.4f} d {r_tgt - r_cur:.4f}")
            print(f"[DEBUG] l_hip_z cur {l_cur:.4f} tgt {l_tgt:.4f} d {l_tgt - l_cur:.4f}")
            print(f"[DEBUG] yaw sliders r {rr['y'].get():.1f} l {lr['y'].get():.1f}")

        step_count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
