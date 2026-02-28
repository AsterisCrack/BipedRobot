"""
Debug script for Biped Robot control with Tkinter UI.
Modeled after try_symmetry.py without symmetry logic.
"""
import argparse
import os
import sys
import math
import tkinter as tk
from tkinter import ttk

import torch

from isaaclab.app import AppLauncher

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

parser = argparse.ArgumentParser(description="Debug Biped Robot in Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--task",
    type=str,
    default="BipedRobot",
    choices=["BipedRobot", "BipedRobotV2"],
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Environment imports
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg

if args_cli.task == "BipedRobot":
    BipedEnvCfg = BipedRobotEnvCfg
else:
    BipedEnvCfg = BipedRobotV2EnvCfg


def _get_joint_limits(env):
    joint_mins = env.joint_limits_min.detach().cpu().tolist()
    joint_maxs = env.joint_limits_max.detach().cpu().tolist()
    return joint_mins, joint_maxs


def _create_ui(joint_names, joint_mins, joint_maxs, default_pos):
    root = tk.Tk()
    root.title("Debug Joint Control")
    root.geometry("600x800")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    sliders_frame = ttk.LabelFrame(main_frame, text="Joint Control (Degrees)", padding="5")
    sliders_frame.pack(fill=tk.X, pady=5)

    slider_vars = []

    def make_slider(idx, name, min_deg, max_deg, default_deg):
        lbl = ttk.Label(sliders_frame, text=name, width=18)
        lbl.grid(row=idx, column=0, padx=2, pady=2, sticky="w")

        var = tk.DoubleVar(value=default_deg)
        s = ttk.Scale(sliders_frame, from_=min_deg, to=max_deg, variable=var, orient="horizontal", length=240)
        s.grid(row=idx, column=1, padx=2, pady=2)

        val_lbl = ttk.Label(sliders_frame, text=f"{default_deg:.1f}", width=6)
        val_lbl.grid(row=idx, column=2, padx=2, pady=2)

        def on_change(*_):
            val_lbl.configure(text=f"{var.get():.1f}")

        var.trace_add("write", on_change)
        return var

    for i, (name, min_rad, max_rad, default_rad) in enumerate(zip(joint_names, joint_mins, joint_maxs, default_pos)):
        min_deg = math.degrees(min_rad)
        max_deg = math.degrees(max_rad)
        default_deg = math.degrees(default_rad)
        slider_vars.append(make_slider(i, name, min_deg, max_deg, default_deg))

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=5)

    def set_zero():
        for var in slider_vars:
            var.set(0.0)

    ttk.Button(btn_frame, text="Reset to Zero", command=set_zero).pack(side=tk.LEFT, padx=5)

    height_var = tk.StringVar(value="Root height: 0.0000")
    height_lbl = ttk.Label(main_frame, textvariable=height_var)
    height_lbl.pack(fill=tk.X, pady=4)

    ui_state = {"open": True, "height_var": height_var}

    def on_close():
        ui_state["open"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    return root, slider_vars, ui_state


def _actions_from_gui(slider_vars, env, task_name):
    num_envs = env.num_envs
    num_actions = env.num_actions
    actions = torch.zeros(num_envs, num_actions, device=env.device, dtype=torch.float32)

    target_deg = [var.get() for var in slider_vars]
    target_rad = torch.tensor([math.radians(val) for val in target_deg], device=env.device, dtype=torch.float32)

    joint_mins, joint_maxs = _get_joint_limits(env)
    mins = torch.tensor(joint_mins, device=env.device, dtype=torch.float32)
    maxs = torch.tensor(joint_maxs, device=env.device, dtype=torch.float32)
    base_action = 2.0 * (target_rad - mins) / (maxs - mins) - 1.0

    base_action = torch.clamp(base_action, -1.0, 1.0)
    actions[:] = base_action
    return actions


def main():
    env_cfg = BipedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = BipedEnv(cfg=env_cfg)
    env.reset()

    joint_names = list(env.robot.joint_names)
    joint_mins, joint_maxs = _get_joint_limits(env)
    default_pos = env.default_joint_pos.detach().cpu().tolist()

    root, slider_vars, ui_state = _create_ui(joint_names, joint_mins, joint_maxs, default_pos)

    obs, _ = env.reset()

    while simulation_app.is_running():
        try:
            root.update()
        except Exception:
            break

        if not ui_state.get("open", True):
            break

        if ui_state.get("height_var") is not None:
            root_height = env.robot.data.root_pos_w[0, 2].item()
            ui_state["height_var"].set(f"Root height: {root_height:.4f}")

        actions = _actions_from_gui(slider_vars, env, args_cli.task)
        obs, _, terminated, truncated, _ = env.step(actions)

        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    simulation_app.close()


if __name__ == "__main__":
    main()
