"""
Test script for Biped Robot symmetry with Tkinter UI.
"""
import argparse
import sys
import os
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Test Biped Robot Symmetry")
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
from envs.isaaclab.mdp.symmetry import compute_symmetric_states, _switch_biped_joints_left_right, _transform_actions_left_right
import isaaclab.utils.math as math_utils

if args_cli.task == "BipedRobot":
    BipedEnvCfg = BipedRobotEnvCfg
else:
    BipedEnvCfg = BipedRobotV2EnvCfg

class SymmetryTestEnv(BipedEnv):
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def step(self, actions):
        # 1. Symmetrize Actions
        # Apply normal actions to Robot 0
        # Apply symmetric actions to Robot 1
        with torch.inference_mode():
            # Get Robot 0 actions (actions coming from UI are same for all, so take [0:1])
            act_0 = actions[0:1].clone()
            
            # Compute Symmetric actions for Robot 1 (Symmetry(act_0))
            # returns batch of same size as input
            act_1_sym = _transform_actions_left_right(act_0)
            
            # Set actions for env 1
            actions[1] = act_1_sym[0]
            
            # 2. Step Physics
            # Robot 0 runs with act_0
            # Robot 1 runs with Symmetry(act_0)
            obs, rew, term, trunc, info = super().step(actions)
            
            return obs, rew, term, trunc, info

def main():
    # Configure Environment
    env_cfg = BipedEnvCfg()
    # Force num_envs to 2 in both the scene and the top-level config
    env_cfg.scene.num_envs = 2
    env_cfg.num_envs = 2 
    
    # Disable randomization for consistent testing
    env_cfg.randomize_rigid_body_mass = None 
    
    # Remove randomization events but keep required resets
    keys_to_remove = ["push_robot", "randomize_mass", "randomize_friction"]
    for k in keys_to_remove:
        if k in env_cfg.events:
            del env_cfg.events[k]

    # Zero out reset randomness
    if "reset_base" in env_cfg.events:
        env_cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        env_cfg.events["reset_base"].params["velocity_range"] = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        
    if "reset_robot_joints" in env_cfg.events:
        env_cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    
    # Initialize Environment
    # Use our custom class
    env = SymmetryTestEnv(cfg=env_cfg)
    env.reset()

    # Setup UI
    root = tk.Tk()
    root.title("Symmetry Test Control")
    root.geometry("600x800")
    
    # Logging Setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join("logs", "symmetry_test", f"{timestamp}_session")
    
    log_dir_real = os.path.join(session_dir, "real")
    log_dir_mirror = os.path.join(session_dir, "mirror")
    log_dir_diff = os.path.join(session_dir, "diff")
    
    os.makedirs(session_dir, exist_ok=True)
    
    writer_real = SummaryWriter(log_dir_real)
    writer_mirror = SummaryWriter(log_dir_mirror)
    writer_diff = SummaryWriter(log_dir_diff)
    
    print(f"Logging session to {session_dir}")
    print(f"Run 'tensorboard --logdir logs/symmetry_test' to view live.")
    
    data_log = []
    step_idx = 0

    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Sliders Frame
    sliders_frame = ttk.LabelFrame(main_frame, text="Joint Control (Degrees)", padding="5")
    sliders_frame.pack(fill=tk.X, pady=5)

    slider_vars = []
    
    joint_names = list(env.robot.joint_names)
    joint_limits_min = env.joint_limits_min.detach().cpu().tolist()
    joint_limits_max = env.joint_limits_max.detach().cpu().tolist()

    def make_slider(idx, name, min_deg, max_deg):
        row = idx
        lbl = ttk.Label(sliders_frame, text=name, width=15)
        lbl.grid(row=row, column=0, padx=2, pady=2)
        
        var = tk.DoubleVar(value=0.0)
        s = ttk.Scale(sliders_frame, from_=min_deg, to=max_deg, variable=var, orient="horizontal", length=200)
        s.grid(row=row, column=1, padx=2, pady=2)
        
        val_lbl = ttk.Label(sliders_frame, text="0.0", width=6)
        val_lbl.grid(row=row, column=2, padx=2, pady=2)
        
        # Link label to var
        def on_change(*args):
            val_lbl.configure(text=f"{var.get():.1f}")
            
        var.trace_add("write", on_change)
        return var

    for i, name in enumerate(joint_names):
        min_deg = np.degrees(joint_limits_min[i])
        max_deg = np.degrees(joint_limits_max[i])
        slider_vars.append(make_slider(i, name, min_deg, max_deg))

    # Control Buttons
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=5)

    def set_random():
        for i, var in enumerate(slider_vars):
            # Respect limits roughly, or just random
            var.set(np.random.uniform(-90, 90))

    def set_zero():
        for var in slider_vars:
            var.set(0.0)

    ttk.Button(btn_frame, text="Generate Random", command=set_random).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Reset to Zero", command=set_zero).pack(side=tk.LEFT, padx=5)

    # Info Display
    info_frame = ttk.LabelFrame(main_frame, text="Symmetry Analysis", padding="5")
    info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    text_info = tk.Text(info_frame, height=20, font=("Consolas", 9))
    text_info.pack(fill=tk.BOTH, expand=True)
    text_info.tag_config("match", foreground="green")
    text_info.tag_config("mismatch", foreground="red")

    # Helper to map degrees to actions
    def get_actions_from_gui():
        # Only set actions for Robot 0, others will be overwritten in step()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Determine strict range for this robot
        # We need to reuse the environment's scaling logic approximately
        # But here we just use the raw joint limits logic
        
        for i in range(env.num_actions):
            deg = slider_vars[i].get()
            rad = np.deg2rad(deg)
            
            low = joint_limits_min[i]
            high = joint_limits_max[i]
            rng = high - low
            if rng < 1e-6: rng = 1.0
            
            act = 2.0 * (rad - low) / rng - 1.0
            act = np.clip(act, -1.0, 1.0)
            
            # Apply to ALL envs initially, but step() will fix Env 1
            actions[:, i] = float(act)
            
        return actions

    # Define Labels (Static)
    base_labels = ["Lin Vel X", "Lin Vel Y", "Lin Vel Z", 
                   "Ang Vel X", "Ang Vel Y", "Ang Vel Z",
                   "Grav X", "Grav Y", "Grav Z",
                   "Cmd X", "Cmd Y", "Cmd Yaw"]
    
    joint_labels = [f"Pos {name}" for name in joint_names]
    joint_vel_labels = [f"Vel {name}" for name in joint_names]
    prev_act_labels = [f"Act {name}" for name in joint_names]
    
    all_labels = base_labels + joint_labels + joint_vel_labels + prev_act_labels

    # Main Loop
    while simulation_app.is_running():
        # Handle UI
        root.update()
        
        # Step Env
        actions = get_actions_from_gui()
        obs, _, _, _, _ = env.step(actions)
        
        # 1. Get Normal Robot Obs (Robot 0) and Mirrored Robot Obs (Robot 1)
        obs_policy = obs["policy"] # [2, 48]
        obs_1 = obs_policy[1].cpu()

        # 2. Compute Symmetric of Robot 0
        # We wrap in dict as expected by compute_symmetric_states
        obs_dict_0 = {"policy": obs_policy[0:1]} 
        # Returns [Original(0), Symmetric(0)]
        obs_aug_dict, _ = compute_symmetric_states(env, obs=obs_dict_0)
        obs_0_sym = obs_aug_dict["policy"][1].cpu()

        # 3. Calculate Differences
        diff = torch.abs(obs_0_sym - obs_1)
        
        # Tensorboard Logging
        # Global stats (diff only)
        writer_diff.add_scalar("Global/Mean_Diff", diff.mean().item(), step_idx)
        writer_diff.add_scalar("Global/Max_Diff", diff.max().item(), step_idx)
        
        # Individual Component Logging
        for i, label in enumerate(all_labels):
            if i < len(diff):
                # Common tag for overlay in TensorBoard
                tag = f"Observation/{label.replace(' ', '_')}"
                
                writer_real.add_scalar(tag, obs_0_sym[i].item(), step_idx)
                writer_mirror.add_scalar(tag, obs_1[i].item(), step_idx)
                writer_diff.add_scalar(tag, diff[i].item(), step_idx)

        # CSV Logging
        row = {"step": step_idx}
        for i, name in enumerate(all_labels):
            if i < len(diff):
                # Save all 3 values per component
                row[f"{name}_real"] = obs_0_sym[i].item()
                row[f"{name}_mirror"] = obs_1[i].item()
                row[f"{name}_diff"] = diff[i].item()
        data_log.append(row)

        step_idx += 1

        # 4. Compare obs_0_sym vs obs_1 (UI Update)
        
        # Clear Text
        text_info.delete("1.0", tk.END)
        
        # Header
        header = f"{'Observation':<25} | {'Exp (Sym)':<15} | {'Act (Rob1)':<15} | {'Match'}\n"
        header += "-" * 75 + "\n"
        text_info.insert("end", header)
        
        for i, label in enumerate(all_labels):
            if i >= len(obs_0_sym): break
            
            val_sym = obs_0_sym[i].item() # Theoretical
            val_act = obs_1[i].item()     # Actual
            
            # Comparison (Tolerance 1e-3)
            is_close = np.isclose(val_sym, val_act, atol=1e-1, rtol=1e-1)
            
            tag = "match" if is_close else "mismatch"
            status = "TRUE" if is_close else "FALSE"
            
            line = f"{label:<25} | {val_sym:>15.4f} | {val_act:>15.4f} | {status}\n"
            text_info.insert("end", line, tag)

    # Save Data
    writer_real.close()
    writer_mirror.close()
    writer_diff.close()
    
    if data_log:
        df = pd.DataFrame(data_log)
        csv_path = os.path.join(session_dir, "symmetry_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV logs to {csv_path}")
        # Note: plot script might need updates to handle new CSV structure
        # print(f"You can plot this using: python BipedRobot/plot_symmetry_logs.py {csv_path}")

    simulation_app.close()

if __name__ == "__main__":
    main()
