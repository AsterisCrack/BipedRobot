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

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Test Biped Robot Symmetry")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Environment imports
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg
from envs.isaaclab.mdp.symmetry import compute_symmetric_states, _switch_biped_joints_left_right, _transform_actions_left_right
from envs.assets.robot.biped_robot import JOINT_LIMITS
import isaaclab.utils.math as math_utils

# Constants
JOINT_NAMES = [
    "r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x",
    "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x"
]

class SymmetryTestEnv(BipedEnv):
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def step(self, actions):
        # 1. Symmetrize Actions
        # Force Robot 1 to take the symmetric actions of Robot 0
        with torch.inference_mode():
            actions_sym = _transform_actions_left_right(actions[0:1])
            actions[1] = actions_sym[0]
            
            # 2. Step Physics
            obs, rew, term, trunc, info = super().step(actions)
            
            # 3. Hard-Copy State for Visual Verification
            # This ensures Robot 1 is EXACTLY the symmetry of Robot 0, correcting any physics drift.
            
            # Read Robot 0 State
            root_pos_0 = self.robot.data.root_pos_w[0].clone()
            root_quat_0 = self.robot.data.root_quat_w[0].clone()
            lin_vel_0 = self.robot.data.root_lin_vel_w[0].clone()
            ang_vel_0 = self.robot.data.root_ang_vel_w[0].clone()
            joint_pos_0 = self.robot.data.joint_pos[0].clone()
            joint_vel_0 = self.robot.data.joint_vel[0].clone()

            # Calculate Robot 1 State (Symmetric)
            
            # Root Position: Shift Y by 1.5m so they are side-by-side
            # If we want pure symmetry, we might want (x, -y, z), but that overlaps if y=0.
            # Here we just visualize the pose symmetry adjacent to the original.
            root_pos_1 = root_pos_0.clone()
            root_pos_1[1] += 1.5 
            
            # Root Orientation: Roll -> -Roll, Pitch -> Pitch, Yaw -> -Yaw
            # Convert Quat to Euler
            r, p, y = math_utils.euler_xyz_from_quat(root_quat_0.unsqueeze(0))
            # Apply Symmetry
            root_quat_1 = math_utils.quat_from_euler_xyz(-r, p, -y).squeeze(0)

            # Velocities (Linear: x -> x, y -> -y, z -> z) ?? 
            # Needs to match the body frame inversion.
            # Use utility if available or just approximate for visual static check.
            # Local velocities are handled by symmetry.py, but here we deal with World frame root.
            # Let's simple-copy velocities for now or zero them if testing static poses.
            lin_vel_1 = lin_vel_0.clone() # Approximation
            ang_vel_1 = ang_vel_0.clone()

            # Joints: Use the symmetry function
            joint_pos_1 = _switch_biped_joints_left_right(joint_pos_0.unsqueeze(0)).squeeze(0)
            joint_vel_1 = _switch_biped_joints_left_right(joint_vel_0.unsqueeze(0)).squeeze(0)
            
            # Apply to Robot 1
            env_ids = torch.tensor([1], device=self.device)
            self.robot.write_root_pose_to_sim(
                torch.cat([root_pos_1.unsqueeze(0), root_quat_1.unsqueeze(0)], dim=-1),
                env_ids=env_ids
            )
            self.robot.write_root_velocity_to_sim(
                torch.cat([lin_vel_1.unsqueeze(0), ang_vel_1.unsqueeze(0)], dim=-1),
                env_ids=env_ids
            )
            self.robot.write_joint_state_to_sim(
                joint_pos_1.unsqueeze(0),
                joint_vel_1.unsqueeze(0),
                env_ids=env_ids
            )
            
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

    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Sliders Frame
    sliders_frame = ttk.LabelFrame(main_frame, text="Joint Control (Degrees)", padding="5")
    sliders_frame.pack(fill=tk.X, pady=5)

    slider_vars = []
    
    def make_slider(idx, name):
        row = idx
        lbl = ttk.Label(sliders_frame, text=name, width=15)
        lbl.grid(row=row, column=0, padx=2, pady=2)
        
        var = tk.DoubleVar(value=0.0)
        s = ttk.Scale(sliders_frame, from_=-90, to=90, variable=var, orient="horizontal", length=200)
        s.grid(row=row, column=1, padx=2, pady=2)
        
        val_lbl = ttk.Label(sliders_frame, text="0.0", width=6)
        val_lbl.grid(row=row, column=2, padx=2, pady=2)
        
        # Link label to var
        def on_change(*args):
            val_lbl.configure(text=f"{var.get():.1f}")
            
        var.trace_add("write", on_change)
        return var

    for i, name in enumerate(JOINT_NAMES):
        slider_vars.append(make_slider(i, name))

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

    # Helper to map degrees to actions
    def get_actions_from_gui():
        # Only set actions for Robot 0, others will be overwritten in step()
        actions = torch.zeros(env.num_envs, 12, device=env.device)
        
        # Determine strict range for this robot
        # We need to reuse the environment's scaling logic approximately
        # But here we just use the raw joint limits logic
        
        for i in range(12):
            deg = slider_vars[i].get()
            rad = np.deg2rad(deg)
            
            low = JOINT_LIMITS[i][0]
            high = JOINT_LIMITS[i][1]
            rng = high - low
            if rng < 1e-6: rng = 1.0
            
            act = 2.0 * (rad - low) / rng - 1.0
            act = np.clip(act, -1.0, 1.0)
            
            # Apply to ALL envs initially, but step() will fix Env 1
            actions[:, i] = float(act)
            
        return actions

    # Main Loop
    while simulation_app.is_running():
        # Handle UI
        root.update()
        
        # Step Env
        actions = get_actions_from_gui()
        obs, _, _, _, _ = env.step(actions)
        
        # Compute Symmetry (Just for text display validation still)
        obs_aug, act_aug = compute_symmetric_states(env, obs=obs, actions=actions)
        
        # Display Text
        if obs_aug is not None and "policy" in obs_aug:
            # Note: obs_aug is [2*N, ...]. 
            # obs_aug[0] is Env 0 Original
            # obs_aug[N] is Env 0 Fliped
            
            proprio_dim = 48
            
            # We want to compare:
            # Robot 0's actual state (policy[0])
            # Robot 1's actual state (policy[1])
            # They SHOULD be symmetric.
            
            pol_0 = obs["policy"][0, :proprio_dim].cpu()
            pol_1 = obs["policy"][1, :proprio_dim].cpu()
            
            msg = "--- Live Validation (Robot 0 vs Robot 1) ---\n"
            msg += f"Robot 1 is explicitly forced to be Symmetry(Robot 0)\n"
            msg += f"{'Feature':<20} | {'Robot 0':<30} | {'Robot 1':<30}\n"
            msg += "-" * 85 + "\n"
            
            # 0-2 Lin Vel
            msg += f"{'Base Lin Vel':<20} | {str(pol_0[0:3].numpy()):<30} | {str(pol_1[0:3].numpy()):<30}\n"
            # 3-5 Ang Vel
            msg += f"{'Base Ang Vel':<20} | {str(pol_0[3:6].numpy()):<30} | {str(pol_1[3:6].numpy()):<30}\n"
            
            msg += "\n--- Joint Positions (Normalized/Relative) ---\n"
            for j in range(12):
                name = JOINT_NAMES[j]
                
                # Check mapping for display:
                # If Robot 0 is r_hip_z (0), Robot 1's corresponding slot (0) holds l_hip_z data (flipped)
                # But physically, Robot 1's joint 0 IS r_hip_z.
                # So if Robot 0 moves r_hip_z to +0.5
                # Robot 1 should move l_hip_z (6) to +0.5? Or -0.5?
                # The symmetry logic swaps LEFT and RIGHT.
                # So Robot 1's Left Leg = Robot 0's Right Leg (flipped)
                # And Robot 1's Right Leg = Robot 0's Left Leg (flipped)
                
                # Let's just print simple values
                val_0 = pol_0[12+j].item()
                val_1 = pol_1[12+j].item()
                
                msg += f"{name:<15} | {val_0:>10.4f} | {val_1:>10.4f}\n"
                
            text_info.delete("1.0", tk.END)
            text_info.insert("1.0", msg)

    simulation_app.close()

if __name__ == "__main__":
    main()
