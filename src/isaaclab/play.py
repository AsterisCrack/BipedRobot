import os
import sys
import torch
import numpy as np
import argparse
import time

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play Biped Robot in Isaac Lab")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BipedRobot", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--config_path", type=str, default="config/train_config.yaml", help="Path to training config")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from algorithms.sac.model import SAC
from algorithms.ddpg.model import DDPG
from algorithms.d4pg.model import D4PG
from algorithms.mpo.model import MPO
from algorithms.ppo.model import PPO
from config.schema import ModelType
from utils import Config
from src.isaaclab.common import BaseIsaacLabWrapper
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg

if args_cli.task == "BipedRobot":
    BipedEnvCfg = BipedRobotEnvCfg
elif args_cli.task == "BipedRobotV2":
    BipedEnvCfg = BipedRobotV2EnvCfg
else:
    raise ValueError(f"Unknown task: {args_cli.task}")

class IsaacLabWrapper(BaseIsaacLabWrapper):
    """Inference wrapper - step() returns a 5-tuple matching the Gymnasium interface."""

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        obs, rew, terminated, truncated, info = self.env.step(actions)
        return self._process_obs(obs), rew, terminated, truncated, info

def play():
    # Load config
    if not os.path.exists(args_cli.config_path):
        rel_path = os.path.join(os.path.dirname(__file__), "../../../", args_cli.config_path)
        if os.path.exists(rel_path):
            args_cli.config_path = rel_path
        else:
            print(f"Config file not found: {args_cli.config_path}")
            return

    config = Config(args_cli.config_path)
    
    # Initialize Isaac Lab Environment
    env_cfg = BipedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    
    if hasattr(config.train, "history_size"):
        env_cfg.history_size = config.train.history_size
    if hasattr(config.train, "use_history"):
        env_cfg.use_history = config.train.use_history
        
    # Apply env_config from yaml
    if hasattr(config.train, "env_config"):
        env_conf = config.train.env_config
        
        # Update commands
        if hasattr(env_conf, "commands") and env_conf.commands:
            # Handle Pydantic model or dict
            cmds = env_conf.commands
            if hasattr(cmds, "model_dump"):
                cmds = cmds.model_dump()
            
            if "base_velocity" in cmds:
                if "ranges" in cmds["base_velocity"]:
                    ranges = cmds["base_velocity"]["ranges"]
                    # Update ranges in env_cfg
                    # env_cfg.commands is a dict
                    if "lin_vel_x" in ranges:
                        env_cfg.commands["base_velocity"]["ranges"]["lin_vel_x"] = tuple(ranges["lin_vel_x"])
                    if "lin_vel_y" in ranges:
                        env_cfg.commands["base_velocity"]["ranges"]["lin_vel_y"] = tuple(ranges["lin_vel_y"])
                    if "ang_vel_z" in ranges:
                        env_cfg.commands["base_velocity"]["ranges"]["ang_vel_z"] = tuple(ranges["ang_vel_z"])
                        
        # Update rewards weights if needed (for logging/debugging)
        if hasattr(env_conf, "reward_weights") and env_conf.reward_weights:
            weights = env_conf.reward_weights
            if hasattr(weights, "model_dump"):
                weights = weights.model_dump()
            env_cfg.rewards.update(weights)
        if hasattr(config.train, "actor_obs"):
            env_cfg.observation_type = config.train.actor_obs
        
        if hasattr(config.train, "critic_obs"):
            # If critic_obs is privileged, we enable privileged info
            # Otherwise (normal or basic), we disable it so it uses the same obs as actor
            env_cfg.critic_has_privileged_info = (config.train.critic_obs == "privileged")
            
    env = BipedEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.headless else None)
    wrapped_env = IsaacLabWrapper(env, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    use_history = config.train.use_history
    history_size = config.train.history_size
    
    # Load model
    checkpoint_path = args_cli.checkpoint
    if not checkpoint_path:
        # Search for latest checkpoint
        # Base checkpoints dir relative to this file
        checkpoints_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
        
        if os.path.exists(checkpoints_root):
             # Find all subdirectories
             dirs = [os.path.join(checkpoints_root, d) for d in os.listdir(checkpoints_root) if os.path.isdir(os.path.join(checkpoints_root, d))]
             if dirs:
                 # Sort by modification time (latest first)
                 dirs.sort(key=os.path.getmtime, reverse=True)
                 latest_dir = dirs[0]
                 
                 print(f"Searching for checkpoints in: {latest_dir}")
                 
                 # Find pt files
                 files = [f for f in os.listdir(latest_dir) if f.endswith(".pt")]
                 if files:
                     # Sort files to find latest step (assuming name step_X.pt)
                     def get_step_num(filename):
                         try:
                             parts = filename.split("_")
                             if len(parts) >= 2:
                                 num = parts[1].replace(".pt","")
                                 return int(num)
                             return 0
                         except:
                             return 0

                     files.sort(key=get_step_num, reverse=True)
                     latest_file = files[0]
                     
                     checkpoint_path = os.path.join(latest_dir, latest_file)
                     print(f"Auto-selected latest checkpoint: {checkpoint_path}")
                 else:
                     print(f"No .pt files found in {latest_dir}")
             else:
                 print(f"No checkpoint directories found in {checkpoints_root}")
    
    if not checkpoint_path:
        print("No checkpoint provided. Please provide a checkpoint path using --checkpoint")
        return

    # Select model class based on config
    model_type = config.train.model
    if model_type == ModelType.SAC:
        ModelClass = SAC
    elif model_type == ModelType.DDPG:
        ModelClass = DDPG
    elif model_type == ModelType.D4PG:
        ModelClass = D4PG
    elif model_type == ModelType.MPO:
        ModelClass = MPO
    elif model_type == ModelType.PPO:
        ModelClass = PPO
    else:
        # Fallback
        if str(model_type).lower() == "sac":
            ModelClass = SAC
        elif str(model_type).lower() == "ddpg":
            ModelClass = DDPG
        elif str(model_type).lower() == "d4pg":
            ModelClass = D4PG
        elif str(model_type).lower() == "mpo":
            ModelClass = MPO
        elif str(model_type).lower() == "ppo":
            ModelClass = PPO
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model = ModelClass(
        env=wrapped_env,
        model_path=checkpoint_path,
        device=device,
        config=config,
        use_history=use_history,
        history_size=history_size
    )
    
    # Play loop
    obs = wrapped_env.start()
    
    print("Starting playback...")
    while simulation_app.is_running():
        # Get action from policy
        # SAC.step() returns action
        action = model.step(obs)
        
        # Step environment
        obs, rew, terminated, truncated, info = wrapped_env.step(action)
        
        # Reset if needed
        if terminated.any() or truncated.any():
            obs = wrapped_env.start()
            
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    play()
