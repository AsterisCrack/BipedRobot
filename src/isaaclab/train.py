
import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Biped Robot in Isaac Lab")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BipedRobot", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to training config")

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

# AppLauncher reinitializes sys.path; re-add project paths after it launches.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# torch_rl_algorithms uses bare imports (from models.networks import ...)
# so its own directory must be on sys.path as well.
_trl_root = os.path.join(_project_root, "torch_rl_algorithms")
if _trl_root not in sys.path:
    sys.path.insert(0, _trl_root)

from torch_rl_algorithms.algorithms.sac.model import SAC
from torch_rl_algorithms.algorithms.ddpg.model import DDPG
from torch_rl_algorithms.algorithms.d4pg.model import D4PG
from torch_rl_algorithms.algorithms.mpo.model import MPO
from torch_rl_algorithms.algorithms.ppo.model import PPO
from config.schema import ModelType
from utils import Config
from utils.residual_home import freeze_residual_map
from src.isaaclab.common import BaseIsaacLabWrapper
from src.isaaclab.env_config import build_env_cfg, describe as describe_env_cfg
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg

if args_cli.task == "BipedRobot":
    BipedEnvCfg = BipedRobotEnvCfg
elif args_cli.task == "BipedRobotV2":
    BipedEnvCfg = BipedRobotV2EnvCfg
else:
    raise ValueError(f"Unknown task: {args_cli.task}")

IsaacLabWrapper = BaseIsaacLabWrapper

def train():
    # Resolve Config Path from Checkpoint if applicable
    if args_cli.checkpoint and "--config_path" not in sys.argv:
        # User provided checkpoint but not config. Try to find config in checkpoint dir.
        ckpt_path = os.path.abspath(args_cli.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_path)
        
        # Check in the checkpoint directory
        found_config = None
        # 1. Find first fide with .yaml or .yml extension
        for name in os.listdir(ckpt_dir):
            if name.endswith(".yaml") or name.endswith(".yml"):
                found_config = os.path.join(ckpt_dir, name)
                break
                    
        if found_config:
            print(f"[INFO] Auto-detected config from checkpoint: {found_config}")
            args_cli.config_path = found_config

    # Load config
    # Adjust path if running from different directory
    if not os.path.exists(args_cli.config_path):
        # Try relative to this file
        rel_path = os.path.join(os.path.dirname(__file__), "../../../", args_cli.config_path)
        if os.path.exists(rel_path):
            args_cli.config_path = rel_path
        else:
            print(f"Config file not found: {args_cli.config_path}")
            return

    config = Config(args_cli.config_path)
    # action_home_deg / action_scale_rad / spawn_height_m may say `auto` (derived from the gait table and the robot model):
    # freeze the numbers into this config NOW, before the env is built and before the config is saved to
    # checkpoints/<run>/config.yaml, so a later re-bake cannot change an existing run's map. No-op for explicit numbers.
    freeze_residual_map(config)
    # TF32 matmuls (train.tf32): the learner is ~90 ms/it of fp32 GEMM at batch 2048.
    torch.backends.cuda.matmul.allow_tf32 = bool(config.train.tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.train.tf32)

    # Initialize Isaac Lab Environment. All config.yaml -> BipedEnvCfg mapping lives in
    # src/isaaclab/env_config.py, shared with play.py, so the two cannot drift apart again.
    env_cfg = build_env_cfg(config, BipedEnvCfg, training=True, num_envs=args_cli.num_envs)
    print("[train] env contract:", describe_env_cfg(env_cfg))

    # Set seeds
    seed = args_cli.seed if args_cli.seed is not None else config.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Pass seed to config for environment
    env_cfg.seed = seed

    # Generate timestamp now so it's available for the video directory
    log_dir = config.train.log_dir
    model_name = config.train.model_name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{model_name}_{timestamp}"
    if config.train.checkpoint_path:
        config.train.checkpoint_path = os.path.join(config.train.checkpoint_path, model_name)

    # Inject camera into scene when --video is requested (camera adds GPU overhead, so opt-in only)
    if args_cli.video:
        from isaaclab.sensors import TiledCameraCfg as _TiledCameraCfg
        from isaaclab.sim import PinholeCameraCfg as _PinholeCameraCfg
        env_cfg.enable_video_camera = True
        # Attach a small follow-camera to the robot's base_link for training video.
        # pos: 0.5m in front of the robot, 0.4m above; looking back toward the robot.
        # Adjust pos/rot to change the viewing angle.
        env_cfg.scene.camera = _TiledCameraCfg(
            prim_path="/World/envs/env_0/Robot/Robot/base_link/Camera",
            offset=_TiledCameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 0.4),
                rot=(0.5, -0.5, 0.5, -0.5),  # looking back (-X) and slightly down; ROS convention
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=_PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=0.6,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 5.0),
            ),
            width=128,
            height=72,
            update_period=0.04,  # 25 Hz — reduces rendering overhead during training
        )

    # Create environment
    env = BipedEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap with video recorder before the RL wrapper so frames are captured at env.step() level
    if args_cli.video:
        import gymnasium as gym
        video_dir = os.path.join(log_dir, model_name, "videos")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[Video] Recording to {video_dir} every {args_cli.video_interval} steps, {args_cli.video_length} frames each.")

    # Wrap environment
    # With replay mirror augmentation on, make the stage-1 obs statistics mirror-symmetric (see
    # BaseIsaacLabWrapper). The wrapper reaches BipedEnv.mirror_obs through env.unwrapped, so the
    # --video RecordVideo wrapper is fine. play/export/probe paths never pass this (stats frozen).
    wrapped_env = IsaacLabWrapper(
        env, config,
        symmetric_obs_stats=bool(getattr(config.train, "symmetry_augmentation", False)),
    )

    # Initialize Model (SAC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_history = config.train.use_history
    history_size = config.train.history_size
    
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
        # Fallback for string values if not using Enum directly or if config loaded as dict
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
        device=device,
        config=config,
        use_history=use_history,
        history_size=history_size,
        model_path=args_cli.checkpoint # Resume if checkpoint provided
    )
    
    # Train
    model.train(
        log_dir=log_dir,
        log_name=model_name,
        steps=config.train.steps,
        save_steps=config.train.save_steps
    )
    
    # Close
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    train()
