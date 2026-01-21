
import os
import sys
import torch
import gymnasium as gym

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Debug Biped Robot in Isaac Lab")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def debug():
    env_cfg = BipedEnvCfg()
    env_cfg.scene.num_envs = 1
    
    env = BipedEnv(cfg=env_cfg)
    print(f"Observation Space Type: {type(env.observation_space)}")
    print(f"Observation Space: {env.observation_space}")
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        print("Keys:", env.observation_space.keys())
        for k, v in env.observation_space.items():
            print(f"Key: {k}, Shape: {v.shape}")
    else:
        if hasattr(env.observation_space, 'shape'):
            print(f"Shape: {env.observation_space.shape}")

    simulation_app.close()

if __name__ == "__main__":
    debug()
