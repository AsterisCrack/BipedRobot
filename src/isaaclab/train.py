
import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

# Add the project root to sys.path to allow imports from BipedRobot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Biped Robot in Isaac Lab")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Biped", help="Name of the task.")
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
from algorithms.fast_sac.model import FastSAC
from algorithms.ddpg.model import DDPG
from algorithms.d4pg.model import D4PG
from algorithms.mpo.model import MPO
from algorithms.ppo.model import PPO
from config.schema import ModelType
from utils import Config
from algorithms.utils import RunningMeanStd
if args_cli.task == "BipedV2":
    from envs.isaaclab.biped_env_cfg_v2 import BipedEnvCfg
    from envs.isaaclab.biped_env_v2 import BipedEnv
elif args_cli.task == "Biped":
    from envs.isaaclab.biped_env_cfg import BipedEnvCfg
    from envs.isaaclab.biped_env import BipedEnv
else:
    raise ValueError(f"Unknown task: {args_cli.task}")

class IsaacLabWrapper:
    def __init__(self, env, config=None):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.config = config
        self.normalize_obs = False
        
        if config and hasattr(config.train, "normalize_obs"):
             self.normalize_obs = config.train.normalize_obs

        # Handle batched spaces from Isaac Lab
        # We need to expose single-env spaces to the algorithms
        import gymnasium.spaces as spaces
        
        # Observation space
        if hasattr(env, "cfg") and hasattr(env.cfg, "observation_space_dim") and len(env.cfg.observation_space_dim) > 1:
             # Asymmetric observations
             self.observation_space = spaces.Dict({
                 "actor": spaces.Box(-np.inf, np.inf, shape=(env.cfg.observation_space_dim["policy"],), dtype=np.float32),
                 "critic": spaces.Box(-np.inf, np.inf, shape=(env.cfg.observation_space_dim["critic"],), dtype=np.float32)
             })
        elif hasattr(env.observation_space, "shape") and len(env.observation_space.shape) == 2:
             self.observation_space = spaces.Box(
                 low=env.observation_space.low[0], 
                 high=env.observation_space.high[0], 
                 shape=(env.observation_space.shape[1],), 
                 dtype=env.observation_space.dtype
             )
        else:
             self.observation_space = env.observation_space
             
        # Initialize scalers if needed
        self.obs_scalers = {}
        if self.normalize_obs:
            if isinstance(self.observation_space, spaces.Dict):
                for key, space in self.observation_space.spaces.items():
                    self.obs_scalers[key] = RunningMeanStd(shape=space.shape, device=self.device)
            else:
                self.obs_scalers["default"] = RunningMeanStd(shape=self.observation_space.shape, device=self.device)

        # Action space
        if hasattr(env.action_space, "shape") and len(env.action_space.shape) == 2:
             self.action_space = spaces.Box(
                 low=env.action_space.low[0], 
                 high=env.action_space.high[0], 
                 shape=(env.action_space.shape[1],), 
                 dtype=env.action_space.dtype
             )
        else:
             self.action_space = env.action_space
        
    def start(self):
        obs, _ = self.env.reset()
        return self._process_obs(obs)
        
    def step(self, actions):
        # Actions are numpy from the agent, convert to torch for Isaac Lab
        # if isinstance(actions, np.ndarray):
        #     actions = torch.from_numpy(actions).to(self.device)
            
        obs, rew, terminated, truncated, info = self.env.step(actions)
        
        resets = terminated | truncated
        
        # Keep as torch tensors for performance
        # rew_np = rew.detach().cpu().numpy()
        # resets_np = resets.detach().cpu().numpy()
        # terminated_np = terminated.detach().cpu().numpy()
        
        processed_obs = self._process_obs(obs)
        
        # Trainer expects infos to contain these keys
        infos = {
            "rewards": rew,
            "resets": resets,
            "terminations": terminated,
            "observations": processed_obs # Next obs
        }
        
        # Pass through logging info if available
        if isinstance(info, dict) and "log" in info:
            infos["log"] = info["log"]
        
        # If dict obs, SAC/DDPG update expects specific keys in kwargs
        if isinstance(processed_obs, dict):
            if "actor" in processed_obs:
                infos["observations_actor"] = processed_obs["actor"]
                infos["next_observations_actor"] = processed_obs["actor"] # It's actually next obs
            if "critic" in processed_obs:
                infos["observations_critic"] = processed_obs["critic"]
                infos["next_observations_critic"] = processed_obs["critic"]
        else:
            infos["next_observations"] = processed_obs
            
        return processed_obs, infos

    def _process_obs(self, obs):
        # Map Isaac Lab keys to what SAC expects
        new_obs = obs
        if isinstance(obs, dict):
            new_obs = {}
            if "policy" in obs:
                new_obs["actor"] = obs["policy"]
            if "critic" in obs:
                new_obs["critic"] = obs["critic"]
            # Pass through other keys if needed
            for k, v in obs.items():
                if k not in ["policy", "critic"]:
                    new_obs[k] = v
                    
        # Normalize if enabled and not evaluating (implied by training loop, but we do it regardless here)
        # Note: In a cleaner impl, we should freeze stats during eval. But wrapped_env doesn't know mode.
        if self.normalize_obs:
            if isinstance(new_obs, dict):
                for k, v in new_obs.items():
                    if k in self.obs_scalers:
                        self.obs_scalers[k].update(v)
                        new_obs[k] = self.obs_scalers[k].normalize(v)
            else:
                 if "default" in self.obs_scalers:
                     self.obs_scalers["default"].update(new_obs)
                     new_obs = self.obs_scalers["default"].normalize(new_obs)
            
        return new_obs

    def save(self, path):
        if self.normalize_obs and self.obs_scalers:
            save_path = path + "_obs_scalers.pt"
            scalers_state = {k: v.state_dict() for k, v in self.obs_scalers.items()}
            torch.save(scalers_state, save_path)
            print(f"Saved observation scalers to {save_path}")

    def load(self, path):
        if self.normalize_obs and self.obs_scalers:
            load_path = path + "_obs_scalers.pt"
            # Handle if path doesn't have .pt extension in the argument but file has it?
            # Usually path passed here assumes just base path or full path. 
            # In utils.py: self.agent.save(save_path) calls torch.save(..., path + '.pt').
            # So save_path in utils usually doesn't have .pt.
            # But here I am appending _obs_scalers.pt.
            
            # If path ends with .pt, strip it
            if path.endswith(".pt"):
                base_path = path[:-3]
                load_path = base_path + "_obs_scalers.pt"
            
            if os.path.exists(load_path):
                scalers_state = torch.load(load_path, map_location=self.device)
                for k, v in scalers_state.items():
                    if k in self.obs_scalers:
                        self.obs_scalers[k].load_state_dict(v)
                print(f"Loaded observation scalers from {load_path}")
            else:
                print(f"Warning: Observation scalers not found at {load_path}")

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
    
    # Initialize Isaac Lab Environment
    env_cfg = BipedEnvCfg()
    
    # Override config values if needed based on train_config
    if hasattr(config.train, "use_rough_terrain"):
        env_cfg.use_rough_terrain = config.train.use_rough_terrain

    if hasattr(config.train, "history_size"):
        env_cfg.history_size = config.train.history_size
        
    if hasattr(config.train, "actor_obs"):
        env_cfg.observation_type = config.train.actor_obs
        
    if hasattr(config.train, "critic_obs"):
        # If critic_obs is privileged, we enable privileged info
        # Otherwise (normal or basic), we disable it so it uses the same obs as actor
        env_cfg.critic_has_privileged_info = (config.train.critic_obs == "privileged")
        
    # Override num_envs if provided in CLI
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        
    # Apply EnvConfig from yaml
    if config.train.env_config:
        env_conf = config.train.env_config
        
        # Mirroring
        if hasattr(env_cfg, "enable_mirroring"):
            env_cfg.enable_mirroring = env_conf.enable_mirroring
        
        # Rewards
        if env_conf.reward_weights and hasattr(env_cfg, "rewards"):
            env_cfg.rewards.update(env_conf.reward_weights)
            
        # Reward Scale
        if hasattr(env_conf, "reward_scale"):
            env_cfg.reward_scale = env_conf.reward_scale
            
        # Commands (Target velocities)
        # Check if commands are defined in env_config (as dict) or as fields
        if hasattr(env_conf, "commands") and env_conf.commands:
             # If commands is a dict (from config.yaml structure)
             cmds = env_conf.commands
             if hasattr(cmds, "model_dump"):
                 cmds = cmds.model_dump()
             
             if "base_velocity" in cmds and "ranges" in cmds["base_velocity"]:
                 ranges = cmds["base_velocity"]["ranges"]
                 if "lin_vel_x" in ranges:
                     env_cfg.commands["base_velocity"]["ranges"]["lin_vel_x"] = tuple(ranges["lin_vel_x"])
                 if "lin_vel_y" in ranges:
                     env_cfg.commands["base_velocity"]["ranges"]["lin_vel_y"] = tuple(ranges["lin_vel_y"])
                 if "ang_vel_z" in ranges:
                     env_cfg.commands["base_velocity"]["ranges"]["ang_vel_z"] = tuple(ranges["ang_vel_z"])

        # Mirroring Indices
        if hasattr(env_conf, "mirror_joint_indices") and env_conf.mirror_joint_indices:
            env_cfg.mirror_joint_indices = env_conf.mirror_joint_indices
            env_cfg.mirror_action_indices = env_conf.mirror_joint_indices

        # Randomization & Events
        if hasattr(env_conf, "enable_perturbations"):
            env_cfg.enable_perturbations = env_conf.enable_perturbations
        if hasattr(env_conf, "push_interval_s"):
            env_cfg.push_interval_s = env_conf.push_interval_s
        if hasattr(env_conf, "push_vel_range"):
            env_cfg.push_vel_range = env_conf.push_vel_range
        if hasattr(env_conf, "enable_physics_randomization"):
            env_cfg.enable_physics_randomization = env_conf.enable_physics_randomization
            
        # Events
        if hasattr(env_conf, "events") and env_conf.events:
            events_dict = env_conf.events
            if hasattr(events_dict, "model_dump"):
                events_dict = events_dict.model_dump()
                
            for key, val in events_dict.items():
                if key in env_cfg.events:
                    # Update params if they exist
                    if "params" in val:
                        env_cfg.events[key].params.update(val["params"])
                    # Update other fields if needed (e.g. interval_range_s)
                    if "interval_range_s" in val:
                        env_cfg.events[key].interval_range_s = tuple(val["interval_range_s"])

        # Noise
        if hasattr(env_conf, "observation_noise_model") and env_conf.observation_noise_model:
            obs_noise = env_conf.observation_noise_model
            if hasattr(obs_noise, "model_dump"):
                obs_noise = obs_noise.model_dump()
            if env_cfg.observation_noise_model and env_cfg.observation_noise_model.noise_cfg:
                if "mean" in obs_noise:
                    env_cfg.observation_noise_model.noise_cfg.mean = obs_noise["mean"]
                if "std" in obs_noise:
                    env_cfg.observation_noise_model.noise_cfg.std = obs_noise["std"]
                    
        if hasattr(env_conf, "action_noise_model") and env_conf.action_noise_model:
            act_noise = env_conf.action_noise_model
            if hasattr(act_noise, "model_dump"):
                act_noise = act_noise.model_dump()
            if env_cfg.action_noise_model and env_cfg.action_noise_model.noise_cfg:
                if "mean" in act_noise:
                    env_cfg.action_noise_model.noise_cfg.mean = act_noise["mean"]
                if "std" in act_noise:
                    env_cfg.action_noise_model.noise_cfg.std = act_noise["std"]

    # Set seeds
    seed = args_cli.seed if args_cli.seed is not None else config.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Pass seed to config for environment
    env_cfg.seed = seed

    # Create environment
    env = BipedEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.headless else None)
    
    # Wrap environment
    wrapped_env = IsaacLabWrapper(env, config)
    
    # Initialize Model (SAC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    use_history = config.train.use_history
    history_size = config.train.history_size
    
    # Select model class based on config
    model_type = config.train.model
    if model_type == ModelType.SAC:
        ModelClass = SAC
    elif model_type == ModelType.FastSAC:
        ModelClass = FastSAC
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
        elif str(model_type).lower() == "fast_sac" or str(model_type).lower() == "fastsac":
            ModelClass = FastSAC
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
    
    # Setup logging
    log_dir = config.train.log_dir
    model_name = config.train.model_name
    
    # Add datetime to model name and checkpoint path to avoid overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"{model_name}_{timestamp}"
    
    # Update checkpoint path in config to include the model name (with timestamp)
    # This ensures checkpoints are saved in a unique subfolder
    if config.train.checkpoint_path:
        config.train.checkpoint_path = os.path.join(config.train.checkpoint_path, model_name)
    
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
