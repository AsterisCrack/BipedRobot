import os
import torch
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from algorithms.utils import RunningMeanStd
import envs.isaaclab

class BaseIsaacLabWrapper:
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
                    
        # Normalize if enabled
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
            # Handle if path doesn't have .pt extension in the argument
            load_path = path
            if path.endswith(".pt"):
                base_path = path[:-3]
                load_path = base_path + "_obs_scalers.pt"
            else:
                load_path = path + "_obs_scalers.pt"
            
            if os.path.exists(load_path):
                scalers_state = torch.load(load_path, map_location=self.device)
                for k, v in scalers_state.items():
                    if k in self.obs_scalers:
                        self.obs_scalers[k].load_state_dict(v)
                print(f"Loaded observation scalers from {load_path}")
            else:
                print(f"Warning: Observation scalers not found at {load_path}")

    def step(self, actions):
        # Actions are numpy from the agent, convert to torch for Isaac Lab
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
            
        obs, rew, terminated, truncated, info = self.env.step(actions)
        
        resets = terminated | truncated
        
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

def make_env(args_cli, config):
    # Map legacy task names to registered IDs
    task_name = args_cli.task
    if task_name == "Biped":
        task_name = "Isaac-Biped-Direct-v0"
    elif task_name == "BipedV2":
        task_name = "Isaac-BipedV2-Direct-v0"

    # Determine device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load env config via Isaac Lab utils
    env_cfg = parse_env_cfg(
        task_name,
        device=device_str,
        num_envs=args_cli.num_envs,
    )
    
    # Override config values if needed based on train_config
    if hasattr(config.train, "history_size"):
        env_cfg.history_size = config.train.history_size
        
    if hasattr(config.train, "actor_obs"):
        env_cfg.observation_type = config.train.actor_obs
        
    if hasattr(config.train, "critic_obs"):
        # If critic_obs is privileged, we enable privileged info
        # Otherwise (normal or basic), we disable it so it uses the same obs as actor
        env_cfg.critic_has_privileged_info = (config.train.critic_obs == "privileged")
    
    # Override num_envs if provided in CLI explicitly (already passed to parse_env_cfg but double check cfg update)
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
        if hasattr(env_conf, "commands") and env_conf.commands:
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
                    if "params" in val:
                        env_cfg.events[key].params.update(val["params"])
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
    if args_cli.seed is not None:
        seed = args_cli.seed
    else:
        seed = config.train.seed
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Pass seed to config for environment
    env_cfg.seed = seed

    # Create environment
    env = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video or args_cli.headless is False else None # Assuming headless logic handled outside or defaulting to None
        # In train.py: render_mode="rgb_array" if args_cli.video or args_cli.headless else None
        # But wait, headless usually means NO render.
        # train.py code: render_mode="rgb_array" if args_cli.video or args_cli.headless else None
        # This looks like a bug or specific intent in train.py? Usually headless=True -> no render.
        # Unless headless implies offscreen rendering?
        # Let's copy the logic from train.py exactly for now to be safe.
    )
    # Re-reading logic from train.py:
    # env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.headless else None)
    
    return env, env_cfg
