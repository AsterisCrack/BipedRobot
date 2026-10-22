import os
import torch
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from torch_rl_algorithms.algorithms.utils import RunningMeanStd
import envs.isaaclab

class BaseIsaacLabWrapper:
    def __init__(self, env, config=None, symmetric_obs_stats=False):
        self.env = env
        # gymnasium wrappers (e.g. RecordVideo) don't forward Isaac Lab specific
        # attributes, so read them from the innermost unwrapped env.
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = base_env.device
        self.num_envs = base_env.num_envs
        self.config = config
        self.normalize_obs = False
        self._eval_mode = False   # when True, scaler statistics are frozen (play/eval only)

        # Symmetric stage-1 statistics. The replay's mirror augmentation mirrors the already
        # NORMALISED obs (perm * sign), which equals the normalised mirrored raw obs only if the
        # scaler satisfies mean_i = s_i * mean_p(i) and var_i = var_p(i). Real-only stats do not
        # (e.g. critic foot forces: R/L sigma ratio 1.3-1.6 -> mirrored samples mis-scaled by up
        # to 0.6 sigma). So, when on, each scaler is updated with the real raw batch AND its raw
        # mirror (equal counts), which makes the pair relations exact for a linear perm/sign
        # mirror. Only the statistics change; the observations returned are the same real ones.
        self._mirror_raw_obs = None
        if symmetric_obs_stats:
            mirror_fn = getattr(base_env, "mirror_obs", None)
            if mirror_fn is None:
                print("[WARN] symmetric_obs_stats requested but the unwrapped env has no mirror_obs(); "
                      "stage-1 observation statistics stay real-sample-only (mirror augmentation will "
                      "be slightly mis-scaled).")
            else:
                self._mirror_raw_obs = mirror_fn

        if config and hasattr(config.train, "normalize_obs"):
             self.normalize_obs = config.train.normalize_obs

        # Handle batched spaces from Isaac Lab
        # We need to expose single-env spaces to the algorithms
        import gymnasium.spaces as spaces

        # Observation space
        if hasattr(base_env, "cfg") and hasattr(base_env.cfg, "observation_space_dim") and len(base_env.cfg.observation_space_dim) > 1:
             # Asymmetric observations
             self.observation_space = spaces.Dict({
                 "actor": spaces.Box(-np.inf, np.inf, shape=(base_env.cfg.observation_space_dim["policy"],), dtype=np.float32),
                 "critic": spaces.Box(-np.inf, np.inf, shape=(base_env.cfg.observation_space_dim["critic"],), dtype=np.float32)
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
                # Raw-space mirror, computed BEFORE any key is normalised in place below.
                mirrored = None
                if self._mirror_raw_obs is not None and not self._eval_mode:
                    mirrored = self._mirror_raw_obs(
                        {k: v for k, v in new_obs.items() if k in ("actor", "critic") and k in self.obs_scalers})
                for k, v in new_obs.items():
                    if k in self.obs_scalers:
                        if not self._eval_mode:   # freeze stats during play/eval
                            if mirrored is not None and k in mirrored:
                                self.obs_scalers[k].update(torch.cat([v, mirrored[k]], dim=0))
                            else:
                                self.obs_scalers[k].update(v)
                        new_obs[k] = self.obs_scalers[k].normalize(v)
            else:
                 if "default" in self.obs_scalers:
                     if not self._eval_mode:
                         if self._mirror_raw_obs is not None:
                             m = self._mirror_raw_obs({"actor": new_obs})["actor"]
                             self.obs_scalers["default"].update(torch.cat([new_obs, m], dim=0))
                         else:
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

    def set_eval_mode(self) -> None:
        """Freeze obs-scaler statistics — call after load() in play/eval to prevent drift."""
        self._eval_mode = True

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
        
        # Only the next_* keys are published. There used to be matching observations_actor /
        # observations_critic keys here holding this same post-step tensor, which reads as
        # "the replay stores (s', a, s')" -- an external audit flagged exactly that as a
        # critical bug. It was never live: DDPG.update() takes `observations` positionally and
        # never reads **kwargs, and the real store pairs self.last_observations (pre-step, what
        # the actor saw) with this observation. Removed so nobody trusts them as `s`.
        # PPO reads next_observations_actor for its GAE bootstrap, where post-step is correct.
        if isinstance(processed_obs, dict):
            if "actor" in processed_obs:
                infos["next_observations_actor"] = processed_obs["actor"]
            if "critic" in processed_obs:
                infos["next_observations_critic"] = processed_obs["critic"]
        else:
            infos["next_observations"] = processed_obs
            
        return processed_obs, infos


# make_env() used to live here: an unused third copy of the config.yaml -> BipedEnvCfg
# mapping (no callers anywhere). That mapping now lives in src/isaaclab/env_config.py.
