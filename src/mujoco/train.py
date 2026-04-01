import numpy as np
from utils import Config, NoConfig
from algorithms.mpo.model import MPO
from algorithms.ddpg.model import DDPG
from algorithms.sac.model import SAC
from algorithms.d4pg.model import D4PG
from envs.distributed import distribute
from envs.mujoco_env import MujocoEnv
import os
import shutil
import torch
import argparse


class EnvBuilder:
    """Top-level picklable environment builder."""
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def __call__(self):
        train_cfg = self.config.train
        return MujocoEnv(
            history_size=train_cfg.history_size,
            sim_frequency=train_cfg.sim_frequency,
            random_config=self.config.randomization,
            seed=self.seed,
            actor_obs=train_cfg.actor_obs,
            critic_obs=train_cfg.critic_obs,
            env_config=train_cfg.env_config)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = config.train.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    worker_groups = config.train.worker_groups
    workers_per_group = config.train.workers_per_group
    max_episode_steps = config.train.max_episode_steps
    
    env_builder = EnvBuilder(config=config, seed=seed)
    
    env = distribute(env_builder, worker_groups, workers_per_group, max_episode_steps=max_episode_steps)
    log_dir = config.train.log_dir
    checkpoint_path = config.train.checkpoint_path
    model_name = config.train.model_name
    
    print(f"Model name: {model_name}")
    
    # Ensure unique log and checkpoint directory
    if not config.train.overwrite_model:
        i = 1
        original_model_name = model_name
        # Check both checkpoint path and log path to be safe
        while os.path.exists(os.path.join(checkpoint_path, model_name)) or os.path.exists(os.path.join(log_dir, model_name)):
            model_name = f"{original_model_name}_{i}"
            i += 1
        if model_name != original_model_name:
            print(f"Model name adjusted to avoid collision: {model_name}")
    else:
        # If overwrite is true, delete existing folders
        checkpoint_dir = os.path.join(checkpoint_path, model_name)
        log_run_dir = os.path.join(log_dir, model_name)
        if os.path.exists(checkpoint_dir):
            print(f"Overwriting checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
        if os.path.exists(log_run_dir):
            print(f"Overwriting log directory: {log_run_dir}")
            shutil.rmtree(log_run_dir)
    
    # Initialize model
    use_history = config.train.use_history
    history_size = config.train.history_size
    model_init = lambda model: model(env=env, device=device, config=config, use_history=use_history, history_size=history_size)
    checkpoint_path_resume = getattr(config.train, "checkpoint", None)
    if checkpoint_path_resume:
        model_init = lambda model: model(env=env, device=device, config=config,
                                         use_history=use_history, history_size=history_size,
                                         model_path=checkpoint_path_resume)

    match config.train.model.lower():
        case "mpo":
            model = model_init(MPO)
        case "ddpg":
            model = model_init(DDPG)
        case "sac":
            model = model_init(SAC)
        case "d4pg":
            model = model_init(D4PG)
        case _:
            raise ValueError("Model not recognized.")
    
    steps = config.train.steps
    test_environment = None
    if config.train.test_environment:
        test_env_builder = EnvBuilder(config=config, seed=seed + 1000)
        test_environment = distribute(test_env_builder, worker_groups=1, workers_per_group=1, 
                                      max_episode_steps=max_episode_steps)

    model.train(
        log_dir=log_dir,
        log_name=model_name,
        steps=steps,
        checkpoint_path=checkpoint_path+model_name,
        seed=seed,
        test_environment=test_environment,
        config=config
    )
    model.save_trainer_state()
    
    
if __name__ == "__main__":
    # config_file = "config/train_history_config.yaml"
    config_file = "config/final/train_config_d4pg.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_file, help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trainer state directory to resume from")
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint:
        config.train.checkpoint = args.checkpoint
    train(config)
