import numpy as np
from torch.optim import Adam
from algorithms.mpo.mpo import MPO
from models.networks import ActorCriticWithTargets
from envs.distributed import distribute
from envs.mujoco_env import MujocoEnv
from algorithms.utils import Trainer
from config.schema import Config, NetworkConfig, NetworkType
import torch
from src.train import EnvBuilder
from utils import NoConfig

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = 42
    seq_length = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    # Create a dummy config for LSTM
    env_config = NoConfig()
    env_config.history_size = 0 
    
    env_builder = EnvBuilder(config=NoConfig(), seed=seed)
    env = distribute(env_builder, 8)
    
    # Initialize networks
    print("Initializing model")
    config = NoConfig()
    config.model = NoConfig()
    config.model.actor_config = NetworkConfig(network_type=NetworkType.LSTM, hidden_size=256, num_layers=2)
    config.model.critic_config = NetworkConfig(network_type=NetworkType.LSTM, hidden_size=256, num_layers=2)
    
    model = ActorCriticWithTargets(env.observation_space, env.action_space, config=config, device=device)
    model.to(device)
    
    # Initialize algorithm
    mpo = MPO(
        action_space=env.action_space,
        model=model,
        recurrent_model=True,
        max_seq_length=seq_length,
        num_workers=8,
        device=device,
    )

    trainer = Trainer(mpo, env, steps=11200)
    
    # Train agent
    trainer.run()
    
    
if __name__ == "__main__":
    train()
