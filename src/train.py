import numpy as np
from torch.optim import Adam
from models.mpo.mpo import MPO
from models.networks import ActorCriticWithTargets
from envs.distributed import distribute
from envs.basic_env import BasicEnv
from models.utils import Trainer
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    # env = BasicEnv(render_mode="human")
    env = distribute(BasicEnv, 8)
    
    # Initialize networks
    print("Initializing model")
    model = ActorCriticWithTargets(env.observation_space, env.action_space, [256, 256], [256, 256], device=device)
    model.to(device)
    
    # Initialize MPO algorithm
    mpo = MPO(
        action_space=env.action_space,
        model=model,
        device=device,
    )

    trainer = Trainer(mpo, env)
    
    # Train agent
    trainer.run()
    
    
if __name__ == "__main__":
    train()


