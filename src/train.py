import numpy as np
from models.mpo.train import MPOTrainer, MPOTrainerLSTM
from models.ddpg.train import DDPGTrainer
from models.sac.train import SACTrainer
from envs.distributed import distribute
from envs.basic_env import BasicEnv
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
    steps=11000
    """trainer = MPOTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        steps=steps,
        seed=seed)
    
    trainer1 = MPOTrainerLSTM(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        num_workers=8,
        device=device,
        steps=steps,
        seed=seed)"""
        
    """trainer = DDPGTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        seed=seed)"""
    
    trainer = SACTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        seed=seed)
    
    # Train agent
    trainer.run()
    
if __name__ == "__main__":
    train()


