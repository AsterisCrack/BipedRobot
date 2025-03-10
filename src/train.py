import numpy as np
from models.mpo.train import MPOTrainer, MPOTrainerLSTM
from models.ddpg.train import DDPGTrainer
from models.sac.train import SACTrainer
from models.d4pg.train import D4PGTrainer
from envs.distributed import distribute
from envs.basic_env import BasicEnv
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set random seed for reproducibility
    seed = 42
    steps = 10200
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    # env = BasicEnv(render_mode="human")
    env = distribute(BasicEnv, 8)
    log_dir = "runs_comparison"
    checkpoint_path = "checkpoints_comparison/"
    trainer_mpo = MPOTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        log_dir=log_dir,
        log_name="mpo",
        steps=steps,
        checkpoint_path=checkpoint_path+"mpo",
        seed=seed)
    
    trainer_lstm = MPOTrainerLSTM(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        num_workers=8,
        device=device,
        log_dir=log_dir,
        log_name="mpo_lstm",
        steps=steps,
        checkpoint_path=checkpoint_path+"mpo_lstm",
        seed=seed)
        
    trainer_ddpg = DDPGTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        log_dir=log_dir,
        log_name="ddpg",
        steps=steps,
        checkpoint_path=checkpoint_path+"ddpg",
        seed=seed)
    
    trainer_sac = SACTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        log_dir=log_dir,
        log_name="sac",
        steps=steps,
        checkpoint_path=checkpoint_path+"sac",
        seed=seed)
    
    trainer_d4pg = D4PGTrainer(
        env=env,
        model_sizes=[[256, 256], [256, 256]],
        device=device,
        log_dir=log_dir,
        log_name="d4pg",
        steps=steps,
        checkpoint_path=checkpoint_path+"d4pg",
        seed=seed)
    
    # Train agent
    # Run in order SAC, D4PG, MPO, DDPG, MPO-LSTM
    trainer_sac.run()
    trainer_d4pg.run()
    trainer_mpo.run()
    trainer_ddpg.run()
    trainer_lstm.run()
    
    
if __name__ == "__main__":
    train()


