import numpy as np
from models.mpo.model import MPO
from models.ddpg.model import DDPG
from models.sac.model import SAC
from models.d4pg.model import D4PG
from envs.distributed import distribute
from envs.basic_env import BasicEnv
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set random seed for reproducibility
    seed = 42
    steps = 7000000
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    # env = BasicEnv(render_mode="human")
    env = distribute(BasicEnv, 8)
    log_dir = "runs_comparison"
    checkpoint_path = "checkpoints_comparison/"
    
    mpo = MPO(env=env, device=device)
    mpo_lstm = MPO(env=env, lstm=True, device=device)
    ddpg = DDPG(env=env, device=device)
    sac = SAC(env=env, device=device)
    d4pg = D4PG(env=env, device=device)
    
    train = lambda model, name: model.train(
        log_dir=log_dir,
        log_name=name,
        steps=steps,
        checkpoint_path=checkpoint_path+name,
        seed=seed)
    # Train agents
    # Run in order SAC, D4PG, MPO, DDPG, MPO-LSTM
    train(sac, "sac")
    sac.save_trainer_state()
    train(d4pg, "d4pg")
    d4pg.save_trainer_state()
    train(mpo, "mpo")
    mpo.save_trainer_state()
    train(ddpg, "ddpg")
    ddpg.save_trainer_state()
    train(mpo_lstm, "mpo_lstm")
    mpo_lstm.save_trainer_state()
    
if __name__ == "__main__":
    train()


