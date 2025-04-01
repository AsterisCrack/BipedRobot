import numpy as np
from models.mpo.model import MPO
from models.ddpg.model import DDPG
from models.sac.model import SAC
from models.d4pg.model import D4PG
from envs.distributed import distribute
from envs.basic_env import BasicEnv
from envs.advanced_env import AdvancedEnv
import os
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
    # env_sequential = distribute(AdvancedEnv, 1, 16)
    env_parallel = distribute(AdvancedEnv, 4, 8)
    log_dir = "runs_reward_tests"
    checkpoint_path = "checkpoints_reward_tests/"
    model_name = "d4pg_advanced"
    i=1
    while model_name in os.listdir(checkpoint_path):
        model_name = "d4pg_advanced" + str(i)
        i += 1
    print(f"Model name: {model_name}")
    d4pg = D4PG(env=env_parallel, device=device)
    
    steps = 20000000
    print("Training ddpg in sequential")
    d4pg.train(
        log_dir=log_dir,
        log_name=model_name,
        steps=steps,
        checkpoint_path=checkpoint_path+model_name,
        seed=seed)
    d4pg.save_trainer_state()
    
    """mpo = MPO(env=env, device=device)
    mpo_lstm = MPO(env=env, lstm=True, device=device)
    ddpg = DDPG(env=env, device=device)
    sac = SAC(env=env, device=device)"""
    """d4pg_seq = D4PG(env=env_sequential, device=device)
    d4pg_par = D4PG(env=env_parallel, device=device)
    
    steps = 1000000
    print("Training ddpg in parallel")
    d4pg_par.train(
        log_dir=log_dir,
        log_name="d4pg_par",
        steps=steps,
        checkpoint_path=checkpoint_path+"d4pg_par",
        seed=seed)
    d4pg_par.save_trainer_state()
    
    print("Training ddpg sequentially")
    d4pg_seq.train(
        log_dir=log_dir,
        log_name="d4pg_seq",
        steps=steps,
        checkpoint_path=checkpoint_path+"d4pg_seq",
        seed=seed)
    d4pg_seq.save_trainer_state()"""
    
    # Train agents
    # Run in order SAC, D4PG, MPO, DDPG, MPO-LSTM
    """
    train = lambda model, name: model.train(
        log_dir=log_dir,
        log_name=name,
        steps=steps,
        checkpoint_path=checkpoint_path+name,
        seed=seed)
    train(sac, "sac")
    sac.save_trainer_state()
    train(d4pg, "d4pg")
    d4pg.save_trainer_state()
    train(mpo, "mpo")
    mpo.save_trainer_state()
    train(ddpg, "ddpg")
    ddpg.save_trainer_state()
    train(mpo_lstm, "mpo_lstm")
    mpo_lstm.save_trainer_state()"""
    
if __name__ == "__main__":
    train()


