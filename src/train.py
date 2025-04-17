import numpy as np
from utils import Config
from models.mpo.model import MPO
from models.ddpg.model import DDPG
from models.sac.model import SAC
from models.d4pg.model import D4PG
from envs.distributed import distribute
from envs.basic_env import BasicEnv
from envs.advanced_env import AdvancedEnv
import os
import torch
import argparse

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set random seed for reproducibility
    seed = 42 or config["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    worker_groups = config["train"]["worker_groups"] or 4
    workers_per_group = config["train"]["workers_per_group"] or 8
    env = distribute(BasicEnv, worker_groups, workers_per_group)
    log_dir = config["train"]["log_dir"] or "runs"
    checkpoint_path = config["train"]["checkpoint_path"] or "checkpoints/" 
    model_name = config["train"]["model_name"] or "model"
    
    if not config["train"]["overwrite_model"]:
        i=1
        orig_model_name = model_name
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        while model_name in os.listdir(checkpoint_path):
            model_name = orig_model_name + str(i)
            i += 1
            
    print(f"Model name: {model_name}")
    
    # Initialize model
    model_sizes = config["model"]["model_sizes"] or [[256, 256], [256, 256]]
    model_init = lambda model: model(env=env, device=device, model_sizes=model_sizes, config=config)
    match config["train"]["model"].lower():
        case "mpo":
            model = model_init(MPO)
        case "ddpg":
            model = model_init(DDPG)
        case "sac":
            model = model_init(SAC)
        case "d4pg":
            model = model_init(D4PG)
        case None:
            print("No model specified, using D4PG")
            model = model_init(D4PG)
        case _:
            raise ValueError("Model not recognized. Please use one of the following: mpo, ddpg, sac, d4pg")
    
    steps = config["train"]["steps"] or 2000000
    test_environment = config["train"]["test_environment"] or False
    if test_environment:
        test_environment = AdvancedEnv()

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
    config_file = "config/train_config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_file, help="Path to the config file")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)


