import argparse
import numpy as np
import torch
from utils import Config
from algorithms.mpo.model import MPO
from config.schema import NetworkConfig, NetworkType
from envs.distributed import distribute
from src.mujoco.train import EnvBuilder


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = config.train.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    worker_groups = config.train.worker_groups
    workers_per_group = config.train.workers_per_group
    max_episode_steps = config.train.max_episode_steps

    env_builder = EnvBuilder(config=config, seed=seed)
    env = distribute(env_builder, worker_groups, workers_per_group, max_episode_steps=max_episode_steps)

    # Override network configs to use LSTM backbone
    lstm_config = NetworkConfig(network_type=NetworkType.LSTM, hidden_size=256, num_layers=2)
    config.model.actor_config = lstm_config
    config.model.critic_config = lstm_config

    seq_length = config.train.history_size or 2

    model = MPO(
        env=env,
        device=device,
        config=config,
        use_history=True,
        history_size=seq_length,
    )

    model.train(
        log_dir=config.train.log_dir,
        log_name=config.train.model_name,
        steps=config.train.steps,
        checkpoint_path=config.train.checkpoint_path + config.train.model_name,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/final/train_config_mpo.yaml",
                        help="Path to the config file")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
