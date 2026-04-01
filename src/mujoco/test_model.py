import argparse
import torch
from utils import Config
from algorithms.mpo.model import MPO
from algorithms.d4pg.model import D4PG
from algorithms.sac.model import SAC
from algorithms.ddpg.model import DDPG
from envs.mujoco_env import MujocoEnv


_ALGORITHM_MAP = {
    "sac": SAC,
    "ddpg": DDPG,
    "d4pg": D4PG,
    "mpo": MPO,
}


def test_model(config, checkpoint_path, episodes=10, episode_length=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = MujocoEnv(render_mode="human", sim_frequency=config.train.sim_frequency)

    algorithm = config.train.model.lower()
    ModelClass = _ALGORITHM_MAP.get(algorithm)
    if ModelClass is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(_ALGORITHM_MAP)}")

    model = ModelClass(env, model_path=checkpoint_path, device=device, config=config)

    mean_reward = 0.0
    for episode in range(episodes):
        obs = env.reset()
        for step in range(episode_length):
            action = model.step(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            mean_reward += reward
            if done:
                break

    print(f"Mean reward over {episodes} episodes: {mean_reward / episodes:.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/final/train_config_sac.yaml",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_final/sac/step_30000000.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=1000)
    args = parser.parse_args()
    config = Config(args.config)
    test_model(config, args.checkpoint, args.episodes, args.episode_length)
