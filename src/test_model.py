from envs.basic_env import BasicEnv
from envs.advanced_env import AdvancedEnv
from models.mpo.model import MPO
from models.d4pg.model import D4PG
import torch

def test_model(model_path, episodes=10, episode_length=int(2e4)):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment with rendering
    env = BasicEnv(render_mode="human", sim_frequency=100)
    # env = AdvancedEnv(render_mode="human")
    # Initialize networks
    model = D4PG(env, model_path=model_path, device=device)

    # Evaluate the model
    mean_reward = 0
    for episode in range(episodes):
        obs = env.reset()
        for step in range(episode_length):
            action = model.step(obs)
            # action = model.actor.get_action(torch.from_numpy(obs).float())
            obs, reward, done, _ = env.step(action)
            env.render()
            mean_reward += reward
            if done:
                break
    print(f"Mean reward over {episodes} episodes: {mean_reward / episodes}")

    # Close environment
    env.close()

if __name__ == "__main__":
    model_path = "checkpoints_optimizers_tests\d4pg_feet_orient_no_height6\step_17000000.pt"
    test_model(model_path)