from envs.basic_env import BasicEnv
from models.mpo.model import MPOModel
import torch

def test_model(model_path, episodes=10, episode_length=int(2e4)):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment with rendering
    env = BasicEnv(render_mode="human")
    
    # Initialize networks
    model = MPOModel(model_path, env, device=device)

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
    #model_path = "models/mpo/checkpoints/Working/step_3665000.pt"
    model_path = "models/mpo/checkpoints/final_model_2/step_7540000.pt"
    test_model(model_path)