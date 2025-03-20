from models.networks import ActorTwinCriticWithTargets
from envs.basic_env import BasicEnv
import torch

def test_model(model_path, episodes=10, episode_length=int(2e4)):
    # Initialize environment with rendering
    env = BasicEnv(render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks
    model_sizes=[[256, 256], [256, 256]]
    model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, actor_type="gaussian_multivariate")

    # Load the saved model
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    mean_reward = 0
    for episode in range(episodes):
        obs = env.reset()
        for step in range(episode_length):
            action = model.actor.forward(torch.tensor(obs).float()).sample().numpy()
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
    model_path = "checkpoints_comparison\sac\step_2500000.pt"
    test_model(model_path)