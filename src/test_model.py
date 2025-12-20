from envs.mujoco_env import MujocoEnv
from algorithms.mpo.model import MPO
from algorithms.d4pg.model import D4PG
from algorithms.sac.model import SAC
from algorithms.ddpg.model import DDPG
import torch

def test_model(model_path, episodes=10, episode_length=1000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment with rendering
    env = MujocoEnv(render_mode="human", sim_frequency=100)
    
    # Initialize networks
    model = SAC(env, model_path=model_path, device=device)

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
    # model_path = "checkpoints_optimizers_tests\d4pg_com_center_contact_time_invariant_no_feet_orient1\step_6900000.pt"
    model_path = "checkpoints_final/sac/step_30000000.pt"
    test_model(model_path)