from envs.basic_env import BasicEnv
from envs.advanced_env import AdvancedEnv
from models.mpo.model import MPO
from models.d4pg.model import D4PG
from models.sac.model import SAC
import torch

def test_model(model_path, episodes=10, episode_length=1000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment with rendering
    randomization = {
        "randomize_dynamics": False,
        "randomize_sensors": False,
        "randomize_perturbations": False,
        "friction": [0.5, 1.5],
        "joint_damping": [0.5, 1,5],
        "mass": [0.5, 1.5],
        "inertia": [0.7, 1.3],
        "imu_noise": 0.01,
        "vel_noise": 0.02,
        "t_perturbation": [0.1, 3],
        "force": [-1, 1]
    }
    
    env = BasicEnv(render_mode="human", sim_frequency=100, randomize_dynamics=randomization["randomize_dynamics"], randomize_sensors=randomization["randomize_sensors"], randomize_perturbations=randomization["randomize_perturbations"], random_config=randomization)
    
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
    # model_path = "checkpoints_optimizers_tests\d4pg_com_center_contact_time_invariant_no_feet_orient1\step_6900000.pt"
    model_path = "checkpoints_linux/4/step_30000000.pt"
    test_model(model_path)