import torch
from models.mpo import MPO
from models.networks import PolicyNetwork, QNetwork
from envs.basic_env import BasicEnv

def test_model(model_path, episodes=10, episode_length=200):
    # Initialize environment with rendering
    env = BasicEnv(render_mode="human")
    
    # Initialize networks
    policy_net = PolicyNetwork(env)
    target_policy_net = PolicyNetwork(env)
    q_net = QNetwork(env)
    target_q_net = QNetwork(env)

    # Initialize MPO algorithm
    mpo = MPO(
        env=env,
        dual_constraint=0.1,
        mean_constraint=0.1,
        var_constraint=0.1,
        learning_rate=1e-4,
        alpha=0.1,
        q_net=q_net,
        target_q_net=target_q_net,
        policy_net=policy_net,
        target_policy_net=target_policy_net,
        episodes=1000,
        episode_length=200,
        lagrange_it=10,
        mb_size=64,
        rerun_mb=10,
        add_act=10
    )

    # Load the saved model
    mpo.load_model(model_path)

    # Evaluate the model
    mean_reward = mpo.eval(episodes, episode_length, render=True)
    print(f"Mean reward over {episodes} episodes: {mean_reward}")

    # Close environment
    env.close()

if __name__ == "__main__":
    model_path = "mpo_model.pt"
    test_model(model_path)