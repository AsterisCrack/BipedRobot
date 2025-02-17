import numpy as np
import torch
from torch.optim import Adam
from models.mpo import MPO
from models.networks import PolicyNetwork, QNetwork
from envs.basic_env import BasicEnv

def train():
    # Initialize environment
    # env = BasicEnv(render_mode="human")
    env = BasicEnv()
    
    # Initialize networks
    policy_net = PolicyNetwork(env)
    target_policy_net = PolicyNetwork(env)
    q_net = QNetwork(env)
    target_q_net = QNetwork(env)

    # Copy weights from policy_net to target_policy_net
    target_policy_net.load_state_dict(policy_net.state_dict())
    target_q_net.load_state_dict(q_net.state_dict())

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
        policy_optimizer=Adam(policy_net.parameters(), lr=1e-3),
        q_optimizer=Adam(q_net.parameters(), lr=1e-4),
        episodes=1000,
        episode_length=200,
        lagrange_it=10,
        mb_size=64,
        rerun_mb=10,
        add_act=10
    )

    # Train MPO
    mpo.train()
    
    # Close environment
    env.close()
    
    
if __name__ == "__main__":
    train()


