import torch
import torch.nn as nn
from models.transformer import TransformerActor, TransformerCritic
from gymnasium import spaces
import numpy as np

def test_transformer():
    device = torch.device("cpu")
    history_size = 5
    obs_dim = 10
    action_dim = 2
    
    # Define spaces
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim * history_size,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    
    # Model parameters
    d_model = 64
    nhead = 4
    num_layers = 1
    dim_feedforward = 128
    
    print(f"Testing TransformerActor...")
    actor = TransformerActor(
        observation_space, action_space, d_model, nhead, num_layers, dim_feedforward, 
        observation_normalizer=None, head_type="gaussian", history_size=history_size, device=device
    )
    
    # Test forward pass
    batch_size = 8
    obs = torch.randn(batch_size, obs_dim * history_size)
    dist = actor(obs)
    action = dist.sample()
    print(f"Actor output shape: {action.shape}")
    assert action.shape == (batch_size, action_dim)
    
    print(f"Testing TransformerCritic...")
    critic = TransformerCritic(
        observation_space, action_space, d_model, nhead, num_layers, dim_feedforward,
        observation_normalizer=None, history_size=history_size, critic_type="deterministic", device=device
    )
    
    actions = torch.randn(batch_size, action_dim)
    value = critic(obs, actions)
    print(f"Critic output shape: {value.shape}")
    assert value.shape == (batch_size,)
    
    print("Transformer test passed!")

if __name__ == "__main__":
    test_transformer()
