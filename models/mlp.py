import torch
import torch.nn as nn
from torch.distributions import Normal
from algorithms.utils import SquashedMultivariateNormalDiag, DistributionalValueHead 
from models.utils.base import Torso

class MLPTorso(Torso):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU, observation_normalizer=None):
        super().__init__(observation_normalizer)
        sizes = [input_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
        self.net = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]
        self.observation_size = input_size # Store for possible pooling

    def forward(self, x):
        x = self.normalize(x)
        return self.net(x)

class MLPActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None, head_type="gaussian"):
        super().__init__()
        self.torso = MLPTorso(observation_space.shape[0], hidden_sizes, observation_normalizer=observation_normalizer)
        self.head_type = head_type
        self.action_size = action_space.shape[0]
        
        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(nn.Linear(self.torso.output_size, self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(nn.Linear(self.torso.output_size, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "gaussian_multivariate":
            self.mean_layer = nn.Sequential(nn.Linear(self.torso.output_size, self.action_size), nn.Identity())
            self.std_layer = nn.Sequential(nn.Linear(self.torso.output_size, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "deterministic":
            self.action_layer = nn.Sequential(nn.Linear(self.torso.output_size, self.action_size), nn.Tanh())

    def forward(self, observations):
        out = self.torso(observations)
        if self.head_type == "gaussian":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return Normal(mean, std)
        elif self.head_type == "gaussian_multivariate":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return SquashedMultivariateNormalDiag(mean, std)
        elif self.head_type == "deterministic":
            return self.action_layer(out)
    
    def get_action(self, observations):
        out = self.forward(observations)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, 'mean'):
             # Return mean for deterministic action, but stay on device
             return out.mean
        # For other distributions, sample or return mean? 
        # Usually get_action implies deterministic/greedy for eval.
        if hasattr(out, 'loc'):
            return out.loc
        return out

class MLPCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None, critic_type="deterministic"):
        super().__init__()
        self.critic_type = critic_type
        
        # Determine input size based on critic type
        if critic_type == "value":
            input_size = observation_space.shape[0]
        else:
            input_size = observation_space.shape[0] + action_space.shape[0]
            
        self.torso = MLPTorso(input_size, hidden_sizes, observation_normalizer=observation_normalizer)
        
        # Legacy temporal pooling components
        self.obs_reducer = nn.Linear(observation_space.shape[0], observation_space.shape[0])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        if critic_type == "deterministic" or critic_type == "value":
            self.value_layer = nn.Linear(self.torso.output_size, 1)
        elif critic_type == "distributional":
            self.value_layer = DistributionalValueHead(-150, 150, 51, self.torso.output_size)

    def forward(self, observations, actions=None):
        if self.torso.observation_normalizer:
            observations = self.torso.observation_normalizer(observations)
        
        if observations.dim() == 3:
            # Sequential observations (legacy pooling logic)
            observations = self.obs_reducer(observations)
            observations = self.pool(observations.transpose(1, 2)).transpose(1, 2)
            observations = observations.reshape(observations.shape[0], -1)

        if self.critic_type == "value":
            out = observations
        else:
            out = torch.cat([observations, actions], dim=-1)
            
        out = self.torso.net(out) 
        
        if self.critic_type == "deterministic" or self.critic_type == "value":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        elif self.critic_type == "distributional":
            return self.value_layer(out)
