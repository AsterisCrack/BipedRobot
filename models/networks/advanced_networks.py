import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from models.utils import MeanStd, SquashedMultivariateNormalDiag, CategoricalWithSupport

# Implementation of networks used in: https://journals.sagepub.com/doi/full/10.1177/02783649241285161
# Model will be fed a 2 second long IO history, wich will be fed into a 1D CNN
# Also, with a short 4 timestep history of the IO and the command which will be fed directly into the MLP

class AdvancedActor(nn.Module):
    def __init__(self, observation_space, long_history_size, short_history_size, action_space, hidden_sizes, cnn_sizes, observation_normalizer=None, head_type="gaussian"):
        super().__init__()
        
        self.long_history_size = long_history_size
        self.short_history_size = short_history_size
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        self.head_type = head_type
        
        self.observation_size = observation_space.shape[0]
        # obs_dim = self.model.nq + self.model.nv + \
        #    short_history_size * (self.model.nq + self.model.nv + self.model.nu) + \
        #    long_history_size * (self.model.nq + self.model.nv + self.model.nu)
        # Extract the observation size from the observation space, we want self.model.nq + self.model.nv
        self.observation_size = (self.observation_size - (self.action_size * (self.long_history_size + self.short_history_size)))\
            // (1 + self.long_history_size + self.short_history_size)
        
        # Main net (1D CNN for long history and MLP for short history)
        assert len(cnn_sizes) == 2, "CNN sizes must be a list of two tuples (kernel_size, out_channels, stride)"
        if self.long_history_size != 0:
            assert len(cnn_sizes[0]) == 3 and len(cnn_sizes[1]) == 3, "Each CNN size tuple must contain (kernel_size, out_channels, stride)"
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=cnn_sizes[0][1], kernel_size=cnn_sizes[0][0], stride=cnn_sizes[0][2]),
                nn.ReLU(),
                nn.Conv1d(in_channels=cnn_sizes[0][1], out_channels=cnn_sizes[1][1], kernel_size=cnn_sizes[1][0], stride=cnn_sizes[1][2]),
                nn.ReLU(),
                nn.Flatten()
            )
            # Main net (MLP with ReLU activations)
            # Calculate the output size of the CNN
            cnn_out_size = ((self.long_history_size*(self.observation_size + self.action_size) - cnn_sizes[0][0]) // cnn_sizes[0][2]) + 1
            cnn_out_size = ((cnn_out_size - cnn_sizes[1][0]) // cnn_sizes[1][2]) + 1
        else:
            cnn_out_size = 0
            self.cnn = None
        
        print("Observation size: ", self.observation_size)
        print("Long history size: ", self.long_history_size)
        print("Short history size: ", self.short_history_size)
        print("CNN out size: ", cnn_out_size)
        sizes = [self.observation_size + self.short_history_size*(self.observation_size + self.action_size) +  cnn_out_size*cnn_sizes[1][1]] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        
        # Policy head
        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(
                nn.Linear(hidden_sizes[-1], self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(
                nn.Linear(hidden_sizes[-1], self.action_size), nn.Softplus())
            self.std_min = 1e-4
            self.std_max = 1
        elif head_type == "gaussian_multivariate":
            self.mean_layer = nn.Sequential(
                nn.Linear(hidden_sizes[-1], self.action_size), nn.Identity())
            self.std_layer = nn.Sequential(
                nn.Linear(hidden_sizes[-1], self.action_size), nn.Softplus())
            self.std_min = 1e-4
            self.std_max = 1
        elif head_type == "deterministic":
            self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_sizes[-1], self.action_size),
            torch.nn.Tanh())

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
            
        # Split observations into long history and short history
        if self.long_history_size != 0:
            long_history = observations[:, -self.long_history_size*(self.observation_size+self.action_size):]
            observations = observations[:, :-self.long_history_size*(self.observation_size+self.action_size)]
        
            cnn_out = self.cnn(long_history.unsqueeze(1))
            # Concatenate the CNN output with the observations
            out = torch.cat([cnn_out, observations], dim=-1)
        else:
            # No long history, just use the short history
            out = observations
            
        out = self.net(out)
        if self.head_type == "gaussian":
            mean = self.mean_layer(out)
            std = self.std_layer(out)
            std = torch.clamp(std, self.std_min, self.std_max)
            return Normal(mean, std)
        elif self.head_type == "gaussian_multivariate":
            mean = self.mean_layer(out)
            std = self.std_layer(out)
            std = torch.clamp(std, self.std_min, self.std_max)
            return SquashedMultivariateNormalDiag(mean, std)
        elif self.head_type == "deterministic":
            return self.action_layer(out)
    
    def get_action(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        out = self.net(observations)
        if self.head_type == "gaussian" or self.head_type == "gaussian_multivariate":
            actions = self.mean_layer(out)
            return actions.detach().cpu().numpy()
        else:
            return self.action_layer(out).detach().cpu().numpy()
    
class AdvancedCritic(nn.Module):
    def __init__(self, observation_space, long_history_size, short_history_size, action_space, hidden_sizes, cnn_sizes, observation_normalizer=None, critic_type="deterministic", DistributionalValueHead=None):
        super().__init__()
        self.long_history_size = long_history_size
        self.short_history_size = short_history_size
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        
        self.observation_size = observation_space.shape[0]
        self.observation_size = (self.observation_size - self.action_size * (self.long_history_size + self.short_history_size))\
            // (1 + self.long_history_size + self.short_history_size)
        
        # Main net (1D CNN for long history and MLP for short history)
        assert len(cnn_sizes) == 2, "CNN sizes must be a list of two tuples (kernel_size, out_channels, stride)"
        if self.long_history_size != 0:
            assert len(cnn_sizes[0]) == 3 and len(cnn_sizes[1]) == 3, "Each CNN size tuple must contain (kernel_size, out_channels, stride)"
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=cnn_sizes[0][1], kernel_size=cnn_sizes[0][0], stride=cnn_sizes[0][2]),
                nn.ReLU(),
                nn.Conv1d(in_channels=cnn_sizes[0][1], out_channels=cnn_sizes[1][1], kernel_size=cnn_sizes[1][0], stride=cnn_sizes[1][2]),
                nn.ReLU(),
                nn.Flatten()
            )
            # Main net (MLP with ReLU activations)
            # Calculate the output size of the CNN
            cnn_out_size = ((self.long_history_size*(self.observation_size + self.action_size) - cnn_sizes[0][0]) // cnn_sizes[0][2]) + 1
            cnn_out_size = ((cnn_out_size - cnn_sizes[1][0]) // cnn_sizes[1][2]) + 1
        else:
            cnn_out_size = 0
            self.cnn = None
            
        sizes = [self.observation_size + self.short_history_size*(self.observation_size + self.action_size) +  cnn_out_size*cnn_sizes[1][1] + self.action_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        
        # Observation sequence reducer head
        self.obs_reducer = nn.Linear(self.observation_size, self.observation_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Value head
        self.critic_type = critic_type
        if critic_type == "deterministic":
            self.value_layer = nn.Linear(hidden_sizes[-1], 1)
        elif critic_type == "distributional":
            self.value_layer = DistributionalValueHead(-150, 150, 51, hidden_sizes[-1])

    def to(self, device):
        """ Moves all components to a specific device """
        super().to(device)
        self.value_layer.to(device)
        return self
    
    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        
        if observations.dim() == 3:
            # observations is a sequence
            observations = self.obs_reducer(observations)
            observations = self.pool(observations.transpose(1, 2)).transpose(1, 2)
            # Now reshape to (batch_size, observation_size)
            observations = observations.reshape(actions.shape[0], -1)
        if self.long_history_size != 0:
            # Split observations into long history and short history
            long_history = observations[:, -self.long_history_size*(self.observation_size+self.action_size):]
            observations = observations[:, :-self.long_history_size*(self.observation_size+self.action_size)]
            
            cnn_out = self.cnn(long_history.unsqueeze(1))
            # Concatenate the CNN output with the observations
            out = torch.cat([cnn_out, observations, actions], dim=-1)
        else:
            # No long history, just use the short history
            out = torch.cat([observations, actions], dim=-1)
        out = self.net(out)
        if self.critic_type == "deterministic":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        elif self.critic_type == "distributional":
            return self.value_layer(out)