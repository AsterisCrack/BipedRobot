import torch
import torch.nn as nn
from torch.distributions import Normal
from algorithms.utils import SquashedMultivariateNormalDiag, DistributionalValueHead
from models.utils.base import Torso

class CNNTorso(Torso):
    def __init__(self, observation_space, history_size, cnn_sizes, observation_normalizer=None):
        super().__init__(observation_normalizer)
        self.history_size = history_size
        
        # Calculate input dim
        if history_size > 0:
            # Note: This logic assumes history is concatenated in specific way.
            # In MujocoEnv it's concatenated.
            self.input_dim = observation_space.shape[0] // history_size
        else:
            self.input_dim = observation_space.shape[0]

        if history_size > 0:
            assert len(cnn_sizes) == 2, "CNN sizes must be a list of two tuples (kernel_size, out_channels, stride)"
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim, out_channels=cnn_sizes[0][1], kernel_size=cnn_sizes[0][0], stride=cnn_sizes[0][2]),
                nn.ReLU(),
                nn.Conv1d(in_channels=cnn_sizes[0][1], out_channels=cnn_sizes[1][1], kernel_size=cnn_sizes[1][0], stride=cnn_sizes[1][2]),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate output size
            l_out = ((history_size - cnn_sizes[0][0]) // cnn_sizes[0][2]) + 1
            l_out = ((l_out - cnn_sizes[1][0]) // cnn_sizes[1][2]) + 1
            self.output_size = l_out * cnn_sizes[1][1]
        else:
            self.cnn = nn.Identity()
            self.output_size = self.input_dim

    def forward(self, x):
        x = self.normalize(x)
        if self.history_size > 0:
            # Reshape to (batch, history_size, input_dim) then transpose to (batch, input_dim, history_size)
            x = x.view(-1, self.history_size, self.input_dim).transpose(1, 2)
            return self.cnn(x)
        return x

class CNNActor(nn.Module):
    def __init__(self, observation_space, history_size, action_space, hidden_sizes, cnn_sizes, observation_normalizer=None, head_type="gaussian"):
        super().__init__()
        self.torso = CNNTorso(observation_space, history_size, cnn_sizes, observation_normalizer)
        self.head_type = head_type
        self.action_size = action_space.shape[0]
        
        # MLP layers after CNN
        sizes = [self.torso.output_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        
        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "gaussian_multivariate":
            self.mean_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], self.action_size), nn.Identity())
            self.std_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "deterministic":
            self.action_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], self.action_size), nn.Tanh())

    def forward(self, observations):
        out = self.torso(observations)
        out = self.net(out)
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
        if hasattr(out, 'mean'):
             return out.mean.detach().cpu().numpy()
        return out.detach().cpu().numpy()

class CNNCritic(nn.Module):
    def __init__(self, observation_space, history_size, action_space, hidden_sizes, cnn_sizes, observation_normalizer=None, critic_type="deterministic"):
        super().__init__()
        self.torso = CNNTorso(observation_space, history_size, cnn_sizes, observation_normalizer)
        self.critic_type = critic_type
        self.action_size = action_space.shape[0]
        
        # Note: CNN Critic traditionally concatenates action AFTER CNN
        sizes = [self.torso.output_size + self.action_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        
        if critic_type == "deterministic":
            self.value_layer = nn.Linear(hidden_sizes[-1], 1)
        elif critic_type == "distributional":
            self.value_layer = DistributionalValueHead(-150, 150, 51, hidden_sizes[-1])

    def forward(self, observations, actions):
        out = self.torso(observations)
        out = torch.cat([out, actions], dim=-1)
        out = self.net(out)
        if self.critic_type == "deterministic":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        elif self.critic_type == "distributional":
            return self.value_layer(out)
