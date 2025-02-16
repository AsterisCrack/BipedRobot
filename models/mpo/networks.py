import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from models.utils import MeanStd

class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None):
        super().__init__()
        self.observation_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        
        # Main net (MLP with ReLU activations)
        sizes = [self.observation_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        
        # Policy head
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], self.action_size), nn.Tanh())
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], self.action_size), nn.Softplus())
        self.std_min = 1e-4
        self.std_max = 1

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        out = self.net(observations)
        mean = self.mean_layer(out)
        std = self.std_layer(out)
        std = torch.clamp(std, self.std_min, self.std_max)
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None):
        super().__init__()
        self.observation_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        
        # Main net (MLP with ReLU activations)
        sizes = [self.observation_size + self.action_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.torso = nn.Sequential(*layers)
        
        # Value head
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        out = torch.cat([observations, actions], dim=-1)
        out = self.torso(out)
        value = self.value_layer(out)
        return torch.squeeze(value, -1)

class ActorCriticWithTargets(nn.Module):
    def __init__(self, obs_space, action_space, actor_sizes, critic_sizes, target_coeff=0.005):
        super().__init__()
        self.observation_normalizer = MeanStd(shape=obs_space.shape)
        self.return_normalizer = None
        self.actor = Actor(obs_space, action_space, actor_sizes, self.observation_normalizer)
        self.critic = Critic(obs_space, action_space, critic_sizes, self.observation_normalizer)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_coeff = target_coeff
        self.assign_targets()

    def assign_targets(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)