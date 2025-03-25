import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from models.utils import MeanStd, SquashedMultivariateNormalDiag, CategoricalWithSupport

class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None, head_type="gaussian"):
        super().__init__()
        self.observation_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        self.head_type = head_type
        
        # Main net (MLP with ReLU activations)
        sizes = [self.observation_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
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
            
        out = self.net(observations)
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

class DistributionalValueHead(torch.nn.Module):
    def __init__(self, vmin, vmax, num_atoms, input_size, return_normalizer=None, fn=None, device=torch.device("cpu")):
        super().__init__()
        self.num_atoms = num_atoms
        self.fn = fn
        self.values = torch.linspace(vmin, vmax, num_atoms).float().to(device)
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')
        self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms)
        if self.fn:
            self.distributional_layer.apply(self.fn)
            
    def to(self, device):
        """ Moves all components to a specific device """
        super().to(device)
        self.values = self.values.to(device)
        self.distributional_layer.to(device)
        return self
    
    def forward(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)
    
class Critic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, observation_normalizer=None, critic_type="deterministic"):
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
            
        out = torch.cat([observations, actions], dim=-1)
        out = self.torso(out)
        if self.critic_type == "deterministic":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        elif self.critic_type == "distributional":
            return self.value_layer(out)
    

class LSTMActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64, num_layers=2, observation_normalizer=None, head_type="gaussian", device=torch.device("cpu")):
        super().__init__()
        self.head_type = head_type
        self.observation_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # LSTM network
        self.lstm = nn.LSTM(self.observation_size, hidden_size, num_layers, batch_first=False)
        self.lstm_relu = nn.ReLU()

        # Policy heads
        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Softplus())
            self.std_min = 1e-4
            self.std_max = 1
        elif head_type == "deterministic":
            self.action_layer = nn.Sequential(
                nn.Linear(hidden_size, self.action_size), nn.Tanh())    

        # Pre-allocate hidden states on the device
        self._init_hidden_states = (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        )

    def forward(self, observations, hidden_state=None):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=observations.shape[1])

        lstm_out, _ = self.lstm(observations, hidden_state)  # Add batch dim
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[-1] # Remove batch dim
            
        # Pass through Relu
        lstm_out = self.lstm_relu(lstm_out)

        if self.head_type == "deterministic":
            return self.action_layer(lstm_out)
        elif self.head_type == "gaussian":
            mean = self.mean_layer(lstm_out)
            std = self.std_layer(lstm_out)
            std = torch.clamp(std, self.std_min, self.std_max)
            return Normal(mean, std)

    def get_action(self, observations, hidden_state):
        """ Selects an action and updates hidden state """
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        action_dist, _ = self.forward(observations, hidden_state)
        actions = action_dist.mean  # Mean action for deterministic control

        return actions.detach().cpu().numpy()
    
    def init_hidden(self, batch_size=1):
        """ Initialize hidden state for LSTM """
        return (
            self._init_hidden_states[0].expand(-1, batch_size, -1).contiguous(),
            self._init_hidden_states[1].expand(-1, batch_size, -1).contiguous()
        )

class LSTMCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64, num_layers=2, observation_normalizer=None, device=torch.device("cpu")):
        super().__init__()
        self.observation_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.observation_normalizer = observation_normalizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Main net (MLP with ReLU activations)
        sizes = [self.observation_size + self.action_size] + [hidden_size] * num_layers
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.torso = nn.Sequential(*layers)

        # Value head
        self.value_layer = nn.Linear(hidden_size, 1)

    def forward(self, observations, actions, hidden_state=None):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        actions = actions.unsqueeze(0).expand(observations.shape[0], -1, -1)
        combined_input = torch.cat([observations, actions], dim=-1)
        out = self.torso(combined_input)
        value = self.value_layer(out)
        value = torch.squeeze(value, -1)

        if out.dim() == 3:
            value = value[-1]  # Return the last value
        return value
    
class ActorCriticWithTargets(nn.Module):
    def __init__(self, obs_space, action_space, actor_sizes, critic_sizes, actor_type="gaussian", critic_type="deterministic", target_coeff=0.005, device=torch.device("cpu")):
        super().__init__()
        self.obs_space = obs_space
        self.observation_normalizer = MeanStd(shape=obs_space.shape)
        self.return_normalizer = None
        self.actor = Actor(obs_space, action_space, actor_sizes, self.observation_normalizer, actor_type)
        self.critic = Critic(obs_space, action_space, critic_sizes, self.observation_normalizer, critic_type)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_coeff = target_coeff
        self.assign_targets()
    
    # Redefine to(device) to move all submodules to the specified device
    def to(self, device):
        super().to(device)
        self.observation_normalizer.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        return self

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
                
class ActorTwinCriticWithTargets(nn.Module):
    def __init__(self, obs_space, action_space, actor_sizes, critic_sizes, actor_type="gaussian", target_coeff=0.005, device=torch.device("cpu")):
        super().__init__()
        self.obs_space = obs_space
        self.observation_normalizer = MeanStd(shape=obs_space.shape)
        self.return_normalizer = None
        self.actor = Actor(obs_space, action_space, actor_sizes, self.observation_normalizer, actor_type)
        self.critic_1 = Critic(obs_space, action_space, critic_sizes, self.observation_normalizer)
        self.critic_2 = copy.deepcopy(self.critic_1)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.target_coeff = target_coeff
        self.assign_targets()
    
    # Redefine to(device) to move all submodules to the specified device
    def to(self, device):
        super().to(device)
        self.observation_normalizer.to(device)
        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_actor.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)
        return self

    def assign_targets(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

    def update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
                
                    
class LSTMActorCriticWithTargets(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64, num_layers=2, seq_length=1, actor_type="gaussian", target_coeff=0.005, device=torch.device("cpu")):
        super().__init__()
        self.obs_space = obs_space
        # Mean std shape is observation space shape * seq_length
        self.observation_normalizer = MeanStd(shape=obs_space.shape)
        self.return_normalizer = None
        
        self.actor = LSTMActor(obs_space, action_space, hidden_size, num_layers, self.observation_normalizer, actor_type, device)
        self.critic = Critic(obs_space, action_space, [hidden_size]*num_layers, self.observation_normalizer)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_coeff = target_coeff
        self.assign_targets()

    def to(self, device):
        """ Moves all components to a specific device """
        super().to(device)
        self.observation_normalizer.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        return self

    def assign_targets(self):
        """ Hard copy actor/critic parameters to targets """
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def update_targets(self):
        """ Soft update of target networks """
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
            
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
                
class LSTMActorTwinCriticWithTargets(nn.Module):
    def __init__(self, obs_space, action_space, actor_type="gaussian", hidden_size=64, num_layers=2, seq_length=1, target_coeff=0.005, device=torch.device("cpu")):
        super().__init__()
        self.obs_space = obs_space
        # Mean std shape is observation space shape * seq_length
        self.observation_normalizer = MeanStd(shape=obs_space.shape)
        self.return_normalizer = None
        
        self.actor = LSTMActor(obs_space, action_space, hidden_size, num_layers, self.observation_normalizer, actor_type, device)
        self.critic_1 = Critic(obs_space, action_space, [hidden_size]*num_layers, self.observation_normalizer)
        self.critic_2 = copy.deepcopy(self.critic_1)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.target_coeff = target_coeff
        self.assign_targets()

    def to(self, device):
        """ Moves all components to a specific device """
        super().to(device)
        self.observation_normalizer.to(device)
        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_actor.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)
        return self

    def assign_targets(self):
        """ Hard copy actor/critic parameters to targets """
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

    def update_targets(self):
        """ Soft update of target networks """
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
            
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)
                
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.mul_(1 - self.target_coeff)
                target_param.data.add_(self.target_coeff * param.data)