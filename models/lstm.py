import torch
import torch.nn as nn
from torch.distributions import Normal
from algorithms.utils import SquashedMultivariateNormalDiag, DistributionalValueHead
from models.utils.base import Torso

class LSTMTorso(Torso):
    def __init__(self, observation_space, hidden_size=64, num_layers=2, observation_normalizer=None, history_size=0, device=torch.device("cpu")):
        super().__init__(observation_normalizer)
        self.observation_size = observation_space.shape[0]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.history_size = history_size

        if self.history_size > 0:
            self.input_dim = self.observation_size // self.history_size
        else:
            self.input_dim = self.observation_size

        # LSTM network
        self.lstm = nn.LSTM(self.input_dim, hidden_size, num_layers, batch_first=False)
        self.output_size = hidden_size

        # Pre-allocate hidden states
        self._init_hidden_states = (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        )

    def forward(self, observations, hidden_state=None):
        observations = self.normalize(observations)

        # Handle history if provided as a flat vector
        if self.history_size > 0:
            x = observations.view(-1, self.history_size, self.input_dim).transpose(0, 1)
        else:
            x = observations.unsqueeze(0)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=x.shape[1])

        lstm_out, hidden_state = self.lstm(x, hidden_state)
        return lstm_out[-1], hidden_state # Return last output and hidden state

    def init_hidden(self, batch_size=1):
        return (
            self._init_hidden_states[0].expand(-1, batch_size, -1).contiguous(),
            self._init_hidden_states[1].expand(-1, batch_size, -1).contiguous()
        )

class LSTMActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64, num_layers=2, observation_normalizer=None, head_type="gaussian", history_size=0, device=torch.device("cpu")):
        super().__init__()
        self.torso = LSTMTorso(observation_space, hidden_size, num_layers, observation_normalizer, history_size, device)
        self.head_type = head_type
        self.action_size = action_space.shape[0]

        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "gaussian_multivariate":
            self.mean_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Identity())
            self.std_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "deterministic":
            self.action_layer = nn.Sequential(nn.Linear(hidden_size, self.action_size), nn.Tanh())

    def forward(self, observations, hidden_state=None):
        out, next_hidden = self.torso(observations, hidden_state)
        # Activation for torso output
        out = torch.relu(out)
        
        if self.head_type == "deterministic":
            return self.action_layer(out)
        elif self.head_type == "gaussian":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return Normal(mean, std)
        elif self.head_type == "gaussian_multivariate":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return SquashedMultivariateNormalDiag(mean, std)

    def get_action(self, observations, hidden_state=None):
        out = self.forward(observations, hidden_state)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, 'mean'):
             return out.mean
        if hasattr(out, 'loc'):
            return out.loc
        return out

class LSTMCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64, num_layers=2, observation_normalizer=None, history_size=0, critic_type="deterministic", device=torch.device("cpu")):
        super().__init__()
        self.torso = LSTMTorso(observation_space, hidden_size, num_layers, observation_normalizer, history_size, device)
        self.critic_type = critic_type
        self.action_size = action_space.shape[0]

        if critic_type == "value":
            self.torso_out_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        else:
            self.torso_out_layer = nn.Sequential(nn.Linear(hidden_size + self.action_size, hidden_size), nn.ReLU())

        if critic_type == "deterministic" or critic_type == "value":
            self.value_layer = nn.Linear(hidden_size, 1)
        elif critic_type == "distributional":
            self.value_layer = DistributionalValueHead(-150, 150, 51, hidden_size)

    def forward(self, observations, actions=None, hidden_state=None):
        out, _ = self.torso(observations, hidden_state)
        out = torch.relu(out)
        
        if self.critic_type == "value":
            combined = out
        else:
            combined = torch.cat([out, actions], dim=-1)
            
        out = self.torso_out_layer(combined)
        
        if self.critic_type == "deterministic" or self.critic_type == "value":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        else:
            return self.value_layer(out)
