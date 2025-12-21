from models.mlp import MLPActor, MLPCritic
from models.cnn import CNNActor, CNNCritic
from models.lstm import LSTMActor, LSTMCritic
from models.transformer import TransformerActor, TransformerCritic
from config.schema import NetworkType, NetworkConfig
import torch

class NetworkFactory:
    @staticmethod
    def get_network_classes(network_type: NetworkType):
        if network_type == NetworkType.MLP:
            return MLPActor, MLPCritic
        elif network_type == NetworkType.CNN:
            return CNNActor, CNNCritic
        elif network_type == NetworkType.LSTM:
            return LSTMActor, LSTMCritic
        elif network_type == NetworkType.TRANSFORMER:
            return TransformerActor, TransformerCritic
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    @staticmethod
    def build_actor(config: NetworkConfig, observation_space, action_space, normalizer, head_type, history_size=0, device=torch.device("cpu")):
        network_type = config.network_type
        if network_type == NetworkType.MLP:
            return MLPActor(observation_space, action_space, config.hidden_sizes, normalizer, head_type)
        elif network_type == NetworkType.CNN:
            # Assuming cnn_sizes is provided in config or use defaults
            cnn_sizes = config.cnn_sizes or [[3, 32, 2], [3, 32, 2]]
            return CNNActor(observation_space, history_size, action_space, config.hidden_sizes, cnn_sizes, normalizer, head_type)
        elif network_type == NetworkType.LSTM:
            return LSTMActor(observation_space, action_space, config.hidden_size, config.num_layers, normalizer, head_type, history_size, device)
        elif network_type == NetworkType.TRANSFORMER:
            return TransformerActor(observation_space, action_space, config.d_model, config.nhead, config.num_layers, config.dim_feedforward, normalizer, head_type, history_size, device)
        else:
            raise ValueError(f"Unknown actor network type: {network_type}")

    @staticmethod
    def build_critic(config: NetworkConfig, observation_space, action_space, normalizer, critic_type, history_size=0, device=torch.device("cpu")):
        network_type = config.network_type
        if network_type == NetworkType.MLP:
            return MLPCritic(observation_space, action_space, config.hidden_sizes, normalizer, critic_type)
        elif network_type == NetworkType.CNN:
            cnn_sizes = config.cnn_sizes or [[3, 32, 2], [3, 32, 2]]
            return CNNCritic(observation_space, history_size, action_space, config.hidden_sizes, cnn_sizes, normalizer, critic_type)
        elif network_type == NetworkType.LSTM:
            return LSTMCritic(observation_space, action_space, config.hidden_size, config.num_layers, normalizer, history_size, critic_type, device)
        elif network_type == NetworkType.TRANSFORMER:
            return TransformerCritic(observation_space, action_space, config.d_model, config.nhead, config.num_layers, config.dim_feedforward, normalizer, history_size, critic_type, device)
        else:
            raise ValueError(f"Unknown critic network type: {network_type}")

