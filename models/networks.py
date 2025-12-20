import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from algorithms.utils import MeanStd
from models.factory import NetworkFactory
from config.schema import NetworkType, NetworkConfig
from gymnasium import spaces
    
class BaseActorCritic(nn.Module):
    def __init__(self, obs_space, target_coeff=0.005):
        super().__init__()
        self.obs_space = obs_space
        self.target_coeff = target_coeff
        
        # Initialize normalizers
        if isinstance(obs_space, spaces.Dict):
            self.actor_obs_space = obs_space["actor"]
            self.critic_obs_space = obs_space["critic"]
            self.actor_observation_normalizer = MeanStd(shape=self.actor_obs_space.shape)
            self.critic_observation_normalizer = MeanStd(shape=self.critic_obs_space.shape)
        else:
            self.actor_obs_space = obs_space
            self.critic_obs_space = obs_space
            self.actor_observation_normalizer = MeanStd(shape=obs_space.shape)
            self.critic_observation_normalizer = self.actor_observation_normalizer

        self.observation_normalizer = self.actor_observation_normalizer if not isinstance(obs_space, spaces.Dict) else None
        self.return_normalizer = None

    def _resolve_configs(self, config, network_type, use_history, actor_sizes, critic_sizes):
        # Resolve Actor Network Config
        if config and hasattr(config, "model") and config.model.actor_config:
            actor_config = config.model.actor_config
        else:
            nt = network_type or (NetworkType.CNN if use_history else NetworkType.MLP)
            actor_config = NetworkConfig(network_type=nt, hidden_sizes=actor_sizes or [256, 256])
        
        # Resolve Critic Network Config
        if config and hasattr(config, "model") and config.model.critic_config:
            critic_config = config.model.critic_config
        else:
            nt = network_type or (NetworkType.CNN if use_history else NetworkType.MLP)
            critic_config = NetworkConfig(network_type=nt, hidden_sizes=critic_sizes or [256, 256])
            
        return actor_config, critic_config

    def to(self, device):
        super().to(device)
        if self.observation_normalizer:
            self.observation_normalizer.to(device)
        if self.actor_observation_normalizer:
            self.actor_observation_normalizer.to(device)
        if self.critic_observation_normalizer and self.critic_observation_normalizer is not self.actor_observation_normalizer:
            self.critic_observation_normalizer.to(device)
        return self

    def _get_sync_pairs(self):
        """ Returns a list of (target_module, source_module) pairs for synchronization. """
        raise NotImplementedError

    def assign_targets(self):
        """ Hard copy parameters to target networks. """
        for target, source in self._get_sync_pairs():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def update_targets(self):
        """ Soft update of target networks. """
        with torch.no_grad():
            for target, source in self._get_sync_pairs():
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.mul_(1 - self.target_coeff)
                    target_param.data.add_(self.target_coeff * param.data)

class ActorCriticWithTargets(BaseActorCritic):
    def __init__(self, obs_space, action_space, actor_sizes=None, critic_sizes=None, actor_type="gaussian", critic_type="deterministic", use_history=False, history_size=0, target_coeff=0.005, device=torch.device("cpu"), network_type=None, config=None):
        super().__init__(obs_space, target_coeff)
        
        actor_config, critic_config = self._resolve_configs(config, network_type, use_history, actor_sizes, critic_sizes)

        # Build networks using factory
        self.actor = NetworkFactory.build_actor(actor_config, self.actor_obs_space, action_space, self.actor_observation_normalizer, actor_type, history_size, device)
        self.critic = NetworkFactory.build_critic(critic_config, self.critic_obs_space, action_space, self.critic_observation_normalizer, critic_type, history_size, device)
            
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.assign_targets()

    def _get_sync_pairs(self):
        return [(self.target_actor, self.actor), (self.target_critic, self.critic)]

class ActorTwinCriticWithTargets(BaseActorCritic):
    def __init__(self, obs_space, action_space, actor_sizes=None, critic_sizes=None, actor_type="gaussian", use_history=False, history_size=0, target_coeff=0.005, device=torch.device("cpu"), network_type=None, config=None):
        super().__init__(obs_space, target_coeff)
        
        actor_config, critic_config = self._resolve_configs(config, network_type, use_history, actor_sizes, critic_sizes)

        # Build networks using factory
        self.actor = NetworkFactory.build_actor(actor_config, self.actor_obs_space, action_space, self.actor_observation_normalizer, actor_type, history_size, device)
        self.critic_1 = NetworkFactory.build_critic(critic_config, self.critic_obs_space, action_space, self.critic_observation_normalizer, "deterministic", history_size, device)
        self.critic_2 = copy.deepcopy(self.critic_1)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.assign_targets()

    def _get_sync_pairs(self):
        return [
            (self.target_actor, self.actor),
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2)
        ]
                