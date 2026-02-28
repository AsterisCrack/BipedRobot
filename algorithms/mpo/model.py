from models.networks import ActorCriticWithTargets
from algorithms.mpo.mpo import MPO as MPOAlgorithm
from algorithms.utils import Model
import torch

class MPO(Model):
    def __init__(self, env, lstm=False, seq_length=2, model_path=None, use_history=False, history_size=0, model_sizes = [[256, 256], [256, 256]], device=torch.device("cpu"), config=None, network_type=None):
        # Initialize networks
        self.model = ActorCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, use_history=use_history, history_size=history_size, network_type=network_type, config=config)

        super().__init__(env, model_path, device, config)
        
        # Copy actor optimizer for the dual variables
        self.dual_optimizer = self.actor_optimizer
        self.agent = MPOAlgorithm(
            action_space=env.action_space,
            model=self.model,
            device=device,
            config=config,
            actor_optimizer=self.actor_optimizer,
            dual_optimizer=self.dual_optimizer,
            critic_optimizer=self.critic_optimizer,
        )