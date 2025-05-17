from models.networks.networks import ActorTwinCriticWithTargets
from models.sac.sac import SAC as SACAlgorithm
from models.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, use_history=False, long_history_size=0, short_history_size=0, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), config=None):
        # Initialize networks
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, actor_type="gaussian_multivariate", use_history=use_history, long_history_size=long_history_size, short_history_size=short_history_size)
        
        super().__init__(env, model_path, device, config)
        
        self.agent = SACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
            config=config,
        )
        
        self._init_trainer()

        