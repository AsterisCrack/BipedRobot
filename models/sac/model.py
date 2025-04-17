from models.networks import ActorTwinCriticWithTargets
from models.sac.sac import SAC as SACAlgorithm
from models.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), config=None):
        # Initialize networks
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, actor_type="gaussian_multivariate")
        
        super().__init__(env, model_path, device, config)
        
        self.agent = SACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
        )
        
        self._init_trainer()

        