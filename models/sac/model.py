from models.networks import ActorTwinCriticWithTargets
from models.sac.sac import SAC as SACAlgorithm
from models.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, device=torch.device("cpu")):
        # Initialize networks
        model_sizes=[[256, 256], [256, 256]]
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, actor_type="gaussian_multivariate")
        
        self.agent = SACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            device=device,
        )

        super().__init__(env, model_path, device)