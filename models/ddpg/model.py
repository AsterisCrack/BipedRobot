from models.networks import ActorCriticWithTargets
from models.ddpg.ddpg import DDPG as DDPGAlgorithm
from models.utils import Model
import torch

class DDPG(Model):
    def __init__(self, env, model_path=None, device=torch.device("cpu")):
        # Initialize networks
        self.model = ActorCriticWithTargets(env.observation_space, env.action_space, [256, 256], [256, 256], actor_type="deterministic", device=device)

        self.model.to(device)
        
        self.agent = DDPGAlgorithm(
            action_space=env.action_space,
            model=self.model,
            device=device,
        )
        
        super().__init__(env, model_path, device)