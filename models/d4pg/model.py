from models.networks import ActorCriticWithTargets
from models.d4pg.d4pg import D4PG as D4PGAlgorithm
from models.utils import Model
import torch

class D4PG(Model):
    def __init__(self, env, model_path=None, device=torch.device("cpu")):
         # Initialize networks
        self.model = ActorCriticWithTargets(env.observation_space, env.action_space, [256, 256], [256, 256], actor_type="deterministic", critic_type="distributional", device=device)

        # Initialize algorithm
        self.agent = D4PGAlgorithm(
            action_space=env.action_space,
            model=self.model,
            device=device,
        )
        
        super().__init__(env, model_path, device)