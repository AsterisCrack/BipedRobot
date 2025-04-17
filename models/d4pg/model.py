from models.networks import ActorCriticWithTargets
from models.d4pg.d4pg import D4PG as D4PGAlgorithm
from models.utils import Model
import torch

class D4PG(Model):
    def __init__(self, env, model_path=None, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), config=None):
         # Initialize networks
        self.model = ActorCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], actor_type="deterministic", critic_type="distributional", device=device)
        
        super().__init__(env, model_path, device, config)
        
        # Initialize algorithm
        self.agent = D4PGAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
        )
        
        self._init_trainer()
        