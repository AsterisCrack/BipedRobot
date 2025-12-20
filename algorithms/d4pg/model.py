from models.networks import ActorCriticWithTargets
from algorithms.d4pg.d4pg import D4PG as D4PGAlgorithm
from algorithms.utils import Model
import torch

class D4PG(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), config=None, network_type=None):
         # Initialize networks
        self.model = ActorCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], actor_type="deterministic", critic_type="distributional", device=device, use_history=use_history, history_size=history_size, network_type=network_type, config=config)
        
        super().__init__(env, model_path, device, config)
        
        # Initialize algorithm
        self.agent = D4PGAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
            config=config,
        )
        
        self._init_trainer()
        