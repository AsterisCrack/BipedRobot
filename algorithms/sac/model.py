from models.networks import ActorTwinCriticWithTargets
from algorithms.sac.sac import SAC as SACAlgorithm
from algorithms.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, device=torch.device("cpu"), config=None):
        # Initialize networks
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, actor_type="gaussian_multivariate", device=device, use_history=use_history, history_size=history_size, config=config)
        
        super().__init__(env, model_path, device, config)
        
        self.agent = SACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
            config=config,
        )
        if hasattr(self.agent, "set_env"):
            self.agent.set_env(env)

        