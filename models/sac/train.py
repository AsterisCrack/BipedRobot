from models.sac.sac import SAC
from models.networks import ActorTwinCriticWithTargets
from models.utils import Trainer
import torch

class SACTrainer(Trainer):
    def __init__(self, env, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), seed=42, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3), test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None, chekpoint_path=None):
        
        # Initialize networks
        model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device, actor_type="gaussian_multivariate")
        model.to(device)
        
        # Initialize MPO algorithm
        mpo = SAC(
            action_space=env.action_space,
            model=model,
            device=device,
        )
        self.model = model
        self.device = device
        self.seed = seed
        
        # Initialize Trainer
        super().__init__(mpo, env, test_environment, steps, epoch_steps, save_steps, test_episodes, show_progress, replace_checkpoint, log, log_dir, log_name, chekpoint_path)
