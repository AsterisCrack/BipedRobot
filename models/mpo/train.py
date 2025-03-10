from models.mpo.mpo import MPO
from models.networks import ActorCriticWithTargets, LSTMActorCriticWithTargets
from models.utils import Trainer
import torch

class MPOTrainer(Trainer):
    def __init__(self, env, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), seed=42, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3), test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None, checkpoint_path=None):
        
        # Initialize networks
        model = ActorCriticWithTargets(env.observation_space, env.action_space, model_sizes[0], model_sizes[1], device=device)
        model.to(device)
        
        # Initialize MPO algorithm
        mpo = MPO(
            action_space=env.action_space,
            model=model,
            device=device,
        )
        self.model = model
        self.device = device
        self.seed = seed
        
        # Initialize Trainer
        super().__init__(mpo, env, test_environment, steps, epoch_steps, save_steps, test_episodes, show_progress, replace_checkpoint, log, log_dir, log_name, checkpoint_path)
        
class MPOTrainerLSTM(Trainer):
    def __init__(self, env, model_sizes=[[256, 256], [256, 256]], num_workers=1, seq_length=2, device=torch.device("cpu"), seed=42, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3), test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None, checkpoint_path=None):
        
        # Initialize networks
        hidden_size = model_sizes[0][-1]
        num_layers = len(model_sizes[0])
        model = LSTMActorCriticWithTargets(env.observation_space, env.action_space, hidden_size, num_layers, seq_length, device=device)
        model.to(device)
        
        # Initialize MPO algorithm
        mpo = MPO(
            action_space=env.action_space,
            model=model,
            recurrent_model=True,
            max_seq_length=seq_length,
            num_workers=num_workers,
            device=device,
        )
        self.model = model
        self.device = device
        self.seed = seed
        
        # Initialize Trainer
        super().__init__(mpo, env, test_environment, steps, epoch_steps, save_steps, test_episodes, show_progress, replace_checkpoint, log, log_dir, log_name, checkpoint_path)
