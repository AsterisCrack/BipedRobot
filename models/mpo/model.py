from models.networks import ActorCriticWithTargets, LSTMActorCriticWithTargets
from models.mpo.mpo import MPO as MPOAlgorithm
from models.utils import Model
import torch

class MPO(Model):
    def __init__(self, env, lstm=False, seq_length=2, model_path=None, device=torch.device("cpu")):
        # Initialize networks
        if lstm:
            # Initialize networks
            model_sizes = [[256, 256], [256, 256]]
            hidden_size = model_sizes[0][-1]
            num_layers = len(model_sizes[0])
            self.model = LSTMActorCriticWithTargets(env.observation_space, env.action_space, hidden_size, num_layers, seq_length, device=device)
        else:
            self.model = ActorCriticWithTargets(env.observation_space, env.action_space, [256, 256], [256, 256], device=device)
        
        self.agent = MPOAlgorithm(
            action_space=env.action_space,
            model=self.model,
            device=device,
        )

        super().__init__(env, model_path, device)