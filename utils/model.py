import torch
# import all models
from models.mpo.model import MPO
from models.ddpg.model import DDPG
from models.sac.model import SAC
from models.d4pg.model import D4PG

class Model():
    def __init__(self, env, type, model_path=None, device=torch.device("cpu")):
        # Initialize networks
        if type.lower() == "mpo":
            self.model = MPO(model_path, env, device=device)
        elif type.lower() == "ddpg":
            self.model = DDPG(model_path, env, device=device)
        elif type.lower() == "sac":
            self.model = SAC(model_path, env, device=device)
        elif type.lower() == "d4pg":
            self.model = D4PG(model_path, env, device=device)
        else:
            raise ValueError("Model type not supported")
        
    def step(self, observation):
        action = self.model.step(observation)
        return action
