import numpy as np
from torch.optim import Adam
from models.mpo.mpo import MPO
from models.mpo.networks import ActorCriticWithTargets
from envs.distributed import distribute
from envs.basic_env import BasicEnv
from models.utils import Trainer

def train():
    # Initialize environment
    # env = BasicEnv(render_mode="human")
    env = distribute(BasicEnv, 1)
    
    # Initialize networks
    model = ActorCriticWithTargets(env.observation_space, env.action_space, [256, 256], [256, 256])

    # Initialize MPO algorithm
    mpo = MPO(
        action_space=env.action_space,
        model=model,
    )

    trainer = Trainer(mpo, env)
    
    # Train agent
    trainer.run()
    
    
if __name__ == "__main__":
    train()


