from models.networks import ActorTwinCriticWithTargets
from algorithms.fast_sac.fast_sac import FastSAC as FastSACAlgorithm
from algorithms.utils import Model
import torch

class FastSAC(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, device=torch.device("cpu"), config=None):
        
        # Override config for FastSAC defaults if config is provided
        if config:
             # Apply FastSAC specific buffer settings if they are not manually set to something else appropriate
             # This is a bit invasive but ensures "FastSAC behavior" by default
             if hasattr(config, "buffer"):
                 # We assume config.buffer is an object or dict-like
                 # Configure for "Large Batch, Single Update"
                 if hasattr(config.buffer, "batch_iterations"):
                    config.buffer.batch_iterations = 1
                 if hasattr(config.buffer, "batch_size"):
                    # Use large batch size if default (1024 or similar small number)
                    # We check if it is small (< 4096) to avoid overriding user intent if they already set it high
                    val = config.buffer.batch_size or 0
                    if val < 4096:
                        config.buffer.batch_size = 65536
                 
                 # Configure steps between batches for efficient GPU usage
                 if hasattr(config.buffer, "steps_between_batches"):
                     val = config.buffer.steps_between_batches or 0
                     if val < 10:
                         config.buffer.steps_between_batches = 64
                         
                 # Ensure return steps is 1 (off-policy usually 1)
                 if hasattr(config.buffer, "return_steps"):
                     config.buffer.return_steps = 1
                     
                 # Increase buffer size if it's small
                 if hasattr(config.buffer, "size"):
                     if (config.buffer.size or 0) < 1000000:
                         config.buffer.size = 2000000

        # Initialize networks
        self.model = ActorTwinCriticWithTargets(
            env.observation_space, 
            env.action_space, 
            actor_type="gaussian_multivariate", 
            device=device, 
            use_history=use_history, 
            history_size=history_size, 
            config=config
        )
        
        super().__init__(env, model_path, device, config)
        
        self.agent = FastSACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
            config=config,
        )
        # Pass environment to agent for symmetry augmentation
        if hasattr(self.agent, "set_env"):
            self.agent.set_env(env)
