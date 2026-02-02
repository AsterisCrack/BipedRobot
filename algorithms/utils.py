import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import NoConfig

def to_tensor(obs, device):
    """Converts observations (array or dict) to torch tensors."""
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
    return torch.as_tensor(obs, dtype=torch.float32, device=device)

class RunningMeanStd:
    def __init__(self, shape, device='cpu', epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def state_dict(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.mean.device)
        self.var = state_dict['var'].to(self.var.device)
        self.count = state_dict['count']
        if isinstance(self.count, torch.Tensor):
             self.count = self.count.to(self.mean.device)

class OptimizerWithScheduler(torch.optim.Optimizer):
    """ Wrapper for optimizers with a scheduler. """
    def __init__(self, optimizer, scheduler, start_step=0):
        self.optimizer = optimizer
        self.scheduler = scheduler(self.optimizer) if scheduler else None
        self.start_step = start_step
    
    def step(self, metrics=None, steps=0):
        # Get current LRs before scheduler step
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        
        self.optimizer.step()
        
        if self.scheduler is not None and steps >= self.start_step:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    self.scheduler.step(metrics)
            else:
                self.scheduler.step()
            
            # Check for LR changes
            new_lrs = [group["lr"] for group in self.optimizer.param_groups]
            for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                if new_lr < old_lr:
                    print(f"[INFO] Learning rate for group {i} decreased from {old_lr:.6e} to {new_lr:.6e}")
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        state = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        
    def __getattr__(self, name):
        return getattr(self.optimizer, name)
    
    def __setattr__(self, name, value):
        if name in ['optimizer', 'scheduler', 'start_step']:
            self.__dict__[name] = value
        else:
            setattr(self.optimizer, name, value)
    
class Model:
    def __init__(self, env, model_path=None, device=torch.device("cpu"), config=None):
        
        self.config = config or NoConfig()
        # Initialize optimizers and schedulers
        model_cfg = self.config["model"]
        actor_lr = model_cfg.get("actor_lr", 1e-3)
        critic_lr = model_cfg.get("critic_lr", 1e-3)
        
        lr_sched_cfg = model_cfg.get("lr_scheduler")
        
        def get_scheduler_fn(cfg):
            if not cfg:
                # Default Cosine
                return lambda optimizer: lr_scheduler.CosineAnnealingLR(optimizer, T_max=5e5, eta_min=1e-5)
            
            sched_type = cfg.get("scheduler_type", "cosine")
            
            if sched_type == "none":
                return None
                
            if sched_type == "plateau":
                 mode = cfg.get("mode", "min")
                 factor = cfg.get("factor", 0.1)
                 patience = cfg.get("patience", 10)
                 threshold = cfg.get("threshold", 1e-4)
                 threshold_mode = cfg.get("threshold_mode", "rel")
                 cooldown = cfg.get("cooldown", 0)
                 min_lr = cfg.get("min_lr", 0)
                 eps = cfg.get("eps", 1e-8)
                 return lambda optimizer: lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode=mode, factor=factor, patience=patience,
                     threshold=threshold, threshold_mode=threshold_mode,
                     cooldown=cooldown, min_lr=min_lr, eps=eps
                 )
            else:
                 # Cosine
                 T_max = cfg.get("T_max", 5e5)
                 if T_max is None: T_max = 5e5
                 eta_min = cfg.get("eta_min", 1e-5)
                 return lambda optimizer: lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        scheduler_fn = get_scheduler_fn(lr_sched_cfg)
        
        # Get start step from config
        start_step = 0
        if lr_sched_cfg:
            start_step = lr_sched_cfg.get("start_step", 0)

        self.actor_optimizer = (
            lambda params: OptimizerWithScheduler(
                torch.optim.Adam(params, lr=actor_lr),
                scheduler_fn,
                start_step
            )
        )
        self.critic_optimizer = (
            lambda params: OptimizerWithScheduler(
                torch.optim.Adam(params, lr=critic_lr),
                scheduler_fn,
                start_step
            )
        )
        """
        self.actor_optimizer = lambda params: torch.optim.Adam(params, lr=actor_lr)
        self.critic_optimizer = lambda params: torch.optim.Adam(params, lr=critic_lr)"""
        
        # Load the saved model
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Load environment state
            if hasattr(env, "load"):
                env.load(model_path)
        
        self.model.to(device)
        
        # Initialize environment with rendering
        self.env = env
        
        self.device = device
    
    def step(self, observation):
        if isinstance(observation, dict):
            if "policy" in observation:
                obs = observation["policy"]
            elif "actor" in observation:
                obs = observation["actor"]
            else:
                obs = observation
        else:
            obs = observation
            
        action = self.model.actor.get_action(obs)
        return action
    
    def train(self, seed=42, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3), test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None, checkpoint_path=None, config=None):
        
        # Initialize trainer
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.trainer = Trainer(self.agent, self.env, test_environment, steps, epoch_steps, save_steps, test_episodes, show_progress, replace_checkpoint, log, log_dir, log_name, checkpoint_path, config or self.config)
        
        self.trainer.run()
    
    def save_trainer_state(self):
        self.trainer.save_trainer_state()
        
    def load_trainer_state(self, path):
        self.trainer.load_trainer_state(path)
        
    
class Buffer:
    '''Replay storing a large number of transitions for off-policy learning
    and using n-step returns.'''

    def __init__(
        self, size=int(1e6), return_steps=1, batch_iterations=50,
        batch_size=1024, discount_factor=0.99, steps_before_batches=int(1e4),
        steps_between_batches=50, seed=None, config=None, device=torch.device("cpu")
    ):
        self.device = device
        self.config = config or NoConfig()
        self.full_max_size = int(self.config["buffer"]["size"] or size)
        
        self.return_steps = self.config["buffer"]["return_steps"] or return_steps
        self.batch_iterations = self.config["buffer"]["batch_iterations"] or batch_iterations
        self.batch_size = self.config["buffer"]["batch_size"] or batch_size
        self.discount_factor = self.config["buffer"]["discount_factor"] or discount_factor
        self.steps_before_batches = self.config["buffer"]["steps_before_batches"] or steps_before_batches
        self.steps_between_batches = self.config["buffer"]["steps_between_batches"] or steps_between_batches
        self.seed = self.config["buffer"]["seed"] or seed
        self.buffers = None
        self.index = 0
        self.size = 0
        self.last_steps = 0

    def ready(self, steps):
        if steps < self.steps_before_batches:
            return False
        return (steps - self.last_steps) >= self.steps_between_batches

    def store(self, **kwargs):
        if 'terminations' in kwargs:
            term = kwargs['terminations']
            if torch.is_tensor(term):
                continuations = 1.0 - term.float()
            else:
                continuations = np.float32(1 - term)
            kwargs['discounts'] = continuations * self.discount_factor
            
        # Flatten dicts
        flattened_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flattened_kwargs[f"{k}_{subk}"] = subv
            else:
                flattened_kwargs[k] = v
        kwargs = flattened_kwargs

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.max_size = self.full_max_size // self.num_workers
            self.buffers = {}
            for key, val in kwargs.items():
                if torch.is_tensor(val):
                    val_shape = val.shape
                else:
                    val_shape = np.array(val).shape
                shape = (self.max_size,) + tuple(int(x) for x in val_shape)
                self.buffers[key] = torch.full(shape, float("nan"), dtype=torch.float32, device=self.device)

        # Store the new values.
        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=self.device)
            else:
                val = val.to(dtype=torch.float32, device=self.device)
            self.buffers[key][self.index] = val
            
        # Accumulate values for n-step returns.
        if self.return_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def accumulate_n_steps(self, kwargs):
        rewards = kwargs['rewards']
        discounts = kwargs['discounts']
        
        # Ensure all are torch tensors on the correct device
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(dtype=torch.float32, device=self.device)
        
        if not torch.is_tensor(discounts):
            discounts = torch.tensor(discounts, dtype=torch.float32, device=self.device)
        else:
            discounts = discounts.to(dtype=torch.float32, device=self.device)

        # Identify next observation keys
        next_obs_keys = [k for k in kwargs.keys() if 'next_observations' in k]
        
        # Convert next obs to tensors
        next_obs_tensors = {}
        for k in next_obs_keys:
            val = kwargs[k]
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=self.device)
            else:
                val = val.to(dtype=torch.float32, device=self.device)
            next_obs_tensors[k] = val

        # Accumulate n-step returns.
        masks = torch.ones(self.num_workers, dtype=torch.float32, device=self.device)
        for i in range(min(self.size, self.return_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= (1 - self.buffers['resets'][index])
            new_rewards = (self.buffers['rewards'][index] +
                        self.buffers['discounts'][index] * rewards)
            self.buffers['rewards'][index] = (
                (1 - masks) * self.buffers['rewards'][index] +
                masks * new_rewards)
            new_discounts = self.buffers['discounts'][index] * discounts
            self.buffers['discounts'][index] = (
                (1 - masks) * self.buffers['discounts'][index] +
                masks * new_discounts)
            
            # Update next observations for all keys
            for k in next_obs_keys:
                self.buffers[k][index] = (
                    (1 - masks)[:, None] *
                    self.buffers[k][index] +
                    masks[:, None] * next_obs_tensors[k])

    def get(self, *keys, steps):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = torch.randint(0, total_size, (self.batch_size,), device=self.device)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}

    def sample(self, batch_size=None, *keys):
        '''Sample a single large batch without generator overhead.'''
        bs = batch_size if batch_size is not None else self.batch_size
        total_size = self.size * self.num_workers
        indices = torch.randint(0, total_size, (bs,), device=self.device)
        rows = indices // self.num_workers
        columns = indices % self.num_workers
        
        # If no keys provided, return all
        if not keys:
            keys = self.buffers.keys()
            
        return {k: self.buffers[k][rows, columns] for k in keys}

        self.last_steps = steps
    
    def save(self, path):
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.buffers, path)
        
    def load(self, path):
        path = path + '.pt'
        self.buffers = torch.load(path)
        self.size = self.buffers['observations'].shape[0]
        self.index = 0
        self.max_size = self.size
        self.full_max_size = self.size
        self.num_workers = self.buffers['observations'].shape[1]
        self.last_steps = 0
        print(f"Loaded buffer from {path}")

class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = torch.nn.functional.softmax(logits, dim=-1)

    @property
    def mean(self):
        return (self.probabilities * self.values).sum(dim=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = torch.cat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = torch.cat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = torch.clamp(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = (delta_values >= 0).float()
        delta_hat = ((delta_sign * delta_values / d_pos) -
                     ((1 - delta_sign) * delta_values / d_neg))
        delta_clipped = torch.clamp(1 - delta_hat, 0, 1)

        return (delta_clipped * self.probabilities[:, None]).sum(dim=2)
    
class SquashedMultivariateNormalDiag:
    def __init__(self, loc, scale):
        self._distribution = torch.distributions.normal.Normal(loc, scale)

    def rsample_with_log_prob(self, shape=()):
        samples = self._distribution.rsample(shape)
        squashed_samples = torch.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)
        return squashed_samples, log_probs

    def rsample(self, shape=()):
        samples = self._distribution.rsample(shape)
        return torch.tanh(samples)

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return torch.tanh(samples)

    def log_prob(self, samples):
        '''Required unsquashed samples cannot be accurately recovered.'''
        raise NotImplementedError(
            'Not implemented to avoid approximation errors. '
            'Use sample_with_log_prob directly.')

    @property
    def loc(self):
        return torch.tanh(self._distribution.mean)

    @property
    def mean(self):
        return self.loc
    
class MeanStd(torch.nn.Module):
    def __init__(self, mean=0, std=1, clip=None, shape=None):
        super().__init__()
        self.mean_val = mean
        self.std_val = std
        self.clip = clip
        self.count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self.new_count = 0
        self.eps = 1e-2
        
        if shape:   
            if isinstance(self.mean_val, (int, float)):
                self.mean_val = torch.full(shape, self.mean_val, dtype=torch.float32)
            else:
                self.mean_val = torch.as_tensor(self.mean_val, dtype=torch.float32)
            if isinstance(self.std_val, (int, float)):
                self.std_val = torch.full(shape, self.std_val, dtype=torch.float32)
            else:
                self.std_val = torch.as_tensor(self.std_val, dtype=torch.float32)
            
            self.mean_sq = torch.square(self.mean_val)
            self._mean = torch.nn.Parameter(self.mean_val.clone(), requires_grad=False)
            self._std = torch.nn.Parameter(self.std_val.clone(), requires_grad=False)

    def forward(self, val):
        with torch.no_grad():
            val = (val - self._mean) / self._std
            if self.clip is not None:
                val = torch.clamp(val, -self.clip, self.clip)
        return val

    def unnormalize(self, val):
        return val * self._std + self._mean

    def record(self, values):
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float32, device=self._mean.device)
        
        # Ensure values are on the same device as parameters
        if values.device != self._mean.device:
            values = values.to(self._mean.device)
            
        # Batch update
        self.new_sum += values.sum(dim=0)
        self.new_sum_sq += torch.square(values).sum(dim=0)
        self.new_count += values.shape[0]

    def update(self):
        new_count = self.count + self.new_count
        if self.new_count == 0:
            return

        new_mean = self.new_sum / self.new_count
        new_mean_sq = self.new_sum_sq / self.new_count
        w_old = self.count / new_count
        w_new = self.new_count / new_count
        
        # If we have a sequential model new_mean and new_std will have size obs_shape * seq_length
        # We need to reshape them to obs_shape (seq_lenght, obs_shape) and then take the mean over the seq_length dimension
        if new_mean.shape != self.mean_val.shape:
            new_mean = new_mean.reshape(-1, self.mean_val.shape[0]).mean(dim=0)
            new_mean_sq = new_mean_sq.reshape(-1, self.mean_val.shape[0]).mean(dim=0)
        
        self.mean_val = w_old * self.mean_val.to(self._mean.device) + w_new * new_mean
        self.mean_sq = w_old * self.mean_sq.to(self._mean.device) + w_new * new_mean_sq
        self.std_val = self._compute_std(self.mean_val, self.mean_sq)
        self.count = new_count
        self.new_count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self._update(self.mean_val, self.std_val)

    def _compute_std(self, mean, mean_sq):
        var = mean_sq - torch.square(mean)
        var = torch.maximum(var, torch.tensor(0.0, device=var.device))
        std = torch.sqrt(var)
        std = torch.maximum(std, torch.tensor(self.eps, device=std.device))
        return std

    def _update(self, mean, std):
        self._mean.data.copy_(mean)
        self._std.data.copy_(std)
        
class DecayingEntropyCoeff:
    def __init__(self, initial=0.2, minimum=0.01, decay_rate=1e-6, start_steps=10000):
        self.initial = initial
        self.minimum = minimum
        self.decay_rate = decay_rate
        self.start_steps = start_steps
        self.step = 0  # Initialize step counter
        self.value = initial  # Initial value of the coefficient

    def __call__(self):
        self.step += 1
        step = self.step
        if step < self.start_steps:
            return self.initial
        decayed = self.initial * np.exp(-self.decay_rate * (step - self.start_steps))
        self.value = max(self.minimum, decayed)
        return max(self.minimum, decayed)

    def update(self):
        # For compatibility with other updaters
        return self()
    
class NormalActionNoise:
    def __init__(self, policy, action_space, scale=0.3, min_scale=0.03, decay_rate=0.000001, start_steps=10000, seed=None, device=torch.device("cpu")):
        self.scale = scale
        self.min_scale = min_scale
        self.decay_rate = decay_rate
        
        self.start_steps = start_steps
        self.policy = policy
        self.action_size = action_space.shape[0]
        # self.np_random = np.random.RandomState(seed) # Not needed for pure torch
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device

    def __call__(self, observations, steps):
        # Infer device if not set or just use self.device
        
        # Determine batch size
        if isinstance(observations, dict):
             # Assume dict obs has 'actor' or similar
             obs_for_len = list(observations.values())[0]
        else:
             obs_for_len = observations
             
        batch_size = obs_for_len.shape[0]

        if steps > self.start_steps:
            actions = self.policy(observations)
            if not isinstance(actions, torch.Tensor):
                 actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

            # Exponential decay of the scale
            scale = max(self.min_scale, self.scale * np.exp(-self.decay_rate * steps))
            
            # Generate noise on the same device as actions
            noises = torch.randn_like(actions) * scale
            actions = torch.clamp(actions + noises, -1, 1)
        else:
            # Random actions
            actions = torch.rand((batch_size, self.action_size), device=self.device) * 2.0 - 1.0
            
        return actions

    def update(self, resets):
        pass

class NoActionNoise:
    def __init__(self, policy, action_space, seed=None, start_steps=10000, device=torch.device("cpu")):
        self.start_steps = start_steps
        self.policy = policy
        self.action_size = action_space.shape[0]
        # self.np_random = np.random.RandomState(seed)
        if seed is not None:
             torch.manual_seed(seed)
        self.device = device

    def __call__(self, observations, steps):
        if isinstance(observations, dict) and "actor" in observations:
            obs_for_len = observations["actor"]
        else:
            obs_for_len = list(observations.values())[0] if isinstance(observations, dict) else observations

        batch_size = obs_for_len.shape[0]

        if steps > self.start_steps:
            actions = self.policy(observations)
            if not isinstance(actions, torch.Tensor):
                 actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = torch.rand((batch_size, self.action_size), device=self.device) * 2.0 - 1.0
        return actions

    def update(self, resets):
        pass

class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, agent, environment, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3),
        test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None,
        checkpoint_path=None, config=None
    ):
        self.config = config or NoConfig()
        self.max_steps = self.config["train"]["steps"] or steps
        self.epoch_steps = self.config["train"]["epoch_steps"] or epoch_steps
        self.save_steps = self.config["train"]["save_steps"] or save_steps
        self.test_episodes = self.config["train"]["test_episodes"] or test_episodes
        self.show_progress = self.config["train"]["show_progress"] or show_progress
        self.replace_checkpoint = self.config["train"]["replace_checkpoint"] or replace_checkpoint
        # Log the training data to tensorboard.
        self.log = self.config["train"]["log"] or log
        log_dir = log_dir or "runs"
        # Logname is dd/MM/YYYY HH:MM:SS if not provided
        self.log_name = log_name or str(time.strftime("%d-%m-%Y %H:%M:%S")).replace(" ", "_").replace(":", "-")
        self.log_dir = os.path.join(log_dir, self.log_name)
        if not os.path.exists(self.log_dir) and self.log:
            os.makedirs(self.log_dir)
        
        # Set up checkpoint path
        print("Checking checkpoint path...")
        print(self.config["train"]["checkpoint_path"])
        self.checkpoint_path = self.config["train"]["checkpoint_path"] or checkpoint_path or f"checkpoints/{time.strftime('%d-%m-%Y_%H:%M:%S')}".replace(" ", "_").replace(":", "-")
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print(f"Checkpoint path: {self.checkpoint_path}")
        
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def _save_checkpoint(self, name):
        '''Saves a checkpoint of the agent.'''
        path = self.checkpoint_path
        if os.path.isdir(path) and self.replace_checkpoint:
            for file in os.listdir(path):
                # If name is the same as the checkpoint name, delete it
                if file.startswith(name):
                    os.remove(os.path.join(path, file))
        print(f"Saving checkpoint {name} to {path}...")
        save_path = os.path.join(path, name)
        self.agent.save(save_path)
        
        # Save environment state (e.g. running mean/std)
        if hasattr(self.environment, "save"):
            self.environment.save(save_path)
        
        # Save the config file used
        if hasattr(self.config, "copy_in_file"):
             self.config.copy_in_file(os.path.join(path, "config.yaml"))
    
    def save_trainer_state(self):
        '''Saves the state of the trainer.'''
        path = self.checkpoint_path + "/trainer_state/"
        print(f"Saving trainer state to {path}...")
        os.makedirs(path, exist_ok=True)
        self.agent.save_train_state(path)
        
        # Save the config file used
        self.config.copy_in_file(path + "config.yaml")
        
    def load_trainer_state(self, path):
        '''Loads the state of the trainer.'''
        assert os.path.exists(path), f"Path {path} does not exist."
        
        self.agent.load_train_state(path)
        
        
    def run(self):
        '''Runs the main training loop.'''

        print(f"Training started at {time.strftime('%d-%m-%Y %H:%M:%S')}")

        # Start the environments.
        observations = self.environment.start()

        if isinstance(observations, dict):
            num_workers = len(list(observations.values())[0])
        else:
            num_workers = len(observations)
            
        # Use torch tensors for tracking if possible, to avoid CPU transfers
        device = self.environment.device if hasattr(self.environment, "device") else torch.device("cpu")
        scores = torch.zeros(num_workers, device=device, dtype=torch.float32)
        total_reward = torch.zeros(num_workers, device=device, dtype=torch.float32)
        lengths = torch.zeros(num_workers, device=device, dtype=torch.int32)
        
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        iterations = 0
        steps_since_save = 0
        last_epoch_time = time.time()
        
        # Trackers for epoch stats
        epoch_rewards = []
        epoch_lengths = []
        current_loss = 0.0
        
        if self.log:
            writer = SummaryWriter(self.log_dir)
        if self.show_progress:
            # Treat max_steps as iterations for the progress bar as requested
            progress_bar = tqdm(total=self.max_steps, desc="Training", unit="iter")
        
        try:
            while True:
                iterations += 1
                # Select actions.
                actions = self.agent.step(observations, self.steps)
                # assert not np.isnan(actions.sum()) # Skip nan check for speed

                # Take a step in the environments.
                observations, infos = self.environment.step(actions)
                agent_infos = self.agent.update(**infos, steps=self.steps)
                
                # Extract loss for progress bar
                if "loss" in agent_infos:
                    current_loss = agent_infos["loss"]
                elif "critic_loss" in agent_infos:
                    current_loss = agent_infos["critic_loss"]
                elif "actor_loss" in agent_infos:
                    current_loss = agent_infos["actor_loss"]
                    
                if self.log:
                    for key, value in agent_infos.items():
                        if isinstance(value, torch.Tensor):
                            value = value.mean().item()
                        elif isinstance(value, np.ndarray):
                            value = value.mean()
                        writer.add_scalar(f'train/{key}', value, self.steps)
                        
                # Log environment extras (e.g. detailed reward terms)
                if self.log and "log" in infos and infos["log"]:
                    for key, value in infos["log"].items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        writer.add_scalar(key, value, self.steps)

                scores += infos['rewards']
                lengths += 1
                self.steps += num_workers
                epoch_steps += 1
                steps_since_save += 1
                
                # Update the progress bar
                if self.show_progress:
                    progress_bar.update(1)

                # Check the finished episodes.
                # Vectorized reset check
                resets = infos['resets']
                if resets.any():
                    reset_indices = torch.nonzero(resets).flatten()
                    
                    # Log finished episodes
                    if self.log:
                        # We can log the mean of finished episodes in this step
                        mean_score = scores[reset_indices].mean().item()
                        mean_length = lengths[reset_indices].float().mean().item()
                        writer.add_scalar('train/episode_score', mean_score, self.steps)
                        writer.add_scalar('train/episode_length', mean_length, self.steps)
                    
                    # Update total reward for tracking
                    total_reward[reset_indices] += scores[reset_indices]
                    
                    # Store for epoch logging
                    epoch_rewards.extend(scores[reset_indices].tolist())
                    epoch_lengths.extend(lengths[reset_indices].tolist())
                    
                    # Reset trackers
                    scores[reset_indices] = 0
                    lengths[reset_indices] = 0
                    episodes += len(reset_indices)
                    
                    # Reset agent observations if needed (usually handled by env, but agent might have memory)
                    # self.agent.reset_observations(reset_indices) # This might need update for vectorized

                # End of the epoch.
                if epoch_steps >= self.epoch_steps:
                    # Evaluate the agent on the test environment.
                    if self.test_environment:
                        self._test()

                    # Log the data.
                    epochs += 1
                    current_time = time.time()
                    epoch_time = current_time - last_epoch_time
                    sps = epoch_steps / epoch_time

                    if self.log:
                        writer.add_scalar('train/epoch_time', epoch_time, self.steps)
                        writer.add_scalar('train/steps_per_second', sps, self.steps)
                        
                    if len(epoch_rewards) > 0:
                        mean_reward = sum(epoch_rewards) / len(epoch_rewards)
                        mean_length = sum(epoch_lengths) / len(epoch_lengths)
                        # Update progress bar description instead of printing
                        if self.show_progress:
                            progress_bar.set_description(f"Epoch {epochs} | Iter {iterations} | Loss: {current_loss:.4f} | Reward: {mean_reward:.2f} | SPS: {sps:.0f}")
                        else:
                            print(f"Epoch {epochs}: SPS: {sps:.2f}, Time: {epoch_time:.2f}s, Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}")
                    else:
                        if self.show_progress:
                            progress_bar.set_description(f"Epoch {epochs} | Iter {iterations} | Loss: {current_loss:.4f} | SPS: {sps:.0f}")
                        else:
                            print(f"Epoch {epochs}: SPS: {sps:.2f}, Time: {epoch_time:.2f}s (No episodes finished)")
                    
                    epoch_rewards = []
                    epoch_lengths = []
                    
                    last_epoch_time = time.time()
                    epoch_steps = 0

                # End of training.
                # Stop based on iterations to match the progress bar and user expectation
                stop_training = iterations >= self.max_steps

                # Save a checkpoint.
                if stop_training or steps_since_save >= self.save_steps:
                    name = f'step_{self.steps}'
                    self._save_checkpoint(name)
                    steps_since_save = self.steps % self.save_steps

                if stop_training:
                    # end training
                    if self.log:
                        writer.close()
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            print(f"Saving checkpoint at step {self.steps}...")
            # Save the trainer state
            self.save_trainer_state()
            print(f"Checkpoint saved at {self.checkpoint_path}")

    def _test(self):
        '''Tests the agent on the test environment.'''

        # Start the environment.
        if not hasattr(self, 'test_observations'):
            self.test_observations = self.test_environment.start()
            if isinstance(self.test_observations, dict):
                batch_size = next(iter(self.test_observations.values())).shape[0]
            else:
                batch_size = self.test_observations.shape[0]
            assert batch_size == 1

        # Test loop.
        for _ in range(self.test_episodes):
            score, length = 0, 0

            while True:
                # Select an action.
                actions = self.agent.test_step(
                    self.test_observations, self.steps)
                assert not np.isnan(actions.sum())
                # logger.store('test/action', actions, stats=True)

                # Take a step in the environment.
                self.test_observations, infos = self.test_environment.step(
                    actions)
                self.agent.test_update(**infos, steps=self.steps)

                score += infos['rewards'][0]
                length += 1

                if infos['resets'][0]:
                    break

            # Log the data.
            # logger.store('test/episode_score', score, stats=True)
            # logger.store('test/episode_length', length, stats=True)

class DistributionalValueHead(torch.nn.Module):
    def __init__(self, vmin, vmax, num_atoms, input_size, return_normalizer=None, fn=None, device=torch.device("cpu")):
        super().__init__()
        self.num_atoms = num_atoms
        self.fn = fn
        self.register_buffer("values", torch.linspace(vmin, vmax, num_atoms).float().to(device))
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')
        self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms)
        if self.fn:
            self.distributional_layer.apply(self.fn)
            
    def forward(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)