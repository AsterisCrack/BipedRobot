import numpy as np
import torch
import os
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

class Model:
    def __init__(self, env, model_path=None, device=torch.device("cpu")):
        # Load the saved model
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.to(device)
        
        # Initialize environment with rendering
        self.env = env
        
        self.device = device
        self.trainer = Trainer(self.agent, self.env)
        
    def step(self, observation):
        action = self.model.actor.forward(torch.tensor(observation).float()).sample().numpy()
        return action
    
    def train(self, seed=42, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3), test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None, checkpoint_path=None):
        
        # Initialize trainer
        self.trainer = Trainer(self.agent, self.env, test_environment, steps, epoch_steps, save_steps, test_episodes, show_progress, replace_checkpoint, log, log_dir, log_name, checkpoint_path)
        
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
        steps_between_batches=50, seed=None
    ):
        self.full_max_size = size
        self.return_steps = return_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches
        self.np_random = np.random.RandomState(seed)
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
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.max_size = self.full_max_size // self.num_workers
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val
        # Accumulate values for n-step returns.
        if self.return_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def accumulate_n_steps(self, kwargs):
        rewards = kwargs['rewards']
        next_observations = kwargs['next_observations']
        discounts = kwargs['discounts']
        masks = np.ones(self.num_workers, np.float32)
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
            self.buffers['next_observations'][index] = (
                (1 - masks)[:, None] *
                self.buffers['next_observations'][index] +
                masks[:, None] * next_observations)

    def get(self, *keys, steps):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}

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
    
class MeanStd(torch.nn.Module):
    def __init__(self, mean=0, std=1, clip=None, shape=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.clip = clip
        self.count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self.new_count = 0
        self.eps = 1e-2
        
        if shape:   
            if isinstance(self.mean, (int, float)):
                self.mean = np.full(shape, self.mean, np.float32)
            else:
                self.mean = np.array(self.mean, np.float32)
            if isinstance(self.std, (int, float)):
                self.std = np.full(shape, self.std, np.float32)
            else:
                self.std = np.array(self.std, np.float32)
            self.mean_sq = np.square(self.mean)
            self._mean = torch.nn.Parameter(torch.as_tensor(
                self.mean, dtype=torch.float32), requires_grad=False)
            self._std = torch.nn.Parameter(torch.as_tensor(
                self.std, dtype=torch.float32), requires_grad=False)

    def forward(self, val):
        with torch.no_grad():
            val = (val - self._mean) / self._std
            if self.clip is not None:
                val = torch.clamp(val, -self.clip, self.clip)
        return val

    def unnormalize(self, val):
        return val * self._std + self._mean

    def record(self, values):
        for val in values:
            self.new_sum += val
            self.new_sum_sq += np.square(val)
            self.new_count += 1

    def update(self):
        new_count = self.count + self.new_count
        new_mean = self.new_sum / self.new_count
        new_mean_sq = self.new_sum_sq / self.new_count
        w_old = self.count / new_count
        w_new = self.new_count / new_count
        
        # If we have a sequential model new_mean and new_std will have size obs_shape * seq_length
        # We need to reshape them to obs_shape (seq_lenght, obs_shape) and then take the mean over the seq_length dimension
        if new_mean.shape != self.mean.shape:
            new_mean = new_mean.reshape(-1, self.mean.shape[0]).mean(axis=0)
            new_mean_sq = new_mean_sq.reshape(-1, self.mean.shape[0]).mean(axis=0)
        
        self.mean = w_old * self.mean + w_new * new_mean
        self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
        self.std = self._compute_std(self.mean, self.mean_sq)
        self.count = new_count
        self.new_count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self._update(self.mean.astype(np.float32), self.std.astype(np.float32))

    def _compute_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    def _update(self, mean, std):
        self._mean.data.copy_(torch.as_tensor(self.mean, dtype=torch.float32))
        self._std.data.copy_(torch.as_tensor(self.std, dtype=torch.float32))

class NormalActionNoise:
    def __init__(self, policy, action_space, scale=0.1, start_steps=10000, seed=None):
        self.scale = scale
        self.start_steps = start_steps
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
            noises = self.scale * self.np_random.normal(size=actions.shape)
            actions = (actions + noises).astype(np.float32)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets):
        pass

class NoActionNoise:
    def __init__(self, policy, action_space, seed=None, start_steps=10000):
        self.start_steps = start_steps
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-np.pi, np.pi, shape)
        return actions

    def update(self, resets):
        pass
    
class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, agent, environment, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3),
        test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None,
        checkpoint_path=None
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        # Log the training data to tensorboard.
        self.log = log
        log_dir = log_dir if log_dir is not None else "runs"
        # Logname is dd/MM/YYYY HH:MM:SS if not provided
        self.log_name = log_name if log_name is not None else str(time.strftime("%d-%m-%Y %H:%M:%S")).replace(" ", "_").replace(":", "-")
        self.log_dir = os.path.join(log_dir, self.log_name)
        
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else f"models/mpo/checkpoints/{time.strftime('%d-%m-%Y_%H:%M:%S')}".replace(" ", "_").replace(":", "-")
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def _save_checkpoint(self, name):
        '''Saves a checkpoint of the agent.'''
        path = self.checkpoint_path
        if os.path.isdir(path) and self.replace_checkpoint:
            for file in os.listdir(path):
                # If name is the same as the checkpoint name, delete it
                if file.startswith(name):
                    os.remove(os.path.join(path, file))
        save_path = os.path.join(path, name)
        self.agent.save(save_path)
    
    def save_trainer_state(self):
        '''Saves the state of the trainer.'''
        path = self.checkpoint_path + "/trainer_state/"
        print(f"Saving trainer state to {path}...")
        os.makedirs(path, exist_ok=True)
        self.agent.save_train_state(path)
        
    def load_trainer_state(self, path):
        '''Loads the state of the trainer.'''
        assert os.path.exists(path), f"Path {path} does not exist."
        
        self.agent.load_train_state(path)
        
        
    def run(self):
        '''Runs the main training loop.'''

        print(f"Training started at {time.strftime('%d-%m-%Y %H:%M:%S')}")

        # Start the environments.
        observations = self.environment.start()

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        total_reward = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0
        last_epoch_time = time.time()
        
        if self.log:
            writer = SummaryWriter(self.log_dir)
        progress_bar = tqdm(total=self.max_steps, desc="Training", unit="step")
        
        try:
            while True:
                # Select actions.
                actions = self.agent.step(observations, self.steps)
                assert not np.isnan(actions.sum())

                # Take a step in the environments.
                observations, infos = self.environment.step(actions)
                agent_infos = self.agent.update(**infos, steps=self.steps)
                if self.log:
                    for key, value in agent_infos.items():
                        if isinstance(value, np.ndarray):
                            value = value.mean()
                        writer.add_scalar(f'train/{key}', value, self.steps)

                scores += infos['rewards']
                lengths += 1
                self.steps += num_workers
                epoch_steps += num_workers
                steps_since_save += num_workers
                
                # Update the progress bar
                progress_bar.update(num_workers)

                # Check the finished episodes.
                for i in range(num_workers):
                    if infos['resets'][i]:
                        # Reset the observations for the worker.
                        self.agent.reset_observations(i)
                        if self.log:
                            writer.add_scalar('train/episode_score', scores[i], self.steps)
                            writer.add_scalar('train/episode_length', lengths[i], self.steps)
                        total_reward[i] += scores[i]
                        scores[i] = 0
                        lengths[i] = 0
                        episodes += 1

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
                        writer.add_scalar('train/epoch_mean_reward', np.mean(total_reward / episodes), self.steps)
                    
                    last_epoch_time = time.time()
                    epoch_steps = 0

                # End of training.
                stop_training = self.steps >= self.max_steps

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
            assert len(self.test_observations) == 1

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