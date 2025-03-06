import numpy as np
import torch
import os
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

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




class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, agent, environment, test_environment=None, steps=int(1e7), epoch_steps=int(5e3), save_steps=int(5e3),
        test_episodes=5, show_progress=True, replace_checkpoint=False, log=True, log_dir=None, log_name=None,
        chekpoint_path=None
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
        self.checkpoint_path = chekpoint_path if chekpoint_path is not None else f"models/mpo/checkpoints/{time.strftime('%d-%m-%Y_%H:%M:%S')}".replace(" ", "_").replace(":", "-")
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