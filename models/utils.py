import numpy as np
import torch
import os
import time

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
        test_episodes=5, show_progress=True, replace_checkpoint=False,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint

        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def run(self):
        '''Runs the main training loop.'''

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations = self.environment.start()

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        total_scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0

        while True:
            # Select actions.
            actions = self.agent.step(observations, self.steps)
            assert not np.isnan(actions.sum())
            # logger.store('train/action', actions, stats=True)

            # Take a step in the environments.
            observations, infos = self.environment.step(actions)
            self.agent.update(**infos, steps=self.steps)

            scores += infos['rewards']
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                # logger.show_progress(self.steps, self.epoch_steps, self.max_steps)
                #print(f"Step: {self.steps}/{self.max_steps}, Epoch step: {epoch_steps}/{self.epoch_steps}, Epoch: {epochs}, Episode: {episodes}")
                pass
            # Check the finished episodes.
            for i in range(num_workers):
                if infos['resets'][i]:
                    # logger.store('train/episode_score', scores[i], stats=True)
                    # logger.store('train/episode_length', lengths[i], stats=True)
                    # print(f"Episode {episodes} finished with score {scores[i]} and length {lengths[i]}")
                    total_scores[i] += scores[i]
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
                # logger.store('train/episodes', episodes)
                # logger.store('train/epochs', epochs)
                # logger.store('train/seconds', current_time - start_time)
                # logger.store('train/epoch_seconds', epoch_time)
                # logger.store('train/epoch_steps', epoch_steps)
                # logger.store('train/steps', self.steps)
                # logger.store('train/worker_steps', self.steps // num_workers)
                # logger.store('train/steps_per_second', sps)
                # logger.dump()
                print(f"Epoch time: {epoch_time}, Steps per second: {sps}, Mean score: {total_scores / episodes}")
                last_epoch_time = time.time()
                epoch_steps = 0

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = "models/mpo/checkpoints"
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f'step_{self.steps}'
                save_path = os.path.join(path, checkpoint_name)
                self.agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

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