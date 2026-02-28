import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, device=torch.device("cpu"), gamma=0.99, gae_lambda=0.95):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
        
    def reset(self):
        self.observations = torch.zeros((self.num_steps, self.num_envs) + self.obs_shape, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.action_shape, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.advantages = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        
        self.step = 0
        
    def add(self, obs, action, reward, done, value, log_prob):
        if self.step >= self.num_steps:
            raise IndexError("Buffer is full")
            
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        
        self.step += 1
        
    def compute_returns_and_advantage(self, last_value, last_done):
        last_value = last_value.to(self.device)
        last_done = last_done.to(self.device)
        
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
                
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            
        self.returns = self.advantages + self.values
        
    def get_generator(self, num_mini_batches=None, mini_batch_size=None):
        batch_size = self.num_steps * self.num_envs
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batches, "Batch size is smaller than number of mini batches"
            mini_batch_size = batch_size // num_mini_batches
            
        sampler = torch.randperm(batch_size, device=self.device)
        
        flatten_obs = self.observations.view(-1, *self.obs_shape)
        flatten_actions = self.actions.view(-1, *self.action_shape)
        flatten_log_probs = self.log_probs.view(-1)
        flatten_advantages = self.advantages.view(-1)
        
        # Normalize advantages
        flatten_advantages = (flatten_advantages - flatten_advantages.mean()) / (flatten_advantages.std() + 1e-8)
        
        flatten_returns = self.returns.view(-1)
        flatten_values = self.values.view(-1)
        
        for i in range(0, batch_size, mini_batch_size):
            indices = sampler[i:i+mini_batch_size]
            
            obs_batch = flatten_obs[indices]
            actions_batch = flatten_actions[indices]
            log_probs_batch = flatten_log_probs[indices]
            advantages_batch = flatten_advantages[indices]
            returns_batch = flatten_returns[indices]
            values_batch = flatten_values[indices]
            
            yield obs_batch, actions_batch, log_probs_batch, advantages_batch, returns_batch, values_batch
