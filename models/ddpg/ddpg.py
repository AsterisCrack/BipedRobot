import os
import torch
import numpy as np
from models.utils import Buffer, NormalActionNoise

class DeterministicPolicyGradient:
    def __init__(self, model, device=torch.device("cpu"), seq_length=1, optimizer=None, gradient_clip=0, recurrent_model = False):
        self.device = device
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip
        
        # Init the model and the optimizer
        self.model = model
        self.variables = [p for p in model.actor.parameters() if p.requires_grad]
        self.optimizer = self.optimizer(self.variables)
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length

    def save_train_state(self, path):
        # Save the model and the dual variables
        # Also save optimizers
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'variables': [v.detach() for v in self.variables],
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Saved mpo model to {path}")
    
    def load_train_state(self, path):
        # Load the model and the dual variables
        # Also load optimizers
        path = path + '.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        for v, state in zip(self.variables, checkpoint['variables']):
            v.data.copy_(state)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded mpo model from {path}")
        
    def __call__(self, observations):
        critic_variables = [p for p in self.model.critic.parameters() if p.requires_grad]

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
            
        actions = self.model.actor(observations)
        values = self.model.critic(observations, actions)
        loss = -values.mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())

class DeterministicQLearning:
    def __init__(self, model, loss=None, optimizer=None, gradient_clip=0, device=torch.device("cpu"), recurrent_model = False, seq_length=1):
        self.device = device
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

        # Init the model and the optimizer
        self.model = model
        self.variables = [p for p in model.critic.parameters() if p.requires_grad]
        self.optimizer = self.optimizer(self.variables)
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length

    def save_train_state(self, path):
        # Save the model and the optimizer
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'variables': [v.detach() for v in self.variables],
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Saved mpo model to {path}")
        
    def load_train_state(self, path):
        # Load the model and the optimizer
        path = path + '.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        for v, state in zip(self.variables, checkpoint['variables']):
            v.data.copy_(state)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded mpo model from {path}")
        
    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        with torch.no_grad():
            if self.recurrent_model:
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                
                next_observations = next_observations.reshape(next_observations.shape[0], self.seq_length, -1)
                next_observations = next_observations.transpose(0, 1)
                
            next_actions = self.model.target_actor(next_observations)
            next_values = self.model.target_critic(
                next_observations, next_actions)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())

class DDPG():
    '''Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    '''
    def __init__(
        self, action_space, model, recurrent_model=False, max_seq_length=1, num_workers=1, seed=None, replay=None, actor_updater=None, critic_updater=None, exploration=None, actor_optimizer=None, critic_optimizer=None, device=torch.device("cpu"), config=None
    ):
        self.model = model
        self.recurrent_model = recurrent_model
        self.max_seq_length = max_seq_length
        self.observation_memory = torch.zeros((max_seq_length, num_workers, model.actor.observation_size), dtype=torch.float32).to(device) if recurrent_model else None
        self.action_memory = np.zeros((max_seq_length, num_workers, model.actor.action_size), dtype=np.float32) if recurrent_model else None
        self.device = device
        self.num_workers = num_workers
        self.replay = Buffer(return_steps=5, seed=seed, device=device, config=config) if replay is None else replay
        self.exploration = exploration or NormalActionNoise(self._policy, action_space, seed=seed)
        self.actor_updater = DeterministicPolicyGradient(model=model, device=device, optimizer=actor_optimizer) if actor_updater is None else actor_updater
        self.critic_updater = DeterministicQLearning(model=model, device=device, optimizer=critic_optimizer) if critic_updater is None else critic_updater

    def save(self, path):
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        path = path + '.pt'
        self.model.load_state_dict(torch.load(path))
    
    def save_train_state(self, path):
        print(f"Saving mpo model to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the model and the replay
        self.replay.save(path+'replay')
        # Save the model and the dual variables
        self.actor_updater.save_train_state(path+'actor')
        self.critic_updater.save_train_state(path+'critic')
        # Save the model
        self.save(path+'model')
        
    def load_train_state(self, path):
        # Load the model and the replay
        self.replay.load(path+'replay')
        # Load the model and the dual variables
        self.actor_updater.load_train_state(path+'actor')
        self.critic_updater.load_train_state(path+'critic')
        # Load the model
        self.load(path+'model')
        print(f"Loaded mpo model from {path}")
    
    def reset_observations(self, workers):
        '''Reset the observations for the recurrent model.'''
        if self.recurrent_model:
            self.observation_memory[:, workers, :] = 0
            self.action_memory[:, workers, :] = 0

    def step(self, observations, steps):
        # Get actions from the actor and exploration method.
        actions = self.exploration(torch.as_tensor(observations, dtype=torch.float32).to(self.device), steps)
        self.last_actions = actions.copy()
        
        # Keep some values for the next update.
        if not self.recurrent_model:
            self.last_observations = observations.copy()
        else:
            # First, store action into last actions
            self.action_memory = np.roll(self.action_memory, shift=-1, axis=0)  # Shift all past obs up
            self.action_memory[-1] = actions  # Insert new observations at the last position
            
            self.last_observations = self.observation_memory.cpu().numpy().copy()
            # Now, observations has 3 dimensions: (sequence_length, batch_size, observation_size)
            # But buffer only stores 2 dimensional data: (batch_size, observation_size)
            # So we need to combine dimensions 0 and 2
            self.last_observations = self.last_observations.transpose(1, 0, 2)
            self.last_observations = self.last_observations.reshape(self.last_observations.shape[0], -1)

        return actions

    def test_step(self, observations, steps):
        # Greedy actions for testing.
        return self._greedy_actions(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        if self.recurrent_model:
            observations = torch.as_tensor(observations, dtype=torch.float32)
            observations = torch.cat((self.observation_memory.cpu().clone(), observations.unsqueeze(0)), dim=0)
            observations = observations[1:].cpu().numpy()

            observations = observations.transpose(1, 0, 2)
            observations = observations.reshape(observations.shape[0], -1)
            
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        infos = dict()
        if self.replay.ready(steps):
            infos = self._update(steps)

        self.exploration.update(resets)
        
        return infos

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations)

    def _policy(self, observations):
        return self._greedy_actions(observations).cpu().numpy()

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        actor_loss = 0
        critic_loss = 0
        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            actor_loss += infos['actor']['loss']
            critic_loss += infos['critic']['loss']

        actor_loss /= self.replay.batch_iterations
        critic_loss /= self.replay.batch_iterations
        
        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        return dict(
            actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach())
        
    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        observations = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        next_observations = torch.as_tensor(next_observations, dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        discounts = torch.as_tensor(discounts, dtype=torch.float32).to(self.device)
        
        critic_infos = self.critic_updater(observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)