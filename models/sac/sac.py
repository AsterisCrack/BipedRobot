import os
import torch
import numpy as np
import models.ddpg.ddpg as ddpg
from models.utils import NoActionNoise

class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, model, action_space, device=torch.device("cpu"), seq_length=1, optimizer=None, entropy_coeff=0.2, gradient_clip=0, recurrent_model = False):
        self.device = device
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length
        self.optimizer = (lambda params: torch.optim.Adam(params, lr=3e-4)) if optimizer is None else optimizer
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

        self.model = model
        self.variables = [param for param in self.model.actor.parameters() if param.requires_grad]
        self.optimizer = self.optimizer(self.variables)

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
        critic_1_variables = [param for param in self.model.critic_1.parameters() if param.requires_grad]
        critic_2_variables = [param for param in self.model.critic_2.parameters() if param.requires_grad]
        critic_variables = critic_1_variables + critic_2_variables

        for var in critic_variables:
            var.requires_grad = False

        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
            
        self.optimizer.zero_grad()
        distributions = self.model.actor(observations)
        if hasattr(distributions, 'rsample_with_log_prob'):
            actions, log_probs = distributions.rsample_with_log_prob()
        else:
            actions = distributions.rsample()
            log_probs = distributions.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        values = torch.min(values_1, values_2)
        loss = (self.entropy_coeff * log_probs - values).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())

class TwinCriticSoftQLearning:
    def __init__(
        self, model, loss=None, optimizer=None, entropy_coeff=0.2, gradient_clip=0, device=torch.device("cpu"), recurrent_model = False, seq_length=1
    ):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.device = device
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length
        self.model = model
        variables_1 = [param for param in self.model.critic_1.parameters() if param.requires_grad]
        variables_2 = [param for param in self.model.critic_2.parameters() if param.requires_grad]
        self.variables = variables_1 + variables_2
        self.optimizer = self.optimizer(self.variables)

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
        if 'variables' in checkpoint:
            for v, state in zip(self.variables, checkpoint['variables']):
                v.data.copy_(state)
        else:
            print("Warning: No variables found in the checkpoint. Loading model only.")
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
            next_distributions = self.model.actor(next_observations)
            if hasattr(next_distributions, 'rsample_with_log_prob'):
                outs = next_distributions.rsample_with_log_prob()
                next_actions, next_log_probs = outs
            else:
                next_actions = next_distributions.rsample()
                next_log_probs = next_distributions.log_prob(next_actions)
            next_log_probs = next_log_probs.sum(dim=-1)
            next_values_1 = self.model.target_critic_1(
                next_observations, next_actions)
            next_values_2 = self.model.target_critic_2(
                next_observations, next_actions)
            next_values = torch.min(next_values_1, next_values_2)
            returns = rewards + discounts * (
                next_values - self.entropy_coeff * next_log_probs)

        self.optimizer.zero_grad()
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        loss_1 = self.loss(values_1, returns)
        loss_2 = self.loss(values_2, returns)
        loss = loss_1 + loss_2

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(
            loss=loss.detach(), q1=values_1.detach(), q2=values_2.detach())

class SAC(ddpg.DDPG):
    '''Soft Actor-Critic.
    SAC: https://arxiv.org/pdf/1801.01290.pdf
    '''

    def __init__(
        self, action_space, model, max_seq_length=1, num_workers=1,seed=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, recurrent_model=False, device=torch.device("cpu")
    ):
        model = model
        exploration = NoActionNoise(policy=self._policy, action_space=action_space, seed=seed) if exploration is None else exploration
        actor_updater = TwinCriticSoftDeterministicPolicyGradient(model=model, action_space=action_space, device=device) if actor_updater is None else actor_updater
        critic_updater = TwinCriticSoftQLearning(model=model, device=device) if critic_updater is None else critic_updater
        
        super().__init__(action_space=action_space, model=model, recurrent_model=recurrent_model, max_seq_length=max_seq_length, num_workers=num_workers, seed=seed, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater, device=device)

    def _stochastic_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        # Send observations to device
        observations = torch.as_tensor(observations, dtype=torch.float32)
        return self._stochastic_actions(observations).cpu().numpy()

    def _greedy_actions(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
        with torch.no_grad():
            return self.model.actor(observations).loc