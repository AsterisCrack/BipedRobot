import os
import torch
import numpy as np
import models.ddpg.ddpg as ddpg
from models.utils import NoActionNoise, Buffer

class DistributionalDeterministicPolicyGradient:
    def __init__(self, model, action_space, device=torch.device("cpu"), seq_length=1, optimizer=None, gradient_clip=0, recurrent_model = False):
        self.device = device
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

        self.model = model
        self.variables = [
            param for param in self.model.actor.parameters() if param.requires_grad]
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
        critic_variables = [param for param in self.model.critic.parameters() if param.requires_grad]

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        actions = self.model.actor(observations)
        value_distributions = self.model.critic(observations, actions)
        values = value_distributions.mean()
        loss = -values.mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        for var in critic_variables:
            var.requires_grad = True

        return dict(loss=loss.detach())

class DistributionalDeterministicQLearning:
    def __init__(self, model, optimizer=None, gradient_clip=0, device=torch.device("cpu"), recurrent_model = False, seq_length=1):
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3))
        self.gradient_clip = gradient_clip

        self.model = model
        self.variables = [
            param for param in self.model.critic.parameters() if param.requires_grad]
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
            next_actions = self.model.target_actor(next_observations)
            next_value_distributions = self.model.target_critic(
                next_observations, next_actions)
            values = next_value_distributions.values
            returns = rewards[:, None] + discounts[:, None] * values
            targets = next_value_distributions.project(returns)

        self.optimizer.zero_grad()
        value_distributions = self.model.critic(observations, actions)
        log_probabilities = torch.nn.functional.log_softmax(
            value_distributions.logits, dim=-1)
        loss = -(targets * log_probabilities).sum(dim=-1).mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach())

class D4PG(ddpg.DDPG):
    '''Distributed Distributional Deterministic Policy Gradients.
    D4PG: https://arxiv.org/pdf/1804.08617.pdf
    '''

    def __init__(
        self, action_space, model, max_seq_length=1, num_workers=1,seed=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, recurrent_model=False, device=torch.device("cpu")
    ):
        model = model
        self.replay = Buffer(return_steps=5) if replay is None else replay
        actor_updater = DistributionalDeterministicPolicyGradient(model=model, action_space=action_space, device=device) if actor_updater is None else actor_updater
        critic_updater = DistributionalDeterministicQLearning(model=model, device=device) if critic_updater is None else critic_updater
        
        super().__init__(action_space=action_space, model=model, recurrent_model=recurrent_model, max_seq_length=max_seq_length, num_workers=num_workers, seed=seed, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater, device=device)