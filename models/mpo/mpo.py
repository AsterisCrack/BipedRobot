import os
import torch
import numpy as np
from models.utils import Buffer

FLOAT_EPSILON = 1e-8

class MaximumAPosterioriPolicyOptimization:
    def __init__(
        self, model, action_space, device=torch.device("cpu"), seq_length=1 , num_samples=20, epsilon=1e-1, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-6, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        actor_optimizer=None, dual_optimizer=None, gradient_clip=0, recurrent_model = False,
        config=None
    ):
        self.config = config
        self.device = device
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.seq_length = seq_length
        if self.config is not None:
            lr = self.config["model"]["actor_lr"]
        self.actor_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=lr))
        self.dual_optimizer = dual_optimizer or (
            lambda params: torch.optim.Adam(params, lr=lr))
        self.gradient_clip = gradient_clip
        self.recurrent_model = recurrent_model
        
        
        # Init the model and the actor optimizer
        self.model = model
        self.actor_variables = [p for p in self.model.actor.parameters() if p.requires_grad]
        self.actor_optimizer = self.actor_optimizer(self.actor_variables)

        # Dual variables.
        self.dual_variables = []
        self.log_temperature = torch.nn.Parameter(torch.as_tensor(
            [self.initial_log_temperature], dtype=torch.float32))
        self.dual_variables.append(self.log_temperature)
        shape = [action_space.shape[0]] if self.per_dim_constraining else [1]
        self.log_alpha_mean = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_mean, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_mean)
        self.log_alpha_std = torch.nn.Parameter(torch.full(
            shape, self.initial_log_alpha_std, dtype=torch.float32))
        self.dual_variables.append(self.log_alpha_std)
        if self.action_penalization:
            self.log_penalty_temperature = torch.nn.Parameter(torch.as_tensor(
                [self.initial_log_temperature], dtype=torch.float32))
            self.dual_variables.append(self.log_penalty_temperature)
        self.dual_optimizer = self.dual_optimizer(self.dual_variables)
        
        # Move dual variables to the same device as the model
        self.log_temperature = self.log_temperature.to(device)
        self.log_alpha_mean = self.log_alpha_mean.to(device)
        self.log_alpha_std = self.log_alpha_std.to(device)
        
    def save_train_state(self, path):
        # Save the model and the dual variables
        # Also save optimizers
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'dual_variables': [v.detach() for v in self.dual_variables],
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'dual_optimizer': self.dual_optimizer.state_dict()
        }, path)
        print(f"Saved mpo model to {path}")
    
    def load_train_state(self, path):
        # Load the model and the dual variables
        # Also load optimizers
        path = path + '.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        for v, state in zip(self.dual_variables, checkpoint['dual_variables']):
            v.data.copy_(state)
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.dual_optimizer.load_state_dict(checkpoint['dual_optimizer'])
        print(f"Loaded mpo model from {path}")
        
    def __call__(self, observations):
        def parametric_kl_and_dual_losses(kl, alpha, epsilon):
            kl_mean = kl.mean(dim=0)
            kl_loss = (alpha.detach() * kl_mean).sum()
            alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
            return kl_loss, alpha_loss

        def weights_and_temperature_loss(q_values, epsilon, temperature):
            tempered_q_values = q_values.detach() / temperature
            
            weights = torch.nn.functional.softmax(tempered_q_values, dim=0)
            weights = weights.detach()

            # Temperature loss (dual of the E-step).
            q_log_sum_exp = torch.logsumexp(tempered_q_values, dim=0)
            num_actions = torch.as_tensor(
                q_values.shape[0], dtype=torch.float32)
            log_num_actions = torch.log(num_actions)
            loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
            loss = temperature * loss

            return weights, loss

        # Use independent normals to satisfy KL constraints per-dimension.
        def independent_normals(distribution_1, distribution_2=None):
            distribution_2 = distribution_2 or distribution_1
            
            mean = distribution_1.mean
            std = distribution_2.stddev
            
            dist = torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    mean, std), -1)
            return dist

        with torch.no_grad():
            self.log_temperature.data.copy_(
                torch.maximum(self.min_log_dual, self.log_temperature))
            self.log_alpha_mean.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_mean))
            self.log_alpha_std.data.copy_(
                torch.maximum(self.min_log_dual, self.log_alpha_std))
            if self.action_penalization:
                self.log_penalty_temperature.data.copy_(torch.maximum(
                    self.min_log_dual, self.log_penalty_temperature))

            if self.recurrent_model:
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                target_distributions = self.model.target_actor(observations)
                # Now undo the flattening of the observations
                observations = observations.transpose(0, 1)
                observations = observations.reshape(observations.shape[0], -1)
                
            else:
                target_distributions = self.model.target_actor(observations)
                
            actions = target_distributions.sample((self.num_samples,))

            # Tile the observations and actions for the critic.
            tiled_observations = observations.unsqueeze(0).expand(self.num_samples, *observations.shape)
            # Merge the first two dimensions for the observations and actions.
            flat_observations = tiled_observations.reshape(tiled_observations.shape[0] * tiled_observations.shape[1], *tiled_observations.shape[2:])
            flat_actions = actions.reshape(actions.shape[0] * actions.shape[1], *actions.shape[2:])
            
            # If recurrent model, we need to revert the flattening of observations
            if self.recurrent_model:
                flat_observations = flat_observations.reshape(flat_observations.shape[0], self.seq_length, -1)
                flat_observations = flat_observations.transpose(0, 1)
                
                # Do the same for observations
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                
            values = self.model.target_critic(flat_observations, flat_actions)
            values = values.view(self.num_samples, -1)

            assert isinstance(
                target_distributions, torch.distributions.normal.Normal)
            target_distributions = independent_normals(target_distributions)

        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.model.actor(observations)
        distributions = independent_normals(distributions)

        temperature = torch.nn.functional.softplus(
            self.log_temperature) + FLOAT_EPSILON
        alpha_mean = torch.nn.functional.softplus(
            self.log_alpha_mean) + FLOAT_EPSILON
        alpha_std = torch.nn.functional.softplus(
            self.log_alpha_std) + FLOAT_EPSILON
        weights, temperature_loss = weights_and_temperature_loss(
            values, self.epsilon, temperature)

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = torch.nn.functional.softplus(
                self.log_penalty_temperature) + FLOAT_EPSILON
            diff_bounds = actions - torch.clamp(actions, -1, 1)
            action_bound_costs = -torch.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = \
                weights_and_temperature_loss(
                    action_bound_costs,
                    self.epsilon_penalty, penalty_temperature.to(self.device))
            
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist)
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist)

        # Compute the decomposed policy losses.
        
        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(
            actions).sum(dim=-1).view(self.num_samples, -1) * weights).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(
            actions).sum(dim=-1).view(self.num_samples, -1) * weights).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_std_distribution.base_dist)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions.base_dist,
                fixed_mean_distribution.base_dist)
        else:
            kl_mean = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_std_distribution)
            kl_std = torch.distributions.kl.kl_divergence(
                target_distributions, fixed_mean_distribution)

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(
            kl_mean, alpha_mean, self.epsilon_mean)
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(
            kl_std, alpha_std, self.epsilon_std)

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        loss = policy_loss + kl_loss + dual_loss

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_variables, self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(
                self.dual_variables, self.gradient_clip)
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        dual_variables = dict(
            temperature=temperature.detach(), alpha_mean=alpha_mean.detach(),
            alpha_std=alpha_std.detach())
        if self.action_penalization:
            dual_variables['penalty_temperature'] = \
                penalty_temperature.detach()

        return dict(
            dual_loss=dual_loss.detach(),
            total_loss=policy_loss.detach(),
            policy_mean_loss=policy_mean_loss.detach(),
            policy_std_loss=policy_std_loss.detach(),
            kl_mean_loss=kl_mean_loss.detach(),
            kl_std_loss=kl_std_loss.detach(),
            alpha_mean_loss=alpha_mean_loss.detach(),
            alpha_std_loss=alpha_std_loss.detach(),
            temperature_loss=temperature_loss.detach(),
            **dual_variables)

class ExpectedSARSA:
    def __init__(
        self, model, num_samples=20, batch_size=1, loss=None, optimizer=None, gradient_clip=0, device=torch.device("cpu"), recurrent_model = False, seq_length=1
    ):
        self.device = device
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.gradient_clip = gradient_clip

        self.model = model
        self.variables = [p for p in self.model.critic.parameters() if p.requires_grad]
        self.optimizer = self.optimizer(self.variables)
        self.recurrent_model = recurrent_model

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
        # Approximate the expected next values.
        with torch.no_grad():
            # If recurrent model, we need to revert the flattening of observations
            if self.recurrent_model:
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                
                next_observations = next_observations.reshape(next_observations.shape[0], self.seq_length, -1)
                next_observations = next_observations.transpose(0, 1)
                next_target_distributions = self.model.target_actor(next_observations)
                # Undo the flattening of the observations
                next_observations = next_observations.transpose(0, 1)
                next_observations = next_observations.reshape(next_observations.shape[0], -1)
            else:
                next_target_distributions = self.model.target_actor(next_observations)
                
            next_actions = next_target_distributions.rsample(
                (self.num_samples,))
        
            # Merge the first two dimensions
            next_actions = next_actions.reshape(next_actions.shape[0] * next_actions.shape[1], *next_actions.shape[2:])
            # Repeat the observations for each sample, creating a batch.
            next_observations = next_observations.unsqueeze(0).expand(self.num_samples, *next_observations.shape)
            # Merge the first two dimensions
            next_observations = next_observations.reshape(next_observations.shape[0] * next_observations.shape[1], *next_observations.shape[2:])
            # If recurrent model, we need to get the correct observations for the critic
            if self.recurrent_model:
                next_observations = next_observations.reshape(next_observations.shape[0], self.seq_length, -1)
                next_observations = next_observations.transpose(0, 1)
                
            next_values = self.model.target_critic(next_observations, next_actions)
            next_values = next_values.view(self.num_samples, -1)
            next_values = next_values.mean(dim=0)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(returns, values)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), q=values.detach())

class MPO():
    '''Maximum a Posteriori Policy Optimization.
    MPO: https://arxiv.org/pdf/1806.06920.pdf
    MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
    '''

    def __init__(
        self, action_space, model, recurrent_model=False, max_seq_length=1, num_workers=1, seed=None, replay=None, actor_updater=None, critic_updater=None, actor_optimizer=None,dual_optimizer=None, critic_optimizer=None, device=torch.device("cpu"), config=None
    ):
        self.model = model
        self.recurrent_model = recurrent_model
        self.max_seq_length = max_seq_length
        self.observation_memory = torch.zeros((max_seq_length, num_workers, model.actor.observation_size), dtype=torch.float32).to(device) if recurrent_model else None
        self.action_memory = np.zeros((max_seq_length, num_workers, model.actor.action_size), dtype=np.float32) if recurrent_model else None
        self.device = device
        self.num_workers = num_workers
        self.replay = Buffer(return_steps=5, seed=seed, device=device, config=config) if replay is None else replay
        self.actor_updater = MaximumAPosterioriPolicyOptimization(self.model, action_space, device, recurrent_model = recurrent_model, seq_length=max_seq_length, actor_optimizer=actor_optimizer, dual_optimizer=dual_optimizer, config=config) \
            if actor_updater is None else actor_updater
        self.critic_updater = ExpectedSARSA(self.model, batch_size=num_workers, recurrent_model = recurrent_model, seq_length=max_seq_length, optimizer=critic_optimizer, config=config) \
            if critic_updater is None else critic_updater
        
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
        actions = self._step(observations)
        actions = actions.cpu().numpy()
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
        # Sample actions for testing.
        return self._test_step(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        # If the model is recurrent we need to change the observations to include it correctly in the replay.
        if self.recurrent_model:
            # We need to reshape the observations to be of size (sequence_length, batch_size, observation_size)
            observations = torch.as_tensor(observations, dtype=torch.float32)
            observations = torch.cat((self.observation_memory.cpu().clone(), observations.unsqueeze(0)), dim=0)
            observations = observations[1:].cpu().numpy()
            
            # Now, observations has 3 dimensions: (sequence_length, batch_size, observation_size)
            # But buffer only stores 2 dimensional data: (batch_size, observation_size)
            # So we need to combine dimensions 0 and 2
            observations = observations.transpose(1, 0, 2)
            observations = observations.reshape(observations.shape[0], -1)

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
            
        return infos

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        if self.recurrent_model:
            # Update the list of last observations for the recurrent model. Last observations must be of size:
            # (sequence_length, batch_size, observation_size)
            self.observation_memory = torch.cat((self.observation_memory, observations.unsqueeze(0)), dim=0)
            self.observation_memory = self.observation_memory[1:]
            with torch.no_grad():
                # Get the action from the model
                actions = self.model.actor(self.observation_memory).sample()
                return actions

        else: 
            with torch.no_grad():
                return self.model.actor(observations).sample()
            
    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            actions = self.model.actor(observations).sample()
            return actions

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')
        
        actor_dual_loss = 0
        actor_loss = 0
        critic_loss = 0
        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            actor_dual_loss += infos['actor']['dual_loss']
            actor_loss += infos['actor']['total_loss']
            critic_loss += infos['critic']['loss']
        actor_dual_loss /= self.replay.batch_iterations
        actor_loss /= self.replay.batch_iterations
        critic_loss /= self.replay.batch_iterations

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
        
        return dict(
            actor_dual_loss=actor_dual_loss.detach(),
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

