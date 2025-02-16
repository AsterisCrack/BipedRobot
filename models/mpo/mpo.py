import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from scipy.optimize import minimize

from models.mpo.networks import ActorCriticWithTargets
from tensorboardX import SummaryWriter

FLOAT_EPSILON = 1e-8

class Buffer:
    '''Replay storing a large number of transitions for off-policy learning
    and using n-step returns.'''

    def __init__(
        self, size=int(1e6), return_steps=1, batch_iterations=50,
        batch_size=100, discount_factor=0.99, steps_before_batches=int(1e4),
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

class MaximumAPosterioriPolicyOptimization:
    def __init__(
        self, model, action_space, num_samples=20, epsilon=1e-1, epsilon_penalty=1e-3,
        epsilon_mean=1e-3, epsilon_std=1e-6, initial_log_temperature=1.,
        initial_log_alpha_mean=1., initial_log_alpha_std=10.,
        min_log_dual=-18., per_dim_constraining=True, action_penalization=True,
        actor_optimizer=None, dual_optimizer=None, gradient_clip=0
    ):
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
        self.actor_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.dual_optimizer = dual_optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-2))
        self.gradient_clip = gradient_clip
        
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
            return torch.distributions.independent.Independent(
                torch.distributions.normal.Normal(
                    distribution_1.mean, distribution_2.stddev), -1)

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

            target_distributions = self.model.target_actor(observations)
            actions = target_distributions.sample((self.num_samples,))

            # Tile the observations and actions for the critic.
            tiled_observations = observations.unsqueeze(0).expand(self.num_samples, *observations.shape)
            # Merge the first two dimensions for the observations and actions.
            flat_observations = tiled_observations.reshape(tiled_observations.shape[0] * tiled_observations.shape[1], *tiled_observations.shape[2:])
            flat_actions = actions.reshape(actions.shape[0] * actions.shape[1], *actions.shape[2:])
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
                    self.epsilon_penalty, penalty_temperature)
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions.
        fixed_std_distribution = independent_normals(
            distributions.base_dist, target_distributions.base_dist)
        fixed_mean_distribution = independent_normals(
            target_distributions.base_dist, distributions.base_dist)

        # Compute the decomposed policy losses.
        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(
            actions).sum(dim=-1) * weights).sum(dim=0)
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
        self, model, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.gradient_clip = gradient_clip

        self.model = model
        self.variables = [p for p in self.model.critic.parameters() if p.requires_grad]
        self.optimizer = self.optimizer(self.variables)

    def __call__(
        self, observations, actions, next_observations, rewards, discounts
    ):
        # Approximate the expected next values.
        with torch.no_grad():
            next_target_distributions = self.model.target_actor(
                next_observations)
            next_actions = next_target_distributions.rsample(
                (self.num_samples,))
            # Merge the first two dimensions
            next_actions = next_actions.reshape(next_actions.shape[0] * next_actions.shape[1], *next_actions.shape[2:])  
            
            # Repeat the observations for each sample, creating a batch.    
            next_observations = next_observations.unsqueeze(0).expand(self.num_samples, *next_observations.shape)
            # Merge the first two dimensions
            next_observations = next_observations.reshape(next_observations.shape[0] * next_observations.shape[1], *next_observations.shape[2:])
            
            next_values = self.model.target_critic(
                next_observations, next_actions)
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
        self, action_space, model, seed=None, replay=None, actor_updater=None, critic_updater=None
    ):
        self.model = model
        self.replay = Buffer(return_steps=5, seed=seed) if replay is None else replay
        self.actor_updater = MaximumAPosterioriPolicyOptimization(self.model, action_space) \
            if actor_updater is None else actor_updater
        self.critic_updater = ExpectedSARSA(self.model) \
            if critic_updater is None else critic_updater

    def step(self, observations, steps):
        actions = self._step(observations)
        actions = actions.numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations, steps):
        # Sample actions for testing.
        return self._test_step(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
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
        if self.replay.ready(steps):
            self._update(steps)

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations).loc

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    #print(key + '/' + k, v.numpy())
                    pass

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations)
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)

