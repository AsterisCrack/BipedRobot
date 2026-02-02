import os
import torch
import numpy as np
from gymnasium import spaces
from algorithms.utils import Buffer, NormalActionNoise, to_tensor
from envs.isaaclab.mdp.symmetry import compute_symmetric_states

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
        
    def __call__(self, observations, critic_observations=None, steps=0):
        critic_variables = [p for p in self.model.critic.parameters() if p.requires_grad]

        for var in critic_variables:
            var.requires_grad = False

        self.optimizer.zero_grad()
        
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
            if critic_observations is not None:
                critic_observations = critic_observations.reshape(critic_observations.shape[0], self.seq_length, -1)
                critic_observations = critic_observations.transpose(0, 1)
            
        actions = self.model.actor(observations)
        
        # Use critic_observations if provided (for privileged critic)
        critic_obs = critic_observations if critic_observations is not None else observations
        values = self.model.critic(critic_obs, actions)
        loss = -values.mean()

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step(loss.item(), steps=steps)

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
        # print(f"Loaded mpo model from {path}")
        
    def __call__(
        self, observations, actions, next_observations, rewards, discounts, next_actor_observations=None, steps=0
    ):
        with torch.no_grad():
            if self.recurrent_model:
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                
                next_observations = next_observations.reshape(next_observations.shape[0], self.seq_length, -1)
                next_observations = next_observations.transpose(0, 1)

                if next_actor_observations is not None:
                    next_actor_observations = next_actor_observations.reshape(next_actor_observations.shape[0], self.seq_length, -1)
                    next_actor_observations = next_actor_observations.transpose(0, 1)
                
            # If actor and critic have different observation spaces, 
            # use next_actor_observations for the target actor.
            actor_obs = next_actor_observations if next_actor_observations is not None else next_observations
            next_actions = self.model.target_actor(actor_obs)
            next_values = self.model.target_critic(
                next_observations, next_actions)
            returns = rewards + discounts * next_values

        self.optimizer.zero_grad()
        values = self.model.critic(observations, actions)
        loss = self.loss(values, returns)

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step(loss.item(), steps=steps)

        return dict(loss=loss.detach(), q=values.detach())

class DDPG():
    '''Deep Deterministic Policy Gradient.
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    '''
    def __init__(
        self, action_space, model, recurrent_model=False, max_seq_length=1, num_workers=1, seed=None, replay=None, actor_updater=None, critic_updater=None, exploration=None, actor_optimizer=None, critic_optimizer=None, device=torch.device("cpu"), config=None
    ):
        self.config = config
        self.model = model
        
        # Determine if symmetry augmentation is enabled from config
        self.use_symmetry = False
        if self.config:
            # Check pydantic model structure generally found in this project
            if hasattr(self.config, "train") and hasattr(self.config.train, "symmetry_augmentation"):
                 self.use_symmetry = self.config.train.symmetry_augmentation
            # Fallback for dict access
            elif hasattr(self.config, "__getitem__"):
                 try:
                     self.use_symmetry = self.config["train"]["symmetry_augmentation"]
                 except:
                     pass
        self.recurrent_model = recurrent_model
        self.max_seq_length = max_seq_length
        self.observation_memory = torch.zeros((max_seq_length, num_workers, model.actor.observation_size), dtype=torch.float32).to(device) if recurrent_model else None
        self.action_memory = np.zeros((max_seq_length, num_workers, model.actor.action_size), dtype=np.float32) if recurrent_model else None
        self.device = device
        self.num_workers = num_workers
        self.replay = Buffer(return_steps=5, seed=seed, device=device, config=config) if replay is None else replay
        
        # Setup Exploration
        if exploration is not None:
            self.exploration = exploration
        else:
            # Default options
            noise_kwargs = {}
            if self.config:
                # Check directly in pydantic model structure if applicable
                model_cfg = getattr(self.config, "model", None)
                if model_cfg:
                     expl_cfg = getattr(model_cfg, "exploration", None)
                     if expl_cfg:
                         # It is likely a pydantic model or dict
                         if hasattr(expl_cfg, "dict"): # Pydantic v1
                             noise_kwargs = expl_cfg.dict()
                         elif hasattr(expl_cfg, "model_dump"): # Pydantic v2
                             noise_kwargs = expl_cfg.model_dump()
                         elif isinstance(expl_cfg, dict):
                             noise_kwargs = expl_cfg
            
            # Filter None values just in case
            noise_kwargs = {k: v for k, v in noise_kwargs.items() if v is not None}
            self.exploration = NormalActionNoise(self._policy, action_space, seed=seed, device=device, **noise_kwargs)

        self.actor_updater = DeterministicPolicyGradient(model=model, device=device, optimizer=actor_optimizer) if actor_updater is None else actor_updater
        self.critic_updater = DeterministicQLearning(model=model, device=device, optimizer=critic_optimizer) if critic_updater is None else critic_updater
        
        self.is_dict_obs = isinstance(model.obs_space, spaces.Dict)
        if self.is_dict_obs:
            self.keys = ('observations_actor', 'observations_critic', 'actions', 
                         'next_observations_actor', 'next_observations_critic', 'rewards', 'discounts')
        else:
            self.keys = ('observations', 'actions', 'next_observations', 'rewards', 'discounts')
            
        self.env = None # Environment reference for symmetry

    def set_env(self, env):
        self.env = env

    def save(self, path):
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        path = path + '.pt'
        self.model.load_state_dict(torch.load(path))
    
    def save_train_state(self, path):
        # print(f"Saving mpo model to {path}")
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
        # print(f"Loaded mpo model from {path}")
    
    def reset_observations(self, workers):
        '''Reset the observations for the recurrent model.'''
        if self.recurrent_model:
            self.observation_memory[:, workers, :] = 0
            self.action_memory[:, workers, :] = 0

    def step(self, observations, steps):
        # Get actions from the actor and exploration method.
        actions = self.exploration(observations, steps)
        self.last_actions = actions.clone()  # Use clone for tensors
        
        # Keep some values for the next update.
        if not self.recurrent_model:
            self.last_observations = observations # Store full dict/array
        else:
            # First, store action into last actions
            # Note: Tensors equivalent of np.roll is tricky in-place, better to re-construct or use indexing
            # self.action_memory = np.roll(self.action_memory, shift=-1, axis=0) # Removed for GPU support
            
            # Recurrent support needs full review for GPU-only support
            # For now, we assume simple case or implement circular buffer later
            # self.last_observations = self.observation_memory.cpu().numpy().copy()
            pass
            
            # Recurrent logic deferred for full GPU overhaul

        return actions


    def test_step(self, observations, steps):
        # Greedy actions for testing.
        return self._greedy_actions(observations)

    def test_update(self, **kwargs):
        resets = kwargs.get('resets')
        if resets is not None:
            self.reset_observations(resets)

    def update(self, observations, rewards, resets, terminations, steps, **kwargs):
        if self.recurrent_model:
            observations = torch.as_tensor(observations, dtype=torch.float32)
            observations = torch.cat((self.observation_memory.clone(), observations.unsqueeze(0)), dim=0)
            observations = observations[1:] #.cpu().numpy()

            observations = observations.transpose(1, 0, 2)
            observations = observations.reshape(observations.shape[0], -1)
            
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Symmetry Augmentation
        if self.use_symmetry and self.env is not None and compute_symmetric_states is not None:
            # Unwrap to get the underlying DirectRLEnv which has the config
            real_env = self.env.env if hasattr(self.env, "env") else self.env
            
            # Check if symmetry is applicable (BipedEnv usually)
            # We wrap non-dict obs if necessary, though BipedEnv usually returns dict
            obs_in = self.last_observations
            next_obs_in = observations
            
            is_dict = isinstance(obs_in, dict)
            
            if not is_dict:
                # Wrap in dict for symmetry function
                obs_in = {"policy": obs_in}
                next_obs_in = {"policy": next_obs_in}
            
            # Compute Symmetry
            # Note: We catch errors to avoid crashing training if symmetry fails for some reason (e.g. mismatch dims)
            try:
                with torch.no_grad():
                     obs_aug, act_aug = compute_symmetric_states(real_env, obs=obs_in, actions=self.last_actions)
                     next_obs_aug, _ = compute_symmetric_states(real_env, obs=next_obs_in, actions=None)
                     
                     if obs_aug is not None and act_aug is not None and next_obs_aug is not None:
                         # Extract the symmetric part (second half)
                         # obs_aug is [2*N, ...], we want [N:, ...]
                         
                         if is_dict:
                             sym_obs = {k: v[v.shape[0]//2:] for k, v in obs_aug.items() if v is not None}
                             sym_next_obs = {k: v[v.shape[0]//2:] for k, v in next_obs_aug.items() if v is not None}
                         else:
                             # Unwrap
                             v = obs_aug["policy"]
                             sym_obs = v[v.shape[0]//2:]
                             nv = next_obs_aug["policy"]
                             sym_next_obs = nv[nv.shape[0]//2:]
                        
                         batch_size = self.last_actions.shape[0]
                         sym_actions = act_aug[batch_size:]
                         
                         # Store Symmetric transition
                         self.replay.store(
                            observations=sym_obs, actions=sym_actions,
                            next_observations=sym_next_obs, rewards=rewards, resets=resets,
                            terminations=terminations)
            except Exception as e:
                print(f"Symmetry augmentation skipped due to error: {e}")
                pass

        # Prepare to update the normalizers.
        if self.is_dict_obs:
            if hasattr(self.model, 'actor_observation_normalizer') and self.model.actor_observation_normalizer:
                self.model.actor_observation_normalizer.record(self.last_observations['actor'])
            if hasattr(self.model, 'critic_observation_normalizer') and self.model.critic_observation_normalizer:
                self.model.critic_observation_normalizer.record(self.last_observations['critic'])
        elif self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)

        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        infos = dict()
        if self.replay.ready(steps):
            infos = self._update(steps)

        self.exploration.update(resets)
        
        return infos
        
    def _ensure_actor_tensor(self, observations):
        if self.is_dict_obs and isinstance(observations, dict):
            observations = observations["actor"]
        return to_tensor(observations, self.device)

    def _greedy_actions(self, observations):
        observations = self._ensure_actor_tensor(observations)
        with torch.no_grad():
            return self.model.actor(observations)

    def _policy(self, observations):
        return self._greedy_actions(observations)

    def _update(self, steps):
        actor_loss = 0
        critic_loss = 0
        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*self.keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(steps=steps, **batch)

            actor_loss += infos['actor']['loss']
            critic_loss += infos['critic']['loss']

        actor_loss /= self.replay.batch_iterations
        critic_loss /= self.replay.batch_iterations
        
        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if hasattr(self.model, 'actor_observation_normalizer') and self.model.actor_observation_normalizer:
            self.model.actor_observation_normalizer.update()
        if hasattr(self.model, 'critic_observation_normalizer') and self.model.critic_observation_normalizer:
            self.model.critic_observation_normalizer.update()
            
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        self.model.update_targets()
        return dict(
            actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach())
        
    def _update_actor_critic(self, **kwargs):
        steps = kwargs.pop('steps', 0)
        if self.is_dict_obs:
            observations = kwargs['observations_actor'] 
            critic_observations = kwargs['observations_critic']
            actions = kwargs['actions']
            next_observations = kwargs['next_observations_actor']
            next_critic_observations = kwargs['next_observations_critic']
            rewards = kwargs['rewards']
            discounts = kwargs['discounts']
        
            critic_infos = self.critic_updater(
                critic_observations, actions, next_critic_observations, 
                rewards, discounts, 
                next_actor_observations=next_observations,
                steps=steps
            )
            actor_infos = self.actor_updater(observations, critic_observations=critic_observations, steps=steps)
        else:
            observations = kwargs['observations']
            actions = kwargs['actions']
            next_observations = kwargs['next_observations']
            rewards = kwargs['rewards']
            discounts = kwargs['discounts']
            
            observations = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            next_observations = torch.as_tensor(next_observations, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
            discounts = torch.as_tensor(discounts, dtype=torch.float32).to(self.device)
            
            critic_infos = self.critic_updater(observations, actions, next_observations, rewards, discounts, steps=steps)
            actor_infos = self.actor_updater(observations, steps=steps)

        return dict(critic=critic_infos, actor=actor_infos)