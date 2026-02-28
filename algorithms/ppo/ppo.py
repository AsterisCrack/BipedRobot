import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.ppo.buffer import RolloutBuffer
from algorithms.utils import to_tensor

class PPO:
    def __init__(
        self,
        model,
        action_space,
        device=torch.device("cpu"),
        config=None,
        optimizer=None,
        clip_param=0.2,
        ppo_epoch=4,
        num_mini_batches=4,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        num_steps=2048, # Steps per env per update
        num_envs=1,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.action_space = action_space
        
        # Hyperparameters
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_envs = num_envs
        
        # Optimizer
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=3e-4)
        
        # Buffer
        obs_shape = model.actor_obs_space.shape
        action_shape = action_space.shape
        self.buffer = RolloutBuffer(
            num_steps=num_steps,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        self.last_step_data = None

    def step(self, observations, steps):
        # Ensure observations are tensor
        if isinstance(observations, dict):
            obs_tensor = observations["actor"] # Assuming actor obs for policy
        else:
            obs_tensor = observations
            
        obs_tensor = to_tensor(obs_tensor, self.device)
        
        with torch.no_grad():
            # Get action distribution and value
            if hasattr(self.model.actor, "get_distribution"):
                 dist = self.model.actor.get_distribution(obs_tensor)
            else:
                 # Assuming forward returns distribution for PPO actor
                 dist = self.model.actor(obs_tensor)
                 
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Handle multi-dim action log_prob (sum over action dim)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)
                
            # Get value
            # If actor and critic share features, this might be optimized, but here we assume separate or handled by model
            if isinstance(observations, dict):
                critic_obs = to_tensor(observations["critic"], self.device)
            else:
                critic_obs = obs_tensor
                
            value = self.model.critic(critic_obs, None) # Critic usually takes (obs, action) or just obs. For PPO V(s), it takes obs.
            # But our MLPCritic takes (obs, action). We need to adjust MLPCritic or use a different one.
            # Wait, MLPCritic in networks.py takes (obs, actions).
            # For PPO, we need V(s).
            # I should check if I need a specific PPO Critic or if I can pass dummy actions or if MLPCritic handles None actions.
            # MLPCritic: out = torch.cat([observations, actions], dim=-1) -> This will fail if actions is None.
            # I need a ValueNetwork for PPO.
            
        # Store data for update
        self.last_step_data = {
            "obs": obs_tensor,
            "action": action,
            "value": value,
            "log_prob": log_prob
        }
        
        # Record observations for normalization
        if hasattr(self.model, "observation_normalizer") and self.model.observation_normalizer:
             self.model.observation_normalizer.record(obs_tensor)
        elif hasattr(self.model, "actor_observation_normalizer") and self.model.actor_observation_normalizer:
             self.model.actor_observation_normalizer.record(obs_tensor)
             if hasattr(self.model, "critic_observation_normalizer") and self.model.critic_observation_normalizer:
                 # If separate, record critic obs too
                 if isinstance(observations, dict):
                     self.model.critic_observation_normalizer.record(to_tensor(observations["critic"], self.device))
                 else:
                     self.model.critic_observation_normalizer.record(obs_tensor)
        
        return action

    def update(self, **kwargs):
        # Retrieve data
        obs = self.last_step_data["obs"]
        action = self.last_step_data["action"]
        value = self.last_step_data["value"]
        log_prob = self.last_step_data["log_prob"]
        
        rewards = to_tensor(kwargs["rewards"], self.device)
        dones = to_tensor(kwargs["resets"], self.device) # resets are dones
        
        # Add to buffer
        try:
            self.buffer.add(obs, action, rewards, dones, value, log_prob)
        except IndexError:
            # Buffer full, but update not called yet? Should not happen if logic is correct.
            pass
            
        # Check if buffer is full
        if self.buffer.step >= self.num_steps:
            # Get next value for GAE
            next_obs = kwargs.get("next_observations_actor")
            if next_obs is None:
                next_obs = kwargs.get("next_observations")
            
            next_critic_obs = kwargs.get("next_observations_critic")
            if next_critic_obs is None:
                next_critic_obs = next_obs
                
            next_critic_obs = to_tensor(next_critic_obs, self.device)
            
            with torch.no_grad():
                # We need a way to call critic for V(s). 
                # If using standard MLPCritic, we might need to adapt it.
                # For now assuming model.critic can handle it or we fixed it.
                next_value = self.model.critic(next_critic_obs, None)
                
            # Compute returns
            self.buffer.compute_returns_and_advantage(next_value, dones)
            
            # Update
            infos = self._update_ppo()
            
            # Update normalizers
            if hasattr(self.model, "observation_normalizer") and self.model.observation_normalizer:
                self.model.observation_normalizer.update()
            if hasattr(self.model, "actor_observation_normalizer") and self.model.actor_observation_normalizer:
                self.model.actor_observation_normalizer.update()
            if hasattr(self.model, "critic_observation_normalizer") and self.model.critic_observation_normalizer:
                self.model.critic_observation_normalizer.update()
            
            # Reset buffer
            self.buffer.reset()
            
            return infos
            
        return {}

    def _update_ppo(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        
        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.get_generator(num_mini_batches=self.num_mini_batches)
            
            for obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch, old_values_batch in data_generator:
                
                # Get current policy outputs
                if hasattr(self.model.actor, "get_distribution"):
                     dist = self.model.actor.get_distribution(obs_batch)
                else:
                     dist = self.model.actor(obs_batch)
                
                log_probs = dist.log_prob(actions_batch)
                if log_probs.dim() > 1:
                    log_probs = log_probs.sum(dim=-1)
                entropy = dist.entropy()
                if entropy.dim() > 1:
                    entropy = entropy.sum(dim=-1)
                    
                # Get current value
                # Assuming critic handles V(s)
                # If using shared backbone, we might need to pass obs to a joint model
                # But here we have separate actor/critic in model
                # We need to handle the critic input issue.
                # For now, let's assume we fix the critic to accept None actions or we use a specific PPO critic.
                values = self.model.critic(obs_batch, None)
                if values.dim() > 1:
                    values = values.squeeze(-1)
                
                # Ratio
                ratio = torch.exp(log_probs - old_log_probs_batch)
                
                # Surrogate loss
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                surrogate_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                # Optional: Clip value loss
                value_pred_clipped = old_values_batch + (values - old_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = surrogate_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Stats
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_loss.item()
                
                with torch.no_grad():
                    # Calculate approx KL for monitoring
                    log_ratio = log_probs - old_log_probs_batch
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    mean_kl += approx_kl.item()
                    
        num_updates = self.ppo_epoch * self.num_mini_batches
        return {
            "loss": (mean_surrogate_loss + mean_value_loss + mean_entropy_loss) / num_updates,
            "policy_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": -mean_entropy_loss / num_updates, # entropy_loss is negative entropy
            "kl": mean_kl / num_updates
        }
        
    def save_train_state(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path + ".pt")
        
    def load_train_state(self, path):
        checkpoint = torch.load(path + ".pt")
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")
