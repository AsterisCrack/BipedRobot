import torch
import algorithms.sac.sac as sac
from algorithms.utils import DecayingEntropyCoeff
import platform

class FastSAC(sac.SAC):
    '''Fast Soft Actor-Critic.
    Optimized for high-throughput parallel environments.
    Uses large batch sizes, single-shot sampling, and torch.compile.
    '''

    def __init__(
        self, action_space, model, max_seq_length=1, num_workers=1, seed=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, recurrent_model=False, actor_optimizer=None, critic_optimizer=None, device=torch.device("cpu"), config=None
    ):
        # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass

        # JIT Compile the model components if available and supported
        # Windows usually lacks Triton support for torch.compile(mode="reduce-overhead")
        is_windows = platform.system() == "Windows"
        
        if hasattr(torch, "compile") and not is_windows:
             print("FastSAC: Enabling torch.compile for actor and critic networks.")
             try:
                 # Compile with reduce-overhead for training loop efficiency
                 model.actor = torch.compile(model.actor, mode="reduce-overhead")
                 model.critic_1 = torch.compile(model.critic_1, mode="reduce-overhead")
                 model.critic_2 = torch.compile(model.critic_2, mode="reduce-overhead")
                 
                 # Targets
                 if hasattr(model, "target_critic_1"):
                     model.target_critic_1 = torch.compile(model.target_critic_1, mode="reduce-overhead")
                 if hasattr(model, "target_critic_2"):
                     model.target_critic_2 = torch.compile(model.target_critic_2, mode="reduce-overhead")
             except Exception as e:
                 print(f"FastSAC: Warning - torch.compile failed: {e}")
        elif is_windows:
            print("FastSAC: Windows detected. Disabling torch.compile (requires Triton). Algorithmic optimizations (large batch, single update) will still be used.")
        
        # Initialize parent (SAC)
        # Note: We don't need to override updaters if they match standard SAC, 
        # but we might want to ensure entropy coeff is handled correctly if changed.
        # Parent init will set up actor_updater and critic_updater.
        super().__init__(
            action_space=action_space, model=model, max_seq_length=max_seq_length, num_workers=num_workers,
            seed=seed, replay=replay, exploration=exploration, actor_updater=actor_updater,
            critic_updater=critic_updater, recurrent_model=recurrent_model, actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer, device=device, config=config
        )

    def _update(self, steps):
        # FastSAC Update Loop override
        # This replaces the loop in ddpg.DDPG._update and sac.SAC._update
        
        # 1. Sample ONE giant batch directly (no generator loop)
        # We use the keys defined in DDPG.__init__
        batch = self.replay.sample(self.replay.batch_size, *self.keys)
        
        # Ensure primitive types if needed (sample returns tensors usually)
        # Convert to tensor just in case, though sample() should return tensors on device
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}
        
        # 2. Update Actor and Critic (Single shot)
        # _update_actor_critic handles the internal logic (calls updaters)
        infos = self._update_actor_critic(**batch)
        
        # 3. Update normalizers
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if hasattr(self.model, 'actor_observation_normalizer') and self.model.actor_observation_normalizer:
            self.model.actor_observation_normalizer.update()
        if hasattr(self.model, 'critic_observation_normalizer') and self.model.critic_observation_normalizer:
            self.model.critic_observation_normalizer.update()
            
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        # 4. Soft update targets
        self.model.update_targets()
        
        # Return loss infos
        return dict(
            actor_loss=infos['actor']['loss'].detach(), 
            critic_loss=infos['critic']['loss'].detach()
        )
