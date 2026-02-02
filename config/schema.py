from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any
from enum import Enum
class NetworkType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
class ModelType(str, Enum):
    SAC = "sac"
    FastSAC = "fast_sac"
    DDPG = "ddpg"
    D4PG = "d4pg"
    MPO = "mpo"
    PPO = "ppo"
class ObsType(str, Enum):
    NORMAL = "normal"
    PRIVILEGED = "privileged"
    BASIC = "basic"
class EnvObjective(str, Enum):
    WALK = "walk"
    TARGET = "target"
    BALANCE = "balance"

class SchedulerType(str, Enum):
    COSINE = "cosine"
    PLATEAU = "plateau"
    NONE = "none"

class LRSchedulerConfig(BaseModel):
    scheduler_type: SchedulerType = Field(default=SchedulerType.COSINE)
    
    # Cosine args
    T_max: Optional[float] = None
    eta_min: float = 1e-5

    # Plateau args
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8
    start_step: int = 0
class NetworkConfig(BaseModel):
    network_type: NetworkType = Field(default=NetworkType.MLP)
    hidden_sizes: List[int] = Field(default=[256, 256])
    cnn_sizes: Optional[List[List[int]]] = None
    # Transformer specific
    d_model: Optional[int] = 128
    nhead: Optional[int] = 4
    num_layers: Optional[int] = 2
    dim_feedforward: Optional[int] = 256
class ModelConfig(BaseModel):
    # Backward compatibility fields
    network_type: Optional[NetworkType] = None
    actor_sizes: Optional[List[int]] = None
    critic_sizes: Optional[List[int]] = None
    cnn_sizes: Optional[List[List[int]]] = None
    
    # New modular config
    actor_config: Optional[NetworkConfig] = None
    critic_config: Optional[NetworkConfig] = None
    
    # Training params
    target_coeff: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    lr_scheduler: Optional[LRSchedulerConfig] = None
    
    class Config:
        extra = "ignore"
class BufferConfig(BaseModel):
    size: float = 1e6
    return_steps: int = 5
    batch_iterations: int = 50
    batch_size: int = 1024
    discount_factor: float = 0.99
    steps_before_batches: float = 1e4
    steps_between_batches: int = 50
    seed: int = 42
    
    class Config:
        extra = "ignore"
class EnvConfig(BaseModel):
    objective: EnvObjective = Field(default=EnvObjective.WALK)
    enable_mirroring: bool = Field(default=False)
    reward_weights: Dict[str, float] = Field(default_factory=dict)
    reward_scale: float = 1.0
    
    # Commands structure matching config.yaml
    commands: Optional[Dict[str, Any]] = None

    # Mirroring
    mirror_joint_indices: Optional[List[int]] = None

    # Randomization & Events
    enable_perturbations: bool = False
    push_interval_s: float = 15.0
    push_vel_range: float = 1.0
    
    enable_physics_randomization: bool = False
    events: Optional[Dict[str, Any]] = None
    
    # Noise
    observation_noise_model: Optional[Dict[str, float]] = None
    action_noise_model: Optional[Dict[str, float]] = None
    
    class Config:
        extra = "allow" # Allow extra fields to avoid validation errors for complex nested dicts
    
class TrainConfig(BaseModel):
    seed: int = 42
    steps: float = 3e7
    max_episode_steps: int = 1000
    worker_groups: int = 8
    workers_per_group: int = 16
    sim_frequency: int = 100
    use_history: bool = False
    history_size: int = 0
    normalize_obs: bool = False
    actor_obs: ObsType = ObsType.NORMAL
    critic_obs: ObsType = ObsType.PRIVILEGED
    log_dir: str = "runs"
    checkpoint_path: str = "checkpoints/"
    model_name: str = "sac"
    overwrite_model: bool = False
    
    model: ModelType = ModelType.SAC
    env_config: EnvConfig = Field(default_factory=EnvConfig)
    
    test_environment: bool = False
    epoch_steps: float = 5e3
    save_steps: float = 5e4
    
    test_episodes: int = 5
    show_progress: bool = False
    replace_checkpoint: bool = False
    log: bool = True
    symmetry_augmentation: bool = False
    
    class Config:
        extra = "ignore"
class RandomizationConfig(BaseModel):
    randomize_dynamics: bool = False
    randomize_sensors: bool = False
    randomize_perturbations: bool = False
    
    friction: Optional[List[float]] = None
    joint_damping: Optional[List[float]] = None
    mass: Optional[List[float]] = None
    inertia: Optional[List[float]] = None
    imu_noise: Optional[float] = None
    vel_noise: Optional[float] = None
    t_perturbation: Optional[List[float]] = None
    force: Optional[List[float]] = None
    
    class Config:
        extra = "allow" # Allow extra fields if any

class PPOConfig(BaseModel):
    clip_param: float = 0.2
    ppo_epoch: int = 4
    num_mini_batches: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    num_steps: int = 2048

class Config(BaseModel):
    train: TrainConfig
    model: ModelConfig
    buffer: Optional[BufferConfig] = None
    ppo: Optional[PPOConfig] = None
    randomization: Optional[RandomizationConfig] = Field(default_factory=RandomizationConfig)