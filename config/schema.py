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
    DDPG = "ddpg"
    D4PG = "d4pg"
    MPO = "mpo"
class ObsType(str, Enum):
    NORMAL = "normal"
    PRIVILEGED = "privileged"
class EnvObjective(str, Enum):
    WALK = "walk"
    TARGET = "target"
    BALANCE = "balance"
class LRSchedulerConfig(BaseModel):
    T_max: float
    eta_min: float
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
    
    # Target velocity randomization (when objective == "target")
    # Velocities are in robot reference frame
    target_x_vel_range: List[float] = Field(default=[-0.5, 0.5])  # Forward/backward (m/s)
    target_y_vel_range: List[float] = Field(default=[-0.3, 0.3])  # Left/right (m/s)
    target_w_vel_range: List[float] = Field(default=[-1.0, 1.0])  # Angular velocity (rad/s)
    
class TrainConfig(BaseModel):
    seed: int = 42
    steps: float = 3e7
    max_episode_steps: int = 1000
    worker_groups: int = 8
    workers_per_group: int = 16
    sim_frequency: int = 100
    use_history: bool = False
    history_size: int = 0
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
class Config(BaseModel):
    train: TrainConfig
    model: ModelConfig
    buffer: BufferConfig
    randomization: Optional[RandomizationConfig] = Field(default_factory=RandomizationConfig)