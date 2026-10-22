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
    PPO = "ppo"
class ObsType(str, Enum):
    NORMAL = "normal"
    PRIVILEGED = "privileged"
    BASIC = "basic"

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

class ExplorationConfig(BaseModel):
    scale: float = 0.3
    min_scale: float = 0.03
    decay_rate: float = 1e-6
    start_steps: int = 10000

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
    exploration: Optional[ExplorationConfig] = None
    
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
    enable_mirroring: bool = Field(default=False)
    reward_weights: Dict[str, float] = Field(default_factory=dict)
    reward_scale: float = 1.0
    
    # Commands structure matching config.yaml
    commands: Optional[Dict[str, Any]] = None

    # Mirroring
    mirror_joint_indices: Optional[List[int]] = None

    # Randomization & Events
    enable_perturbations: bool = False
    # IGNORED. Never wired to anything: the real push is events.push_robot in biped_env_cfg
    # (interval 1-5 s, +-0.4 m/s x). Set it via `events: {push_robot: {interval_range_s, params}}`.
    # Kept only so run configs saved with these keys still load.
    push_interval_s: float = 15.0
    push_vel_range: float = 1.0
    
    enable_physics_randomization: bool = False
    events: Optional[Dict[str, Any]] = None
    
    # Noise
    # Per-channel observation noise std. Keys must match biped_env_cfg.observation_noise_std;
    # anything listed here overrides that default, anything omitted keeps it.
    observation_noise_std: Optional[Dict[str, float]] = None
    action_noise_model: Optional[Dict[str, float]] = None
    quantize_joint_obs: bool = True

    # Curriculum
    curriculum_enabled: bool = False
    curriculum_dr_start_steps: int = 3_000_000
    curriculum_dr_full_steps: int  = 20_000_000
    curriculum_cmd_ramp_steps: int = 5_000_000
    curriculum_init_ramp_steps: int = 5_000_000
    curriculum_dr_events: List[str] = Field(default_factory=list)
    action_delay_steps_range: List[int] = Field(default_factory=lambda: [0, 0])
    servo_obs_delay_steps: int = 0
    imu_obs_delay_steps_range: List[int] = Field(default_factory=lambda: [0, 0])
    imu_accel_bias: float = 0.0
    imu_gyro_bias: float = 0.0
    # Renamed to mount_offset_rad (now applied to the command path too). A non-zero value here
    # raises in src/isaaclab/env_config.py rather than being silently ignored.
    encoder_offset_rad: float = 0.0
    mount_offset_rad: float = 0.0
    imu_misalignment_deg: float = 0.0

    # Servo model and backlash (see actuator/ and envs/assets/robotV2/bam_actuator.py)
    actuator: Optional[Dict[str, Any]] = None
    backlash: Optional[Dict[str, Any]] = None
    action_map: str = "absolute"
    # action_map: residual -> target = clamp(home + scale*a, limits). Dicts keyed by joint type
    # (hip_yaw, hip_roll, hip_pitch, knee, ankle_roll, ankle_pitch); l/r share a value.
    action_home_deg: Optional[Dict[str, float]] = None
    action_scale_rad: Optional[Dict[str, float]] = None
    spawn_height_m: Optional[float] = None

    # Gait clock without a gait table. speed_proportional: rate = gait_clock_base_freq * |v_xy|.
    # fixed_period: constant rate 2*pi/gait_clock_period_s, held while |v_xy| and |wz| are both
    # below gait_clock_stand_threshold. A gait table (reference) always uses its own period.
    gait_clock_mode: str = "speed_proportional"
    gait_clock_period_s: float = 0.7
    gait_clock_stand_threshold: float = 0.05

    # Gait-table imitation (see utils/motion_reference.py GaitTableReference). None = off.
    reference: Optional[Dict[str, Any]] = None

    # Schedules (common_step_counter units). None leaves the env default alone.
    standing_envs_start: float = 0.02
    standing_envs_end: Optional[float] = None
    standing_envs_ramp_steps: int = 0
    action_rate_weight_start: float = -0.01
    action_rate_weight_end: Optional[float] = None
    action_rate_ramp_steps: int = 0

    swing_min_clearance: float = 0.01
    swing_max_clearance: float = 0.07
    swing_max_time_s: float = 0.5
    # Reward-side contact flag: |F| > 1 N OR foot clearance < this [m] (imitation_contact, feet_slide). None/0 = force only.
    contact_height_thresh_m: Optional[float] = None
    # hard (default) | soft: soft contact score c = max(force_flag, clamp(1 - clearance / thresh, 0, 1)) for
    # imitation_contact (1 - |c - table|) and as the feet_slide weight. Needs contact_height_thresh_m > 0.
    contact_height_mode: str = "hard"
    # mean (default) | min over the two feet for imitation_foot_height and imitation_contact.
    imitation_foot_agg: str = "mean"
    track_lin_std_rel: float = 0.0
    track_lin_std_min: float = 0.07
    track_ang_std_rel: float = 0.0
    track_ang_std_min: float = 0.15
    # command (default) | reference: centre of the linear velocity-tracking kernel (reference = the gait table's
    # base_lin_vel_b at the current command and phase; needs env_config.reference and track_lin_std_rel > 0).
    track_lin_target: str = "command"
    feet_air_time_mode: str = "per_step"
    feet_air_time_min_s: float = 0.15
    feet_air_time_max_s: float = 0.4
    com_centering_std: float = 0.04
    com_foot_margin: float = 0.03
    # foot_separation_mode: origin_one_sided (old) | sole_band (two-sided band on sole centres)
    foot_separation_mode: str = "origin_one_sided"
    foot_sep_target_m: float = 0.095
    foot_sep_tol_m: float = 0.025
    foot_sep_scale_m: float = 0.02
    foot_sole_offset_m: List[float] = Field(default_factory=lambda: [0.0163, 0.0211, -0.0316])
    undesired_contacts_sat_n: float = 20.0
    undesired_contacts_thresh_n: float = 1.0
    leg_contact_thresh_n: float = 0.5
    leg_contact_sat_n: float = 5.0
    # foot_proximity (E21): sole-rectangle half extents [x, y] (m, centred on foot_sole_offset_m), penalty margin (m),
    # and the optional overlap termination (s of continuous footprint overlap; 0 = off)
    foot_sole_half_extents_m: List[float] = Field(default_factory=lambda: [0.0450, 0.0186])
    foot_proximity_margin_m: float = 0.006
    foot_overlap_termination_s: float = 0.0

    class Config:
        extra = "allow" # Allow extra fields to avoid validation errors for complex nested dicts
    
class TrainConfig(BaseModel):
    seed: int = 42
    steps: float = 3e7
    max_episode_steps: int = 1000
    worker_groups: int = 8
    workers_per_group: int = 16
    sim_frequency: int = 100
    use_rough_terrain: bool = False
    use_terrain_curriculum: bool = False
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
    caps_temporal_weight: float = 0.0  # CAPS: weight on ||mu(s_t) - mu(s_t+1)||^2 (0 = off)
    caps_spatial_weight: float = 0.0  # CAPS: weight on ||mu(s) - mu(s + eps)||^2 (0 = off)
    caps_spatial_sigma: float = 0.05  # CAPS: std of eps in normalized actor-input space
    symmetry_loss_weight: float = 0.0  # actor mirror loss ||mu(M s) - M mu(s)||^2 (0 = off; needs symmetry_augmentation)
    # Exactly mirror-equivariant SAC actor: loc/scale are projected 1/2 (f(s) + M f(M s)) in pre-tanh space
    # (torch_rl_algorithms/models/equivariant.py). Needs an env with a mirror transform (BipedEnv) and the
    # gaussian_multivariate head. Default off = plain actor. Saved with the run config, so play / export /
    # sim2real / policy_gate rebuild the same actor.
    equivariant_actor: bool = False
    
    class Config:
        extra = "ignore"

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