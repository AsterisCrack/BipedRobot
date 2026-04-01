# Configuration Reference

All config files are YAML, validated against the Pydantic schema in `schema.py`. The top-level sections are `train`, `model`, `buffer` (off-policy), and `ppo` (PPO only).

---

## `train`

Core training settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"sac"` | Algorithm: `sac` `fast_sac` `ddpg` `d4pg` `mpo` `ppo` |
| `seed` | int | `42` | Global random seed |
| `steps` | float | `3e7` | Total environment steps |
| `max_episode_steps` | int | `1000` | Episode length before forced reset |
| `worker_groups` | int | `8` | Number of parallel worker groups (MuJoCo) |
| `workers_per_group` | int | `16` | Workers per group (MuJoCo) |
| `sim_frequency` | int | `100` | MuJoCo simulation frequency (Hz) |
| `actor_obs` | string | `"normal"` | `normal` or `privileged` |
| `critic_obs` | string | `"privileged"` | `normal` or `privileged` |
| `use_history` | bool | `false` | Stack last N observations |
| `history_size` | int | `0` | Steps to stack (0 = off) |
| `log_dir` | string | `"runs"` | TensorBoard log directory |
| `checkpoint_path` | string | `"checkpoints/"` | Checkpoint save root |
| `model_name` | string | `"sac"` | Run name - becomes subdirectory |
| `overwrite_model` | bool | `false` | Overwrite existing checkpoint dir |
| `test_environment` | bool | `false` | Periodic evaluation rollouts during training |
| `epoch_steps` | float | `5e3` | Steps per training epoch |
| `save_steps` | float | `5e4` | Steps between checkpoint saves |
| `log` | bool | `true` | Enable TensorBoard logging |
| `env_config` | object | - | Environment and reward config (see below) |

### `train.env_config`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_mirroring` | bool | `false` | Left-right symmetry augmentation |
| `reward_weights` | dict | `{}` | Override default reward weights by name |
| `reward_scale` | float | `1.0` | Global reward multiplier |
| `enable_perturbations` | bool | `false` | Random velocity pushes |
| `push_interval_s` | float | `15.0` | Seconds between pushes |
| `push_vel_range` | float | `1.0` | Max push velocity (m/s) |
| `commands.base_velocity.ranges.lin_vel_x` | `[min, max]` | `[-0.5, 1.0]` | X velocity command range (m/s) |
| `commands.base_velocity.ranges.lin_vel_y` | `[min, max]` | `[-0.3, 0.3]` | Y velocity command range (m/s) |
| `commands.base_velocity.ranges.ang_vel_z` | `[min, max]` | `[-0.5, 0.5]` | Yaw rate command range (rad/s) |

**Reward term names** (use these as keys under `reward_weights`):

| Term | Default weight | Sign |
|------|---------------|------|
| `track_lin_vel_xy_exp` | `2.0` | + |
| `track_ang_vel_z_exp` | `1.0` | + |
| `termination_penalty` | `-10.0` | − |
| `lin_vel_z_l2` | `-0.1` | − |
| `ang_vel_xy_l2` | `-0.05` | − |
| `flat_orientation_l2` | `-2.0` | − |
| `action_rate_l2` | `-0.01` | − |
| `dof_torques_l2` | `-0.002` | − |
| `dof_acc_l2` | `-1e-6` | − |
| `dof_pos_limits` | `-1.0` | − |
| `feet_air_time` | `1.0` | + |
| `feet_slide` | `-0.1` | − |
| `joint_deviation_hip` | `-0.2` | − |
| `joint_deviation_ankle_roll` | `-0.2` | − |
| `step_length` | `0.0` | + (disabled) |
| `swing_foot_height` | `0.0` | + (disabled) |
| `torso_centering` | `0.0` | + (disabled) |
| `knee_bend_touchdown` | `0.0` | + (disabled) |

---

## `model`

Network architecture and optimizer settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `actor_lr` | float | `1e-4` | Actor learning rate |
| `critic_lr` | float | `1e-4` | Critic learning rate |
| `target_coeff` | float | `0.005` | Soft target update rate (tau) |
| `actor_config` | object | - | Actor network (see NetworkConfig) |
| `critic_config` | object | - | Critic network (see NetworkConfig) |
| `lr_scheduler` | object | `null` | LR scheduler (optional) |
| `exploration` | object | - | Exploration noise schedule (DDPG/D4PG) |

### NetworkConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network_type` | string | `"mlp"` | `mlp` `cnn` `lstm` `transformer` |
| `hidden_sizes` | `[int]` | `[256, 256]` | Layer sizes |
| `cnn_sizes` | `[[filters, kernel, stride]]` | `null` | CNN layer specs |
| `d_model` | int | `128` | Transformer embedding dim |
| `nhead` | int | `4` | Transformer attention heads |
| `num_layers` | int | `2` | Transformer layers |
| `dim_feedforward` | int | `256` | Transformer FFN dim |

**MLP:**
```yaml
model:
  actor_config:
    network_type: "mlp"
    hidden_sizes: [256, 256]
  critic_config:
    network_type: "mlp"
    hidden_sizes: [256, 256]
  actor_lr: 3e-4
  critic_lr: 3e-4
```

**LSTM:**
```yaml
model:
  actor_config:
    network_type: "lstm"
    hidden_sizes: [256]
  critic_config:
    network_type: "lstm"
    hidden_sizes: [256]
```

**Transformer actor, MLP critic:**
```yaml
model:
  actor_config:
    network_type: "transformer"
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 256
  critic_config:
    network_type: "mlp"
    hidden_sizes: [256, 256]
```

### `model.exploration` (DDPG / D4PG)

```yaml
model:
  exploration:
    scale: 0.3          # initial noise std
    min_scale: 0.03     # minimum noise std
    decay_rate: 1e-6    # per-step decay
    start_steps: 10000  # pure random exploration before training
```

### `model.lr_scheduler`

```yaml
model:
  lr_scheduler:
    scheduler_type: "cosine"  # cosine | plateau | none
    T_max: 1e7
    eta_min: 1e-6
    start_step: 5e5
```

---

## `buffer`

Replay buffer for off-policy algorithms (SAC, DDPG, D4PG, MPO).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `size` | float | `1e6` | Buffer capacity |
| `batch_size` | int | `1024` | Batch size |
| `return_steps` | int | `5` | N-step return horizon |
| `batch_iterations` | int | `50` | Gradient steps per cycle |
| `discount_factor` | float | `0.99` | Discount γ |
| `steps_before_batches` | float | `1e4` | Warmup before training |
| `steps_between_batches` | int | `50` | Env steps between updates |

---

## `ppo`

Only used when `train.model: "ppo"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_param` | float | `0.2` | Clipping epsilon |
| `ppo_epoch` | int | `4` | Gradient epochs per rollout |
| `num_mini_batches` | int | `4` | Mini-batches per epoch |
| `value_loss_coef` | float | `0.5` | Value loss weight |
| `entropy_coef` | float | `0.01` | Entropy bonus |
| `gamma` | float | `0.99` | Discount |
| `gae_lambda` | float | `0.95` | GAE lambda |
| `max_grad_norm` | float | `0.5` | Gradient clip |
| `num_steps` | int | `2048` | Rollout length |

---

## Physics Randomization (MuJoCo)

```yaml
randomization:
  randomize_dynamics: true
  friction: [0.5, 1.5]       # scale range for geom friction
  joint_damping: [0.8, 1.2]  # scale range for joint damping
  mass: [0.9, 1.1]            # scale range for body mass
  inertia: [0.9, 1.1]        # scale range for body inertia
```

---

## Complete Example

```yaml
train:
  seed: 42
  model: "sac"
  steps: 3.0e7
  max_episode_steps: 1000
  worker_groups: 8
  workers_per_group: 16
  sim_frequency: 100
  actor_obs: "normal"
  critic_obs: "privileged"
  log_dir: "runs"
  checkpoint_path: "checkpoints/"
  model_name: "sac_run1"
  save_steps: 1.0e5
  env_config:
    enable_mirroring: true
    reward_weights:
      track_lin_vel_xy_exp: 2.0
      track_ang_vel_z_exp: 1.0
      termination_penalty: -10.0
      feet_air_time: 1.0
      feet_slide: -0.1
      flat_orientation_l2: -2.0
      action_rate_l2: -0.01
      dof_torques_l2: -0.002
      joint_deviation_hip: -0.2
      joint_deviation_ankle_roll: -0.2
    commands:
      base_velocity:
        ranges:
          lin_vel_x: [-0.5, 1.0]
          lin_vel_y: [-0.3, 0.3]
          ang_vel_z: [-0.5, 0.5]

model:
  actor_config:
    network_type: "mlp"
    hidden_sizes: [256, 256]
  critic_config:
    network_type: "mlp"
    hidden_sizes: [256, 256]
  actor_lr: 3.0e-4
  critic_lr: 3.0e-4

buffer:
  size: 1.0e6
  batch_size: 1024
  return_steps: 5
  discount_factor: 0.99
  steps_before_batches: 1.0e4

randomization:
  randomize_dynamics: true
  friction: [0.7, 1.3]
  joint_damping: [0.9, 1.1]
  mass: [0.95, 1.05]
```
