# Configuration

A run is defined by one YAML file. This page lists every field, what it does, and how to change
the things you will most likely want to change. The provided configs are templates: copy one, edit
it, and pass it with `--config_path`.

```bash
cp config/config.yaml config/my_run.yaml
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/my_run.yaml --headless
```

| Jump to | |
|---|---|
| [How configs are loaded and checked](#how-configs-are-loaded-and-checked) | schema, `env_config.py`, what a typo does |
| [Provided configs](#provided-configs) | which file to start from |
| [`train`](#train-run-settings) | run settings, symmetry, smoothness regularisers |
| [`model`, `exploration`, `buffer`, `ppo`](#algorithm-network-exploration-buffer) | algorithm, network, replay |
| [`train.env_config`](#trainenv_config-the-environment) | rewards, commands, noise, randomization, servo model, action map, gait clock, reference |
| [Recipes](#recipes) | switch algorithm, change network, add a reward, debug without randomization, resume |

---

## How configs are loaded and checked

```
config.yaml ──► utils.Config ──► config/schema.py (pydantic) ──► train.py ──► src/isaaclab/env_config.py ──► BipedEnvCfg
                numbers like         types, enums,              resolves        copies train.env_config
                "1e5" -> 1e5         defaults                   `auto` values   into the Isaac Lab env
```

1. [`utils/__init__.py`](../utils/__init__.py) reads the YAML, turns strings such as `3e7` or `1e-4`
   into numbers (plain YAML would keep them as text), and validates the result against
   [`schema.py`](schema.py). A wrong type or an unknown enum value (for example `model: sacc`) fails
   at startup.
2. `train.py` resolves any `auto` value of the [action map](#action-map) and writes the numbers back
   into the config.
3. [`env_config.py`](../src/isaaclab/env_config.py) is the one place where `train.env_config`
   becomes an environment config. `train.py`, `play.py` and the diagnostics all go through it, so
   evaluation runs the same observation and actuation pipeline as training. `play.py` and
   `jitter_probe.py` switch off pushes, physics randomization and action noise; everything else
   (sensor noise, delays, biases, servo model, action map) stays on.
4. The config actually used is saved as `checkpoints/<run>/config.yaml` (with every default filled
   in and every `auto` replaced by its number). `play`, `export_onnx`, `sim2real` and `policy_gate`
   read that copy, so a run stays reproducible even if you edit the original file.

### What happens to a typo

Not every unknown key is an error. Know which ones are:

| Where | Unknown key or bad value |
|---|---|
| Top-level section other than `train`, `model`, `buffer`, `ppo` (e.g. `exploration:`) | Silently dropped |
| Unknown key inside `train`, `model`, `buffer`, `ppo` | Silently dropped |
| Wrong type, or bad `train.model` / `network_type` / `scheduler_type` / `actor_obs` / `critic_obs` | Error at startup |
| Unknown key directly under `train.env_config` | Accepted by the schema, but **has no effect** unless `env_config.py` copies it |
| `commands`, `commands.base_velocity`, `...ranges` | Error: unknown key |
| `actuator`, `backlash` | Error: unknown key |
| `reference`, `reference.std` | Error: unknown key |
| `observation_noise_std` channel, `observation_vibration_std` channel | Error: unknown channel |
| `limit_shape`, `action_map`, `gait_clock_mode`, `feet_air_time_mode`, `contact_height_mode`, `imitation_foot_agg`, `foot_separation_mode`, `backlash.encoder_side` | Error: unknown value |
| `reward_weights` name that no term uses | Silently unused |
| `reward_weights` with the wrong sign (a penalty above 0, a reward below 0) | Error listing every offender |
| `events` name that is not an active event | Silently ignored |
| `action_home_deg` / `action_scale_rad` with other than the six joint types | Error |
| `action_home_deg` / `action_scale_rad` while `action_map` is not `residual` | Error |
| `encoder_offset_rad` non-zero | Error (renamed to `mount_offset_rad`) |

When a setting seems to do nothing, check the key against [`schema.py`](schema.py) and the
`_SCALAR_FIELDS` list in [`env_config.py`](../src/isaaclab/env_config.py).

### Units

| Quantity | Unit |
|---|---|
| `train.steps`, `epoch_steps`, `save_steps` | **Iterations**: one iteration is one step of the whole vectorised environment (all robots at once, 20 ms of robot time) |
| `curriculum_*_steps`, `*_ramp_steps` | Control steps of the vectorised environment, the same unit as iterations |
| `buffer.size`, `buffer.steps_*` | Transitions / environment steps summed over all robots (iterations × `--num_envs`) |
| `exploration.start_steps`, `lr_scheduler.start_step` | Environment steps summed over all robots |
| `lr_scheduler.T_max` | Gradient steps (the scheduler advances once per optimiser step) |
| Checkpoint file name `step_<N>.pt`, TensorBoard x-axis | Environment steps summed over all robots |

---

## Provided configs

| Config | Reference motion | Use |
|---|---|---|
| [`config.yaml`](config.yaml) | no | Start here. Velocity tracking with hand-written gait rewards, rough terrain, full randomization, residual action map with explicit numbers, `cadence_law` gait clock |
| [`config_gait_reference.yaml`](config_gait_reference.yaml) | yes | `config.yaml` plus a `reference:` block and the `imitation_*` rewards. The hand-written swing, air-time and gait-phase rewards are off, the action map is `auto` (derived from the table), and the actor is mirror-equivariant |
| [`config_gait_reference_resume.yaml`](config_gait_reference_resume.yaml) | yes | Continue or fine-tune a run: action map frozen to numbers, curricula at full strength from step 1 (the step counter restarts at 0 on resume). Its numbers belong to one particular earlier run; for your own run, start from that run's saved `config.yaml` instead (see [Resume](#resume-or-fine-tune)) |
| [`config_gait_reference_no_rand.yaml`](config_gait_reference_no_rand.yaml) | yes | Debugging without randomization: flat ground, no pushes, no physics randomization, zero sensor noise, biases and backlash, no action delay. A fixed 1-step encoder lag and a 0-1 step IMU lag remain. It uses the older `centered` action map and no equivariant actor, so its checkpoints are not compatible with the residual-map configs |
| [`ablations/`](ablations/) | no | A generator plus generated copies of `config.yaml`; see below |

Reference motion is optional. `config.yaml` trains with no gait table at all; the gait-reference
configs only add the `reference:` block and the `imitation_*` terms on top.

### `ablations/`

[`generate.py`](ablations/generate.py) writes full config files, each one `config.yaml` with a
few overrides deep-merged on top. Each ablation is a whole file (not a command-line override), because
the saved run config must be exactly what ran. To make your own variant, add an entry to the
`ABLATIONS` dict (an override dictionary shaped like the config) and run:

```bash
python config/ablations/generate.py     # writes config/ablations/<name>.yaml
```

The `.yaml` files already in that folder are snapshots generated from earlier versions of
`config.yaml` (they use the `centered` action map). Regenerate before using one as a baseline.

---

## Skeleton

```yaml
train:                      # run settings, symmetry, regularisers
  model: "sac"
  env_config:               # everything about the environment (see below)
    reward_weights: {...}
    commands: {...}
    actuator: {...}
    ...
model:                      # network, learning rates
  actor_config: {...}
  critic_config: {...}
buffer:                     # replay settings (off-policy algorithms)
  size: 1000000
ppo:                        # only when train.model: "ppo"
  clip_param: 0.2
```

---

## `train`: run settings

| Key | Default | Meaning | Example |
|---|---|---|---|
| `model` | `sac` | Algorithm: `sac` `ddpg` `d4pg` `mpo` `ppo` | `"sac"` |
| `seed` | `42` | Random seed (`--seed` on the command line overrides it) | `42` |
| `steps` | `3e7` | Training length in iterations (see [Units](#units)) | `500000` |
| `epoch_steps` | `5e3` | Iterations per epoch (one progress line and log entry per epoch) | `1000` |
| `save_steps` | `5e4` | Iterations between checkpoints | `3000` |
| `model_name` | `"sac"` | Run name; a timestamp is appended | `"BipedV2_SAC"` |
| `checkpoint_path` | `"checkpoints/"` | Root for `checkpoints/<model_name>_<timestamp>/` | `"checkpoints/"` |
| `log_dir` | `"runs"` | TensorBoard root | `"runs"` |
| `log` | `true` | TensorBoard scalars. **`false` has no effect**: the trainer treats a false config value as "not set" and falls back to its own default, which is on | `true` |
| `log_interval` | `10` | Write scalars every N iterations, as means over those iterations | `10` |
| `show_progress` | `false` | **No effect**: the tqdm progress bar is always shown (same reason as `log`) | `true` |
| `replace_checkpoint` | `false` | Overwrite a checkpoint file of the same name when saving | `true` |
| `tf32` | `true` | TF32 matmuls in the networks (faster on recent GPUs); `false` = exact fp32 | `true` |
| `use_rough_terrain` | `false` | Generated rough ground (noise, waves, gentle slopes, height patches, flat patches) instead of a plane | `true` |
| `use_terrain_curriculum` | `false` | Progressive terrain difficulty. Leave off: difficulty is then drawn per patch, which acts as randomization | `false` |
| `use_history` | `false` | Stack the last frames of observation for the actor | `true` |
| `history_size` | `0` | Frames to stack (needs `use_history`). Actor input = 50 × `history_size` | `5` |
| `normalize_obs` | `false` | Running mean/std normalisation of the observations (stats saved next to each checkpoint; export needs them) | `true` |
| `critic_obs` | `privileged` | `privileged` adds 10 simulator-only dims to the critic input (base velocity, height, foot forces); `normal` does not | `"privileged"` |
| `actor_obs` | `normal` | Stored but currently has no effect in Isaac Lab: the actor always sees the 50-dim proprioceptive frame | `"normal"` |

The 50-dim frame is: IMU acceleration (3), gyro (3), projected gravity (3), command (3), joint
positions relative to the home pose (12), joint velocities (12), previous action (12), gait clock
`[sin, cos]` (2).

Fields used only by the legacy MuJoCo trainer (no effect in Isaac Lab): `worker_groups`,
`workers_per_group`, `sim_frequency`, `max_episode_steps`. Not read by `src/isaaclab/train.py`
either: `overwrite_model`, `test_environment`, `test_episodes`. The Isaac Lab episode length is set
in `biped_env_cfg.py`. `train.env_config.action_rate_limit` (set to `0` in `config.yaml`) is not read
by any code: the schema accepts it, nothing uses it. The robot-side rate limiter is a deployment
setting, tested by the gate's `--rate`.

### Symmetry and smoothness

| Key | Default | Meaning | Example |
|---|---|---|---|
| `symmetry_augmentation` | `false` | Add mirrored transitions to the replay buffer and mirror-symmetric observation statistics | `true` |
| `equivariant_actor` | `false` | Make the SAC actor exactly mirror-symmetric by construction (needs a mirror-capable env and the SAC actor) | `true` |
| `symmetry_loss_weight` | `0.0` | Extra actor loss on the mirror error; needs `symmetry_augmentation`. Pointless with `equivariant_actor` | `0.0` |
| `caps_temporal_weight` | `0.0` | CAPS smoothness: penalise the change of the actor output between consecutive steps | `1.0` |
| `caps_spatial_weight` | `0.0` | CAPS: penalise the change of the output for a small input perturbation | `1.0` |
| `caps_spatial_sigma` | `0.05` | Std of that perturbation (in normalised observation space) | `0.1` |

---

## Algorithm, network, exploration, buffer

### `model`

| Key | Default | Meaning | Example |
|---|---|---|---|
| `actor_config` | | Actor network (fields below). If omitted, the actor is an MLP, or a CNN when `use_history` is on | |
| `critic_config` | | Critic network. Same fallback | |
| `actor_lr` | `1e-4` | Actor learning rate | `3.0e-4` |
| `critic_lr` | `1e-4` | Critic learning rate | `3.0e-4` |
| `target_coeff` | `0.005` | Soft target-network update rate | `0.01` |
| `lr_scheduler` | cosine, see note | Learning-rate schedule | |
| `exploration` | | Exploration noise (DDPG and D4PG only) | |

Network fields (`actor_config` / `critic_config`):

| Key | Default | Meaning |
|---|---|---|
| `network_type` | `mlp` | `mlp` `cnn` `lstm` `transformer` |
| `hidden_sizes` | `[256, 256]` | MLP layer sizes (also the head of a CNN) |
| `cnn_sizes` | `[[3,32,2],[3,32,2]]` | Two entries of `[kernel, channels, stride]` |
| `d_model`, `nhead`, `num_layers`, `dim_feedforward` | `128`, `4`, `2`, `256` | Transformer size |

```yaml
model:
  actor_config:
    network_type: mlp
    hidden_sizes: [1024, 512, 256]
  critic_config:
    network_type: mlp
    hidden_sizes: [1024, 512, 256]
  actor_lr: 3.0e-4
  critic_lr: 3.0e-4
  lr_scheduler:
    scheduler_type: "none"       # none | cosine | plateau
```

The MLP and the transformer are the backbones to use with the Isaac Lab environment. The critic
sees one frame (no history), so keep it an `mlp`. Always write `actor_config` and `critic_config`
yourself (see the fallback above).

`lstm` and `cnn` pass schema validation but do not work from YAML as shipped:

- **`lstm`** fails when the network is built (`'NetworkConfig' object has no attribute
  'hidden_size'`). The schema's `NetworkConfig` has no `hidden_size`, and a `hidden_size:` you write
  in YAML is silently dropped. The LSTM classes themselves are fine when built directly from the
  `torch_rl_algorithms` library, which has its own `NetworkConfig` with that field.
- **`cnn`** builds, then fails on the first forward pass (`Kernel size can't be greater than actual
  input size`) with the default `cnn_sizes` unless `history_size` is at least 7. For a shorter
  history, give smaller kernels, for example `cnn_sizes: [[2, 32, 1], [2, 32, 1]]` at
  `history_size: 5`.

Only an MLP actor (plain or mirror-equivariant) can be exported with `export_onnx.py`.

**`lr_scheduler`.** If the block is missing, the learning rate follows a cosine decay (`T_max` 5e5
gradient steps, `eta_min` 1e-5). Write `scheduler_type: "none"` to keep it constant.

| Key | Default | Applies to |
|---|---|---|
| `scheduler_type` | `cosine` | `cosine` `plateau` `none` |
| `T_max`, `eta_min` | `5e5` if unset, `1e-5` | cosine |
| `mode`, `factor`, `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps` | `min`, `0.1`, `10`, `1e-4`, `rel`, `0`, `0`, `1e-8` | plateau |
| `start_step` | `0` | Environment step (summed over robots) from which the scheduler acts |

### `exploration` (DDPG, D4PG)

Put it **inside `model:`**. SAC samples from its own stochastic policy and ignores this block; it
only takes uniform random actions for its first 10,000 environment steps (summed over robots), a value
you cannot set. A top-level `exploration:` block, as in the provided configs, is not part of the schema
and is dropped without a message.

```yaml
model:
  exploration:
    scale: 0.3          # initial action-noise std
    min_scale: 0.03     # floor
    decay_rate: 1.0e-6  # exponential decay per step
    start_steps: 10000  # uniform random actions for this many steps first
```

### `buffer` (off-policy: SAC, DDPG, D4PG, MPO)

Required for these algorithms: leaving the block out (or writing `buffer:` with no value) crashes the
run with `'NoneType' object is not subscriptable`. `buffer: {}` is enough to take every default below.

| Key | Default | Meaning | Example |
|---|---|---|---|
| `size` | `1e6` | Replay capacity (transitions) | `1000000` |
| `batch_size` | `1024` | Batch per gradient step | `2048` |
| `return_steps` | `5` | n-step return horizon | `5` |
| `discount_factor` | `0.99` | Discount | `0.99` |
| `batch_iterations` | `50` | Gradient steps per update round | `10` |
| `steps_between_batches` | `50` | Environment steps between update rounds | `4` |
| `steps_before_batches` | `1e4` | Environment steps collected before the first update | `4096` |
| `seed` | `42` | Sampler seed | `42` |

One update round (`batch_iterations` gradient steps) runs after an iteration once
`steps_between_batches` environment steps have passed since the last round. Because every iteration
advances the counter by `--num_envs` steps, any value at or below `--num_envs` means one round per
iteration. To raise the update-to-data ratio (more learning per sample, slower iterations), raise
`batch_iterations`.

### `ppo`

Used only with `train.model: "ppo"` (no `buffer`).

| Key | Default | Meaning |
|---|---|---|
| `clip_param` | `0.2` | Clipping epsilon |
| `ppo_epoch` | `4` | Epochs per rollout |
| `num_mini_batches` | `4` | Mini-batches per epoch |
| `value_loss_coef` | `0.5` | Value-loss weight |
| `entropy_coef` | `0.01` | Entropy bonus |
| `gamma` | `0.99` | Discount |
| `gae_lambda` | `0.95` | GAE lambda |
| `max_grad_norm` | `0.5` | Gradient clip |
| `num_steps` | `2048` | Rollout length |

---

## `train.env_config`: the environment

Keys you leave out take the schema default (listed below) or, for blocks such as `commands`, the
environment default in [`biped_env_cfg.py`](../envs/isaaclab/biped_env_cfg.py).

### Rewards

`reward_weights` maps a term name to its weight. **A term you do not list is off** (weight 0). The
reward is `reward_scale × Σ weight × term`. Penalties must have a weight ≤ 0 and rewards ≥ 0; a
sign guard refuses to start otherwise. All terms are implemented in
[`envs/isaaclab/rewards/rewards.py`](../envs/isaaclab/rewards/rewards.py) (pure PyTorch, shared with
MuJoCo).

```yaml
train:
  env_config:
    reward_scale: 1.0
    reward_weights:
      track_lin_vel_xy_exp: 2.0
      track_ang_vel_z_exp: 1.5
      upright: 0.2
      flat_orientation_l2: -1.5
      action_rate_l2: -0.01
      foot_flat_l2: -5.0
      # unlisted terms are off
```

| Group | Term (sign) | What it scores |
|---|---|---|
| Tracking | `track_lin_vel_xy_exp` (+) | Body-frame x/y velocity vs the command, `exp(-err²/std²)` |
| | `track_ang_vel_z_exp` (+) | Yaw rate vs the command |
| Stability | `termination_penalty` (−) | 1 on the step an episode ends in a fall |
| | `upright` (+) | 1 when vertical, decays with tilt |
| | `flat_orientation_l2` (−) | Squared tilt |
| | `lin_vel_z_l2` (−) | Squared vertical base speed |
| | `ang_vel_xy_l2` (−) | Squared roll/pitch rate |
| | `angular_momentum` (−) | Mass-weighted body angular velocity (limb wobble) |
| | `base_height_l2` (−) | Squared error to the target base height (= `spawn_height_m`) |
| Smoothness, effort | `action_rate_l2` (−) | Squared change of the action per step. Scheduled, see [schedules](#schedules) |
| | `action_l2` (−) | Squared action magnitude |
| | `dof_torques_l2` (−) | Squared joint torques |
| | `ankle_torques_l2` (−) | Squared ankle torques |
| | `dof_acc_l2` (−) | Squared joint accelerations |
| | `dof_pos_limits` (−) | Distance beyond the soft joint limits |
| Pose | `dof_pos_l2` (−) | Squared deviation of all joints from the home pose |
| | `joint_deviation_hip`, `_hip_pitch`, `_ankle_roll`, `_ankle_pitch`, `_knee` (−) | L1 deviation of that joint group from the home pose |
| | `stand_pose_l2` (−) | Squared joint error to the gait table's stand pose while the command is ~0. **Needs a reference** |
| | `torso_centering` (+) | Keeps the base over the feet |
| | `com_support_centering` (+) | `exp(-½(d/std)²)` on the distance of the centre of mass outside the support polygon |
| Feet and gait | `feet_air_time` (+) | Reward for swing duration, see `feet_air_time_mode` |
| | `swing_foot_height` (+) | Foot clearance above its resting height, only while moving and while the swing is shorter than `swing_max_time_s` |
| | `gait_phase_contact` (+) | Foot contacts match the gait clock while moving |
| | `step_length` (+) | Squared stride at touchdown while moving |
| | `knee_bend_touchdown` (+) | Knee flexion during the swing, paid at touchdown |
| | `feet_slide` (−) | Planar foot speed while in contact |
| | `foot_slip` (−) | Planar speed of the sole at the contact point of loaded feet |
| | `foot_drag` (−) | Unloaded, low, moving foot (scuffing) |
| | `foot_flat_l2`, `foot_flat_l1` (−) | Roll tilt of the soles in contact (`sin²`, `abs(sin)`); the L1 form still has a gradient near flat |
| | `foot_separation` (−) | Lateral foot spacing outside a band, see `foot_separation_mode` |
| | `foot_proximity` (−) | The two sole rectangles closer than `foot_proximity_margin_m` (terrain-independent) |
| Contacts | `undesired_contacts` (−) | Force on bodies other than the feet, graded and saturating |
| | `leg_contact_l2` (−) | Legs pressing on each other |
| Animation | `track_joint_pos_exp`, `track_joint_vel_exp` (+) | Tracking of a joint animation (`animation:` block). Leave at 0 unless you use one |
| Imitation | `imitation_joint_pos`, `_joint_vel`, `_foot_pos`, `_contact`, `_orientation`, `_lin_vel`, `_ang_vel`, `_foot_height` (+) | Match the gait table at the current command and phase, each in [0, 1]. **Only exist when `reference:` is set** |

**Reward shape parameters.** These sit next to `reward_weights` under `env_config`:

| Key | Default | Meaning |
|---|---|---|
| `reward_scale` | `1.0` | Global multiplier |
| `track_lin_std_rel`, `track_lin_std_min` | `0.0`, `0.07` | Linear tracking kernel width `std = clamp(rel × abs(cmd), min)`. `rel: 0` uses a fixed width (0.2 m/s) |
| `track_ang_std_rel`, `track_ang_std_min` | `0.0`, `0.15` | Same for yaw (fixed width 0.5 rad/s when `rel: 0`) |
| `track_lin_target` | `command` | `command`, or `reference`: centre the kernel on the gait table's own base velocity (needs `reference` and `track_lin_std_rel > 0`) |
| `swing_min_clearance`, `swing_max_clearance`, `swing_max_time_s` | `0.01`, `0.07`, `0.5` | `swing_foot_height` window [m] above the resting foot height, and the longest swing that still pays [s] |
| `feet_air_time_mode` | `per_step` | `per_step` pays every step a foot is up; `touchdown` pays once at landing |
| `feet_air_time_min_s`, `feet_air_time_max_s` | `0.15`, `0.4` | Swing durations that bound the `touchdown` payment |
| `com_centering_std`, `com_foot_margin` | `0.04`, `0.03` | Kernel width and foot half-extent for `com_support_centering` [m] |
| `foot_separation_mode` | `origin_one_sided` | `sole_band`: two-sided band on the sole-centre spacing; `origin_one_sided`: penalise only a too-small link-origin spacing |
| `foot_sep_target_m`, `foot_sep_tol_m`, `foot_sep_scale_m` | `0.095`, `0.025`, `0.02` | Band centre, half-width and penalty scale for `sole_band` |
| `foot_sole_offset_m` | `[0.0163, 0.0211, -0.0316]` | Sole centre `[x, lateral, z]` in the foot link frame |
| `foot_sole_half_extents_m` | `[0.045, 0.0186]` | Sole rectangle half sizes `[x, y]` for `foot_proximity`, `foot_slip`, `foot_drag` |
| `foot_proximity_margin_m` | `0.006` | Gap below which `foot_proximity` starts to penalise |
| `foot_drag_height_m` | `0.005` | Clearance below which an unloaded moving foot counts as dragged |
| `foot_overlap_termination_s` | `0.0` | End the episode (as a fall) after this long with overlapping footprints; 0 = off |
| `undesired_contacts_thresh_n`, `undesired_contacts_sat_n` | `1.0`, `20.0` | Force where `undesired_contacts` starts and saturates [N] |
| `leg_contact_thresh_n`, `leg_contact_sat_n` | `0.5`, `5.0` | Same for `leg_contact_l2` |
| `contact_height_thresh_m` | `null` | Reward-side contact flag: a foot counts as in contact if loaded **or** within this clearance. Used by `imitation_contact` and `feet_slide`. `null`/0 = force only |
| `contact_height_mode` | `hard` | `soft`: smooth contact score in [0, 1] instead of a flag (needs `contact_height_thresh_m > 0`) |
| `imitation_foot_agg` | `mean` | `mean` or `min` over the two feet for `imitation_foot_height` and `imitation_contact`; `min` lets the worse foot decide |

### Commands

```yaml
    commands:
      base_velocity:
        resampling_time_range: [4.0, 8.0]   # s between new commands per robot
        heading_control_stiffness: 0.5
        heading_max_ang_vel: 0.5            # yaw-rate ceiling while heading control is active
        rel_standing_envs: 0.02             # fraction told to stand still
        rel_heading_envs: 1.0               # fraction that steer by a heading target
        rel_turn_in_place_envs: 0.15        # vx = vy = 0, |wz| from the top of the range
        limit_shape: ellipse_xy             # or box
        ranges:
          lin_vel_x: [-0.2, 0.2]            # m/s
          lin_vel_y: [-0.15, 0.15]
          ang_vel_z: [-1.2, 1.2]            # rad/s
          heading: [-3.14159, 3.14159]
```

| Key | Meaning |
|---|---|
| `limit_shape` | `box`: each axis limited on its own. `ellipse_xy`: also `(vx/vx_max)² + (vy/vy_max)² ≤ 1`, so the combined forward + sideways corners are not asked for. `wz` keeps its own range |
| `rel_standing_envs` | Starting fraction; [`standing_envs_*`](#schedules) can ramp it |
| `rel_heading_envs` | Robots in heading mode derive their yaw command from a random heading target, at `heading_control_stiffness × error`, clamped to `heading_max_ang_vel` |
| `ranges` | Sampling ranges. The config is the source of truth for the command limit: the same shape and ranges are written into the exported bundle, so the robot clamps commands the same way |

Keys you omit keep the environment defaults, which differ from `config.yaml`. With a gait table,
keep the ranges inside the range the table was baked for.

### Observation noise and vibration

White noise is added per sensor channel, in that channel's own units, in training and in `play`.
Channels you omit keep the environment default (not zero); write `0.0` to turn one off.

```yaml
    observation_noise_std:
      imu_acc: 0.15            # m/s^2
      gyro: 0.02               # rad/s
      projected_gravity: 0.01
      commands: 0.0            # not a sensor
      joint_pos: 0.0015        # rad
      joint_vel: 0.25          # rad/s
      previous_actions: 0.0    # not a sensor
      gait_clock: 0.0          # generated
    observation_vibration_std:        # band-limited sensor vibration; remove the block to disable
      imu_acc: 1.5
      gyro: 1.0
      joint_vel: 0.8
    observation_vibration_hz: [3.0, 10.0]
    quantize_joint_obs: true
    action_noise_model: {mean: 0.0, std: 0.01}
```

| Key | Default | Meaning |
|---|---|---|
| `observation_noise_std` | env default | Std per channel: `imu_acc`, `gyro`, `projected_gravity`, `commands`, `joint_pos`, `joint_vel`, `previous_actions`, `gait_clock` |
| `observation_vibration_std` | off | Upper bound of the vibration std for `imu_acc`, `gyro`, `joint_vel`. Each robot draws an amplitude in `U(0, 1) × std` per episode |
| `observation_vibration_hz` | `[3.0, 10.0]` | Range of the vibration's centre frequency. The signal is a narrow-band resonator per dimension, so unlike white noise it does not average out over the history stack |
| `quantize_joint_obs` | `true` | Round joint position and velocity to the servo's 4096-counts-per-revolution grid |
| `action_noise_model` | `{0, 0.01}` | Gaussian noise `{mean, std}` added to the policy action. Training only |

### Delays and biases

| Key | Default | Meaning |
|---|---|---|
| `action_delay_steps_range` | `[0, 0]` | Action delay in control steps `[lo, hi]`, drawn per robot at reset. Ramped in by the curriculum when `action_delay` is in `curriculum_dr_events` |
| `imu_obs_delay_steps_range` | `[0, 0]` | IMU lag in control steps, drawn per robot at reset |
| `servo_obs_delay_steps` | `0` | Encoder lag in control steps, the same for all robots |
| `imu_accel_bias` | `0.0` | Constant accelerometer offset per episode, uniform in ± this [m/s²] |
| `imu_gyro_bias` | `0.0` | Constant gyro offset per episode [rad/s] |
| `mount_offset_rad` | `0.0` | Per-servo zero error, drawn once per robot, applied to observation **and** command [rad] |
| `imu_misalignment_deg` | `0.0` | IMU board tilt about a random axis, angle in `[0, max]`, drawn once per robot |

Biases differ from noise on purpose: a constant error survives the history stack, so the policy
has to stay upright with a consistently wrong reading.

### Physics randomization, pushes and curriculum

Two master switches:

| Key | Default | Switches off |
|---|---|---|
| `enable_perturbations` | `false` | `push_robot` |
| `enable_physics_randomization` | `false` | `randomize_mass`, `randomize_friction`, `randomize_com`, `randomize_payload`, `randomize_joint_params`, `randomize_bam_actuator`, and the per-episode `kp_fw_range` draw |

Both default to **false** in the schema, so set them explicitly. Turning the second one off also
removes the per-servo draws of voltage, friction, lash and dead zone; backlash then falls back to
the middle of `lash_deg_range`.

`mujoco_ground_friction` (default: the MJCF value, 1.0) sets the floor friction of the MuJoCo
evaluation environment used by `sim2real.py` and `policy_gate.py`. Isaac Lab ignores it.

**Events.** The randomization lives in the `events` of
[`biped_env_cfg.py`](../envs/isaaclab/biped_env_cfg.py). Override an event's parameters from YAML
under `events:`:

```yaml
    events:
      randomize_friction:
        params:
          static_friction_range: [0.25, 0.8]
          dynamic_friction_range: [0.2, 0.7]
      push_robot:
        interval_range_s: [2.0, 5.0]
```

| Event | When | Randomizes (env default) |
|---|---|---|
| `reset_base` | reset | Spawn position and yaw, tiny initial velocity |
| `reset_robot_joints` | reset | Initial joint noise. With the curriculum on, its range comes from `curriculum_init_*` |
| `push_robot` | every 3 to 6 s | Velocity kick (x/y ±0.3 m/s, yaw ±0.6 rad/s) |
| `randomize_mass` | reset | Leg-link masses, scale 0.92 to 1.08 |
| `randomize_friction` | reset | Foot-ground friction (static 0.6 to 1.1, dynamic 0.5 to 1.0) and restitution |
| `randomize_com` | reset | Base centre-of-mass offset |
| `randomize_payload` | reset | Extra base mass |
| `randomize_joint_params` | startup | Joint armature, scale 0.85 to 1.15 |
| `randomize_bam_actuator` | startup | Servo population; set from the `actuator` and `backlash` blocks, not from `events` |

Events whose names are not active are ignored without an error, and parameter names must match the
event function. **While `curriculum_enabled` is true**, an event that the curriculum ramps (listed in
`curriculum_dr_events`, or any of `push_robot`, `randomize_mass`, `randomize_friction`,
`randomize_com`, `randomize_payload` when that list is empty) has its parameters rewritten from fixed
constants in `biped_env_cfg.py` (`curriculum_dr_*`), so editing it under `events:` has no effect.
`reset_robot_joints` is always rewritten that way while the curriculum is on. With
`curriculum_enabled: false` every `events:` override applies as written.

**Curriculum.** Randomization can be ramped in so that early training is easier:

```yaml
    curriculum_enabled: true
    curriculum_dr_start_steps: 0         # zero randomization before this step
    curriculum_dr_full_steps: 20000      # full randomization from this step
    curriculum_cmd_ramp_steps: 0         # 0 = full command range from the start
    curriculum_init_ramp_steps: 0        # 0 = full initial-joint noise from the start
    curriculum_dr_events: ["randomize_com", "randomize_payload", "action_delay", "servo_stiffness"]
```

| Key | Default | Meaning |
|---|---|---|
| `curriculum_enabled` | `false` | Off: everything runs at full width from step 0 |
| `curriculum_dr_start_steps`, `curriculum_dr_full_steps` | `3_000_000`, `20_000_000` | Linear ramp of the listed randomization from 0 to 100% (control steps) |
| `curriculum_cmd_ramp_steps` | `5_000_000` | Ramp from a small forward-only command range to the configured `ranges`; 0 = full range at once |
| `curriculum_init_ramp_steps` | `5_000_000` | Ramp of the initial joint noise; 0 = full at once |
| `curriculum_dr_events` | `[]` | Which randomization is ramped: event names, plus `action_delay` (action delay grows from 0 to `action_delay_steps_range`) and `servo_stiffness` (the sim gain widens out from `kp_fw` to `kp_fw_range` on both sides). **Empty means everything the curriculum knows is ramped**: `push_robot`, `randomize_mass`, `randomize_friction`, `randomize_com`, `randomize_payload`, `action_delay` and `servo_stiffness`. The startup events (`randomize_joint_params`, `randomize_bam_actuator`) are never ramped |

Those `curriculum_dr_*` sizes (centre of mass, payload, and the ranges used if you list mass, friction
or `push_robot`) are constants in `biped_env_cfg.py`, not YAML keys. To control friction, mass or pushes
from `events:` while the curriculum is on, name the events you do want ramped in `curriculum_dr_events`
and leave these out (an empty list ramps them all).

### Schedules

Linear ramps independent of the curriculum switch, in control steps. A ramp of 0 jumps straight to the
end value.

| Key | Default | Meaning |
|---|---|---|
| `standing_envs_start`, `standing_envs_end`, `standing_envs_ramp_steps` | `0.02`, `null`, `0` | Fraction of robots commanded to stand still. Active only when `end` is set; then it ramps `start` to `end` and overrides `rel_standing_envs` |
| `action_rate_weight_start`, `action_rate_weight_end`, `action_rate_ramp_steps` | `-0.01`, `null`, `0` | Ramp of the `action_rate_l2` weight. Active only when `end` is set; then the `reward_weights.action_rate_l2` value is ignored |

### Servo model and backlash

The simulated servo is an identified model of the real STS3215 (BAM). Both blocks need a robot with
a BAM actuator (`BipedRobotV2`). Method, parameters and validation are in
[`actuator/README.md`](../actuator/README.md).

```yaml
    actuator:
      params_file: actuator/params/sts3215_12v/m6.json
      kp_fw: 32
      kp_fw_range: [20, 34]
      vin_range: [10.2, 12.6]
      friction_scale_range: [0.8, 1.2]
      torque_scale: 1.0
      stall_torque_cap_nm: 2.94
      tau_ext_source: observer
      deadband_counts: 1
      deadband_counts_range: [1, 1]
    backlash:
      enabled: true
      lash_deg_range: [1.0, 2.5]
      encoder_side: output
      mujoco_lash_deg: 1.5
      impulse_cap: 2.94
```

| Key | Meaning |
|---|---|
| `params_file` | BAM parameter JSON, path relative to the repo root. `python -m bam.fit --output <file>` writes this format, so a new identification is a path change |
| `kp_fw` | Firmware P register. Must match the value on the robot |
| `kp_fw_range` | Firmware gain used in simulation, one value per robot drawn at every reset (needs `enable_physics_randomization`), to cover model uncertainty; the robot still runs `kp_fw`. Ramped in by the `servo_stiffness` curriculum entry |
| `vin_range` | Supply voltage range [V], drawn per robot. Set it to the real battery |
| `friction_scale_range` | Per-servo multiplier on the whole friction budget |
| `torque_scale` | The servos' `TORQUE_LIMIT / 1000`; scales the duty ceiling |
| `stall_torque_cap_nm` | Cap the stall torque [N·m] (applied through the winding resistance, so no-load speed is unchanged). `null` trusts the parameter file |
| `tau_ext_source` | `observer` (default) or `gravity` (debug) for the external-torque estimate |
| `deadband_counts`, `deadband_counts_range` | Firmware dead zone in encoder counts (4096 per revolution): the nominal value (MuJoCo, nominal robot) and the per-servo draw (Isaac Lab). The servo does not move while `abs(goal - encoder)` is within it |
| `backlash.enabled` | Model gear lash |
| `backlash.lash_deg_range` | Total play per joint [deg], drawn per servo; its midpoint is the nominal value when physics randomization is off |
| `backlash.encoder_side` | `output`: the encoder reads the link through the play (the policy sees it). `servo`: the encoder reads the shaft before the play |
| `backlash.mujoco_lash_deg` | Fixed play used by the MuJoCo environment |
| `backlash.impulse_cap` | Cap on the re-engagement knock torque [N·m] |

### Action map

How the policy's output in [-1, 1] becomes a joint target.

| `action_map` | Target |
|---|---|
| `residual` | `clamp(home + scale × a, joint limits)`. `a = 0` is the home pose, which is also the reset pose and the origin of `joint_pos_rel` and the deviation rewards |
| `centered` | `a = 0` is the default pose, `a = ±1` reaches the joint limits |
| `absolute` | The schema default: `a` spans the whole joint-limit box, so `a = 0` is the limit midpoint. Legacy; set the map explicitly |

For `residual`, give a home pose and a scale for each of the six joint types (left and right share
a value):

```yaml
    action_map: residual
    action_home_deg:                # the pose for action = 0
      hip_yaw: 0.0
      hip_roll: 0.0
      hip_pitch: -10.0
      knee: -26.0
      ankle_roll: 0.0
      ankle_pitch: -16.0
    action_scale_rad:               # a number, or {pos, neg} for different reach on each side
      hip_yaw: 0.45
      hip_roll: 0.45
      hip_pitch: {pos: 0.6, neg: 0.8}
      knee: {pos: 0.55, neg: 0.8}
      ankle_roll: 0.35
      ankle_pitch: {pos: 0.6, neg: 0.35}
    spawn_height_m: 0.286           # base height at reset
```

| Key | Default | Meaning |
|---|---|---|
| `action_home_deg` | `null` | Home pose per joint type [deg]; `residual` only |
| `action_scale_rad` | `null` | Reach per joint type [rad]. `{pos: x, neg: y}`: `x` scales `a > 0` (joint angle above home), `y` scales `a < 0` |
| `spawn_height_m` | `null` | Base height at spawn [m]; `residual` defaults to 0.286. The `base_height_l2` target follows it |
| `action_home_deg: auto`, `action_scale_rad: auto`, `spawn_height_m: auto` | | Derive the value instead of writing it (below). Each key can be `auto` on its own |
| `action_scale_mode` | `symmetric` | For `auto` scales: `symmetric` = one scale per joint type; `asymmetric` = separate `pos` and `neg` sized to the gait |
| `spawn_clearance_m` | `0.0054` | `auto` spawn height: lowest sole point above the floor [m] |
| `scale_headroom` | `0.96` | `auto` scale: the widest baked joint excursion (plus reset noise) uses at most this fraction of `±1` |

`auto` (see [`utils/residual_home.py`](../utils/residual_home.py)):

- `action_home_deg` takes the gait table's stand pose (needs a `reference`).
- `action_scale_rad` takes the narrowest value on a 0.05 rad grid that covers every baked cell inside
  the command ranges (needs a `reference`).
- `spawn_height_m` comes from forward kinematics of the home pose; it does not need a table.

`train.py` replaces `auto` by the resulting numbers before the config is saved, so a later re-bake
cannot change an existing run's map.

Three things to know:

- **Changing the map invalidates existing checkpoints and bundles.** The same action then means a
  different joint target. The export records the map, and the robot-side runtime refuses a mismatch.
- The environment also low-pass filters the targets (EMA, `action_filter_alpha`). That constant is in
  `biped_env_cfg.py`, not in YAML.
- `action_home_deg` and `action_scale_rad` raise an error with `centered` and `absolute`.

### Gait clock

The policy gets a 2-dim gait clock `[sin φ, cos φ]`. Without a gait table, `gait_clock_mode` picks how
φ advances:

| `gait_clock_mode` | Rate |
|---|---|
| `speed_proportional` (schema default) | Proportional to the commanded planar speed, so it stops when standing. The constant is in `biped_env_cfg.py` (`gait_clock_base_freq`) |
| `fixed_period` | Constant cycle time `gait_clock_period_s` while moving, held while the command is below `gait_clock_stand_threshold` |
| `cadence_law` | The cycle time follows the command: `gait_clock_period_max_s` just above the stand threshold down to `gait_clock_period_min_s` at the edge of the command ranges. Exported as the same cadence grid a gait table carries |

```yaml
    gait_clock_mode: cadence_law
    gait_clock_period_min_s: 0.6
    gait_clock_period_max_s: 0.9
    gait_clock_stand_threshold: 0.05     # m/s and rad/s
    # gait_clock_period_s: 0.7           # fixed_period only
```

With a [`reference:`](#reference-gait-table) the clock is the table's own (its period or per-command
cadence grid, held below `reference.stand_threshold`), and the `gait_clock_*` keys are ignored.
Periods must lie between 0.2 s and 5 s.

### Reference (gait table)

Optional. Adds imitation of a gait table baked in
[Gait Studio](../robot_motion_reference/gait_studio/README.md). Without a `reference:` block the
`imitation_*` terms do not exist and the training is purely reward-driven.

```yaml
    reference:
      npz_path: robot_motion_reference/processed/gait_table.npz   # relative to the repo root
      stand_threshold: 0.05     # command magnitude below which the clock holds and the stand pose is the target
      rsi_prob: 0.8             # fraction of resets that start mid-gait on the table (training only)
      rsi_joint_noise: 0.05     # rad of noise added to those starts
      std:                      # kernel width per imitation term
        {joint_pos: 0.15, joint_vel: 1.5, foot_pos: 0.02, orientation: 0.08,
         lin_vel: 0.1, ang_vel: 0.3, foot_height: 0.010}
    reward_weights:
      imitation_joint_pos: 0.6
      imitation_foot_pos: 0.4
      imitation_contact: 0.5
      imitation_foot_height: 1.5
      # ... see config_gait_reference.yaml for the full set
```

| Key | Meaning |
|---|---|
| `npz_path` | The baked table. Must exist (bake with `python -m gait_studio.bake`) |
| `stand_threshold` | Below this command magnitude the clock holds and the robot is expected to stand |
| `rsi_prob` | Reference state initialisation: that fraction of resets starts on the table. `play.py` always starts from the default pose |
| `rsi_joint_noise` | Joint noise on those starts [rad] |
| `std` | Width of each imitation kernel (`exp(-mean(err²)/std²)`). Keys: `joint_pos`, `joint_vel`, `foot_pos`, `orientation`, `lin_vel`, `ang_vel`, `foot_height` |

The training config stays the source of truth for the commands and the action map. The table only
supplies motion and cadence.

---

## Recipes

### Switch the algorithm

```yaml
train:
  model: "ddpg"            # sac | ddpg | d4pg | mpo | ppo
model:
  exploration: {scale: 0.3, min_scale: 0.03, decay_rate: 1.0e-6, start_steps: 10000}
```

Off-policy algorithms need the `buffer:` block; PPO needs `ppo:` instead. `equivariant_actor` and the
CAPS terms are SAC-only. The tools downstream of training (`export_onnx`, `verify_onnx_bundle`,
`jitter_probe`, `profile_train`) assume a SAC actor with an MLP, so a different algorithm can be
trained and watched in `play.py` and `sim2real.py`, but not exported with them.

### Change the network

```yaml
model:
  actor_config:
    network_type: transformer      # attends over the history frames
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 256
  critic_config:
    network_type: mlp
    hidden_sizes: [256, 256]
train:
  use_history: true
  history_size: 5
```

Bigger MLP: raise `hidden_sizes`. More or fewer frames: change `history_size` (the actor input is
`50 × history_size`). A transformer actor trains and plays, but
cannot be exported to a bundle; use an `mlp` actor for anything you want to deploy.

### Add or disable a reward

- **Turn a term on or off:** give it a weight, or set it to `0`. Mind the sign (penalties ≤ 0).
- **Tune its shape:** the parameters in [Reward shape parameters](#rewards).
- **Add a new term:** write the function in `rewards.py`, register it in the environment and give it
  a weight. [`envs/README.md`](../envs/README.md) walks through it.

### Widen the randomization

```yaml
    enable_physics_randomization: true
    enable_perturbations: true
    events:
      randomize_friction:
        params: {static_friction_range: [0.2, 1.2], dynamic_friction_range: [0.15, 1.0]}
      randomize_mass:
        params: {mass_distribution_params: [0.85, 1.15]}
      push_robot:
        interval_range_s: [2.0, 4.0]
    observation_noise_std: {imu_acc: 0.3, gyro: 0.04}
    action_delay_steps_range: [0, 2]
    actuator: {kp_fw_range: [32, 50], vin_range: [10.5, 12.6]}
    backlash: {enabled: true, lash_deg_range: [0.5, 3.0]}
```

Wider randomization is harder to learn from. Keep `curriculum_enabled: true` so it ramps in, and
check `curriculum_dr_events`: an event listed there (or every event, if the list is empty) takes its
size from the constants in `biped_env_cfg.py`, not from your `events:` override.

### Make a no-randomization debug run

Use [`config_gait_reference_no_rand.yaml`](config_gait_reference_no_rand.yaml), or switch things off
in your own config:

```yaml
train:
  use_rough_terrain: false
  env_config:
    enable_perturbations: false
    enable_physics_randomization: false
    observation_noise_std: {imu_acc: 0.0, gyro: 0.0, projected_gravity: 0.0, joint_pos: 0.0, joint_vel: 0.0}
    observation_vibration_std: {}
    action_noise_model: {mean: 0.0, std: 0.0}
    action_delay_steps_range: [0, 0]
    imu_obs_delay_steps_range: [0, 0]
    imu_accel_bias: 0.0
    imu_gyro_bias: 0.0
    mount_offset_rad: 0.0
    imu_misalignment_deg: 0.0
    backlash: {enabled: false, lash_deg_range: [0, 0], mujoco_lash_deg: 0}
    curriculum_enabled: false
```

### Resume or fine-tune

```bash
python src/isaaclab/train.py --task BipedRobotV2 \
    --checkpoint checkpoints/<run>/step_<N>.pt --headless
```

Without `--config_path`, `train.py` takes the `config.yaml` saved in the checkpoint's folder. That
file has the action map as plain numbers, which is what you want. To change something, copy that
file, edit it, and pass it with `--config_path`. Do not change the action map, the observation
layout or `history_size`: the loaded policy would see a different robot.

The environment's step counter restarts at 0 on resume, so curricula and schedules start again. To
continue at full strength, collapse them:

```yaml
    curriculum_dr_full_steps: 1
    action_rate_weight_start: -0.1     # = action_rate_weight_end, no ramp
```

[`config_gait_reference_resume.yaml`](config_gait_reference_resume.yaml) is an example of this.

### Change the command range

```yaml
    commands:
      base_velocity:
        ranges:
          lin_vel_x: [-0.1, 0.4]
```

The range is also what the exported bundle clamps to. With a gait table, bake the table for at least
this range (and re-bake after changing it), and with an `auto` action map the scales are derived from
the cells inside it.

### Add a reference motion to your config

Copy the `reference:` block and the `imitation_*` weights from
[`config_gait_reference.yaml`](config_gait_reference.yaml), set the hand-written gait terms
(`gait_phase_contact`, `feet_air_time`, `swing_foot_height`) to 0 as the table now provides that, and
optionally set `action_home_deg`, `action_scale_rad` and `spawn_height_m` to `auto`.

### Check a config before a long run

`check_actuator.py` builds the environment from your config the way `train.py` does and reports
whether the servo model, the action map and the standing behaviour are sane:

```bash
python src/isaaclab/check_actuator.py --headless --config_path config/my_run.yaml
```

See [`src/README.md`](../src/README.md) for this and the other tools.
