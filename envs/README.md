# Environments

The two simulation environments the policy lives in, the robot assets they load, and the shared
reward functions.

- **Isaac Lab (`BipedEnv`)** trains the policy: thousands of robots in parallel on the GPU.
- **MuJoCo (`BipedEnvV2`)** is one nominal robot on the CPU, the independent check: sim-to-sim
  playback, the pre-deploy policy gate and the tests.

Both expose the same observation and action, run the same servo model
([`actuator/`](../actuator/README.md)) and share one reward file.

| | Isaac Lab `BipedEnv` | MuJoCo `BipedEnvV2` |
|---|---|---|
| Robots, rates | Many (GPU), 200 Hz physics, 50 Hz control | One (CPU), same rates |
| Observation, action, servo | 50-dim frame, 12 actions in `[-1, 1]`, BAM servo | Identical; backlash is a passive hinge |
| Randomization | Noise, delays, biases, physics DR, pushes | None: nominal robot, fixed one-step IMU lag |
| Rewards | All terms | A subset, for reporting |
| Episode ends | Tilt over about 45 deg, NaN state (optional foot overlap) | Tilt over about 45 deg, both feet airborne for 1 s |

---

## Contents

- `isaaclab/`: `biped_env.py` (`BipedEnv`), `biped_env_cfg.py` (defaults, event table, terrain), `mdp/`
  (`commands.py`, `events.py`, `symmetry.py`), `rewards/rewards.py` (all reward functions), `physx_cache.py`.
- `mujoco/`: `biped_env_v2.py` (`BipedEnvV2`), `mujoco_env.py` (`MujocoEnv`, the older V1 robot),
  `base_env.py` (model loading, viewer), `distributed.py` (V1 wrappers).
- `assets/robotV2/` (current robot: MJCF, USD, URDF, `ArticulationCfg`, `BamActuator`), `assets/robot/`
  (V1), and V1 helpers in `utils/` and `rewards/`.

`BipedRobotV2` is the maintained robot and task. The V1 assets and `BipedRobot` task are kept for
reference and are not expected to run unchanged (the code now looks joints up by V2 names).

---

## Using the environments

The Isaac environment needs Isaac Sim launched first, so build it inside a script (see
`src/isaaclab/play.py`). MuJoCo is plain Python:

```python
# Isaac Lab
from src.isaaclab.env_config import build_env_cfg
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg
env = BipedEnv(build_env_cfg(config, BipedRobotV2EnvCfg, training=True, num_envs=64))

# MuJoCo
from utils import Config
from envs.mujoco.biped_env_v2 import BipedEnvV2
config = Config("config/config.yaml")
env = BipedEnvV2(history_size=5, use_history=True, env_config=config.train.env_config)
obs, _ = env.reset(seed=0)               # obs["actor"] (250,), obs["critic"] (60,)
env.set_command(0.1, 0.0, 0.0)           # vx, vy, wz; limited like in training
obs, reward, terminated, truncated, info = env.step(action)    # action: (12,) in [-1, 1]
```

**How a config reaches the environment.** `train.env_config` becomes a `BipedEnvCfg` in one function,
[`src/isaaclab/env_config.py`](../src/isaaclab/env_config.py) (`build_env_cfg`), used by training,
`play.py` and the diagnostics; MuJoCo reads the same `env_config` object.

- With `training=False` (playback) everything is applied **except** pushes, physics randomization
  and action noise: the nominal robot runs through the same pipeline.
- An unknown key inside `commands`, `actuator`, `backlash`, `reference` or the noise blocks raises. A
  key reaches the environment only if `env_config.py` copies it (`_SCALAR_FIELDS` or a block handler);
  other `BipedEnvCfg` attributes (`decimation`, `action_filter_alpha`, `gait_clock_base_freq`,
  default event parameters) are code-level defaults. Fields are typed in
  [`config/schema.py`](../config/schema.py) and listed in [`config/README.md`](../config/README.md).

---

## Observations

### Actor frame (50 dims)

| Slice | Dims | Content |
|---|---|---|
| `0:3` | 3 | IMU linear acceleration, body frame, specific force (about +9.81 on z when upright) |
| `3:6` | 3 | IMU angular velocity, body frame |
| `6:9` | 3 | Projected gravity, body frame, unit vector |
| `9:12` | 3 | Command `[vx, vy, wz]` |
| `12:24` | 12 | `joint_pos - default_joint_pos`; the default is the home pose for `action_map: residual` |
| `24:36` | 12 | Joint velocity |
| `36:48` | 12 | Previous action (the one applied at the last control step) |
| `48:50` | 2 | Gait clock `[sin(phi), cos(phi)]` |

Joints are ordered left/right interleaved by type: `l_hip_yaw, r_hip_yaw, l_hip_roll, r_hip_roll,
l_hip_pitch, r_hip_pitch, l_knee, r_knee, l_ankle_roll, r_ankle_roll, l_ankle_pitch, r_ankle_pitch`.

With `train.use_history: true` the actor input is the last `history_size` frames stacked, oldest
first, so the newest frame is the last 50 values (`history_size: 5` gives 250). A reset zero-fills
the stack. The critic gets no history.

### Privileged critic observation (60 dims)

The critic sees the same frame (without noise and vibration) plus 10 values the real robot cannot
measure, never deployed: `50:53` base linear velocity (body frame), `53` base height (world z),
`54:60` net contact force on each foot in the body frame, right foot then left.

### Where each sensor effect is applied

Ground truth goes through these effects before it becomes the frame. Rewards and terminations always
read the ground truth.

| Effect | YAML key | Drawn | Applied to |
|---|---|---|---|
| White noise | `observation_noise_std.<channel>` | Every step | Actor only |
| Band-limited vibration | `observation_vibration_std`, `_hz` | Per robot, redrawn at reset | Actor only |
| Encoder quantisation (4096 counts/rev) | `quantize_joint_obs` | | Joint position and velocity |
| IMU delay | `imu_obs_delay_steps_range` | Per robot, per episode | Acceleration, gyro, gravity together |
| Joint delay | `servo_obs_delay_steps` | Fixed | Joint position and velocity |
| IMU bias | `imu_accel_bias`, `imu_gyro_bias` | Per episode, constant | Acceleration, gyro |
| IMU tilt | `imu_misalignment_deg` | Per robot, once | Acceleration, gyro, gravity |
| Servo zero error | `mount_offset_rad` | Per robot and joint, once | Joint position **and** the command |

- Noise channels: `imu_acc`, `gyro`, `projected_gravity`, `commands`, `joint_pos`, `joint_vel`,
  `previous_actions`, `gait_clock`, each in its own units. Commands, previous actions and the clock
  are not sensors, so they default to 0. Noise is added to the current frame before it enters the
  history.
- Vibration (per channel; `imu_acc`, `gyro`, `joint_vel` in the shipped config) is a narrow-band
  resonator per dimension with a random centre frequency and per-robot amplitude, so it does not
  average out over the stack like white noise. An empty block disables it.
- The IMU acceleration is the mean over the control step (`imu_step_averaged`), not the last substep.
  With `backlash.encoder_side: servo` the joint observations are the servo-shaft angles instead of
  the output link (see [`actuator/`](../actuator/README.md)).

**Symmetry.** `mirror_joint_perm` / `mirror_joint_signs` on the env cfg describe the left-right mirror;
`BipedEnv._build_mirror_transform` builds the tables for the frame and critic observation. They drive
`train.symmetry_augmentation` and `train.equivariant_actor`. On V2 all signs are `+1` (a pure
permutation); the gait clock is negated, because swapping left and right shifts the phase by pi.

---

## Action pipeline

```
policy action a, tanh output in [-1, 1]
  + Gaussian action noise                  training only
  -> clamp to [-1, 1]
  -> action map                            joint target, in encoder space
  -> action delay, 0..N control steps      per robot, per episode
  -> EMA filter, alpha = 0.4               every physics substep
  -> minus mount offset                    per joint
  -> BAM servo                             every physics substep -> torque
  -> PhysX
```

| `action_map` | Target | `a = 0` is |
|---|---|---|
| `residual` | `clamp(home + scale * a, limits)`; `scale` may differ for `a > 0` and `a < 0` | The home pose |
| `centered` | `default + a * span`, `span` = distance from the default to the limit on that side | The default pose |
| `absolute` | Spreads `[-1, 1]` over the whole joint-limit box | The box midpoint |

```yaml
action_map: residual
action_home_deg:  {hip_yaw: 0.0, hip_roll: 0.0, hip_pitch: -10.0, knee: -26.0, ankle_roll: 0.0, ankle_pitch: -16.0}
action_scale_rad: {hip_yaw: 0.45, knee: {pos: 0.55, neg: 0.8}}   # per joint type; a number or {pos, neg}; ...
spawn_height_m: 0.286                  # base height at reset; also the base_height_l2 target
```

For `residual` the home pose also becomes the articulation's reset pose and the origin of
`joint_pos_rel` and the deviation rewards. Home, scales and spawn height may be `auto` with a gait
table; `train.py` freezes them into numbers before the run config is saved. The filter runs every
physics substep, so `alpha = 0.4` is about 0.87 per 20 ms control step. Joint limits are
`_JOINT_LIMITS_DEG` in [`assets/robotV2/biped_robot.py`](assets/robotV2/biped_robot.py).

Changing the action map, home pose, scales or filter makes every existing checkpoint wrong (the same
action means a different target). The exporter writes the map into the bundle and the Jetson refuses
a mismatch.

---

## Step, termination and reset

`BipedEnv.step`: (1) add action noise, then `_pre_physics_step` (update commands, map the action,
apply the delay); (2) four times, `_apply_action` (filter, set the target), actuator torque, PhysX
step; (3) `_get_dones`, then `_get_rewards`; (4) `_reset_idx` for finished robots;
(5) `_get_observations`.

**Termination.** `terminated` when the base tilts more than 0.784 rad (about 45 deg) or the state is
NaN; `truncated` after `episode_length_s` (20 s, 1000 control steps). With
`foot_overlap_termination_s > 0` it also ends when the two foot footprints overlap for that long.
`termination_penalty` fires on the step the tilt is exceeded.

**Reset.** A robot gets a random position within 0.5 m and yaw, a small random joint offset (the init
range), a new command, a random gait phase, zeroed action history and observation stack, an action
filter seeded with the current encoder reading (no reset transient), and fresh per-episode draws
(delays, biases, vibration, servo stiffness). With a gait table, `reference.rsi_prob` of the resets
start mid-gait on the table.

TensorBoard gets `Reward_Terms/<term>` (mean weighted reward per second), `Episode_Termination/*`,
`Torque/*`, `Actuator/*`, `Diagnostics/*` (gait quality) and `Curriculum/*`.

---

## Commands

`UniformVelocityCommand` ([`mdp/commands.py`](isaaclab/mdp/commands.py)) holds a body-frame command
`[vx, vy, wz]` per robot and resamples it when its timer runs out.

Config keys (`train.env_config.commands.base_velocity`, see [`config/README.md`](../config/README.md)):
`resampling_time_range`, `ranges`, `limit_shape`, `rel_standing_envs`, `rel_turn_in_place_envs`
(vx = vy = 0, `|wz|` in [0.4, 1] x max), `rel_heading_envs`, `heading_control_stiffness`,
`heading_max_ang_vel`.

- **`box`** samples each axis uniformly. **`ellipse_xy`** samples `(vx, vy)` uniformly inside
  `(vx/vx_b)^2 + (vy/vy_b)^2 <= 1` (`vx_b` is the forward or backward bound by sign); `wz` keeps its box.
- **Heading mode** sets `wz = heading_error * heading_control_stiffness` every step, clamped to the
  `ang_vel_z` range and `heading_max_ang_vel`. Standing and turn-in-place robots are exempt.
- The limit lives in [`utils/command_limit.py`](../utils/command_limit.py), shared by the sampler,
  `BipedEnvV2.set_command`, the export and the Jetson runtime; the training config is its source of
  truth. `standing_envs_start/end/ramp_steps` ramp `rel_standing_envs` during training.

---

## Gait clock

The phase `phi` is a policy input (`[sin, cos]`) and drives `gait_phase_contact`. It starts at a
random phase at reset and advances every control step; in `fixed_period` and `cadence_law` mode it is
held while both `|v_xy|` and `|wz|` are below `gait_clock_stand_threshold` (with a gait table, below
`reference.stand_threshold`).

| `gait_clock_mode` | Cadence |
|---|---|
| `speed_proportional` | `gait_clock_base_freq * |v_xy|` Hz, so it stops at zero command (`gait_clock_base_freq` is a code-level default) |
| `fixed_period` | One cycle every `gait_clock_period_s` |
| `cadence_law` | Period moves linearly with the command magnitude, from `gait_clock_period_max_s` (slow) to `gait_clock_period_min_s` (fastest command) |

With a gait table (`reference:` block) the table's own cadence grid is used; it shares one format with
the cadence law ([`utils/cadence_law.py`](../utils/cadence_law.py)), so MuJoCo, the export and the
Jetson clock the policy the same way.

---

## Randomization

### Events

Most physics randomization is an Isaac Lab event in the `events` table of
[`biped_env_cfg.py`](isaaclab/biped_env_cfg.py):

| Event | Mode | What it draws |
|---|---|---|
| `push_robot` | interval | Velocity kick on the base, linear and yaw |
| `randomize_mass` | reset | Scale on the 12 leg links |
| `randomize_friction` | reset | Friction and restitution of the robot's bodies (the effective foot-ground friction) |
| `randomize_com` | reset | Base COM offset from the nominal COM (no accumulation) |
| `randomize_payload` | reset | Extra mass on the base |
| `randomize_joint_params` | startup | Joint armature scale |
| `randomize_bam_actuator` | startup | Supply voltage per robot; friction scale, backlash, dead zone per servo |

`startup` events are manufacturing properties of one robot, drawn once, so thousands of different
robots train together; `reset` events are redrawn every episode. The servo events are configured through
the `actuator:` and `backlash:` blocks. `enable_perturbations` gates `push_robot` and
`enable_physics_randomization` gates the rest. Override any event under `events:` in the YAML, for
example `events: {randomize_friction: {params: {static_friction_range: [0.25, 0.8]}}, push_robot:
{interval_range_s: [3.0, 6.0]}}` (unknown event names are ignored, parameter names are only checked
when the event runs).

Terrain (`train.use_rough_terrain`) is a generated mix of flat, rough, wavy, sloped and tiled patches;
the robot is blind, so it is a disturbance, not an obstacle course.

Drawn in code, not as events: firmware P gain (`actuator.kp_fw_range`, needs
`enable_physics_randomization`), action delay, IMU delay, biases and vibration (per episode, in
`BipedEnv._reset_idx`); mount offset and IMU tilt (once per robot, in `BipedEnv.__init__`).

### Curriculum

Keys: `curriculum_enabled`, `curriculum_dr_start_steps` / `curriculum_dr_full_steps` (randomization is 0
until the first, full at the second), `curriculum_cmd_ramp_steps` and `curriculum_init_ramp_steps` (command
and joint-reset ranges ramp up from a narrow start; 0 = full from the start), `curriculum_dr_events`
(shipped: `randomize_com`, `randomize_payload`, `action_delay`, `servo_stiffness`).

Steps count calls of `env.step`, not robots. `curriculum_dr_events` picks what ramps (empty means
all); nothing ramps unless `curriculum_enabled` is true. `action_delay` and `servo_stiffness` ramp the
upper end of their ranges in code through `BipedEnv._dr_ramp`. `standing_envs_*` and
`action_rate_weight_*` are separate linear schedules that do not need `curriculum_enabled`.

For a ramped event (`push_robot`, `randomize_mass`, `randomize_friction`, `randomize_com`,
`randomize_payload`), the range comes from the `curriculum_dr_*` constants in `biped_env_cfg.py`, not
from the event term or an `events:` override. Isaac Lab's event manager keeps its own copy of the event
configs, so `_apply_curriculum` pushes each rewrite into it through `_sync_event_term`. For
`randomize_friction`, which samples its friction buckets once, that also re-draws the buckets.

---

## Rewards

Every reward term is a function in [`rewards/rewards.py`](isaaclab/rewards/rewards.py), which needs
only PyTorch, so both environments call it (MuJoCo wraps its state as batch-of-one tensors).
`BipedEnv._get_rewards` computes the terms, multiplies each by its YAML weight, sums them and applies
`reward_scale`. Terms with weight `0` are skipped where they are expensive. Variants and shape
parameters (`track_*_std_rel`, `feet_air_time_mode`, contact thresholds, ...) are in `EnvConfig` in
[`config/schema.py`](../config/schema.py).

| Group | Terms |
|---|---|
| Command tracking | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` |
| Stability | `termination_penalty`, `flat_orientation_l2`, `upright`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `angular_momentum`, `base_height_l2` |
| Smoothness and effort | `action_rate_l2`, `action_l2`, `dof_torques_l2`, `ankle_torques_l2`, `dof_acc_l2`, `dof_pos_limits` |
| Joint posture | `dof_pos_l2`, `joint_deviation_hip` / `_hip_pitch` / `_ankle_roll` / `_ankle_pitch` / `_knee`, `stand_pose_l2` |
| Gait shaping | `feet_air_time`, `swing_foot_height`, `gait_phase_contact`, `step_length`, `knee_bend_touchdown`, `com_support_centering`, `torso_centering` |
| Feet and contacts | `feet_slide`, `foot_slip`, `foot_drag`, `foot_flat_l2`, `foot_flat_l1`, `foot_separation`, `foot_proximity`, `undesired_contacts`, `leg_contact_l2` |
| Reference motion | `imitation_*` (eight terms, need a gait table), `track_joint_pos_exp`, `track_joint_vel_exp` (need an `animation:` file) |

**The sign guard.** The total is `sum(weight * term)` and every term is non-negative, so a penalty
needs a negative weight; a positive one silently turns it into a bonus. `REWARD_WEIGHT_SIGN` in
`rewards.py` records each term's expected sign (`"foot_flat_l2": -1` penalty, `"track_lin_vel_xy_exp":
1` reward). `BipedEnv.__init__` and `BipedEnvV2.__init__` call `validate_reward_weight_signs`; a wrong
sign raises one `ValueError` listing every offender, including the scheduled
`action_rate_weight_start/end`. Zero is always valid. Unknown names are ignored, so a misspelt weight
is not an error: it just never reaches a term. Check `Reward_Terms/` in TensorBoard.

---

## The MuJoCo environment

`BipedEnvV2` ([`mujoco/biped_env_v2.py`](mujoco/biped_env_v2.py)) loads `assets/robotV2/robot_v2.xml`
and reproduces the Isaac pipeline step by step (same observation, action map, filter, BAM servo, gait
clock, command limit and step-averaged IMU, read from the same config keys) with none of the training
randomization: no noise, pushes, mass or friction draws, biases or mount offset. The robot is nominal,
with a fixed supply voltage, a one-step IMU lag and no joint lag. Backlash is a real passive hinge in
series with each servo; `robot_v2.backlash.xml` is generated from the MJCF when `backlash.enabled` is
true and `mujoco_lash_deg` is above 0, so do not edit it. `mujoco_ground_friction` sets the sliding
friction of every geom (the MJCF floor uses 1.0, grippier than real soles).

These constants are hard-coded at module level in `biped_env_v2.py`, **not** read from the config, and
must be kept equal to the Isaac side by hand:

| Quantity | Isaac | MuJoCo |
|---|---|---|
| Physics step, decimation | `sim.dt`, `decimation` | `PHYSICS_DT`, `DECIMATION` |
| Filter alpha | `action_filter_alpha` | `ACTION_FILTER_ALPHA` |
| Joint limits | `_JOINT_LIMITS_DEG` | `JLIM_MIN`, `JLIM_MAX` |
| Speed-proportional clock | `gait_clock_base_freq` | `GAIT_CLOCK_FREQ` |
| IMU and joint lag | `imu_obs_delay_steps_range`, `servo_obs_delay_steps` | `IMU_OBS_DELAY`, `SERVO_OBS_DELAY` |

`MujocoEnv` is the older V1 environment (V1 robot, 48-dim observation without a gait clock, its own
critic layout), used by the CPU training scripts in `src/mujoco/`, not by the sim-to-sim checks.
`mujoco/__init__.py` registers it as `Mujoco-v0` with a stale entry point
(`envs.mujoco_env:MujocoEnv`), so import the class directly instead of using `gym.make`.

---

## Robot assets

`assets/robotV2/`: `robot.usd` (Isaac Lab; references the URDF import in `usd/Robot/`), `robot_v2.xml`
(MuJoCo: radians, 12 motors, `imu_site`, floor), the generated `robot_v2.backlash.xml`, `biped_robot.py`
(`ArticulationCfg`, `_JOINT_LIMITS_DEG`) and `bam_actuator.py` (`BamActuator`, the servo model as an
Isaac Lab actuator).

The URDF and meshes are not in this repo: they live in
[BipedRobotHardware](https://github.com/AsterisCrack/BipedRobotHardware), included as the `robot_hardware/`
submodule (`robot_hardware/robot_description/`). Mesh paths in `robot_v2.xml` are relative to the file
(`../../../robot_hardware/robot_description/meshes/...`), and the generated `robot_v2.backlash.xml` is
written next to it, so both load from any checkout that has the submodule.

After changing the URDF in the hardware repo, copy the new inertials into `robot_v2.xml` and re-import the
URDF in Isaac Sim to regenerate `usd/Robot/`.

---

## Tutorial: add a reward term

Example: a penalty on knee speed, `knee_vel_l2`.

**1. Write the function** in `rewards/rewards.py`: batched tensors in, one value per robot `(N,)` out.
Most functions are `@torch.jit.script`, which needs a type on every argument. Return a non-negative
quantity and let the weight carry the sign.

```python
@torch.jit.script
def knee_vel_l2(knee_vel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(knee_vel), dim=1)      # (N,) >= 0
```

**2. Register its sign** in `REWARD_WEIGHT_SIGN`: `"knee_vel_l2": -1`.

**3. Compute it in `BipedEnv._get_rewards`** and add it to the `reward_terms` dict. Read fresh state
from `self.robot.data` (attributes such as `self.joint_vel` are refreshed in `_get_observations`, so
they are one step old here). New state needs a buffer created in `__init__` and cleared in `_reset_idx`.

```python
r_knee_vel = (rewards.knee_vel_l2(self.robot.data.joint_vel[:, self.knee_indices])
              if self._weighted("knee_vel_l2") else self._zero_term)
reward_terms["knee_vel_l2"] = r_knee_vel * self.cfg.rewards.get("knee_vel_l2", 0.0)   # in the dict
```

**4. Give it a weight** under `train.env_config.reward_weights: {knee_vel_l2: -0.001}`; a term logs as
`Reward_Terms/knee_vel_l2` only if it appears there.

**5. Optional, MuJoCo** (a subset of terms only; training uses the Isaac reward): add
`total += _w("knee_vel_l2", R.knee_vel_l2(jv_t[:, KNEE_IDX]))` to `BipedEnvV2._compute_reward` and the
name to `_MUJOCO_REWARD_KEYS`.

**6. Check it** with `python -m pytest tests/test_reward_sign_guard.py tests/test_reward_shaping.py -q`
(the guard test fails if `_get_rewards` reads a weight missing from `REWARD_WEIGHT_SIGN`, or the
reverse), and add a CPU test next to `test_reward_shaping.py`.

A tunable parameter goes in three places: a field on `BipedEnvCfg`, a field on `EnvConfig` in
`config/schema.py`, and the name in `_SCALAR_FIELDS` in `src/isaaclab/env_config.py`.

---

## Tutorial: add or change a randomization

**Change a range:** override it in the YAML `events:` block. **Switch groups off:**
`enable_perturbations: false` removes `push_robot`; `enable_physics_randomization: false` removes the
rest, and the servo then runs nominal (mid-range voltage and lash).

**Add a new event.**

1. Use an Isaac Lab `isaaclab.envs.mdp:randomize_*` term, or add a function to
   [`mdp/events.py`](isaaclab/mdp/events.py) with signature `(env, env_ids, **params)`. Draw absolute
   values from a cached nominal: `current + noise` on every reset is a random walk (this is why
   `randomize_com` is custom).
2. Add an `EventTerm` to `events` in `biped_env_cfg.py`: `mode="startup"` for a property of one
   physical robot, `"reset"` for a per-episode draw, `"interval"` for repeated disturbances.
3. Add its name to the list in `BipedEnvCfg.__post_init__` if `enable_physics_randomization: false`
   should remove it. Its parameters are then overridable from the YAML `events:` block.
4. To ramp it, do it in code like the action delay (multiply its range by
   `self._dr_ramp("my_event")`); adding it to `_apply_curriculum` would hit the limitation above.

**A per-robot or per-episode quantity in code** (a new bias or delay): draw it in `BipedEnv.__init__`
(once per robot) or `_reset_idx` (per episode), apply it in `_get_observations` or `_apply_action`, and
add the config field as for a reward parameter. **MuJoCo** is nominal on purpose; for a robustness
case (say a heavier torso) change the model in `BipedEnvV2.reset()`.

---

## Tutorial: add a robot

The environment is written for the V2 robot (12 joints named by type, left/right interleaved).
Supported today: **a variant of it** with the same joint and body names but different geometry, mass,
limits or servo parameters. A different joint count or topology also changes the observation size, the
action, the mirror tables, the exporter and the Jetson runtime (see the observation tutorial).

1. **Asset folder** `assets/<name>/`. Convert your URDF to USD with Isaac Lab's URDF converter. Copy
   `assets/robotV2/biped_robot.py` and edit the spawn path, `_JOINT_LIMITS_DEG` (row *i* is bound to
   DOF *i* by position), the initial base height and the actuator (`BamActuatorCfg`, `params_file`).
2. **Env cfg.** In `biped_env_cfg.py` subclass `BipedEnvCfg` like `BipedRobotV2EnvCfg`: scene with your
   `ArticulationCfg` (and IMU prim and offset), `joint_limits`, foot, base and joint name regexes,
   `mirror_joint_perm` / `mirror_joint_signs`. Validate the mirror with `src/tools/`; wrong signs
   collapse training.
3. **Register the task** by hand in `src/isaaclab/train.py`, `play.py`, `jitter_probe.py` and
   `profile_train.py`, then run with `--task <name>` (the default task is the V1 `BipedRobot`; pass
   `--task BipedRobotV2` for the current robot).
4. **MuJoCo.** An MJCF with the 12 motors named as `JOINT_NAMES` in `biped_env_v2.py`, bodies
   `base_link`, `l_foot_link_1`, `r_foot_link_1`, site `imu_site` and a floor plane. Update
   `JLIM_MIN/MAX` and pass `xml_path=` to `BipedEnvV2`.
5. **Export and check.** `export_onnx.py` reads the limits from `assets/robotV2/biped_robot.py` and has
   the joint names built in; point it at your files. Run `src/isaaclab/check_actuator.py` (it builds
   `BipedRobotV2EnvCfg`, so adapt the import) before a training run. A new servo model is a separate
   step: identify it with [`actuator/`](../actuator/README.md) and set `actuator.params_file`.

---

## Tutorial: change the observations

The frame layout lives in `BipedEnv._get_observations` and is rebuilt in four other places. Example:
add one actor-visible channel `foo` (1 dim).

1. `BipedEnv.__init__`: update `obs_proprio_dim` (50 to 51). Append `foo` in `_get_observations`
   after the gait clock so existing slices keep their indices.
2. If `foo` has noise or vibration: add a channel to `observation_noise_std` in `biped_env_cfg.py` and
   to the two channel lists in `BipedEnv.__init__` that build `obs_noise_std` and `vib_std`.
3. `_build_mirror_transform`: the tables are written index by index for 50 and 60 dims. Extend them,
   with the right sign for `foo` under mirroring.
4. MuJoCo: `BipedEnvV2._build_obs`, `base_actor_dim` / `base_critic_dim` in `__init__`, the lag buffers
   in `_reset_sensor_lag`, and `FRAME = 50` in `src/mujoco/policy_gate.py`.
5. Export: the `OBS_LAYOUT` slices and the `proprio_dim != 50` check in `src/isaaclab/export_onnx.py`
   (`verify_onnx_bundle.py` reads the layout from the bundle).
6. Robot: in [BipedRobotJetson](https://github.com/AsterisCrack/BipedRobotJetson),
   `policy/observation.py` assembles the frame and `policy/deploy_config.py` validates it. The robot
   needs a real source for `foo`, and the bundle format must change so an old runtime refuses it.

This breaks every existing checkpoint and bundle (the actor input size and the saved
`<checkpoint>_obs_scalers.pt` shape change) and tests that fix the sizes, such as
`tests/test_symmetric_obs_stats.py`. A **critic-only** change (`obs_priv_dim`) is lighter: update
`BipedEnv.__init__`, `_get_observations`, the critic mirror table and `BipedEnvV2._build_obs`; export
and Jetson are unaffected (the critic is never exported), but old checkpoints still stop loading.

---

## Tests

```bash
python -m pytest tests -q --ignore=tests/test_env.py --ignore=tests/test_transformer.py
```

`test_env.py` and `test_transformer.py` import modules that no longer exist and fail at collection.
The rest covers the reward sign guard and reward functions, command limit, action maps (Isaac and
MuJoCo), gait clock modes and export, the previous-action slot, and mirror tables and statistics.
