# Environments

This directory contains both simulation environments and everything that plugs into them: rewards, robot assets, physics randomization, and observation mirroring.

---

## Contents

```
envs/
├── mujoco/
│   ├── mujoco_env.py     # MujocoEnv - single Gymnasium environment (CPU)
│   ├── base_env.py       # GLFW rendering, contact detection
│   └── distributed.py   # Wraps MujocoEnv in sequential or multiprocess workers
│
├── isaaclab/
│   ├── biped_env.py      # BipedEnv - GPU parallel environment (4096 envs)
│   ├── biped_env_cfg.py  # Config dataclasses for V1 / V2 robot
│   ├── rewards/
│   │   └── rewards.py    # Shared reward functions - pure PyTorch, @torch.jit.script
│   └── mdp/
│       └── commands.py   # UniformVelocityCommand generator
│
├── rewards/
│   └── mujoco_reward.py  # Calls rewards.py with numpy→tensor conversion per step
│
├── utils/
│   ├── mirroring.py      # Left-right obs/action symmetry augmentation
│   └── randomizer.py     # Per-episode physics randomization (MuJoCo)
│
└── assets/
    ├── robot/            # V1 robot (torso_link as base)
    │   ├── Robot_description/urdf/robot_mujoco.xml
    │   ├── biped_robot.py     (Isaac Lab ArticulationCfg)
    │   └── robot.usd
    └── robotV2/          # V2 robot (base_link as base)
        ├── Robot_description/...
        ├── biped_robot.py
        └── robot.usd
```

---

## How the Two Environments Relate

Both environments are designed to be **functionally identical** from a policy's perspective:

| | `MujocoEnv` | `BipedEnv` |
|--|-------------|------------|
| Backend | MuJoCo + Gymnasium | Isaac Lab / PhysX |
| N envs | 1 (distributed externally) | 4096 in parallel |
| Obs layout | 48-dim (identical) | 48-dim (identical) |
| Action space | `[-1, 1]^12` | `[-1, 1]^12` |
| Action mapping | `ctrl_min + ((a+1)/2) * ctrl_range` | `joint_min + ((a+1)/2) * joint_range` |
| Reward | `mujoco_reward.py` calls `rewards.py` | `biped_env.py` calls `rewards.py` |
| Termination | tilt > 45° or height < 0.15 m | tilt > 45° or both feet airborne |

The reward functions in `envs/isaaclab/rewards/rewards.py` are pure PyTorch with no Isaac Lab dependency. MuJoCo calls them by converting its numpy state to single-batch `(1, ...)` tensors, then calling `.item()` on the result.

---

## Observation Space (48 dims)

| Slice | Dims | Source | Notes |
|-------|------|--------|-------|
| `lin_acc` | 3 | IMU / finite-diff of base vel | Body frame |
| `ang_vel_b` | 3 | IMU / qvel[3:6] | Body frame |
| `projected_gravity` | 3 | `R_inv @ [0,0,-1]` | Body frame, indicates tilt |
| `commands` | 3 | Sampled per episode | `[vx, vy, wz]` velocity target |
| `joint_pos_rel` | 12 | `qpos[7:] - default_joint_pos` | Relative to neutral pose |
| `joint_vel` | 12 | `qvel[6:]` | Joint velocity |
| `prev_actions` | 12 | Last applied action | Closes the action loop |

Privileged critic obs appends additional information on top of the 48 actor dims. In MuJoCo this is richer (friction, damping, body masses); in Isaac Lab it's a single contact force scalar.

---

## Reward Terms

All 18 terms live in `envs/isaaclab/rewards/rewards.py`. Each is a `@torch.jit.script` function that takes batched tensors and returns a per-env scalar.

| Term | Sign | What it rewards/penalizes |
|------|------|--------------------------|
| `track_lin_vel_xy_exp` | + | Exponential tracking of commanded XY velocity (sigma=0.25) |
| `track_ang_vel_z_exp` | + | Exponential tracking of commanded yaw rate (sigma=0.25) |
| `termination_penalty` | − | Fall or excessive tilt |
| `lin_vel_z_l2` | − | Vertical base velocity (bouncing) |
| `ang_vel_xy_l2` | − | Roll/pitch angular velocity |
| `flat_orientation_l2` | − | Non-flat base orientation (projected gravity xy) |
| `action_rate_l2` | − | Jerkiness: change in action between steps |
| `dof_torques_l2` | − | Joint torque magnitude |
| `dof_acc_l2` | − | Joint acceleration |
| `dof_pos_limits` | − | Joint positions approaching soft limits |
| `feet_air_time` | + | Alternating gait: foot air time up to 0.5 s threshold |
| `feet_slide` | − | Foot sliding while in contact |
| `joint_deviation_hip` | − | Hip joints deviating from default position |
| `joint_deviation_ankle_roll` | − | Ankle roll joints deviating from default |
| `step_length` | + | Stride length at touchdown (optional, default weight 0) |
| `swing_foot_height` | + | Foot lift during swing phase (optional, default weight 0) |
| `torso_centering` | + | COM staying between feet (optional, default weight 0) |
| `knee_bend_touchdown` | + | Knee flexion at touchdown - encourages soft landing |

Default weights are defined in `envs/rewards/mujoco_reward.py` (`_DEFAULT_WEIGHTS`) and `envs/isaaclab/biped_env.py` (via config). Any weight can be overridden in YAML - set to `0.0` to disable a term entirely.

### Adding a new reward term

1. Add a `@torch.jit.script` function to `envs/isaaclab/rewards/rewards.py`:

```python
@torch.jit.script
def my_new_term(some_tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.sum(torch.square(some_tensor), dim=1)
```

2. Call it in `BipedEnv._get_rewards()` (`envs/isaaclab/biped_env.py`):

```python
r_my_term = rewards.my_new_term(self.some_buffer)
reward_terms["my_new_term"] = r_my_term * self.cfg.rewards.get("my_new_term", 0.0)
```

3. Call it in `MujocoReward.compute()` (`envs/rewards/mujoco_reward.py`), converting MuJoCo state to a tensor:

```python
r_my_term = rwd.my_new_term(_t(env.data.some_field)).item()
terms["my_new_term"] = r_my_term
```

4. Add the default weight to `_DEFAULT_WEIGHTS` in `mujoco_reward.py`.

5. Set the weight in YAML:
```yaml
rewards:
  my_new_term: 0.5
```

---

## Adding a Custom Robot

### MuJoCo

1. Create a MuJoCo XML file for your robot. The XML must define:
   - A `free` joint at the root (6-DOF floating base)
   - 12 position-controlled actuators named with `<position>` tags
   - Geom names `l_foot` and `r_foot` for foot contact detection
   - A geom named `floor`

2. Update the `xml_path` in `MujocoEnv.__init__()`:
   ```python
   xml_path = "envs/assets/my_robot/robot.xml"
   ```

3. Update joint group names for the per-group deviation rewards (hip, knee, ankle) in `MujocoEnv.__init__()`:
   ```python
   self.hip_indices        = [_qidx(n) for n in ["my_hip_l", "my_hip_r", ...]]
   self.knee_indices       = [_qidx(n) for n in ["my_knee_l", "my_knee_r"]]
   self.ankle_roll_indices = [_qidx(n) for n in ["my_ankle_roll_l", "my_ankle_roll_r"]]
   ```

4. If the action dimension changes from 12, update `cfg.action_space = N` in `_setup_spaces()`.

### Isaac Lab

1. Create a USD file for your robot. Export from URDF using Isaac Lab's URDF converter or from your CAD tool.

2. Create an `ArticulationCfg` in a new `biped_robot.py` file (copy from `envs/assets/robot/biped_robot.py` as template). Set `prim_path`, joint names, actuator configs, and initial state.

3. Create a new `EnvCfg` in `envs/isaaclab/biped_env_cfg.py`:
   ```python
   @configclass
   class MyRobotEnvCfg(BipedEnvCfg):
       base_body_name: str = "my_base_link"
       right_foot_body_name: str = "my_right_foot"
       left_foot_body_name: str = "my_left_foot"
       joint_limits: list = [(-1.5, 1.5), ...]  # 12 joints
   ```

4. Register it in `src/isaaclab/train.py`:
   ```python
   task_map = {
       "BipedRobot": BipedEnvCfg,
       "BipedRobotV2": BipedRobotV2EnvCfg,
       "MyRobot": MyRobotEnvCfg,  # add this
   }
   ```

5. Run with `--task MyRobot`.

---

## Physics Randomization (MuJoCo)

Controlled via the `randomization` section in the YAML config:

```yaml
randomization:
  randomize_dynamics: true
  friction: [0.5, 1.5]      # multiply original friction by U(0.5, 1.5)
  joint_damping: [0.8, 1.2] # multiply original damping by U(0.8, 1.2)
  mass: [0.9, 1.1]          # multiply original body mass by U(0.9, 1.1)
  inertia: [0.9, 1.1]       # multiply original inertia by U(0.9, 1.1)
```

Each value is a `[min_scale, max_scale]` multiplier applied to the original model value. Applied fresh every `reset()`.

Isaac Lab randomization is configured in `biped_env_cfg.py` via `EventTermCfg` entries (`randomize_rigid_body_mass`, `randomize_rigid_body_material`, `push_by_setting_velocity`).

---

## Obs/Action Mirroring

`envs/utils/mirroring.py` implements left-right symmetry augmentation. When enabled (set `enable_mirroring: true` in config), 50% of episodes use a mirrored policy - the observation is flipped left-right and the action is correspondingly flipped before being applied. This doubles effective data without any hardware changes.

Enabled via config:
```yaml
train:
  env_config:
    enable_mirroring: true
```
