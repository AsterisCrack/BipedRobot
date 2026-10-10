# Changes to port into the current (BAM) version

These changes were made against the old checkout. Everything outside `robot_motion_reference/`
is listed here, so it can be re-applied by hand. The retargeting submodule's changes (the new
`gait_studio/` package, a README note, and the baked table in `processed/`) live in that repo.

**What this adds:** imitation of a command-conditioned gait table baked by Gait Studio. It
brings five things:
1. A fixed-period gait clock.
2. `imitation_*` reward terms.
3. Reference state initialisation (RSI).
4. A config that zeroes the hand-written gait terms.
5. Deploy-side clock metadata.

When no table is configured, behaviour is unchanged, apart from the two bug fixes marked **(fix)**.

---

## New files

### `config/config_gait_reference.yaml`
A copy of `config/config.yaml` with the following changes. Port these as a diff onto your current
config rather than copying the file.

- `model_name: BipedV2_SAC_gaitref`.
- A new `env_config.reference` block:
  ```yaml
  reference:
    npz_path: robot_motion_reference/processed/gait_table.npz
    stand_threshold: 0.05
    rsi_prob: 0.8
    rsi_joint_noise: 0.05
    std: {joint_pos: 0.15, joint_vel: 1.5, foot_pos: 0.02, orientation: 0.08, lin_vel: 0.1, ang_vel: 0.3}
  ```
- New reward weights:

  | term | weight |
  |---|---|
  | `imitation_joint_pos` | 1.5 |
  | `imitation_foot_pos` | 1.0 |
  | `imitation_contact` | 0.5 |
  | `imitation_orientation` | 0.5 |
  | `imitation_lin_vel` | 0.3 |
  | `imitation_ang_vel` | 0.2 |
  | `imitation_joint_vel` | 0.1 |

- Set to 0: `gait_phase_contact`, `feet_air_time`, `swing_foot_height`, all `joint_deviation_*`,
  `com_support_centering`, `dof_pos_l2`, `foot_separation`. The reference now provides foot
  clearance, stance width, posture and contact timing. Velocity tracking and every
  stability/smoothness term are kept.
- `commands.base_velocity.rel_standing_envs: 0.10` (was 0.02). This gives the standing reference
  enough samples. It only takes effect with the `train.py` fix below.
- `commands.base_velocity.ranges`: `lin_vel_x [-0.2, 0.3]`, `lin_vel_y [-0.15, 0.15]`,
  `ang_vel_z [-1.2, 1.2]`. Turning was raised from ±0.5 so the robot turns faster. Turn-in-place
  stays inside the servo envelope up to about 0.6 rad/s; at 1.2 rad/s the hip-yaw servo is at about
  150% (flagged, but allowed). These match the table grid in
  `robot_motion_reference/gait_studio/configs/robot_v2.yaml`. Gait Studio's feasibility analysis
  shows the old range (vx up to 0.5) is far beyond the STS3215 speed/torque envelope for a clean
  gait. If you widen the ranges, widen the bake grid too. Commands outside the grid get the edge
  reference.

### `tests/test_gait_table_reference.py`
Tests for the GPU sampler; numpy and torch only. They cover:
- exact interpolation of a synthetic linear table
- phase wrap
- clamping to the grid
- the standing frame and joint binding
- whether the real baked table, when present, matches the npz and is left/right symmetric

### `CHANGES.md`
This file.

---

## Modified files

### `utils/motion_reference.py`
Added `class GaitTableReference` at the end of the file. `MotionReference` is unchanged.
- It loads the baked npz to the GPU as a flat `[(vx, vy, wz, frame), D]` table, plus the
  standing frame, grid axes, period, field layout and joint names.
- `sample(commands[N,3], phase[N] in [0,1), standing[N] bool) -> [N, D]`. Interpolation is
  trilinear in the command (clamped to the grid) and linear in phase (wrapping). Standing envs get
  the stand frame.
- `bind_to_robot(names)` and `joint_field(ref, "joint_pos" | "joint_vel")` return values in
  articulation order. `field(ref, name)` returns any other field.
- Inputs are cast to float32 on the table's device.

### `envs/isaaclab/rewards/rewards.py`
Two jit kernels appended at the end of the file:
- `imitation_mse_exp(value, ref, std) = exp(-mean((value - ref)^2) / std^2)`
- `imitation_contact_match(feet_in_contact, ref_contact)`: the fraction of feet whose contact
  state matches `ref_contact > 0.5`

### `envs/isaaclab/biped_env_cfg.py`
1. In `BipedEnvCfg`, right after the `animation_*` fields, add:
   - `reference_npz_path: str | None = None`
   - `reference_stand_threshold = 0.05`
   - `reference_rsi_prob = 0.8`
   - `reference_rsi_joint_noise = 0.05`
   - `reference_std` (dict): `joint_pos 0.15`, `joint_vel 1.5`, `foot_pos 0.02`,
     `orientation 0.08`, `lin_vel 0.1`, `ang_vel 0.3`
2. At the end of the file, add the module-level function `apply_reference_config(env_cfg, env_conf)`.
   It copies the yaml `env_config.reference` block onto the cfg, and partial `std` dicts merge into
   the defaults. Both `train.py` and `play.py` call it, because the gait clock is a policy input
   and must be identical in both.

### `envs/isaaclab/biped_env.py`
1. Import `GaitTableReference` next to `MotionReference`.
2. In `__init__`, after the gait-phase buffers, add `self.ref_frame = None`. It holds the last
   sampled reference, for debugging and visualisation.
3. In `__init__`, right after the command generator, add
   `self._cmd_full_ranges = {k: tuple(v) for k, v in self.command_generator.ranges.items()}`.
   See fix (a).
4. At the end of `__init__`, after the clip-reference block, load `self.gait_ref` when
   `cfg.reference_npz_path` is set:
   - `bind_to_robot(self.robot.joint_names)`
   - print the grid, frame count, period and style digest
5. In `_get_observations`, the old speed-proportional phase update runs **only if
   `self.gait_ref is None`**.
6. New methods:
   - **`_reference_standing()`**: true when |cmd_xy| < thr and |wz| < thr.
   - **`_advance_gait_phase()`**: `phi += moving * dt * 2*pi / period`, mod 2π. The phase holds
     while standing.
   - **`_imitation_terms(feet_pos, feet_in_contact)`**: samples the reference at
     `(commands, gait_phase / 2π, standing)` and returns the seven `imitation_*` terms. It reads
     `robot.data` directly, because the cached `self.joint_pos` etc. are refreshed in
     `_get_observations` and are therefore one step stale in `_get_rewards`. The env's foot order
     is [right, left] and the table's is [left, right], so the code reindexes with `[:, [1, 0]]`.
   - **`_reference_state_init(env_ids)`**: for `rsi_prob` of the resets it:
     - writes the reference joints (plus uniform noise, clamped to the action limits) and joint
       velocities
     - sets base height = `env_origin_z + base_height + 5 mm`
     - derives roll/pitch from the reference projected gravity, keeping the yaw that was drawn
     - sets base linear/angular velocity from the reference
     - re-seeds `filtered_targets`, `servo_obs_buffer` and `previous_actions` (the reference pose
       mapped into action space)
7. `_get_rewards`:
   - When a table is loaded, call `_advance_gait_phase()` first. The reward then compares the
     post-step state with the reference at the post-step phase, and the next observation shows
     that phase.
   - Compute `imitation = self._imitation_terms(feet_pos, feet_in_contact)` and add each term to
     `reward_terms`, weighted by `self.cfg.rewards.get(name, 0.0)`. They are logged like every
     other term.
8. `_reset_idx`: call `self._reference_state_init(env_ids)` **after** `_resample_commands`, as the
   last line.
9. **(fix a)** `_apply_curriculum` command ramp: the full ranges now come from
   `self._cmd_full_ranges` (the configured ranges) instead of hardcoded vx (-0.3, 0.5), vy ±0.3 and
   wz ±0.3. The vx start range is clipped into the full range. Your current config's ranges equal
   the old hardcoded values, so its behaviour is identical.

Unchanged and still valid: the phase observation `[sin, cos]` (50-dim obs) and the mirror signs
(φ → φ + π). The table is built so that `sin > 0` means right stance, matching
`gait_phase_contact` and `_build_mirror_transform`. It is symmetrised at bake time, so the SAC
symmetry augmentation stays exact.

### `src/isaaclab/train.py`
1. Import `apply_reference_config` from `envs.isaaclab.biped_env_cfg`.
2. Call `apply_reference_config(env_cfg, env_conf)` just before the "Animation (imitation)" block.
3. **(fix b)** In the commands block, also copy `resampling_time_range`,
   `heading_control_stiffness`, `rel_standing_envs` and `rel_heading_envs` from
   `commands.base_velocity`. Previously only `ranges` was copied, so the yaml values for these
   never reached the env, and the cfg defaults applied instead: (10, 10) s, 0.3, 0.02, 1.0.
   **This changes your baseline runs:** `config.yaml` asks for 4–8 s resampling and heading
   stiffness 0.5.

### `src/isaaclab/play.py`
Import and call `apply_reference_config(env_cfg, env_conf)` at the top of the `env_config` block.
This keeps playback on the same gait clock as training.

### `src/isaaclab/common.py`
In `make_env`, call `apply_reference_config` after the reward scale. `make_env` currently has no
callers; the call is there for consistency.

### `config/schema.py`
In `EnvConfig`, add `reference: Optional[Dict[str, Any]] = None`.

### `src/isaaclab/export_onnx.py`
1. New helpers:
   - `_scan_block(path, block)`: an indentation-aware scalar scan, needed because the trainer's
     yaml does not `safe_load`
   - `_load_gait_clock(ckpt_dir, explicit, speed_freq)`
   - `_load_command_ranges(ckpt_dir, explicit)`
2. `deploy_config.json` gains a `gait_clock` object. With a table:
   ```json
   {"mode": "fixed_period", "period": 0.72, "stand_threshold": 0.05,
    "table": "...", "style_digest": "...", "phase_convention": "..."}
   ```
   Without one: `{"mode": "speed_proportional", "freq": 5.0, ...}`.
3. `command_ranges` now comes from the run's config instead of being hardcoded.
4. `gait_clock_freq` is kept for older runtimes.

### `envs/mujoco/biped_env_v2.py`
1. New module function `gait_clock_from_config(env_config) -> (period | None, stand_threshold)`.
   It reads `env_config.reference.npz_path` and takes the period from the table.
2. `__init__` sets `self.gait_period` and `self.gait_stand_threshold`.
3. `step()` uses the fixed-period, hold-while-standing clock when `gait_period` is set, and the old
   speed clock otherwise. This keeps sim-to-sim (`src/mujoco/sim2real.py`) consistent. The MuJoCo
   env does not compute the imitation rewards; it is an inference target.

---

## Deployment runtime (BipedRobotJetson): not edited here

The repo was not on disk. Make this change in the observation builder (`policy/observation.py`),
where the gait clock is advanced:

```python
gc = deploy_cfg.get("gait_clock", {"mode": "speed_proportional", "freq": deploy_cfg["gait_clock_freq"]})
if gc["mode"] == "fixed_period":
    thr = gc["stand_threshold"]
    moving = (math.hypot(cmd[0], cmd[1]) >= thr) or (abs(cmd[2]) >= thr)
    phase = (phase + moving * dt * 2 * math.pi / gc["period"]) % (2 * math.pi)
else:
    phase = (phase + dt * 2 * math.pi * gc["freq"] * math.hypot(cmd[0], cmd[1])) % (2 * math.pi)
obs_phase = [math.sin(phase), math.cos(phase)]
```

Advance the phase once per control step, before building the observation, the same as training.
Also clamp joystick commands to `deploy_cfg["command_ranges"]`.

---

## Workflow

```bash
# 1. design + bake (laptop, no Isaac)
cd robot_motion_reference && python -m gait_studio.app       # Bake tab, or: python -m gait_studio.bake
# 2. train
python src/isaaclab/train.py --task BipedV2 --config_path config/config_gait_reference.yaml
# 3. export: deploy_config.json now carries gait_clock + command_ranges
python src/isaaclab/export_onnx.py --config_path config/config_gait_reference.yaml
```

Things to watch in the first run:
- `Reward_Terms/imitation_*` should rise quickly; RSI helps.
- Velocity tracking should stay at or above baseline.
- `Torque/frac_saturated` should stay low.
- If a style uses trunk pitch, the reference conflicts with `flat_orientation_l2` and `upright`.
  Lower those or keep trunk pitch near 0.
- Set `actuator.armature` in `gait_studio/configs/robot_v2.yaml` to the value your BAM sim uses,
  so the Studio's feasibility checks match the training physics.
