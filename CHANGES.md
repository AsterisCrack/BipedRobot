# Gait-table imitation (ported to the BAM checkout)

These changes were first written against the old checkout. Everything outside
`robot_motion_reference/` is listed here. The retargeting submodule's changes (the `gait_studio/`
package, a README note, and the baked table in `processed/`) live in that repo.

## Sim-to-real: shakes and falls after 3-4 s (2026-10-12, run `BipedV2_SAC_2026-10-11_19-07-51`)

The policy was stable in both sims. On the robot at zero command it stood, shook harder and harder,
and fell. In MuJoCo, standing at zero command (4 runs, 10 s):

| servo | extra actuation delay | falls | body wobble |
|---|---|---|---|
| as trained (capped) | 0 / +20 / +40 / +60 ms | 0 / 0 / 1 / 4 | 0.12 / 0.40 / 2.6 / 3.6 rad/s |
| stiff (uncapped, like the bench) | +20 ms (what the Jetson had) | **4, at 1.7-7.7 s** | 4.0 rad/s |

Extra *observation* delay of the same size was tolerated. A flipped pitch gyro also fails this way,
but the gyro was confirmed correct.

**Causes and fixes:**
- **Jetson actuated one cycle late.** The bus loop wrote the PREVIOUS cycle's command before running
  the policy, so each action landed ~23 ms late; training had 0. Now the write comes after the policy
  hook (`BipedRobotJetson/hardware/servo_bus_manager.py`).
- **The trained servo is ~2x softer than the real one.** The stall cap is applied by raising R, which
  halves small-signal stiffness (17 vs 37 N-m/rad); the bench validated the uncapped fit. New sim-only
  DR `actuator.kp_fw_range: [32, 70]` (per robot, startup) spans both, keeping the cap. The robot
  still runs P = 32.
- **No latency margin.** `action_delay_steps_range: [0, 2]` (was [0, 0]; the old "[1,3] hurt" test was
  confounded) and `imu_obs_delay_steps_range: [1, 2]` (was [0, 1]; BNO055 fusion is never 0).
- **Jetson:** arm-time servo register audit, and every run recorded to `data/traces/` (downloadable from
  the Control tab).

Retrain with `config/config.yaml`. Policies trained before this assumed the old timing. The Jetson
fix alone also shortens latency for the current policy, which helps it too.

**Follow-up (2026-10-13): with both new terms at full width from step 0, the run learned to stand.**
Run `BipedV2_SAC_2026-10-12_12-32-16` vs the walking run: speed error while moving 0.30 vs 0.13 m/s,
foot clearance 1.0 vs 4.3 mm, about the same episode length. Walking was too hard to discover from
scratch. Both terms now follow the DR curriculum:
- **Delay:** action delay starts at 0 and reaches [0, 2].
- **Gain:** servo gain starts at P = 32 and reaches [32, 70]. It is now drawn per episode in
  `_reset_idx` instead of once at startup.

They reach full width at `curriculum_dr_full_steps` (40k control steps, ~164M transitions at 4096
envs). They opt in through `curriculum_dr_events: [..., "action_delay", "servo_stiffness"]`. Watch
`Curriculum/action_delay_max_steps` and `Curriculum/servo_kp_max` in TensorBoard.

Pushes were checked: they fire in training (58/64 envs pushed within 6 s, ~1.5 per env, ~0.39 m/s) and
are off in play by design. `play.py --pushes` turns them on to watch push recovery.

## Sim-to-sim parity (2026-10-11, run `BipedV2_SAC_gaitref_2026-10-10_18-40-27`)

**Physics is in parity.** With identical initial state and identical actions (no policy):

| test | Isaac | MuJoCo |
|---|---|---|
| zero-g floating, 88 steps | joint trajectory diff 0.0002 rad rms (max 0.0008), torques equal to 1e-3 | |
| standing on stand pose | joint err 0.004 rad, roll 0.2-0.6 deg, feet down 99% | 0.006 rad, 0.6 deg, 99% |
| open-loop walk at 0.1 m/s | joint err 0.033 rad, contact match 0.74, yaw drift -12 deg | 0.032 rad, 0.76, -12 deg |

So robot model, inertias, BAM servo and action pipeline match. Contact differs in detail (Isaac feet
are a convex decomposition, MuJoCo uses a soft-contact convex hull) but gives the same open-loop behavior.

**Fixed: the accelerometer the policy trained on was 4x too large (motion part).** Isaac Lab's `Imu`
divides the velocity change since its last recompute by the last physics dt. With lazy sensor
updates it recomputes once per control step, so it was a 20 ms delta-v over 5 ms. MuJoCo and the
robot's BNO055 report the real value, so every checkpoint before this fix saw a different
accelerometer in MuJoCo and on hardware. Now `biped_env.StepAveragedImu` (and MuJoCo, matching)
reports the mean acceleration over the control step: rms 5.95 vs 5.86, correlation 0.92 (was 0.67).
The current checkpoint in MuJoCo, fed the accelerometer it was trained on: with 1.5 deg lash, falls
go from 3/6 to 0/6 and vx from 0.015 to 0.072 m/s at a 0.1 command. **Retrain; checkpoints before this fix
must not go on the robot.** To evaluate an old checkpoint: `jitter_probe.py --legacy-imu`,
`sim2real.py --legacy-imu` (`imu_step_averaged: False` / `imu_acc_dt = PHYSICS_DT`).

**Fixed: `curriculum_*_ramp_steps: 0` gave the START range at step 0, not the full one.** The first
reset of every run sampled vx 0-0.2, vy = wz = 0 (and the smallest init-pose range).

**Remaining gap: backlash.** Isaac's lost-motion model barely changes behavior. MuJoCo's lash hinge
(`solreflimit` = 2*dt) adds chatter and wobble: wobble 1.4 -> 1.6/1.7/1.9 rad/s for lash
0.5/1.0/1.5 deg on the current checkpoint, with no falls. Measure the real lash with the Jetson's
SysID "Screen all joints" and set `lash_deg_range` / `mujoco_lash_deg` from it.

Yaw drift with a fixed `--wz 0` is expected: training commands mostly come from heading mode, so the
policy tracks a yaw rate, not a heading. Without a heading loop (the operator, on the robot) slow drift accumulates.

## Jitter investigation (2026-10-10, run `BipedV2_SAC_gaitref_2026-10-10_12-46-59`)

**Where the jitter comes from: the network.** `src/isaaclab/jitter_probe.py` runs a checkpoint
deterministically, on flat terrain, at a fixed command, and reports per-joint chatter. On the
184M-step checkpoint at vx = 0.1:

| setting | rms action change / step | body wobble [rad/s] | vx reached |
|---|---|---|---|
| Isaac baseline | 0.48 | 1.20 | 0.045 |
| Isaac, backlash off | 0.46 | 1.28 | 0.058 |
| Isaac, action noise off | 0.48 | 1.20 | 0.070 |
| Isaac, both off | 0.46 | 1.33 | 0.061 |
| MuJoCo, lash 1.5 deg | 0.52 | 3.49 (1 fall in 4) | 0.165 |
| MuJoCo, lash off | 0.44 | 2.03 | 0.042 |

The action, on a [-1, 1] scale, swings by ~0.45 every 20 ms with every simulator suspect removed
(about 50 deg of knee target per step). Backlash adds to it in MuJoCo, where it roughly doubles
body wobble. The training logs agree: `action_rate_l2` = -5.7/s at weight -0.05 is the same ~0.44.

**Fixed:**
- **The policy could not see its own last action.** `previous_actions` was advanced in `step()`
  after the observation was built, so the obs that chose a_t held a_{t-2}, while `action_rate_l2`
  penalised a_t - a_{t-1}, a difference the policy never observed. Now advanced at the end of
  `_get_rewards` (MuJoCo: before `_build_obs`), so the obs carries a_t. That is what the Jetson
  already feeds, so this was also a sim/real mismatch. **Checkpoints trained before this fix are
  incompatible; retrain.** Test: `tests/test_prev_action_obs.py`.
- **Backlash knock** capped at the servo's capped stall torque (`impulse_cap` 6.0 -> 2.94, also
  clamped in `bam_actuator.py`). 6.0 was sized for the uncapped 6.3 N-m fit.
- **Heading-mode yaw.** New `commands.base_velocity.heading_max_ang_vel` (0.5 in the gait-table
  config). With `ang_vel_z` at +-1.2 for turn-in-place, heading control turned most moving envs at
  ~1 rad/s. The heading rate is now also applied right after a resample, so the first obs and the
  RSI reference see the real command, not the uniform +-1.2 draw.
- `push_interval_s` / `push_vel_range` were never wired to anything. They're no longer copied, and
  the schema says so. The real push is `events.push_robot` (+-0.4 m/s x, every 1-5 s); override it
  through the `events:` block.
- Action noise is off in play/eval. `play.py` and `sim2real.py` default to the checkpoint's own
  `config.yaml`.

**Next, one variable per run** (after a run with the fixes above):
1. `action_rate_weight_end` -0.05 -> -0.3
2. `backlash.enabled: false`
3. `events: {push_robot: {interval_range_s: [8, 15], params: {velocity_range: {x: [-0.15, 0.15], y: [-0.1, 0.1], yaw: [-0.3, 0.3]}}}}`
4. `use_rough_terrain: false`
5. Re-bake with more `step_height`, if it still shuffles (the style lifts only 1 cm).

Not changed: RSI drops the robot from env_origin + table height + 5 mm, the same exposure to terrain
relief as the normal reset (fixed 0.29 m). Slope envs spawn on a flat 1.5 m platform. Item 4 removes
it. Known and deferred: in `torch_rl_algorithms`, symmetry mirroring is applied after observation
normalisation, with non-mirrored stats.

## Status: ported to the BAM checkout

Differences from the original notes below (they are accurate for the old layout only):
- **Config plumbing.** There is no `apply_reference_config`. `src/isaaclab/env_config.py`
  (`_apply_reference`) is the one place `env_config.reference` reaches `BipedEnvCfg`, for train,
  play and `make_env` alike, so `train.py`, `play.py` and `common.py` are unchanged. RSI is forced
  off when `training=False` (playback starts from the default pose). Fix (b) (command keys
  reaching the env) already existed in a different form there.
- **RSI placement.** `_reference_state_init` is the last call in `_reset_idx`, after
  `_resample_commands`. It writes joints before anything reads them and does **not** touch the BAM
  servo: its reset is deferred, and the first `motor()` call seeds `q_smooth`, the command-delay
  ring and the back-EMF state from the pose and velocity written there. It re-seeds
  `filtered_targets` (encoder space, `+ mount_offset`), `action_delay_buffer`, `servo_obs_buffer`,
  `previous_actions` (via the new `targets_to_actions`) and `feet_in_contact_prev`.
- **Action history** is advanced at the end of `_get_rewards`, before `_reset_idx`, so the RSI
  seed survives into the first observation (see "Jitter investigation" above; this replaces the
  earlier `step()` keep-mask).
- **Export.** `deploy_config.json` carries `gait_clock` and the run's own `command_ranges`
  (`export_onnx.py`: `_scan_block`, `_load_gait_clock`, `_load_command_ranges`). A table-trained
  bundle is `format_version` 3, so a runtime that only knows `gait_clock_freq` refuses it instead of
  clocking the policy wrongly. The exporter warns if the table was re-baked after the checkpoint.
- **Command ranges.** Forward speed is capped at 0.2 m/s in both the bake grid and
  `config_gait_reference.yaml` (vy +-0.15, wz +-1.2). Keep the two equal.
- **Run command.** `--task` is `BipedRobotV2` (the default `BipedRobot` is the V1 robot):
  `python src/isaaclab/train.py --task BipedRobotV2 --config_path config/config_gait_reference.yaml`.
- **Servo model.** Gait Studio now judges gaits against the same BAM m6 servo as training, read
  from `actuator/params/sts3215_12v/m6.json` (see `robot_motion_reference/gait_studio/README.md`).
  Its feasibility numbers are worst-frame and conservative; the open-loop physics preview is the
  better check of whether the servos can follow a style.

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
python src/isaaclab/train.py --task BipedRobotV2 --config_path config/config_gait_reference.yaml
# 3. export: deploy_config.json now carries gait_clock + command_ranges
python src/isaaclab/export_onnx.py --config_path config/config_gait_reference.yaml
```

Things to watch in the first run:
- `Reward_Terms/imitation_*` should rise quickly; RSI helps.
- Velocity tracking should stay at or above baseline.
- `Torque/frac_saturated` should stay low.
- If a style uses trunk pitch, the reference conflicts with `flat_orientation_l2` and `upright`.
  Lower those or keep trunk pitch near 0.
- `Torque/frac_saturated` matters more under BAM: the duty ceiling and back-EMF saturate easily, and
  the 2.94 N-m stall cap is conservative rather than measured. Compare against the 0.39 of the
  working non-imitation run; `Actuator/mean_abs_friction` shows what the servo spends on friction.
- Gait Studio takes the servo from the same BAM params file as training, so there is no armature
  to keep in sync by hand. Keep `gait_studio/configs/robot_v2.yaml` `actuator:` equal to
  `env_config.actuator` in the training config (params file, `kp_fw`, stall cap).
