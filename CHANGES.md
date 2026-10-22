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

## Real-trace follow-up (2026-10-14)

Trace of `BipedV2_SAC_2026-10-13_19-42-03` @122.88M at cmd 0: action change 0.41/step (sim 0.04), 73% of steps clipped by the Jetson
limiter. Deploy pipeline verified (ONNX replay 1e-6, obs in distribution). MuJoCo reproduces the real chatter with +2 steps of action
delay on top of the trained [0,2]; backlash, speed quantization, history seeding and the limiter alone do not.

- `action_delay_steps_range: [0, 2]` -> `[1, 3]` (still ramped by the DR curriculum).
- New `action_rate_limit` (env_config, default 0 = off; `config.yaml` 0.5). Applied in `_pre_physics_step` against the previously applied
  action, before the EMA -- same rule as the Jetson `SafetyMonitor.limit_action`. Exported as `max_action_rate`; the Jetson takes it from
  the bundle (older bundles keep its default 0.5). The servo slew limit (4.9 rad/s ~ 0.05 action/step on the knee) is far tighter than
  0.5, so this is a safety clip, not a smoothing tool. Note SAC stores the policy's raw action, not the clipped one. [SUPERSEDED 2026-10-18: the in-env limiter code was removed from training and the exporter no longer writes `max_action_rate`; the Jetson keeps its own 0.5 safety clip, and a bundle value is used only if it is > 0.]
- `action_rate_weight_end` -0.05 -> -0.1 (-0.3 stopped walking before).
- Jetson: history is zero-filled at start (as in training) instead of seeded with copies of the standing frame.

## 25 Hz bang-bang (2026-10-15)

Trace `BipedV2_SAC_2026-10-14_20-58-31` @49.15M at cmd 0. The raw policy output flips sign every step (lag-1 autocorrelation -0.90),
and the Jetson 0.5 limiter fires on 100% of steps, so the applied action is a two-value square wave. Targets jump 13-23 deg/step,
against the servo's 5.6 deg/step slew limit. MuJoCo reproduces it: vx 0.00 at cmd 0.2, which is why it "refuses to walk".
- **Cause:** the in-env `action_rate_limit` added on 10-14. SAC stores the raw action, so anything past the band executes the same, and the
  policy collapsed onto "opposite extreme of my last action". The prev-action Jacobian (newest frame) has spectral radius 3.0
  (10-13 run: 1.04, marginal). It is now `action_rate_limit: 0`. The Jetson keeps its 0.5 safety clip (a bundle value of 0 no longer disables it).
- **Correction to the 10-14 entry:** the "+2 steps of delay reproduces the real chatter" MuJoCo result was an artifact. The scratch script
  put the DELAYED action in the policy's prev-action slot. With Jetson semantics the 10-13 policy is fine at +3 steps in MuJoCo, so the
  real-robot chatter of that run is not explained by delay. `action_delay_steps_range [1, 3]` stays, but it is not evidence-based.
- **New `src/mujoco/policy_gate.py`:** pre-deploy gate on the exported bundle. It reports the prev-action spectral radius and runs a MuJoCo
  closed loop with the Jetson limiter and +0..3 delay. It fails the 10-14 bundle; the 10-13 bundle fails on radius only (1.04). Run it
  before every deploy.
- **New CAPS loss** in the SAC actor (`train.caps_temporal_weight`, `caps_spatial_weight`, `caps_spatial_sigma`; default off). It uses the true
  1-step next obs (separate replay key, because `return_steps: 5` makes `next_observations` n-step) and masks resets.
- **Jetson:** stale-sensor detection. Per-servo and IMU stamps feed `check_observation(sensor_age_s)`, so 8+ missed 50 Hz cycles fault
  STALE_SENSOR. Missing servos refuse arm/start.

## Low-speed walking: fixed-period clock + residual action map (2026-10-17)

No run since 10-11 walks at command 0.1 (new `policy_gate.py --cmds` / `sweep_all.py` sweep); most only move from 0.15-0.2 up. Even the "walkers"
(10-13_19-42-03 at 0.2-0.3) step at ~7 touchdowns/s with 70-100 ms flights and no aerial phase (debounced), i.e. a shuffle/vibration, not a gait
(the baked table is ~2.4 touchdowns/s). The sweep tool evaluates each checkpoint with its own run config (clock, map).
- **Fixed-period gait clock** (`gait_clock_mode: fixed_period`, period 0.7 s, held below `gait_clock_stand_threshold` for both |v_xy| and |wz|). The old
  clock ran at 5*|v| Hz: a 2 s cycle with 1 s single support at 0.1 m/s. Isaac advances it in `_get_rewards` (same order as the table path), MuJoCo in
  `step()`, Jetson already had it. Export writes the v3 `gait_clock` block. Old run configs lacking the keys stay `speed_proportional`.
- **Residual action map** (`action_map: residual`): `target = clamp(home + scale_j * a)`, home deg hip_yaw 0 / hip_roll 0 / hip_pitch -10 / knee -26 /
  ankle_roll 0 / ankle_pitch -16 (flat sole: hip - knee + ankle = 0), scale rad 0.6 / 0.45 / 0.8 / 0.6 / 0.45 / 0.6, spawn z 0.286. Checked against all 819 baked
  gaits: max |a| 0.90. 0.1 of action is now 2.6-4.6 deg (was 3.5-6.25, 24x lopsided at the knee). `default_joint_pos` = home, so obs, `joint_deviation_*` and `dof_pos_l2`
  follow it. Export format v4 (`default_joint_pos`, `action_scale_rad`); Jetson parses it, subtracts the bundle default in `joint_pos_rel`, and arms to home.
  Old checkpoints/bundles are incompatible. `centered` still works.
- TODO (hardware): bench-check the crouch on the robot with feet off the ground; `robot.yaml default_position_deg` must be the servo reading at URDF zero.
  `robot.yaml` `crouch` is still wrong (left untouched).

## Reward economics for low speed (2026-10-17)

Run `2026-10-17_13-56-29` (residual map + fixed-period clock): the MuJoCo sweep shows the early "walking" (12M: vx/cmd ~0.8 at 0.1-0.3) is a ~7 touchdown/s
vibration with raw action changes of 0.8-1.06 per step; by 123M (action_rate weight -0.1, delay 3, DR 1) the action change is 0.002 and the robot stands
motionless at every command. Standing earned most of the reward: tracking exp(-e^2/0.2^2) pays 0.78 for standing at cmd 0.1 and yaw exp(-e^2/0.5^2) pays 0.85-0.96.
- **Relative tracking kernels** (`track_lin_std_rel/min` 0.7/0.07, `track_ang_std_rel/min` 0.7/0.15): std = max(rel*|cmd|, min). Standing at cmd >= 0.1 now earns ~0.13,
  half speed ~0.6. Old fixed std when rel = 0 (default).
- **Touchdown-paid air time** (`feet_air_time_mode: touchdown`, weight 3.0): paid once at landing, clamp(air - 0.15, 0, 0.25) s, single stance, |cmd_xy| > 0.05. A 0.1 s flicker
  pays 0, a foot held up forever pays 0, a 0.3 s swing pays 0.45 (weight included).
- **swing_foot_height re-enabled** (weight 1.0) but saturating at 2 cm (`swing_max_clearance` 0.02; baked gait lifts 1.0-1.4 cm) and only while the swing lasted <= `swing_max_time_s` 0.5 s.
- MuJoCo env keeps the old per-step air-time and fixed-std tracking (reward magnitudes differ there; no crash).
- `policy_gate.py --rate 0` now means "no limiter" (it froze the actions before).
- Open: the action-rate ramp (-0.01 -> -0.1) and the vibration local optimum. Candidates: CAPS temporal (`train.caps_temporal_weight`), a higher starting action-rate weight.

## Jetson sync-up (2026-10-18)

Audit of `BipedRobotJetson` against the v4 residual bundle / fixed clock / no in-env limiter; a real v4 bundle exported from run `13-56-29` loads, matches the training
action formula, and passes `policy_gate.py` at command 0 (spectral radius 0.95). Fixes (verified, 134 Jetson tests pass):
- Web/WS motion guard: servo moves and register/calibration writes return 409 while ARMING/ARMED/RUNNING and during the whole arm() sequence (torque disable and estop stay open); `/servos/simulation` gated too.
- arm(): present positions are written as goals (ARM_SPEED) before torque is enabled; refuses on stale/missing servo stamps. **Bench-check first with the feet off the ground.**
- Limits: bundle rate used only if > 0 (clamped 0.05..2.0, else 0.5), reset on every load; tilt = min(bundle, 0.784); bounds on the limit API; UI sliders initialised from the runner and send only the changed field.
- Status line (format / action map / clock / home), fuller trace meta, selftest zero-fill + home comparison, docs (CLAUDE.md, README).
- Not changed: `config/robot.yaml` poses / `default_position_deg` (stand = straight legs, `crouch` wrong) — yours to recalibrate.

## Posture exploit: edge standing + leg bracing (2026-10-18)

Real traces of `17-08-33` @245M (4 runs, cmd 0): at step 0 the robot is exactly at home (calibration OK), the policy's first action drives both ankle rolls to ~+26 deg (action
saturated) and the robot falls forward within 1-5 s; ONNX replay of all traces matches to 1e-6 (no pipeline bug). MuJoCo reproduces the same pose and never falls: standing on the
INNER foot edges (sole tilt ~14 deg), hip_roll -10.8 deg, foot origins 17 mm apart, with the two ankle links pressing on each other (5.9 N, mostly lateral; feet carry the full vertical
load). The old centered-map policy (10-13, also fell on hardware) braces even more (foot-foot 15 N, ankle-ankle 7 N, outer-edge standing). So inter-leg bracing is common to the policies that fail.
- Why nothing stopped it: `undesired_contacts` reads the floor-filtered force matrix, so leg-on-leg contact is invisible to it; `foot_separation` measures the foot link origin (the
  ankle_pitch joint, 31.6 mm above the sole) with a one-sided 0.09 m target (= ~132 mm sole spacing, wider than the 105 mm hips; the gait table's 0.09 is sole-centre spacing) and a tiny
  weight (<= 0.09/step); nothing sees sole flatness.
- New (all switchable, shipped on in `config.yaml`): `foot_flat_l2` -3.0 (sole roll tilt while in contact); `foot_separation_mode: sole_band` (two-sided on the sole-centre spacing: target 0.095 m,
  tol 0.025, weight -0.1); `leg_contact_l2` -0.5 (net force on non-foot leg links; foot-on-foot is NOT covered with rough terrain on, because the floor filter is cleared); `undesired_contacts`
  -1.0 with a 5 N saturation; CAPS on (temporal 1.0, spatial 1.0, sigma 0.1; about +1 GB replay memory and an estimated +20-35% per SAC update).
- `policy_gate.py` now also fails policies with a saturated posture (|a| > 0.95 on > 40% of steps) or inter-leg contact > 1 N; it fails both `17-08-33` and `10-13`. A sideways-gravity-bias line is
  informational only (it did not discriminate: 10-13 passes it yet fell on hardware).
- Watch in the first Isaac logs: `leg_contact_l2` should be ~0 at flat standing; the peak penalty with thrashing random legs can be large vs `termination_penalty` -0.2 (early-termination incentive).

## Gait-table config synced with config.yaml (2026-10-18)

`config/config_gait_reference.yaml` had last been edited on 2026-10-14. It now carries everything learned since, on top of its `reference:` block and `imitation_*` weights:
- **Residual action map**, home = the baked table's stand pose rounded (deg: yaw 0, roll -2.5, hip_pitch -8, knee -22, ankle_roll 2.5, ankle_pitch -14; flat sole), scales rad 0.6 / 0.4 / 0.75 / 0.56 / 0.45 / 0.55,
  spawn 0.286. a = 0 is the reference stand (imitation_joint_pos 0.998 at a = 0; the config.yaml crouch home would give 0.906). Max |a| over all 819 baked cells (both legs, clamped to the limits) <= 0.85.
  The table's lateral cells and ankle_pitch's -35 deg limit are the extremes. The map is a different contract from config.yaml's: old checkpoints/bundles do not load.
- Relative tracking kernels, `foot_flat_l2` -3.0, `leg_contact_l2` -0.5, `undesired_contacts` -1.0 with a 5 N saturation, `foot_separation` -0.1 in `sole_band` mode, CAPS 1.0 / 1.0 / 0.1,
  `action_delay_steps_range [0, 3]` + `imu_obs_delay_steps_range [1, 2]` + `kp_fw_range [32, 70]` ramped by the DR curriculum (20k steps), action-rate schedule -0.01 -> -0.1, `ang_vel_z` +-1.2.
- Kept different on purpose: `standing_envs_start` 0.10, stronger stability weights (flat_orientation -2.0, lin_vel_z -2, upright 0.5), hand-written gait terms zeroed, `lin_vel_y` +-0.15 (= bake grid).
- `foot_sep_target_m 0.105 / foot_sep_tol_m 0.055` (band 50..160 mm instead of 70..120): the table's own lateral steps (|vy| >= 0.1) move the sole centres 0.048-0.175 m apart, so the default band was violated in 43% of table frames.
  Those same cells make the two legs touch (ankle/foot collision geoms overlap by up to 16 mm in ~6% of their frames), so `leg_contact_l2` costs a little there (<= ~0.06/step): re-bake with a wider stance or cap `lin_vel_y` at +-0.05 if it matters.
- Open-loop MuJoCo replay of the table (feeding `targets_to_actions(table q)` for 5 s): joint tracking rms 1-4 deg, no action clipping. A verifier's replay (backlash off, 10 s): in-place, vx 0.05, vx 0.1, vx -0.1, wz 0.6 and vy 0.1 stay up; vx 0.15 falls at 7.8 s, vx 0.2 at 1.9 s, (0.1, 0.05) at 2.0 s. The table is not self-balancing (its own report flags 819/819 cells: ZMP margin negative everywhere), so the policy must add balance feedback, especially for vx >= 0.15 and mixed cells. An earlier claim that vx 0.05/0.1 fall did not reproduce.
- `tests/test_residual_action_map.py`: the reference config is residual now; new test checks the map covers every baked cell.

## Gait-reference tuning: foot lift and imitation share (2026-10-19)

Run `gaitref_2026-10-18_17-42-34` transfers to the real robot, but lifts the feet ~0.8 mm (table: 10-14 mm) so they slide, and imitation terms are 63% of the
positive reward. Causes: foot height above ground was never imitated (`imitation_foot_pos` is base-frame and averaged over 6 values; an 11 mm miss costs ~1%);
`imitation_contact` and `feet_slide` used `|F| > 1 N`, so a foot dragged with < 1 N counted as airborne (it earned the contact reward and escaped the slide penalty).
- New `imitation_foot_height` (weight 0.5, `reference.std.foot_height` 0.004 m): robot sole clearance (foot z above its last force-contact z) vs the table's `foot_height`.
- New `contact_height_thresh_m` (0.002): for `imitation_contact` and `feet_slide` only, a foot also counts as in contact when its clearance is < 2 mm.
- Imitation weights cut: joint_pos 1.5 -> 0.6, foot_pos 1.0 -> 0.4, orientation 0.5 -> 0.2 (overlaps `upright`/`flat_orientation_l2`); expected share ~50%.
- `Diagnostics/dragged_foot_frac` logs the exploit. Known: after a mid-swing RSI reset the swinging foot reads 0 clearance until its first touchdown (~1% of an episode).

## Wrong-sign foot_flat_l2 (2026-10-19)

Run `gaitref_2026-10-19_20-20-42` degraded after ~40M (sole spacing 0.079 -> 0.041 m, imitation_joint_pos 13.8 -> 2.6, torque saturation 0.08 -> 0.20) while
`Reward_Terms/foot_flat_l2` rose from 0.7 to ~10: the config had `foot_flat_l2: 1.0`, and the term is >= 0 (sin^2 of sole roll), so it REWARDED edge standing (estimated
RMS sole roll ~6 deg at 10M, 20-30 deg after 70M). An earlier, smaller dip (18-38M) coincides with the DR ramp reaching full at ~41M. Fixed to -1.0.
- New reward-weight sign guard (`rewards.REWARD_WEIGHT_SIGN`, `validate_reward_weight_signs`): the Isaac env refuses to start if a penalty weight is > 0 or a reward weight
  is < 0 (also `action_rate_weight_start/end`). The MuJoCo env checks only the keys it implements, so `policy_gate.py` can still load old checkpoints. `play.py` on the
  10-19 checkpoint now needs `--config_path` (its own config has the wrong sign).
- That run also had `use_rough_terrain: true`, which hides foot-on-foot contact from `leg_contact_l2`.

## Foot-lift gradient (2026-10-20)

Run `gaitref_2026-10-20_09-32-16` still shuffles: raw swing clearance ~5 mm in the first few M steps, ~2 mm from ~7M, ~1.3 mm from 40M; `dragged_foot_frac` 0.16-0.20.
`imitation_foot_height` at std 4 mm gave no gradient from dragging (1 mm vs a 10 mm target scores 0.006, 3 mm 0.047).
- `reference.std.foot_height` 0.004 -> 0.010 (1 mm -> 0.44, 5 mm -> 0.78, 10 mm -> 1.0, 20 mm -> 0.37); weight 0.5 -> 1.5 (lift-vs-drag gap ~0.31/step, was ~0.22).
- `contact_height_thresh_m` 0.002 -> 0.004 (note: `dragged_foot_frac` uses the same threshold, so it is not comparable with earlier runs).
- `upright` 0.5 -> 0.2 (~95% saturated: a near-constant bonus).

## Asymmetric gait: mirror loss, symmetric normaliser stats, critic_2 normaliser bug (2026-10-20)

The gait-reference policy lifts one leg and drags the other (yaw drift). Measured: the gait table is exactly mirror-equivariant and the reward terms are L/R-consistent,
but the ACTOR itself is asymmetric (mirror error |pi(Ms) - M pi(s)| ~0.6x |a| at 49M, ~0.3x at 221M; the transferred 10-18 policy still drags R and drifts ~100 deg
in 8 s in MuJoCo). Symmetry augmentation alone does not enforce symmetry.
- New SAC mirror loss `train.symmetry_loss_weight` (gait reference config: 2.0): w * mean(sum_a (mu(M_s s) - M_a mu(s))^2) on the deterministic mean, same mirror as the augmentation.
  Logged as `train/actor_symmetry_loss`. Needs symmetry_augmentation on (raises otherwise). With `--video` the RecordVideo wrapper hides mirror_obs from model.py, so it now
  fails loudly instead of silently dropping the augmentation.
- The augmentation mirrors stage-1-NORMALISED observations, which is exact only if the normaliser stats are mirror-symmetric. They were not: the critic's foot-force slots had
  R/L sigma ratios 1.3-1.6. Now the stage-1 RunningMeanStd (src/isaaclab/common.py, flag from train.py when symmetry_augmentation) and the stage-2 network MeanStd (ddpg.py) also
  record the mirrored observations, so the stats are symmetric and mirroring is exact.
- BUG: critic_2 was a deepcopy of critic_1 with a PRIVATE input normaliser that was never updated (identity for the whole run: critic_2 saw un-normalised inputs while
  critic_1 saw normalised ones). Fixed in torch_rl_algorithms/models/networks.py (shared normaliser), with a load_state_dict override so old checkpoints don't overwrite
  critic_1's stats with critic_2's identity ones.
- Start a FRESH run: resuming keeps the old asymmetric stage-1 stats (diluted only as 1/count) and shifts critic_2's input scale.

## Limp: exactly equivariant actor + worst-foot rewards (2026-10-20)

Ground-truth test in MuJoCo (physically mirrored robot state): the mirror function is CORRECT (obs error 0, mirrored actions give mirrored trajectories to 1e-6; joint
signs +1 confirmed from the MJCF axes). The policy limps with the SAME (left) leg from mirrored starts (15/16 random start pairs): the actor itself is not mirror-equivariant
(|pi(Ms) - M pi(s)| ~22% of |a| with augmentation + mirror loss). Projecting it, 1/2(pi(s) + M pi(Ms)), removes the limp. Open-loop table replay reaches the table's
10-14 mm lift with the servo model, so the hardware is not the limit. (The MJCF base_link COM is 1.74 mm off-centre in y; negligible here.)
- `train.equivariant_actor: true` (torch_rl_algorithms/models/equivariant.py): the actor outputs loc = 1/2(loc(s) + M loc(Ms)), scale = 1/2(scale(s) + P scale(Ms)), exactly
  equivariant in training and in the exported ONNX (Jetson unchanged; ~2x actor compute). Mirror tables are checkpoint buffers. Plain checkpoints are remapped on load.
  `symmetry_loss_weight` is 0 (identically zero with the projection); symmetry augmentation stays on.
- `imitation_foot_agg: min`: imitation_foot_height and imitation_contact take the WORSE foot (a limp now costs ~0.84/step vs a perfect gait, was ~0.49).
- `contact_height_mode: soft`, `contact_height_thresh_m` 0.006: contact score max(force, 1 - clearance/6 mm) instead of a hard 4 mm cliff (feet_slide weight and imitation_contact).
- RSI reset datum on flat terrain: env origin z + 0.0316 m (measured foot-origin rest height), so a mid-swing reset foot no longer reads 0 clearance.
- Old runs with symmetry_loss_weight > 0 in their config.yaml need it set to 0 (in a copy) for policy_gate / sim2real to construct the model.

## Command tracking vs gait sway: reference-velocity target (2026-10-20)

`track_lin_vel_xy_exp` compares the INSTANTANEOUS base xy velocity with the constant command (kernel std = clamp(0.7 |cmd|, 0.07)). Walking sways, and the gait table does too:
per cycle its lateral velocity has std 0.062-0.072 m/s and its forward velocity 0.03-0.05 m/s, while the cycle mean equals the command exactly. So the reward fights the reference:
a PERFECT copy of the table scores 0.38 at vx 0.1 (standing still: 0.13) and 0.28 at vx 0.05 (standing: 0.60, standing beats perfect walking).
- MuJoCo run 17-25-00, step 61.44M, command 0.1: mean vx 0.073, but the instantaneous kernel scores 0.40 against 0.86 on the mean velocity. A vy 0.1 command is not tracked at all (0.004).
- `track_lin_target: command | reference` (default `command` = old behaviour, bit-identical). `reference` (needs the gait table and `track_lin_std_rel` > 0, else ValueError at env init)
  centres the kernel on the table's `base_lin_vel_b` xy at the current (command, phase); the width stays clamp(rel |cmd|, min) from the COMMAND. The measured velocity is the fresh
  `robot.data.root_lin_vel_b` (`self.base_lin_vel_b` is refreshed in `_get_observations`, one step stale at reward time; `command` mode keeps using it). A perfect copy scores 1.0;
  standing scores 0.08-0.33 (forward commands: not above the old value), so the walking-minus-standing gap is >= 0.67 at every tested command (old: -0.32 at vx 0.05, 0.25 at vx 0.1).
- `_get_rewards` now samples the table frame once, right after the clock advance, and shares it between this term and `_imitation_terms` (new optional `ref` argument): same phase for both.
  `track_ang_vel_z_exp` is unchanged (the table's yaw-rate sway is ~0.005 rad/s). `Diagnostics/vel_err_moving_ms` stays the honest error against the command.
- `config_gait_reference.yaml` sets `track_lin_target: reference`. The MuJoCo env's own reward (fixed std 0.15) is not changed. Start a fresh run: the reward scale of `track_lin_vel_xy_exp` changes.
- Not a skill regression: `Reward_Terms/track_*` in run 17-25-00 peak at ~21M and then decay (`Diagnostics/vel_err_moving_ms` 0.16 -> 0.21). That coincides with the DR/delay curriculum
  completing (delay max 3 at 35M, dr_scale 1 at 42M), and clean MuJoCo replays of step 24.58M and 61.44M are equal within seed noise.
- Tests: `tests/test_track_lin_reference_target.py` (real table: perfect copy 1.0, standing gap, old-mode bug documented, validation, config, call order).

## Human gait scaling: speed-dependent cadence, rocker-model heel-to-toe, direction-aware side steps (2026-10-21)

Gait Studio (submodule, see its README) now follows human gait laws scaled to this leg (L = 0.21 m hip-pitch height; the whole command range is Fr <= 0.02, human very slow walking):
- **Cadence follows the command** (`gait_studio/cadence.py`): s/L = 1.15 v'^beta (Grieve 1968, Hof 1996; 1-beta = `cadence_exponent` 0.5), sideways scored at 1/0.45 of its speed
  (Handford & Srinivasan 2014), turning capped at `max_step_yaw` per step. Cycle 0.90 s / 9 cm steps at vx 0.2, 1.27 s at 0.1, 1.80 s at 0.05; `period_max` (default 2.0 s, the
  human law) and `period_min` are style sliders. A style saved with the old global `period` loads as period_min = period_max = period (same cadence as before).
- **Heel-to-toe from the rocker model** (`gait_studio/rollover.py`, Perry's three rockers): the stance foot's pitch follows the shank, computed from the plan (hip from the CoM target,
  ankle from the footstep, two-link knee). Heel strike = the shank's lean at touchdown (~19 deg at vx 0.2, ~10 deg at 0.1, as in humans heel-strike angle ~ leg angle), heel off
  when the lean passes `heel_off_angle` or the hip is beyond `max_leg_reach`; toe strike / heel last backward; none sideways or in place. Replaces `heel_strike` / `toe_off`.
- **Side steps**: pelvis twist scales with the forward step length (none in place or sideways), swing lift drops to `lateral_lift` (closing step 1/1.5 of that), no outward bulge,
  default 2 deg hip hike on the Sideways anchor.

Pipeline (the cycle length now depends on the command):
- **Table v2** (`gait_studio/bake.py`): cycles phase-normalised (`n_phase` 64 samples) plus `freq[nvx, nvy, nwz]` cycles/s, symmetrised (freq(vx,-vy,-wz) == freq(vx,vy,wz), which
  the obs-mirror augmentation needs). The `dt` key is gone (an old env would now fail loudly instead of running one period).
- **Isaac** (`utils/motion_reference.py` `GaitTableReference.frequency`, `biped_env.py` `_advance_gait_phase`): phi += step_dt 2 pi freq(cmd), freq trilinear in the command and
  clamped to the grid like the table. v1 tables give a constant freq (unchanged behaviour).
- **MuJoCo** (`envs/mujoco/biped_env_v2.py`): `gait_clock_from_config` returns a numpy `CadenceGrid` (constant for v1 tables and `fixed_period`).
- **Export** (`export_onnx.py`): a v2 table with a varying cadence exports `gait_clock.mode: cadence_grid` (axes + freq) as **bundle format v5**; a constant cadence stays
  `fixed_period` (v3/v4). **Jetson** (`policy/cadence.py`, `deploy_config.py`, `observation.py`): accepts v5, validates the grid (increasing axes, shape, 0.2-5 Hz).
- Parity: `tests/data/cadence_grid_golden.json` (and its copy in BipedRobotJetson/tests/data) is checked against scipy's interpolator, the Isaac torch path, the MuJoCo helper
  and the Jetson clock. Tests: `tests/test_gait_table_reference.py`, `tests/test_fixed_period_clock.py`, `tests/test_cadence_export.py`, Jetson `tests/test_cadence_clock.py`.
- Untrained. Checkpoints trained on a v1 table keep working with it (and with v3/v4 bundles). A v2 table needs a fresh run. One change per run: bake with
  `rollover_gain: 0` first (cadence only), then the rocker heel-to-toe, then the side-step knobs.

## Gait Studio: turning "hula hoop" and lateral sway (2026-10-21, untrained)

- **Turning circled.** The body yaws over the planted stance foot, so in the body frame the foot slides front to back. With the ZMP held at the foot centre, the CoM followed:
  - body-frame fore-aft p-p 16-60 mm turning in place, against 3 mm walking straight;
  - with `sway_gain` 0.3 it became a 13 x 12 mm circle.
  - The rocker amplified it: 1° pitches threw the ZMP ~5 cm onto a pivot edge, the CoM then moved the hip and the next pass asked for 8°.
- **Fixes.**
  - The centre of pressure now rolls along the foot to cancel the rotation (`footsteps._turn_shift`).
  - Rocker pitches < 1.5° become flat (soft deadband).
  - The ZMP reaches the pivot edge at 3° instead of 1.5°.
  - Now <= 3 mm fore-aft turning in place; straight-walking heel strike unchanged (~14° at 0.1, ~21° at 0.2).
- **Sway was ~0.26-0.39 leg lengths p-p** (humans walking this slowly ~0.1; at normal speed 2.5-3.5 cm, ~0.03 L). Causes:
  - a 9 cm default stance (0.43 L, wider than the 6.0 cm hip-pitch spacing);
  - the ZMP at the sole centre.
- **Sway fixes.**
  - New `zmp_inward` knob (default 0.6 toward the inner sole edge, ~10 mm margin).
  - `stance_width` default 0.06.
  - Sway is now 27-41 mm p-p (0.13-0.20 L) with the ZMP inside the feet.
  - Longer double support does not help (periodic LIPM: 1-3 mm).
  - `sway_gain` < 1 puts the ZMP outside the foot: the user's gait_table_style.yaml (0.3) grades ZMP margin -7 to -9.5 mm.
- Reference change, so a fresh training run is needed.

## Stand at command 0: non-saturating stand-pose anchor (2026-10-21)

- **Diagnosis** (MuJoCo, run `gaitref_2026-10-21_07-17-11`, step 98.3M). At command 0 the policy leaves the table stand pose (= the residual-map home) for a braced stance:
  - joint deviation from the stand pose, deg, env order l/r pairs (hip_yaw, hip_roll, hip_pitch, knee, ankle_roll, ankle_pitch) = `[-28,-18, -4,-13, 33,15, 26,24, -22,20, 2,19]`;
  - knees on their +5 deg stop, ankle roll / hip roll actions saturated, legs pressing each other with 8-15 N, sole gap 50 mm;
  - holding the stand pose (action 0) instead: 0 N leg contact, sole separation 91 mm, never falls under 0.2-0.3 m/s kicks with backlash; the brace falls 3-5 of 6;
  - when walking the policy stays near home (< 11 deg).
- **Mechanism.** Every stand anchor is flat at the brace, so no local gradient leads back: `imitation_joint_pos = exp(-mean(e^2)/0.15^2)` ~ 0.003, `imitation_foot_pos` ~ 0, `leg_contact_l2` and `undesired_contacts` saturate at 5 N.
- **Change.** New penalty `stand_pose_l2` (`rewards.stand_pose_l2`): per env, sum over the 12 joints of `(q - q_ref)^2` [rad^2] x the standing mask (`_reference_standing()`), with `q = robot.data.joint_pos` and `q_ref` = the table frame `_get_rewards` already samples once per step (the stand frame while standing).
  - Never saturates: ~1.5 rad^2 at the observed brace (weight -1.0 costs ~1.5/step; 1.56 from the rounded per-joint numbers), ~0.015 for 2 deg on every joint, 0 for walking envs.
  - Weight sign negative (`REWARD_WEIGHT_SIGN`), 0 by default (absent from `config.yaml`), `-1.0` in `config_gait_reference.yaml`. A nonzero weight without a gait reference raises a `ValueError` at init. Logged as `Reward_Terms/stand_pose_l2`.
  - No other weight or term changed. Tests: `tests/test_stand_pose_l2.py`.
- **Table version.** The checkpoint was trained on table v1 (`gait_table.npz` from 2026-10-15); the repo's table was re-baked to v2 on 2026-10-21 20:26 (same `stand` frame / `GaitTableReference` API), so this change needs a fresh run on the v2 table.

## Residual map follows the v2 table stand (2026-10-21)

- **Why.** The v2 re-bake moved the table stand (deg, L = R: hip_yaw 2.10, hip_roll -6.18, hip_pitch -17.72, knee -36.55, ankle_roll 6.53, ankle_pitch -18.71; base_height 0.2745 m) away from the old residual home (hip_roll -2.5, hip_pitch -8, knee -22, ankle_roll 2.5, ankle_pitch -14, hip_yaw 0):
  - action 0 was no longer the stand: with `stand_pose_l2` at -1.0 it cost ~0.22/step at a = 0 (knee 14.6 deg off);
  - the table needed |a| up to 1.48 (old scales), i.e. cells outside the action range.
- **New map** (`config_gait_reference.yaml`, `action_home_deg` = the user's rounded v2 stand with hip_yaw 2.1 instead of 0.0):
  - home (deg): hip_yaw 2.1, hip_roll -6.2, hip_pitch -17.7, knee -36.6, ankle_roll 6.5, ankle_pitch -18.7. Flat sole: hip_pitch - knee + ankle_pitch = 0.2 deg, hip_roll + ankle_roll = 0.3 deg. |a_stand| < 0.002.
  - `action_scale_rad`: hip_yaw 0.55, hip_roll 0.55, hip_pitch 0.75, knee 0.70, ankle_roll 0.60, ankle_pitch 0.45 (was 0.6 / 0.4 / 0.75 / 0.56 / 0.45 / 0.55).
  - `spawn_height_m`: 0.28 (was 0.286): same ~5.4 mm sole clearance as before; RSI resets use table base_height + 0.005 = 0.2795.
- **Scale rule.** Smallest value on a 0.05 grid with max |a| <= 0.85 over every baked cell inside the TRAINING command ranges (|vx| <= 0.2, |vy| <= 0.15, |wz| <= 1.2; a bound that is not a grid point also takes the first grid point beyond it: vx +-0.22, vy +-0.164, wz +-1.3), both legs, table clamped to the joint limits.
  - Required (home = rounded home): hip_yaw 0.529, hip_roll 0.526, hip_pitch 0.743, knee 0.689, ankle_roll 0.571, ankle_pitch 0.335 -> 0.55 / 0.55 / 0.75 / 0.70 / 0.60 / 0.35. The binding cells are command-box corners (vx -0.22 with vy +-0.164 for hip_pitch and knee). ankle_pitch is kept at 0.45 on purpose (balance authority beyond the table).
  - max |a| inside the training range with these scales: yaw 0.82, roll 0.81, hip_pitch 0.84, knee 0.84, ankle_roll 0.81, ankle_pitch 0.63.
  - max |a| over the WHOLE grid (vx +-0.3, vy +-0.2, wz +-1.5): yaw 0.84, roll 0.89, hip_pitch 1.05, knee 1.04, ankle_roll 0.90, ankle_pitch 0.63. Widening the command ranges toward the grid edge needs hip_pitch >= 0.95 and knee >= 0.90 (at 0.85).
- **MuJoCo check** (`BipedEnvV2` built from the edited config; zero action at command 0; BAM servo, 2.94 N*m cap; no policy). Spawn clearance of the lowest sole point right after reset: 5.4 mm at 0.28 in the Isaac frame (MuJoCo's XML root sits 10 mm above Isaac's 0.29, so it reads 15.4 mm there; the old home at 0.286 reads 15.43 mm, so the clearance is unchanged).
  - 10 s hold, backlash on / off: 0 falls in all four runs, base height drift after 1 s -0.015 mm / 0 mm (settled z 0.2735 / 0.2741 vs table stand 0.2745; old home 0.2798 / 0.2802 vs 0.2807).
  - mean |servo torque| over 2-10 s, new home, backlash on, as a fraction of the cap: knee 0.096 N*m (3.3%), ankle_pitch 0.129 (4.4%), ankle_roll 0.082 (2.8%), hip_roll 0.020 (0.7%), hip_pitch 0.017 (0.6%), hip_yaw 0.000; max over all joints and steps 0.36 N*m (12.4%). Old home: knee 1.4%, ankle_pitch 4.8%, ankle_roll 2.0%, max 12.1%. Backlash off: new / old max 12.7% / 11.1%.
  - knee sag (measured - home, 2-10 s mean): -0.72 deg with backlash (old home -0.49), -0.18 deg without (old -0.04); the other joints sag < 0.7 deg (backlash on).
  - velocity kicks in body x / y (+-0.2, +-0.3 m/s) and yaw +-0.6 rad/s every 4 s, 6 seeds x 20 s: 0 falls with the new home and with the old one, backlash on and off. Single-kick margin (open loop, a = 0, backlash on): falls at |vx| = 0.4 (3/3 seeds) for both homes, none at 0.35 or at |vy| = 0.4; |vx| and |vy| = 0.5 fall for both, except old home +vy 0.5.
- **Test fixes** (no protection weakened):
  - `test_residual_action_map.py`: reference-config constants and spawn height updated; `test_reference_config_map_covers_every_baked_cell` now checks the cells inside the training command ranges read from the config (same selection rule, max |a| <= 0.86, |a_stand| < 0.05); the flat-sole assertion on the reference home allows the 0.2-0.3 deg residual of the rounded values (was exactly 0). The config.yaml crouch fixtures and `RESIDUAL_SPAWN_HEIGHT_M` are untouched.
  - `test_track_lin_reference_target.py`: the two grid tests snap their nominal commands to the nearest baked cell instead of assuming v1 grid points. Perfect copy still scores 1.0 per frame in reference mode; standing scores 0.05-0.30 there (gap >= 0.70, threshold 0.5). The old-kernel bug is documented at the smallest moving cell above the 0.05 stand threshold (vx 0.02, vy 0.055): standing scores 0.50, a perfect copy 0.26, so the negative gap survives on v2; at the vx 0.1 forward cell the perfect copy scores 0.41 (standing 0.13), and no snapped cell reaches 0.6.
  - `test_imitation_foot_height.py`: the v2 stance foot height dips to -0.04 mm; the check is now |min| < 0.1 mm (kernel std 10 mm).
  - `test_reward_sign_guard.py`: `foot_flat_l2` is deliberately 0 in the gait-reference config; the check is `<= 0` (the +1.0 sign error stays rejected by the other tests).
  - Suite: 161 passed (`tests`, excluding `test_env.py` and `test_transformer.py`).
- **Old checkpoints and bundles are incompatible** (the map is baked into them; the Jetson refuses a mismatch): retrain and re-export. The Jetson reads home / scales from the bundle, so no Jetson code change.

## Feet pressing together at command 0: terrain-independent foot-gap penalty (2026-10-22)

- **Diagnosis** (MuJoCo, `src/mujoco/policy_gate.py` semantics, run `gaitref_2026-10-22_07-18-30`, step 73.7M). At command 0 the feet press together in 100% of steps: foot-foot normal force 13.8 N, `mj_geomDistance` between the two foot geoms -1.2 mm (holding the home pose instead: +31.5 mm). The policy straightens the knees ~17 deg, toes in 7-12 deg (hip yaw) and pitches the ankles +11-13 deg to bring the feet together. While walking there is almost no inter-leg contact.
- **Root cause.** Training uses `use_rough_terrain: true`, and `biped_env_cfg.py` then clears `contact_forces.filter_prim_paths_expr`, so `force_matrix_w` is None. `undesired_contacts` and `leg_contact_l2` (`_leg_contact_forces`) can no longer separate floor force from other-foot force and leave the FEET out; the non-foot leg bodies are still covered. Foot-to-foot contact was therefore completely unpenalised on rough terrain. On FLAT terrain the old force-based foot-foot term also works (it uses the ground-plane filter); rough terrain was the blind spot.
- **Change.** New penalty `foot_proximity` that needs no contact sensor:
  - `rewards.foot_footprint_gap(pos_l, quat_l, pos_r, quat_r, centre_offset, half_extents)`: signed gap [m] between the two foot footprints, each an oriented rectangle in the world horizontal plane (centre = link origin + R @ `foot_sole_offset_m`, projected to xy; axes from the foot yaw). Apart: exact minimum 2D distance (corner-to-rectangle). Overlapping: minus the SAT penetration depth. Vectorised, `torch.jit.script`. It is 2D, so a swing foot passing over the other foot's footprint reads <= 0 although the 3D geometry is apart.
  - `rewards.foot_proximity_penalty(gap, margin_m)` = `clamp((margin - gap)/margin, 0, 2)^2`: 0 beyond the margin, 1 touching, 4 at a margin of overlap or more.
  - `BipedEnv._get_rewards`: key `foot_proximity` (penalty, `REWARD_WEIGHT_SIGN` -1), gap from the fresh `robot.data.body_pos_w / body_quat_w` of `self.feet_indices` ([right, left]), once per step. Config: reward weight `foot_proximity: -2.0`, env_config `foot_proximity_margin_m: 0.006`, `foot_sole_half_extents_m: [0.0450, 0.0186]` (new cfg / schema / `_SCALAR_FIELDS` keys). Logged as `Reward_Terms/foot_proximity`, `Diagnostics/foot_gap_mm` (episodic mean) and `Diagnostics/foot_overlap_frac` (fraction of steps with gap <= 0).
  - Optional termination, DEFAULT OFF: `foot_overlap_termination_s` (0 = off; validated >= 0). When > 0, an env whose gap has been <= 0 for that many seconds in a row terminates as `died` (so `termination_penalty` applies). The counter advances once per step in `_get_dones` (Isaac Lab's `DirectRLEnv.step` calls `_get_dones` BEFORE `_get_rewards`, both after the physics step; both read `robot.data`, so they see the same state) and is zeroed in `_reset_idx`. Logged as `Episode_Termination/foot_overlap` (episodes ended ONLY by it; `base_contact` no longer counts them, and is unchanged while the termination is off). `config_gait_reference.yaml` keeps it at 0.0; 0.25 s is the suggested value if the penalty alone does not stop the feet pressing together.
  - `policy_gate.py` pose report: new line with the mean footprint gap [mm] and the fraction of steps with gap <= 0, computed with the SAME `rewards.foot_footprint_gap`. No new PASS/FAIL criterion. On this checkpoint: cmd0 13.79 N foot-foot, footprint gap -2.2 mm, 100% of steps <= 0; walk 0.2 m/s 17.3 mm, 4% of steps <= 0.
- **Footprint model.** From the foot collision mesh (`r/l_foot_link_1.stl` through `robot_v2.xml`, foot link frame): whole-mesh footprint x -28.7 .. +61.3 mm (90.0), |y| 2.0 .. 39.3 mm (37.3); the sole face (z -31.6 mm) is |y| 3.0 .. 39.2 mm (a 1 mm chamfer per side). Centre = the existing `foot_sole_offset_m` (16.3, 21.1, -31.6) mm; half extents (45.0, 18.6) mm. 18.6 splits the sole (18.1) and the wider upper foot (19.1), which keeps the error vs MuJoCo within ~1 mm for both a flat foot (+0.9 mm at home) and a rolled foot pressing on its sole edge (-1.0 mm at cmd 0). `tests/test_foot_proximity.py` re-derives centre and extents from the MJCF and asserts the cfg matches within 1 mm.
- **Validation against MuJoCo** (policy above, checkpoint `step_73728000.pt`, 3 seeds x 10 s per command, first 1 s dropped = 9450 frames; gap fed with the MuJoCo foot BODY `xpos / xquat`, compared with `mj_geomDistance(foot_l, foot_r, distmax 0.05)`). 480 frames where `mj_geomDistance` returns exactly 0.0 although no contact exists (a MuJoCo artefact: no contact, no force) and 105 frames at the 50 mm clamp are excluded from the error statistics (8865 frames used):

  | command | MuJoCo contact (F > 0) | gap <= 0 | MuJoCo dist mean [mm] | footprint gap mean [mm] | corr | error gap-MuJoCo mean [mm] (p5 / p95) | 2D overlap, 3D apart |
  |---|---|---|---|---|---|---|---|
  | (0, 0, 0) | 100.0% | 100.0% | -1.2 | -2.2 | 1.000 | -0.99 (-0.99 / -0.99) | 0 |
  | (0.1, 0, 0) | 0.0% | 0.0% | 27.4 | 26.6 | 0.999 | -0.67 (-1.43 / +0.27) | 0 |
  | (0.2, 0, 0) | 0.2% | 1.6% | 22.6 | 21.7 | 0.910 | -3.12 (-10.27 / +1.26) | 18 (1.3%) |
  | (-0.1, 0, 0) | 0.0% | 0.0% | 11.8 | 9.7 | 0.979 | -2.09 (-3.43 / -0.58) | 0 |
  | (0, 0.1, 0) | 0.0% | 0.0% | 15.3 | 14.7 | 0.994 | -0.38 (-0.91 / +0.97) | 0 |
  | (0, 0, 0.6) | 0.0% | 0.0% | 21.6 | 21.0 | 0.998 | -0.27 (-0.99 / +0.98) | 0 |
  | hold home (a = 0) | 0.0% | 0.0% | 31.5 | 32.4 | n/a (constant) | +0.90 | 0 |
  | all | | | | | 0.990 (Spearman 0.982) | mean -0.91, std 1.87, p1 -9.8, p5 -3.6, p50 -0.8, p95 +0.9, p99 +1.4 | 18 (0.2%) |

  - Every one of the 1353 frames with MuJoCo contact has gap <= -2.2 mm (min -4.5); every frame with `mj_geomDistance` <= 2 mm has gap <= -2.2 mm; at <= 3 mm the max is -0.2 mm. The gap is slightly pessimistic (negative error): it says "overlap" up to ~1-3 mm before the geoms touch, never after.
  - At home: footprint 32.4 mm vs MuJoCo 31.5 mm. At command 0: -2.2 vs -1.2 mm.
  - Swing foot above the other foot's footprint (gap <= 0 while MuJoCo says apart): 18 frames of 8865 (0.2%), all at command 0.2 (the swing foot 18-24 mm above the other foot, MuJoCo distance 0.6-4.1 mm). That is also where the largest errors are (p5 -10 mm).
  - Against the exact 2D projection of the foot MESH (convex hull), the rectangle model is 0.4-1.0 mm conservative; the remaining spread is the 3D height ignored by a 2D footprint.
  - Gap of the policy per command (min / p5 / mean [mm]): cmd0 -2.2 / -2.2 / -2.2; vx 0.1 3.7 / 7.6 / 26.6; vx 0.2 -4.8 / 4.7 / 21.7; vx -0.1 2.7 / 4.9 / 9.7 (46% of frames < 8 mm); vy 0.1 7.0 / 8.7 / 14.7; wz 0.6 5.6 / 9.8 / 21.0; home 32.4.
- **Margin.** Footprint gap over the gait table itself (`gait_table.npz` v2): every cell the config's command ranges select. `config_gait_reference.yaml` now trains on the whole bake grid (vx +-0.3, vy +-0.2, wz +-1.5 = 16 x 12 x 16 cells; an earlier pass of this entry used |vx| <= 0.2, |vy| <= 0.15, |wz| <= 1.2 and got min 8.8 mm), all 64 phase frames = 196,608 poses, set through MuJoCo FK (table `joint_pos`, base at the table base height with the `projected_gravity` orientation, yaw 0; checked against the table's own `foot_pos_b` to 1e-4 m): min 6.39 mm, p0.1 9.12, p1 11.37, p5 17.05, median 49.1, stand frame 32.8 mm (worst cell vx -0.3, vy -0.018, wz -1.3; per-cell minimum p1 8.2, p5 10.0). The margin is the largest whole mm (<= 10 mm) that penalises no reference frame: `foot_proximity_margin_m = 0.006` (7 mm would penalise 16 frames, 10 mm 0.48% of frames). The test reads the ranges from the yaml and fails if the margin is not exactly that value, so narrowing or widening the ranges forces a re-derivation.
- **Tests.** `tests/test_foot_proximity.py` (22 tests, command ranges read from the yaml): exact axis-aligned gaps, brute-force check on 250 random tilted / rotated poses (direction sweep for the signed value, sampled points for the distance), overlap <= 0, penalty values, cfg vs MJCF within 1 mm, the table's reference gap >= margin, sign guard, config values (-2.0, 0.0), source-level wiring (fresh `robot.data` pose, [right, left] order, counter written once per step and only in `_get_dones` / `_reset_idx`). Suite (`tests`, excluding `test_env.py`, `test_transformer.py`): 183 passed, 0 failed. `test_residual_action_map.py` was updated to the yaml's deliberate scales (hip_pitch 0.95, knee 0.90) and its coverage test now asserts the real constraint, per joint max |a| + `rsi_joint_noise` / scale < 1.0 over the cells the config's ranges select (whole grid: max |a| yaw 0.84, roll 0.89, hip_pitch 0.83, knee 0.81, ankle_roll 0.90, ankle_pitch 0.63; with noise 0.93 / 0.98 / 0.88 / 0.86 / 0.98 / 0.74), instead of the fixed 0.86 bound.
- **Needs a fresh run:** the penalty is new, so the existing checkpoints do not contain it. No other weight changed.
