# Sim2real work programme

Separate from `TODO.md`, which tracks deferred odds and ends. This file is one coherent
programme: **make the simulated servo behave like the real one, then rebuild the training
env on top of it.**

Written 2026-10-01, after the deployment chain was verified end to end and the robot still
walked badly.

---

## The diagnosis

Everything in the deployment path has now been checked and is correct: the IMU mount rotation
(derived from three static poses, quaternion composition proven against the gravity path to
1e-15), every joint direction sign (confirmed against the URDF-driven 3D viewer, which is a
valid oracle because the Jetson URDF and the V2 training URDF match on all 12 joints, axes and
origins), ONNX parity with the training actor (2.5e-6), and the servo velocity channel.

So the gap is physics fidelity. The reference project `RobotAnalysis/microduck_rl` — a
14-servo **Dynamixel XL330** robot at 7.4 V, *not* STS3215, so none of its numbers transfer —
states the thesis directly in its README:

> *"At this scale — tiny servos driving a ~800 g biped — **actuator fidelity is most of the
> sim2real gap**, which is why the actuator is modeled down to its voltage control law instead
> of an ideal PD."*

And its (deleted, recoverable via `git show ae72594^:docs/sim2real/sim2real_analysis.md`,
621 lines — **worth reading in full**) analysis describes our exact symptom:

> *"Policy trained with weak sim motor: learns **large actions** because a large position error
> is needed to generate enough torque. Real strong motor: same large position error →
> **1.29× more torque** → overshoots → oscillates."*

**We built that mismatch.** `DCMotorCfg` uses `effort_limit = 0.98` N·m — the *rated
continuous* figure — while the real STS3215 delivers up to 2.94 N·m transiently. Our sim motor
is the weak one precisely in the fast-transient regime that governs gait.

They explicitly **rejected** randomizing `action_scale` to paper over this: *"it treats the
symptom (policy over-actuation) rather than the cause (wrong motor model)."*

---

## Status

Updated 2026-10-02. Everything not blocked on the printed arm is built and tested; what is
left is hardware.

| # | Item | State |
|---|---|---|
| 0 | Acceleration-cap audit + clear | **code done, needs a hardware run** (`tools/servo_registers.py`) |
| 1 | This document | done |
| 2 | SysID rig, logger, validation against published 12 V params | **tooling done**: Jetson SysID tab, `actuator/validate.py`. Needs the printed arm |
| 3 | Full identification (only if 2 fails) | conditional on 2 |
| 4 | Port BAM to Isaac + parity check | **done**. Kernel == BAM to 1e-16; MuJoCo and Isaac pendulums == BAM to 0.03 mrad |
| 5 | Targets A / B / C | **implemented** on the placeholder params; each behind its own config key |

**Next on hardware, in order:**

1. `tools/servo_registers.py`: acceleration cap and torque-ceiling audit.
2. Re-baseline the robot with **D = 0** (see below).
3. Retrain.
4. Print the arm → SysID grid → `validate.py`.
5. Screen all 12 joints → set `backlash.lash_deg_range`.

### What changed the deployed robot this round (re-baseline before judging anything)

- **Servo PID is now P-only: `d: 0` on all 12** (`BipedRobotJetson/config/robot.yaml`). BAM's
  firmware law is P-only, so a policy trained on BAM must run on servos without D.
  `PolicyRunner.arm()` refuses a v2 bundle whose `requires_pid` doesn't match.
- **`_sync_write` wrote the wrong register.** The 0x2A goal block is position | **time** |
  speed (Feetech SDK: 42 / 44 / 46). Every write was 4 bytes, so the "speed" landed in
  goal *time*, and goal speed was never set. It now writes 6 bytes, `time = 0` and speed,
  as the SDK's own `WritePosEx` does. `ARM_SPEED = 200` is therefore now a real speed
  limit (~17.6 deg/s), and the arming ramp will feel different.
- **Action map is "centered"** (`a = 0` is the standing pose). This invalidates every existing
  checkpoint and `walk_v1`. Bundles are now format v2 and carry `action_map` + `actuator`; the
  Jetson refuses a bundle whose map it doesn't know, and keeps the old map for v1 bundles.

### Findings from building it, worth knowing

- **The placeholder fit implies ~6.3 N*m duty-limited stall at 12 V**, about 2x the
  datasheet's 2.94. BAM doesn't model current or overload protection, and
  `TORQUE_LIMIT < 100%` would cap it. The register tool audits that and prints the
  `torque_scale` to set.
- **Effective inertia must be `1/(M^-1)_ii`, not `M_ii`.** The stiction clip is a one-step
  deadbeat, and on a floating base the diagonal overestimates what one joint's torque moves.
  **PhysX's mass matrix excludes armature**, which `BamActuator` probes at startup.
- **Isaac has a standing-jitter floor of ~0.11 deg** even with the legacy DC motor (contact
  solver). BAM adds ~15%; backlash adds ~30% more. All of these are under 2.5 encoder counts.
  `check_actuator.py --legacy-dcmotor` re-measures it.
- **Three hand-written copies of the config mapping had drifted** (`train.py`, `play.py`,
  and a dead `common.make_env`). That drift is how `resampling_time_range`,
  `rel_standing_envs` and the rest went dead, and why `play.py` evaluated a different env.
  Now there is one builder, `src/isaaclab/env_config.py`. It raises on unknown keys, and
  `play.py` runs the same pipeline minus pushes/physics DR.
- **MJCF angles default to DEGREES.** `backlash.py` reads the compiler's `angle` setting;
  writing radians into a degree model silently shrank the lash ~57x.

### How to verify any of this yourself

```bash
python -m actuator.parity.test_kernel_vs_bam                      # needs bam (actuator/requirements.txt)
python -m actuator.parity.mujoco_pendulum --lash-deg 1.5
python actuator/parity/isaac_pendulum.py --headless               # Isaac python
python src/isaaclab/check_actuator.py --headless                  # full robot, startup checks
```

---

## Gait regression under BAM (2026-10-03) — fixed, then attributed

The first BAM run trained cleanly (episode length 831 vs 838) but walked badly:
- feet turned inward;
- body far ahead of the feet;
- unstable, asymmetric, poor turning.

Against the old DC-motor run:
- `track_lin_vel` was **up** (58 vs 38) and yaw tracking was halved (18 vs 37).
- `dof_acc` was 13×, `ankle_torques` 6×, hip/ankle-pitch deviation 2.5–4×.
- `swing_foot_height` was 0.17 vs 1.32.

That is a "fall forward" gait: lean ahead, shuffle, and catch with ankle torque.

**Fixed (bugs):**
- **`com_support_centering` never worked.** It was `exp(-1.0·dist²)` with dist in metres,
  which is a 0.7 m length scale on a 30 cm robot: 5 cm off support still scored 0.9975.
  It is now `exp(-dist²/(2·std²))` with `com_centering_std = 0.04`, and the weight is raised
  0.1 → 0.5 (a first guess). This predates BAM; the old weak motor simply couldn't hold a lean.
- **`swing_foot_height` datum (introduced in the BAM round).** It was a global EMA of
  in-contact foot height relative to the env origin, so it absorbed terrain elevation
  (46.5 mm on rough vs ~34 mm true). That made the effective dead zone ~22 mm, so the feet
  stopped lifting. The datum is now **each foot's own last touchdown height**: terrain-local,
  with no global state.

**Decided:** `actuator.stall_torque_cap_nm: 2.94` caps the duty-limited stall at the
datasheet. The placeholder implies 6.33 N·m, and the lean was held at a mean of 1.2 N·m,
above the 0.98 continuous rating. Set the cap to `null` once our own fit lands.

**New diagnostics** (TensorBoard `Diagnostics/*`; signs verified by `check_actuator.py`):
- `com_ahead_m`
- `com_lateral_abs_m`
- `lean_fwd_deg`
- `toe_in_deg`
- `swing_clearance_mm`
- `contact_asym`

**Attribution runs**, generated by `config/ablations/generate.py`:
- 25k iterations each, ≈100M on the TensorBoard axis.
- Ramps are compressed to 15k so every run ends in the final environment.
- Each run reverts exactly one suspect from `B`.

| Run | Change vs `B` | Question |
|---|---|---|
| `B` | (fixes + cap + diagnostics) | new baseline |
| `B_nolash` | backlash off | does backlash drive a preloaded lean/toe-in posture? |
| `B_ar005` | `action_rate_weight_end -0.05` | does the 30× smoothness penalty cause the shuffle? |
| `B_fullstrength` | `stall_torque_cap_nm: null` | is the lean held by unrealistic torque? |

**How to read them:**
- Compare `com_ahead_m`, `lean_fwd_deg`, `toe_in_deg`, `swing_clearance_mm`, `contact_asym`
  and `track_ang_vel_z_exp`.
- The old DC-motor run walked upright, so the target is `com_ahead_m` near 0 to slightly
  positive while walking.
- Pick the winning configuration for one full-length run.

```bash
python -m src.isaaclab.train --task BipedRobotV2 --headless --num_envs 4096 --config_path config/ablations/B.yaml
```

### Round 1 result (all four run) — lean fixed, stepping NOT

`B` fixed the posture: `com_ahead_m` 5 mm, stands straight. But **no run learned to step.**
All four plateaued at ~6 mm peak foot lift from 5M steps onward, against the DC-motor run
reaching swing reward 0.56 by 10M and 1.15 by 20M. This is "stuck from step one", not
"undertrained".

| Run | swing reward @30–37M | peak lift | episode score | verdict |
|---|---|---|---|---|
| DC (2026-09-28) | **1.139** | — | 1677 | the target |
| BAM (2026-10-02) | 0.242 | — | 756 | |
| `B` | 0.061 | 6.3 mm | 699 | lean fixed, won't step |
| `B_nolash` | 0.063 | 7.5 mm | 791 | ≈ B → **backlash cleared** |
| `B_ar005` | 0.072 | 6.5 mm | **1158** | only improvement |
| `B_fullstrength` | 0.002 | 2.9 mm | 406 | **total collapse** |

`B_fullstrength` is the most informative: given enough torque to hold a centred CoM,
standing still was strictly optimal, so it simply stopped moving.

### Round 2 diagnosis — stepping had been made unprofitable

The reward budget (weighted, 30–37M) shows the economics the policy actually faced:

| term | DC | `B` |
|---|---|---|
| `action_rate_l2` | −2.01 | **−18.22** |
| `com_support_centering` | 3.89 | **11.75** |
| `swing_foot_height` | **1.14** | **0.06** |

The policy was paid ~300× more to hold still than to lift a foot, so it shuffled: sliding
satisfies `track_lin_vel_xy` (29.7) while `feet_slide` costs only −0.24. Three causes, all
introduced by us, all now fixed in `config.yaml`:

1. **`swing_min_clearance` 0.01 → 0.0.** The dead zone sat ABOVE the actual 0.0063 m
   clearance, so `clamp(clearance − min, 0)` was zero for most swings: **no gradient at the
   operating point**, and weight 1.5 multiplied nothing. The dead zone existed to stop an
   unloaded foot from earning reward, and the per-foot touchdown datum already makes that
   structurally impossible. Keeping both was double protection that killed the signal.
2. **`action_rate_weight_end` −0.3 → −0.05.** Largest single term in the budget.
   `B_ar005` set exactly this and was the only run that improved.
3. **`com_support_centering` is now anisotropic**, `std_fwd` 0.12 / `std_lat` 0.04.
   Carrying the CoM ahead of the support is *what drives a step*; an isotropic 0.04 m scale
   forbids walking outright. Lateral excursion has no such role — with the feet ~5 cm apart
   a sideways CoM is a fall — so the lateral response is **bit-identical** to before and the
   fall protection is untouched. Cost of walking with the CoM 5 cm forward: 0.271 → 0.042
   per step.

Kept: the 2.94 N·m stall cap (`B_fullstrength` was worst) and backlash (`B_nolash` cleared it).

```bash
python -m src.isaaclab.train --task BipedRobotV2 --headless --num_envs 4096 --config_path config/ablations/C.yaml
```

**Kill criterion:** the DC run passed swing reward 0.5 by 10M. If `C` is still under ~0.3 at
20M, the economics are still wrong — do not let it run to completion.

### Round 2 result (`C`, ran to 292M) — reward moved, behaviour did not

| | DC (walks) | `B` | `C` |
|---|---|---|---|
| `swing_foot_height` | 1.352 | 0.061 | 0.288 |
| **`swing_clearance_mm`** | — | **6.3** | **6.06** |
| `track_ang_vel_z_exp` | 37.4 | 11.3 | 22.4 |
| `foot_sep_body_m` | 0.051 | 0.043 | **0.036** |
| `com_support_centering` | 4.28 | 11.75 | **18.45** |

**Foot lift did not change.** The swing reward rose 5× only because removing the dead zone
made it pay for the *same* shuffle. It plateaued by 15M and stayed flat for another 280M.
Stance narrowed to 3.6 cm — under the 4.33 cm hip width, so the feet are tucked inboard,
which is the instability visible on screen.

**The headline tracking number was lying.** `track_lin_vel_xy_exp` is an exp kernel averaged
over all envs, and a standing env scores ~1.0 for free. At `rel_standing_envs` 0.25 that
inflated it ~22%:

| | headline | standing | moving-only | implied error |
|---|---|---|---|---|
| DC | 38.1 | 2% | 0.433 | ~0.32 m/s |
| `C` | **46.6** | 25% | **0.412** | ~0.33 m/s |

`C` scores *above* the walking run while tracking slightly *worse*. Commands span only
−0.3…+0.5 m/s, so a 0.33 m/s error means it is barely following at all.

### Round 3 diagnosis — the term taxed the weight shift that starts every step

`com_support_centering` measured distance from the support **centre**. To lift a foot the
CoM must first move over the *other* foot — roughly half a stance width off-centre, yet
perfectly stable. That cost ~0.09/step against a ~0.045/step swing reward, so **stepping was
a net loss regardless of how the swing term was tuned**, which is why two rounds moved the
reward but never the behaviour.

It now scores distance outside the support **polygon** (point-to-segment, 0 while inside),
verified on 10 synthetic poses — weight fully on one foot went 0.823 → 1.000 while a 10 cm
sideways CoM still scores 0.53. Direction-agnostic, so the fore/aft split is gone.

Also changed: `standing_envs_end` 0.25 → 0.10, and two honest diagnostics —
`Diagnostics/vel_err_moving_ms` and `yaw_err_moving_rads`, raw error in physical units over
commanded-to-move envs only (nan elsewhere, so the episodic mean ignores standing envs).
`Diagnostics/com_outside_mm` reports what the reward now actually scores.

```bash
python -m src.isaaclab.train --task BipedRobotV2 --headless --num_envs 4096 --config_path config/ablations/D.yaml
```

**Judge `D` on `swing_clearance_mm`, not on reward values.** Lift has sat at ~6 mm through
three configurations while reward numbers moved freely. If it is not clearly above ~15 mm by
20M, the blocker is not reward shaping — test whether the BAM servo can physically swing the
leg at gait cadence before tuning anything else.

### Round 3 result (`D`) — kill criterion hit

Foot lift 6.8 mm. Three rounds of reward work moved every reward number and never the
behaviour. That is the signature of a physical constraint, not a weight.

New honest diagnostics on `D`: `vel_err_moving_ms` ≈ 0.16 (tolerable), but
**`yaw_err_moving_rads` ≈ 0.86–1.0 against commands of only ±0.3 rad/s** — yaw, not linear
velocity, is where command-following actually fails.

## Round 4 — the real bug: the stall cap was also a speed cap

`stall_torque_cap_nm` was applied by clamping the duty ceiling, i.e. the **voltage**. In

    tau = kt·V/R − kt²·dq/R

voltage sets stall torque **and** no-load speed together:

| | stall | no-load speed |
|---|---|---|
| STS3215 datasheet @12 V | 2.94 N·m | **5.82 rad/s** |
| BAM m6 raw fit | 6.33 N·m | 5.15 rad/s |
| duty-capped (`B`…`D`) | 2.94 N·m | **2.39 rad/s** |
| **R-capped (`E`)** | 2.94 N·m | **5.15 rad/s** |

The datasheet quotes 2.94 N·m *and* 5.82 rad/s simultaneously; capping by duty cannot
express that. A knee needs ~2 rad/s **unloaded** to swing a leg at ~1.5 Hz cadence, so at
2.39 rad/s no-load the robot physically could not step once gravity, 0.2 N·m of friction and
leg inertia were added. Fingerprint: `Torque/frac_saturated` 0.55–0.60 across `B`…`D` against
the working DC run's 0.39.

The cap now raises `R` instead (`apply_stall_cap`), which lowers stall torque and leaves
speed alone — what a real higher-resistance motor does. Sanity check: the implied real servo
is kt 2.0 / R 7.92; the capped m6 is kt 2.26 / R 8.96, so stall, speed and the back-EMF
damping term kt²/R all land close to the datasheet part. The old duty cap left damping at
1.23 vs the real 0.51 — it was over-damped as well as too slow.

`E` is `D` plus this one fix; the config files are identical, the change is in code.

```bash
python -m src.isaaclab.train --task BipedRobotV2 --headless --num_envs 4096 --config_path config/ablations/E.yaml
```

Watch `swing_clearance_mm` (must clear ~15 mm), `Torque/frac_saturated` (should fall toward
the DC run's 0.39) and `yaw_err_moving_rads`.

**Rule going forward: never express an actuator limit as a duty/voltage scale.** Duty scales
torque and speed together; a datasheet specifies them independently. Torque limits belong in
`R` or `kt`, not in the PWM ceiling.

---

## 0. The acceleration cap — check this before anything else

The STS3215 ships with an acceleration limit active, via **two** registers:

| Register | Name | Notes |
|---|---|---|
| `0x29` (41) | `ACCELERATION` | documented, unit 100 step/s², 0 = unlimited |
| `0x55` (85) | `MAX_ACCELERATION` | **undocumented**, factory value 50 |

`0x55` gates `0x29`, and writing `0x55` alone does nothing — `0x29` must be written after it.
Both are RAM, so both **revert on every power cycle**. Source: Rhoban BAM PR #20
(`T-K-233/bam:add-sts3215-12v`), which hit this while identifying a 12 V STS3215 and measured
the out-of-the-box ceiling at roughly **10–28 rad/s²**.

**The Jetson repo never wrote either register.** `registers.py` defined `ACCELERATION` and
`servo.py` had a `set_acceleration()` method, but it was never called; `0x55` was absent from
the map entirely.

Why it matters: a bang-bang move of distance `d` in time `T` needs `a = 4d/T²`. At 50 Hz a
0.1 rad step needs **~1000 rad/s²**. Under a 28 rad/s² cap the servo covers **0.16° per control
step** — it cannot follow the policy at all, and the phase lag that creates is exactly how
sluggish tracking turns into oscillation.

**Done:** `MAX_ACCELERATION = 0x55` added to the register map; `Robot.initialize()` now writes
`0x55 = 0` then `0x29 = 0` on every servo at every startup; `tools/servo_registers.py` audits,
clears and step-tests.

**To run on hardware:**

```bash
venv/bin/python tools/servo_registers.py                 # audit: what are they set to?
venv/bin/python tools/servo_registers.py --clear         # clear them
venv/bin/python tools/servo_registers.py --step-test 10  # prove it (torque engages)
```

Expected: if the cap was active, time-to-90% on a 10° step drops from hundreds of
milliseconds to within a control period or two. **This must be settled before any
identification** — a servo under a hidden acceleration cap would corrupt the SysID data too.

---

## The actuator model: BAM

Rhoban's BAM (`github.com/Rhoban/bam`) is a voltage-level, identified DC-motor model with
nonlinear friction, not a PD controller. Its kernel, per physics step:

```
max_step = max_velocity * dt
q_smooth = clamp(q_target, q_smooth - max_step, q_smooth + max_step)   # stateful slew limiter
duty     = clamp((q_smooth - q) * kp * error_gain * error_gain_ratio, -0.97, 0.97)
voltage  = vin * duty
torque   = kt*voltage/R - kt^2*dq/R                                     # drive minus back-EMF
friction_budget = friction_base + load terms + stribeck + friction_viscous*|dq|
tau_stop = (I_eff/dt)*dq + torque + tau_ext
ctrl     = torque - sign(tau_stop) * min(|tau_stop|, friction_budget)   # never injects energy
```

### What we already had

`actuator/__pycache__/bam_sts3215.cpython-311.pyc` survived a lost source file and was read
out of its bytecode (CPython 3.11, same as the local interpreter). It shows a prior attempt:

- `motor_name = "feetech_sts3215_7_4V"`, `model = "m1"` — BAM ships a Feetech parameter set
- `_KP_FW = 32` — exactly the `p: 32` in every `config/robot.yaml` servo entry
- default `vin = 7.4`; `prepare_mujoco_model` zeroes joint `damping`/`frictionloss` and
  rewrites the actuator as a force source
- `ACTUATOR_NAMES` are V1 joint names (`r_hip_z`, `r_hip_x`, …), so it predates V2

### Facts that shape the work

**A fitted 12 V parameter set already exists** — BAM PR #20, with M1–M6 files under
`feetech_sts3215_12V` and registry key `sts3215_12v`. Its M6:

```
kt 2.2617   R 4.1581   armature 0.02342   friction_base 0.14907
max_velocity 4.9164    command_delay 0.005565
```

The PR explicitly warns **7.4 V params do not transfer to 12 V** (they predict the arm falling
where the 12 V servo holds it). So the job is **validate the published fit**, not identify from
scratch — a far smaller dataset.

**The firmware law is P-only.** `kp = 32`, `error_gain = 0.166`, `max_pwm = 0.97`. **No D and
no I term**, plus the slew limiter above. BAM's own recorder writes `D = 0`.
**Our robot runs D = 16**, which is outside the model — it must go to zero on the bench *and*
on the robot, or the identified model describes a different servo than the one we deploy.
(Re-baseline the robot after changing it.) Waveshare's 12 V ST3025 reuses `error_gain = 0.166,
kp = 32` verbatim, so these are family-wide constants.

Per-radian loop gain for reference: STS3215 `32 × 0.166 = 5.31`/rad vs XL330
`400 × 0.0028774 = 1.15`/rad.

**Only five per-sample fields are read by the fitter**: `timestamp`, `position`, `speed`,
`goal_position`, `torque_enable` — plus top-level `mass`, `arm_mass`, `length`, `kp`, `vin`,
`dt`. `load`, `input_volts` and `temp` are written by their recorder but never consumed. We do
not need the current register on the critical path (log it for diagnostics anyway).

**Trajectories are exactly 6.0 s**, from a fixed registry: `sin_time_square`, `lift_and_drop`
(cubic to −π/2 then **torque off** — the free-fall segment, essential for the drive/backdrive
split and the dominant error source across all models), `up_and_down`, `sin_sin`. The protocol
also sweeps **kp over [4, 8, 16, 32]**, holding one out for validation (`--validation_kp 8`).
That dimension validates the *firmware* model, not just the friction fit.

**Model ladder:** M1 Coulomb-viscous → M2 +Stribeck → M3 +load-dependent (needs multiple
masses) → M4 → M5 +directional (needs drive *and* backdrive) → M6 +quadratic. All models always
identify `kt`, `R`, `armature`, `q_offset`, `command_delay`, `friction_base`, `friction_viscous`;
STS3215 adds `error_gain_ratio` and `max_velocity`.

**Pitfalls, all documented upstream:**

- `kt`, `R` and `error_gain_ratio` are **degenerate** — logs only constrain
  `kt·error_gain_ratio/R` and `kt²/R`, so all three scale together at identical MAE. Run ≥3
  restarts and keep only physically plausible values.
- `q_offset` is fitted **once globally** across all logs — never remount the servo mid-dataset.
- `command_delay` is rig-specific (bus, baud, control period). Do not reuse it on the robot.
- All logs must share an identical `dt`.
- **Filter for PWM saturation before fitting.** Their stall criterion
  `(|ω| < 0.05 rad/s) & (|err| > 0.2 rad) & (|V| > 3)` dropped 53 recordings to 22 and had been
  causing a **29% torque underestimate** — the root of their entire `action_scale` saga.
- Do not inherit their recorder's bugs: hardcoded `/dev/ttyACM0` and ID 1, `--vin` defaulting
  to 15.0, `arm_mass` never written, and an `all_record.py` that lists a `brutal` trajectory
  missing from the registry.

**Published 12 V hold-out quality** (kp=8 held out): ~10 mrad overall for M6, ~21–25 mrad on
`lift_and_drop`, 5–7 mrad elsewhere. M1 is ~2× worse, almost entirely on `lift_and_drop`.
If our validation lands near this, we skip identification.

### The rig

Any rigid single-DOF pendulum is **exactly** equivalent to a point pendulum, so the arm
geometry is unconstrained. BAM with `arm_mass = 0` computes `inertia = mass·L²` and gravity
coefficient `mass·L`. Given the real inertia about the pivot `I` and the real first moment
`S = Σ mᵢdᵢ` (from the arm URDF plus the weighed payload):

```
L = I / S        mass = S² / I        arm_mass = 0
```

Exact for any arm shape and any weight position, so a single arm with three weight holes gives
three genuine load conditions with no modelling error.

What *does* still bind: the arm must be **stiff** (first bending mode well clear of the ~3 Hz
excitation band — a 15×6 mm PLA bar with 250 g at 150 mm resonates near 9 Hz, which the fit
would absorb as fictitious friction), the payload must be **weighed**, and the servo must not
be remounted mid-dataset.

---

## Target A — BAM actuator, ported to Isaac

Swap `DCMotorCfg` for a `BamActuator(ActuatorBase)`. Isaac Lab supports this cleanly:
`ActuatorBaseCfg.class_type` is a plain `type` with no registry,
`Articulation._process_actuators_cfg` instantiates whatever it names, the implicit/explicit
branch tests `isinstance(actuator, ImplicitActuator)` so a subclass stays explicit, and
`compute(control_action, joint_pos, joint_vel)` is the right hook. All gain tensors are already
`(num_envs, num_joints)`, so per-env randomization is free.

This replaces the `effort_limit` / `stiffness` / `damping` guesswork entirely, and makes
`randomize_actuator_gains` redundant — the reference turned kp/kd DR **off** once BAM was in,
expressing uncertainty as `vin_range` (6.5–8.2 V for them) plus a friction scale instead.

**The one real porting risk:** the stiction clip needs effective inertia. MuJoCo reads
`dof_invweight0`; Isaac has no direct equivalent. Compute it once at startup from the
articulation's generalized mass matrix, or approximate from armature plus link inertia. Getting
it wrong makes stiction either vanish or inject energy.

**Parity gate:** identical goal trajectories from identical initial states through MuJoCo+BAM
and Isaac+`BamActuator`, comparing position traces. BAM's own harness flags `> 0.01` as *"likely
a kernel bug"* — adopt that threshold.

## Target B — Backlash

The reference models it as a **serial passive hinge** on the same axis as each actuated joint,
with hard limits — not a torque deadband. Link rotation = `qpos[servo] + qpos[backlash]`.
Generated MJCF values:

```xml
<joint damping="0.01" frictionloss="0" armature="0.001" limited="true"
       range="-0.0174533 0.0174533"
       solreflimit="0.01 1" solimplimit="0.95 0.999 0.0001 0.5 2"/>
```

Their comment on the stiffness choice is load-bearing: with a range this small the default
solref lets the joint overshoot its limits ~2× under load; `0.01` is 2× sim dt, the stiffest
stable setting.

**The best idea in that repo: the encoder reads *through* the backlash.** The magnetic encoder
sits on the output side, so the firmware position loop never sees the servo winding through the
dead zone — `cmd.pos += qpos[backlash]`, while `cmd.vel` stays motor-side because back-EMF and
friction are rotor physics. Observations are remapped to match. **The STS3215 sensor is also
output-side, so this applies to us directly**, and it matters more for us because our play is
larger than their ±1°.

- **MuJoCo**: port `add_backlash.py` — it operates standalone on any MJCF.
- **Isaac** (implemented as `LostMotion` in `actuator/bam_kernel.py`): a second passive DOF
  per joint would double the DOF count, and a stiff explicit spring on ~1e-4 kg*m^2 output links
  is unstable at 200 Hz. So the kernel tracks a virtual rotor through the gap instead:
  - in free play no torque is transmitted;
  - in contact the full motor torque is transmitted;
  - at re-engagement the rotor's momentum arrives as a capped impulse.

  A reversal costs 15–30 ms of torque-free dead time for 0.5–3 deg of lash. lash = 0 is an
  exact no-op (tested). An earlier plan for a simple position-error deadband was dropped:
  the encoder is output-side, so the firmware DOES see the error during free play. What is
  missing during free play is transmission, not error.
- Use the **measured** lash from the 12-joint screen, not their ±1°. Theirs is fixed, not
  randomized, deliberately kept as an unconfounded A/B axis — run the A/B before randomizing.

## Target C — The config set

**Prerequisite, or none of the command work functions:** `src/isaaclab/train.py:167-180` (and
its duplicate in `common.py:220-232`) copies only `lin_vel_x`, `lin_vel_y`, `ang_vel_z`.
`resampling_time_range`, `rel_standing_envs`, `heading_control_stiffness` and `rel_heading_envs`
in `config.yaml` are **silently discarded**; the live values are the `biped_env_cfg.py:529-531`
defaults — `(10.0, 10.0)`, `0.3`, `0.02`.

| Change | From | To | Source |
|---|---|---|---|
| Action map | absolute over the joint-limit box | **residual around the home pose** | reference uses `scale = 1.0` residuals |
| `action_rate_l2` | −0.01 | raised on a curriculum | *"the most important regulariser for sim2real"*; they ramp −0.1 → −1.0 over 1500 iters |
| `joint_vel` obs noise | 0.05 | **~0.25 rad/s** | theirs is ±0.25 vs ±0.001 on joint_pos — a 250× ratio, modelling *which* sensor is bad |
| `rel_standing_envs` | 0.02 | **curriculum 0.02 → 0.25** | no stillness reward at all; standing is taught by the spawn mix |
| Turn-in-place bucket | none | **15%**, `vx=vy=0`, `|ωz| ∈ [0.4·max, max]` | uniform sampling makes this ~2% of data; spin-on-the-spot was otherwise untrained |
| IMU misalignment DR | none | **0–6°**, constant per env | trains tolerance to magnitude only; cannot fix a systematic bias |
| Mount offset | observation only | **observation *and* command path** | see below |
| Command-range curriculum | — | **do not add one** | they measured a widening ramp *causing* a post-iter-1000 decline |

**Why the action map matters:** `a = 0` currently maps to the joint-limit *midpoint* — knee
−57.5°. Standing straight requires `a ≈ +0.92` on both knees every single step, near tanh
saturation. `action_scale = 0.7` is already marked DEAD in `biped_env_cfg.py:173-177`.

**Mount offset on both paths.** `encoder_offset` is applied to `joint_pos_obs` only
(`biped_env.py:438-439`), which models a sensor that lies to the policy while the PD tracks the
true angle. Real hardware cannot do that: the servo closes its loop on *its own* encoder, so a
mis-zeroed servo drives the true joint somewhere displaced and the leg's kinematic zero moves.
Keeping the existing convention `r = q + d`, the command site becomes
`set_joint_position_target(filtered_targets - mount_offset)`. Check: next step
`r = q + d = (t − d) + d = t` — self-consistent in encoder space, with true `q` displaced, which
is the whole point. Also move sampling from per-reset to startup, matching its own docstring.

The reference applies its encoder bias to the observation only too; both reviewing agents
flagged that as a likely gap rather than a deliberate choice. Heed their related rule: *"If an
obs is remapped to a sensor view, any tracking REWARD on the same quantity must measure the same
view — otherwise the policy is punished for correcting what it sees."* Audit `dof_pos_l2` and
the `joint_deviation_*` terms when this lands.

---

## Independent bug: `swing_foot_height` pays for standing still

**Unrelated to the servo work, and the clearest single explanation for the standing jitter.**

`rewards/rewards.py:3-20` takes no `commands` argument — it is **the only gait-shaping term with
no command gate**, while `feet_air_time`, `gait_phase_contact` and `step_length` all gate on
`cmd_norm > threshold`. It also measures the **foot-link origin in world Z**, not the sole and
not height above terrain.

Summing the V2 MJCF chain: `−0.00015 − 0.040776 − 0.0339 − 0.072052 − 0.072184 − 0.0339 =
−0.2536 m`. Against a ~0.29 m base spawn that puts the foot-link origin at **z ≈ 0.036 m while
standing** — above its own `min_height = 0.02`. Contact is `|F| > 1.0 N` against a 16.3 N robot.

**So a foot flat on the floor carrying under 1 N earns `1.5 × (0.036 − 0.02) ≈ +0.025/step`,
with zero lift.** Rocking side to side to alternately unweight each foot farms it. Against that,
the total smoothness cost of a 3° 5 Hz tremor on all 12 joints is ~0.030/step, 97% of which is
`dof_acc_l2`; `action_rate_l2` contributes 0.0003.

This is a regression: raising the weight 0.15 → 1.5 and the cap 0.04 → 0.09 fixed the
straight-leg gait and amplified this payout tenfold.

**Fix:** gate on command norm like every other gait term; measure relative to the env's terrain
origin (also closing `TODO.md` item 7, where rough terrain pays a bonus proportional to patch
elevation); and raise `min_height` above the measured resting foot-origin height — **from a
startup diagnostic, not from the arithmetic above.**

Related: `dof_vel_l2` does not exist anywhere in the repo, so joint velocity is completely
unpenalised.

---

## Process rules worth adopting

- **Export at ~2000 iterations, not the final checkpoint.** Their most counter-intuitive
  finding: *"Final checkpoint: robot shakes at action_scale=0.65. ~2000 iterations: clean
  transfer, no shaking."* Past the point of robust walking the optimizer starts exploiting
  sim-specific contact timing and numerical artefacts, and training metrics do not show it.
- **Pin simulator versions exactly and hardware-test before bumping.** They lost months to a
  backend update that silently broke transfer while training curves looked fine.
- **Verify physics assumptions in sim before training.** A target pose must be a stable
  equilibrium: hold its ctrl for 3 s from noisy inits and check **tilt**, not just height — a
  settle test that only records z reports fallen states as resting fine.
- **Domain randomization must not accumulate across resets.** They hit the same COM bug we did:
  *"An accumulating CoM randomizer once degraded every long run for months."*
- **Zero `Return_Delay_Time`** on every servo — already handled by `tools/return_delay.py`.

## Notes against `TODO.md`

- **Item 2's archaeology note is resolved.** The lost `bam_sts3215` source has been recovered
  from its `.pyc`; see above.
- Item 10 is already closed (`dof_acc_l2` live at `-1.0e-06`, `config.yaml:90`). Worth knowing
  that it is the *largest* of the three smoothness penalties at jitter frequencies —
  `action_rate_l2` at -0.01 contributes ~1% of what `dof_acc_l2` does. The genuinely absent
  term is `dof_vel_l2`, which appears nowhere in the repo.
