# actuator/

The Feetech STS3215 servo model the policy trains against, plus everything needed to
identify the real servos and prove each simulator reproduces the model. Background and the
full work programme are in `../TODO_SIM2REAL.md`.

## Layout

```
params/                BAM params JSONs -- the servo identification
  sts3215_12v/m1..m6     ACTIVE. BAM PR #20's 12 V fit, validated against our own
                         hardware 2026-10-09 (see Results, and PROVENANCE.md)
  sts3215_7_4v/          reference only, wrong voltage for this robot
  PROVENANCE.md          commits, licence, known quirks of the published fit
bam_kernel.py          the servo model in batched torch (BamKernel, LostMotion, BamServo)
backlash.py            add serial passive lash hinges to an MJCF (MuJoCo backlash)
validate.py            "do the published params describe OUR servos?" (bam.process + bam.mae)
                       plus recorded-vs-fitted figures and the torque-on/off error split
logset.py              load a session's logs + rig metadata. numpy only, so it imports in
                       the bam venv and the Isaac one alike
plotstyle.py           one look for every figure; palette checked with the viz validator
plots.py               the two figure forms: overlay grid, and "where is the error"
replay.py              replay recorded logs through MuJoCo vs hardware; orchestrates Isaac
replay_isaac.py        the Isaac worker replay.py spawns (Kit cannot share a process
                       with MuJoCo, so they never run in one interpreter)
parity/
  test_kernel_vs_bam.py  kernel vs BAM's own simulator, float64, bit-exact
  test_lost_motion.py    Isaac backlash behaviour (lash = 0 is an exact no-op, dead time, bounded)
  mujoco_pendulum.py     kernel inside MuJoCo vs BAM
  isaac_pendulum.py      BamActuator inside Isaac/PhysX vs BAM
  pendulum_ref.py        the shared reference pendulum
rig/rig.yaml           <- YOU PUT THE RIG NUMBERS HERE
rig/pendulum_arm.urdf  placeholder; replace with the Fusion2URDF export
rig/loads.json         generated; copy to the Jetson's data/sysid/loads.json
rig.py                 real arm -> the equivalent point pendulum BAM wants
data/<session>/        raw recordings copied off the Jetson
requirements.txt       bam (pinned), optuna, matplotlib, pytest. Tooling only;
                       training needs none of it.
```

## Switching to a new identification

The servo model is one file. `python -m bam.fit ... --output <file>` writes exactly the
params format used here, so moving from the placeholder to our own fit is:

```yaml
# config/config.yaml
env_config:
  actuator:
    params_file: actuator/params/ours/m6.json
```

Isaac (`envs/assets/robotV2/bam_actuator.py`) and the MuJoCo eval env both read that key, and
`export_onnx.py` records the file and its SHA-256 in every bundle.

## The model, and where each sim gets its inputs

`bam_kernel.py` ports BAM's `simulate.py` + `model.py`, the code path the parameters were
fitted against: firmware slew limiter → P-only duty → DC motor with back-EMF → M1..M6
friction budget → stiction clip. It never imports a simulator. Each caller supplies the
two things only it can know:

| | external torque on the joint | effective inertia |
|---|---|---|
| pendulum tests | exact gravity torque | `m·L² + armature` |
| MuJoCo | `−qfrc_bias + qfrc_constraint` (as `bam/mujoco.py` does) | `1/(M⁻¹)ᵢᵢ` via `mj_solveM` |
| Isaac | lagged observer: `I·q̈ − τ_applied` (previous substep) | `1/(M⁻¹)ᵢᵢ` from PhysX, plus armature |

The inertia is `1/(M⁻¹)ᵢᵢ`, **not** the diagonal `Mᵢᵢ`. The stiction clip is a one-step
deadbeat on velocity, and on a floating base `Mᵢᵢ` (every other DOF locked) overestimates
what a torque on one joint actually moves, so it over-corrects into chatter. PhysX's mass
matrix **excludes** armature; `BamActuator` checks this on startup and logs the result.

**Backlash.** MuJoCo gets the faithful model: a passive hinge in series with each servo, with
stiff limits and **the encoder reading through the lash** (the STS3215 sensor is
output-side). Isaac can't add a DOF per joint cheaply, so it uses `LostMotion`. That model
tracks a virtual rotor through the gap, transmits no torque in free play, and delivers an
impulse at re-engagement. Its one known approximation is that the output keeps the rotor's
inertia during free play.

## Parity chain

Run these after touching the kernel or either simulator integration. `bam` must be
importable (see `requirements.txt`).

```bash
python -m actuator.parity.test_kernel_vs_bam   # kernel == BAM            -> 1e-16 rad
python -m actuator.parity.test_lost_motion     # backlash behaviour       (torch only)
python -m actuator.parity.mujoco_pendulum --lash-deg 1.5    # MuJoCo == BAM    -> 0.03 mrad
python actuator/parity/isaac_pendulum.py --headless         # Isaac == BAM     -> 0.03 mrad (Isaac python)
```

The thresholds are BAM's own: 1e-9 rad for the bit-exact test, and 10 mrad MAE (*"likely a
kernel bug"*) for real integrators. The arrow values are what they measured on 2026-10-02
with the M6 placeholder. All four are mutation-tested: dropping the slew limiter, the
viscous term or the M6 sign gate, or shifting the delay by one sample, each fails the
kernel test.

For the full robot in Isaac, use `src/isaaclab/check_actuator.py`.

## Workflow: from the rig to a validated model

1. Design the arm in Fusion. Set the material density so CAD mass matches the weighed
   print, then export with Fusion2URDF. Put the URDF origin **on the servo rotation
   axis**, which removes a parallel-axis step and a sign trap.
2. Fill in `rig/rig.yaml`: the URDF path, the rotation axis, the weighed payload mass, and
   the hole radii.
3. `python -m actuator.rig --config actuator/rig/rig.yaml`
   This prints an `(I, S) -> (mass, length)` table and writes `rig/loads.json`.
4. Copy `loads.json` to the Jetson at `data/sysid/loads.json`.
5. On the Jetson's **SysID tab**:
   - Start a session for the bench servo's id.
   - Capture the hanging zero.
   - Record the grid: 4 trajectories × kp {4, 8, 16, 32} × 3 holes, 6 s each (~5 min of data).
     The tab shows coverage.
   - Download the session zip.

   **Never remount the servo mid-session**: BAM fits one offset across all of it.
6. Unzip into `actuator/data/<session>/`, then
   `python -m actuator.validate --session actuator/data/<session>`.
   - **Near the published hold-out** (~10 mrad M6, ~20 mrad M1): the placeholder describes
     our servos, so keep it.
   - **Several times worse**: fit our own (command in `validate.py`), drop it in
     `params/ours/`, and point `params_file` at it.
7. On the robot, suspended: the tab's **Screen all joints** measures per-joint hysteresis
   (backlash + friction deadband). Use it to set `backlash.lash_deg_range` in
   `config/config.yaml`.

## Running the analysis

Two venvs, because no single one has every dependency: `bam` needs its own (it pins an old
numpy), while MuJoCo and Isaac live in the Isaac env. Each tool below says which it needs.

**Score a session and plot recorded vs fitted** (needs `bam`: `actuator/.venv`):

```bash
# from the BipedRobot root
actuator/.venv/Scripts/python.exe -m actuator.validate --session actuator/data/<session>
```

Screens out PWM-saturated recordings, resamples with `bam.process`, scores every model with
`bam.mae`, then rolls the best one out over every log and writes the figures. Options:

| flag | |
|---|---|
| `--plot-model m6` | plot a named model instead of the best-scoring one |
| `--plot-model none` | skip the rollout and figures (scoring only) |
| `--params <dir>` | a different params directory |
| `--keep-saturated` | skip the saturation screen |

**Replay through the simulators** (needs MuJoCo + Isaac: the Isaac env python):

```bash
python -m actuator.replay --session actuator/data/<session>            # MuJoCo + Isaac
python -m actuator.replay --session ... --sims mujoco                  # skip the Kit launch
python -m actuator.replay --session ... --reuse                        # redraw from cached .npz
python -m actuator.replay --session ... --stall-cap 2.94               # as config.yaml runs it
```

Each simulator runs in its own subprocess (MuJoCo's DLL will not load inside a running
Isaac Kit process) and caches to `<session>/_sim/<sim>_<model>.npz`. `--reuse` keeps those,
which matters because the Isaac pass costs a Kit launch plus ~58k physics steps. Replays are
open-loop: the simulator gets the recorded goal and `torque_enable`, starts from the
recorded initial state, and is never corrected afterwards.

## Results: the 2026-10-09 identification

Session `data/bench_servo_2`, 48 recordings: 4 trajectories x kp {4, 8, 16, 32} x 3 holes
(50/100/150 mm), 6 s each at ~250 Hz, one STS3215 on the pendulum rig at 11.49 V.

### Verdict: keep `sts3215_12v/m6.json`. No refit needed.

The published BAM PR #20 parameters describe these servos **better than they described
the rig they were fitted on** -- 5.5 mrad against a 10 mrad published hold-out.

| model | torque-on mean | torque-on median | torque-off mean | session mean | published |
|---|---|---|---|---|---|
| m1 | 6.14 | 6.57 | 722 | 125.6 | 21 |
| m2 | 6.01 | 5.76 | 756 | 131.1 | - |
| m3 | 6.70 | 5.49 | 737 | 128.8 | - |
| m4 | 6.43 | 5.88 | 728 | 127.0 | - |
| m5 | **5.44** | **4.38** | 758 | 131.0 | - |
| **m6** | **5.53** | 4.53 | 762 | 131.8 | **10** |

All mrad. m5 edges m6 by 0.09 mrad, which is noise -- and m5 pays for it with three
friction terms pinned at ~0 (`load_friction_motor` 2.5e-08, `load_friction_motor_stribeck`
4.7e-08, `friction_stribeck` 5.9e-05) and `alpha` stuck on its lower bound at 1.0000002.
m6 is the only model with a published hold-out close to our result, and it is already what
`config/config.yaml`, the parity chain and `PROVENANCE.md` point at. **m6 stays.**

### The one real gap: stiction

![where the error is](media/sysid_summary_m6.png)

40 of 48 recordings land under 10 mrad; then a cliff to ~1000 mrad for eight. Every one of
those eight is `lift_and_drop` at hole 1 or 2 -- the trajectory that cuts torque at -90
degrees. Split by regime, m6 scores **5.5 mrad powered and 762 mrad unpowered**.

![recorded vs m6, hole 2](media/sysid_fit_m6_h2.png)

Top row, after the grey band starts: the real arm (blue) stays where torque left it, and
the model (orange) sweeps back to hanging. Measured drift after torque-off is 1.5 to 120
mrad at **every** hole, including 0.433 N.m of gravity torque at h2. The ~330:1 gearbox is
simply not back-drivable, and the published friction budget does not hold it.

This costs nothing for tracking -- the policy never runs unpowered -- but it is wrong for
impacts, falls, and a leg loaded by the ground. Fix it by raising stiction if that
behaviour starts to matter; do not refit the whole model for it.

### Both simulators reproduce the kernel

![simulator parity](media/sysid_sim_parity_m6.png)

Replaying all 48 logs open-loop through each simulator, against the same hardware:

| | torque-on MAE | session mean |
|---|---|---|
| BAM reference (`bam.simulate`) | 5.53 mrad | 131.8 mrad |
| MuJoCo (`replay.py`) | 5.53 mrad | 131.6 mrad |
| Isaac / PhysX (`replay_isaac.py`) | 5.53 mrad | 131.6 mrad |

MuJoCo and Isaac agree with each other to **6.1e-05 rad (0.06 mrad)** worst case over
57,600 steps -- a hundredth of the error against hardware. The servo model is the same
servo in all three places, so a policy trained in Isaac is trained against the servo that
is actually on the robot.

### Final parameters (`params/sts3215_12v/m6.json`)

| | value | |
|---|---|---|
| `kt` | 2.2617 | N.m/A (through the gearbox) |
| `R` | 4.1581 | ohm |
| `armature` | 0.023423 | kg.m^2, rotor inertia reflected through the gearbox |
| `max_velocity` | 4.9164 | rad/s, firmware clamp |
| `command_delay` | 0.005565 | s |
| `error_gain_ratio` | 1.0737 | firmware P scaling |
| `friction_base` | 0.14907 | N.m |
| `friction_viscous` | 0.037772 | N.m/(rad/s) |
| `load_friction_external` | 0.15470 | |
| `load_friction_external_stribeck` | 0.25712 | |
| `load_friction_motor_stribeck` | 0.14451 | |
| `load_friction_motor_quad` | 0.0017805 | |
| `load_friction_external_quad` | 0.00068938 | |
| `dtheta_stribeck` | 0.115928 | rad/s |
| `alpha` | 9.9855 | |
| `q_offset` | -0.1 | **ignored**: the publishing rig's calibration error, zeroed here |

As deployed: `kp_fw: 32`, `torque_scale: 1.0`, `stall_torque_cap_nm: 2.94`.

### Speed and torque

| | at 12.0 V | at 11.49 V (measured) | source |
|---|---|---|---|
| No-load speed, motor line | 5.15 rad/s (49 rpm) | 4.93 rad/s (47 rpm) | `0.97*V/kt` |
| **Effective max speed** | **4.92 rad/s (47 rpm)** | **4.92 rad/s (47 rpm)** | `max_velocity` clamp, binds first |
| Stall torque, raw fit | 6.33 N.m | 6.06 N.m | `0.97*V*kt/R` |
| **Stall torque, as configured** | **2.94 N.m** | 2.82 N.m | datasheet cap in `config.yaml` |
| Continuous rating | 0.98 N.m | | datasheet |

**Trust the speed; treat the torque as a datasheet assumption.** The recordings reach a
measured peak of **5.37 rad/s**, slightly past the motor line because gravity drives the
pendulum down faster than the motor alone would -- so ~5 rad/s is corroborated by data.

Torque is **not**. The heaviest load on the rig was 0.433 N.m, about 15% of the datasheet
stall and 44% of the continuous rating, so nothing here exercises the servo anywhere near
stall. The raw fit implies 6.33 N.m, roughly 2x the datasheet; `stall_torque_cap_nm: 2.94`
keeps the sim honest by raising `R` (which caps torque and leaves no-load speed alone --
scaling the duty ceiling instead would wrongly slow the servo down too). Validating the
real stall needs a stalled-torque rig, which this one is not.

## Reading the figures

Both `validate.py` and `replay.py` write into `<session>/plots/`:

| File | What it shows |
|---|---|
| `fit_<model>_h<hole>.png` | recorded vs the BAM rollout, one panel per (trajectory, kp) |
| `summary_<model>.png` | sorted per-recording error, then error by trajectory / kp / hole |
| `sim_<model>_h<hole>.png` | recorded vs MuJoCo vs Isaac, same grid |
| `sim_summary_<model>.png` | the same breakdown, per simulator |

**Read the torque-on number, not the session mean.** A session average mixes two
physically different regimes, and they are not comparable:

* **Torque on** is the only regime the policy ever runs in. This is the number that says
  whether the servo model is good.
* **Torque off** (the grey band, the back half of `lift_and_drop`) has nothing driving
  the joint. It measures whether the model's *stiction* matches the real gearbox, and on
  a ~330:1 STS3215 it usually does not: the real arm stays where it was left, the model
  lets it fall. Eight such recordings can drag a 5 mrad session to 130 mrad while saying
  nothing about tracking.

That split is why `validate.py` prints both and bases its verdict on the torque-on mean.
It matters for backdriven behaviour -- impacts, falls, a leg pushed by the ground -- so
it is worth fixing eventually, but it is not a reason to refit a model that tracks well.

## Why the arm is reduced to two numbers

For a single revolute joint the complete rigid-body dynamics are

```
I * qdd = tau_motor - g * S * sin(q) - friction
```

`I` is the inertia about the rotation axis, and `S = sum(m_i d_i)` is the gravity-torque
coefficient. Every geometric detail (holes, pockets, screws, the double-sided servo mount)
enters **only** through those two scalars. Two bodies with equal `I` and `S` about the same
axis are dynamically indistinguishable, so no experiment on that joint can separate them.

BAM's log format is parameterised as `(mass, arm_mass, length)`, which describes *its* rig:
a rod with a tip weight. Forcing an arbitrary arm into that shape is what would lose
information. Setting `arm_mass = 0` makes it a pure point pendulum, where
`inertia = mass*L^2` and the gravity coefficient is `mass*L`. That is a bijection with `(I, S)`:

```
L = I / S          mass = S^2 / I
```

`rig.py` does this, and the round-trip is exact to machine precision (~3e-16), not
approximate. Modelling the payload as a point mass is a genuine approximation, but a tiny
one: a compact 250 g weight at 150 mm contributes ~0.6% of its parallel-axis term from its
own inertia. Set `payload.inertia_kg_m2` if you modelled it properly.

The one property `(I, S)` **cannot** represent is **flexibility**. A bending arm is not a
rigid body, and its mode gets absorbed into the fit as fictitious friction. Keep the section
deep in the swing plane: a 15x6 mm PLA bar with 250 g at 150 mm resonates near 9 Hz, inside
the excitation band, while 15x12 mm moves it to ~26 Hz.

## Sizing

`rig.py` reports peak gravity torque as a fraction of the 0.98 N.m continuous rating and
warns at both ends. Too high and recordings saturate PWM, the data contamination that cost
the reference project a 29% torque underestimate; `validate.py` screens saturated recordings
out regardless. Too low and the load-dependent friction terms are poorly identified.

Useful scale check: the published STS3215 `armature` is ~0.023 kg m^2 (rotor inertia
reflected through roughly 330:1), which is about 4x a 250 g / 150 mm pendulum. The motor's
own inertia dominates the arm's, so excitation comes from **gravity torque S, not inertia**.
Get the payload mass and hole radii right; a few percent on the inertia tensor does not
matter.
