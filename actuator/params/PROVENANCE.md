# Parameter provenance

Every file here is in BAM's own params JSON schema, which is exactly what
`python -m bam.fit --output <file>` writes. To switch the simulator to a new fit, point
`actuator_params` in `config/config.yaml` at that file; no code changes are needed.

## Validation against our own hardware — 2026-10-09

These parameters started as a placeholder. They are now **the identification we ship**,
because a bench session on our own servo says they already describe it.

- **Session:** `actuator/data/bench_servo_2`, 48 recordings. 4 trajectories x kp {4, 8, 16,
  32} x 3 holes (50/100/150 mm), 6 s each at ~250 Hz, one STS3215, 11.49 V measured.
- **Rig:** our pendulum arm (31.73 g, CoM 65.5 mm from the axis) plus a 280.17 g payload,
  reduced to its equivalent point pendulum by `rig.py`.
- **Result, torque on:** m6 5.53 mrad MAE, m5 5.44, m1 6.14. The m6 hold-out published with
  this fit is 10 mrad, so it describes our servo better than the rig it came from.
- **Chosen: m6.** m5's 0.09 mrad edge is noise and it carries three friction terms pinned at
  ~0 plus `alpha` on its lower bound. m6 is the model with a published reference.
- **Simulator parity:** replaying all 48 logs through MuJoCo and through Isaac/PhysX gives
  5.53 mrad each, agreeing with one another to 6.1e-05 rad.

**The caveat that matters.** A session mean over this data is 131 mrad, and that number is
meaningless on its own: 8 of the 48 recordings are `lift_and_drop` at holes 1-2, where
torque is cut at -90 degrees. The real arm *stays where it was left* (measured drift 1.5 to
120 mrad at every hole, against 0.433 N·m of gravity torque at h2) because the ~330:1
gearbox will not back-drive. The model lets it fall. So **stiction here is too low**, which
is invisible to a tracking policy and wrong for impacts, falls and a loaded leg.

Also unresolved: nothing in this session goes near stall. The heaviest load was 0.433 N·m,
about 15% of the datasheet stall, so the 6.3 N·m the fit implies is still untested. That is
why `stall_torque_cap_nm: 2.94` stays on.

Reproduce with `python -m actuator.validate --session actuator/data/bench_servo_2` and
`python -m actuator.replay --session actuator/data/bench_servo_2`.

## `sts3215_12v/` — the shipped fit

- **Source:** Rhoban BAM pull request #20, *"add STS3215 12V params"*. It is open and not merged.
  - Fork/branch: `T-K-233/bam`, `add-sts3215-12v`
  - Commit: `ca3b3a3bae9373017ff75e37f3e486cdb782049b`
  - Path: `bam/params/feetech_sts3215_12V/m{1..6}.json`
- **What it is:** an identification of *someone else's* 12 V STS3215 on *their* pendulum
  rig. The arms were 0.10 m / 20 g and 0.15 m / 22.5 g, the tip masses were 0.21 / 0.41 /
  0.61 kg, there were 97 logs, and kp = 8 was held out.
- **Values are byte-for-byte as published**, only re-indented. Do not hand-edit them.
  Replace the whole file with our own `bam.fit` output once identification has run.
- **Reported hold-out MAE:** about 10 mrad for M6 and about 20 mrad for M1, almost all of it
  on `lift_and_drop`.

Known features of this fit, recorded so nobody "fixes" them by accident:

- `q_offset` sits at the −0.1 bound in m1, m2, m3, m5 and m6. It is a testbench calibration
  term, and **our kernel ignores it** on the robot.
- m1 `friction_base` is at its 0.2 upper bound. m2 and m6 `alpha` sit at ~10, the bound,
  which turns the Stribeck curve into a near step. m4 `dtheta_stribeck` sits at its 1.0 bound.
- `kt`, `R` and `error_gain_ratio` are degenerate as a group. The logs only constrain
  `kt·error_gain_ratio/R` and `kt²/R`, so do not read physical meaning into `kt` alone.
- With `vin = 12`, the duty-limited stall torque is `0.97·12·kt/R` ≈ **6.3 N·m**. That is about
  2× the 2.94 N·m datasheet stall. BAM does not model the servo's current or overload
  protection. Our own validation will show whether the real servo reaches this.

## `sts3215_7_4v/` — reference only

- **Source:** Rhoban BAM `main`, commit `e9a619d56da5236206f4de6ceec2c1ee1b497b5c`, path
  `bam/params/feetech_sts3215_7_4V/`.
- This is the set the lost `actuator/bam_sts3215.py` referenced (`motor_name =
  "feetech_sts3215_7_4V"`).
- **Do not use it for this robot.** The PR #20 author found these values predict the arm
  falling where a real 12 V servo holds it.

## Licence

BAM is © 2025 Marc Duclusaud & Grégoire Passault and is licensed under Apache-2.0. The full
text is in `LICENSE-BAM`. `actuator/bam_kernel.py` is a port of BAM's `model.py`,
`actuator.py`, `feetech/actuator.py` and `simulate.py`, and carries a modification notice
to that effect.
