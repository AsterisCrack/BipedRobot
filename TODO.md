# TODO — deferred work

Known issues and improvements deliberately deferred, with enough context to act on each
without re-deriving it. Each entry names the file and the reason it was parked.

> **See also `TODO_SIM2REAL.md`** — the actuator-fidelity work programme (BAM servo model,
> backlash, the acceleration-cap finding, and the training-config changes that depend on them).
> That is an active plan rather than a backlog, so it lives in its own file. Several items
> below are superseded by it, most notably #1 and #2 on `effort_limit` and the actuator model.

---

## Sim-to-real fidelity

### 1 + 2. Actuator model — DONE (2026-09-25)
Was: `ImplicitActuatorCfg(stiffness=80, damping=8, effort_limit=3.0, velocity_limit=5.0)`.
`effort_limit=3.0` was the **stall** rating; continuous is ~0.98 N·m @ 12 V. Implicit
actuators also silently discard `velocity_limit`, so the sim servo delivered full torque at
any speed.

Now `DCMotorCfg(stiffness=20, damping=2.0, effort_limit=0.98, velocity_limit=4.71,
saturation_effort=2.94)` in `envs/assets/robotV2/biped_robot.py`. Envelope: flat 0.98 N·m to
the corner at 3.14 rad/s, then linear droop to zero at 4.71.

`stiffness` had to move with `effort_limit` — at kp=80 a 0.98 N·m ceiling saturates after
0.70° of error (bang-bang). At kp=20 it saturates at 2.8°, and holding a 30° knee bend in
single support (~0.42 N·m) costs 1.2° of droop.

Verified before switching: Isaac Lab zeroes the PhysX drive gains for explicit actuators
(`articulation.py:1736-1740`) so there is no double PD; `set_joint_position_target` still
feeds the actuator; `randomize_actuator_gains` still applies because it writes
`actuator.stiffness` directly (`events.py:631`).

**Known consequence:** honest torque caps knee bend at roughly 60° in single support
(0.81 N·m of 0.98). 90° needs 1.15 N·m and is not holdable. If the gait work wants deeper
bend than that, the constraint is real and the answer is a lighter robot or a bigger servo,
not a bigger number in the config. Watch `Torque/frac_saturated`.

**Still open here:** friction is a plain Coulomb `friction=0.2`. The Rhoban/BAM identified
STS3215 model (Stribeck / load-dependent / quadratic) would be better, but every published
fit is for the **7.4 V** variant and this robot is **12 V** — do not mix those. Archaeology:
`actuator/__pycache__/bam_sts3215.cpython-311.pyc` exists but its source is gone; an
identified model for the real servo existed at some point and is worth recovering.

### 3. Missing DR: effort_limit / voltage sag — DONE (2026-09-25)
Isaac Lab has no built-in effort-limit randomizer (`randomize_actuator_gains` only covers
kp/kd), so `envs/isaaclab/mdp/events.py` now provides `randomize_actuator_effort_limit`,
wired as the `randomize_effort_limit` event at `scale (0.8, 1.1)`.

`ActuatorBase.effort_limit` is a per-env `(num_envs, num_joints)` tensor and is what explicit
actuators clip against internally, so writing it is sufficient — no sim-side write needed, and
`effort_limit_sim` is deliberately left at its 1e9 park value to avoid double-clipping. The
nominal limit is cached on first use so repeated scaling cannot compound toward zero.

**Encoder / IMU bias — DONE (2026-09-26).** `imu_accel_bias` (0.2 m/s²), `imu_gyro_bias`
(0.02 rad/s) and `encoder_offset_rad` (0.0175 ≈ 1°) are drawn uniformly in ±range at each
reset and held **constant for the whole episode**.

Constant is the point. `observation_noise_model` resamples every step, so across the 5-frame
history stack it largely averages out; the policy can filter it. A real MPU-6050 has a fixed
null offset per power-cycle, and each servo's encoder zero carries a fixed assembly error — so
the policy must stay upright while reading a consistently *wrong* value. That is the harder and
more realistic failure mode.

Applied to the observation copies only (`joint_pos_obs` / `imu_acc_obs` / `imu_gyro_obs`),
never to the `self.*` attributes the rewards and `_get_dones` read.

**Still missing:** IMU drift (bias that walks *within* an episode) and gyro scale-factor error.

### 4. `randomize_friction` band is narrow — DONE (2026-09-25, widened again 2026-09-28)
Now `(0.3, 1.4)` static and dynamic, plus `restitution_range` `(0.0, 0.3)` (was pinned at 0).

**How friction actually reaches the contact:** we randomize the *robot's* material, not the
floor's. But `sim.physics_material` uses `friction_combine_mode="multiply"` and the terrain
material is exactly `1.0`, so `mu_effective = mu_robot x 1.0` — the robot's draw passes
through unchanged and the range above IS the foot-ground mu. **This only holds while terrain
friction is 1.0**; change it and the whole range scales silently.

0.5 was still a reasonable floor. Wet tile/laminate is mu ~0.2-0.3, which is exactly where a
flat-footed biped slips, so the lower bound went to 0.3.

### 4b. DR widening pass — DONE (2026-09-28)
Triggered by occasional falls. Full inventory widened:

| Term | Was | Now |
|---|---|---|
| `randomize_mass` (12 leg links, scale) | 0.8–1.2 | **0.75–1.25** |
| `randomize_actuator_gains` (kp/kd, scale) | 0.8–1.2 | **0.7–1.3** |
| `randomize_effort_limit` (scale) | 0.8–1.1 | **0.7–1.15** |
| `randomize_friction` static/dynamic | 0.5–1.3 | **0.3–1.4** |
| `randomize_friction` restitution | 0.0–0.0 | **0.0–0.3** |
| `curriculum_dr_com_range` | ±0.01 m | **±0.02 m** |
| `curriculum_dr_payload_max` | 0.15 kg | **0.3 kg** (18% of body) |
| `push_robot` x / y | ±0.3 / ±0.2 | **±0.4 / ±0.3** |
| `push_robot` yaw | **absent** | **±0.6 rad/s** |

**NEW: `randomize_joint_params`** — `randomize_joint_parameters` scaling joint `friction`
(0.5–1.5 on nominal 0.2) and `armature` (0.85–1.15 on nominal 0.04). This was the largest
unmodelled physical variation: stiction differs substantially between nominally-identical
STS3215s and directly sets how much torque is lost before a joint moves at all.

`mode="startup"`, **not** `"reset"`, deliberately — these are manufacturing properties. A given
servo's stiction does not change between episodes, so re-drawing per reset models something
impossible. One draw per env gives 4096 distinct robots and the shared policy must be robust
across that population; it also avoids the per-reset CPU round-trip the term warns about.

**Which knob is live matters.** `curriculum_dr_events = ["randomize_com", "randomize_payload"]`,
so `_apply_curriculum` overwrites *only* those two event params every reset — editing their
`EventTerm` params does nothing, you must edit `curriculum_dr_com_range` /
`curriculum_dr_payload_max`. The reverse holds for mass/gains/friction/push: their `EventTerm`
params are live and the `curriculum_dr_*` values for them are documentation only.

**Still missing:** backlash/deadband (~0.3–1° gear lash, no Isaac primitive, needs a custom
event), IMU drift within an episode, gyro scale-factor error, per-env randomized servo lag
(currently fixed at 1 step for all envs), and spatially-varying terrain friction patches.
`action_delay_steps_range` is still `[0,0]` — it was tested at `[1,3]` and hurt, but that was
under massless links and a 3x overstrong actuator, so it is worth retesting.

---

## Reward / gait

### 5. Siekmann periodic reward composition
The one genuinely valuable idea from the external report — but **the report describes it
wrongly**, and the errors would have broken the implementation:

- It is **two per-leg offset sine signals** `sin(2*pi*(phi+theta_L))`, `sin(2*pi*(phi+theta_R))`
  with `theta_L=0, theta_R=0.5` — **not** `(sin phi, cos phi)`. That encoding belongs to a
  different paper (2207.12644); the report swapped its own citations.
- Real coefficients are **penalties only**: `c_swing^frc = -1`, `c_stance^spd = -1`, with the
  complementary terms zeroed. The report's symmetric reward-and-penalise form is its invention.
- `I_i(phi)` is a **binary** variable whose phase *boundaries* are Von Mises-distributed;
  training uses `E[I_i(phi)]`, a smoothed square wave. A two-logistic-gate surrogate is fine
  but should be labelled as such.
- "stance ratio ~= 0.6" is **invented** — the paper's `r` is the *swing* ratio and it publishes
  no per-gait table. Treat as a starting guess.

Paper: Siekmann et al., ICRA 2021, arXiv:2011.01387.

### 6. `gait_phase_contact` is 0 while the phase clock is still in the observation
Obs dims 48-49 carry a clock no reward uses — 2 wasted input dims and a free noise source.
Either re-enable the weight or drop the clock from obs. Now safe to re-enable: the feet-order
inversion and the phase-mirror sign bug that would have corrupted it are both fixed.

### 7. `swing_foot_height` uses absolute world Z — MOOT while terrain is flat
Terrain is now `MeshPlaneTerrainCfg(proportion=1.0)`, so world Z equals height above ground and
the term is correct as written. **Becomes a real bug the moment rough terrain is re-enabled**
(see item 17) — it would then pay a bonus proportional to the env's patch elevation.
Original description below.
`rewards.py` — reads `feet_pos_w[:,:,2]`. On rough terrain (`use_rough_terrain: true`) this
pays a bonus proportional to whatever elevation the env's patch sits at. Active at weight 0.15.
Same class of bug in `base_height_l2` (currently disabled at 0).

### 8. `foot_separation` heading dependence — FIXED (2026-09-25)
Was measured in **world Y** at weight −3.0. Reset yaw is `U(−pi, pi)` and heading commands keep
rotating the robot, so whenever it faced world ±Y the measured separation collapsed to ~0 and
charged the full penalty for a perfectly good stance — roughly half the time.

Now measured along **body +Y**, verified from the URDF rather than guessed: `l_hip_yaw` sits at
origin `y = +0.021672` and `r_hip_yaw` at `y = −0.021672` relative to `base_link` — 4.33 cm
apart along Y, matching the "±21.7 mm" figure in the code comments. +Y is left; standard
REP-103 (X forward, Y left, Z up).

**Correcting the record:** an earlier body-frame attempt was reverted on the belief that body Y
was not the lateral axis. That was wrong. The formula was correct; the run it was judged in
carried three other changes and a collapsing policy, and `Reward_Terms` are deflated by
`actual/max` episode length, so a short episode alone shrinks the logged value ~10x.

Bonus confirmation from the same URDF block: `l_hip_yaw` axis is `[0,0,1]`, `r_hip_yaw` is
`[0,0,-1]`. With the sagittal reflection M = diag(1,−1,1), `M·â_L = [0,0,1]`, so
`â_R = −M·â_L` → `q_R = +q_L` → mirror sign **+1**. Independently confirms the all-`+1`
`mirror_joint_signs` for V2.

### 9. `undesired_contacts` binary cliff — FIXED (2026-09-26)
Was `any(force > 1.0)` over all non-foot bodies: a 2 N brush cost the same as a 40 N impact,
with zero gradient on both sides of the threshold. Matters more now the gait is dynamic — V2's
hips are 4.3 cm apart so the legs legitimately pass close during swing.

Now `sum(clamp((force - threshold)/saturation, 0, 1))` with `saturation = 20 N`. A body at
≥21 N still costs 1.0 (identical deterrent for real impacts, existing weights preserved), but a
light graze costs proportionally little and there is a usable gradient between.

**Still open — cannot separate ground contact from self-collision.** `enabled_self_collisions`
is True, and the only way to distinguish the two is `force_matrix_w`, which needs
`filter_prim_paths_expr` set. `__post_init__` clears that whenever `use_rough_terrain` is true —
and it currently *is* true even though `ROUGH_TERRAINS_CFG` now generates a flat plane. So
setting `use_rough_terrain: false` would restore the filter and also give `feet_slide` back its
3-frame contact history. Worth doing, but it changes the terrain path, so it needs its own run.

### 10. `dof_acc_l2` had no config key — DONE by user (2026-09-25)
Now set to `-1.0e-06` in `config.yaml`.

### 11. "Both feet airborne" termination — STALE, does not exist (checked 2026-09-25)
The external audit claimed `_get_dones` terminates on dual flight. It does not — the current
tree terminates only on tilt > 0.784 rad, timeout, and a NaN guard. Nothing to do.

### 12. Termination penalty almost never fired — FIXED (2026-09-25)
Far worse than the "constant mismatch" the audit described. The reward and the termination
disagreed on three things at once:

| | `_get_rewards` | `_get_dones` |
|---|---|---|
| Threshold | `0.78` | `0.784` |
| Source | `self.projected_gravity_b` (IMU copy) | `self.robot.data.projected_gravity_b` |
| Freshness | **one step stale** | current |

`self.projected_gravity_b` is assigned only in `_get_observations`, which runs *after*
`_get_rewards`. So on the step the robot crossed the tilt limit, the reward read the previous
step's orientation — still upright — and charged nothing, while `_get_dones` read fresh data
and terminated. The penalty systematically missed the one step it exists for.

Evidence: `termination_penalty` logged −0.0022, but a single firing at weight −5.0 over the
20 s normaliser is −0.25 — i.e. it fired ~1% as often as it should.

Now both call a shared `_tilt_exceeded()` reading the articulation directly. The helper also
clamps before `acos`, which previously could return NaN on normalisation round-off (and NaN
compares False, silently suppressing the termination).

**Expect `Reward_Terms/termination_penalty` to jump by ~100x** on the next run. That is the
fix working, not a regression — a real −5.0 stability signal is now live for the first time,
and it may noticeably change behaviour.

---

## Training / algorithm

### 13. Replay buffer window — RE-ASSESSED, probably NOT urgent (2026-09-26)
The original framing ("holds under a quarter of one episode, SAC is nearly on-policy, likely a
large cheap win") is misleading. Corrected numbers, with symmetry augmentation doubling the
worker axis to 8192:

`max_size = 1e6 // 8192 = 122` rows = **2.44 s** at 50 Hz, against ~19 s episodes.

**Why "less than one episode" does not mean what it sounds like.** The buffer is
`(rows, workers, dim)`. Those 8192 workers reset at independent times, so at any instant they
span *all* episode phases — some at step 10, some at step 900. The critic therefore does see
late-episode states; it just doesn't see long contiguous stretches of a single trajectory. With
`return_steps: 5` only 5 consecutive rows are ever needed, and 122 is ample.

**The real property is low reuse, not low coverage.** Each iteration draws
`batch_iterations 10 × batch_size 2048 = 20,480` from 1e6 slots, so over the 122 iterations it
takes to refresh the buffer each transition is sampled ~2.5 times. That is high-throughput /
low-reuse — the same regime FastTD3 and PQL deliberately operate in, leaning on data volume
rather than sample reuse. Not pathological.

**Cost if we did want to change it.** Per (row, worker): actor obs 250 + critic obs 60 +
next actor 250 + next critic 60 + actions 12 + rewards/discounts/resets/terminations 4 = 636
floats = 2.5 KB. Current total ≈ **2.54 GB** on GPU. Holding a full 19 s episode would need
~20 GB — impossible. Doubling `buffer.size` to 2e6 buys 4.9 s for +2.5 GB.

Note the actor/critic obs are stored twice (obs and next_obs). Normally `next_obs[t] == obs[t+1]`
and could be indexed rather than duplicated, saving ~49% — but `accumulate_n_steps` overwrites
`next_observations` with the state *n* steps ahead, so the duplication is load-bearing given the
n-step design.

**Verdict:** leave it. Revisit only if a diagnostic actually implicates replay staleness.

### 14. `exploration:` block is at the wrong nesting level
`config.yaml` has it top-level; `schema.py:74` expects it under `model:`. Pydantic's default
`extra="ignore"` drops it silently.

**Currently moot** — SAC hard-wires `NoActionNoise`, which has no scale/min_scale/decay_rate.
Would bite immediately if you switch to `model: "ddpg"` or `"d4pg"`, which do read it.

One live side effect: `start_steps: 4096` is ignored; `NoActionNoise` uses its hardcoded
`10000`. Negligible (env-steps, ~2-3 iterations at 4096 envs).

### 15. `MPO.update` has no `**kwargs` — DONE (2026-09-26)
Added `**kwargs` so the trainer's `agent.update(**infos, steps=...)` no longer raises
`TypeError` if MPO is ever selected.

### 16. `try_symmetry.py` is not a valid oracle
Fixed already: labels, 50-dim coverage, tolerance 1e-4. **Still broken:** env 1's commands and
gait phase are resampled independently, so dims 9-11 and 48-49 can never match even with a
perfect mirror. To be a real oracle it must pin env 1's commands to `mirror(commands_0)` and
its phase to `phi_0 + pi`. Also needs a GUI + human; no headless pass/fail mode.

---

## Environment / config hygiene

### 17. Terrain — RESCALED AND RE-ENABLED (2026-09-27)
Rebuilt `ROUGH_TERRAINS_CFG` for a 29 cm blind biped. Old version was flat-only because the
amplitudes were unusable.

**What was wrong:** 20 mm random bumps and 25 mm waves. At a 5.9× scale to a 1.7 m human that
is 12–15 cm — curbs, walked blind. Also `vertical_scale = 0.005` meant a 10 mm feature had only
**two** height levels, so "rough" was really a blocky two-tone staircase.

**Now:** worst feature ~12 mm (≈7 cm human-scale, i.e. rough grass), `vertical_scale = 0.001`,
six sub-terrains — flat 25%, fine rough 20%, waves 20%, slopes up/down 10% each, random grid
15%. `downsampled_scale=0.2` on the rough terrain interpolates between coarse samples so it is
rolling texture rather than per-pixel noise the foot can catch on.

Sizing rule used: keep the worst feature under ~20–25% of swing-foot clearance (now ~5–7 cm
after the `swing_foot_height` fix).

**`HfRandomUniformTerrainCfg` still ignores `difficulty`** (documented in `hf_terrains.py`), so
it always runs at full range — hence its deliberately small 8 mm ceiling.

**Progressive curriculum deliberately NOT enabled.** Setting `TerrainGeneratorCfg.curriculum=True`
only helps combined with `use_terrain_curriculum: true` → `max_init_terrain_level=0`. But
`_compute_env_origins_curriculum` assigns each env to a **patch origin**, so that would pack all
4096 envs onto the 20 row-0 patches (~205 each) rather than spreading over 400. With
`curriculum=False` each patch draws `difficulty ~ U(0,1)`, which already gives the full
easy→hard spread as randomization. To enable progression safely you would need many more
patches (raise `num_cols`), or to confirm inter-env collision filtering makes co-located envs
harmless.

**Still open:** `use_rough_terrain: true` clears `filter_prim_paths_expr`, which kills
`force_matrix_w` — so `undesired_contacts` cannot separate ground contact from self-collision,
and `feet_slide` loses its 3-frame contact history. Now unavoidable since rough terrain is
wanted; would need the filter pointed at the generated terrain prim instead of the ground plane.

### 17-old. Terrain — SIDESTEPPED by user (2026-09-25): now flat only
`ROUGH_TERRAINS_CFG` sub-terrains are commented out and replaced with
`MeshPlaneTerrainCfg(proportion=1.0)`, so the roughness problem is moot for now.

If rough terrain is re-enabled, the original issues still apply:
`HfRandomUniformTerrainCfg` **ignores the `difficulty` parameter** (documented in
`hf_terrains.py:29`), so its 20 mm steps at 10 cm pitch are always max difficulty — on a 30 cm
robot with ~5 cm feet and **no height scanner**. `TerrainGeneratorCfg.curriculum` is also never
set (defaults `False`), so all patches draw difficulty uniformly. The cfg comment claims
"max 8mm bumps, 12mm waves"; the commented config is 20 mm / 25 mm.

Also note `use_rough_terrain: true` still clears `filter_prim_paths_expr`, which kills
`force_matrix_w_history` — so `feet_slide` loses its 3-frame contact history and
`undesired_contacts` falls back to raw `net_forces_w` and fires on leg-to-leg self-contact.

### 18. Command resampling slower than the episode — DONE by user (2026-09-25)
`resampling_time_range` now `[4.0, 8.0]` (was `[8.0, 12.0]`), so the policy sees multiple
commands per episode and actually trains on transitions.

### 19c. `randomize_com` was a random walk — ROOT CAUSE FOUND (2026-09-26)
This is why enabling COM randomization always stopped the robot walking, across multiple
attempts and months of suspicion. **The magnitude was never the problem — the accumulation was.**

`isaaclab.envs.mdp:randomize_rigid_body_com` (events.py:426-429) does:

```python
coms = asset.root_physx_view.get_coms().clone()   # CURRENT com
coms[env_ids[:, None], body_ids, :3] += rand_samples   # ADD
```

Unlike `randomize_rigid_body_mass` — a `ManagerTermBase` that resets to `default_mass` first —
this plain function never resets to a nominal. Under `mode="reset"` each episode stacks another
offset onto the last. Its docstring's "use only during initialization" advice is really about
this non-idempotency, not the CPU round-trip.

Uniform ±1 cm has σ = 0.01/√3 = 5.77 mm per reset, compounding as σ√N:

| Resets/env | base_link drift (1σ) | Whole-body COM shift |
|---|---|---|
| 29 (~40M-step run) | 3.1 cm | 1.2 cm |
| 100 | 5.8 cm | 2.2 cm |
| 400 | 11.5 cm | 4.4 cm |

Whole-body shift is `0.633/1.661 = 0.381 ×` the base offset. Past ~100 resets the COM is
displaced by roughly half a foot length — unbounded, different per env, and invisible to the
policy since COM is not observed. Some envs become genuinely unbalanceable.

**Fix:** `envs/isaaclab/mdp/events.py: randomize_rigid_body_com_absolute` caches the nominal
COM on first use and writes `nominal + offset`, so every episode is an independent draw.

**Watch `Diagnostics/com_drift_max_m`** — it must stay flat near the configured range. If it
climbs steadily, something is compounding again.

**General lesson:** any Isaac Lab event used with `mode="reset"` must be checked for whether it
resets to a nominal or accumulates. `randomize_rigid_body_mass`, `randomize_actuator_gains` and
`randomize_joint_parameters` reset; `randomize_rigid_body_com` does not. Our custom
`randomize_actuator_effort_limit` caches a nominal for exactly this reason.

### 19. `randomize_com` was commented out — DONE (2026-09-25)
Re-enabled at `±0.01 m` on `base_link`. The curriculum plumbing already existed
(`_apply_curriculum` reads `curriculum_dr_com_range`) and the name was already in
`curriculum_dr_events`, so it ramps 0 → full over `curriculum_dr_full_steps` rather than
biting from step 0.

**Two things to watch.** (a) You previously reported that enabling `randomize_com` made things
"much worse" — but that was measured when leg links were going massless ~17-30% of the time
and the actuator was 3x too strong, so the observation is confounded. If it still hurts,
`curriculum_dr_com_range` is the single knob. (b) Isaac Lab's own docstring warns
`randomize_rigid_body_com` "uses CPU tensors" and is recommended for init only — at 4096 envs
resetting often this may cost throughput. Watch `train/steps_per_second`.

### 19b. `curriculum_dr_mass_range` was a landmine — FIXED (2026-09-25)
`_apply_curriculum` scaled the mass range as `(lo * dr, hi * dr)`, which is correct for an
`add` operation but catastrophic now that `randomize_mass` uses `scale`: the neutral point is
1.0, not 0, so low `dr` would have driven the scale factors toward **zero** and reproduced the
massless-link bug through a different path. Now interpolates toward 1.0 like the gains and
friction terms. `curriculum_dr_mass_range` also changed `(-0.2, 0.4)` kg → `(0.8, 1.2)` scale.
Note this only ever fired if `"randomize_mass"` was added to `curriculum_dr_events`, which it
currently is not — it was latent.

### 20. `reset_base` / `reset_robot_joints` are applied twice per episode
`super()._reset_idx()` runs all reset-mode events via the EventManager, then `biped_env.py`
manually calls `reset_joints_by_offset` and `reset_root_state_uniform` again. Last write wins,
so wasteful rather than wrong. Note the EventManager path uses `reset_joints_by_scale`, which
is a **no-op for V2** (all default joint positions are 0.0, and scaling zero gives zero).

### 21. Dead config values — MARKED IN-FILE (2026-09-26)
All now carry a comment at the declaration naming the real knob, rather than being silently
inert. Not deleted: `train.py`/`common.py` transfer them via `hasattr`, so removing the fields
would make those assignments create unknown attributes on a `configclass`.

- `action_scale = 0.7` — marked; becomes live only if the residual action map is adopted.
- `push_interval_s` / `push_vel_range` — marked; real cadence is the `push_robot` event's
  `interval_range_s=(1.0, 5.0)`, real magnitude is its `params["velocity_range"]`. The dead
  `self.push_interval_steps` computation in `biped_env.py` was removed outright.
- `velocity_limit` — **no longer dead**: `DCMotorCfg` uses it for the torque-speed droop. It
  was only discarded under the old `ImplicitActuatorCfg`.
- `--config_path` default was `config/train_config.yaml`, which does not exist. Fixed to
  `config/config.yaml` in both `train.py` and `play.py`.

Original list below.
- `action_scale = 0.7` (`biped_env_cfg.py:130`) — declared, never read anywhere.
- `push_interval_s` / `push_vel_range` — `push_interval_steps` is computed and never used; the
  real cadence is the event's own `interval_range_s=(1.0, 5.0)`.
- `velocity_limit=5.0` — silently discarded by `ImplicitActuator` (see item 2).
- `reset_robot_joints` `position_range`/`velocity_range` — overwritten every reset by
  `_apply_curriculum` with `curriculum_init_range_max` (0.10). Marked in-file.
- Stale "removed" comments on `feet_air_time` / `swing_foot_height`, which are both **active**.

### 22. Joint-limit table positional binding — VERIFIED CORRECT (2026-09-25)
`biped_env.py` builds `joint_limits_min/max` from `cfg.joint_limits` by row index with no
reference to `robot.joint_names`, and `mirror_joint_perm` rests on the same assumption. The
startup diagnostic confirms the runtime order matches the documented one exactly:

```
0 l_hip_yaw    1 r_hip_yaw     2 l_hip_roll    3 r_hip_roll
4 l_hip_pitch  5 r_hip_pitch   6 l_knee        7 r_knee
8 l_ankle_roll 9 r_ankle_roll 10 l_ankle_pitch 11 r_ankle_pitch
```

So the limits table and the adjacent-pair mirror permutation are both correct. Feet order also
confirmed `[r_foot_link_1, l_foot_link_1]` after the `preserve_order=True` fix.

The binding is still positional and unasserted, so it could silently break if the USD is ever
regenerated with a different joint order. The diagnostic print remains; consider promoting it
to an assert against a hardcoded expected list.

### 23. `Reward_Terms/*` scaling is misleading
`biped_env.py:775` divides by `max_episode_length * step_dt` (20 s), not actual episode length.
Terms are deflated by `actual/max` and **rise as episodes lengthen even at a constant reward
rate**. Ratios between terms are still valid; absolute values are not. Also, `reward_scale` is
applied to `total_reward` only, so per-term curves won't sum to the logged return if it != 1.

The new `Torque/*` scalars do normalise by actual episode length, so they are directly
comparable against the 0.98 N·m rating. Consider doing the same for `Reward_Terms/*` — it
would change the scale of every historical curve, so it needs a deliberate cutover.

### 27. `Curriculum/*` scalars were never reaching TensorBoard — FIXED (2026-09-25)
`_apply_curriculum` runs at the top of `_reset_idx` and writes `Curriculum/dr_scale` etc. into
`extras["log"]`, but line ~785 then did `self.extras["log"] = dict()`, wiping them before they
were ever emitted. This is why the DR curriculum state had to be derived analytically instead
of read off a chart. Now carried across explicitly by prefix.

---

## Cross-simulator (MuJoCo)

### 24. Isaac vs MuJoCo parity — LARGELY FIXED (2026-09-28)
MuJoCo (`envs/mujoco/biped_env_v2.py` + `envs/assets/robotV2/robot_v2.xml`) is the sim-to-real
validation path, so it has to be the same robot as Isaac.

**The original TODO entry was substantially wrong.** Audited against the live files:

| Claim | Reality |
|---|---|
| Control rate 100 Hz | **False.** 200 Hz physics / decimation 4 = 50 Hz. Already matched. |
| Action maps to ±pi | **False.** Affine onto the same Isaac degree table. Already matched. |
| `lin_acc` excludes gravity | **False.** Subtracts `(0,0,-9.81)` = specific force, same as Isaac. |
| `_x`/`_y` axes swapped vs URDF | **False.** Verified all 12 joints; axes and body y-offsets match the URDF exactly. |
| L/R bodies swapped | **False.** Every `l_*` body carries `+y` pose/inertial, every `r_*` carries `−y`. |
| Mass 1.47 kg, 3 `<inertial>` commented out | **Wrong numbers, real bug underneath** — see below. |

**Actually fixed:**
- **`base_link` had no `<inertial>` at all.** MuJoCo auto-computed mass from mesh volume at the
  default 1000 kg/m³: **0.7214 kg vs the URDF's 0.633299**. Total 1.749 vs 1.661 kg (+5.3%),
  all of it in the torso where it most affects balance. Added the URDF's tag; total is now
  **1.660997 kg**, matching Isaac to 3 µg.
- **PD gains were 4× off**: kp 80→**20**, kd 8→**2.0**.
- **Torque was a flat `clip(±3.0)`** — full torque at any speed, the single biggest way a sim
  servo flatters the real one. Replaced with `_clip_effort_dcmotor`, a port of Isaac's
  `DCMotor._clip_effort`: flat ±0.98 N·m to the corner at 3.14 rad/s, linear droop to 0 at
  4.71. Verified numerically against the Isaac formula.
- **Hard `qvel` clamp at ±5 rad/s removed.** Isaac does not clamp joint velocity — for an
  explicit actuator `velocity_limit` feeds the torque-speed curve, it is not a state
  constraint (the USD's own limit is ~100 rad/s). Overwriting `qvel` mid-episode also
  injects/removes energy the integrator never accounts for. The envelope now enforces the
  ceiling the physical way.
- **`GAIT_CLOCK_FREQ` was still 1.5** while Isaac moved to 5.0 (item 30). This feeds observation
  dims 48-49, so the policy was being fed a clock at the wrong rate. Now 5.0.
- **IMU offset sign flip.** `_imu_offset_b` was hardcoded `[+0.023673, 0, 0.025133]` while the
  MJCF site and Isaac's `ImuCfg` both use **−**0.023673. It feeds the `ω × r` tangential term,
  so IMU acceleration was wrong whenever the base rotated — i.e. throughout any gait. Now read
  from `model.site("imu_site").pos` so it cannot drift again.
- `ctrlrange` ±3 → **±2.94** (stall torque, the true physical ceiling) as a backstop.

**Still open — needs verification before touching:** the MJCF `<joint range=>` values are the
wide CAD limits (knee ±120°, hip_pitch ±90°) rather than Isaac's action-space table
(knee −120/+5). **This may be correct parity, not a bug.** Isaac has the same two-tier
structure: `cfg.joint_limits` drives the action map and the `dof_pos_limits` reward, while the
articulation's *physical* limits come from the USD (`soft_joint_pos_limit_factor=0.9`) and are
read separately into `self.joint_pos_limits`, which is never compared against the table. If the
USD limits are also the wide CAD ones, MuJoCo is already right. **Check Isaac's actual
`soft_joint_pos_limits` before changing anything here.**

**Sensor lag — PORTED (2026-09-28).** `SERVO_OBS_DELAY = 1`, `IMU_OBS_DELAY = 1` in
`biped_env_v2.py`, applied in `_build_obs` via ring buffers seeded at reset from the current
reading (zero-filling would feed a fabricated zero projected-gravity for the first D steps —
an orientation that reads as "lying on its side").

**Deliberately deterministic, not randomized.** MuJoCo stands in for *one* physical robot, so
it gets a fixed lag rather than Isaac's per-episode draw. Isaac samples IMU lag from `[0,1]`;
1 is the conservative end. Set either constant to 0 to A/B the effect on a trained policy.

Rationale: the policy is *trained* with these delays and has learned to compensate for a known
phase lag. Handing it zero-lag observations is as much a distribution shift as too much lag —
removing a disturbance a policy was trained with is not "easier".

**Still not in MuJoCo, on purpose:** the per-episode sensor *biases* (`imu_accel_bias`,
`imu_gyro_bias`, `encoder_offset_rad`), observation noise, pushes, and all dynamics
randomization. `Randomizer` is constructed with a `None` config (`biped_env_v2.py:223`) so
`randomize()` returns immediately — MuJoCo runs one nominal, undisturbed robot by design.
**Consequence: a MuJoCo score is not comparable to an Isaac score.** MuJoCo is also flat
(`<geom type="plane">`) while Isaac trains on six rough sub-terrains.

Reward constants (`swing_foot_height` window/weight, `foot_separation` min_sep) have also
drifted from Isaac. Those affect only logged eval metrics, not policy behaviour.

### 25. `lin_acc` as the first three actor obs — RESOLVED, no change needed (2026-09-27)
Isaac's `ImuCfg.gravity_bias` defaults to `(0, 0, 9.81)` and `imu.py:201` adds it to the
finite-differenced acceleration. So `lin_acc_b` is **specific force** — exactly what a real
MPU-6050 reports. MuJoCo does the same thing (`a_world - (0,0,-9.81)`, rotated to body).
Both sims already match the hardware convention; the TODO's premise was wrong.

---

## Repo hygiene

### 26. Broken imports / install
- ~~`requirements.txt` hyphens vs underscores~~ — FIXED 2026-09-26.
- `src/mujoco/train.py`, `train_lstm.py`, `tests/test_env.py` cannot import — stale
  `envs.distributed` / `envs.mujoco_env` paths; `envs/` has no `__init__.py`.
- `--config_path` defaults to the non-existent `config/train_config.yaml`.

---

### 31. `swing_foot_height` was saturated against its own cap — FIXED (2026-09-26)
**Root cause of the stiff, straight-leg gait.** The call used
`min_height=0.02, max_height=0.04`, so the reward clamps at 0.02 per foot. Measured
`raw_avg = 0.0165` — **82% of the cap**. The policy lifted its foot to 3.65 cm and stopped,
because past 4 cm there is no gradient whatsoever.

Foot clearance and knee flexion are geometrically coupled: you cannot raise the foot without
shortening the leg (knee bend) or hiking the hip. **Capping clearance therefore caps knee
bend**, which is exactly the symptom. It also starves `gait_phase_contact`, since a clean
swing/stance alternation needs the foot actually off the ground.

The weight was also below break-even: knee flexion to ~0.6 rad costs ~0.024/step in
`joint_deviation_knee` + `dof_pos_l2`, while clearance at weight 0.15 paid ~0.0075/step.
**Lifting higher was a net loss and the policy was behaving optimally by refusing.**

Fix: `max_height` 0.04 → 0.09, weight 0.15 → 1.5 (break-even is ~0.5, so 3x margin).

**Diagnostic pattern worth reusing:** when a clamped reward sits near its cap, it has no
gradient left and is silently inert at the margin. Check every `clamp`/`clip` in rewards.py
against its measured value — `feet_air_time` (threshold 0.3) and `knee_bend_on_touchdown`
(max_bend 1.0) are the other clamped terms.

### 30. `gait_clock_base_freq` was miscalibrated — FIXED (2026-09-26)
The phase advances as `dphi = dt * 2*pi * freq * v_cmd`, so a cycle takes `T = 1/(freq*v_cmd)`
and the implied step length is `v_cmd * T / 2 = 1/(2*freq)` — **independent of speed**. The
frequency therefore sets stride length directly.

`freq = 1.5` implied a **0.33 m stride** on a robot with 0.2 m legs: physically unreachable, so
`gait_phase_contact` was unsatisfiable no matter what the policy did. Set to **5.0** → 0.10 m.

This is the third independent defect that made the earlier `gait_phase_contact` attempts fail,
alongside inverted feet indices and the un-mirrored phase clock. All three are now fixed, which
is why re-enabling it is worth a real try.

### 28. `foot_separation` — NOT inert; min_sep raised (2026-09-26)
Resolved by `Diagnostics/foot_sep_body_m`: it starts at 0.07 and decays to **0.0433 m**, which
is exactly the URDF hip spacing (2 × 0.021672). So the policy drives `hip_roll` to zero and
parks at the neutral pose — and since 0.0433 < 0.07 the term *was* firing at −0.081/step all
along. The flat-zero reading in the failed run was an artefact of the broken-COM run, not of
the body-frame formula.

Note the policy is leaving free reward on the table: escaping to 0.07 needs only ~3.9° of
abduction, costing ~0.016/step against the 0.081/step saved. That it doesn't is a measure of
how deep the pose-regularization basin is — useful evidence in its own right.

`min_sep` raised 0.07 → **0.09** to roughly double the outward gradient.

Original note below.
The body-frame fix (item 8) is mathematically right but made the term read **exactly 0**: true
lateral separation is always ≥ `min_sep = 0.07`, so the clamp never opens. At `action = 0`,
`hip_roll` sits at the joint-limit midpoint of +0.175 rad, which on ~0.2 m legs already spreads
the feet to roughly 11 cm.

The old world-Y version fired only because it measured a heading-projected (and therefore
smaller) quantity — it was reading `true_sep × |cos(yaw)|`, i.e. partly an artefact.

`Diagnostics/foot_sep_body_m` now logs the unclamped body-frame separation in metres. Read it,
then set `min_sep` just above the observed mean so the term actually shapes stance. If the
observed value is already comfortably wide, the honest conclusion is that this term has no job
on V2 and should be set to 0 rather than left as a no-op.

Note it was NOT the cause of the 2026-09-25 regression — removing it *gained* ~85 points of
episode_score.

### 29. LESSON: check the baseline is current before arguing from arithmetic
I argued `upright: 0.5` created a "standing attractor" by computing that standing scores
~560/episode and comparing against a green baseline of **515**. That baseline was several runs
stale; the actual current baseline was **2368**. Standing was never competitive, the argument
was void, and acting on it cost 1384 points of episode_score and dropped episode_length
962 → 341.

Before reasoning from reward arithmetic, re-read the *current* `train/episode_score`.

---

## Settled — do not re-investigate

Recorded so these aren't re-litigated. All verified against the current tree.

| Claim | Verdict |
|---|---|
| Replay stores `(s', a, s')` | **False.** Dead keys in `common.py`; `ddpg.py:299-302` stores correct `(s, a, s')`. |
| `episode_sums` never reset | **False.** Top-level loop over all keys, `biped_env.py:777-779`. |
| `JOINT_LIMITS_V2` is a ±180° placeholder | **False.** Real asymmetric limits since commit `66737e4`. |
| Timeout bootstrapping conflates truncation/termination | **False.** Already correct. |
| Q-value divergence / deadly triad | **False.** `critic_mean_q` 71.32 vs `mean_target` 71.32. |
| SAC alpha collapsing to zero | **False.** Settled at 0.0556, flat. |
| tanh saturation corrupting entropy control | **False.** `log_prob` ~= 1.0/dim vs a 13.8/dim cap. |
| `last_steps` dead code changes update cadence | **True but inert.** `steps` advances by `num_workers` per iteration, so the gate passes regardless. |
| Huber loss needed for the critic | **No.** Critic is well-calibrated; MSE is fine. |
