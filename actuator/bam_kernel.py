"""
actuator/bam_kernel.py -- the Feetech STS3215 servo model, batched in PyTorch.

WHAT THIS IS
------------
A port of Rhoban BAM's voltage-level servo model. The firmware P loop, the DC motor
with back-EMF, and the M1..M6 friction budget all run here, vectorised over
(num_envs, num_joints). Isaac (envs/assets/robotV2/bam_actuator.py), the MuJoCo eval
env and the parity tests all share this one implementation, so they cannot drift
apart.

It ports BAM at commit e9a619d of github.com/Rhoban/bam:
    model.py            Model.compute_frictions      -> BamKernel.friction_budget
    feetech/actuator.py STS3215Actuator.compute_ctrl -> BamKernel.motor
    actuator.py         VoltageControlledActuator    -> BamKernel.motor (torque eqn)
    simulate.py         Simulator.step (stop clip)   -> BamKernel.stop_clip
    simulate.py         fractional_delay_shift       -> causal ring in BamKernel.motor
BAM is Copyright 2025 Marc Duclusaud & Gregoire Passault, Apache-2.0 (see
actuator/params/LICENSE-BAM). This file is a modified, re-implemented derivative.

WHICH BAM IT FOLLOWS, AND WHY
-----------------------------
BAM has three non-identical references: simulate.py, mujoco.py and mjlab.py. This
file follows simulate.py and model.py, because those are what the parameters were
FITTED against:
  * the friction budget includes viscous: min(|tau_stop|, frictionloss + c*|dq|);
  * the M6 quadratic term keeps model.py's sign gate (mjlab.py drops it);
  * friction uses the CURRENT step's motor torque, not the previous one.
actuator/parity/test_kernel_vs_bam.py checks it against BAM's own Simulator to
float64 rounding.

WHAT THE CALLER SUPPLIES
------------------------
The kernel knows the servo and nothing about the rigid body attached to it.
Each step the caller passes two things:
  tau_ext      every torque on the joint except this servo's motor and friction
               (gravity, contacts, coupling). The pendulum test computes it exactly;
               MuJoCo reads it from qfrc_bias/qfrc_constraint; Isaac estimates it
               with a lagged dynamics observer.
  inertia_eff  the joint's effective inertia, INCLUDING armature. Only the stiction
               clip uses it, to work out the torque that would stop the joint
               within one step.
The returned torque is motor + friction. The simulator applies it as a joint effort
and integrates everything else itself.

DELIBERATE DEPARTURES FROM BAM
------------------------------
  * q_offset is ignored. It is a testbench calibration term ("no meaning on a real
    robot"), and the 12 V fit pins it at its +-0.1 bound anyway.
  * command_delay is applied through a causal ring buffer instead of shifting the
    whole goal sequence ahead of time. The two give identical results, because both
    hold the first sample before enough history exists.
  * friction_scale and torque_scale are per-env/per-joint multipliers for domain
    randomisation and the servo's TORQUE_LIMIT register. At 1.0 they are exact no-ops.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch

# --- STS3215 firmware constants (bam/feetech/actuator.py) ---------------------------
# error_gain turns kp * (position error in rad) into a PWM duty cycle. BAM measured it
# on an oscilloscope. Waveshare's ST3025 reuses it verbatim, so it is family-wide.
ERROR_GAIN = 0.166
# Largest duty cycle the firmware will output, also oscilloscope-measured.
MAX_PWM = 0.97

# Model variant -> which friction terms exist (bam/model.py `models`).
_MODEL_FLAGS = {
    #       load_dependent directional stribeck quadratic
    "m1": (False, False, False, False),
    "m2": (False, False, True, False),
    "m3": (True, False, False, False),
    "m4": (True, False, True, False),
    "m5": (True, True, True, False),
    "m6": (True, True, True, True),
}


@dataclass(frozen=True)
class BamParams:
    """One BAM params JSON, as written by ``bam.fit --output``.

    ``from_json`` rejects unknown keys and missing keys the model needs. A misspelt
    key would otherwise silently leave a friction term at zero.
    """
    model: str
    kt: float
    R: float
    error_gain_ratio: float
    armature: float
    max_velocity: float
    command_delay: float
    friction_base: float
    friction_viscous: float
    # Stribeck
    friction_stribeck: float = 0.0
    dtheta_stribeck: float = 1.0
    alpha: float = 1.0
    # Load-dependent, non-directional (m3, m4)
    load_friction_base: float = 0.0
    load_friction_stribeck: float = 0.0
    # Load-dependent, directional (m5, m6)
    load_friction_motor: float = 0.0
    load_friction_external: float = 0.0
    load_friction_motor_stribeck: float = 0.0
    load_friction_external_stribeck: float = 0.0
    # Quadratic (m6)
    load_friction_motor_quad: float = 0.0
    load_friction_external_quad: float = 0.0
    # Carried for provenance, never used in the maths.
    q_offset: float = 0.0
    actuator: str = ""
    source: str = ""

    @property
    def flags(self) -> tuple[bool, bool, bool, bool]:
        return _MODEL_FLAGS[self.model]

    @property
    def stall_torque(self) -> float:
        """Duty-limited stall torque at 12 V [N*m], useful for sanity-checking a fit."""
        return MAX_PWM * 12.0 * self.kt / self.R

    @classmethod
    def from_json(cls, path: str | Path) -> "BamParams":
        path = Path(path)
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown BAM parameter(s) {sorted(unknown)}")
        model = data.get("model")
        if model not in _MODEL_FLAGS:
            raise ValueError(f"{path}: 'model' must be one of {sorted(_MODEL_FLAGS)}, got {model!r}")

        load_dep, directional, stribeck, quadratic = _MODEL_FLAGS[model]
        required = {"kt", "R", "error_gain_ratio", "armature", "max_velocity",
                    "command_delay", "friction_base", "friction_viscous"}
        if stribeck:
            required |= {"friction_stribeck", "dtheta_stribeck", "alpha"}
        if load_dep and not directional:
            required |= {"load_friction_base"} | ({"load_friction_stribeck"} if stribeck else set())
        if load_dep and directional:
            required |= {"load_friction_motor", "load_friction_external"}
            if stribeck:
                required |= {"load_friction_motor_stribeck", "load_friction_external_stribeck"}
        if quadratic:
            required |= {"load_friction_motor_quad", "load_friction_external_quad"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path}: model {model} needs {sorted(missing)}")

        return cls(**data, source=str(path))


def _full(shape, value, device, dtype) -> torch.Tensor:
    t = torch.as_tensor(value, device=device, dtype=dtype)
    return t.expand(shape).clone() if t.dim() else torch.full(shape, float(t), device=device, dtype=dtype)


class BamKernel:
    """Batched STS3215 model. All state tensors are ``(num_envs, num_joints)``.

    Call order, exactly once per physics step:
        tau_motor = kernel.motor(q_goal, q_enc, dq)              # stateful
        tau_fric  = kernel.stop_clip(tau_motor, tau_ext, dq, inertia_eff)
        apply tau_motor + tau_fric as the joint effort
    or use ``step()``, which does both. ``motor`` is stateful (the firmware slew limiter
    and the command-delay ring), so it must not be called twice in one physics step.

    Per-env knobs, writable at any time (``tensor[env_ids] = ...``):
        kp             firmware P register (32 on this robot)
        vin            supply voltage [V], shape (N, 1)
        friction_scale multiplies the whole friction budget (DR)
        torque_scale   TORQUE_LIMIT/1000: scales the duty ceiling
    """

    def __init__(
        self,
        params: BamParams,
        num_envs: int,
        num_joints: int,
        dt: float,
        kp: float | torch.Tensor = 32.0,
        vin: float = 12.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.p = params
        self.dt = float(dt)
        self.num_envs, self.num_joints = num_envs, num_joints
        self.device, self.dtype = torch.device(device), dtype
        shape = (num_envs, num_joints)

        self.kp = _full(shape, kp, self.device, dtype)
        self.vin = _full((num_envs, 1), vin, self.device, dtype)
        self.friction_scale = torch.ones(shape, device=self.device, dtype=dtype)
        self.torque_scale = torch.ones(shape, device=self.device, dtype=dtype)

        self.q_smooth = torch.zeros(shape, device=self.device, dtype=dtype)
        self.duty = torch.zeros(shape, device=self.device, dtype=dtype)
        self.frictionloss = torch.zeros(shape, device=self.device, dtype=dtype)

        # Command delay as a fractional shift: (1-f)*goal[k-d0] + f*goal[k-d0-1].
        d = self.p.command_delay / self.dt
        self._d0 = int(math.floor(d))
        self._frac = d - self._d0
        self._ring_len = self._d0 + 2
        self._ring = torch.zeros((self._ring_len, *shape), device=self.device, dtype=dtype)
        self._head = 0

        # Every env starts "pending": its slew state and delay history are seeded from
        # the first motor() call, the way the servo behaves when it powers on.
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------------------
    def reset(self, env_ids=None) -> None:
        """Defer a state reset to the next ``motor()`` call.

        The reset seeds from the q and goal passed to that call. Like BAM, the reset has
        to be deferred because the post-reset joint position isn't known yet.
        """
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self._pending[:] = True
        else:
            self._pending[env_ids] = True

    # ------------------------------------------------------------------------------
    def motor(self, q_goal: torch.Tensor, q: torch.Tensor, dq: torch.Tensor,
              torque_enable: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Firmware + DC motor: goal and encoder state -> motor torque [N*m].

        ``q`` is what the servo's own encoder reads. When backlash is modelled that is
        the output-side angle; the caller handles that distinction.
        """
        p = self.p
        if self._pending.any():
            idx = self._pending
            self.q_smooth[idx] = q[idx]
            self._ring[:, idx] = q_goal[idx]
            self._pending[idx] = False

        # Command delay (bam/simulate.py fractional_delay_shift, causal form).
        self._head = (self._head + 1) % self._ring_len
        self._ring[self._head] = q_goal
        if self._d0 == 0 and self._frac == 0.0:
            goal = q_goal
        else:
            i0 = (self._head - self._d0) % self._ring_len
            i1 = (self._head - self._d0 - 1) % self._ring_len
            goal = (1.0 - self._frac) * self._ring[i0] + self._frac * self._ring[i1]

        # Firmware slew limiter: the internal target moves at most max_velocity*dt.
        max_step = p.max_velocity * self.dt
        self.q_smooth = torch.clamp(goal, self.q_smooth - max_step, self.q_smooth + max_step)

        # P-only position loop to PWM duty. Same operand order as BAM, for float parity.
        duty = (self.q_smooth - q) * self.kp * ERROR_GAIN * p.error_gain_ratio
        lim = MAX_PWM * self.torque_scale
        self.duty = torch.clamp(duty, -lim, lim)
        volts = self.vin * self.duty

        # DC motor with back-EMF: tau = kt*V/R - kt^2*dq/R
        torque = p.kt * volts / p.R
        torque = torque - (p.kt ** 2) * dq / p.R
        return torque * torque_enable

    # ------------------------------------------------------------------------------
    def friction_budget(self, motor_torque: torch.Tensor, external_torque: torch.Tensor,
                        dtheta: torch.Tensor) -> tuple[torch.Tensor, float]:
        """bam/model.py Model.compute_frictions, line for line. Returns (frictionloss, viscous)."""
        p = self.p
        load_dependent, directional, stribeck, quadratic = p.flags

        if directional:
            gearbox_torque = torch.abs(external_torque * p.load_friction_external
                                       - motor_torque * p.load_friction_motor)
            if stribeck:
                gearbox_torque_stribeck = torch.abs(
                    external_torque * p.load_friction_external_stribeck
                    - motor_torque * p.load_friction_motor_stribeck)
        else:
            gearbox_torque = torch.abs(external_torque - motor_torque)

        if stribeck:
            stribeck_coeff = torch.exp(-(torch.abs(dtheta / p.dtheta_stribeck) ** p.alpha))

        frictionloss = torch.full_like(dtheta, p.friction_base)
        if load_dependent:
            if directional:
                frictionloss = frictionloss + gearbox_torque
            else:
                frictionloss = frictionloss + p.load_friction_base * gearbox_torque

        if stribeck:
            frictionloss = frictionloss + stribeck_coeff * p.friction_stribeck
            if load_dependent:
                if directional:
                    frictionloss = frictionloss + gearbox_torque_stribeck * stribeck_coeff
                else:
                    frictionloss = frictionloss + (p.load_friction_stribeck * gearbox_torque
                                                   * stribeck_coeff)
                if quadratic:
                    enable_quadratic = (torch.sign(external_torque) != torch.sign(motor_torque))
                    direction_motor = torch.abs(external_torque) < torch.abs(motor_torque)
                    direction_external = torch.abs(external_torque) > torch.abs(motor_torque)
                    gearbox_torque2_motor = p.load_friction_external_quad * torch.abs(external_torque) ** 2
                    gearbox_torque2_external = p.load_friction_motor_quad * torch.abs(motor_torque) ** 2
                    frictionloss = frictionloss + (
                        stribeck_coeff
                        * (direction_motor.to(dtheta.dtype) * gearbox_torque2_motor
                           + direction_external.to(dtheta.dtype) * gearbox_torque2_external)
                        * enable_quadratic.to(dtheta.dtype)
                    )

        return frictionloss, p.friction_viscous

    # ------------------------------------------------------------------------------
    def stop_clip(self, motor_torque: torch.Tensor, external_torque: torch.Tensor,
                  dq: torch.Tensor, inertia_eff: torch.Tensor | float) -> torch.Tensor:
        """Friction torque to apply (bam/simulate.py Simulator.step).

        tau_stop is the torque that would bring the joint to rest within one step.
        Friction opposes it and is capped at the budget, so it can stop the joint but
        never reverse it. That cap is what produces genuine stiction, and it can never
        add energy.
        """
        frictionloss, damping = self.friction_budget(motor_torque, external_torque, dq)
        self.frictionloss = frictionloss
        budget = (frictionloss + damping * torch.abs(dq)) * self.friction_scale
        tau_stop = (inertia_eff / self.dt) * dq + (motor_torque + external_torque)
        return -torch.sign(tau_stop) * torch.minimum(torch.abs(tau_stop), budget)

    # ------------------------------------------------------------------------------
    def step(self, q_goal, q_enc, dq, external_torque, inertia_eff, torque_enable=1.0):
        """motor() + stop_clip(). Returns (total actuator torque, motor torque, friction torque)."""
        tau_m = self.motor(q_goal, q_enc, dq, torque_enable)
        tau_f = self.stop_clip(tau_m, external_torque, dq, inertia_eff)
        return tau_m + tau_f, tau_m, tau_f


class LostMotion:
    """Gear backlash for single-DOF simulators (Isaac): the rotor travels through a dead zone.

    WHY NOT THE FAITHFUL MODEL
    --------------------------
    MuJoCo gets the faithful model: a passive hinge in series with each servo, with stiff
    limits (actuator/backlash.py). Isaac can't use either faithful option:
      * an extra passive DOF per joint would double the DOF count and re-index every
        joint-indexed term in the env;
      * a stiff spring coupling the rotor to the output link is explicit, and the output
        links are ~1e-4 kg*m^2, so at 200 Hz it would be violently unstable.
    So the rotor is tracked here as a virtual angle and velocity, theta_m and omega_m,
    and the output stays the PhysX DOF:
      * free play (rotor inside the gap): no servo torque reaches the output, and the
        rotor accelerates on its own, J = armature, under motor torque and gearbox
        friction;
      * contact (rotor at an edge, pushing into it): rotor and output move together and
        the full motor torque is transmitted;
      * at the moment of contact the rotor's excess momentum, J*(omega_m - dq)/dt, goes
        to the output as an impulse torque, capped at `impulse_cap` for stability.
    The encoder is on the output side, so the firmware never sees the rotor winding
    through the gap. This is the effect that matters for a 50 Hz position policy: a
    reversal costs a torque-free dead time, followed by a knock.

    KNOWN APPROXIMATION
    -------------------
    PhysX keeps the armature on the output DOF permanently, so during free play the
    output carries rotor inertia that it physically wouldn't. That makes the output
    sluggish where it should wander. The MuJoCo-vs-Isaac pendulum parity run with lash
    on is what measures this gap.

    With lash = 0 the gap is pinned shut, every step is contact, and the transmitted
    torque is exactly the motor torque. That makes it an exact no-op, which is tested.
    """

    def __init__(self, num_envs: int, num_joints: int, device, dtype,
                 impulse_cap: float = 6.0):
        shape = (num_envs, num_joints)
        self.lash = torch.zeros(shape, device=device, dtype=dtype)   # total play [rad]
        self.theta_m = torch.zeros(shape, device=device, dtype=dtype)
        self.omega_m = torch.zeros(shape, device=device, dtype=dtype)
        self.contact = torch.ones(shape, dtype=torch.bool, device=device)
        self.side = torch.ones(shape, device=device, dtype=dtype)   # +1 upper edge, -1 lower
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.impulse_cap = impulse_cap

    def reset(self, env_ids=None) -> None:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self._pending[:] = True
        else:
            self._pending[env_ids] = True

    def emf_velocity(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Rotor velocity, which is what generates back-EMF. Call before motor()."""
        if self._pending.any():
            idx = self._pending
            self.theta_m[idx] = q[idx]
            self.omega_m[idx] = dq[idx]
            self.contact[idx] = True
            self.side[idx] = 1.0
            self._pending[idx] = False
        # In contact the rotor rides on the output, so its velocity is the output's
        # CURRENT velocity. omega_m holds last step's value, and using it there would
        # put a one-step lag on back-EMF.
        return torch.where(self.contact, dq, self.omega_m)

    def transmit(self, tau_motor: torch.Tensor, rotor_friction: torch.Tensor,
                 q: torch.Tensor, dq: torch.Tensor, armature: float, dt: float) -> torch.Tensor:
        """Advance the rotor one step and return the servo torque reaching the output.

        rotor_friction is the gearbox friction the rotor sees in free play. In contact,
        friction acts on the output instead and is applied by the caller.
        """
        half = 0.5 * self.lash
        zero = torch.zeros_like(tau_motor)
        rigid = self.lash <= 0.0          # no lash: pinned shut, an exact pass-through

        # A contact holds while the motor still pushes into the edge it is against. The
        # contact force then stays positive. Gearbox friction belongs to the combined
        # rotor+output body; the caller applies it on the output side. It must NOT be
        # folded into the rotor here: below the friction budget that would "stick" the
        # rotor while the output coasts away, breaking contact every few steps.
        holds = self.contact & (self.side * tau_motor > 0.0)

        # Everything not held is free: rotor and output move independently. Envs that are
        # separating this step start from the contact state they are leaving.
        omega0 = torch.where(self.contact, dq, self.omega_m)
        theta0 = torch.where(self.contact, q + self.side * half, self.theta_m)
        omega_free = omega0 + ((tau_motor + rotor_friction) / armature) * dt
        gap = (theta0 + omega_free * dt) - q
        hit = (~holds) & ((gap >= half) | (gap <= -half))
        hit_side = torch.where(gap >= half, torch.ones_like(gap), -torch.ones_like(gap))

        contact = holds | hit | rigid
        side = torch.where(holds | rigid, self.side, hit_side)
        impact = hit & ~self.contact & ~rigid
        impulse = torch.clamp(armature * (omega_free - dq) / dt, -self.impulse_cap, self.impulse_cap)

        tau_out = torch.where(contact, tau_motor, zero) + torch.where(impact, impulse, zero)
        self.theta_m = torch.where(contact, q + side * half, theta0 + omega_free * dt)
        self.omega_m = torch.where(contact, dq, omega_free)
        self.contact, self.side = contact, side
        return tau_out


class BamServo:
    """BamKernel plus optional LostMotion, composed the same way for every simulator.

    The simulator supplies tau_ext and inertia_eff (see the module docstring), and
    applies the returned torque as the joint effort. Nothing here knows which
    simulator it is running in.
    """

    def __init__(self, params: BamParams, num_envs: int, num_joints: int, dt: float,
                 kp=32.0, vin: float = 12.0, backlash: bool = False,
                 impulse_cap: float = 6.0, device="cpu", dtype=torch.float32):
        self.kernel = BamKernel(params, num_envs, num_joints, dt, kp=kp, vin=vin,
                                device=device, dtype=dtype)
        self.lost_motion = (LostMotion(num_envs, num_joints, device, dtype, impulse_cap)
                            if backlash else None)
        self.armature = params.armature
        self.dt = float(dt)
        shape = (num_envs, num_joints)
        self.tau_motor = torch.zeros(shape, device=device, dtype=dtype)
        self.tau_friction = torch.zeros(shape, device=device, dtype=dtype)
        self.tau_total = torch.zeros(shape, device=device, dtype=dtype)

    def reset(self, env_ids=None) -> None:
        self.kernel.reset(env_ids)
        if self.lost_motion is not None:
            self.lost_motion.reset(env_ids)

    def step(self, q_goal, q_enc, dq, external_torque, inertia_eff, torque_enable=1.0):
        """One physics step. Returns the torque to apply to the joint [N*m]."""
        k, lm = self.kernel, self.lost_motion
        if lm is None:
            self.tau_motor = k.motor(q_goal, q_enc, dq, torque_enable)
            self.tau_friction = k.stop_clip(self.tau_motor, external_torque, dq, inertia_eff)
            self.tau_total = self.tau_motor + self.tau_friction
            return self.tau_total

        emf_vel = lm.emf_velocity(q_enc, dq)
        self.tau_motor = k.motor(q_goal, q_enc, emf_vel, torque_enable)
        # Gearbox friction on the rotor alone, in case it is in free play this step.
        rotor_friction = k.stop_clip(self.tau_motor, torch.zeros_like(dq), lm.omega_m, self.armature)
        tau_tx = lm.transmit(self.tau_motor, rotor_friction, q_enc, dq, self.armature, self.dt)
        out_friction = k.stop_clip(tau_tx, external_torque, dq, inertia_eff)
        self.tau_friction = torch.where(lm.contact, out_friction, torch.zeros_like(out_friction))
        self.tau_total = tau_tx + self.tau_friction
        return self.tau_total

    @property
    def duty(self) -> torch.Tensor:
        return self.kernel.duty

    @property
    def duty_saturated(self) -> torch.Tensor:
        return self.kernel.duty.abs() >= (MAX_PWM * self.kernel.torque_scale) * (1.0 - 1e-6)


def apply_stall_cap(params: BamParams, stall_torque_cap_nm: float | None) -> BamParams:
    """Return params whose duty-limited stall torque is capped, WITHOUT slowing the servo.

    The motor line is  tau = kt*V/R - kt^2*dq/R,  so
        stall torque  = kt*V/R          (falls as R rises)
        no-load speed = V/kt            (independent of R)
    Raising R therefore caps the torque and leaves the speed alone, which is what a real
    higher-resistance motor does.

    Do NOT cap by scaling the duty ceiling (torque_scale) instead: that scales V, which
    drops stall torque and no-load speed TOGETHER. That was the 2026-10-03 bug. On the m6
    fit, capping 6.33 -> 2.94 N*m by duty also cut no-load speed 5.15 -> 2.39 rad/s, while
    the real STS3215 datasheet quotes 2.94 N*m AND 5.82 rad/s at the same time. A knee
    needs ~2 rad/s unloaded to swing a leg at gait cadence, so the sim robot could no
    longer step: foot lift sat at ~6 mm across four reward configurations and
    Torque/frac_saturated ran 0.55-0.60 against the working run's 0.39.

    torque_scale stays what it says it is: the servos' TORQUE_LIMIT register.
    """
    if stall_torque_cap_nm is None or stall_torque_cap_nm >= params.stall_torque:
        return params
    return replace(params, R=params.R * params.stall_torque / float(stall_torque_cap_nm))


def load_params(path: str | Path) -> BamParams:
    """Load a params JSON, resolving paths relative to the BipedRobot repo root if needed."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent.parent / path
    return BamParams.from_json(path)
