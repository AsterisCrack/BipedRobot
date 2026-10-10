"""
Isaac Lab actuator for the Feetech STS3215, built on the BAM servo model.

Replaces DCMotorCfg. The servo maths lives in actuator/bam_kernel.py, shared with the
MuJoCo eval env and the parity tests. This file only adapts it to Isaac Lab:

  * Isaac calls compute() once per 200 Hz physics substep. That matches the dt BAM's
    parameters were fitted at (bam.process resamples to 0.005 s).
  * Isaac gives an actuator neither dt nor the articulation, so dt is read from the
    SimulationContext and the env injects the articulation with attach().
  * External torque. BAM's MuJoCo backend reads qfrc_bias + qfrc_constraint, which
    PhysX doesn't expose per joint. Here it comes from a one-substep-lagged dynamics
    observer:
        tau_ext[t] = I_eff * qdd[t-1] - tau_applied[t-1]
    with qdd from data.joint_acc, the finite difference over the previous substep. The
    estimate lumps gravity, contacts and coupling from the other joints together, which
    is exactly what a single-DOF friction model wants to know. bam/mujoco.py works from
    the previous step's solve too, so the lag is the same.
    tau_ext_source="gravity" uses PhysX gravity compensation instead, for debugging.
  * Effective inertia for the stiction clip is 1 / (M^-1)_ii, NOT the diagonal M_ii.
    The clip is a one-step deadbeat on joint velocity: it computes the torque that stops
    the joint within one substep. M_ii is the inertia with every OTHER dof locked. On a
    floating-base robot, a torque on one joint also moves the base and the other links,
    so the inertia it actually sees is 1/(M^-1)_ii, which is always <= M_ii. Using M_ii
    over-corrects every substep, and that is chatter: standing robots showed a residual
    |I*qdd| of ~0.08 N*m. BAM's own MuJoCo code reads dof_invweight0 for the same reason.
    PhysX's matrix EXCLUDES armature (probed once on the first compute()), so armature is
    added to the diagonal before inverting. The 18x18 batched inverse runs every substep
    and is cheap next to the physics.

Without attach() (play/export scripts that never call it), it falls back to
inertia = armature and tau_ext = 0, and warns once.

Friction: the kernel owns it entirely, so PhysX joint friction is forced to 0 here.
Otherwise it would be counted twice.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from actuator.bam_kernel import BamServo, apply_stall_cap, load_params

logger = logging.getLogger(__name__)


class BamActuator(ActuatorBase):
    """Explicit actuator: position target in, BAM servo torque out."""

    cfg: "BamActuatorCfg"

    def __init__(self, cfg: "BamActuatorCfg", *args, **kwargs):
        self.params = apply_stall_cap(load_params(cfg.params_file), cfg.stall_torque_cap_nm)
        # The kernel owns friction, and armature comes from the identified params unless
        # the cfg overrides it. Both are written to PhysX by the articulation from these
        # actuator values.
        cfg.friction = 0.0
        cfg.dynamic_friction = 0.0
        cfg.viscous_friction = 0.0
        if cfg.armature is None:
            cfg.armature = self.params.armature
        super().__init__(cfg, *args, **kwargs)

        import isaaclab.sim as sim_utils
        dt = sim_utils.SimulationContext.instance().get_physics_dt()
        self.servo = BamServo(
            self.params, self._num_envs, self.num_joints, dt,
            kp=cfg.kp_fw, vin=sum(cfg.vin_range) / 2.0,
            backlash=cfg.backlash_enabled, impulse_cap=cfg.backlash_impulse_cap,
            device=self._device,
        )
        self.servo.kernel.torque_scale[:] = cfg.torque_scale
        if cfg.backlash_enabled:
            self.servo.lost_motion.lash[:] = cfg.backlash_nominal_rad

        self._articulation = None
        self._dof_offset = 0
        self._armature_in_mass_matrix: bool | None = None
        self._warned_detached = False
        self.tau_ext = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.inertia_eff = self.armature.clone()
        # Per-joint torque enable (1 = powered). The firmware slew state keeps advancing while
        # unpowered, exactly as in BAM. Used by the pendulum parity test's lift_and_drop, and
        # to simulate a limp or faulted servo.
        self.torque_enable = torch.ones(self._num_envs, self.num_joints, device=self._device)
        logger.info("BamActuator: %s (%s), dt=%.4f, kp=%s, stall@12V=%.2f N*m, backlash=%s",
                    self.params.source, self.params.model, dt, cfg.kp_fw,
                    self.params.stall_torque, cfg.backlash_enabled)

    # ------------------------------------------------------------------------------
    def attach(self, articulation) -> None:
        """Give the actuator its articulation, for the mass matrix and joint_acc."""
        self._articulation = articulation
        self._dof_offset = 0 if articulation.is_fixed_base else 6

    def _joint_ids(self):
        ids = self.joint_indices
        return slice(None) if isinstance(ids, slice) else ids

    def _mass_matrix(self) -> torch.Tensor:
        return self._articulation.root_physx_view.get_generalized_mass_matrices()

    def _mass_diag(self) -> torch.Tensor:
        diag = torch.diagonal(self._mass_matrix(), dim1=-2, dim2=-1)[:, self._dof_offset:]
        return diag[:, self._joint_ids()]

    def _effective_inertia(self) -> torch.Tensor:
        """1 / (M^-1)_ii per actuated joint, armature included (see module docstring)."""
        M = self._mass_matrix()
        if not self._armature_in_mass_matrix:
            arm = torch.zeros(M.shape[:2], device=M.device, dtype=M.dtype)
            arm[:, self._dof_offset:] = self._articulation.data.joint_armature.to(M.dtype)
            M = M + torch.diag_embed(arm)
        inv_diag = torch.diagonal(torch.linalg.inv(M), dim1=-2, dim2=-1)[:, self._dof_offset:]
        return 1.0 / inv_diag[:, self._joint_ids()]

    def _sim_armature(self) -> torch.Tensor:
        return self._articulation.data.joint_armature[:, self._joint_ids()]

    def _probe_armature(self) -> None:
        """Once: does PhysX's mass-matrix diagonal already contain armature?

        Decided from magnitudes, with no writes to the sim. Writing armature and reading
        the matrix back in the same substep may not refresh it, and that would produce a
        confident wrong answer. BAM's armature (~0.023 kg*m^2) exceeds every link inertia
        about its own joint axis on this robot (the hip, carrying the whole leg, is about
        0.005). So if every diagonal entry is at least ~armature, the armature is in it.
        """
        diag = self._mass_diag()
        # Per-env PhysX values: randomize_joint_params scales armature in the sim but
        # never updates the actuator's own copy (self.armature).
        arm = self._sim_armature()
        ratio = (diag / arm).min().item()
        # Included -> ratio >= 1. Excluded -> ratio <= ~0.25. Split the gap.
        self._armature_in_mass_matrix = ratio >= 0.6
        per_joint = ", ".join(f"{n}={v:.4f}" for n, v in zip(self.joint_names, diag[0].tolist()))
        logger.warning("BamActuator: mass-matrix diagonal %s armature (min diag/armature = %.2f). "
                       "env0 diag: %s", "INCLUDES" if self._armature_in_mass_matrix else "EXCLUDES",
                       ratio, per_joint)

    # ------------------------------------------------------------------------------
    def reset(self, env_ids: Sequence[int]):
        self.servo.reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        # Zero the observer's memory too: joint_acc is zeroed by the state write, and
        # subtracting a pre-reset torque would hand the first post-reset step a bogus tau_ext.
        self.tau_ext[idx] = 0.0
        self.servo.tau_total[idx] = 0.0

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor,
                joint_vel: torch.Tensor) -> ArticulationActions:
        if self._articulation is not None:
            if self._armature_in_mass_matrix is None:
                self._probe_armature()
            self.inertia_eff = self._effective_inertia()
            if self.cfg.tau_ext_source == "gravity":
                g = self._articulation.root_physx_view.get_gravity_compensation_forces()
                self.tau_ext = -g[:, self._dof_offset:][:, self._joint_ids()]
            else:
                qdd = self._articulation.data.joint_acc[:, self._joint_ids()]
                self.tau_ext = self.inertia_eff * qdd - self.servo.tau_total
        elif not self._warned_detached:
            logger.warning("BamActuator: no articulation attached; using inertia=armature, tau_ext=0")
            self._warned_detached = True

        effort = self.servo.step(control_action.joint_positions, joint_pos, joint_vel,
                                 self.tau_ext, self.inertia_eff, self.torque_enable)
        if control_action.joint_efforts is not None:
            effort = effort + control_action.joint_efforts

        self.computed_effort = effort
        self.applied_effort = effort
        control_action.joint_efforts = effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    @property
    def duty_saturated(self) -> torch.Tensor:
        return self.servo.duty_saturated


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """STS3215 via BAM. To switch to a new identification, change params_file; nothing else."""

    class_type: type = BamActuator

    # Explicit actuator: PhysX drive gains must be zero (the kernel is the controller).
    stiffness: float = 0.0
    damping: float = 0.0

    params_file: str = MISSING
    """BAM params JSON, as written by ``bam.fit --output``. Relative to the BipedRobot repo root."""

    kp_fw: float = 32.0
    """Firmware P register value. Must match config/robot.yaml on the Jetson."""

    vin_range: tuple[float, float] = (11.1, 12.6)
    """Supply voltage range [V], drawn per env at startup. ASSUMED 3S pack: set to the real battery."""

    friction_scale_range: tuple[float, float] = (0.9, 1.1)
    """Per-env, per-joint multiplier on the whole friction budget, drawn at startup."""

    torque_scale: float = 1.0
    """TORQUE_LIMIT/1000 on the real servos; scales the duty ceiling. Audit with tools/servo_registers.py."""

    stall_torque_cap_nm: float | None = None
    """Cap the duty-limited stall torque at this value [N*m], e.g. the datasheet's 2.94 at 12 V.
    The placeholder fit implies ~6.3 N*m, about 2x the datasheet, and a policy trained on it
    learned a forward lean held by torque the real servo can't sustain. None = trust the
    params. Kept separate from torque_scale, which is the real servos' TORQUE_LIMIT.

    Applied by raising R (see apply_stall_cap), so no-load speed is preserved. It is NOT a
    duty-ceiling scale -- that would slow the servo as well as weaken it."""

    tau_ext_source: str = "observer"
    """'observer' (default, lagged dynamics observer) or 'gravity' (PhysX gravity only, for debugging)."""

    backlash_enabled: bool = False
    backlash_nominal_rad: float = 0.0
    """Total lash until randomize_bam_actuator draws per-joint values."""
    backlash_impulse_cap: float = 6.0
    """Cap on the re-engagement impulse torque [N*m], roughly the 12 V stall torque."""
