"""
actuator/parity/test_lost_motion.py -- behaviour checks for the Isaac backlash model.

There is no BAM reference to compare LostMotion against: BAM itself has no backlash.
These tests pin down the properties it must have instead:

  1. lash = 0 is an EXACT no-op. A BamServo with backlash on and zero lash reproduces
     the backlash-free servo bit-for-bit on a full pendulum rollout. This guards the
     no-backlash path, which is the one test_kernel_vs_bam.py validates.
  2. On a direction reversal the output receives NO servo torque while the rotor
     crosses the gap, and the length of that dead time scales with the lash.
  3. It stays bounded: no NaNs, and the joint still reaches its target.

Run: python -m actuator.parity.test_lost_motion   (torch only, no bam needed)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from actuator.bam_kernel import BamServo, load_params       # noqa: E402

PARAMS = _REPO / "actuator" / "params" / "sts3215_12v" / "m6.json"
DT = 0.005
MASS, LENGTH, G = 0.25, 0.10, -9.80665


def _rollout(servo: BamServo, goals: list[float], gravity: bool = True):
    """Pendulum, same explicit-Euler integrator as bam/simulate.py. Returns per-step traces."""
    p = servo.kernel.p
    inertia = MASS * LENGTH ** 2 + p.armature
    q = torch.zeros(1, 1, dtype=torch.float64)
    dq = torch.zeros(1, 1, dtype=torch.float64)
    qs, taus, contacts = [], [], []
    for g in goals:
        goal = torch.full((1, 1), g, dtype=torch.float64)
        bias = (MASS * G * LENGTH * torch.sin(q)) if gravity else torch.zeros_like(q)
        tau = servo.step(goal, q, dq, bias, inertia)
        dq = torch.clamp(dq + ((tau + bias) / inertia) * DT, -100.0, 100.0)
        q = q + dq * DT
        qs.append(float(q))
        taus.append(float(tau))
        lm = servo.lost_motion
        contacts.append(bool(lm.contact[0, 0]) if lm is not None else True)
    return qs, taus, contacts


def _servo(backlash: bool, lash_deg: float = 0.0) -> BamServo:
    s = BamServo(load_params(PARAMS), 1, 1, DT, kp=32.0, vin=12.0, backlash=backlash,
                 dtype=torch.float64)
    if s.lost_motion is not None:
        s.lost_motion.lash[:] = math.radians(lash_deg)
    return s


def _sin_time_square() -> list[float]:
    return [math.sin((k * DT) ** 2) for k in range(1200)]


def test_zero_lash_is_exact_noop():
    goals = _sin_time_square()
    q_ref, tau_ref, _ = _rollout(_servo(False), goals)
    q_lm, tau_lm, _ = _rollout(_servo(True, 0.0), goals)
    assert q_ref == q_lm, "lash=0 changed the position trace"
    assert tau_ref == tau_lm, "lash=0 changed the torque trace"


def _dead_time_after_reversal(lash_deg: float) -> int:
    """Ramp up slowly, then step the goal down. Count steps where the output got no servo torque.

    The slow ramp matters. After a static hold, the rotor comes to rest somewhere
    arbitrary inside the gap, so the dead time would depend on that position rather
    than on the lash. While ramping, gearbox friction keeps the motor pushing, which
    holds the rotor against the upper edge. A reversal from there has to cross the
    full gap.
    """
    goals = [0.001 * k for k in range(200)] + [-0.5] * 200
    _, _, contacts = _rollout(_servo(True, lash_deg), goals, gravity=False)
    assert contacts[199], "rotor was not in contact when the reversal started"
    # The servo's command delay (~5.6 ms) keeps contact held for a step or so after the
    # goal flips. Count the free-play run from the moment contact is actually lost.
    after = contacts[200:]
    start = next((i for i, c in enumerate(after) if not c), None)
    if start is None:
        return 0
    return next((i for i, c in enumerate(after[start:]) if c), len(after) - start)


def test_reversal_has_dead_time_growing_with_lash():
    d1, d3 = _dead_time_after_reversal(1.0), _dead_time_after_reversal(3.0)
    assert d1 >= 1, f"no free-play dead time with 1 deg lash (got {d1} steps)"
    assert d3 > d1, f"dead time did not grow with lash: 1 deg -> {d1}, 3 deg -> {d3}"


def test_bounded_and_reaches_target():
    goals = [0.5] * 400
    qs, taus, _ = _rollout(_servo(True, 2.5), goals)
    assert all(math.isfinite(v) for v in qs + taus), "non-finite state with backlash"
    assert abs(qs[-1] - 0.5) < math.radians(3.0), f"did not settle near target: {qs[-1]:.4f}"


if __name__ == "__main__":
    test_zero_lash_is_exact_noop()
    print("  OK  lash = 0 is bit-identical to the backlash-free servo")
    for lash in (0.5, 1.0, 1.5, 2.5, 3.0):
        print(f"      lash {lash:3.1f} deg -> {_dead_time_after_reversal(lash):2d} torque-free "
              f"steps after a reversal ({_dead_time_after_reversal(lash) * DT * 1000:.0f} ms)")
    test_reversal_has_dead_time_growing_with_lash()
    print("  OK  dead time grows with lash")
    test_bounded_and_reaches_target()
    print("  OK  bounded, settles at target with 2.5 deg lash")
