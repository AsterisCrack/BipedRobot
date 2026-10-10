"""
actuator/parity/mujoco_pendulum.py -- the BAM kernel inside a real MuJoCo integrator vs bam.simulate.

test_kernel_vs_bam.py proves the kernel equals BAM inside BAM's own explicit-Euler loop.
This checks the next link in the chain: the kernel driving MuJoCo exactly the way
envs/mujoco/biped_env_v2.py drives it, with
  external torque = -qfrc_bias + qfrc_constraint   (from the previous solve, as in BAM)
  inertia         = qM diagonal (armature included)
  integrator      = implicitfast at 200 Hz
should reproduce the reference rollout. The two integrators are not identical (MuJoCo vs
explicit Euler), so the bar is BAM's own "likely a kernel bug" threshold: position MAE
below 0.01 rad.

Optionally (--lash-deg) the same pendulum gets a serial backlash hinge
(actuator/backlash.py), and the run reports how much that moves the trajectory. There
is no reference for that case; it shows the size of the effect.

Run:  python -m actuator.parity.mujoco_pendulum [--model m6] [--lash-deg 1.5]
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from actuator.backlash import LASH_SUFFIX, add_backlash          # noqa: E402
from actuator.bam_kernel import BamServo, load_params            # noqa: E402

from actuator.parity.pendulum_ref import (                 # noqa: E402
    DT, KP, LENGTH, MASS, PARAMS_DIR, THRESH, TRAJECTORIES, VIN, _reference,
)

# Point mass on a massless rod, hinge about y, q = 0 hanging down. Matches bam's Pendulum
# testbench: inertia m*L^2 about the pivot, gravity torque -m*g*L*sin(q).
_PENDULUM_XML = f"""
<mujoco model="pendulum">
  <compiler angle="radian"/>
  <option timestep="{DT}" integrator="implicitfast" gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="servo" type="hinge" axis="0 -1 0" damping="0" frictionloss="0"/>
      <geom type="sphere" size="0.005" pos="0 0 -{LENGTH}" mass="{MASS}"/>
    </body>
  </worldbody>
  <actuator><motor name="servo" joint="servo" ctrllimited="false"/></actuator>
</mujoco>
"""


def _mujoco(model_name: str, traj_name: str, lash_deg: float = 0.0) -> np.ndarray:
    from bam.trajectory import trajectories
    traj = trajectories[traj_name]
    params = load_params(PARAMS_DIR / f"{model_name}.json")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pendulum.xml"
        path.write_text(_PENDULUM_XML)
        if lash_deg > 0.0:
            path = add_backlash(path, Path(tmp) / "pendulum_lash.xml", ["servo"], lash_deg)
        model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    dof = model.joint("servo").dofadr[0]
    lash_dof = model.joint("servo" + LASH_SUFFIX).dofadr[0] if lash_deg > 0.0 else None
    model.dof_armature[dof] = params.armature
    servo = BamServo(params, 1, 1, DT, kp=KP, vin=VIN, dtype=torch.float64)
    madr = model.dof_Madr[dof]
    mujoco.mj_forward(model, data)

    out = []
    for k in range(int(round(traj.duration / DT))):
        q_enc = data.qpos[dof] + (data.qpos[lash_dof] if lash_dof is not None else 0.0)
        out.append(q_enc)
        goal, enable = traj(k * DT)
        t = lambda x: torch.tensor([[float(x)]], dtype=torch.float64)
        tau = servo.step(t(goal), t(q_enc), t(data.qvel[dof]),
                         t(-data.qfrc_bias[dof] + data.qfrc_constraint[dof]),
                         t(data.qM[madr]), torque_enable=float(enable))
        data.ctrl[0] = float(tau)
        mujoco.mj_step(model, data)
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m6")
    ap.add_argument("--lash-deg", type=float, default=0.0)
    a = ap.parse_args()
    worst = 0.0
    for tr in TRAJECTORIES:
        ref = _reference(a.model, tr)
        mj = _mujoco(a.model, tr)
        mae = float(np.mean(np.abs(mj - ref)))
        worst = max(worst, mae)
        line = f"  {'OK ' if mae < THRESH else 'BAD'} {a.model} {tr:<16} MuJoCo vs bam.simulate MAE = {mae * 1000:6.2f} mrad"
        if a.lash_deg > 0.0:
            lash = _mujoco(a.model, tr, a.lash_deg)
            line += f"   |  +{a.lash_deg:.1f} deg lash moves it by {np.mean(np.abs(lash - mj)) * 1000:6.2f} mrad"
        print(line)
    print(f"\n  worst MAE = {worst * 1000:.2f} mrad (threshold {THRESH * 1000:.0f} mrad)")
    return 0 if worst < THRESH else 1


if __name__ == "__main__":
    raise SystemExit(main())
