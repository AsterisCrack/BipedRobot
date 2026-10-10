"""
actuator/parity/test_kernel_vs_bam.py -- prove bam_kernel.py IS the model the params were fitted to.

Every other actuator check (MuJoCo, Isaac, hardware) assumes the kernel reproduces BAM.
This test establishes that directly. It rolls out a pendulum two ways:

    reference  bam.simulate.Simulator.rollout_log(simulate_control=True), the exact
               code path bam.fit uses to score a parameter set
    ours       BamKernel inside the same explicit-Euler integrator

and runs every model (m1..m6) on every built-in trajectory, at two kp values in one
batch so the per-env vectorisation is exercised too. Both run in float64, and the
position traces must agree to 1e-9 rad. Any real porting error -- a dropped friction
term, a sign gate, a delay off by one sample -- shows up as orders of magnitude more.

Run (needs the bam package; see actuator/requirements.txt):
    python -m actuator.parity.test_kernel_vs_bam
    python -m pytest actuator/parity/test_kernel_vs_bam.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from actuator.bam_kernel import BamKernel, load_params      # noqa: E402

from bam.feetech.actuator import STS3215Actuator             # noqa: E402
from bam.model import models                                 # noqa: E402
from bam.simulate import Simulator                           # noqa: E402
from bam.testbench import Pendulum                           # noqa: E402
from bam.trajectory import trajectories                      # noqa: E402

PARAMS_DIR = _REPO / "actuator" / "params" / "sts3215_12v"
TRAJECTORIES = ["sin_time_square", "lift_and_drop", "up_and_down", "sin_sin"]
DT = 0.005
VIN = 12.0
KPS = [32, 8]
# A mid-range rig load: 250 g point mass at 100 mm, ~0.25 N*m peak gravity torque.
MASS, ARM_MASS, LENGTH = 0.25, 0.0, 0.10
G = -9.80665
TOL = 1e-9


def _make_log(traj_name: str, kp: int) -> dict:
    traj = trajectories[traj_name]
    n = int(round(traj.duration / DT))
    entries = []
    for k in range(n):
        goal, enable = traj(k * DT)
        entries.append({"position": 0.0, "speed": 0.0, "goal_position": float(goal),
                        "torque_enable": bool(enable)})
    return {"dt": DT, "mass": MASS, "arm_mass": ARM_MASS, "length": LENGTH,
            "kp": kp, "vin": VIN, "entries": entries}


def _reference(model_name: str, log: dict) -> np.ndarray:
    m = models[model_name]()
    m.set_actuator(STS3215Actuator(Pendulum))
    m.load_parameters(str(PARAMS_DIR / f"{model_name}.json"))
    m.q_offset.value = 0.0          # the kernel ignores q_offset; compare like with like
    positions, _, _ = Simulator(m).rollout_log(copy.deepcopy(log), simulate_control=True)
    return np.asarray(positions, dtype=np.float64)


def _ours(model_name: str, logs: list[dict]) -> np.ndarray:
    """All logs at once, one per env, with 1 joint. Same integrator as bam/simulate.py."""
    params = load_params(PARAMS_DIR / f"{model_name}.json")
    n_env = len(logs)
    kp = torch.tensor([[float(lg["kp"])] for lg in logs], dtype=torch.float64)
    k = BamKernel(params, n_env, 1, DT, kp=kp, vin=VIN, dtype=torch.float64)

    inertia = MASS * LENGTH ** 2 + (ARM_MASS / 3) * LENGTH ** 2 + params.armature
    q = torch.zeros(n_env, 1, dtype=torch.float64)
    dq = torch.zeros(n_env, 1, dtype=torch.float64)
    n = len(logs[0]["entries"])
    out = np.zeros((n, n_env))
    for i in range(n):
        out[i] = q[:, 0].numpy()
        goal = torch.tensor([[lg["entries"][i]["goal_position"]] for lg in logs], dtype=torch.float64)
        enable = torch.tensor([[float(lg["entries"][i]["torque_enable"])] for lg in logs],
                              dtype=torch.float64)
        tau_m = k.motor(goal, q, dq, enable)
        bias = (MASS + ARM_MASS / 2) * G * LENGTH * torch.sin(q)
        tau_f = k.stop_clip(tau_m, bias, dq, inertia)
        net = (tau_m + bias) + tau_f           # bam: net = motor + bias; net += static
        dq = torch.clamp(dq + (net / inertia) * DT, -100.0, 100.0)
        q = q + dq * DT
    return out


def run() -> dict[tuple[str, str], float]:
    results = {}
    for model_name in ["m1", "m2", "m3", "m4", "m5", "m6"]:
        for traj in TRAJECTORIES:
            logs = [_make_log(traj, kp) for kp in KPS]
            ours = _ours(model_name, logs)
            for j, lg in enumerate(logs):
                ref = _reference(model_name, lg)
                results[(model_name, f"{traj}@kp{lg['kp']}")] = float(np.max(np.abs(ours[:, j] - ref)))
    return results


def test_kernel_matches_bam():
    bad = {k: v for k, v in run().items() if not v < TOL}
    assert not bad, f"kernel diverges from bam.simulate: {bad}"


if __name__ == "__main__":
    res = run()
    width = max(len(k[1]) for k in res)
    for (m, t), err in res.items():
        flag = "OK " if err < TOL else "BAD"
        print(f"  {flag} {m}  {t:<{width}}  max|dq| = {err:.3e} rad")
    worst = max(res.values())
    print(f"\n  worst = {worst:.3e} rad  (tolerance {TOL:.0e})")
    sys.exit(0 if worst < TOL else 1)
