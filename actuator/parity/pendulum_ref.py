"""
actuator/parity/pendulum_ref.py -- the shared reference pendulum and BAM's rollout of it.

Kept free of simulator imports on purpose: MuJoCo's DLL fails to load inside a running
Isaac Kit process (a library conflict), so isaac_pendulum.py can't import anything that
imports mujoco. mujoco_pendulum.py and isaac_pendulum.py both take their constants and the
reference trajectory from here, so the two parity numbers are directly comparable.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

PARAMS_DIR = _REPO / "actuator" / "params" / "sts3215_12v"
DT, VIN, KP = 0.005, 12.0, 32
MASS, LENGTH = 0.25, 0.10
TRAJECTORIES = ["sin_time_square", "lift_and_drop", "up_and_down", "sin_sin"]
THRESH = 0.01


def _reference(model_name: str, traj_name: str) -> np.ndarray:
    from bam.feetech.actuator import STS3215Actuator
    from bam.model import models
    from bam.simulate import Simulator
    from bam.testbench import Pendulum
    from bam.trajectory import trajectories
    traj = trajectories[traj_name]
    entries = [{"position": 0.0, "speed": 0.0, "goal_position": float(traj(k * DT)[0]),
                "torque_enable": bool(traj(k * DT)[1])} for k in range(int(round(traj.duration / DT)))]
    log = {"dt": DT, "mass": MASS, "arm_mass": 0.0, "length": LENGTH, "kp": KP, "vin": VIN, "entries": entries}
    m = models[model_name]()
    m.set_actuator(STS3215Actuator(Pendulum))
    m.load_parameters(str(PARAMS_DIR / f"{model_name}.json"))
    m.q_offset.value = 0.0
    q, _, _ = Simulator(m).rollout_log(copy.deepcopy(log), simulate_control=True)
    return np.asarray(q)
