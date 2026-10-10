"""
actuator/logset.py -- load SysID logs and their rig metadata, once, for every consumer.

validate.py (actuator/.venv, has bam) and replay.py (env_isaaclab, has torch+mujoco+
isaaclab) both need the same three things: which (trajectory, kp, hole) a file is, the
rig the recording was made on, and the recorded arrays. Neither venv has the other's
heavy dependency, so this module is numpy + stdlib only and imports from neither side.

Logs are BAM's processed format (bam.process output): fixed dt, entries resampled onto
it. The raw Jetson logs have the same keys but irregular timestamps, so they load too --
only `dt` differs in meaning. Prefer processed, since that is what the MAE is computed on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 2026-10-09_18h21m34s_lift_and_drop_kp4_h0.json -> (lift_and_drop, 4, 0)
NAME_RE = re.compile(r"_([a-z_]+)_kp(\d+)_h(\d+)\.json$")


@dataclass
class Log:
    """One recording: what it is, the rig it was on, and what the servo did."""
    path: Path
    name: str
    trajectory: str
    kp: int
    hole: int
    # Rig, as rig.py computed it: the point-pendulum equivalent BAM was given.
    mass: float
    length: float
    vin: float
    dt: float
    # Arrays, all length n.
    t: np.ndarray
    goal: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    torque_enable: np.ndarray

    @property
    def n(self) -> int:
        return len(self.q)

    @property
    def label(self) -> str:
        return f"{self.trajectory} kp{self.kp} h{self.hole}"


def parse_name(path: Path) -> tuple[str, int, int]:
    m = NAME_RE.search(path.name)
    if not m:
        raise ValueError(f"{path.name}: not a SysID recording name (<traj>_kp<N>_h<N>.json)")
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_log(path: str | Path) -> Log:
    path = Path(path)
    traj, kp, hole = parse_name(path)
    d = json.loads(path.read_text())
    e = d["entries"]
    dt = float(d["dt"])
    get = lambda k, default=0.0: np.array([x.get(k, default) for x in e], dtype=float)
    return Log(
        path=path, name=path.name, trajectory=traj, kp=kp, hole=hole,
        mass=float(d["mass"]), length=float(d["length"]),
        vin=float(d["vin"]), dt=dt,
        t=np.arange(len(e), dtype=float) * dt,
        goal=get("goal_position"), q=get("position"), dq=get("speed"),
        torque_enable=np.array([bool(x.get("torque_enable", True)) for x in e]),
    )


def load_session(session: str | Path, holes=None, kps=None, trajectories=None) -> list[Log]:
    """Load a session's processed logs, sorted by (hole, kp, trajectory).

    Falls back to the session root if there is no processed/ dir, so a session can be
    inspected before validate.py has resampled it.
    """
    session = Path(session)
    folder = session / "processed"
    if not folder.is_dir():
        folder = session
    logs = []
    for p in sorted(folder.glob("*.json")):
        if not NAME_RE.search(p.name):
            continue                      # mae.json, loads.json, sidecars
        lg = load_log(p)
        if holes is not None and lg.hole not in holes:
            continue
        if kps is not None and lg.kp not in kps:
            continue
        if trajectories is not None and lg.trajectory not in trajectories:
            continue
        logs.append(lg)
    if not logs:
        raise SystemExit(f"no SysID recordings found under {folder}")
    return sorted(logs, key=lambda l: (l.hole, l.kp, l.trajectory))


def axes_of(logs: list[Log]) -> tuple[list[str], list[int], list[int]]:
    """The (trajectories, kps, holes) present, each sorted -- the grid the plots use."""
    return (sorted({l.trajectory for l in logs}),
            sorted({l.kp for l in logs}),
            sorted({l.hole for l in logs}))
