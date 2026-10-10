"""
actuator/replay.py -- replay recorded SysID logs through MuJoCo and Isaac, against hardware.

validate.py answers "do the published parameters describe our servos?" by rolling the
model out in BAM's own integrator. That leaves the question this file answers: does the
servo model still behave the same once it is embedded in the simulator the policy
actually trains in? A kernel that matches BAM but drifts inside PhysX would train a
policy against a servo the robot does not have.

Each log is replayed open-loop: the simulator gets the recorded goal sequence and the
recorded torque_enable, starts from the recorded initial state, and runs. The rig is the
log's own point-pendulum equivalent (mass, length from rig.py), so hole h2 really is a
heavier swing than h0. Nothing is fed back from the recording after t=0.

PROCESS LAYOUT
--------------
MuJoCo's DLL will not load inside a running Isaac Kit process, so the two never share one
interpreter. This file is the orchestrator AND the MuJoCo worker; replay_isaac.py is the
Isaac worker. The orchestrator spawns each as a subprocess, collects the .npz each writes,
and draws the comparison. It imports neither simulator itself.

    python -m actuator.replay --session actuator/data/bench_servo_2
    python -m actuator.replay --session ... --sims mujoco        # skip Isaac (no Kit launch)
    python -m actuator.replay --session ... --stall-cap 2.94     # datasheet cap, as in config
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

# Point mass on a massless rod, hinge about y, q = 0 hanging down -- bam's Pendulum
# testbench: inertia m*L^2 about the pivot, gravity torque -m*g*L*sin(q). The mass is
# carried by <inertial> with negligible diagonal, so it is a true point mass; the geom
# is massless and collision-free, present only so the body is well formed.
PENDULUM_XML = """
<mujoco model="pendulum">
  <compiler angle="radian"/>
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="servo" type="hinge" axis="0 -1 0" damping="0" frictionloss="0"/>
      <inertial pos="0 0 -{length}" mass="{mass}" diaginertia="1e-9 1e-9 1e-9"/>
      <geom type="sphere" size="0.004" pos="0 0 -{length}" mass="0"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><motor name="servo" joint="servo" ctrllimited="false"/></actuator>
</mujoco>
"""


def run_mujoco(logs, params_file: str, stall_cap: float | None) -> dict:
    """Drive the BAM kernel inside MuJoCo exactly as envs/mujoco/biped_env_v2.py does."""
    import mujoco
    import torch

    from actuator.bam_kernel import BamServo, apply_stall_cap, load_params

    params = apply_stall_cap(load_params(params_file), stall_cap)
    out = {}
    for lg in logs:
        model = mujoco.MjModel.from_xml_string(
            PENDULUM_XML.format(dt=lg.dt, mass=lg.mass, length=lg.length))
        data = mujoco.MjData(model)
        dof = model.joint("servo").dofadr[0]
        madr = model.dof_Madr[dof]
        model.dof_armature[dof] = params.armature
        servo = BamServo(params, 1, 1, lg.dt, kp=float(lg.kp), vin=float(lg.vin),
                         dtype=torch.float64)
        # Start where the recording started, so only the dynamics are on trial.
        data.qpos[dof], data.qvel[dof] = float(lg.q[0]), float(lg.dq[0])
        mujoco.mj_forward(model, data)

        def tt(x):
            return torch.tensor([[float(x)]], dtype=torch.float64)

        q_out = np.empty(lg.n)
        for k in range(lg.n):
            q_out[k] = data.qpos[dof]
            tau = servo.step(tt(lg.goal[k]), tt(data.qpos[dof]), tt(data.qvel[dof]),
                             tt(-data.qfrc_bias[dof] + data.qfrc_constraint[dof]),
                             tt(data.qM[madr]), torque_enable=float(lg.torque_enable[k]))
            data.ctrl[0] = float(tau)
            mujoco.mj_step(model, data)
        out[lg.name] = q_out
    print(f"  mujoco: {len(out)} logs replayed", flush=True)
    return out


def _plot(session: Path, sims: dict, model: str) -> list:
    from actuator import plots, plotstyle
    from actuator.logset import axes_of, load_session

    plotstyle.apply()
    logs = load_session(session)
    # Roles, in the order plotstyle validated them for overlay.
    colors = dict(zip(sims, (plotstyle.SIM1, plotstyle.SIM2)))
    out_dir = session / "plots"
    written = []
    _, _, holes = axes_of(logs)
    for h in holes:
        sub = [lg for lg in logs if lg.hole == h]
        mm = sub[0]
        written.append(plots.overlay_grid(
            sub, sims, colors,
            title=f"{session.name} - hardware vs simulators ({model}), hole {h}",
            subtitle=f"open-loop replay of the recorded goal; {mm.mass * 1000:.0f} g at "
                     f"{mm.length * 1000:.1f} mm, vin {mm.vin:.2f} V"
                     f"   |   grey band = torque off",
            out=out_dir / f"sim_{model}_h{h}.png"))
    written.append(plots.mae_summary(
        logs, sims, colors, title=f"{session.name} - simulator error vs hardware ({model})",
        out=out_dir / f"sim_summary_{model}.png"))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", default="m6")
    ap.add_argument("--params", default=None,
                    help="params JSON (default: params/sts3215_12v/<model>.json)")
    ap.add_argument("--sims", default="mujoco,isaac", help="comma list of mujoco,isaac")
    ap.add_argument("--stall-cap", type=float, default=None,
                    help="cap duty-limited stall torque [N*m], e.g. 2.94 (datasheet)")
    ap.add_argument("--reuse", action="store_true",
                    help="keep any existing _sim/<sim>_<model>.npz instead of re-running it")
    ap.add_argument("--worker", choices=["mujoco"], default=None,
                    help="internal: run one simulator and write its npz")
    a = ap.parse_args()

    session = Path(a.session).resolve()
    params = a.params or str(_REPO / "actuator" / "params" / "sts3215_12v" / f"{a.model}.json")
    cache = session / "_sim"
    cache.mkdir(parents=True, exist_ok=True)

    from actuator.logset import load_session
    logs = load_session(session)

    if a.worker == "mujoco":
        np.savez(cache / f"mujoco_{a.model}.npz", **run_mujoco(logs, params, a.stall_cap))
        return 0

    wanted = [s.strip() for s in a.sims.split(",") if s.strip()]
    if len(wanted) > 2:
        print("at most two simulators can be overlaid (see plotstyle)", file=sys.stderr)
        return 1

    # Each simulator runs in its own interpreter: MuJoCo and Isaac Kit cannot coexist.
    for sim in wanted:
        npz = cache / f"{sim}_{a.model}.npz"
        if a.reuse and npz.exists():
            print(f"[replay] {sim}: reusing {npz.name}", flush=True)
            continue
        if sim == "mujoco":
            cmd = [sys.executable, "-m", "actuator.replay", "--session", str(session),
                   "--model", a.model, "--params", params, "--worker", "mujoco"]
        else:
            cmd = [sys.executable, str(_REPO / "actuator" / "replay_isaac.py"), "--headless",
                   "--session", str(session), "--model", a.model, "--params", params]
        if a.stall_cap is not None:
            cmd += ["--stall-cap", str(a.stall_cap)]
        print(f"[replay] launching {sim} worker...", flush=True)
        r = subprocess.run(cmd, cwd=str(_REPO))
        if r.returncode != 0 or not npz.exists():
            print(f"[replay] {sim} worker failed (exit {r.returncode}); skipping it",
                  file=sys.stderr)

    sims = {}
    for sim in wanted:
        npz = cache / f"{sim}_{a.model}.npz"
        if npz.exists():
            with np.load(npz) as z:
                sims[sim] = {k: z[k] for k in z.files}
    if not sims:
        print("[replay] no simulator produced results", file=sys.stderr)
        return 1

    print(f"\n[replay] MAE vs hardware, {a.model}:")
    by = {lg.name: lg for lg in logs}
    for sim, curves in sims.items():
        on_err, all_err = [], []
        for name, y in curves.items():
            lg = by[name]
            e = np.abs(y - lg.q[:len(y)])
            all_err.append(e.mean())
            m = lg.torque_enable[:len(y)]
            if m.any():
                on_err.append(e[m].mean())
        print(f"    {sim:8} torque-on {np.mean(on_err) * 1000:7.2f} mrad   "
              f"all {np.mean(all_err) * 1000:7.2f} mrad   ({len(curves)} logs)")

    for f in _plot(session, sims, a.model):
        print(f"    wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
