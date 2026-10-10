"""
actuator/replay_isaac.py -- the Isaac worker for replay.py. Not meant to be run directly.

Replays recorded SysID logs through BamActuator on a real PhysX pendulum and writes
<session>/_sim/isaac_<model>.npz, which replay.py collects and plots. Separate from
replay.py because Isaac Kit must be launched before anything imports isaaclab, and
because MuJoCo's DLL will not load inside a running Kit process.

One articulation is built per weight hole, since the holes differ in mass and length and
those are baked into the URDF at spawn time. They all live in one scene, so Kit is
launched once; the holes not under test are left unpowered and simply hang.

Run via replay.py, or directly with the Isaac python:
    python actuator/replay_isaac.py --headless --session actuator/data/<session>
"""
import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--model", default="m6")
parser.add_argument("--params", default=None)
parser.add_argument("--stall-cap", type=float, default=None)
parser.add_argument("--tau-ext", choices=["observer", "gravity"], default="observer")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
while _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("config", None)   # Kit's namespace 'config' shadows ours (see check_actuator.py)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import isaaclab.sim as sim_utils                              # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg      # noqa: E402

from actuator.logset import load_session                       # noqa: E402
from envs.assets.robotV2.bam_actuator import BamActuatorCfg    # noqa: E402

# World-fixed base, revolute about y, point mass hanging at -z. Matches bam's Pendulum
# and the MuJoCo model in replay.py, so the two simulator curves are comparable.
_URDF = """<?xml version="1.0"?>
<robot name="bam_pendulum">
  <link name="base"><inertial><origin xyz="0 0 0"/><mass value="1.0"/>
    <inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="servo" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 1.0"/><axis xyz="0 -1 0"/>
    <limit lower="-6.28" upper="6.28" effort="1000" velocity="1000"/>
  </joint>
  <link name="arm"><inertial><origin xyz="0 0 -{length}"/><mass value="{mass}"/>
    <inertia ixx="1e-9" iyy="1e-9" izz="1e-9" ixy="0" ixz="0" iyz="0"/></inertial></link>
</robot>
"""


def main() -> None:
    session = os.path.abspath(args.session)
    params = args.params or os.path.join(
        _ROOT, "actuator", "params", "sts3215_12v", f"{args.model}.json")
    logs = load_session(session)

    dts = {lg.dt for lg in logs}
    if len(dts) != 1:
        raise SystemExit(f"logs disagree on dt ({sorted(dts)}); the sim timestep is fixed")
    dt = dts.pop()

    # One rig per hole: (mass, length) are spawn-time properties.
    holes = sorted({lg.hole for lg in logs})
    rig = {h: next((lg.mass, lg.length) for lg in logs if lg.hole == h) for h in holes}

    tmp = tempfile.mkdtemp()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, gravity=(0.0, 0.0, -9.80665)))

    arts = {}
    for h in holes:
        mass, length = rig[h]
        urdf = os.path.join(tmp, f"pendulum_h{h}.urdf")
        with open(urdf, "w") as f:
            f.write(_URDF.format(mass=mass, length=length))
        vin = float(next(lg.vin for lg in logs if lg.hole == h))
        cfg = ArticulationCfg(
            prim_path=f"/World/Pendulum_h{h}",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf, fix_base=True, merge_fixed_joints=False,
                make_instanceable=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0)),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=2),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 2.0 * h, 0.0), joint_pos={"servo": 0.0}),
            actuators={"servo": BamActuatorCfg(
                joint_names_expr=["servo"], params_file=params,
                kp_fw=32.0, vin_range=(vin, vin), friction_scale_range=(1.0, 1.0),
                stall_torque_cap_nm=args.stall_cap, tau_ext_source=args.tau_ext)},
        )
        arts[h] = Articulation(cfg)

    sim.reset()
    acts = {}
    for h, art in arts.items():
        acts[h] = art.actuators["servo"]
        acts[h].attach(art)
        acts[h].torque_enable[:] = 0.0          # idle rigs hang limp

    dev = arts[holes[0]].device
    out = {}
    for lg in logs:
        art, act = arts[lg.hole], acts[lg.hole]
        # kp and vin are per-recording; the kernel holds them as tensors, so they can be
        # retuned between logs without rebuilding the articulation.
        act.servo.kernel.kp[:] = float(lg.kp)
        act.servo.kernel.vin[:] = float(lg.vin)
        art.write_joint_state_to_sim(
            torch.full((1, 1), float(lg.q[0]), device=dev),
            torch.full((1, 1), float(lg.dq[0]), device=dev))
        art.reset()
        act.reset(None)
        art.update(dt)

        q_out = np.empty(lg.n)
        for k in range(lg.n):
            q_out[k] = float(art.data.joint_pos[0, 0])
            art.set_joint_position_target(
                torch.full((1, 1), float(lg.goal[k]), device=dev))
            act.torque_enable[:] = float(lg.torque_enable[k])
            art.write_data_to_sim()
            sim.step(render=False)
            for other in arts.values():
                other.update(dt)
        act.torque_enable[:] = 0.0
        out[lg.name] = q_out
        print(f"  isaac {lg.label:30} done", flush=True)

    dest = os.path.join(session, "_sim", f"isaac_{args.model}.npz")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, **out)
    print(f"  isaac: {len(out)} logs replayed -> {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
