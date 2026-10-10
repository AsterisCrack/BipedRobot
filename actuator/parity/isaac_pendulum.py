"""
actuator/parity/isaac_pendulum.py -- the Isaac BamActuator on a real PhysX pendulum vs bam.simulate.

The last link in the parity chain:
    test_kernel_vs_bam.py   kernel == BAM, inside BAM's own explicit-Euler loop
    mujoco_pendulum.py      kernel inside MuJoCo == BAM, within 0.01 rad MAE
    THIS                    BamActuator inside Isaac/PhysX == BAM, within 0.01 rad MAE
Isaac is where the policy trains, and it is the one simulator where the integration
can't be bit-exact:
  * external torque comes from the lagged dynamics observer, not qfrc_bias;
  * effective inertia comes from the PhysX generalized mass matrix (fixed base here, so
    no floating-base offset), plus the armature probe;
  * friction is an explicit effort that PhysX's TGS solver integrates.
So this measures that integration error against ground truth. The pendulum (250 g
point mass at 100 mm, hanging down at q = 0) is identical to the MuJoCo test's, so all
three numbers are comparable.

Run with the Isaac Lab python, from the BipedRobot root, with bam importable:
    python actuator/parity/isaac_pendulum.py --headless [--model m6] [--tau-ext gravity]
"""
import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="m6")
parser.add_argument("--tau-ext", choices=["observer", "gravity"], default="observer")
parser.add_argument("--out", default=None, help="write the result table to this file too")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
while _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("config", None)   # Kit's namespace 'config' shadows ours (see check_actuator.py)

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402

import isaaclab.sim as sim_utils                 # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg   # noqa: E402

from actuator.parity.pendulum_ref import (       # noqa: E402
    DT, KP, LENGTH, MASS, PARAMS_DIR, THRESH, TRAJECTORIES, _reference,
)
from envs.assets.robotV2.bam_actuator import BamActuatorCfg   # noqa: E402

# World-fixed base, revolute about y, point mass hanging at -z. Matches bam's Pendulum.
_URDF = f"""<?xml version="1.0"?>
<robot name="bam_pendulum">
  <link name="base"><inertial><origin xyz="0 0 0"/><mass value="1.0"/>
    <inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="servo" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 1.0"/><axis xyz="0 -1 0"/>
    <limit lower="-6.28" upper="6.28" effort="1000" velocity="1000"/>
  </joint>
  <link name="arm"><inertial><origin xyz="0 0 -{LENGTH}"/><mass value="{MASS}"/>
    <inertia ixx="1e-8" iyy="1e-8" izz="1e-8" ixy="0" ixz="0" iyz="0"/></inertial></link>
</robot>
"""


def main() -> None:
    tmp = tempfile.mkdtemp()
    urdf = os.path.join(tmp, "bam_pendulum.urdf")
    with open(urdf, "w") as f:
        f.write(_URDF)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, gravity=(0.0, 0.0, -9.80665)))
    cfg = ArticulationCfg(
        prim_path="/World/Pendulum",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf, fix_base=True, merge_fixed_joints=False, make_instanceable=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"servo": 0.0}),
        actuators={"servo": BamActuatorCfg(
            joint_names_expr=["servo"], params_file=str(PARAMS_DIR / f"{args.model}.json"),
            kp_fw=float(KP), vin_range=(12.0, 12.0), friction_scale_range=(1.0, 1.0),
            tau_ext_source=args.tau_ext)},
    )
    art = Articulation(cfg)
    sim.reset()
    act = art.actuators["servo"]
    act.attach(art)

    from bam.trajectory import trajectories
    lines, worst = [], 0.0
    for tr in TRAJECTORIES:
        traj = trajectories[tr]
        # reset to hanging, at rest
        art.write_joint_state_to_sim(torch.zeros(1, 1, device=art.device), torch.zeros(1, 1, device=art.device))
        art.reset()
        art.update(DT)
        q_out = []
        for k in range(int(round(traj.duration / DT))):
            q_out.append(float(art.data.joint_pos[0, 0]))
            goal, enable = traj(k * DT)
            art.set_joint_position_target(torch.full((1, 1), float(goal), device=art.device))
            act.torque_enable[:] = float(enable)
            art.write_data_to_sim()
            sim.step(render=False)
            art.update(DT)
        ref = _reference(args.model, tr)
        mae = float(np.mean(np.abs(np.asarray(q_out) - ref)))
        worst = max(worst, mae)
        lines.append(f"  {'OK ' if mae < THRESH else 'BAD'} {args.model} {tr:<16} Isaac ({args.tau_ext}) vs "
                     f"bam.simulate MAE = {mae * 1000:6.2f} mrad")
    lines.append(f"\n  armature in PhysX mass matrix: {act._armature_in_mass_matrix}; "
                 f"inertia_eff = {act.inertia_eff[0, 0].item():.5f} (BAM: "
                 f"{MASS * LENGTH ** 2 + act.params.armature:.5f})")
    lines.append(f"  worst MAE = {worst * 1000:.2f} mrad (threshold {THRESH * 1000:.0f} mrad)")
    text = "\n".join(lines)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
