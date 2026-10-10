"""
actuator/backlash.py -- add gear backlash to an MJCF as a passive hinge in series with each servo.

This is the faithful backlash model, and MuJoCo is where it is used (envs/mujoco/biped_env_v2.py).
Isaac can't express it cheaply and uses actuator/bam_kernel.py LostMotion instead; see that
docstring for why. Ported from the reference project's add_backlash.py.

For every actuated joint J, a sibling hinge J_lash is added in the same body, on the same
axis. MuJoCo composes joints in one body in order, so the link turns by q_J + q_J_lash:
  * J keeps the actuator and the rotor armature. That is the motor side.
  * J_lash wanders freely inside +-lash/2, then slams into its limits (gear teeth meeting).
    Only the link's own inertia rides on it, so in free play the output is genuinely light.
  * THE ENCODER READS THROUGH THE LASH. The STS3215's magnetic sensor is on the output
    side, so the firmware error and the observations both use q_J + q_J_lash. Back-EMF and
    gearbox friction stay motor side (dq_J), because they are rotor physics.

Limit parameters are the reference's, and they are load-bearing. With a range this small,
the default solref (0.02, 1) lets the joint overshoot its limits ~2x under load. 0.01 is
2*sim_dt (sim dt is 0.005), the stiffest stable setting, and solimp raises the impedance so
the teeth contact is nearly rigid. armature 0.001 is small but non-zero, for solver
conditioning.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

LASH_SUFFIX = "_lash"


def add_backlash(mjcf_in: str | Path, mjcf_out: str | Path, joint_names: list[str],
                 lash_deg: float) -> Path:
    """Write a copy of mjcf_in with a lash hinge after each joint in joint_names.

    The output is written next to the input by convention, so relative meshdir and asset
    paths stay valid. lash_deg is the TOTAL play; each hinge's range is +-lash_deg/2.
    """
    mjcf_in, mjcf_out = Path(mjcf_in), Path(mjcf_out)
    # Comments are stripped first. MuJoCo's parser tolerates '--' inside a comment, which
    # robot_v2.xml contains, but the XML spec forbids it and ElementTree refuses it. The
    # generated file is machine-only, so losing the comments costs nothing.
    text = re.sub(r"<!--.*?-->", "", mjcf_in.read_text(encoding="utf-8"), flags=re.DOTALL)
    root = ET.fromstring(text)
    tree = ET.ElementTree(root)
    # Joint ranges are in whatever unit the MJCF compiler declares, and MuJoCo DEFAULTS TO
    # DEGREES. Writing radians into a degree-unit model shrinks the play ~57x without any error
    # (robot_v2.xml says angle="radian"; a bare test MJCF does not).
    compiler = root.find("compiler")
    in_degrees = (compiler is None) or (compiler.get("angle", "degree") == "degree")
    half = (lash_deg if in_degrees else math.radians(lash_deg)) / 2.0
    wanted = set(joint_names)
    found: set[str] = set()

    for body in root.iter("body"):
        children = list(body)
        for idx, child in enumerate(children):
            if child.tag != "joint" or child.get("name") not in wanted:
                continue
            name = child.get("name")
            found.add(name)
            lash = ET.Element("joint", {
                "name": name + LASH_SUFFIX,
                "type": "hinge",
                "pos": child.get("pos", "0 0 0"),
                "axis": child.get("axis", "0 0 1"),
                "limited": "true",
                "range": f"{-half:.10f} {half:.10f}",  # compiler units
                "damping": "0.01",
                "armature": "0.001",
                "frictionloss": "0",
                "solreflimit": "0.01 1",
                "solimplimit": "0.95 0.999 0.0001 0.5 2",
            })
            body.insert(list(body).index(child) + 1, lash)

    missing = wanted - found
    if missing:
        raise ValueError(f"{mjcf_in}: joints not found for backlash: {sorted(missing)}")
    tree.write(mjcf_out, encoding="unicode")
    return mjcf_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add serial passive backlash hinges to an MJCF")
    ap.add_argument("mjcf_in")
    ap.add_argument("mjcf_out")
    ap.add_argument("--joints", required=True, help="comma-separated actuated joint names")
    ap.add_argument("--lash-deg", type=float, required=True, help="total play per joint, degrees")
    a = ap.parse_args()
    out = add_backlash(a.mjcf_in, a.mjcf_out, [j.strip() for j in a.joints.split(",")], a.lash_deg)
    print(f"wrote {out}")
