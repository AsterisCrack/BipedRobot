#!/usr/bin/env python3
"""
actuator/rig.py -- turn the real pendulum arm into the numbers BAM's log format wants.

WHY A CONVERSION IS NEEDED AT ALL
---------------------------------
For a single revolute joint the complete rigid-body dynamics are

    I * qdd = tau_motor - g * S * sin(q) - friction

with exactly two scalars: I, the inertia about the rotation axis, and
S = sum(m_i * d_i), the gravity-torque coefficient. Holes, pockets, screws, the
double-sided servo mount -- all of it enters ONLY through those two numbers. Two
bodies with equal I and S about the same axis are dynamically indistinguishable;
no experiment on that joint can separate them.

BAM's testbench is parameterised as (mass, arm_mass, length), which is a rod plus
a tip weight -- its own rig's shape. Forcing an arbitrary arm into that form is
what would lose information. Setting arm_mass = 0 makes it a pure point pendulum,

    inertia = mass * L**2          gravity coefficient = mass * L

which is a bijection with (I, S) for positive values:

    L = I / S                      mass = S**2 / I

So this is an exact reparameterisation, not an approximation. The ONE property it
cannot represent is flexibility -- a bending arm is not a rigid body, and its mode
would be absorbed into the fit as fictitious friction. Keep the arm stiff.

Usage:
    python -m actuator.rig --config actuator/rig/rig.yaml
    python -m actuator.rig --config actuator/rig/rig.yaml --json actuator/rig/loads.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

G = 9.80665


@dataclass
class ArmInertial:
    """<inertial> of the bare arm, in the URDF link frame."""
    mass: float
    com: tuple[float, float, float]
    # Inertia tensor about the COM, in the link frame (URDF rpy is assumed zero;
    # Fusion2URDF emits it that way. A non-zero rpy would need rotating first.)
    ixx: float
    iyy: float
    izz: float
    ixy: float
    ixz: float
    iyz: float

    def tensor(self) -> list[list[float]]:
        return [[self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz]]


def parse_inertial(urdf_path: Path, link_name: str | None) -> tuple[str, ArmInertial]:
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    chosen = None
    for link in links:
        if link.find("inertial") is None:
            continue
        if link_name is None or link.get("name") == link_name:
            chosen = link
            break
    if chosen is None:
        raise SystemExit(
            f"no <inertial> found in {urdf_path}"
            + (f" for link '{link_name}'" if link_name else "")
            + f"; links present: {[l.get('name') for l in links]}")

    inertial = chosen.find("inertial")
    mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    com = (0.0, 0.0, 0.0)
    if origin is not None and origin.get("xyz"):
        com = tuple(float(v) for v in origin.get("xyz").split())  # type: ignore[assignment]
        rpy = origin.get("rpy")
        if rpy and any(abs(float(v)) > 1e-9 for v in rpy.split()):
            raise SystemExit(
                f"<inertial><origin rpy='{rpy}'> is non-zero. This script assumes the "
                f"inertia tensor is expressed in the link frame. Re-export with zero rpy, "
                f"or rotate the tensor before using it.")
    i = inertial.find("inertia")
    return chosen.get("name") or "?", ArmInertial(
        mass=mass, com=com,
        ixx=float(i.get("ixx")), iyy=float(i.get("iyy")), izz=float(i.get("izz")),
        ixy=float(i.get("ixy", 0.0)), ixz=float(i.get("ixz", 0.0)),
        iyz=float(i.get("iyz", 0.0)),
    )


def axis_inertia_about_com(arm: ArmInertial, axis: tuple[float, float, float]) -> float:
    """n^T I n -- the scalar inertia about an axis through the COM."""
    n = _unit(axis)
    t = arm.tensor()
    return sum(n[a] * t[a][b] * n[b] for a in range(3) for b in range(3))


def perpendicular_distance(point: tuple[float, float, float],
                           axis: tuple[float, float, float]) -> float:
    """Distance from a point to the line through the origin along `axis`.

    This single number is both the parallel-axis lever and the gravity lever:
    the COM sweeps a circle of this radius about the rotation axis.
    """
    n = _unit(axis)
    along = sum(point[k] * n[k] for k in range(3))
    perp = [point[k] - along * n[k] for k in range(3)]
    return math.sqrt(sum(v * v for v in perp))


def _unit(v) -> list[float]:
    norm = math.sqrt(sum(float(c) ** 2 for c in v))
    if norm < 1e-12:
        raise SystemExit(f"axis {v} has zero length")
    return [float(c) / norm for c in v]


@dataclass
class Load:
    hole_m: float
    inertia_kg_m2: float      # I about the rotation axis, arm + payload
    first_moment_kg_m: float  # S
    mass: float               # BAM equivalent point mass
    length: float             # BAM equivalent length
    arm_mass: float           # always 0.0 -- see module docstring
    static_torque_nm: float   # peak gravity torque, = g * S (arm horizontal)


def build_loads(arm: ArmInertial, axis, payload_mass: float, payload_inertia: float,
                holes: list[float]) -> tuple[float, float, list[Load]]:
    d_perp = perpendicular_distance(arm.com, axis)
    i_arm_com = axis_inertia_about_com(arm, axis)
    i_arm_pivot = i_arm_com + arm.mass * d_perp ** 2      # parallel axis
    s_arm = arm.mass * d_perp

    loads = []
    for r in holes:
        i_total = i_arm_pivot + payload_mass * r ** 2 + payload_inertia
        s_total = s_arm + payload_mass * r
        if i_total <= 0 or s_total <= 0:
            raise SystemExit(f"hole {r}: non-positive I={i_total} or S={s_total}")
        loads.append(Load(
            hole_m=r,
            inertia_kg_m2=i_total,
            first_moment_kg_m=s_total,
            mass=s_total ** 2 / i_total,
            length=i_total / s_total,
            arm_mass=0.0,
            static_torque_nm=G * s_total,
        ))
    return i_arm_pivot, s_arm, loads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("actuator/rig/rig.yaml"))
    ap.add_argument("--json", type=Path, default=None,
                    help="write the load table here (default: alongside the config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    arm_cfg, pay_cfg, servo_cfg = cfg["arm"], cfg["payload"], cfg.get("servo", {})

    urdf = Path(arm_cfg["urdf"])
    if not urdf.is_absolute():
        urdf = args.config.parent / urdf
    link_name, arm = parse_inertial(urdf, arm_cfg.get("link"))

    # Rescale if the printed part does not weigh what CAD says. Correct only when
    # the density error is uniform -- true for constant infill, false for an
    # unmodelled steel insert.
    measured = arm_cfg.get("measured_mass_kg")
    rescaled = False
    if measured:
        ratio = float(measured) / arm.mass
        if abs(ratio - 1.0) > 1e-3:
            for f in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                setattr(arm, f, getattr(arm, f) * ratio)
            arm.mass = float(measured)
            rescaled = True

    axis = tuple(arm_cfg.get("axis", [0.0, 0.0, 1.0]))
    payload_mass = float(pay_cfg["mass_kg"])
    holes = [float(h) for h in pay_cfg["holes_m"]]
    i_arm, s_arm, loads = build_loads(
        arm, axis, payload_mass, float(pay_cfg.get("inertia_kg_m2", 0.0)), holes)

    print(f"\n  arm      : {urdf.name}  link '{link_name}'")
    print(f"  axis     : {tuple(round(c, 4) for c in _unit(axis))} through the URDF origin")
    print(f"  arm mass : {arm.mass * 1000:.1f} g"
          + (f"   (CAD tensor rescaled x{float(measured) / (arm.mass / (float(measured) / arm.mass)):.4f})"
             if rescaled else "   (CAD mass used as-is)"))
    print(f"  COM      : {perpendicular_distance(arm.com, axis) * 1000:.2f} mm from the axis")
    print(f"  arm alone: I = {i_arm:.6e} kg m^2   S = {s_arm:.6e} kg m")
    print(f"  payload  : {payload_mass * 1000:.1f} g\n")

    print(f"  {'hole':>7}{'I [kg m^2]':>14}{'S [kg m]':>12}"
          f"{'-> mass [kg]':>14}{'length [m]':>12}{'peak tau':>11}")
    for ld in loads:
        print(f"  {ld.hole_m * 1000:>5.0f}mm{ld.inertia_kg_m2:>14.6f}"
              f"{ld.first_moment_kg_m:>12.6f}{ld.mass:>14.4f}{ld.length:>12.4f}"
              f"{ld.static_torque_nm:>10.3f}N")

    peak = max(ld.static_torque_nm for ld in loads)
    print(f"\n  Peak gravity torque {peak:.3f} N.m "
          f"({peak / 0.98 * 100:.0f}% of the 0.98 N.m continuous rating).")
    if peak > 0.85:
        print("  WARNING: close to the continuous rating. Saturated frames are the data")
        print("           contamination that caused the reference project a 29% torque error.")
    elif peak < 0.15:
        print("  WARNING: weak excitation -- load-dependent friction will be poorly identified.")

    out = args.json or args.config.parent / "loads.json"
    out.write_text(json.dumps({
        "source_urdf": urdf.name,
        "arm_mass_kg": arm.mass,
        "arm_inertia_about_axis_kg_m2": i_arm,
        "payload_mass_kg": payload_mass,
        "vin": servo_cfg.get("vin"),
        "kp_sweep": servo_cfg.get("kp_sweep"),
        # Each entry drops straight into a BAM raw log header.
        "loads": [asdict(ld) for ld in loads],
    }, indent=2) + "\n")
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
