"""Regenerate command_limit_golden.json (pure python maths, independent of utils/command_limit.py).

    python tests/data/make_command_limit_golden.py

The input set is fixed on purpose: the Jetson and Gait Studio copies reproduce the same file.
"""
import json
import math
import os

R1 = {"lin_vel_x": [-0.2, 0.2], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-1.2, 1.2]}
R2 = {"lin_vel_x": [-0.1, 0.3], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-1.0, 1.0]}
CMDS = [(0, 0, 0), (0.2, 0, 0), (-0.2, 0, 0), (0, 0.15, 0), (0, -0.15, 0), (0.2, 0.15, 0),
        (0.2, -0.15, 0.5), (-0.2, 0.15, -1.2), (0.1, 0.075, 0.3), (0.3, 0.3, 2.0),
        (0.14, 0.1, 0), (-0.05, 0.14, 0.9), (1e-9, 0, 0)]
SHAPES = ["box", "ellipse_xy"]


def limit(cmd, ranges, shape):
    vx, vy, wz = (float(c) for c in cmd)
    vx = min(max(vx, ranges["lin_vel_x"][0]), ranges["lin_vel_x"][1])
    vy = min(max(vy, ranges["lin_vel_y"][0]), ranges["lin_vel_y"][1])
    wz = min(max(wz, ranges["ang_vel_z"][0]), ranges["ang_vel_z"][1])
    if shape == "ellipse_xy":
        bx = ranges["lin_vel_x"][1] if vx >= 0 else -ranges["lin_vel_x"][0]
        by = ranges["lin_vel_y"][1] if vy >= 0 else -ranges["lin_vel_y"][0]
        s = math.sqrt((vx / bx) ** 2 + (vy / by) ** 2)
        if s > 1.0:
            vx, vy = vx / s, vy / s
    return [vx, vy, wz]


def main(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "command_limit_golden.json")
    cases = []
    for rname, r in (("R1", R1), ("R2", R2)):
        for shape in SHAPES:
            for cmd in CMDS:
                cases.append({"ranges": rname, "shape": shape, "cmd": [float(c) for c in cmd],
                              "out": limit(cmd, r, shape)})
    with open(path, "w", newline="\n") as f:
        f.write('{\n  "R1": ' + json.dumps(R1) + ',\n  "R2": ' + json.dumps(R2) + ',\n  "cases": [\n')
        f.write(",\n".join("    " + json.dumps(c) for c in cases))   # json floats are repr: full precision
        f.write("\n  ]\n}\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
