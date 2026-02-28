"""
Create a joint mapping template from the robot URDF.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET


def _parse_revolute_joints(urdf_path: str) -> list[dict]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.get("type", "")
        if joint_type != "revolute":
            continue
        name = joint.get("name")
        axis_elem = joint.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "1 0 0")
            axis = [float(x) for x in axis_str.split()]
        joints.append(
            {
                "robot_joint": name,
                "fbx_joint": "",
                "axis": axis,
                "sign": 1.0,
                "offset": 0.0,
                "scale": 1.0,
            }
        )
    return joints


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a joint mapping template from a URDF.")
    parser.add_argument(
        "--urdf",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "envs",
            "assets",
            "robot",
            "Robot_description",
            "urdf",
            "Robot.urdf",
        ),
        help="Path to the robot URDF file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "joint_map_template.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    joint_entries = _parse_revolute_joints(urdf_path)
    payload = {
        "robot_urdf": urdf_path,
        "notes": "Fill fbx_joint for each robot_joint. Axis is in the robot joint frame.",
        "joints": joint_entries,
    }

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote joint map template: {out_path}")


if __name__ == "__main__":
    main()
