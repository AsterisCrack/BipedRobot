"""
Generic multi end-effector IK helper based on DifferentialIKController.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms


@dataclass
class EndEffectorConfig:
    name: str
    joint_names: Iterable[str]
    joint_weights: Iterable[float] | None = None


class MultiEndEffectorIK:
    """Numerical IK helper for multiple end-effectors using DLS."""

    def __init__(
        self,
        robot,
        end_effectors: Iterable[EndEffectorConfig],
        lambda_val: float = 0.05,
    ) -> None:
        self.robot = robot
        self.device = robot.device
        try:
            self.num_envs = int(self.robot.num_envs)
        except Exception:
            self.num_envs = int(self.robot.data.joint_pos.shape[0])

        self._ee_names: list[str] = []
        self._ee_body_ids: list[int] = []
        self._ee_joint_ids: list[torch.Tensor] = []
        self._ee_joint_weights: list[torch.Tensor | None] = []
        self._ik_controllers: list[DifferentialIKController] = []

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": lambda_val},
        )

        for ee in end_effectors:
            if ee.name not in self.robot.body_names:
                raise ValueError(f"End-effector body not found: {ee.name}")
            body_id = self.robot.body_names.index(ee.name)
            joint_ids = [self.robot.joint_names.index(name) for name in ee.joint_names]
            if ee.joint_weights is not None:
                weights = torch.tensor(list(ee.joint_weights), device=self.device, dtype=torch.float32)
                if weights.numel() != len(joint_ids):
                    raise ValueError(f"joint_weights size mismatch for {ee.name}")
            else:
                weights = None

            self._ee_names.append(ee.name)
            self._ee_body_ids.append(body_id)
            self._ee_joint_ids.append(torch.tensor(joint_ids, device=self.device, dtype=torch.long))
            self._ee_joint_weights.append(weights)
            self._ik_controllers.append(
                DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)
            )

    def compute(
        self,
        target_pos_w: Dict[str, torch.Tensor],
        target_quat_w: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute joint targets for all end-effectors.

        Args:
            target_pos_w: Dict of end-effector name -> position (N, 3) in world frame.
            target_quat_w: Dict of end-effector name -> quaternion (N, 4) in world frame.

        Returns:
            Joint position targets (N, num_joints) for the full robot.
        """
        joint_pos = self.robot.data.joint_pos.clone()

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))

        for idx, ee_name in enumerate(self._ee_names):
            ee_body_id = self._ee_body_ids[idx]
            joint_ids = self._ee_joint_ids[idx]
            joint_weights = self._ee_joint_weights[idx]
            ik = self._ik_controllers[idx]

            ee_pos_w = self.robot.data.body_pos_w[:, ee_body_id]
            ee_quat_w = self.robot.data.body_quat_w[:, ee_body_id]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            des_pos_w = target_pos_w[ee_name]
            des_quat_w = target_quat_w[ee_name]
            des_pos_b, des_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, des_pos_w, des_quat_w
            )

            command = torch.cat([des_pos_b, des_quat_b], dim=-1)
            ik.set_command(command=command)

            ee_jacobi_idx = ee_body_id - 1
            jacobian_w = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]
            jacobian_b = jacobian_w.clone()
            jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

            if joint_weights is not None:
                jacobian_b = jacobian_b * joint_weights.view(1, 1, -1)

            joint_pos[:, joint_ids] = ik.compute(
                ee_pos=ee_pos_b,
                ee_quat=ee_quat_b,
                jacobian=jacobian_b,
                joint_pos=joint_pos[:, joint_ids],
            )

        return joint_pos
