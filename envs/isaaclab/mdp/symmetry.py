"""Functions to specify the symmetry in the observation and action space for Biped Robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # DirectRLEnv is the base class for BipedEnv
    from isaaclab.envs import DirectRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: DirectRLEnv,
    obs: dict[str, torch.Tensor] | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    left-right symmetry transformations. The symmetry transformations are beneficial for 
    reinforcement learning tasks by providing additional diverse data without requiring 
    additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        # We handle dictionary observations (policy, critic)
        obs_aug = {}
        batch_size = 0
        
        for key, val in obs.items():
            if val is None:
                obs_aug[key] = None
                continue
                
            batch_size = val.shape[0]
            # Augment batch size by 2 (Original, Left-Right)
            # Unlike quadruped, we don't do Front-Back symmetry
            current_obs_aug = val.repeat(2, 1) if val.dim() > 1 else val.repeat(2)

            # -- original (Already set by repeat, but 0:batch_size is original)
            
            # -- left-right
            # Apply transformation to the second half of the batch
            # We assume the policy/critic obs follow the standard proprioceptive layout
            # defined in BipedEnv._get_observations
            if key == "policy" or key == "critic":
                # Only transform if dimensions match expected proprioception (48) + optional privilege
                # If privilege is appended at the end, we need to handle it or ignore it dependent on structure
                # For now, we apply proprio transformation to the first 48 dims
                
                proprio_dim = 48
                if current_obs_aug.shape[1] >= proprio_dim:
                    transformed_proprio = _transform_proprio_obs_left_right(
                        current_obs_aug[batch_size:, :proprio_dim]
                    )
                    current_obs_aug[batch_size:, :proprio_dim] = transformed_proprio
                    
                    # TODO: Handle privileged info symmetry if needed
                    # Privileged info (contacts) also needs swapping
            
            obs_aug[key] = current_obs_aug
            
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # Augment batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        
        # -- original
        actions_aug[:batch_size] = actions[:]
        
        # -- left-right
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_proprio_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    Args:
        obs: The observation tensor to be transformed (assumed shape [N, 48]).

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    
    # Indices based on BipedEnv._get_observations:
    # 0-2: base_lin_vel_b
    # 3-5: base_ang_vel_b
    # 6-8: projected_gravity_b
    # 9-11: commands
    # 12-23: joint_pos
    # 24-35: joint_vel
    # 36-47: previous_actions
    
    # lin vel (x, y, z) -> (x, -y, z)
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    
    # ang vel (x, y, z) -> (-x, y, -z) 
    # Roll is inverted, Pitch is same, Yaw is inverted
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    
    # projected gravity (x, y, z) -> (x, -y, z)
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    
    # velocity command (vx, vy, wz) -> (vx, -vy, -wz)
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    
    # joint pos
    obs[:, 12:24] = _switch_biped_joints_left_right(obs[:, 12:24])
    
    # joint vel
    obs[:, 24:36] = _switch_biped_joints_left_right(obs[:, 24:36])
    
    # last actions
    obs[:, 36:48] = _switch_biped_joints_left_right(obs[:, 36:48])

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_biped_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.
"""

def _switch_biped_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """
    Switches left and right joints in the given joint data tensor for the biped robot.
    """
    joint_data_switched = torch.zeros_like(joint_data)
    
    # Indices 0, 2, 4, 6, 8, 10 are one side
    # indices 1, 3, 5, 7, 9, 11 are the other side
    
    # Indices
    side_a_indices = [0, 2, 4, 6, 8, 10]
    side_b_indices = [1, 3, 5, 7, 9, 11]
    
    # Swap
    joint_data_switched[..., side_b_indices] = joint_data[..., side_a_indices]
    joint_data_switched[..., side_a_indices] = joint_data[..., side_b_indices]

    # Flip Signs
    flip_indices = [0, 1, 2, 3, 10, 11]
    
    joint_data_switched[..., flip_indices] *= -1.0

    return joint_data_switched
