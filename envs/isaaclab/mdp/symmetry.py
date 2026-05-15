"""Left-right symmetry augmentation for BipedEnv observations and actions.

All robot-specific joint permutations and sign conventions live in the env's
`mirror_joint_perm` / `mirror_joint_signs` cfg fields and are pre-compiled in
`BipedEnv._build_mirror_transform()`.  This module simply delegates to those
already-correct, robot-specific methods so there is a single source of truth.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: "DirectRLEnv",
    obs: dict[str, torch.Tensor] | None = None,
    actions: torch.Tensor | None = None,
):
    """Return left-right mirrored copies of *obs* and *actions*.

    Delegates to ``env.mirror_obs`` / ``env.mirror_action`` so the transform is
    always consistent with the robot loaded in the environment.

    Args:
        env:     The BipedEnv instance (must have ``mirror_obs`` / ``mirror_action``).
        obs:     Observation dict (``"policy"`` and/or ``"critic"`` keys).
        actions: Action tensor of shape ``(N, 12)``.

    Returns:
        Tuple ``(obs_mirrored, actions_mirrored)``.  Either element is ``None``
        if the corresponding input was ``None``.
    """
    if not (hasattr(env, "mirror_obs") and hasattr(env, "mirror_action")):
        raise AttributeError(
            "compute_symmetric_states requires the environment to expose "
            "mirror_obs() and mirror_action() methods (available in BipedEnv)."
        )

    obs_aug     = env.mirror_obs(obs)     if obs     is not None else None
    actions_aug = env.mirror_action(actions) if actions is not None else None
    return obs_aug, actions_aug


# ---------------------------------------------------------------------------
# Low-level helpers — accept explicit perm/sign tensors so callers outside
# the env (e.g. try_symmetry.py) can still use them after fetching the
# tensors from the env.
# ---------------------------------------------------------------------------

def switch_joints_left_right(
    joint_data: torch.Tensor,
    joint_perm: torch.Tensor,
    joint_signs: torch.Tensor,
) -> torch.Tensor:
    """Swap left↔right joints and apply reflection sign flips.

    Args:
        joint_data:  ``(..., 12)`` tensor of joint positions, velocities, or actions.
        joint_perm:  ``(12,)`` long tensor — permutation index for L↔R swap
                     (from ``env._mirror_action_perm``).
        joint_signs: ``(12,)`` float tensor — sign multipliers
                     (from ``env._mirror_action_signs``).

    Returns:
        Transformed tensor of the same shape.
    """
    return joint_data[..., joint_perm] * joint_signs


def transform_actions_left_right(
    actions: torch.Tensor,
    joint_perm: torch.Tensor,
    joint_signs: torch.Tensor,
) -> torch.Tensor:
    """Mirror a batch of actions left-right.

    Args:
        actions:     ``(N, 12)`` action tensor.
        joint_perm:  from ``env._mirror_action_perm``.
        joint_signs: from ``env._mirror_action_signs``.
    """
    return switch_joints_left_right(actions, joint_perm, joint_signs)
