"""Custom event terms for the biped environment.

Isaac Lab ships randomizers for mass, friction, COM and actuator gains, but not for the
actuator *effort limit*. That matters here: the STS3215 is modelled with a DCMotorCfg whose
``effort_limit`` is the rated continuous torque (0.98 N-m @ 12 V), and that limit is now the
binding constraint on what the robot can do. Its real value is not a constant -- it drops with
battery sag, varies unit to unit, and derates as the servo heats up.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com_absolute(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Set a random COM offset measured from the body's NOMINAL com.

    Isaac Lab's own ``randomize_rigid_body_com`` reads the *current* com and ``+=`` the sample
    (events.py:426-429). Unlike ``randomize_rigid_body_mass`` it never resets to a default, so
    under ``mode="reset"`` it compounds every episode -- a random walk, not a randomization.
    Uniform +-1 cm has sigma 5.77 mm per reset, so drift grows as 5.77mm * sqrt(N): ~3 cm by 29
    resets, ~6 cm by 100, ~12 cm by 400. On a 30 cm robot that eventually displaces the
    whole-body com by more than half a foot length, unbounded and invisible to the policy.
    (Its docstring's "use only during initialization" advice is really about this
    non-idempotency, not about the CPU round-trip.)

    This version caches the nominal com on first use and always writes nominal + offset, so
    every episode is an independent draw from a fixed distribution.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Cache the nominal com once. Must happen before any randomization has been applied.
    if not hasattr(asset, "_nominal_coms"):
        asset._nominal_coms = asset.root_physx_view.get_coms().clone()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = torch.empty((len(env_ids), 1, 3), device="cpu").uniform_(0.0, 1.0)
    rand_samples = ranges[:, 0] + rand_samples * (ranges[:, 1] - ranges[:, 0])

    coms = asset.root_physx_view.get_coms().clone()
    # Assign from nominal rather than accumulate onto the current value.
    coms[env_ids[:, None], body_ids, :3] = (
        asset._nominal_coms[env_ids[:, None], body_ids, :3] + rand_samples
    )
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_actuator_effort_limit(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float],
    operation: str = "scale",
):
    """Randomize the per-env actuator effort limit.

    ``ActuatorBase.effort_limit`` is a ``(num_envs, num_joints)`` tensor, and for explicit
    actuators (DCMotor) it is what the model clips against internally -- so writing it here is
    sufficient; no sim-side write is needed. ``effort_limit_sim`` is deliberately left alone,
    since for explicit actuators Isaac Lab parks it at 1e9 to avoid double-clipping.

    Args:
        asset_cfg: The articulation to randomize.
        distribution_params: ``(low, high)`` of the uniform distribution.
        operation: ``"scale"`` (multiply the nominal limit) or ``"abs"`` (set it directly).
            ``"scale"`` is strongly preferred -- it stays correct if the nominal torque is
            ever retuned, whereas ``"abs"`` silently decouples from it.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)

    low, high = distribution_params
    for actuator in asset.actuators.values():
        # Cache the nominal limit on first use. Without this, repeated scaling would compound
        # every reset and the limit would random-walk toward zero over training.
        if not hasattr(actuator, "_nominal_effort_limit"):
            actuator._nominal_effort_limit = actuator.effort_limit.clone()

        nominal = actuator._nominal_effort_limit[env_ids]
        samples = torch.empty_like(nominal).uniform_(low, high)
        if operation == "scale":
            actuator.effort_limit[env_ids] = nominal * samples
        elif operation == "abs":
            actuator.effort_limit[env_ids] = samples
        else:
            raise ValueError(
                f"randomize_actuator_effort_limit: unknown operation '{operation}'."
                " Expected 'scale' or 'abs'."
            )
