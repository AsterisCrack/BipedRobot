"""
src/isaaclab/env_config.py -- the ONE place config.yaml's env_config becomes a BipedEnvCfg.

WHY THIS FILE EXISTS
--------------------
train.py, play.py and common.make_env each carried their own hand-written copy of this
mapping, and the copies drifted:
  * all three copied only lin_vel_x / lin_vel_y / ang_vel_z out of the command block, so
    resampling_time_range, rel_standing_envs, heading_control_stiffness, rel_heading_envs
    and ranges.heading were silently discarded;
  * play.py ignored noise, delays, biases and every curriculum key, so playback ran a
    different env from the one the policy was trained in;
  * quantize_joint_obs was declared in the schema but never transferred anywhere.
Any key that is part of the policy contract (action map, actuator model, sensor lag) has
to reach play exactly as it reached training. Otherwise evaluation scores a robot the
policy never saw.

Rules:
  * training=True applies everything.
  * training=False (play/eval) applies everything EXCEPT pushes, physics DR and action noise, so
    playback runs the nominal robot through the identical observation/actuation pipeline.
  * An unknown key in a nested block raises, rather than being ignored. Ignored keys are
    exactly how the command settings went dead.
"""
from __future__ import annotations

import math
from typing import Any


def _as_dict(x) -> dict:
    if x is None:
        return {}
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return dict(x)


def _check_keys(block: str, given: dict, allowed: set[str]) -> None:
    unknown = set(given) - allowed
    if unknown:
        raise ValueError(f"env_config.{block}: unknown key(s) {sorted(unknown)}; allowed {sorted(allowed)}")


_COMMAND_KEYS = {"resampling_time_range", "heading_control_stiffness", "rel_standing_envs",
                 "rel_heading_envs", "rel_turn_in_place_envs", "heading_max_ang_vel", "ranges"}
_RANGE_KEYS = {"lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"}
_ACTUATOR_KEYS = {"params_file", "kp_fw", "kp_fw_range", "vin_range", "friction_scale_range", "torque_scale",
                  "tau_ext_source", "stall_torque_cap_nm"}
_BACKLASH_KEYS = {"enabled", "lash_deg_range", "impulse_cap", "mujoco_lash_deg"}
_REFERENCE_KEYS = {"npz_path", "stand_threshold", "rsi_prob", "rsi_joint_noise", "std"}

# Plain scalar/list fields copied verbatim when present.
_SCALAR_FIELDS = (
    "curriculum_enabled", "curriculum_dr_start_steps", "curriculum_dr_full_steps",
    "curriculum_cmd_ramp_steps", "curriculum_init_ramp_steps", "curriculum_dr_events",
    "action_delay_steps_range", "servo_obs_delay_steps", "imu_obs_delay_steps_range",
    "imu_accel_bias", "imu_gyro_bias", "mount_offset_rad", "imu_misalignment_deg",
    "quantize_joint_obs", "action_map",
    "standing_envs_start", "standing_envs_end", "standing_envs_ramp_steps",
    "action_rate_weight_start", "action_rate_weight_end", "action_rate_ramp_steps",
    "swing_min_clearance", "swing_max_clearance",
    "com_centering_std", "com_foot_margin",
    "reward_scale",
)


def build_env_cfg(config, env_cfg_cls, *, training: bool, num_envs: int | None = None,
                  pushes: bool | None = None):
    """Construct and fully configure a BipedEnvCfg from a loaded utils.Config.

    pushes=None follows `training` (on in training if the config enables them, off in play);
    True/False forces them, e.g. to watch push recovery in play."""
    env_conf = getattr(config.train, "env_config", None)
    # These flags are consumed by __post_init__ (the DR gate pops events there), so they
    # must go through the constructor. Assigning them afterwards would be too late.
    env_cfg = env_cfg_cls(
        use_rough_terrain=getattr(config.train, "use_rough_terrain", False),
        use_terrain_curriculum=getattr(config.train, "use_terrain_curriculum", False),
        enable_perturbations=(training if pushes is None else pushes)
                             and bool(getattr(env_conf, "enable_perturbations", True)),
        enable_physics_randomization=training and bool(
            getattr(env_conf, "enable_physics_randomization", True)),
    )

    if hasattr(config.train, "use_history"):
        env_cfg.use_history = config.train.use_history
    if hasattr(config.train, "history_size"):
        env_cfg.history_size = config.train.history_size
    if hasattr(config.train, "actor_obs"):
        env_cfg.observation_type = config.train.actor_obs
    if hasattr(config.train, "critic_obs"):
        env_cfg.critic_has_privileged_info = (config.train.critic_obs == "privileged")
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs

    if env_conf is not None:
        apply_env_config(env_cfg, env_conf, training=training)
    if not training:
        # Action noise is exploration-style DR. The robot executes the policy's action exactly.
        env_cfg.action_noise_model = None
    return env_cfg


def apply_env_config(env_cfg, env_conf, *, training: bool) -> None:
    if hasattr(env_cfg, "enable_mirroring") and hasattr(env_conf, "enable_mirroring"):
        env_cfg.enable_mirroring = env_conf.enable_mirroring

    weights = _as_dict(getattr(env_conf, "reward_weights", None))
    if weights:
        env_cfg.rewards.update(weights)

    for field in _SCALAR_FIELDS:
        if hasattr(env_conf, field) and getattr(env_conf, field) is not None:
            setattr(env_cfg, field, getattr(env_conf, field))

    if getattr(env_conf, "encoder_offset_rad", None):
        raise ValueError("env_config.encoder_offset_rad was renamed to mount_offset_rad. It is now "
                         "a per-robot constant applied to the command path as well as the "
                         "observation -- see BipedEnv.__init__")

    _apply_animation(env_cfg, env_conf)
    _apply_reference(env_cfg, env_conf, training=training)
    _apply_commands(env_cfg, env_conf)
    _apply_actuator(env_cfg, env_conf)
    _apply_noise(env_cfg, env_conf)

    if getattr(env_conf, "mirror_joint_indices", None):
        env_cfg.mirror_joint_indices = env_conf.mirror_joint_indices
        env_cfg.mirror_action_indices = env_conf.mirror_joint_indices

    events = _as_dict(getattr(env_conf, "events", None))
    for key, val in events.items():
        if key in env_cfg.events:
            if "params" in val:
                env_cfg.events[key].params.update(val["params"])
            if "interval_range_s" in val:
                env_cfg.events[key].interval_range_s = tuple(val["interval_range_s"])


def _apply_animation(env_cfg, env_conf) -> None:
    anim = _as_dict(getattr(env_conf, "animation", None))
    for src, dst in (("npz_path", "animation_npz_path"), ("loop", "animation_loop"),
                     ("random_start", "animation_random_start"), ("speed", "animation_speed"),
                     ("pos_std", "animation_pos_std"), ("vel_std", "animation_vel_std")):
        if src in anim:
            setattr(env_cfg, dst, anim[src])


def _apply_reference(env_cfg, env_conf, *, training: bool) -> None:
    """env_config.reference -> reference_* fields. The gait clock is a policy input, so the table
    reaches train and play identically (both go through build_env_cfg). RSI is training-only:
    playback starts from the default pose, like the real robot."""
    ref = _as_dict(getattr(env_conf, "reference", None))
    if not ref:
        return
    _check_keys("reference", ref, _REFERENCE_KEYS)
    for src, dst in (("npz_path", "reference_npz_path"), ("stand_threshold", "reference_stand_threshold"),
                     ("rsi_prob", "reference_rsi_prob"), ("rsi_joint_noise", "reference_rsi_joint_noise")):
        if src in ref:
            setattr(env_cfg, dst, ref[src])
    std = ref.get("std") or {}
    unknown = set(std) - set(env_cfg.reference_std)
    if unknown:
        raise ValueError(f"env_config.reference.std: unknown key(s) {sorted(unknown)}; "
                         f"allowed {sorted(env_cfg.reference_std)}")
    env_cfg.reference_std = {**env_cfg.reference_std, **std}
    if not training:
        env_cfg.reference_rsi_prob = 0.0


def _apply_commands(env_cfg, env_conf) -> None:
    cmds = _as_dict(getattr(env_conf, "commands", None))
    if not cmds:
        return
    _check_keys("commands", cmds, {"base_velocity"})
    bv = cmds.get("base_velocity") or {}
    _check_keys("commands.base_velocity", bv, _COMMAND_KEYS)
    target = env_cfg.commands["base_velocity"]
    for key in _COMMAND_KEYS - {"ranges"}:
        if key in bv:
            val = bv[key]
            target[key] = tuple(val) if isinstance(val, (list, tuple)) else val
    ranges = bv.get("ranges") or {}
    _check_keys("commands.base_velocity.ranges", ranges, _RANGE_KEYS)
    for key, val in ranges.items():
        target["ranges"][key] = tuple(val)


def _apply_actuator(env_cfg, env_conf) -> None:
    """actuator/backlash blocks -> the robot's BamActuatorCfg and its startup DR event."""
    act = _as_dict(getattr(env_conf, "actuator", None))
    lash = _as_dict(getattr(env_conf, "backlash", None))
    _check_keys("actuator", act, _ACTUATOR_KEYS)
    _check_keys("backlash", lash, _BACKLASH_KEYS)
    actuators = getattr(env_cfg.scene.robot, "actuators", {}) or {}
    bam_cfgs = [c for c in actuators.values() if hasattr(c, "params_file")]
    if not bam_cfgs:
        if act or lash:
            raise ValueError("env_config.actuator/backlash given, but this robot has no BAM actuator")
        return

    for cfg in bam_cfgs:
        for key in _ACTUATOR_KEYS:
            if key in act:
                val = act[key]
                setattr(cfg, key, tuple(val) if isinstance(val, list) else val)
        if lash:
            cfg.backlash_enabled = bool(lash.get("enabled", False))
            lo_deg, hi_deg = lash.get("lash_deg_range", (0.0, 0.0))
            # Nominal = mid range: used when physics DR is off (e.g. play), since the startup
            # event that draws per-joint values is popped by the DR gate then.
            cfg.backlash_nominal_rad = math.radians(0.5 * (lo_deg + hi_deg)) if cfg.backlash_enabled else 0.0
            if "impulse_cap" in lash:
                cfg.backlash_impulse_cap = float(lash["impulse_cap"])

    ev = env_cfg.events.get("randomize_bam_actuator")
    if ev is not None:
        first = bam_cfgs[0]
        ev.params["vin_range"] = tuple(first.vin_range)
        ev.params["friction_scale_range"] = tuple(first.friction_scale_range)
        if first.backlash_enabled:
            lo_deg, hi_deg = lash.get("lash_deg_range", (0.0, 0.0))
            ev.params["lash_range_rad"] = (math.radians(lo_deg), math.radians(hi_deg))
        else:
            ev.params["lash_range_rad"] = None


def _apply_noise(env_cfg, env_conf) -> None:
    obs_noise = _as_dict(getattr(env_conf, "observation_noise_std", None))
    if obs_noise:
        unknown = set(obs_noise) - set(env_cfg.observation_noise_std)
        if unknown:
            raise ValueError(f"unknown observation_noise_std channels: {sorted(unknown)}; "
                             f"expected a subset of {sorted(env_cfg.observation_noise_std)}")
        env_cfg.observation_noise_std = {**env_cfg.observation_noise_std, **obs_noise}

    act_noise = _as_dict(getattr(env_conf, "action_noise_model", None))
    if act_noise and env_cfg.action_noise_model and env_cfg.action_noise_model.noise_cfg:
        if "mean" in act_noise:
            env_cfg.action_noise_model.noise_cfg.mean = act_noise["mean"]
        if "std" in act_noise:
            env_cfg.action_noise_model.noise_cfg.std = act_noise["std"]


def describe(env_cfg) -> dict[str, Any]:
    """The contract-relevant settings, for a startup print (and to diff train vs play)."""
    bv = env_cfg.commands["base_velocity"]
    out = {k: bv.get(k) for k in sorted(_COMMAND_KEYS - {"ranges"})}
    out["ranges"] = dict(bv.get("ranges", {}))
    for field in ("action_map", "mount_offset_rad", "imu_misalignment_deg", "servo_obs_delay_steps",
                  "imu_obs_delay_steps_range", "action_delay_steps_range", "quantize_joint_obs",
                  "enable_perturbations", "enable_physics_randomization"):
        out[field] = getattr(env_cfg, field, None)
    for name, c in (getattr(env_cfg.scene.robot, "actuators", {}) or {}).items():
        if hasattr(c, "params_file"):
            out[f"actuator.{name}"] = {k: getattr(c, k) for k in
                                       ("params_file", "kp_fw", "kp_fw_range", "vin_range", "friction_scale_range",
                                        "torque_scale", "stall_torque_cap_nm", "tau_ext_source",
                                        "backlash_enabled",
                                        "backlash_nominal_rad")}
    return out
