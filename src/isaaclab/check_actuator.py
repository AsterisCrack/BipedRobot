"""
src/isaaclab/check_actuator.py -- startup checks for the BAM servo model and the new env pieces.

Run after any change to the actuator, the action map or the sim2real config, and BEFORE
spending a training run on it. It builds the env through the same builder as train.py, has
every env stand still under a zero command, and reports a PASS/FAIL line for each of:

  1. the actuator really is BamActuator, and which params file it loaded
  2. whether PhysX's mass-matrix diagonal includes armature (decides inertia_eff)
  3. action map: a = 0 reproduces the default pose exactly
  4. mount offset sign: the ENCODER tracks the target, so the TRUE joint sits displaced by -d
  5. external-torque observer: standing still, I*qdd = tau_ext + tau_applied should be ~0
  6. the resting foot-origin height (the swing_foot_height datum)
  7. the sampled servo population (vin, friction scale, lash) and duty saturation
  8. it stays upright and finite

Usage (Isaac Lab python, from the BipedRobot root):
    python src/isaaclab/check_actuator.py --headless
    python src/isaaclab/check_actuator.py --headless --num_envs 64 --seconds 4
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the BAM actuator and sim2real env pieces")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--config_path", type=str, default="config/config.yaml")
parser.add_argument("--no-backlash", action="store_true", help="disable lost-motion backlash (A/B)")
parser.add_argument("--tau-ext", choices=["observer", "gravity"], default=None,
                    help="override the external-torque source (A/B the observer)")
parser.add_argument("--legacy-dcmotor", action="store_true",
                    help="baseline: swap in the pre-BAM DCMotorCfg (kp 20, kd 2, 0.98 N*m) and only "
                         "report the standing metrics")
parser.add_argument("--friction-scale", type=float, default=None,
                    help="force every servo's friction scale (0 = friction off, A/B)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Kit puts two directories named `config` on sys.path (omni.isaac.kit and
# isaacsim.simulation_app), and together they form a NAMESPACE package that shadows
# BipedRobot/config. The repo root must sit at the FRONT of sys.path even if it is already
# on it further down, which is why this is a remove-then-insert and not "if not in".
for p in (os.path.join(_ROOT, "torch_rl_algorithms"), _ROOT):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
sys.modules.pop("config", None)

import math     # noqa: E402

import torch    # noqa: E402

from envs.isaaclab.biped_env import BipedEnv                      # noqa: E402
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg         # noqa: E402
from src.isaaclab.env_config import build_env_cfg, describe       # noqa: E402
from utils import Config                                           # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    RESULTS.append((bool(ok), msg))
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")


def main() -> None:
    config = Config(os.path.join(_ROOT, args_cli.config_path))
    freeze_residual_map(config, _ROOT, warn_unfrozen=True)
    # Measure the MODEL, not the curriculum: flat ground, no pushes, no random initial joint
    # offsets. Everything servo-side (the BAM population, lash, mount offset, IMU
    # misalignment, sensor lag) stays exactly as in training. A robot holding zero action
    # under pushes simply falls over, and then none of the standing-still checks below
    # hold.
    config.train.use_rough_terrain = False
    if args_cli.tau_ext:
        config.train.env_config.actuator = {**config.train.env_config.actuator,
                                            "tau_ext_source": args_cli.tau_ext}
    if args_cli.friction_scale is not None:
        fs = float(args_cli.friction_scale)
        config.train.env_config.actuator = {**config.train.env_config.actuator,
                                            "friction_scale_range": [fs, fs]}
    if args_cli.no_backlash and config.train.env_config.backlash:
        config.train.env_config.backlash = {**config.train.env_config.backlash, "enabled": False}
    cfg = build_env_cfg(config, BipedRobotV2EnvCfg, training=True, num_envs=args_cli.num_envs)
    cfg.events.pop("push_robot", None)
    cfg.curriculum_init_range_min = cfg.curriculum_init_range_max = 0.0
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    if args_cli.legacy_dcmotor:
        from isaaclab.actuators import DCMotorCfg
        cfg.scene.robot.actuators["legs"] = DCMotorCfg(
            joint_names_expr=[".*"], stiffness=20.0, damping=2.0, armature=0.04, friction=0.2,
            effort_limit=0.98, velocity_limit=4.71, saturation_effort=2.94)
        cfg.events.pop("randomize_bam_actuator", None)
    print("\n[check] env contract:", describe(cfg), "\n")
    env = BipedEnv(cfg=cfg)
    env.reset()
    n, J = env.num_envs, env.num_joints

    # 1. actuator
    acts = env.bam_actuators
    if args_cli.legacy_dcmotor:
        _standing_baseline(env, n, J)
        return
    check(len(acts) == 1, f"one BamActuator attached (found {len(acts)})")
    if not acts:
        return
    act = acts[0]
    eff = act.servo.kernel.torque_scale[0, 0].item()
    print(f"        params: {act.params.source} ({act.params.model}), duty-limited stall @12V = "
          f"{act.params.stall_torque:.2f} N*m raw, {act.params.stall_torque * eff:.2f} N*m effective "
          f"(cap {act.cfg.stall_torque_cap_nm}, torque_scale {act.cfg.torque_scale})")

    # 3. action map (before any stepping)
    zero_t = env.actions_to_targets(torch.zeros(n, J, device=env.device))
    err = (zero_t - env.default_joint_pos).abs().max().item()
    check(err < 1e-6, f"action map '{env.action_map}': a = 0 -> default pose (max err {err:.2e} rad)")
    one_t = env.actions_to_targets(torch.ones(n, J, device=env.device))
    if env.action_map == "residual":
        want = torch.minimum(env.default_joint_pos + env.action_scale, env.joint_limits_max)
        check((one_t - want).abs().max().item() < 1e-6, "a = +1 -> min(home + scale, upper joint limit)")
        neg_scale = env.action_scale if env.action_scale_neg is None else env.action_scale_neg
        neg_t = env.actions_to_targets(-torch.ones(n, J, device=env.device))
        want_neg = torch.maximum(env.default_joint_pos - neg_scale, env.joint_limits_min)
        check((neg_t - want_neg).abs().max().item() < 1e-6, "a = -1 -> max(home - neg scale, lower joint limit)")
    else:
        check((one_t - env.joint_limits_max).abs().max().item() < 1e-6, "a = +1 -> upper joint limits")

    # Stand still: zero command everywhere, no heading control, no resampling.
    gen = env.command_generator
    gen.rel_turn_in_place_envs = 0.0
    steps = int(args_cli.seconds / env.step_dt)
    resid, ext, sat, tilt_max = [], [], [], 0.0
    resid_j = torch.zeros(J, device=env.device)
    dq_prev, flips = None, []
    q_hist = []
    for _ in range(steps):
        gen.time_left[:] = 1e9
        gen.in_heading_mode[:] = False
        env.commands[:] = 0.0
        env.step(torch.zeros(n, J, device=env.device))
        qdd = env.robot.data.joint_acc
        resid.append((act.inertia_eff * qdd).abs().mean().item())
        resid_j += (act.inertia_eff * qdd).abs().mean(dim=0)
        dq = env.robot.data.joint_vel
        if dq_prev is not None:
            flips.append(((dq * dq_prev) < 0).float().mean().item())   # velocity sign reversals
        dq_prev = dq.clone()
        q_hist.append(env.robot.data.joint_pos.clone())
        ext.append(act.tau_ext.abs().mean().item())
        sat.append(act.duty_saturated.float().mean().item())
        g = env.robot.data.projected_gravity_b
        tilt_max = max(tilt_max, math.degrees(torch.acos((-g[:, 2]).clamp(-1, 1)).max().item()))

    # 2. armature probe result
    check(act._armature_in_mass_matrix is not None,
          f"armature probe ran: mass-matrix diagonal "
          f"{'INCLUDES' if act._armature_in_mass_matrix else 'EXCLUDES'} armature")
    print("        inertia_eff env0:", [round(v, 4) for v in act.inertia_eff[0].tolist()])

    # 4. mount offset: the encoder (q + d) tracks the encoder-space target, so the true joint
    # sits displaced by -d. Regress (q - t) on -d: the slope should be ~1 for the right sign
    # and ~-1 for a flipped one. Backlash scatters the residual by up to +-lash/2, which is
    # why this uses a slope, not a tight per-joint tolerance.
    q = env.robot.data.joint_pos
    if getattr(env, "_servo_side_encoder", False):
        # backlash.encoder_side servo: the firmware regulates the servo SHAFT, so that is what tracks t - d; the
        # link scatters around it by up to +-lash/2 (comparable to d) and would bias the slope low.
        q = q.clone()
        for act in env.bam_actuators:
            ids = act._joint_ids()
            q[:, ids] = act.encoder_reading(q[:, ids], env.robot.data.joint_vel[:, ids])[0]
    d = env.mount_offset
    t = env.filtered_targets
    enc_err = ((q + d) - t).abs().mean().item()
    true_err = (q - t).abs().mean().item()
    if env.mount_offset_range > 0:
        x, y = (-d).flatten(), (q - t).flatten()
        slope = (torch.dot(x, y) / torch.dot(x, x)).item()
        check(0.6 < slope < 1.4 and enc_err < true_err,
              f"mount offset sign: slope of (q - t) on -d = {slope:.2f} (want ~1, flipped = -1); "
              f"|enc - t| {math.degrees(enc_err):.3f} deg < |q - t| {math.degrees(true_err):.3f} deg")
    else:
        print("        mount offset disabled; sign check skipped")

    # 5. observer self-consistency while standing
    late = slice(len(resid) // 2, None)
    r_mean = sum(resid[late]) / len(resid[late])
    e_mean = sum(ext[late]) / len(ext[late])
    # Standing still is judged on POSITION, against the encoder's resolution. |I*qdd| looks
    # alarming but qdd amplifies high frequencies: 0.08 N*m at I ~ 0.025 is a ~15 Hz
    # micro-vibration of hundredths of a degree, invisible to a 0.088 deg encoder. The
    # first version of this check used that ratio and flagged harmless contact jitter.
    # Isaac has a CONTACT-SOLVER JITTER FLOOR that this check has to sit above. Measured
    # 2026-10-02 on flat ground, 16 envs, zero action (run with --legacy-dcmotor to re-measure):
    #     legacy DCMotor (kp 20, kd 2)   0.106 deg mean, 0.268 worst   <- the floor
    #     BAM, no backlash               0.122 deg mean, 0.292 worst   (+15%, servo-model noise)
    #     BAM + 0.5-2.5 deg backlash     0.157 deg mean, 0.390 worst   (free play, physical)
    # So the bar is ~2.5 encoder LSB: it passes all three, and real servo chatter, which
    # runs to whole degrees, still fails it by an order of magnitude.
    lsb_deg = 360.0 / 4096.0
    qh = torch.stack(q_hist[len(q_hist) // 2:])                  # (T, N, J), last half
    jitter_deg = torch.rad2deg(qh.std(dim=0))                   # per env, per joint
    check(jitter_deg.mean().item() < 2.5 * lsb_deg,
          f"standing still: joint position jitter {jitter_deg.mean().item():.4f} deg mean, "
          f"{jitter_deg.max().item():.4f} deg worst (bar {2.5 * lsb_deg:.3f} deg = 2.5 encoder LSB; "
          f"Isaac's legacy-actuator floor is ~0.106)")
    print(f"        observer (info): standing |I*qdd| = {r_mean:.4f} N*m vs |tau_ext| = {e_mean:.4f} N*m")
    print("        per-joint |I*qdd| (N*m):",
          ", ".join(f"{nm.replace('_joint', '')}={v:.3f}" for nm, v in
                    zip(env.robot.joint_names, (resid_j / steps).tolist())))
    f_late = flips[len(flips) // 2:]
    print(f"        joint velocity sign reversals per policy step: {sum(f_late) / max(1, len(f_late)) * 100:.0f}% "
          f"of joints (chatter indicator; a settled robot is near 0)")
    print(f"        mean |joint vel| at end: {env.robot.data.joint_vel.abs().mean().item():.4f} rad/s")

    # 6. swing datum
    feet = env.robot.data.body_pos_w[:, env.feet_indices, 2] - env.scene.env_origins[:, 2:3]
    in_c = torch.norm(env.contact_sensor.data.net_forces_w[:, env._feet_ids], dim=-1) > 1.0
    if in_c.any():
        print(f"        resting foot-origin height = {feet[in_c].mean().item() * 1000:.1f} mm above "
              f"terrain (swing window: +{cfg.swing_min_clearance * 1000:.0f}..+{cfg.swing_max_clearance * 1000:.0f} mm above it)")

    # 7. population + saturation
    k = act.servo.kernel
    print(f"        vin [{k.vin.min().item():.2f}, {k.vin.max().item():.2f}] V, "
          f"friction_scale [{k.friction_scale.min().item():.2f}, {k.friction_scale.max().item():.2f}]")
    lm = act.servo.lost_motion
    if lm is not None:
        print(f"        lash [{math.degrees(lm.lash.min().item()):.2f}, {math.degrees(lm.lash.max().item()):.2f}] deg, "
              f"free play now {(~lm.contact).float().mean().item() * 100:.0f}% of joints")
    s_mean = sum(sat[late]) / len(sat[late])
    check(s_mean < 0.05, f"standing duty saturation {s_mean * 100:.1f}% of joints (want ~0)")

    # 8. sanity
    finite = torch.isfinite(env.robot.data.joint_pos).all().item()
    check(finite, "joint state finite")
    g = env.robot.data.projected_gravity_b
    tilt_now = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
    check(tilt_max < 20.0, f"stays upright standing with zero action: max tilt {tilt_max:.1f} deg, "
                           f"{int((tilt_now > 20).sum())}/{n} envs above 20 deg at the end")

    # 9. gait diagnostics: standing values, plus SIGN checks. Each angle convention is
    # checked against a known input, not trusted from the derivation.
    from envs.isaaclab.rewards import rewards as R
    import isaaclab.utils.math as mu
    feet = env.robot.data.body_pos_w[:, env.feet_indices]
    in_c = torch.norm(env.contact_sensor.data.net_forces_w[:, env._feet_ids], dim=-1) > 1.0
    off = R.com_support_offset(env.robot.data.body_com_pose_w,
                               env.robot.data.default_mass.to(env.device), feet, in_c)
    dist = R.com_support_distance(env.robot.data.body_com_pose_w,
                                  env.robot.data.default_mass.to(env.device), feet, in_c,
                                  foot_margin=float(getattr(env.cfg, "com_foot_margin", 0.03)))
    d = {k: v.mean().item() for k, v in env.gait_diagnostics(off, dist).items()}
    print(f"        standing: com_ahead {d['com_ahead_m'] * 1000:+.1f} mm, com_lateral "
          f"{d['com_lateral_abs_m'] * 1000:.1f} mm, outside-polygon {d['com_outside_mm']:.1f} mm, "
          f"lean {d['lean_fwd_deg']:+.2f} deg, toe_in {d['toe_in_deg']:+.2f} deg")
    check(abs(d["com_ahead_m"]) < 0.03, f"standing CoM within 3 cm of support fore/aft ({d['com_ahead_m'] * 1000:+.1f} mm)")
    check(d["com_outside_mm"] < 10.0, f"standing CoM inside the support polygon ({d['com_outside_mm']:.1f} mm outside)")
    # lean: pure maths on a known rotation. +10 deg about body +Y must put the nose DOWN
    # (body x has negative world z) and read lean_fwd = +10.
    q = mu.quat_from_angle_axis(torch.tensor([math.radians(10.0)]), torch.tensor([[0.0, 1.0, 0.0]]))
    nose_z = mu.quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))[0, 2].item()
    g_b = mu.quat_apply_inverse(q, torch.tensor([[0.0, 0.0, -1.0]]))
    lean = math.degrees(math.asin(g_b[0, 0].item()))
    check(nose_z < 0 and abs(lean - 10.0) < 1e-3, f"lean sign: +10 deg about +Y is nose-down (nose z {nose_z:+.3f}) and reads {lean:+.2f}")
    # toe-in: command both hip yaws to +0.3 action (centered map, + side). That is +q on
    # both, i.e. toes OUT, so toe_in_deg must go clearly negative.
    yaw = lambda qq: torch.atan2(2 * (qq[..., 0] * qq[..., 3] + qq[..., 1] * qq[..., 2]),
                                 1 - 2 * (qq[..., 2] ** 2 + qq[..., 3] ** 2))
    # The foot link frames are not upright (Fusion export), so yaw-from-quaternion of a foot
    # frame is not its rotation about vertical. Instead, take the world-frame rotation BETWEEN
    # two poses, dq = q_now * conj(q_start). That is a near-pure vertical rotation whatever the
    # frame's orientation, so its yaw is the actual turn.
    qf0 = env.robot.data.body_quat_w[:, env.feet_indices].clone()
    qb0 = env.robot.data.root_quat_w.clone()
    act_vec = torch.zeros(n, J, device=env.device)
    act_vec[:, env.hip_yaw_lr] = 0.3
    for _ in range(40):
        gen.time_left[:] = 1e9
        gen.in_heading_mode[:] = False
        env.commands[:] = 0.0
        env.step(act_vec)
    feet = env.robot.data.body_pos_w[:, env.feet_indices]
    off = R.com_support_offset(env.robot.data.body_com_pose_w,
                               env.robot.data.default_mass.to(env.device), feet, in_c)
    toe = env.gait_diagnostics(off)["toe_in_deg"].mean().item()
    # Cross-check against the feet's world positions: toes out means the feet's forward
    # axes splay apart, so the foot frames' x axes diverge (left yaws +, right yaws -).
    turn = lambda q1, q0: torch.rad2deg(yaw(mu.quat_mul(q1, mu.quat_conjugate(q0))))
    d_foot = turn(env.robot.data.body_quat_w[:, env.feet_indices], qf0)           # [right, left]
    d_base = turn(env.robot.data.root_quat_w, qb0).unsqueeze(-1)
    d_rel = d_foot - d_base
    splay = (d_rel[:, 1] - d_rel[:, 0]).mean().item()    # + = toes apart (out)
    print(f"        toe-out test: hip yaw {-toe:+.1f} deg, foot turn rel. base L {d_rel[:, 1].mean().item():+.1f} / "
          f"R {d_rel[:, 0].mean().item():+.1f} deg")
    check(toe < -5.0 and splay > 10.0,
          f"toe-in sign: hip yaw +0.3 action reads toe_in {toe:+.1f} deg (want < -5) "
          f"and the feet physically splay out by {splay:+.1f} deg (want > +10)")

    bad = [m for ok, m in RESULTS if not ok]
    print(f"\n[check] {len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + ("" if not bad else ":"))
    for m in bad:
        print(f"   FAIL  {m}")
    env.close()


def _standing_baseline(env, n, J) -> None:
    """Standing jitter and tilt only, for an actuator without the BAM diagnostics."""
    gen = env.command_generator
    gen.rel_turn_in_place_envs = 0.0
    steps = int(args_cli.seconds / env.step_dt)
    q_hist, tilt_max = [], 0.0
    for _ in range(steps):
        gen.time_left[:] = 1e9
        gen.in_heading_mode[:] = False
        env.commands[:] = 0.0
        env.step(torch.zeros(n, J, device=env.device))
        q_hist.append(env.robot.data.joint_pos.clone())
        g = env.robot.data.projected_gravity_b
        tilt_max = max(tilt_max, math.degrees(torch.acos((-g[:, 2]).clamp(-1, 1)).max().item()))
    qh = torch.stack(q_hist[len(q_hist) // 2:])
    jit = torch.rad2deg(qh.std(dim=0))
    print(f"  BASELINE (legacy DCMotor) standing jitter {jit.mean().item():.4f} deg mean, "
          f"{jit.max().item():.4f} deg worst; max tilt {tilt_max:.1f} deg")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
