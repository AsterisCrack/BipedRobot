"""foot_slip / foot_drag (E25): contact-point slip and scuffing penalties (CPU, no Isaac). Run from BipedRobot/:
    python -m pytest tests/test_foot_slip_drag.py -q

Why: feet_slide penalised the speed of the foot COM frame of every foot in (soft) contact. With the gait table's heel-toe
rocker that frame moves 2-5x faster than the sole slips (MuJoCo, run gaitref_2026-10-22_18-48-09: 76-79 mm/s measured vs
17-29 mm/s true contact-point slip), so the term fought the reference. foot_slip is the planar speed of the sole at the
contact (lowest sole corners, rolling excluded); foot_drag is the same speed on a low UNLOADED foot.
"""
import math
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

MJCF = os.path.join(ROOT, "envs", "assets", "robotV2", "robot_v2.xml")
OFF = [0.0163, 0.0211, -0.0316]       # == cfg.foot_sole_offset_m
HALF = [0.0450, 0.0186]               # == cfg.foot_sole_half_extents_m


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _t(x):
    return torch.as_tensor(np.asarray(x, np.float64))


def _rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat_axis(axis, ang):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(ang / 2)], a * math.sin(ang / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _slip(pos, quat, vel, ang):
    """(slip (F,), z_low (F,)) for ONE env of F = 2 feet, [right, left]."""
    s, z = R.foot_sole_slip_speed(_t(pos)[None], _t(quat)[None], _t(vel)[None], _t(ang)[None], OFF, HALF)
    return s[0].numpy(), z[0].numpy()


def _corners_np(p, q, sign):
    ox, oy, oz = OFF
    loc = np.array([[ox + sx * HALF[0], sign * oy + sy * HALF[1], oz] for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return p + loc @ _rotmat(q).T


ID = np.array([1.0, 0.0, 0.0, 0.0])
FLAT_Z = -OFF[2]                      # origin height of a foot lying flat on the floor


# ----------------------------------------------------------------------------- the speed function
@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_flat_foot_translating_gives_its_speed(yaw):
    v = np.array([0.06, -0.08, 0.0])
    q = _quat_axis([0, 0, 1], yaw)
    pos = np.array([[0.0, -0.05, FLAT_Z], [0.0, 0.05, FLAT_Z]])
    s, z = _slip(pos, np.stack([q, q]), np.stack([v, v]), np.zeros((2, 3)))
    assert np.allclose(s, np.linalg.norm(v[:2]), atol=1e-9)
    assert np.allclose(z, 0.0, atol=1e-9)                       # the sole is on the floor


def test_vertical_motion_is_not_slip():
    pos = np.array([[0.0, -0.05, FLAT_Z], [0.0, 0.05, FLAT_Z]])
    s, _ = _slip(pos, np.stack([ID, ID]), np.array([[0.0, 0.0, 0.3]] * 2), np.zeros((2, 3)))
    assert np.allclose(s, 0.0, atol=1e-9)


@pytest.mark.parametrize("edge", ["heel", "toe"])
def test_rolling_about_an_edge_without_slip_is_not_slip_but_origin_speed_is(edge):
    """The foot pitches about its heel (toe) edge, which stays still on the floor: the sole does not slip, but the link origin
    (31.6 mm above the sole) moves at omega * ~32 mm. The weighting on the lowest corners must read ~0."""
    sign_x = -1.0 if edge == "heel" else +1.0
    theta, omega = (-0.10, -1.0) if edge == "heel" else (+0.10, +1.0)   # heel roll lifts the toe: pitch about +y is negative
    q = _quat_axis([0, 1, 0], theta)
    Rm = _rotmat(q)
    sign = [-1.0, 1.0]
    pos, vel, ang, quat, origin_speed = [], [], [], [], []
    for k in range(2):
        pivot_local = np.array([OFF[0] + sign_x * HALF[0], sign[k] * OFF[1], OFF[2]])
        pivot = np.array([0.1 * k, (-1 + 2 * k) * 0.05, 0.0])             # world point on the floor, at rest
        p = pivot - Rm @ pivot_local                                      # origin position
        w = np.array([0.0, omega, 0.0])
        v = np.cross(w, p - pivot)                                        # the pivot has zero velocity
        pos.append(p); vel.append(v); ang.append(w); quat.append(q); origin_speed.append(np.linalg.norm(v[:2]))
    s, z = _slip(np.array(pos), np.array(quat), np.array(vel), np.array(ang))
    assert np.all(np.array(origin_speed) > 0.02), origin_speed          # the old origin-speed measure sees "sliding"
    assert np.all(s < 5e-4), s                                           # < 0.5 mm/s: the sole is not slipping
    assert np.all(np.abs(z) < 1e-9)                                      # lowest corner = the pivot edge, on the floor


def test_rolling_while_the_edge_really_slides_is_slip():
    theta, omega = -0.10, -1.0
    q = _quat_axis([0, 1, 0], theta)
    Rm = _rotmat(q)
    pivot_local = np.array([OFF[0] - HALF[0], OFF[1], OFF[2]])
    p = -Rm @ pivot_local
    w = np.array([0.0, omega, 0.0])
    v = np.cross(w, p) + np.array([0.04, 0.0, 0.0])                      # the heel edge slides forward at 40 mm/s
    pos = np.stack([p, p + [0, 0.1, 0]]); vel = np.stack([v, v]); ang = np.stack([w, w]); quat = np.stack([q, q])
    # left foot (index 1) sole centre is at +y: use the same pose, the heel edge is the lowest in both
    s, _ = _slip(pos, quat, vel, ang)
    assert s[0] == pytest.approx(0.04, abs=2e-3)


def test_corner_velocity_matches_finite_difference_of_the_rigid_motion():
    """Independent check of the corner frames, lateral signs and omega x r: move a random rigid body, differentiate the
    numpy corner positions (central difference), apply the same lowest-corner weights, compare with the torch function."""
    rng = np.random.default_rng(5)
    dt = 1e-5
    for _ in range(20):
        sign = [-1.0, 1.0]
        p0 = rng.uniform(-0.1, 0.1, (2, 3)) + [0, 0, 0.04]
        q0 = np.array([_qmul(_quat_axis(rng.normal(size=3), rng.uniform(-0.3, 0.3)), _quat_axis([0, 0, 1], rng.uniform(-3, 3)))
                       for _ in range(2)])
        v = rng.uniform(-0.3, 0.3, (2, 3)); w = rng.uniform(-3, 3, (2, 3))
        got, zlow = _slip(p0, q0, v, w)
        for k in range(2):
            def at(t):
                q = _qmul(_quat_axis(w[k], np.linalg.norm(w[k]) * t), q0[k])
                return _corners_np(p0[k] + v[k] * t, q, sign[k])
            vc = (at(dt) - at(-dt)) / (2 * dt)
            c0 = at(0.0)
            wgt = np.exp(-(c0[:, 2] - c0[:, 2].min()) / 0.002)
            ref = float((wgt * np.linalg.norm(vc[:, :2], axis=1)).sum() / wgt.sum())
            assert got[k] == pytest.approx(ref, abs=2e-6)
            assert zlow[k] == pytest.approx(c0[:, 2].min(), abs=1e-12)


def test_batched_float32_and_jit_consistent():
    rng = np.random.default_rng(1)
    n = 32
    pos = rng.uniform(-0.1, 0.1, (n, 2, 3)); quat = rng.normal(size=(n, 2, 4)); quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    vel = rng.uniform(-0.3, 0.3, (n, 2, 3)); ang = rng.uniform(-2, 2, (n, 2, 3))
    s64, z64 = R.foot_sole_slip_speed(_t(pos), _t(quat), _t(vel), _t(ang), OFF, HALF)
    s32, z32 = R.foot_sole_slip_speed(*(_t(a).float() for a in (pos, quat, vel, ang)), OFF, HALF)
    assert s64.shape == (n, 2) and z64.shape == (n, 2) and torch.isfinite(s32).all()
    assert torch.allclose(s64, s32.double(), atol=2e-5) and torch.allclose(z64, z32.double(), atol=2e-5)
    one = np.array([_slip(pos[i], quat[i], vel[i], ang[i])[0] for i in range(n)])
    assert np.allclose(s64.numpy(), one, atol=1e-12)


# ----------------------------------------------------------------------------- the two penalties
def test_unloaded_feet_give_zero_foot_slip_and_loaded_feet_sum():
    slip = torch.tensor([[0.05, 0.02], [0.05, 0.02], [0.05, 0.02]], dtype=torch.float64)
    loaded = torch.tensor([[False, False], [True, False], [True, True]])
    out = R.foot_slip(slip, loaded)
    assert torch.allclose(out, torch.tensor([0.0, 0.05, 0.07], dtype=torch.float64))
    assert torch.allclose(R.foot_slip(slip, loaded.float()), out)       # float load flags work too


def test_drag_is_zero_above_h_and_when_loaded_and_linear_in_speed_and_clearance():
    h = 0.005
    f = lambda sp, clr, ld: float(R.foot_drag(torch.tensor([sp], dtype=torch.float64), torch.tensor([clr], dtype=torch.float64),
                                              torch.tensor([ld]), h)[0])
    # a single unloaded foot (the second foot is loaded -> contributes 0)
    one = lambda sp, clr, load0: f([sp, 0.3], [clr, 0.0], [load0, True])
    assert one(0.2, h, False) == pytest.approx(0.0)                     # at h: nothing
    assert one(0.2, 2 * h, False) == pytest.approx(0.0)                 # above h
    assert one(0.2, 0.0, True) == pytest.approx(0.0)                    # loaded
    assert one(0.2, 0.0, False) == pytest.approx(0.2)                   # on the floor, unloaded: the full speed
    assert one(0.2, -0.003, False) == pytest.approx(0.2)                # below the datum: clamped, not amplified
    assert one(0.4, 0.002, False) == pytest.approx(2 * one(0.2, 0.002, False))      # linear in speed
    assert one(0.2, 0.5 * h, False) == pytest.approx(0.1)               # linear fade in clearance
    assert one(0.0, 0.0, False) == pytest.approx(0.0)                   # not moving: no drag
    both = R.foot_drag(torch.tensor([[0.1, 0.2]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64),
                       torch.zeros(1, 2, dtype=torch.bool), h)
    assert float(both[0]) == pytest.approx(0.3)                         # sums over both feet


def test_loaded_flag_matches_the_feet_slide_contact_flag():
    hist = torch.zeros(3, 4, 2, 1, 3, dtype=torch.float64)              # (N, T, F, M, 3)
    hist[0, 2, 0, 0, 2] = 5.0                                           # env 0 right foot loaded at one history step
    hist[1, :, 1, 0, 2] = 0.5                                           # env 1 left foot below 1 N
    flag = R.feet_loaded_flag(hist)
    assert flag.tolist() == [[True, False], [False, False], [False, False]]
    # same quantity as feet_slide's internal flag: a unit planar speed returns the flag
    vel = torch.zeros(3, 2, 3, dtype=torch.float64); vel[..., 0] = 1.0
    sl = R.feet_slide(hist, vel)
    assert sl.tolist() == flag.double().sum(dim=1).tolist()


# ----------------------------------------------------------------------------- weights, config, wiring
def test_sign_guard_entries():
    assert R.REWARD_WEIGHT_SIGN["foot_slip"] == -1 and R.REWARD_WEIGHT_SIGN["foot_drag"] == -1
    assert R.reward_weight_sign_errors({"foot_slip": 1.0, "foot_drag": 0.5}) and len(R.reward_weight_sign_errors({"foot_slip": 1.0, "foot_drag": 0.5})) == 2
    assert not R.reward_weight_sign_errors({"foot_slip": -10.0, "foot_drag": -5.0, "feet_slide": 0.0})
    assert not R.reward_weight_sign_errors({"foot_slip": 0.0, "foot_drag": 0.0, "feet_slide": -0.7})      # E28 config values


def test_gait_reference_yaml_weights_load():
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    w = dict(ec.reward_weights)
    # E28: back to the 10-22_18-48 setup: feet_slide on, the E25 terms off (still in the code + classified + sign-guarded; diagnostics log)
    assert w["feet_slide"] == -0.7
    assert w["foot_slip"] == -4.0 and w["foot_drag"] == -2.0   # 2026-10-24: foot_drag back on, milder than E25 (-5)
    assert 0.0 < float(ec.foot_drag_height_m) <= 0.005
    R.validate_reward_weight_signs(w, where="config_gait_reference.yaml")


def test_cfg_schema_and_env_config_agree_on_drag_height():
    import ast
    from config.schema import EnvConfig
    tree = ast.parse(_src("envs/isaaclab/biped_env_cfg.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnvCfg")
    val = next(ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "foot_drag_height_m")
    assert val == EnvConfig().foot_drag_height_m == 0.005
    assert '"foot_drag_height_m"' in _src("src/isaaclab/env_config.py")


def test_env_wiring_is_source_level_correct():
    src = _src("envs/isaaclab/biped_env.py")
    assert 'self.cfg.rewards.get("foot_slip", 0.0)' in src and 'self.cfg.rewards.get("foot_drag", 0.0)' in src
    call = src[src.index("rewards.foot_sole_slip_speed("):]
    call = call[:call.index("list(self._foot_sole_half))")]
    # FRESH articulation state, and the LINK-frame velocity (body_lin_vel_w is the COM frame in this Isaac Lab)
    for name in ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "body_link_ang_vel_w"):
        assert f"_rd.{name}[:, self._feet_idx_t]" in call, name
    for stale in ("self.base_", "self.joint_pos", "self.joint_efforts", "body_lin_vel_w", "body_ang_vel_w"):
        assert stale not in call, stale
    assert "rewards.foot_slip(foot_slip_speed, foot_loaded)" in src
    assert "rewards.foot_drag(foot_slip_speed, sole_low_z - self.foot_sole_ground_z, foot_loaded, self._foot_drag_h)" in src
    assert "rewards.feet_loaded_flag(cs.force_matrix_w_history" in src
    # diagnostics go through the shared diag_keys / diag_sums / diag_counts mechanism
    assert '"foot_slip_mms", "foot_drag_mms")' in src and 'diag["foot_slip_mms"]' in src and 'diag["foot_drag_mms"]' in src
    assert "self.foot_sole_ground_z = torch.where(fresh," in src       # datum seeded with the same reset mask as foot_ground_z
    # the old term is still computed (weight 0 just zeroes it)
    assert 'rewards.feet_slide(' in src and 'self.cfg.rewards.get("feet_slide", 0.0)' in src


# ----------------------------------------------------------------------------- MuJoCo (cheap)
def _model():
    mujoco = pytest.importorskip("mujoco")
    try:
        return mujoco, mujoco.MjModel.from_xml_path(MJCF)
    except Exception as e:                                              # mesh paths in the MJCF are absolute
        pytest.skip(f"MJCF meshes not loadable here: {e}")


def test_sole_corners_lie_on_the_mujoco_sole_for_both_feet():
    """At the zero pose the foot links are level: the 4 corners (right -y, left +y) sit on the lowest mesh vertices' plane and
    inside the foot's xy extent (within 1 mm)."""
    mujoco, m = _model()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.3
    mujoco.mj_forward(m, d)
    for body, sign in (("r_foot_link_1", -1.0), ("l_foot_link_1", +1.0)):
        b = m.body(body).id
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == b)
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        W = d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T
        C = _corners_np(d.xpos[b], d.xquat[b], sign)
        assert np.allclose(C[:, 2], W[:, 2].min(), atol=1e-3)
        assert (C[:, 0].min() >= W[:, 0].min() - 1e-3) and (C[:, 0].max() <= W[:, 0].max() + 1e-3)
        assert (C[:, 1].min() >= W[:, 1].min() - 1e-3) and (C[:, 1].max() <= W[:, 1].max() + 1e-3)


def test_mujoco_xbody_velocity_is_the_origin_velocity_not_the_com():
    """The MuJoCo calibration (policy_gate.FootSlipProbe) feeds mj_objectVelocity(mjOBJ_XBODY, flg_local=0): [w; v] at the body
    ORIGIN in world axes (mjOBJ_BODY would be the COM). Same as mj_jacBody @ qvel."""
    mujoco, m = _model()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    d.qpos[2] = 0.3
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = rng.normal(size=m.nv)
    mujoco.mj_forward(m, d)
    b = m.body("r_foot_link_1").id
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, b)
    assert np.allclose(res[3:], jp @ d.qvel, atol=1e-9) and np.allclose(res[:3], jr @ d.qvel, atol=1e-9)
    res_com = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res_com, 0)
    assert not np.allclose(res_com[3:], res[3:], atol=1e-4)             # the COM differs from the origin


def test_probe_in_the_mujoco_env_stands_with_zero_slip_and_drag():
    pytest.importorskip("onnxruntime")
    from src.mujoco import policy_gate as pg
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    ec.backlash = {**(ec.backlash or {}), "enabled": False}              # enabling it rewrites a generated XML
    env = pg.BipedEnvV2(history_size=2, use_history=True, env_config=ec, seed=0)
    try:
        env.reset(seed=0)
        env.set_command(0.0, 0.0, 0.0)
        probe = pg.FootSlipProbe(env, OFF, HALF)
        for _ in range(25):
            env.step(np.zeros(12, np.float32))
            probe.update()
        s = probe.series()[10:]
        ix = {k: i for i, k in enumerate(pg.FootSlipProbe.NAMES)}
        assert s[:, ix["n_loaded"]].min() >= 0
        assert np.isfinite(s).sum() > 0
        assert float(np.nanmax(s[:, ix["drag"]])) < 5e-3                # standing: nothing scuffs
        assert float(np.nanmax(s[:, ix["slip"]])) < 0.05
    finally:
        env.close()
