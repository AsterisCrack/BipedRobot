"""foot_proximity (E21): terrain-independent geometric foot-to-foot gap (CPU, no Isaac). Run from BipedRobot/:
    python -m pytest tests/test_foot_proximity.py -q

Why: with use_rough_terrain the contact sensor has no filter prims, so undesired_contacts / leg_contact_l2 leave the
feet out and feet pressing together cost nothing (run gaitref_2026-10-22_07-18-30: 100% of steps at command 0, 13.8 N,
MuJoCo -1.2 mm). rewards.foot_footprint_gap is purely geometric (two oriented sole rectangles in the horizontal plane).
"""
import ast
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

TABLE = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
MJCF = os.path.join(ROOT, "envs", "assets", "robotV2", "robot_v2.xml")
OFF = [0.0163, 0.0211, -0.0316]       # == cfg.foot_sole_offset_m
HALF = [0.0450, 0.0186]               # == cfg.foot_sole_half_extents_m
IDQ = [1.0, 0.0, 0.0, 0.0]


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _cfg_literal(name, rel="envs/isaaclab/biped_env_cfg.py"):
    """Default of a BipedEnvCfg class attribute, read from the source (the cfg module needs Isaac to import)."""
    tree = ast.parse(_src(rel))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnvCfg")
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == name:
            return ast.literal_eval(n.value)
    raise AssertionError(f"{name} not found in BipedEnvCfg")


def _t(x):
    return torch.tensor(np.asarray(x, np.float32))


def _quat(yaw=0.0, pitch=0.0, roll=0.0):
    """[w, x, y, z] for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def _rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _gap(pl, ql, pr, qr, off=OFF, half=HALF):
    f = lambda a: _t(a)[None]
    return float(R.foot_footprint_gap(f(pl), f(ql), f(pr), f(qr), off, half)[0])


def _ref_rect(pos, q, sign):
    """Independent numpy model: 4 corners (xy) and the centre of one foot's footprint rectangle."""
    Rm = _rotmat(q)
    c = (np.asarray(pos, np.float64) + Rm @ np.array([OFF[0], sign * OFF[1], OFF[2]]))[:2]
    yaw = math.atan2(Rm[1, 0], Rm[0, 0])
    ax, ay = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    corners = np.array([c + sx * HALF[0] * ax + sy * HALF[1] * ay for sx in (1, -1) for sy in (1, -1)])
    return c, ax, ay, corners


def _ref_signed_gap(A, B, n_dir=7200):
    """Brute force: max over directions u of the separation of the two vertex sets projected on u. For convex
    shapes this is the Euclidean distance when apart and minus the penetration depth when overlapping."""
    th = np.linspace(0.0, math.pi, n_dir, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], axis=1)
    pa, pb = A @ U.T, B @ U.T
    sep = np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0))
    return float(sep.max())


def _ref_sampled_dist(pa, qa, pb, qb, step=0.0015):
    """Min distance between densely SAMPLED points of the two rectangles (apart cases only)."""
    def pts(c, ax, ay):
        gx = np.arange(-HALF[0], HALF[0] + 1e-9, step)
        gy = np.arange(-HALF[1], HALF[1] + 1e-9, step)
        X, Y = np.meshgrid(gx, gy)
        return c + X.reshape(-1, 1) * ax + Y.reshape(-1, 1) * ay
    ca, axa, aya, _ = _ref_rect(pa, qa, +1)
    cb, axb, ayb, _ = _ref_rect(pb, qb, -1)
    PA, PB = pts(ca, axa, aya), pts(cb, axb, ayb)
    d = np.sqrt(((PA[:, None, :] - PB[None, :, :]) ** 2).sum(-1)).min()
    return float(d)


# ----------------------------------------------------------------------------- the gap function
@pytest.mark.parametrize("half_sep", [0.04, 0.05, 0.0, 0.06, 0.08])
def test_axis_aligned_lateral_gap_is_exact(half_sep):
    # left/right link origins at +-half_sep; sole centres sit OUTBOARD by OFF[1]; each rect has half width HALF[1]
    expected = 2.0 * (half_sep + OFF[1]) - 2.0 * HALF[1]
    g = _gap([0, half_sep, 0.03], IDQ, [0, -half_sep, 0.03], IDQ)
    assert g == pytest.approx(expected, abs=1e-6)


def test_axis_aligned_diagonal_gap_is_corner_distance():
    # left foot 0.12 m ahead and well apart laterally: nearest points are two corners
    dx = 0.12 - 2.0 * HALF[0]
    dy = 2.0 * (0.05 + OFF[1]) - 2.0 * HALF[1]
    g = _gap([0.12, 0.05, 0.0], IDQ, [0, -0.05, 0.0], IDQ)
    assert g == pytest.approx(math.hypot(dx, dy), abs=1e-6)


def test_fore_aft_offset_gap():
    # left foot 0.10 m ahead at the home stance: 10 mm apart along x, 5 mm apart along y -> corner distance
    g = _gap([0.10, 0.0, 0.0], IDQ, [0.0, 0.0, 0.0], IDQ)
    gx = 0.10 - 2.0 * HALF[0]
    gy = 2.0 * OFF[1] - 2.0 * HALF[1]
    assert gx > 0 and gy > 0
    assert g == pytest.approx(math.hypot(gx, gy), abs=1e-6)


def test_coincident_footprints_and_exact_touch():
    # origins placed so that BOTH sole centres coincide (left centre y = y_l + OFF[1], right = y_r - OFF[1])
    g = _gap([0, -2.0 * OFF[1], 0.0], IDQ, [0, 0.0, 0.0], IDQ)
    assert g == pytest.approx(-2.0 * HALF[1], abs=1e-6)       # SAT depth = the thinner full width
    # centres exactly one footprint width apart: touching
    g0 = _gap([0, HALF[1] - OFF[1], 0.0], IDQ, [0, -HALF[1] + OFF[1], 0.0], IDQ)
    assert g0 == pytest.approx(0.0, abs=1e-6)


def test_signed_gap_is_continuous_through_contact():
    ys = np.linspace(-OFF[1] + 0.001, 0.03, 121)        # from just before the sole centres cross
    gs = np.array([_gap([0, y, 0.0], IDQ, [0, -y, 0.0], IDQ) for y in ys])
    assert np.all(np.diff(gs) > 0)
    assert np.max(np.abs(np.diff(gs))) < 2.01 * (ys[1] - ys[0]) + 1e-6      # slope 2: no jump at 0
    assert gs.min() < 0.0 < gs.max()


def test_rotated_and_tilted_poses_match_brute_force():
    rng = np.random.default_rng(7)
    n_apart = n_overlap = 0
    for _ in range(250):
        pl = [rng.uniform(-0.08, 0.08), rng.uniform(-0.03, 0.05), rng.uniform(0.0, 0.05)]
        pr = [rng.uniform(-0.08, 0.08), -rng.uniform(-0.03, 0.05), rng.uniform(0.0, 0.05)]
        ql = _quat(rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
        qr = _quat(rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
        g = _gap(pl, ql, pr, qr)
        _, _, _, A = _ref_rect(pl, ql, +1)
        _, _, _, B = _ref_rect(pr, qr, -1)
        ref = _ref_signed_gap(A, B)
        assert g == pytest.approx(ref, abs=2e-4), (g, ref)             # float32 vs 0.025 deg direction sweep
        if ref > 0.002:
            n_apart += 1
            if n_apart <= 25:                                          # sampled points on a 1.5 mm grid
                assert g == pytest.approx(_ref_sampled_dist(pl, ql, pr, qr), abs=2.5e-3)
        else:
            n_overlap += 1
    assert n_apart >= 20 and n_overlap >= 20          # the random draw exercises both branches


def test_overlap_is_nonpositive_and_symmetric_in_sign():
    g = _gap([0.02, 0.0, 0.0], _quat(0.3), [0.0, 0.0, 0.0], _quat(-0.2))
    assert g <= 0.0


def test_batched_and_jit_consistent():
    rng = np.random.default_rng(3)
    N = 64
    pl = np.c_[rng.uniform(-0.1, 0.1, N), rng.uniform(0, 0.08, N), rng.uniform(0, 0.05, N)]
    pr = np.c_[rng.uniform(-0.1, 0.1, N), -rng.uniform(0, 0.08, N), rng.uniform(0, 0.05, N)]
    ql = np.array([_quat(rng.uniform(-1, 1), rng.uniform(-.3, .3), rng.uniform(-.3, .3)) for _ in range(N)])
    qr = np.array([_quat(rng.uniform(-1, 1), rng.uniform(-.3, .3), rng.uniform(-.3, .3)) for _ in range(N)])
    batch = R.foot_footprint_gap(_t(pl), _t(ql), _t(pr), _t(qr), OFF, HALF).numpy()
    one = np.array([_gap(pl[i], ql[i], pr[i], qr[i]) for i in range(N)])
    assert batch.shape == (N,) and np.allclose(batch, one, atol=1e-6)
    assert np.isfinite(batch).all()


# ----------------------------------------------------------------------------- the penalty
def test_penalty_values():
    m = 0.01
    gap = torch.tensor([0.05, 0.01, 0.005, 0.0, -0.005, -0.01, -0.03])
    p = R.foot_proximity_penalty(gap, m)
    assert torch.allclose(p, torch.tensor([0.0, 0.0, 0.25, 1.0, 2.25, 4.0, 4.0]), atol=1e-6)
    assert float(p.max()) <= 4.0 and float(p.min()) >= 0.0


def test_foot_proximity_is_a_penalty_in_the_sign_guard():
    assert R.REWARD_WEIGHT_SIGN["foot_proximity"] == -1
    assert R.reward_weight_sign_errors({"foot_proximity": 2.0})
    assert not R.reward_weight_sign_errors({"foot_proximity": -2.0})


# ----------------------------------------------------------------------------- footprint vs the MJCF
def _mjcf_foot_geometry():
    mujoco = pytest.importorskip("mujoco")
    try:
        m = mujoco.MjModel.from_xml_path(MJCF)
    except Exception as e:                                    # mesh paths in the MJCF are absolute
        pytest.skip(f"MJCF meshes not loadable here: {e}")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = {}
    for side in ("l", "r"):
        b = m.body(f"{side}_foot_link_1").id
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == b)
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        W = d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T
        B = (W - d.xpos[b]) @ d.xmat[b].reshape(3, 3)          # vertices in the foot LINK frame
        sole = B[B[:, 2] < B[:, 2].min() + 0.002]
        out[side] = dict(lo=B[:, :2].min(0), hi=B[:, :2].max(0), sole_lo=sole[:, :2].min(0),
                         sole_hi=sole[:, :2].max(0), sole_z=float(B[:, 2].min()))
    return out


def test_cfg_footprint_matches_mjcf_within_1mm():
    geo = _mjcf_foot_geometry()
    half, off = _cfg_literal("foot_sole_half_extents_m"), _cfg_literal("foot_sole_offset_m")
    assert list(half) == HALF and list(off) == OFF            # the constants above are the cfg defaults
    for side, sign in (("l", +1.0), ("r", -1.0)):
        g = geo[side]
        full_half = (g["hi"] - g["lo"]) / 2.0                  # half extents of the whole mesh footprint
        sole_centre = (g["sole_lo"] + g["sole_hi"]) / 2.0      # centre of the sole face
        assert abs(half[0] - full_half[0]) < 1e-3 and abs(half[1] - full_half[1]) < 1e-3, (side, full_half)
        assert abs(off[0] - sole_centre[0]) < 1e-3 and abs(sign * off[1] - sole_centre[1]) < 1e-3, (side, sole_centre)
        assert abs(off[2] - g["sole_z"]) < 1e-3
        # the sole face itself is within 1 mm of the rectangle on every side except the chamfer
        assert abs(half[0] - (g["sole_hi"][0] - g["sole_lo"][0]) / 2.0) < 1e-3
        assert abs(half[1] - (g["sole_hi"][1] - g["sole_lo"][1]) / 2.0) < 1e-3 + 0.5e-3


def test_schema_and_yaml_agree_with_cfg():
    from config.schema import EnvConfig  # noqa: F401  (declares the keys)
    s = EnvConfig()
    assert list(s.foot_sole_half_extents_m) == _cfg_literal("foot_sole_half_extents_m")
    assert s.foot_proximity_margin_m == _cfg_literal("foot_proximity_margin_m")
    assert s.foot_overlap_termination_s == _cfg_literal("foot_overlap_termination_s") == 0.0


# ----------------------------------------------------------------------------- reference gait is never penalised
def test_gait_table_reference_gap_is_not_penalised():
    mujoco = pytest.importorskip("mujoco")
    if not os.path.exists(TABLE):
        pytest.skip("gait table not present")
    try:
        m = mujoco.MjModel.from_xml_path(MJCF)
    except Exception as e:
        pytest.skip(f"MJCF meshes not loadable here: {e}")
    d = mujoco.MjData(m)
    z = np.load(TABLE, allow_pickle=True)
    names = [str(n) for n in z["field_names"]]
    widths = [int(w) for w in z["field_widths"]]
    start = dict(zip(names, np.cumsum([0] + widths[:-1])))
    sl = lambda a, n: a[..., start[n]:start[n] + widths[names.index(n)]]
    adr = [m.joint(str(n)).qposadr[0] for n in z["joint_names"]]
    lb, rb = m.body("l_foot_link_1").id, m.body("r_foot_link_1").id

    def pose(fr):
        up = -sl(fr, "projected_gravity").astype(np.float64)       # world up in the body frame, yaw = 0
        b = np.array([0.0, up[2], -up[1]])
        b /= np.linalg.norm(b)
        Rm = np.stack([np.cross(b, up), b, up])                    # rows = world axes in the body frame
        d.qpos[:] = 0.0
        d.qpos[adr] = sl(fr, "joint_pos")
        d.qpos[0:3] = [0.0, 0.0, float(sl(fr, "base_height")[0])]
        mujoco.mju_mat2Quat(d.qpos[3:7], Rm.flatten())
        mujoco.mj_kinematics(m, d)
        return np.concatenate([d.xpos[lb], d.xquat[lb], d.xpos[rb], d.xquat[rb]])

    gv, gy, gw = (z[k] for k in ("grid_vx", "grid_vy", "grid_wz"))

    # the cells the CONFIG's command ranges select (read from the yaml), plus the first grid point beyond a bound that is
    # not itself a grid point (a command in range interpolates from it)
    cmd = _load_env_conf().commands["base_velocity"]["ranges"]
    bound = {k: max(abs(float(v)) for v in cmd[c]) for k, c in
             (("grid_vx", "lin_vel_x"), ("grid_vy", "lin_vel_y"), ("grid_wz", "ang_vel_z"))}

    def idx(g, lim):
        keep = {i for i, v in enumerate(g) if abs(v) <= lim + 1e-6}
        for sgn in (1, -1):
            inside = [g[i] for i in keep if np.sign(g[i]) == sgn]
            beyond = [i for i, v in enumerate(g) if np.sign(v) == sgn and abs(v) > lim + 1e-6]
            if beyond and not any(abs(abs(v) - lim) < 1e-6 for v in inside):
                keep.add(min(beyond, key=lambda i: abs(g[i])))
        return sorted(keep)
    ix, iy, iw = idx(gv, bound["grid_vx"]), idx(gy, bound["grid_vy"]), idx(gw, bound["grid_wz"])
    assert ix and iy and iw
    data = z["data"]
    P = np.array([pose(data[i, j, k, f]) for i in ix for j in iy for k in iw for f in range(data.shape[3])] +
                 [pose(z["stand"])], np.float32)
    Pt = torch.tensor(P)
    gap = R.foot_footprint_gap(Pt[:, 0:3], Pt[:, 3:7], Pt[:, 7:10], Pt[:, 10:14], OFF, HALF).numpy()
    margin = _cfg_literal("foot_proximity_margin_m")
    assert float(_load_env_conf().foot_proximity_margin_m) == margin       # cfg default == the gait-reference config
    # Re-bake robust: the margin is NOT pinned to the table's minimum gap. It only has to (a) leave every reference frame
    # inside the command ranges unpenalised, (b) stay >= 5 mm so the penalty still acts before contact, (c) match the cfg.
    headroom = math.floor(min(0.010, gap.min()) * 1000.0 + 1e-9) / 1000.0
    print(f"\nfoot_proximity: table min footprint gap {gap.min() * 1000.0:.2f} mm over {len(gap)} frames; "
          f"configured margin {margin * 1000.0:.1f} mm; the table would allow up to {headroom * 1000.0:.0f} mm")
    assert gap.min() > margin, (gap.min(), margin)             # (a) the reference gait is never penalised
    assert float(R.foot_proximity_penalty(torch.tensor(gap), margin).max()) == 0.0
    assert margin >= 0.005, margin                             # (b) still acts before contact
    assert gap[-1] >= 0.020                    # stand frame clear of the margin (2026-10-23 table: 27.8 mm)


# ----------------------------------------------------------------------------- config + wiring (source level)
def _load_env_conf(name="config_gait_reference.yaml"):
    from utils import Config
    return Config(os.path.join(ROOT, "config", name)).train.env_config


def test_gait_reference_config_values():
    ec = _load_env_conf()
    assert float(ec.reward_weights["foot_proximity"]) == -2.0
    assert float(ec.foot_overlap_termination_s) == 0.0
    assert float(ec.foot_proximity_margin_m) == _cfg_literal("foot_proximity_margin_m")
    assert list(ec.foot_sole_half_extents_m) == _cfg_literal("foot_sole_half_extents_m")
    assert not R.reward_weight_sign_errors(dict(ec.reward_weights))


def test_env_config_plumbs_the_new_fields():
    tree = ast.parse(_src("src/isaaclab/env_config.py"))
    fields = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_SCALAR_FIELDS")
    for k in ("foot_sole_half_extents_m", "foot_proximity_margin_m", "foot_overlap_termination_s",
              "foot_sole_offset_m"):
        assert k in fields, k
        _cfg_literal(k)                                              # and the cfg declares it


def _method(name):
    tree = ast.parse(_src("envs/isaaclab/biped_env.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnv")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return fn, ast.get_source_segment(_src("envs/isaaclab/biped_env.py"), fn)


def test_reward_reads_fresh_robot_data_for_both_feet():
    _, helper = _method("_foot_footprint_gap")
    # _feet_idx_t: device-tensor copy of feet_indices (a Python-list index syncs the GPU)
    assert "self.robot.data.body_pos_w[:, self._feet_idx_t]" in helper
    assert "self.robot.data.body_quat_w[:, self._feet_idx_t]" in helper
    assert "self._feet_idx_t = _ids(self.feet_indices)" in _src("envs/isaaclab/biped_env.py")
    assert "rewards.foot_footprint_gap(" in helper
    # feet_indices is [right, left]; the function takes (left, right)
    assert "pos[:, 1], quat[:, 1], pos[:, 0], quat[:, 0]" in helper
    assert "[self.cfg.right_foot_body_name, self.cfg.left_foot_body_name]" in _src("envs/isaaclab/biped_env.py")
    _, rew = _method("_get_rewards")
    assert "self._foot_footprint_gap()" in rew
    assert "rewards.foot_proximity_penalty(" in rew
    assert '"foot_proximity": r_foot_proximity * self.cfg.rewards.get("foot_proximity", 0.0)' in rew
    for k in ("foot_gap_mm", "foot_overlap_frac"):
        assert f'"{k}"' in _src("envs/isaaclab/biped_env.py")


def test_overlap_termination_counter_is_updated_once_per_step():
    src = _src("envs/isaaclab/biped_env.py")
    writers = {}
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnv")
    for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            tgt = []
            if isinstance(node, ast.Assign):
                tgt = node.targets
            elif isinstance(node, ast.AugAssign):
                tgt = [node.target]
            for t in tgt:
                base = t.value if isinstance(t, ast.Subscript) else t
                if isinstance(base, ast.Attribute) and base.attr == "foot_overlap_count":
                    writers.setdefault(fn.name, []).append(ast.get_source_segment(src, node))
    assert set(writers) == {"__init__", "_get_dones"}, writers     # _reset_idx zeroes it with index_fill_ (below)
    assert len(writers["_get_dones"]) == 1                      # exactly one update per step ...
    assert "_get_rewards" not in writers                        # ... and the reward never advances it
    _, dones = _method("_get_dones")
    assert "if self._foot_overlap_steps > 0:" in dones          # off by default: counter untouched
    assert "self._foot_footprint_gap() <= 0.0" in dones
    assert "died = died | overlap_done" in dones                # counts as `died` -> termination_penalty applies
    reset = _method("_reset_idx")[1]
    i = reset.index("self.foot_overlap_count,")
    assert "buf.index_fill_(0, env_ids, 0)" in reset[i:i + 300]     # zeroed for the reset envs
    assert "Episode_Termination/foot_overlap" in _method("_reset_idx")[1]
    init = _method("__init__")[1]
    assert "must be >= 0" in init and "foot_overlap_termination_s" in init


def test_policy_gate_uses_the_training_gap_function():
    src = _src("src/mujoco/policy_gate.py")
    assert "from envs.isaaclab.rewards import rewards as _train_rewards" in src
    assert "_train_rewards.foot_footprint_gap(" in src
    assert "foot_overlap_frac" in src and "foot_gap_mean_mm" in src
    assert "foot_gap" not in src[src.index("def _verdict"):src.index("def _case_pose")]     # informational, no new FAIL
