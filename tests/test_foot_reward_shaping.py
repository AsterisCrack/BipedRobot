"""Foot/leg reward shaping pure functions (CPU, no Isaac): foot_flat_roll_l2, the sole-centre
foot separation and its two-sided band, undesired_contacts saturation parameters, leg_contact_l2,
and that the switchable defaults reproduce the old behaviour. Run from BipedRobot/:
    python -m pytest tests/test_foot_reward_shaping.py -q
"""
import ast
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402


def qx(a):
    return torch.tensor([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])


def qy(a):
    return torch.tensor([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


def qz(a):
    return torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


IDQ = torch.tensor([1.0, 0.0, 0.0, 0.0])


def feet_q(right, left):
    return torch.stack([right, left]).unsqueeze(0)          # (1, 2, 4), index 0 = right


def contact(r, l):
    return torch.tensor([[r, l]], dtype=torch.bool)


# ---------------- foot_flat_roll_l2 ----------------

def test_flat_feet_cost_nothing():
    out = R.foot_flat_roll_l2(feet_q(IDQ, IDQ), IDQ.unsqueeze(0), contact(True, True))
    assert out.shape == (1,) and out.item() == pytest.approx(0.0, abs=1e-7)


def test_roll_14deg_is_sin_squared_per_foot():
    a = math.radians(14.0)
    expect = math.sin(a) ** 2                                # 0.0585
    assert expect == pytest.approx(0.058, abs=1e-3)
    one = R.foot_flat_roll_l2(feet_q(qx(a), IDQ), IDQ.unsqueeze(0), contact(True, True)).item()
    both = R.foot_flat_roll_l2(feet_q(qx(a), qx(-a)), IDQ.unsqueeze(0), contact(True, True)).item()
    assert one == pytest.approx(expect, rel=1e-5)
    assert both == pytest.approx(2 * expect, rel=1e-5)       # sign of the roll does not matter


def test_airborne_foot_is_ignored():
    a = math.radians(14.0)
    q = feet_q(qx(a), qx(a))
    root = IDQ.unsqueeze(0)
    assert R.foot_flat_roll_l2(q, root, contact(True, False)).item() == pytest.approx(math.sin(a) ** 2, rel=1e-5)
    assert R.foot_flat_roll_l2(q, root, contact(False, False)).item() == pytest.approx(0.0, abs=1e-9)


def test_pitch_tilt_is_not_roll():
    out = R.foot_flat_roll_l2(feet_q(qy(math.radians(20)), qy(math.radians(-20))), IDQ.unsqueeze(0),
                              contact(True, True))
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_torso_roll_does_not_read_as_foot_tilt():
    # flat foot (world-flat) under a torso rolled 10 deg: horizontal projection of body Y -> 0
    root = qx(math.radians(10)).unsqueeze(0)
    assert R.foot_flat_roll_l2(feet_q(IDQ, IDQ), root, contact(True, True)).item() == pytest.approx(0.0, abs=1e-7)
    # a foot rolled WITH the torso is 10 deg off the ground
    out = R.foot_flat_roll_l2(feet_q(qx(math.radians(10)), IDQ), root, contact(True, True)).item()
    assert out == pytest.approx(math.sin(math.radians(10)) ** 2, rel=1e-5)


def test_roll_is_measured_about_the_body_forward_axis_when_yawed():
    a = math.radians(14.0)
    root = qz(math.pi / 2).unsqueeze(0)       # robot faces world +Y; its lateral axis is world -X
    roll = R.foot_flat_roll_l2(feet_q(qy(a), IDQ), root, contact(True, True)).item()    # about body X
    pitch = R.foot_flat_roll_l2(feet_q(qx(a), IDQ), root, contact(True, True)).item()   # about body Y
    assert roll == pytest.approx(math.sin(a) ** 2, rel=1e-5)
    assert pitch == pytest.approx(0.0, abs=1e-7)


# ---------------- sole-centre separation / band ----------------

def _feet(sole_sep, rq=IDQ, lq=IDQ):
    """Right/left link origins (identity foot orientation by default) whose sole centres are
    sole_sep apart: sole centre = origin +- 0.0211 outboard."""
    half = (sole_sep - 2 * 0.0211) / 2
    pos = torch.tensor([[[0.0, -half, 0.0], [0.0, half, 0.0]]])
    return pos, feet_q(rq, lq)


def _sep(pos, q, root=None):
    root = IDQ.unsqueeze(0) if root is None else root
    return R.foot_sole_centre_lateral_sep(pos, q, root).item()


def test_home_pose_numbers():
    # home: link origins 64.7 mm apart -> sole centres 106.9 mm apart (audit)
    pos = torch.tensor([[[0.0, -0.03235, 0.0], [0.0, 0.03235, 0.0]]])
    q = feet_q(IDQ, IDQ)
    assert _sep(pos, q) == pytest.approx(0.1069, abs=1e-4)
    assert R.foot_separation_sole_band_penalty(pos, q, IDQ.unsqueeze(0)).item() == pytest.approx(0.0)
    # the old origin measure reads 25 mm short of its 0.09 target at the same pose
    old = R.foot_separation_penalty(pos, IDQ.unsqueeze(0), min_sep=0.09).item()
    assert old == pytest.approx(0.09 - 0.0647, abs=1e-4)


def test_band_penalty_values():
    root = IDQ.unsqueeze(0)

    def pen(s):
        pos, q = _feet(s)
        return R.foot_separation_sole_band_penalty(pos, q, root).item()

    assert pen(0.107) == pytest.approx(0.0)
    assert pen(0.07) == pytest.approx(0.0, abs=1e-6)         # inner edge of the band (0.095 - 0.025)
    assert pen(0.12) == pytest.approx(0.0, abs=1e-6)         # outer edge
    assert pen(0.04) == pytest.approx(((0.055 - 0.025) / 0.02) ** 2, rel=1e-4)    # 2.25
    assert pen(0.16) == pytest.approx(((0.065 - 0.025) / 0.02) ** 2, rel=1e-4)    # 4.0
    assert pen(0.30) == pytest.approx(9.0)                   # clamp at 3 -> 9 (max)
    assert pen(-0.02) == pytest.approx(9.0)                  # crossed feet read negative -> max
    assert pen(0.04) > 0 and pen(0.16) > 0                   # two-sided


def test_band_params_are_used():
    pos, q = _feet(0.15)
    root = IDQ.unsqueeze(0)
    wide = R.foot_separation_sole_band_penalty(pos, q, root, 0.15, 0.0, 0.02).item()
    assert wide == pytest.approx(0.0, abs=1e-6)
    tight = R.foot_separation_sole_band_penalty(pos, q, root, 0.095, 0.025, 0.01).item()
    assert tight == pytest.approx(9.0)    # (0.055 - 0.025) / 0.01 = 3, the clamp


def test_left_sole_sign_follows_the_urdf():
    # left foot rolled +theta about x: link (0, +0.0211, -0.0316) -> y' = 0.0211 cos t + 0.0316 sin t
    t = math.radians(26.0)
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    left_only = _sep(pos, feet_q(IDQ, qx(t)))
    right_flat = -0.0211
    expect = (0.0211 * math.cos(t) + 0.0316 * math.sin(t)) - right_flat
    assert left_only == pytest.approx(expect, abs=1e-6)
    # right foot rolled -theta (the mirror) outboard: y' = -(0.0211 cos t + 0.0316 sin t)
    right_only = _sep(pos, feet_q(qx(-t), IDQ))
    assert right_only == pytest.approx(0.0211 + (0.0211 * math.cos(t) + 0.0316 * math.sin(t)), abs=1e-6)


def test_left_right_mirror_symmetry():
    th, ph = math.radians(26.0), math.radians(-7.0)
    yl, yr = 0.031, 0.036
    pos_a = torch.tensor([[[0.0, -yr, 0.0], [0.0, yl, 0.01]]])
    qa = feet_q(qx(ph), qx(th))                               # right Rx(ph), left Rx(th)
    # mirror across the sagittal plane: y -> -y, angle about x -> -angle, feet swap sides
    pos_b = torch.tensor([[[0.0, -yl, 0.01], [0.0, yr, 0.0]]])
    qb = feet_q(qx(-th), qx(-ph))
    assert _sep(pos_a, qa) == pytest.approx(_sep(pos_b, qb), abs=1e-6)
    pa = R.foot_separation_sole_band_penalty(pos_a, qa, IDQ.unsqueeze(0)).item()
    pb = R.foot_separation_sole_band_penalty(pos_b, qb, IDQ.unsqueeze(0)).item()
    assert pa == pytest.approx(pb, abs=1e-6)


def test_sep_is_body_frame_not_world():
    pos, q = _feet(0.107)
    for yaw in (0.0, 1.0, 2.5, -2.0):
        # rotate the whole stance with the base: same body-frame separation
        c, s = math.cos(yaw), math.sin(yaw)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos_w = pos @ rot.T
        q_w = torch.stack([torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])] * 2).unsqueeze(0)
        assert _sep(pos_w, q_w, qz(yaw).unsqueeze(0)) == pytest.approx(0.107, abs=1e-5)


# ---------------- undesired_contacts / leg_contact_l2 ----------------

def _old_undesired(f, threshold=1.0, saturation=20.0):
    forces = torch.norm(f, dim=-1) if f.dim() == 3 else f
    return torch.sum(torch.clamp((forces - threshold) / saturation, min=0.0, max=1.0), dim=-1)


def test_undesired_defaults_reproduce_old():
    g = torch.Generator().manual_seed(0)
    f = torch.rand(7, 11, generator=g) * 40.0
    assert torch.allclose(R.undesired_contacts(f), _old_undesired(f))
    v = torch.randn(7, 11, 3, generator=g) * 15.0
    assert torch.allclose(R.undesired_contacts(v), _old_undesired(v))
    assert torch.allclose(R.undesired_contacts(f, threshold=1.0, saturation=20.0), _old_undesired(f))


def test_undesired_sat_5_makes_a_6N_lean_cost_1():
    f = torch.tensor([[5.9, 0.0, 0.0]])
    assert R.undesired_contacts(f).item() == pytest.approx((5.9 - 1.0) / 20.0)             # old: 0.245
    assert R.undesired_contacts(f, threshold=1.0, saturation=5.0).item() == pytest.approx(0.98)
    f2 = torch.tensor([[0.5, 100.0]])
    assert R.undesired_contacts(f2, 1.0, 5.0).item() == pytest.approx(1.0)                 # still saturates
    assert R.undesired_contacts(torch.tensor([[1.0]]), 1.0, 5.0).item() == pytest.approx(0.0)


def test_leg_contact_l2_shape():
    f = torch.tensor([[0.0, 0.4, 0.5, 2.75, 5.0, 50.0]])
    per = [R.leg_contact_l2(f[:, i:i + 1]).item() for i in range(6)]
    assert per[0] == per[1] == per[2] == 0.0                 # free below 0.5 N
    assert per[3] == pytest.approx(0.25, rel=1e-5)           # halfway to saturation, squared
    assert per[4] == pytest.approx(1.0) and per[5] == pytest.approx(1.0)
    assert R.leg_contact_l2(f).item() == pytest.approx(sum(per))
    assert R.leg_contact_l2(f, thresh_n=1.0, sat_n=10.0).item() == pytest.approx(
        ((2.75 - 1.0) / 9.0) ** 2 + ((5.0 - 1.0) / 9.0) ** 2 + 1.0, rel=1e-5)


def test_foot_separation_penalty_unchanged():
    pos = torch.tensor([[[0.0, -0.01, 0.0], [0.0, 0.02, 0.0]], [[0.0, -0.06, 0.0], [0.0, 0.06, 0.0]]])
    root = IDQ.unsqueeze(0).repeat(2, 1)
    out = R.foot_separation_penalty(pos, root, min_sep=0.09)
    assert out[0].item() == pytest.approx(0.09 - 0.03, abs=1e-6)
    assert out[1].item() == pytest.approx(0.0)


# ---------------- config plumbing: defaults == old behaviour ----------------

def _cfg_defaults():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env_cfg.py"), encoding="utf-8").read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "BipedEnvCfg")
    out = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            try:
                out[n.target.id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


NEW_KEYS = {
    "foot_separation_mode": "origin_one_sided", "foot_sep_target_m": 0.095, "foot_sep_tol_m": 0.025,
    "foot_sep_scale_m": 0.02, "foot_sole_offset_m": [0.0163, 0.0211, -0.0316],
    "undesired_contacts_sat_n": 20.0, "undesired_contacts_thresh_n": 1.0,
    "leg_contact_thresh_n": 0.5, "leg_contact_sat_n": 5.0,
}


def test_env_cfg_defaults_keep_old_behaviour():
    d = _cfg_defaults()
    for k, v in NEW_KEYS.items():
        assert d[k] == v, k


def test_schema_and_env_config_carry_the_keys():
    pytest.importorskip("pydantic")
    from config.schema import EnvConfig
    from src.isaaclab import env_config
    e = EnvConfig()
    for k, v in NEW_KEYS.items():
        assert getattr(e, k) == v, k
        assert k in env_config._SCALAR_FIELDS, k


def test_shipped_config_yaml_values():
    yaml = pytest.importorskip("yaml")
    d = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8"))
    t = d["train"]
    ec, w = t["env_config"], t["env_config"]["reward_weights"]
    assert ec["foot_separation_mode"] == "sole_band" and ec["undesired_contacts_sat_n"] == 5.0
    assert (w["foot_flat_l2"], w["foot_separation"], w["undesired_contacts"], w["leg_contact_l2"]) == (-3.0, -0.1, -1.0, -0.5)
    assert (t["caps_temporal_weight"], t["caps_spatial_weight"], t["caps_spatial_sigma"]) == (1.0, 1.0, 0.1)
