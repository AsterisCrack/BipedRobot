"""action_map: residual -- the home pose, the map, its inverse, the config plumbing, the MuJoCo env
and the bundle exporter. CPU only; Isaac itself is NOT imported (the Isaac env's two map methods
are lifted out of biped_env.py with ast and run on a stand-in `self`).

Run from BipedRobot/:  python -m pytest tests/test_residual_action_map.py -q
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import math
import os
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.residual_action import (RESIDUAL_SPAWN_HEIGHT_M, actions_to_targets_np,  # noqa: E402
                                   check_residual_map, expand_by_type, home_rad, scale_rad,
                                   targets_to_actions_np)

HOME_DEG = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -10.0, "knee": -26.0,
            "ankle_roll": 0.0, "ankle_pitch": -16.0}
SCALE = {"hip_yaw": 0.6, "hip_roll": 0.45, "hip_pitch": 0.8, "knee": 0.6,
         "ankle_roll": 0.45, "ankle_pitch": 0.6}
# config/config_gait_reference.yaml: home = the v2 baked table's stand pose (rounded); the scales are the user's choice,
# sized so that every baked cell inside the training command ranges (now the whole v2 grid) AND the RSI joint noise stay
# inside [-1, 1] (ankle_pitch 0.45 is a deliberate floor above the 0.35 the table needs)
REF_HOME_DEG = {"hip_yaw": 2.1, "hip_roll": -6.2, "hip_pitch": -17.7, "knee": -36.6,
                "ankle_roll": 6.5, "ankle_pitch": -18.7}
REF_SCALE = {"hip_yaw": 0.55, "hip_roll": 0.55, "hip_pitch": 0.95, "knee": 0.90,
             "ankle_roll": 0.60, "ankle_pitch": 0.45}
REF_SPAWN_HEIGHT_M = 0.28                        # same ~5.4 mm sole clearance as the previous home at 0.286
LIM_DEG = [[-45, 45], [-45, 45], [-25, 45], [-25, 45], [-90, 30], [-90, 30],
           [-120, 5], [-120, 5], [-60, 60], [-60, 60], [-35, 50], [-35, 50]]
LO = np.radians([lo for lo, _ in LIM_DEG])
HI = np.radians([hi for _, hi in LIM_DEG])
HOME, SC = home_rad(HOME_DEG), scale_rad(SCALE)


def test_expand_order_and_flat_sole():
    h = expand_by_type(HOME_DEG, "x")
    assert h == [0, 0, 0, 0, -10, -10, -26, -26, 0, 0, -16, -16]
    # Flat sole: hip_pitch - knee + ankle_pitch = 0 (URDF axes)
    assert h[4] - h[6] + h[10] == 0.0
    with pytest.raises(ValueError):
        expand_by_type({**HOME_DEG, "extra": 1.0}, "x")
    with pytest.raises(ValueError):
        expand_by_type({k: v for k, v in HOME_DEG.items() if k != "knee"}, "x")


def test_forward_inverse_roundtrip_and_home():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (500, 12))
    t = actions_to_targets_np(a, HOME, SC, LO, HI)
    assert np.all(t >= LO - 1e-12) and np.all(t <= HI + 1e-12)
    inside = (t > LO + 1e-9) & (t < HI - 1e-9)                  # unclamped entries invert exactly
    back = targets_to_actions_np(t, HOME, SC)
    assert np.allclose(back[inside], a[inside], atol=1e-9)
    assert np.allclose(actions_to_targets_np(np.zeros(12), HOME, SC, LO, HI), HOME)
    assert np.allclose(targets_to_actions_np(HOME, HOME, SC), 0.0)


def test_clamp_at_limits_and_action_clip():
    t = actions_to_targets_np(np.full(12, 5.0), HOME, SC, LO, HI)       # a clipped to 1 first
    assert np.allclose(t, np.minimum(HOME + SC, HI))
    assert t[6] == pytest.approx(HI[6])                                  # knee: -26 deg + 0.6 rad > +5 deg
    t = actions_to_targets_np(np.full(12, -1.0), HOME, SC, LO, HI)
    assert np.all(t >= LO - 1e-12)
    assert np.allclose(targets_to_actions_np(HI + 1.0, HOME, SC), 1.0)  # inverse clamps to [-1, 1]
    assert np.allclose(targets_to_actions_np(LO - 1.0, HOME, SC), -1.0)


def test_symmetric_home_assert():
    check_residual_map(HOME, SC, LO, HI)
    bad = HOME.copy()
    bad[6] += 0.1                                                        # left knee != right knee
    with pytest.raises(ValueError, match="symmetric"):
        check_residual_map(bad, SC, LO, HI)
    with pytest.raises(ValueError, match="inside"):
        check_residual_map(HI, SC, LO, HI)
    with pytest.raises(ValueError):
        scale_rad({**SCALE, "knee": 0.0})
    assert np.all(HOME[0::2] == HOME[1::2])                              # home[l] == home[r]


def _lift_methods(*names):
    """actions_to_targets / targets_to_actions out of the real biped_env.py source."""
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"torch": torch}
    exec(compile(ast.Module(fns, []), "biped_env_methods", "exec"), ns)
    return [ns[n] for n in names]


def test_isaac_env_methods_match_numpy_formula():
    fwd, inv = _lift_methods("actions_to_targets", "targets_to_actions")
    f32 = torch.float32
    me = SimpleNamespace(
        action_map="residual", default_joint_pos=torch.tensor(HOME, dtype=f32),
        action_scale=torch.tensor(SC, dtype=f32),
        joint_limits_min=torch.tensor(LO, dtype=f32), joint_limits_max=torch.tensor(HI, dtype=f32))
    a = torch.tensor(np.random.default_rng(1).uniform(-1.3, 1.3, (64, 12)), dtype=f32)
    t = fwd(me, a)
    ref = actions_to_targets_np(a.numpy().astype(np.float64), HOME, SC, LO, HI)
    assert np.allclose(t.numpy(), ref, atol=1e-6)
    assert np.allclose(fwd(me, torch.zeros(3, 12)).numpy(), HOME, atol=1e-6)
    assert np.allclose(inv(me, t).numpy(), targets_to_actions_np(t.numpy().astype(np.float64), HOME, SC), atol=1e-5)


def test_isaac_env_accepts_residual_and_checks_home():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    assert '("centered", "absolute", "residual")' in src
    assert "check_residual_map" in src and "default_joint_pos differs from action_home_deg" in src


# ---------------------------------------------------------------- env_config plumbing (no Isaac)
@dataclasses.dataclass
class _Init:
    pos: tuple = (0.0, 0.0, 0.29)
    joint_pos: dict = dataclasses.field(default_factory=lambda: {".*": 0.0})

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class _Robot:
    init_state: _Init = dataclasses.field(default_factory=_Init)

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def _fake_env_cfg():
    return SimpleNamespace(scene=SimpleNamespace(robot=_Robot()), joint_limits=np.radians(LIM_DEG).tolist(),
                           mirror_joint_perm=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
                           mirror_joint_signs=[1.0] * 12, target_base_height=0.29)


def _conf(**kw):
    base = dict(action_map="absolute", action_home_deg=None, action_scale_rad=None, spawn_height_m=None)
    return SimpleNamespace(**{**base, **kw})


def test_env_config_writes_home_into_init_state():
    from src.isaaclab.env_config import _apply_action_home
    cfg = _fake_env_cfg()
    _apply_action_home(cfg, _conf(action_map="residual", action_home_deg=HOME_DEG, action_scale_rad=SCALE))
    jp = cfg.scene.robot.init_state.joint_pos
    assert len(jp) == 12 and jp["l_knee.*"] == jp["r_knee.*"] == pytest.approx(math.radians(-26))
    assert jp["l_ankle_pitch.*"] == pytest.approx(math.radians(-16))
    assert jp["r_hip_pitch.*"] == pytest.approx(math.radians(-10))
    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, RESIDUAL_SPAWN_HEIGHT_M)
    assert cfg.target_base_height == RESIDUAL_SPAWN_HEIGHT_M == pytest.approx(0.29 - 0.0039, abs=1e-4)
    cfg = _fake_env_cfg()                                                # explicit spawn height wins
    _apply_action_home(cfg, _conf(action_map="residual", action_home_deg=HOME_DEG, action_scale_rad=SCALE,
                                  spawn_height_m=0.3))
    assert cfg.scene.robot.init_state.pos[2] == 0.3


def test_env_config_leaves_other_maps_untouched():
    from src.isaaclab.env_config import _apply_action_home
    for amap in ("centered", "absolute"):
        cfg = _fake_env_cfg()
        before = copy.deepcopy(cfg.scene.robot)
        _apply_action_home(cfg, _conf(action_map=amap))
        assert cfg.scene.robot == before and cfg.target_base_height == 0.29
    with pytest.raises(ValueError):                       # residual without its keys
        _apply_action_home(_fake_env_cfg(), _conf(action_map="residual"))
    with pytest.raises(ValueError):                       # keys without residual
        _apply_action_home(_fake_env_cfg(), _conf(action_map="centered", action_home_deg=HOME_DEG,
                                                  action_scale_rad=SCALE))


def test_config_yaml_parses_residual_defaults():
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config.yaml")).train.env_config
    assert ec.action_map == "residual"
    assert dict(ec.action_home_deg) == HOME_DEG and dict(ec.action_scale_rad) == SCALE
    assert ec.spawn_height_m == pytest.approx(0.286)
    # the gait-table config is residual too, with the baked table's (rounded) STAND pose as home
    ec2 = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    assert ec2.action_map == "residual" and ec2.spawn_height_m == pytest.approx(REF_SPAWN_HEIGHT_M)
    assert dict(ec2.action_home_deg) == REF_HOME_DEG and dict(ec2.action_scale_rad) == REF_SCALE
    h = expand_by_type(ec2.action_home_deg, "x")
    assert abs(h[4] - h[6] + h[10]) < 0.5 and abs(h[2] + h[8]) < 0.5               # flat sole (pitch and roll), v2 stand: 0.2 / 0.3 deg
    check_residual_map(home_rad(ec2.action_home_deg), scale_rad(ec2.action_scale_rad), LO, HI)


REAL_TABLE = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")


def _cells_in_range(grid, bound):
    """Grid points a command with |c| <= bound can interpolate from: every |g| <= bound plus, when the bound is not itself a
    grid point, the first grid point beyond it (trilinear sampling blends the two neighbours)."""
    grid = np.asarray(grid, dtype=np.float64)
    keep = np.abs(grid) <= bound + 1e-6
    beyond = np.abs(grid)[~keep]
    if beyond.size and not np.any(np.isclose(np.abs(grid), bound, atol=1e-6)):
        keep |= np.isclose(np.abs(grid), beyond.min(), atol=1e-6)
    return keep


@pytest.mark.skipif(not os.path.exists(REAL_TABLE), reason="baked table not present")
def test_reference_config_map_covers_every_baked_cell():
    """The gait-table config's residual map must express every baked cell inside the TRAINING command ranges (both legs,
    clamped to the joint limits) with room for the RSI joint noise: per joint, max |a| + rsi_joint_noise / scale < 1.0, i.e.
    reference-state initialisation never needs a clamped action (the real constraint; the old fixed 0.86 bound was a proxy
    for it). Re-baking with a wider style, or widening the command ranges, fails this, not training."""
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    cmd = ec.commands["base_velocity"]["ranges"]
    z = np.load(REAL_TABLE, allow_pickle=False)
    names = [str(s) for s in z["field_names"]]
    assert names[0] == "joint_pos"
    bounds = {k: max(abs(float(v)) for v in cmd[c]) for k, c in
              (("grid_vx", "lin_vel_x"), ("grid_vy", "lin_vel_y"), ("grid_wz", "ang_vel_z"))}
    mx, my, mw = (_cells_in_range(z[k], bounds[k]) for k in ("grid_vx", "grid_vy", "grid_wz"))
    assert mx.any() and my.any() and mw.any()
    q = z["data"][..., :12].astype(np.float64)[mx][:, my][:, :, mw].reshape(-1, 12)
    home, sc = home_rad(REF_HOME_DEG), scale_rad(REF_SCALE)
    a = (np.clip(q, LO, HI) - home) / sc
    noise = float(ec.reference["rsi_joint_noise"])                                 # rad, added to the RSI joint targets
    need = np.abs(a).max(axis=0) + noise / sc
    assert np.all(np.abs(a).max(axis=0) <= 1.0) and np.all(need < 1.0), (np.abs(a).max(axis=0), need)
    stand = (z["stand"][:12].astype(np.float64) - home) / sc
    assert np.abs(stand).max() < 0.05                                              # a = 0 is the table's stand pose


# ---------------------------------------------------------------- MuJoCo env
mujoco = pytest.importorskip("mujoco")
JPOS_REL = slice(12, 24)


def _jl():
    from envs.mujoco.biped_env_v2 import JLIM_MAX, JLIM_MIN
    return JLIM_MIN.astype(np.float64), JLIM_MAX.astype(np.float64)


def _mj_env(action_map="residual"):
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    ec.action_map = action_map
    if action_map == "residual":
        ec.action_home_deg, ec.action_scale_rad, ec.spawn_height_m = dict(HOME_DEG), dict(SCALE), None
    else:                                           # the reference config is residual itself: drop its map keys
        ec.action_home_deg, ec.action_scale_rad, ec.spawn_height_m = None, None, None
    return BipedEnvV2(history_size=1, use_history=False, env_config=ec)


def test_mujoco_reset_is_home_and_obs_rel_zero():
    env = _mj_env()
    obs, _ = env.reset(seed=0)
    assert np.allclose(env._jpos(), HOME, atol=1e-6)
    assert np.allclose(env._default_joint_pos, HOME, atol=1e-6)
    assert np.allclose(obs["critic"][JPOS_REL], 0.0, atol=1e-6)
    assert np.allclose(env._actions_to_targets(np.zeros(12)), HOME, atol=1e-6)
    assert np.allclose(env.filtered_targets, HOME, atol=1e-6)
    ref = _mj_env("centered")
    ref.reset(seed=0)
    assert float(env._qpos()[2]) - float(ref._qpos()[2]) == pytest.approx(RESIDUAL_SPAWN_HEIGHT_M - 0.29, abs=1e-6)
    assert np.allclose(ref._jpos(), 0.0, atol=1e-6)                      # other maps unchanged


def test_mujoco_targets_follow_training_formula():
    env = _mj_env()
    env.reset(seed=0)
    lo, hi = _jl()
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.uniform(-1.2, 1.2, 12).astype(np.float32)
        assert np.allclose(env._actions_to_targets(np.clip(a, -1, 1)),
                           actions_to_targets_np(a, HOME, SC, lo, hi), atol=1e-5)
        obs, *_ = env.step(a)
        assert np.all(np.isfinite(obs["critic"]))


def test_mujoco_centered_unchanged():
    env = _mj_env("centered")
    env.reset(seed=0)
    assert np.all(env._default_joint_pos == 0.0)
    t = env._actions_to_targets(np.full(12, 0.5, dtype=np.float32))
    assert np.allclose(t, 0.5 * _jl()[1], atol=1e-5)


def test_mujoco_residual_requires_keys():
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    ec.action_map = "residual"
    ec.action_home_deg, ec.action_scale_rad = None, None                 # the reference config carries them now
    with pytest.raises(ValueError):
        BipedEnvV2(env_config=ec)


# ---------------------------------------------------------------- exporter (torch only, no Isaac)
def test_export_contract_and_format_version(tmp_path):
    from src.isaaclab import export_onnx as ex
    (tmp_path / "config.yaml").write_text(textwrap.dedent("""\
        train:
          env_config:
            action_map: residual   # comment
            action_home_deg:
              hip_yaw: 0.0
              hip_roll: 0.0
              hip_pitch: -10.0
              knee: -26.0
              ankle_roll: 0.0
              ankle_pitch: -16.0
            action_scale_rad: {hip_yaw: 0.6, hip_roll: 0.45, hip_pitch: 0.8, knee: 0.6, ankle_roll: 0.45, ankle_pitch: 0.6}
        """))
    c = ex._load_policy_contract(str(tmp_path), None, None)
    assert c["action_map"] == "residual"
    assert np.allclose(c["residual"]["home_rad"], HOME) and np.allclose(c["residual"]["scale_rad"], SC)
    assert ex._format_version(c, {"mode": "fixed_period"}) == 4
    assert ex._format_version(c, {"mode": "speed_proportional"}) == 4
    assert ex._format_version({"action_map": "centered"}, {"mode": "fixed_period"}) == 3
    assert ex._format_version({"action_map": "centered"}, {"mode": "speed_proportional"}) == 2
    assert ex._format_version({"action_map": "absolute"}, {"mode": "speed_proportional"}) == 2
    # unchanged maps keep the old contract (no "residual" entry)
    (tmp_path / "config.yaml").write_text("action_map: centered\n")
    assert "residual" not in ex._load_policy_contract(str(tmp_path), None, None)
