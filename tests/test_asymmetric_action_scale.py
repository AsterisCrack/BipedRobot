"""Asymmetric residual action scales: target = home + s_pos*a (a >= 0) / home + s_neg*a (a < 0), clamped to the limits;
inverse a = (q - home)/s_pos if q >= home else (q - home)/s_neg, clipped to [-1, 1]. OFF by default: s_pos == s_neg (a plain
number per joint type) must be bit-identical to the symmetric map in every consumer. CPU only; Isaac itself is NOT imported
(the Isaac env's methods are lifted out of biped_env.py with ast and run on a stand-in `self`).

Run from BipedRobot/:  python -m pytest tests/test_asymmetric_action_scale.py -q
"""
from __future__ import annotations

import ast
import dataclasses
import os
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import Config  # noqa: E402
from utils import residual_home as rh  # noqa: E402
from utils.residual_action import (JOINT_TYPES, actions_to_targets_np, check_residual_map,  # noqa: E402
                                   expand_by_type, home_rad, is_asymmetric_scale, plain_scale_cfg, scale_pair_rad,
                                   scale_rad, targets_to_actions_np)

HOME_DEG = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -10.0, "knee": -26.0, "ankle_roll": 0.0, "ankle_pitch": -16.0}
SCALE = {"hip_yaw": 0.6, "hip_roll": 0.45, "hip_pitch": 0.8, "knee": 0.6, "ankle_roll": 0.45, "ankle_pitch": 0.6}
ASYM = {"hip_yaw": {"pos": 0.45, "neg": 0.65}, "hip_roll": {"pos": 0.65, "neg": 0.20}, "hip_pitch": {"pos": 1.0, "neg": 0.85},
        "knee": {"pos": 0.95, "neg": 0.35}, "ankle_roll": {"pos": 0.20, "neg": 0.65}, "ankle_pitch": {"pos": 0.55, "neg": 0.30}}
LIM_DEG = [[-45, 45], [-45, 45], [-25, 45], [-25, 45], [-90, 30], [-90, 30],
           [-120, 5], [-120, 5], [-60, 60], [-60, 60], [-35, 50], [-35, 50]]
LO, HI = np.radians([lo for lo, _ in LIM_DEG]), np.radians([hi for _, hi in LIM_DEG])
HOME = home_rad(HOME_DEG)
SC = scale_rad(SCALE)
POS, NEG = scale_pair_rad(ASYM)
TABLE = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
REF_YAML = os.path.join(ROOT, "config", "config_gait_reference.yaml")
needs_table = pytest.mark.skipif(not os.path.exists(TABLE), reason="baked table not present")


# ------------------------------------------------------------------------------------------------ parsing
def test_scale_parsing_both_forms():
    pos, neg = scale_pair_rad(SCALE)
    assert neg is None and np.array_equal(pos, SC) and not is_asymmetric_scale(SCALE)
    pos, neg = scale_pair_rad(ASYM)
    assert neg is not None and is_asymmetric_scale(ASYM)
    assert list(pos[6:8]) == [0.95, 0.95] and list(neg[6:8]) == [0.35, 0.35]       # l = r, policy order
    mixed = {**SCALE, "knee": {"pos": 0.9, "neg": 0.4}}                              # numbers and pairs side by side
    pos, neg = scale_pair_rad(mixed)
    assert np.array_equal(pos[:6], SC[:6]) and np.array_equal(neg[:6], SC[:6]) and (pos[6], neg[6]) == (0.9, 0.4)
    same = {**SCALE, "knee": {"pos": 0.6, "neg": 0.6}}                               # pos == neg everywhere = symmetric
    assert scale_pair_rad(same)[1] is None and np.array_equal(scale_rad(same), SC)
    assert plain_scale_cfg(mixed)["hip_yaw"] == 0.6 and plain_scale_cfg(mixed)["knee"] == {"pos": 0.9, "neg": 0.4}
    assert plain_scale_cfg(same) == SCALE
    with pytest.raises(ValueError, match="asymmetric"):
        scale_rad(ASYM)                                                              # the symmetric accessor refuses pairs


@pytest.mark.parametrize("bad", [
    {**SCALE, "knee": {"pos": 0.5}},                                                 # one side only
    {**SCALE, "knee": {"pos": 0.5, "neg": 0.5, "x": 1.0}},
    {**SCALE, "knee": {"pos": 0.5, "neg": 0.0}},
    {**SCALE, "knee": {"pos": -0.5, "neg": 0.5}},
    {**SCALE, "knee": {"pos": float("nan"), "neg": 0.5}},
    {k: v for k, v in ASYM.items() if k != "knee"},                                  # missing type
    {**ASYM, "extra": 0.5},
])
def test_bad_scales_rejected(bad):
    with pytest.raises(ValueError):
        scale_pair_rad(bad)


def test_check_residual_map_checks_both_sides_and_mirror():
    check_residual_map(HOME, POS, LO, HI, scale_neg=NEG)
    with pytest.raises(ValueError, match="> 0"):
        check_residual_map(HOME, POS, LO, HI, scale_neg=np.where(np.arange(12) == 6, 0.0, NEG))
    bad = NEG.copy()
    bad[6] += 0.1                                                                    # left knee neg != right knee neg
    with pytest.raises(ValueError, match="symmetric"):
        check_residual_map(HOME, POS, LO, HI, scale_neg=bad)


# ------------------------------------------------------------------------------------------------ numpy forward / inverse
def test_forward_inverse_round_trip_both_sides_and_clamp():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (2000, 12))
    t = actions_to_targets_np(a, HOME, POS, LO, HI, scale_neg=NEG)
    assert np.all(t >= LO - 1e-12) and np.all(t <= HI + 1e-12)
    assert np.all(((t >= HOME) == (a >= 0)) | (t == HOME) | (t == LO) | (t == HI))     # side follows the sign of a
    inside = (t > LO + 1e-9) & (t < HI - 1e-9)
    back = targets_to_actions_np(t, HOME, POS, scale_neg=NEG)
    assert np.allclose(back[inside], a[inside], atol=1e-9)                           # unclamped entries invert exactly
    # explicit values on each side
    up = actions_to_targets_np(np.full(12, 0.5), HOME, POS, LO, HI, scale_neg=NEG)
    dn = actions_to_targets_np(np.full(12, -0.5), HOME, POS, LO, HI, scale_neg=NEG)
    assert np.allclose(up[:6], HOME[:6] + 0.5 * POS[:6]) and np.allclose(dn[:6], HOME[:6] - 0.5 * NEG[:6])
    assert np.array_equal(actions_to_targets_np(np.zeros(12), HOME, POS, LO, HI, scale_neg=NEG), HOME)
    assert np.array_equal(targets_to_actions_np(HOME, HOME, POS, scale_neg=NEG), np.zeros(12))
    # clamp at the limits: a is clipped to +-1 first, the target to the joint limits after; the inverse clips to [-1, 1]
    top = actions_to_targets_np(np.full(12, 5.0), HOME, POS, LO, HI, scale_neg=NEG)
    bot = actions_to_targets_np(np.full(12, -5.0), HOME, POS, LO, HI, scale_neg=NEG)
    assert np.allclose(top, np.minimum(HOME + POS, HI)) and np.allclose(bot, np.maximum(HOME - NEG, LO))
    assert top[6] == pytest.approx(HI[6])                                            # knee a = +1: -26 deg + 0.95 rad > +5 deg
    assert np.array_equal(targets_to_actions_np(HI + 1.0, HOME, POS, scale_neg=NEG), np.ones(12))
    assert np.array_equal(targets_to_actions_np(LO - 1.0, HOME, POS, scale_neg=NEG), -np.ones(12))
    # a target exactly at home inverts to 0 from either side, and a limit-clamped a = +-1 round trips to >= its clip
    assert np.all(np.abs(targets_to_actions_np(top, HOME, POS, scale_neg=NEG)) <= 1.0)


def test_symmetric_numpy_is_bit_identical_to_the_original_expressions():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1.4, 1.4, (500, 12))
    old_f = np.clip(HOME + SC * np.clip(a, -1.0, 1.0), LO, HI)                       # the pre-asymmetric expressions
    t = old_f + rng.normal(0, 0.1, old_f.shape)
    old_i = np.clip((np.asarray(t) - HOME) / SC, -1.0, 1.0)
    assert np.array_equal(actions_to_targets_np(a, HOME, SC, LO, HI), old_f)
    assert np.array_equal(targets_to_actions_np(t, HOME, SC), old_i)
    # the asymmetric branch with equal sides gives the same bits as well
    assert np.array_equal(actions_to_targets_np(a, HOME, SC, LO, HI, scale_neg=SC.copy()), old_f)
    assert np.array_equal(targets_to_actions_np(t, HOME, SC, scale_neg=SC.copy()), old_i)


# ------------------------------------------------------------------------------------------------ Isaac env methods (lifted)
def _lift(*names):
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in fns) == sorted(names)
    ns = {"torch": torch, "np": np}
    exec(compile(ast.Module(fns, []), "biped_env_methods", "exec"), ns)
    return [ns[n] for n in names], fns


def _isaac_self(scale, scale_neg, dtype=torch.float32):
    t = lambda x: torch.tensor(x, dtype=dtype)
    return SimpleNamespace(action_map="residual", default_joint_pos=t(HOME), action_scale=t(scale),
                           action_scale_neg=None if scale_neg is None else t(scale_neg),
                           joint_limits_min=t(LO), joint_limits_max=t(HI))


def test_isaac_symmetric_is_bit_identical_to_the_original_formula():
    (fwd, inv), _ = _lift("actions_to_targets", "targets_to_actions")
    rng = np.random.default_rng(2)
    a = torch.tensor(rng.uniform(-1.4, 1.4, (256, 12)), dtype=torch.float32)
    t = torch.tensor(rng.uniform(LO - 0.3, HI + 0.3, (256, 12)), dtype=torch.float32)
    for neg in (None, SC):                                                           # no neg tensor / neg == pos
        me = _isaac_self(SC, neg)
        old_f = torch.clamp(me.default_joint_pos + me.action_scale * torch.clamp(a, -1.0, 1.0),
                            me.joint_limits_min, me.joint_limits_max)                # the pre-asymmetric expression
        old_i = torch.clamp((t - me.default_joint_pos) / me.action_scale, -1.0, 1.0)
        assert torch.equal(fwd(me, a), old_f) and torch.equal(inv(me, t), old_i)


def test_isaac_asymmetric_matches_numpy_and_round_trips():
    (fwd, inv), _ = _lift("actions_to_targets", "targets_to_actions")
    me = _isaac_self(POS, NEG)
    a = torch.tensor(np.random.default_rng(3).uniform(-1.3, 1.3, (128, 12)), dtype=torch.float32)
    t = fwd(me, a)
    ref = actions_to_targets_np(a.numpy().astype(np.float64), HOME, POS, LO, HI, scale_neg=NEG)
    assert np.allclose(t.numpy(), ref, atol=1e-6)
    assert np.allclose(fwd(me, torch.zeros(3, 12)).numpy(), HOME, atol=1e-6)
    t64 = t.numpy().astype(np.float64)
    assert np.allclose(inv(me, t).numpy(), targets_to_actions_np(t64, HOME, POS, scale_neg=NEG), atol=1e-5)
    inside = (t64 > LO + 1e-6) & (t64 < HI - 1e-6)
    assert np.allclose(inv(me, t).numpy()[inside], np.clip(a.numpy(), -1, 1)[inside], atol=1e-5)
    # the env's other users of the map: [N, 12] batches of any N, and an unbatched [12] row
    assert fwd(me, a[:1]).shape == (1, 12) and fwd(me, a[0]).shape == (12,)


def test_isaac_methods_stay_vectorised_and_sync_free():
    _, fns = _lift("actions_to_targets", "targets_to_actions")
    for fn in fns:
        src = ast.unparse(fn)
        for banned in (".item(", ".cpu(", ".tolist(", ".numpy(", ".any(", ".all(", "if (", "for "):
            assert banned not in src, (fn.name, banned)
        assert "torch.where" in src                                                  # per-element side choice, no host branching


def test_isaac_init_residual_map_builds_both_tensors():
    (init,), _ = _lift("_init_residual_map")
    names = [("l_" if i % 2 == 0 else "r_") + JOINT_TYPES[i // 2] for i in range(12)]
    for scale_cfg, want_neg in ((SCALE, None), (ASYM, NEG), ({**SCALE, "knee": {"pos": 0.6, "neg": 0.6}}, None)):
        me = SimpleNamespace(
            robot=SimpleNamespace(joint_names=names), device="cpu",
            cfg=SimpleNamespace(action_home_deg=HOME_DEG, action_scale_rad=scale_cfg, mirror_joint_perm=None,
                                mirror_joint_signs=None),
            default_joint_pos=torch.tensor(HOME, dtype=torch.float32),
            joint_limits_min=torch.tensor(LO, dtype=torch.float32), joint_limits_max=torch.tensor(HI, dtype=torch.float32))
        init(me)
        assert np.allclose(me.action_scale.numpy(), scale_pair_rad(scale_cfg)[0], atol=1e-7)
        if want_neg is None:
            assert me.action_scale_neg is None
        else:
            assert np.allclose(me.action_scale_neg.numpy(), want_neg, atol=1e-7) and me.action_scale_neg.dtype == torch.float32


def test_rsi_previous_action_uses_targets_to_actions():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    assert "a_ref = self.targets_to_actions(self.filtered_targets[ids])" in src      # the inverse is side-aware via the method


# ------------------------------------------------------------------------------------------------ env_config plumbing
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


def test_env_config_passes_both_forms_through():
    from src.isaaclab.env_config import _apply_action_home
    conf = lambda sc: SimpleNamespace(action_map="residual", action_home_deg=HOME_DEG, action_scale_rad=sc, spawn_height_m=None)
    cfg = _fake_env_cfg()
    _apply_action_home(cfg, conf(SCALE))
    assert cfg.action_scale_rad == SCALE and all(type(v) is float for v in cfg.action_scale_rad.values())
    cfg = _fake_env_cfg()
    _apply_action_home(cfg, conf(ASYM))
    assert cfg.action_scale_rad == ASYM
    cfg = _fake_env_cfg()                                                            # equal sides collapse to the number
    _apply_action_home(cfg, conf({**SCALE, "knee": {"pos": 0.6, "neg": 0.6}}))
    assert cfg.action_scale_rad == SCALE
    with pytest.raises(ValueError):                                                  # a malformed pair is refused
        _apply_action_home(_fake_env_cfg(), conf({**SCALE, "knee": {"pos": 0.6}}))


# ------------------------------------------------------------------------------------------------ MuJoCo env
def _mj_env(scale_cfg):
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    ec = Config(REF_YAML).train.env_config
    ec.action_map = "residual"
    ec.action_home_deg, ec.action_scale_rad, ec.spawn_height_m = dict(HOME_DEG), scale_cfg, None
    ec.backlash = {**(ec.backlash or {}), "enabled": False}                          # MuJoCo backlash stays disabled
    return BipedEnvV2(history_size=1, use_history=False, env_config=ec)


def test_mujoco_symmetric_is_bit_identical_and_asymmetric_follows_the_formula():
    pytest.importorskip("mujoco")
    from envs.mujoco.biped_env_v2 import JLIM_MAX, JLIM_MIN
    sym = _mj_env(dict(SCALE))
    assert sym._residual_scale_neg is None
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.uniform(-1.4, 1.4, 12).astype(np.float32)
        old = np.clip(sym._residual_home + sym._residual_scale * np.clip(a, -1.0, 1.0), JLIM_MIN, JLIM_MAX)
        assert np.array_equal(sym._actions_to_targets(a), old)
    asym = _mj_env(ASYM)
    assert np.array_equal(asym._residual_scale, POS) and np.array_equal(asym._residual_scale_neg, NEG)
    asym.reset(seed=0)
    assert np.allclose(asym._jpos(), HOME, atol=1e-6)                                # reset pose = home, whatever the scales
    assert np.array_equal(asym._actions_to_targets(np.zeros(12, dtype=np.float32)), np.clip(HOME, JLIM_MIN, JLIM_MAX))
    for _ in range(10):
        a = rng.uniform(-1.4, 1.4, 12).astype(np.float32)
        want = actions_to_targets_np(a, asym._residual_home, POS, JLIM_MIN, JLIM_MAX, scale_neg=NEG)
        assert np.array_equal(asym._actions_to_targets(a), want)
        obs, *_ = asym.step(a)
        assert np.all(np.isfinite(obs["critic"]))


# ------------------------------------------------------------------------------------------------ resolver
def _synthetic_table(dev_pos, dev_neg, home12, frames=8):
    """One baked cell whose joints swing up to +dev_pos / down to -dev_neg (rad, per joint type) around the home."""
    swing = np.zeros((frames, 12))
    for i in range(12):
        swing[0, i], swing[1, i] = dev_pos[i // 2], -dev_neg[i // 2]
    q = (np.asarray(home12) + swing)[None, None, None]
    return {"grids": (np.array([0.0]), np.array([0.0]), np.array([0.0])), "q": q}


def test_resolver_per_side_minimality_and_coverage_synthetic():
    lim = np.stack([LO, HI], axis=1)
    noise, headroom = 0.05, 0.96
    dev_pos = np.array([0.10, 0.30, 0.52, 0.0, 0.17, 0.80])
    dev_neg = np.array([0.30, 0.0, 0.10, 0.44, 0.17, 0.20])
    tab = _synthetic_table(dev_pos, dev_neg, HOME, 6)
    rs = rh.required_scales_asym(tab, HOME, [0.0, 0.0, 0.0], noise, headroom, lim)
    for side, dev in (("pos", dev_pos), ("neg", dev_neg)):
        s = rs[f"scale_{side}"]
        assert np.allclose(rs[f"dev_rad_{side}"], dev)
        assert np.all(dev / s + noise / s <= headroom + 1e-9)                       # coverage with the RSI noise
        smaller = s - rh.SCALE_STEP
        floor = s <= rh.SCALE_STEP + 1e-12
        assert np.all(floor | (dev / smaller + noise / smaller > headroom))         # one 0.05 step less would not cover
        assert np.allclose(np.round(s / rh.SCALE_STEP) * rh.SCALE_STEP, s)           # on the 0.05 grid
    sym = rh.required_scales(tab, HOME, [0.0, 0.0, 0.0], noise, headroom, lim)
    assert np.array_equal(np.maximum(rs["scale_pos"], rs["scale_neg"]), sym["scale"])   # the old rule = the wider side
    assert np.all(rs["scale_pos"] <= sym["scale"] + 1e-12) and np.all(rs["scale_neg"] <= sym["scale"] + 1e-12)


def test_resolver_clamps_the_table_to_the_limits_per_side():
    lim = np.stack([LO, HI], axis=1)
    big = np.full(6, 3.0)                                                            # the table asks for far more than the limit
    tab = _synthetic_table(big, big, HOME, 4)
    rs = rh.required_scales_asym(tab, HOME, [0.0, 0.0, 0.0], 0.05, 0.96, lim)
    assert np.allclose(rs["dev_rad_pos"], (HI - HOME)[0::2]) and np.allclose(rs["dev_rad_neg"], (HOME - LO)[0::2])


def _ec(mode=None, **kw):
    base = dict(action_map="residual", action_home_deg="auto", action_scale_rad="auto", spawn_height_m=0.286,
                reference={"npz_path": TABLE, "rsi_joint_noise": 0.05},
                commands={"base_velocity": {"ranges": {"lin_vel_x": [-0.2, 0.2], "lin_vel_y": [-0.15, 0.15],
                                                        "ang_vel_z": [-1.2, 1.2]}}})
    if mode is not None:
        base["action_scale_mode"] = mode
    return SimpleNamespace(**{**base, **kw})


@needs_table
def test_resolver_current_table_per_side_minimality_coverage_and_default_mode():
    home, scale_default, _, rep0 = rh.resolve_residual_map(_ec(), ROOT)
    home_s, scale_sym, _, _ = rh.resolve_residual_map(_ec("symmetric"), ROOT)
    assert scale_default == scale_sym and "scale_mode" not in rep0                  # default = today's symmetric rule
    assert all(isinstance(v, float) for v in scale_sym.values())
    home_a, scale_asym, _, rep = rh.resolve_residual_map(_ec("asymmetric"), ROOT)
    assert home_a == home and rep["scale_mode"] == "asymmetric"
    s = rep["scale"]
    # independent recomputation of each side's deviation straight from the table
    tab = rh.load_table(TABLE, ROOT)
    bounds, _ = rh.command_bounds(_ec(), tab["grids"])
    masks = [rh.select_grid_cells(g, b) for g, b in zip(tab["grids"], bounds)]
    q = tab["q"][masks[0]][:, masks[1]][:, :, masks[2]].reshape(-1, 12)
    lim = np.radians(rh.joint_limits_deg(ROOT))
    d = np.clip(q, lim[:, 0], lim[:, 1]) - home_rad(home)
    for i, t in enumerate(JOINT_TYPES):
        dev = {"pos": max(d[:, 2 * i].max(), d[:, 2 * i + 1].max(), 0.0), "neg": max((-d[:, 2 * i]).max(), (-d[:, 2 * i + 1]).max(), 0.0)}
        for side in ("pos", "neg"):
            sc = scale_asym[t][side]
            assert dev[side] == pytest.approx(s[f"dev_rad_{side}"][i], abs=1e-12)
            assert dev[side] / sc + 0.05 / sc <= 0.96 + 1e-9, (t, side)             # coverage incl. RSI noise
            lower = sc - rh.SCALE_STEP
            assert sc <= rh.SCALE_STEP + 1e-9 or dev[side] / lower + 0.05 / lower > 0.96, (t, side)   # minimal
            assert sc <= scale_sym[t] + 1e-9                                         # never wider than the symmetric value
        assert max(scale_asym[t].values()) == pytest.approx(scale_sym[t])
    check_residual_map(home_rad(home_a), *scale_pair_rad(scale_asym)[:1], LO, HI, scale_neg=scale_pair_rad(scale_asym)[1])


@needs_table
def test_resolver_modes_validation_and_explicit_passthrough():
    with pytest.raises(ValueError, match="action_scale_mode"):
        rh.resolve_residual_map(_ec("sideways"), ROOT)
    # explicit scales are used as written in either mode (a note says the mode did not apply); pairs pass through
    ec = _ec("asymmetric", action_home_deg=dict(HOME_DEG), action_scale_rad=dict(ASYM))
    home, scale, _, rep = rh.resolve_residual_map(ec, ROOT)
    assert scale == ASYM and any("only applies" in n for n in rep["notes"])
    ec = _ec("asymmetric", action_home_deg=dict(HOME_DEG), action_scale_rad=dict(SCALE))
    assert rh.resolve_residual_map(ec, ROOT)[1] == SCALE
    with pytest.raises(ValueError):
        rh.resolve_residual_map(_ec(action_home_deg=dict(HOME_DEG), action_scale_rad={**SCALE, "knee": {"pos": 0.5}}), ROOT)
    assert "+0.95/-0.35" in rh.summary(HOME_DEG, ASYM, 0.286) and "knee=0.6 " in rh.summary(HOME_DEG, SCALE, 0.286) + " "


@needs_table
def test_freeze_writes_pos_neg_dicts_and_the_run_config_round_trips(tmp_path):
    cfg = Config(REF_YAML)
    ec = cfg.train.env_config
    assert ec.action_scale_mode == "symmetric"                                       # the reference config does not switch it on
    ec.action_scale_mode = "asymmetric"
    rep = rh.freeze_residual_map(cfg, ROOT)
    assert rep is not None and not rh.any_auto(ec)
    assert all(set(v) == {"pos", "neg"} for v in ec.action_scale_rad.values())
    assert all(isinstance(ec.action_home_deg[t], float) for t in JOINT_TYPES)
    saved = tmp_path / "config.yaml"
    cfg.copy_in_file(str(saved))                                                     # what train.py saves into the run dir
    doc = yaml.safe_load(saved.read_text())["train"]["env_config"]
    assert doc["action_scale_rad"] == {t: dict(ec.action_scale_rad[t]) for t in JOINT_TYPES}
    # export reads that file back as plain numbers: block-style nested dicts
    from src.isaaclab import export_onnx as ex
    c = ex._load_policy_contract(str(tmp_path), None, None)
    pos, neg = scale_pair_rad(ec.action_scale_rad)
    assert np.array_equal(c["residual"]["scale_rad"], pos) and np.array_equal(c["residual"]["scale_neg_rad"], neg)
    assert ex._format_version(c, {"mode": "fixed_period"}) == 6
    assert ex._format_version(c, {"mode": "cadence_grid"}) == 6                       # v6 wins over v5: the scale keys changed
    # the same Config, symmetric: no neg scale, the old format versions
    cfg2 = Config(REF_YAML)
    rh.freeze_residual_map(cfg2, ROOT)
    assert all(isinstance(v, float) for v in cfg2.train.env_config.action_scale_rad.values())
    cfg2.copy_in_file(str(saved))
    c2 = ex._load_policy_contract(str(tmp_path), None, None)
    assert c2["residual"]["scale_neg_rad"] is None
    assert ex._format_version(c2, {"mode": "fixed_period"}) == 4 and ex._format_version(c2, {"mode": "cadence_grid"}) == 5


# ------------------------------------------------------------------------------------------------ export contract (no checkpoint)
def _run_cfg(tmp_path, scale_yaml):
    (tmp_path / "config.yaml").write_text(textwrap.dedent("""\
        train:
          env_config:
            action_map: residual
            action_home_deg: {hip_yaw: 0.0, hip_roll: 0.0, hip_pitch: -10.0, knee: -26.0, ankle_roll: 0.0, ankle_pitch: -16.0}
            action_scale_rad: %s
        """) % scale_yaml)


def test_export_contract_flow_and_inline_forms(tmp_path):
    from src.isaaclab import export_onnx as ex
    _run_cfg(tmp_path, "{hip_yaw: 0.6, hip_roll: 0.45, hip_pitch: 0.8, knee: 0.6, ankle_roll: 0.45, ankle_pitch: 0.6}")
    c = ex._load_policy_contract(str(tmp_path), None, None)
    assert c["residual"]["scale_neg_rad"] is None and c["residual"]["scale_rad"] == [float(x) for x in SC]
    assert ex._format_version(c, {"mode": "fixed_period"}) == 4 and ex._format_version(c, {"mode": "cadence_grid"}) == 5
    _run_cfg(tmp_path, "{hip_yaw: {pos: 0.45, neg: 0.65}, hip_roll: {pos: 0.65, neg: 0.2}, hip_pitch: {pos: 1.0, neg: 0.85}, "
                       "knee: {pos: 0.95, neg: 0.35}, ankle_roll: {pos: 0.2, neg: 0.65}, ankle_pitch: {pos: 0.55, neg: 0.3}}")
    c = ex._load_policy_contract(str(tmp_path), None, None)
    assert c["residual"]["scale_rad"] == [float(x) for x in POS] and c["residual"]["scale_neg_rad"] == [float(x) for x in NEG]
    assert ex._format_version(c, {"mode": "fixed_period"}) == 6
    assert ex._format_version({"action_map": "centered"}, {"mode": "cadence_grid"}) == 5       # only residual + neg gives v6


# ------------------------------------------------------------------------------------------------ schema
def test_schema_accepts_both_forms_and_the_mode():
    from config.schema import EnvConfig
    e = EnvConfig(action_map="residual", action_home_deg=HOME_DEG, action_scale_rad=SCALE)
    assert e.action_scale_mode == "symmetric" and e.action_scale_rad == SCALE
    e = EnvConfig(action_map="residual", action_home_deg=HOME_DEG, action_scale_rad=ASYM, action_scale_mode="asymmetric")
    assert e.action_scale_rad == ASYM and e.action_scale_mode == "asymmetric"
    e = EnvConfig(action_scale_rad={**SCALE, "knee": {"pos": 0.9, "neg": 0.4}})                # mixed
    assert e.action_scale_rad["knee"] == {"pos": 0.9, "neg": 0.4} and e.action_scale_rad["hip_yaw"] == 0.6
    assert EnvConfig(action_scale_rad="auto").action_scale_rad == "auto"
    assert EnvConfig().action_scale_rad is None and EnvConfig().action_scale_mode == "symmetric"
    with pytest.raises(Exception):
        EnvConfig(action_scale_mode="both")
    with pytest.raises(Exception):
        EnvConfig(action_scale_rad={**SCALE, "knee": {"pos": "x"}})
    # round trip through a saved config
    dumped = yaml.safe_load(yaml.safe_dump(e.model_dump(mode="json")))
    assert EnvConfig(**dumped).action_scale_rad == e.action_scale_rad


def test_configs_in_the_repo_keep_the_symmetric_default():
    ref = Config(REF_YAML).train.env_config
    assert ref.action_scale_rad == "auto" and ref.action_scale_mode == "symmetric"   # present, NOT switched on
    base = Config(os.path.join(ROOT, "config", "config.yaml")).train.env_config
    # config.yaml (no table) uses explicit {pos, neg} scales since 2026-10-25; the mode only matters for `auto`
    assert base.action_scale_mode == "symmetric" and is_asymmetric_scale(dict(base.action_scale_rad))
    assert "action_scale_mode: symmetric" in open(REF_YAML, encoding="utf-8").read()
