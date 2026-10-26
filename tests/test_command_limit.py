"""Elliptical (vx, vy) command limit: utils/command_limit.py and everything that applies it.

CPU only, no Isaac Sim (the Isaac command sampler is loaded with the Isaac-only imports stubbed).
Run from BipedRobot/:  python -m pytest tests/test_command_limit.py -q
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.command_limit import SHAPES, limit_command, sample_in_limit, validate_shape  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "data", "command_limit_golden.json")
R_REF = {"lin_vel_x": (-0.2, 0.2), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-1.2, 1.2)}


# --------------------------------------------------------------------------- the function
def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def test_golden_numpy_and_torch_float64():
    g = _golden()
    assert len(g["cases"]) == 2 * 2 * 13
    for c in g["cases"]:
        ranges, want = g[c["ranges"]], np.asarray(c["out"])
        got = limit_command(np.asarray(c["cmd"]), ranges, c["shape"])
        assert np.abs(got - want).max() < 1e-12, c
        got_t = limit_command(torch.tensor(c["cmd"], dtype=torch.float64), ranges, c["shape"])
        assert np.abs(got_t.numpy() - want).max() < 1e-12, c
    # batched, same numbers
    sel = [c for c in g["cases"] if c["ranges"] == "R2" and c["shape"] == "ellipse_xy"]
    cmds, want = np.asarray([c["cmd"] for c in sel]), np.asarray([c["out"] for c in sel])
    assert np.abs(limit_command(cmds, g["R2"], "ellipse_xy") - want).max() < 1e-12
    assert np.abs(limit_command(torch.as_tensor(cmds), g["R2"], "ellipse_xy").numpy() - want).max() < 1e-12


def test_golden_file_is_reproducible_from_the_fixed_input_set(tmp_path):
    """The Jetson / Gait Studio copies regenerate the file from the same inputs."""
    spec = importlib.util.spec_from_file_location(
        "make_golden", os.path.join(ROOT, "tests", "data", "make_command_limit_golden.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "g.json"
    mod.main(str(out))
    assert json.load(open(out)) == _golden()
    assert open(out, "rb").read() == open(GOLDEN, "rb").read()


def test_ellipse_semantics():
    def f(c, r=R_REF, s="ellipse_xy"):
        return limit_command(np.asarray(c, dtype=float), r, s)

    assert np.allclose(f((0.2, 0.15, 0.0)), (0.2 / math.sqrt(2), 0.15 / math.sqrt(2), 0.0))   # corner -> 71%
    assert np.allclose(f((0.2, 0.0, 0.5)), (0.2, 0.0, 0.5))                                  # axes untouched
    assert np.allclose(f((0.0, 0.15, 0.0)), (0.0, 0.15, 0.0))
    assert np.allclose(f((0.3, 0.3, 2.0)), (0.2 / math.sqrt(2), 0.15 / math.sqrt(2), 1.2))   # box clamp, then ellipse
    d = f((0.2, 0.15, 0.0))
    assert d[1] / d[0] == pytest.approx(0.15 / 0.2)                                          # direction kept
    # asymmetric vx: backward bound for vx < 0, forward for vx >= 0
    r2 = {"lin_vel_x": (-0.1, 0.3), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-1.0, 1.0)}
    assert np.allclose(f((-0.1, 0.15, 0), r2), (-0.1 / math.sqrt(2), 0.15 / math.sqrt(2), 0))
    assert np.allclose(f((0.3, -0.15, 0), r2), (0.3 / math.sqrt(2), -0.15 / math.sqrt(2), 0))


def test_box_is_clamp_only_and_zero_bound_is_guarded():
    assert np.allclose(limit_command(np.array([0.5, -0.5, 3.0]), R_REF, "box"), (0.2, -0.15, 1.2))
    r = {"lin_vel_x": (-0.2, 0.0), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-1.0, 1.0)}   # hi_x = 0: no forward
    out = limit_command(np.array([[0.1, 0.1, 0.0], [-0.2, 0.15, 0.0]]), r, "ellipse_xy")
    assert np.isfinite(out).all() and np.allclose(out[0], (0.0, 0.1, 0.0))
    assert np.allclose(out[1], (-0.2 / math.sqrt(2), 0.15 / math.sqrt(2), 0.0))
    t = limit_command(torch.tensor([[0.1, 0.1, 0.0]]), r, "ellipse_xy")
    assert torch.isfinite(t).all() and torch.allclose(t, torch.tensor([[0.0, 0.1, 0.0]]))


def test_shape_validation():
    assert validate_shape(None) == "box" and set(SHAPES) == {"box", "ellipse_xy"}
    for bad in ("ellipse", "Box", "", 3):
        with pytest.raises(ValueError):
            validate_shape(bad)
    with pytest.raises(ValueError):
        limit_command(np.zeros(3), R_REF, "circle")
    with pytest.raises(ValueError):
        sample_in_limit(4, R_REF, "circle")


def test_limit_is_idempotent_and_dtype_preserving():
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, (500, 3))
    once = limit_command(c, R_REF, "ellipse_xy")
    assert np.allclose(limit_command(once, R_REF, "ellipse_xy"), once, atol=1e-12)
    t32 = limit_command(torch.as_tensor(c, dtype=torch.float32), R_REF, "ellipse_xy")
    assert t32.dtype == torch.float32 and np.allclose(t32.numpy(), once, atol=1e-6)
    assert limit_command(np.array([1, 1, 1]), R_REF, "box").dtype == np.float64   # int in


@pytest.mark.parametrize("ranges", [R_REF, {"lin_vel_x": (-0.1, 0.3), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.5, 0.5)}])
def test_sample_in_limit_uniform_inside(ranges):
    n = 60000
    for x in (sample_in_limit(n, ranges, "ellipse_xy", rng=np.random.default_rng(1)),
              sample_in_limit(n, ranges, "ellipse_xy", device="cpu").numpy()):
        assert np.allclose(limit_command(x, ranges, "ellipse_xy"), x, atol=1e-6)       # all inside
        lo_x, hi_x = ranges["lin_vel_x"]
        lo_y, hi_y = ranges["lin_vel_y"]
        bx = np.where(x[:, 0] >= 0, hi_x, -lo_x)
        by = np.where(x[:, 1] >= 0, hi_y, -lo_y)
        s2 = (x[:, 0] / bx) ** 2 + (x[:, 1] / by) ** 2
        assert s2.max() <= 1 + 1e-6
        # uniform on the unit disc: P(s > 0.9) = 0.19, E[s^2] = 0.5; wz uniform and independent
        assert 0.17 < (s2 > 0.81).mean() < 0.21 and abs(s2.mean() - 0.5) < 0.01
        wlo, whi = ranges["ang_vel_z"]
        assert x[:, 2].min() >= wlo and x[:, 2].max() <= whi and abs(x[:, 2].mean() - 0.5 * (wlo + whi)) < 0.02
        assert abs(np.corrcoef(np.abs(x[:, 0]), np.abs(x[:, 2]))[0, 1]) < 0.03
    box = sample_in_limit(n, ranges, "box", rng=np.random.default_rng(1))
    assert (limit_command(box, ranges, "ellipse_xy") != box).any()                     # the box does have corners


def test_sample_in_limit_torch_honours_generator():
    a = sample_in_limit(100, R_REF, "ellipse_xy", rng=torch.Generator().manual_seed(5))
    b = sample_in_limit(100, R_REF, "ellipse_xy", rng=torch.Generator().manual_seed(5))
    assert isinstance(a, torch.Tensor) and torch.equal(a, b)


# --------------------------------------------------------------------------- configs
def _config(name):
    pytest.importorskip("yaml")
    from utils import Config
    return Config(os.path.join(ROOT, "config", name)).train.env_config


@pytest.mark.parametrize("name", ["config.yaml", "config_gait_reference.yaml"])
def test_configs_select_the_ellipse_and_plumb_it(name):
    ec = _config(name)
    assert ec.commands["base_velocity"]["limit_shape"] == "ellipse_xy"
    from src.isaaclab import env_config
    assert "limit_shape" in env_config._COMMAND_KEYS


def test_env_config_rejects_a_typo_and_old_configs_default_to_box():
    from src.isaaclab import env_config

    class Cfg:  # the bits of BipedEnvCfg that _apply_commands touches
        def __init__(self):
            self.commands = {"base_velocity": {"ranges": {}, "limit_shape": "box"}}

    for bad in ("elipse_xy", "ellipse"):
        with pytest.raises(ValueError):
            env_config._apply_commands(Cfg(), types.SimpleNamespace(commands={"base_velocity": {"limit_shape": bad}}))
    c = Cfg()
    env_config._apply_commands(c, types.SimpleNamespace(commands={"base_velocity": {"limit_shape": "ellipse_xy"}}))
    assert c.commands["base_velocity"]["limit_shape"] == "ellipse_xy"
    c = Cfg()
    env_config._apply_commands(c, types.SimpleNamespace(commands={"base_velocity": {"rel_heading_envs": 0.5}}))
    assert c.commands["base_velocity"]["limit_shape"] == "box"       # key absent: the class default stays


# --------------------------------------------------------------------------- Isaac sampler
def _load_sampler():
    """envs/isaaclab/mdp/commands.py with the Isaac-only imports stubbed while it loads."""
    imath = types.ModuleType("isaaclab.utils.math")
    imath.wrap_to_pi = lambda x: torch.atan2(torch.sin(x), torch.cos(x))
    imath.quat_from_euler_xyz = imath.quat_apply = None
    imark = types.ModuleType("isaaclab.markers")
    imark.VisualizationMarkers = object
    mp = pytest.MonkeyPatch()
    for name, mod in (("isaaclab", types.ModuleType("isaaclab")), ("isaaclab.utils", types.ModuleType("isaaclab.utils")),
                      ("isaaclab.utils.math", imath), ("isaaclab.markers", imark)):
        mp.setitem(sys.modules, name, mod)
    try:
        spec = importlib.util.spec_from_file_location(
            "_mdp_commands_under_test", os.path.join(ROOT, "envs", "isaaclab", "mdp", "commands.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        mp.undo()
    return mod


@pytest.fixture(scope="module")
def sampler_mod():
    return _load_sampler()


def _gen(mod, name, n, shape=None, **bv_over):
    bv = dict(_config(name).commands["base_velocity"])
    bv["ranges"] = dict(bv["ranges"])
    if shape is not None:
        bv["limit_shape"] = shape
    bv.update(bv_over)
    cfg = types.SimpleNamespace(commands={"base_velocity": bv})
    return mod.UniformVelocityCommand(cfg, n, torch.device("cpu"), 0.02)


def _s(cmd, ranges):
    lo_x, hi_x = ranges["lin_vel_x"]
    lo_y, hi_y = ranges["lin_vel_y"]
    vx, vy = cmd[:, 0], cmd[:, 1]
    bx = torch.where(vx >= 0, torch.tensor(float(hi_x)), torch.tensor(-float(lo_x)))
    by = torch.where(vy >= 0, torch.tensor(float(hi_y)), torch.tensor(-float(lo_y)))
    return torch.sqrt((vx / bx) ** 2 + (vy / by) ** 2)


@pytest.mark.parametrize("name", ["config.yaml", "config_gait_reference.yaml"])
def test_sampler_all_inside_and_covers_the_boundary(sampler_mod, name):
    n = 20000
    gen = _gen(sampler_mod, name, n)
    assert gen.limit_shape == "ellipse_xy"
    ids = torch.arange(n)
    torch.manual_seed(0)
    out = []
    for _ in range(4):
        gen.resample(ids)
        gen.apply_heading(torch.tensor([1.0, 0, 0, 0]).repeat(n, 1))
        out.append(gen.commands.clone())
    cmd = torch.cat(out)
    r = gen.ranges
    s = _s(cmd, r)
    assert s.max() <= 1 + 1e-5
    assert cmd[:, 0].min() >= r["lin_vel_x"][0] and cmd[:, 1].abs().max() <= r["lin_vel_y"][1] + 1e-6
    moving = cmd[:, :2].abs().sum(-1) > 0               # not the standing / turn-in-place buckets
    assert 0.8 < moving.float().mean() < 0.9            # 2% standing + 15% turn-in-place
    frac = (s[moving] > 0.9).float().mean().item()
    assert frac > 0 and 0.15 < frac < 0.23              # uniform on the disc: 1 - 0.9^2 = 0.19
    assert abs((s[moving] ** 2).mean().item() - 0.5) < 0.02
    # the same ranges as a box would leave the ellipse in ~21% of draws
    box = _gen(sampler_mod, name, n, shape="box")
    torch.manual_seed(0)
    box.resample(ids)
    assert (_s(box.commands, r) > 1.0).float().mean() > 0.15


def test_box_sampler_is_bit_identical_to_the_old_one(sampler_mod):
    """limit_shape box: same RNG calls in the same order as before the limit existed."""
    n = 512

    def old_resample(g, env_ids):    # the pre-limit UniformVelocityCommand.resample, verbatim
        t_min, t_max = g.resampling_time_range
        g.time_left[env_ids] = torch.empty(len(env_ids), device=g.device).uniform_(t_min, t_max)
        r_x = g.ranges.get("lin_vel_x", (-0.5, 0.5))
        r_y = g.ranges.get("lin_vel_y", (-0.25, 0.25))
        r_z = g.ranges.get("ang_vel_z", (-1.0, 1.0))
        r_head = g.ranges.get("heading", (-3.14159, 3.14159))
        g.commands[env_ids, 0] = torch.empty(len(env_ids), device=g.device).uniform_(*r_x)
        g.commands[env_ids, 1] = torch.empty(len(env_ids), device=g.device).uniform_(*r_y)
        g.commands[env_ids, 2] = torch.empty(len(env_ids), device=g.device).uniform_(*r_z)
        is_standing = torch.rand(len(env_ids), device=g.device) < g.rel_standing_envs
        is_heading = torch.rand(len(env_ids), device=g.device) < g.rel_heading_envs
        g.commands[env_ids] = torch.where(is_standing.unsqueeze(-1), 0.0, g.commands[env_ids])
        is_turn = (torch.rand(len(env_ids), device=g.device) < g.rel_turn_in_place_envs) & (~is_standing)
        if is_turn.any():
            turn_ids = env_ids[is_turn]
            wz_max = max(abs(float(r_z[0])), abs(float(r_z[1])))
            mag = torch.empty(len(turn_ids), device=g.device).uniform_(g.turn_in_place_min_frac * wz_max, wz_max)
            sign = torch.where(torch.rand(len(turn_ids), device=g.device) < 0.5, -1.0, 1.0)
            g.commands[turn_ids, 0] = 0.0
            g.commands[turn_ids, 1] = 0.0
            g.commands[turn_ids, 2] = mag * sign
        g.in_heading_mode[env_ids] = is_heading & (~is_standing) & (~is_turn)
        g.heading_target[env_ids] = torch.empty(len(env_ids), device=g.device).uniform_(*r_head)

    for name in ("config.yaml", "config_gait_reference.yaml"):
        new, old = _gen(sampler_mod, name, n, shape="box"), _gen(sampler_mod, name, n, shape="box")
        ids = torch.arange(n)
        quat = torch.tensor([0.9, 0.1, 0.2, 0.3]).repeat(n, 1)
        torch.manual_seed(123)
        new.resample(ids)
        new.apply_heading(quat)
        torch.manual_seed(123)
        old_resample(old, ids)
        old.apply_heading(quat)
        for attr in ("commands", "time_left", "heading_target", "in_heading_mode"):
            assert torch.equal(getattr(new, attr), getattr(old, attr)), (name, attr)
    # an old config without the key is a box
    cfg = types.SimpleNamespace(commands={"base_velocity": {"ranges": dict(R_REF)}})
    assert sampler_mod.UniformVelocityCommand(cfg, 4, torch.device("cpu"), 0.02).limit_shape == "box"
    with pytest.raises(ValueError):
        _gen(sampler_mod, "config.yaml", 4, shape="ellipse")


def test_sampler_curriculum_uses_current_ranges(sampler_mod):
    n = 8000
    gen = _gen(sampler_mod, "config_gait_reference.yaml", n, rel_standing_envs=0.0, rel_turn_in_place_envs=0.0)
    gen.update_ranges({"lin_vel_x": (-0.05, 0.1), "lin_vel_y": (-0.05, 0.05)})
    torch.manual_seed(1)
    gen.resample(torch.arange(n))
    assert _s(gen.commands, gen.ranges).max() <= 1 + 1e-5
    assert gen.commands[:, 0].max() <= 0.1 and gen.commands[:, 1].abs().max() <= 0.05 + 1e-7


# --------------------------------------------------------------------------- gait table
def _table_helpers():
    from tests import test_gait_table_reference as t
    return t


def test_gait_table_applies_the_config_limit_before_locating(tmp_path):
    from utils.motion_reference import GaitTableReference
    t = _table_helpers()
    path = t._synthetic(tmp_path)
    ranges = {"lin_vel_x": (-0.2, 0.3), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-1.0, 1.0)}
    plain, limited = GaitTableReference(path, t.DEV), GaitTableReference(path, t.DEV)
    limited.set_command_limit(ranges, "ellipse_xy")
    raw = torch.tensor([[0.3, 0.1, 0.0], [0.1, -0.1, 0.5], [0.05, 0.02, 0.0]])
    proj = torch.as_tensor(limit_command(raw.numpy().astype(np.float64), ranges, "ellipse_xy"), dtype=torch.float32)
    assert not torch.allclose(proj[0], raw[0])
    phase, stand = torch.tensor([0.3, 0.3, 0.3]), torch.tensor([False] * 3)
    got, want, unlimited = limited.sample(raw, phase, stand), plain.sample(proj, phase, stand), plain.sample(raw, phase, stand)
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got[:2], unlimited[:2], atol=1e-4)
    assert torch.allclose(got[2], unlimited[2])                        # inside the ellipse: untouched
    # frequency and precomputed corners see the same command
    assert torch.allclose(limited.frequency(raw), plain.frequency(proj))
    c = limited.corners(raw)
    assert torch.allclose(limited.sample(raw, phase, stand, corners=c), got)
    # box = no change
    boxed = GaitTableReference(path, t.DEV)
    boxed.set_command_limit(ranges, "box")
    assert torch.equal(boxed.sample(raw, phase, stand), unlimited)
    with pytest.raises(ValueError):
        boxed.set_command_limit(ranges, "oval")


def test_gait_table_projects_the_documented_corner(tmp_path):
    """(0.2, 0.15, 0) on the Gait Studio box reads the frame of (0.1414, 0.1061, 0)."""
    from utils.motion_reference import GaitTableReference
    t = _table_helpers()
    path = t._synthetic(tmp_path)
    ranges = {k: tuple(v) for k, v in R_REF.items()}
    plain, limited = GaitTableReference(path, t.DEV), GaitTableReference(path, t.DEV)
    limited.set_command_limit(ranges, "ellipse_xy")
    phase, stand = torch.tensor([0.4]), torch.tensor([False])
    corner = torch.tensor([[0.2, 0.15, 0.0]])
    proj = torch.tensor([[0.2 / math.sqrt(2), 0.15 / math.sqrt(2), 0.0]])
    assert torch.allclose(limited.sample(corner, phase, stand), plain.sample(proj, phase, stand), atol=1e-6)


def _v3_table(tmp_path, shape="ellipse_xy", rx=(-0.2, 0.2), ry=(-0.15, 0.15), rz=(-1.2, 1.2)):
    t = _table_helpers()
    z = dict(np.load(t._synthetic(tmp_path)))
    z.update(version=np.int32(3), command_limit_shape=np.array(shape), command_ranges_x=np.array(rx),
             command_ranges_y=np.array(ry), command_ranges_z=np.array(rz))
    p = tmp_path / f"v3_{shape}.npz"
    np.savez(p, **z)
    return str(p)


def test_table_metadata_mismatch_warns_and_never_defines_the_limit(tmp_path):
    from utils.motion_reference import GaitTableReference
    t = _table_helpers()
    ranges = {k: tuple(v) for k, v in R_REF.items()}
    ok = GaitTableReference(_v3_table(tmp_path), t.DEV)
    assert ok.version == 3 and ok.baked_limit["shape"] == "ellipse_xy"
    assert ok.check_command_limit(ranges, "ellipse_xy") == []
    msgs = ok.check_command_limit(ranges, "box")                                 # shape differs
    assert len(msgs) == 1 and msgs[0].startswith("WARNING") and "limit_shape" in msgs[0]
    msgs = ok.check_command_limit({**ranges, "lin_vel_y": (-0.2, 0.2)}, "ellipse_xy")   # range differs
    assert len(msgs) == 1 and msgs[0].startswith("WARNING") and "lin_vel_y" in msgs[0]
    # a table baked as a box, config ellipse: warning
    box_tab = GaitTableReference(_v3_table(tmp_path, shape="box"), t.DEV)
    assert any(m.startswith("WARNING") for m in box_tab.check_command_limit(ranges, "ellipse_xy"))
    # the table's metadata never reaches the limit the reference applies
    assert box_tab._limit is None and ok._limit is None
    # v1/v2 table (no metadata): silent for a box config, one-line note otherwise
    old = GaitTableReference(t._synthetic(tmp_path), t.DEV)
    assert old.baked_limit is None and old.check_command_limit(ranges, "box") == []
    note = old.check_command_limit(ranges, "ellipse_xy")
    assert len(note) == 1 and note[0].startswith("note:") and "WARNING" not in note[0]


def test_env_wires_the_limit_into_the_table_and_checks_it():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    assert "g.set_command_limit(self._cmd_full_ranges, self.command_generator.limit_shape)" in src
    assert "g.check_command_limit(self._cmd_full_ranges, self.command_generator.limit_shape)" in src


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")),
                    reason="baked table not present")
def test_real_table_loads_whatever_its_version():
    from utils.motion_reference import GaitTableReference
    ref = GaitTableReference(os.path.join("robot_motion_reference", "processed", "gait_table.npz"),
                             torch.device("cpu"), root_dir=ROOT)
    assert ref.version in (2, 3)


# --------------------------------------------------------------------------- MuJoCo
def _mj_env(name):
    pytest.importorskip("mujoco")
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    ec = _config(name)
    ec.backlash = {**(ec.backlash or {}), "enabled": False}
    env = BipedEnvV2(history_size=1, use_history=False, env_config=ec)
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("name", ["config.yaml", "config_gait_reference.yaml"])
def test_mujoco_set_command_and_sampling_use_the_limit(name):
    env = _mj_env(name)
    ranges, shape = env.command_limit()
    assert shape == "ellipse_xy"
    eff = env.set_command(0.2, 0.15, 0.0)
    lim = limit_command(np.array([0.2, 0.15, 0.0]), ranges, "ellipse_xy")
    assert np.allclose(eff, lim, atol=1e-6) and np.allclose(env.commands, lim, atol=1e-6)
    if name == "config_gait_reference.yaml":     # vx 0.2, vy 0.15 (the Gait Studio box)
        assert abs(env.commands[0] - 0.14142) < 1e-4 and abs(env.commands[1] - 0.10607) < 1e-4
        assert abs(env.commands[2]) < 1e-9
    assert np.allclose(env.set_command(0.1, 0.05, 0.3), (0.1, 0.05, 0.3), atol=1e-6)     # inside: as given
    # sampling is reproducible from reset(seed) and stays inside
    seen = []
    for seed in (0, 0, 1):
        env.reset(seed=seed)
        seen.append(env.commands.copy())
    assert np.allclose(seen[0], seen[1]) and not np.allclose(seen[0], seen[2])
    pts = []
    for _ in range(300):
        env._sample_commands()
        pts.append(env.commands.copy())
    pts = np.array(pts, dtype=np.float64)
    assert np.abs(limit_command(pts, ranges, "ellipse_xy") - pts).max() < 1e-5
    # the gait clock reads env.commands, i.e. the limited command
    if env.gait_clock is not None:
        cmd = env.set_command(0.2, 0.15, 0.0)
        assert env.gait_clock.frequency(env.commands[:3]) == pytest.approx(env.gait_clock.frequency(cmd))


def test_mujoco_box_config_keeps_set_command_as_given():
    env = _mj_env("config.yaml")
    env.config.commands["base_velocity"]["limit_shape"] = "box"
    assert np.allclose(env.set_command(0.2, 0.2, 5.0), (0.2, 0.2, 5.0), atol=1e-6)   # no clamp, as before


# --------------------------------------------------------------------------- export
_RANGES_BLOCK = ("        ranges:\n          lin_vel_x: [-0.2, 0.2]\n          lin_vel_y: [-0.15, 0.15]\n"
                 "          ang_vel_z: [-1.2, 1.2]\n          heading: [-3.14, 3.14]\n")


def _run_dir(tmp_path, name, extra=""):
    d = tmp_path / name
    d.mkdir()
    (d / "config.yaml").write_text(
        "train:\n  env_config:\n    commands:\n      base_velocity:\n" + extra + _RANGES_BLOCK)
    return str(d)


def test_export_writes_command_limit_and_absent_means_box(tmp_path):
    from src.isaaclab import export_onnx as ex

    def limit(run):
        return ex._load_command_limit(run, None, ex._load_command_ranges(run, None))

    run = _run_dir(tmp_path, "new", "        limit_shape: ellipse_xy\n")
    assert limit(run) == {"shape": "ellipse_xy", "lin_vel_x": [-0.2, 0.2], "lin_vel_y": [-0.15, 0.15],
                          "ang_vel_z": [-1.2, 1.2]}
    old = limit(_run_dir(tmp_path, "old", "        rel_heading_envs: 1.0\n"))
    assert old["shape"] == "box" and old["lin_vel_x"] == [-0.2, 0.2]
    with pytest.raises(ValueError):
        limit(_run_dir(tmp_path, "bad", "        limit_shape: oval\n"))
    none = tmp_path / "none"
    none.mkdir()
    assert limit(str(none))["shape"] == "box"                                    # no run config at all
    # the key lands in deploy_config.json next to command_ranges, without a format bump
    src = open(os.path.join(ROOT, "src", "isaaclab", "export_onnx.py"), encoding="utf-8").read()
    assert '"command_limit": command_limit,' in src
    assert ex._format_version({"action_map": "centered"}, {"mode": "speed_proportional"}) == 2
