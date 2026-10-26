"""`auto` residual map (utils/residual_home.py): home = the table's stand frame, scales = narrowest 0.05 step with headroom,
spawn height = forward kinematics of the home. CPU only (mujoco for FK / the MuJoCo env), no Isaac.

Run from BipedRobot/:  python -m pytest tests/test_residual_home_resolver.py -q
"""
from __future__ import annotations

import copy
import math
import os
import sys
import textwrap
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
mujoco = pytest.importorskip("mujoco")
from utils import Config  # noqa: E402
from utils import residual_home as rh  # noqa: E402
from utils.residual_action import JOINT_TYPES, RESIDUAL_SPAWN_HEIGHT_M, expand_by_type, home_rad  # noqa: E402

TABLE = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
REF_YAML = os.path.join(ROOT, "config", "config_gait_reference.yaml")
BASE_YAML = os.path.join(ROOT, "config", "config.yaml")
needs_table = pytest.mark.skipif(not os.path.exists(TABLE), reason="baked table not present")

CROUCH = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -10.0, "knee": -26.0, "ankle_roll": 0.0, "ankle_pitch": -16.0}
SCALE = {"hip_yaw": 0.6, "hip_roll": 0.45, "hip_pitch": 0.8, "knee": 0.6, "ankle_roll": 0.45, "ankle_pitch": 0.6}
CONFIG_YAML_SCALE = {"hip_yaw": 0.45, "hip_roll": 0.45, "hip_pitch": {"pos": 0.6, "neg": 0.8}, "knee": {"pos": 0.55, "neg": 0.8}, "ankle_roll": 0.35, "ankle_pitch": {"pos": 0.6, "neg": 0.35}}   # config/config.yaml since 2026-10-25
# homes with a documented spawn clearance (Isaac frame): the hand home of the 2026-10-23 17:36 bake, the home before the v2
# table (CHANGES.md: "reads 15.43 mm" in MuJoCo = 5.43 mm here), and the 2026-10-21 v2 home at 0.28 (5.4 mm)
HAND_HOME = {"hip_yaw": 1.5, "hip_roll": -6.8, "hip_pitch": -6.5, "knee": -19.2, "ankle_roll": 6.9, "ankle_pitch": -12.6}
OLD_HOME = {"hip_yaw": 0.0, "hip_roll": -2.5, "hip_pitch": -8.0, "knee": -22.0, "ankle_roll": 2.5, "ankle_pitch": -14.0}
V2_HOME = {"hip_yaw": 2.1, "hip_roll": -6.2, "hip_pitch": -17.7, "knee": -36.6, "ankle_roll": 6.5, "ankle_pitch": -18.7}


def _ec(**kw):
    base = dict(action_map="residual", action_home_deg=None, action_scale_rad=None, spawn_height_m=None,
                reference={"npz_path": TABLE, "rsi_joint_noise": 0.05},
                commands={"base_velocity": {"ranges": {"lin_vel_x": [-0.2, 0.2], "lin_vel_y": [-0.15, 0.15],
                                                        "ang_vel_z": [-1.2, 1.2]}}})
    return SimpleNamespace(**{**base, **kw})


def _auto_ec(**kw):
    return _ec(action_home_deg="auto", action_scale_rad="auto", spawn_height_m="auto", **kw)


def _write_table(tmp_path, name="table.npz", stand_delta_deg=None, scale_joint_pos=None):
    """Copy of the real table with an edited stand frame (list of 12 deg deltas, policy order) / joint_pos."""
    z = dict(np.load(TABLE, allow_pickle=False))
    if stand_delta_deg is not None:
        order = [list(z["joint_names"]).index(n) for n in rh.POLICY_JOINT_NAMES]
        inv = np.argsort(order)                              # policy index -> table index
        st = z["stand"].astype(np.float64)
        st[:12] += np.radians(np.asarray(stand_delta_deg, dtype=np.float64))[inv]
        z["stand"] = st.astype(np.float32)
    if scale_joint_pos is not None:
        z["data"] = z["data"].copy()
        z["data"][..., :12] *= scale_joint_pos
    path = os.path.join(str(tmp_path), name)
    np.savez(path, **z)
    return path


# ------------------------------------------------------------------------------------------------ pass-through
def test_explicit_values_pass_through_untouched():
    ec = _ec(action_home_deg=dict(CROUCH), action_scale_rad=dict(SCALE), spawn_height_m=0.286)
    home, scale, spawn, rep = rh.resolve_residual_map(ec, ROOT)
    assert home == CROUCH and scale == SCALE and spawn == 0.286
    assert rep["sources"] == {"action_home_deg": "explicit", "action_scale_rad": "explicit", "spawn_height_m": "explicit"}
    assert rh.resolve_if_auto(ec, "test") is None
    # unset spawn height keeps the old residual default; other maps return nothing for home / scale
    assert rh.resolve_residual_map(_ec(action_home_deg=dict(CROUCH), action_scale_rad=dict(SCALE)), ROOT)[2] \
        == RESIDUAL_SPAWN_HEIGHT_M
    assert rh.resolve_residual_map(_ec(action_map="centered"), ROOT)[:3] == (None, None, None)
    assert rh.resolve_residual_map(_ec(action_map="centered", spawn_height_m=0.3), ROOT)[:3] == (None, None, 0.3)


def test_config_yaml_explicit_map_is_not_touched_by_freeze():
    cfg = Config(BASE_YAML)
    before = copy.deepcopy(cfg.config_obj.model_dump(mode="json"))
    assert rh.freeze_residual_map(cfg, ROOT) is None
    assert cfg.config_obj.model_dump(mode="json") == before
    ec = cfg.train.env_config
    assert dict(ec.action_home_deg) == CROUCH and dict(ec.action_scale_rad) == CONFIG_YAML_SCALE and ec.spawn_height_m == 0.286


def test_partial_auto_leaves_explicit_keys_alone():
    ec = _ec(action_home_deg=dict(CROUCH), action_scale_rad=dict(SCALE), spawn_height_m="auto")
    home, scale, spawn, rep = rh.resolve_residual_map(ec, ROOT)
    assert home == CROUCH and scale == SCALE and 0.28 < spawn < 0.29        # spawn is derived from the (explicit) home
    assert "table" not in rep                                               # no table needed for an auto spawn alone


# ------------------------------------------------------------------------------------------------ home = stand frame
@needs_table
def test_auto_home_is_the_table_stand_frame():
    home, _, _, rep = rh.resolve_residual_map(_auto_ec(), ROOT)
    with np.load(TABLE, allow_pickle=False) as z:
        names = [str(s) for s in z["joint_names"]]
        stand = z["stand"][:12].astype(np.float64)
    for t in JOINT_TYPES:
        l, r = (np.degrees(stand[names.index(f"{s}_{t}" + ("" if t == "hip_yaw" else "_joint"))]) for s in ("l", "r"))
        assert home[t] == pytest.approx(round(0.5 * (l + r), 1), abs=1e-9), t
        assert abs(home[t] - 0.5 * (l + r)) <= 0.05 + 1e-9
    assert rep["stand_lr_asym_deg"] < rh.LR_WARN_DEG                         # this table is L/R symmetric
    assert all(math.copysign(1.0, v) > 0 or v != 0.0 for v in home.values())    # no -0.0 in the yaml


@needs_table
def test_auto_home_follows_a_rebake_and_warns_on_asymmetry(tmp_path):
    p = _write_table(tmp_path, stand_delta_deg=[0, 0, 0, 0, 2.0, 2.0, 0, 0, 0, 0, 0, 0])
    h0 = rh.resolve_residual_map(_auto_ec(), ROOT)[0]
    h1 = rh.resolve_residual_map(_auto_ec(reference={"npz_path": p, "rsi_joint_noise": 0.05}), ROOT)[0]
    assert h1["hip_pitch"] == pytest.approx(h0["hip_pitch"] + 2.0, abs=0.101)
    assert all(h1[t] == h0[t] for t in JOINT_TYPES if t != "hip_pitch")
    p2 = _write_table(tmp_path, "asym.npz", stand_delta_deg=[0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])   # right hip pitch +1 deg
    with pytest.warns(UserWarning, match="not L/R symmetric"):
        h2, _, _, rep = rh.resolve_residual_map(_auto_ec(reference={"npz_path": p2, "rsi_joint_noise": 0.05}), ROOT)
    assert rep["stand_lr_asym_deg"] == pytest.approx(1.0, abs=1e-3)
    assert h2["hip_pitch"] == pytest.approx(h0["hip_pitch"] + 0.5, abs=0.101)        # mean of L and R


def test_policy_joint_layout_matches_the_envs():
    from envs.mujoco.biped_env_v2 import JLIM_MAX, JLIM_MIN, JOINT_NAMES
    assert tuple(JOINT_NAMES) == rh.POLICY_JOINT_NAMES
    lim = rh.joint_limits_deg(ROOT)
    assert lim.shape == (12, 2)
    from tests.test_residual_action_map import LIM_DEG
    assert np.array_equal(lim, np.asarray(LIM_DEG, dtype=np.float64))               # same source as the residual-map test
    assert np.allclose(np.radians(lim[:, 0]), JLIM_MIN, atol=1e-3) and np.allclose(np.radians(lim[:, 1]), JLIM_MAX, atol=1e-3)


# ------------------------------------------------------------------------------------------------ scales
def test_select_grid_cells_rule():
    g = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    assert rh.select_grid_cells(g, 0.2).tolist() == [False, True, True, True, True, True, False]      # bound on the grid
    assert rh.select_grid_cells(g, 0.15).tolist() == [False, True, True, True, True, True, False]     # + first point beyond
    assert rh.select_grid_cells(g, 0.5).all()


@needs_table
def test_auto_scale_satisfies_headroom_and_is_minimal():
    ec = _auto_ec()
    home, scale, _, rep = rh.resolve_residual_map(ec, ROOT)
    tab = rh.load_table(TABLE, ROOT)
    lim = np.radians(rh.joint_limits_deg(ROOT))
    noise, head = 0.05, rep["headroom"]
    for bounds in ([0.2, 0.15, 1.2], [0.1, 0.07, 0.5], [0.0, 0.0, 0.0]):
        ec2 = _auto_ec(commands={"base_velocity": {"ranges": {"lin_vel_x": [-bounds[0], bounds[0]],
                                                              "lin_vel_y": [-bounds[1], bounds[1]],
                                                              "ang_vel_z": [-bounds[2], bounds[2]]}}})
        h2, s2, _, r2 = rh.resolve_residual_map(ec2, ROOT)
        assert h2 == home                                                         # home never depends on the command range
        q = tab["q"]
        sel = [rh.select_grid_cells(gr, b) for gr, b in zip(tab["grids"], bounds)]
        q = q[sel[0]][:, sel[1]][:, :, sel[2]].reshape(-1, 12)
        dev = np.abs(np.clip(q, lim[:, 0], lim[:, 1]) - home_rad(h2)).max(axis=0)
        for i, t in enumerate(JOINT_TYPES):
            s = s2[t]
            d = max(dev[2 * i], dev[2 * i + 1])
            assert abs(s / 0.05 - round(s / 0.05)) < 1e-6 and s >= 0.05 - 1e-9     # on the 0.05 grid
            assert (d + noise) / s <= head + 1e-9, (bounds, t)                      # max|a| + noise/scale <= headroom
            if s > 0.05 + 1e-9:
                assert (d + noise) / (s - 0.05) > head, (bounds, t)                 # one step narrower would violate it
        if bounds == [0.2, 0.15, 1.2]:
            assert s2 == scale
        else:
            assert all(s2[t] <= scale[t] + 1e-9 for t in JOINT_TYPES)               # narrower commands never need more


@needs_table
def test_auto_scale_follows_the_table_and_the_noise(tmp_path):
    base = rh.resolve_residual_map(_auto_ec(), ROOT)[1]
    wide = _write_table(tmp_path, "wide.npz", scale_joint_pos=1.5)                  # swings 50% larger, stand frame untouched
    s_wide = rh.resolve_residual_map(_auto_ec(reference={"npz_path": wide, "rsi_joint_noise": 0.05}), ROOT)[1]
    assert all(s_wide[t] >= base[t] for t in JOINT_TYPES) and any(s_wide[t] > base[t] for t in JOINT_TYPES)
    noisy = rh.resolve_residual_map(_auto_ec(reference={"npz_path": TABLE, "rsi_joint_noise": 0.2}), ROOT)[1]
    assert all(noisy[t] >= base[t] for t in JOINT_TYPES) and any(noisy[t] > base[t] for t in JOINT_TYPES)
    tight = rh.resolve_residual_map(_auto_ec(scale_headroom=0.8), ROOT)[1]
    assert all(tight[t] >= base[t] for t in JOINT_TYPES) and any(tight[t] > base[t] for t in JOINT_TYPES)


# ------------------------------------------------------------------------------------------------ spawn height (FK)
def _clearance_mm(home, base_z):
    return (base_z + rh.lowest_sole_z(home, ROOT)["mesh_z"]) * 1e3


def test_fk_reproduces_the_historical_clearances():
    assert _clearance_mm(HAND_HOME, 0.286) == pytest.approx(5.5, abs=0.3)     # hand home of the 17:36 bake at 0.286
    assert _clearance_mm(OLD_HOME, 0.286) == pytest.approx(5.43, abs=0.3)     # "the old home at 0.286 reads 15.43 mm" (MuJoCo frame)
    assert _clearance_mm(V2_HOME, 0.28) == pytest.approx(5.4, abs=0.3)        # the 2026-10-21 v2 home at 0.28
    assert _clearance_mm(None, 0.29) == pytest.approx(5.4, abs=0.3)           # the asset's straight-leg spawn at 0.29


def test_fk_mesh_vertices_agree_with_the_sole_rectangle():
    for home in (HAND_HOME, OLD_HOME, V2_HOME, CROUCH, None):
        fk = rh.lowest_sole_z(home, ROOT)
        assert abs(fk["mesh_z"] - fk["rect_z"]) < 1e-4, (home, fk)           # < 0.1 mm


def test_spawn_for_the_hand_set_crouch_of_config_yaml():
    """config.yaml keeps its explicit 0.286. NOTE its comment ("0.29 minus the 3.9 mm the crouch lowers the base", the same ~5.4 mm
    clearance) does not hold for the -10 / -26 / -16 crouch: FK gives 6.67 mm at 0.286 (that 5.4 mm belonged to the older home
    -8 / -22 / -14). `auto` puts it at the requested clearance."""
    assert _clearance_mm(CROUCH, 0.286) == pytest.approx(6.67, abs=0.1)
    z, fk = rh.spawn_height_for(CROUCH, 0.0054, ROOT)
    assert z == pytest.approx(0.2847, abs=0.0002)
    assert fk["clearance_m"] == pytest.approx(0.0054, abs=6e-5) and fk["clearance_rect_m"] == pytest.approx(0.0054, abs=2e-4)
    z2 = rh.resolve_residual_map(_ec(action_home_deg=dict(CROUCH), action_scale_rad=dict(SCALE), spawn_height_m="auto"), ROOT)[2]
    assert z2 == z


def test_spawn_clearance_key_and_independence_from_the_table(tmp_path):
    ec = _ec(action_home_deg=dict(HAND_HOME), action_scale_rad=dict(SCALE), spawn_height_m="auto")
    z0 = rh.resolve_residual_map(ec, ROOT)[2]
    ec.spawn_clearance_m = 0.0104                                              # +5 mm clearance -> +5 mm root height
    assert rh.resolve_residual_map(ec, ROOT)[2] == pytest.approx(z0 + 0.005, abs=1.1e-4)
    ec.spawn_clearance_m = 0.0054
    ec.reference = None                                                        # no table at all: spawn needs only the home
    assert rh.resolve_residual_map(ec, ROOT)[2] == z0
    assert rh.resolve_residual_map(_ec(action_map="absolute", spawn_height_m="auto"), ROOT)[2] == pytest.approx(
        rh.spawn_height_for(None, 0.0054, ROOT)[0])


@needs_table
def test_auto_spawn_of_the_current_table_gives_the_requested_clearance():
    home, _, spawn, rep = rh.resolve_residual_map(_auto_ec(), ROOT)
    fk = rep["spawn_fk"]
    assert fk["clearance_m"] == pytest.approx(0.0054, abs=6e-5)               # rounding the height to 0.1 mm
    assert spawn + fk["mesh_z"] == pytest.approx(fk["clearance_m"])
    with np.load(TABLE, allow_pickle=False) as z:
        names = [str(s) for s in z["field_names"]]
        w = [int(x) for x in z["field_widths"]]
        base_h = float(z["stand"][sum(w[:names.index("base_height")]):][0])
    assert spawn == pytest.approx(base_h + 0.005, abs=0.001)                  # the table's own RSI rule: stand height + 5 mm


# ------------------------------------------------------------------------------------------------ freezing
@needs_table
def test_freeze_replaces_auto_by_numbers_and_the_saved_config_has_no_auto(tmp_path, capsys):
    cfg = Config(REF_YAML)
    ec = cfg.train.env_config
    assert (ec.action_home_deg, ec.action_scale_rad, ec.spawn_height_m) == ("auto", "auto", "auto")
    rep = rh.freeze_residual_map(cfg, ROOT)
    assert rep is not None and "frozen into the run config" in capsys.readouterr().out
    home, scale, spawn, _ = rh.resolve_residual_map(_auto_ec(), ROOT)
    assert dict(ec.action_home_deg) == home and dict(ec.action_scale_rad) == scale and ec.spawn_height_m == spawn
    assert all(isinstance(v, float) for v in ec.action_home_deg.values())
    assert isinstance(ec.spawn_height_m, float)
    # what the trainer writes next to the checkpoint (copy_in_file) holds numbers only and reloads to the same numbers
    out = os.path.join(str(tmp_path), "config.yaml")
    cfg.copy_in_file(out)
    saved = yaml.safe_load(open(out, encoding="utf-8"))["train"]["env_config"]
    assert saved["action_home_deg"] == home and saved["action_scale_rad"] == scale and saved["spawn_height_m"] == spawn
    assert not any(isinstance(saved[k], str) for k in ("action_home_deg", "action_scale_rad", "spawn_height_m"))
    again = Config(out).train.env_config
    assert dict(again.action_home_deg) == home and again.spawn_height_m == spawn
    assert rh.freeze_residual_map(Config(out), ROOT) is None                                      # idempotent: nothing left to do
    assert rh.freeze_residual_map(cfg, ROOT) is None


@needs_table
def test_a_rebake_cannot_change_a_frozen_run(tmp_path):
    cfg = Config(REF_YAML)
    rh.freeze_residual_map(cfg, ROOT)
    frozen = (dict(cfg.train.env_config.action_home_deg), dict(cfg.train.env_config.action_scale_rad),
              cfg.train.env_config.spawn_height_m)
    p = _write_table(tmp_path, stand_delta_deg=[0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], scale_joint_pos=1.4)
    cfg.train.env_config.reference = {**cfg.train.env_config.reference, "npz_path": p}           # "re-baked" table
    assert rh.freeze_residual_map(cfg, ROOT) is None
    ec = cfg.train.env_config
    assert (dict(ec.action_home_deg), dict(ec.action_scale_rad), ec.spawn_height_m) == frozen
    # while a fresh `auto` against the new table does move
    assert rh.resolve_residual_map(_auto_ec(reference={"npz_path": p, "rsi_joint_noise": 0.05}), ROOT)[0] != frozen[0]


def test_train_entry_freezes_before_building_the_env_and_saving_the_config():
    src = open(os.path.join(ROOT, "src", "isaaclab", "train.py"), encoding="utf-8").read()
    i_load, i_freeze = src.index("config = Config(args_cli.config_path)"), src.index("freeze_residual_map(config)")
    assert i_load < i_freeze < src.index("build_env_cfg(config, BipedEnvCfg") < src.index("config.train.checkpoint_path = ")
    for rel in ("src/isaaclab/play.py", "src/isaaclab/check_actuator.py", "src/isaaclab/jitter_probe.py",
                "src/isaaclab/profile_train.py", "src/mujoco/sim2real.py"):
        assert "freeze_residual_map(" in open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8").read(), rel


# ------------------------------------------------------------------------------------------------ low-level consumers
@needs_table
def test_isaac_env_config_resolves_auto_defensively(capsys):
    from tests.test_residual_action_map import _conf, _fake_env_cfg
    from src.isaaclab.env_config import _apply_action_home
    home, scale, spawn, _ = rh.resolve_residual_map(_auto_ec(), ROOT)
    cfg = _fake_env_cfg()
    _apply_action_home(cfg, _conf(action_map="residual", action_home_deg="auto", action_scale_rad="auto",
                                  spawn_height_m="auto", reference={"npz_path": TABLE, "rsi_joint_noise": 0.05},
                                  commands=_auto_ec().commands))
    assert "NOT frozen" in capsys.readouterr().err
    assert dict(cfg.action_home_deg) == home and dict(cfg.action_scale_rad) == scale
    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, spawn) and cfg.target_base_height == spawn
    jp = cfg.scene.robot.init_state.joint_pos
    assert jp["l_knee.*"] == pytest.approx(math.radians(home["knee"]))
    with pytest.raises(ValueError, match="only apply to action_map: residual|auto only applies"):
        _apply_action_home(_fake_env_cfg(), _conf(action_map="centered", action_home_deg="auto", action_scale_rad="auto"))


def _mj_env_from(ec):
    from envs.mujoco.biped_env_v2 import BipedEnvV2
    ec.backlash = {**(ec.backlash or {}), "enabled": False}                    # MuJoCo backlash stays disabled
    return BipedEnvV2(history_size=1, use_history=False, env_config=ec)


def _lowest_foot_vertex_z(env):
    m, d = env.model, env.data
    zs = []
    for g in env._r_foot_geoms + env._l_foot_geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[a]: m.mesh_vertadr[a] + m.mesh_vertnum[a]]
            zs.append(float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()))
    return min(zs)


@needs_table
def test_mujoco_env_resolves_auto_and_spawns_at_the_requested_clearance(capsys):
    home, scale, spawn, rep = rh.resolve_residual_map(_auto_ec(), ROOT)
    ec = Config(REF_YAML).train.env_config                                     # the gait-reference config, unfrozen
    assert ec.action_home_deg == "auto"
    env = _mj_env_from(ec)
    assert "NOT frozen" in capsys.readouterr().err
    env.reset(seed=0)
    assert np.allclose(env._jpos(), home_rad(home), atol=1e-6) and np.allclose(env._default_joint_pos, home_rad(home), atol=1e-6)
    assert np.allclose(env._actions_to_targets(np.zeros(12)), home_rad(home), atol=1e-6)
    assert np.allclose(env._residual_scale, np.asarray(expand_by_type(scale, "s")))
    assert env._spawn_dz == pytest.approx(spawn - 0.29)
    # MuJoCo's XML root sits 10 mm above Isaac's 0.29, so the lowest sole point reads clearance + 10 mm there
    assert float(env._qpos()[2]) == pytest.approx(0.30 + spawn - 0.29)
    assert _lowest_foot_vertex_z(env) * 1e3 == pytest.approx(rep["spawn_fk"]["clearance_m"] * 1e3 + 10.0, abs=0.01)
    # the same env built from the frozen numbers is identical
    cfg = Config(REF_YAML)
    rh.freeze_residual_map(cfg, ROOT)
    env2 = _mj_env_from(cfg.train.env_config)
    env2.reset(seed=0)
    assert np.allclose(env2._qpos()[:7], env._qpos()[:7], atol=1e-9) and np.allclose(env2._jpos(), env._jpos())
    assert "NOT frozen" not in capsys.readouterr().err


@needs_table
def test_export_contract_resolves_auto_from_a_hand_made_config(tmp_path, capsys):
    from src.isaaclab import export_onnx as ex
    home, scale, _, _ = rh.resolve_residual_map(_auto_ec(), ROOT)
    (tmp_path / "config.yaml").write_text(textwrap.dedent(f"""\
        train:
          env_config:
            action_map: residual
            action_home_deg: auto
            action_scale_rad: auto
            reference:
              npz_path: {TABLE.replace(os.sep, '/')}
              rsi_joint_noise: 0.05
            commands:
              base_velocity:
                ranges:
                  lin_vel_x: [-0.2, 0.2]
                  lin_vel_y: [-0.15, 0.15]
                  ang_vel_z: [-1.2, 1.2]
        """))
    c = ex._load_policy_contract(str(tmp_path), None, None)
    assert "NOT frozen" in capsys.readouterr().err
    assert np.allclose(c["residual"]["home_rad"], home_rad(home)) and np.allclose(
        c["residual"]["scale_rad"], np.asarray(expand_by_type(scale, "s")))


# ------------------------------------------------------------------------------------------------ errors
def test_auto_home_without_a_reference_path_raises_clearly():
    with pytest.raises(ValueError, match=r"action_home_deg: auto needs env_config\.reference\.npz_path"):
        rh.resolve_residual_map(_ec(action_home_deg="auto", action_scale_rad=dict(SCALE), reference=None), ROOT)
    with pytest.raises(ValueError, match=r"action_scale_rad: auto needs env_config\.reference\.npz_path"):
        rh.resolve_residual_map(_ec(action_home_deg=dict(CROUCH), action_scale_rad="auto", reference={}), ROOT)


def test_other_resolver_errors(tmp_path):
    with pytest.raises(ValueError, match="auto only applies to action_map: residual"):
        rh.resolve_residual_map(_ec(action_map="centered", action_home_deg="auto"), ROOT)
    with pytest.raises(FileNotFoundError, match="gait table not found"):
        rh.resolve_residual_map(_ec(action_home_deg="auto", action_scale_rad="auto",
                                    reference={"npz_path": str(tmp_path / "nope.npz")}), ROOT)
    with pytest.raises(ValueError, match="needs env_config.action_home_deg"):
        rh.resolve_residual_map(_ec(action_scale_rad=dict(SCALE)), ROOT)
    with pytest.raises(ValueError, match="needs env_config.action_scale_rad"):
        rh.resolve_residual_map(_ec(action_home_deg=dict(CROUCH)), ROOT)
    with pytest.raises(ValueError, match="exactly the keys"):
        rh.resolve_residual_map(_ec(action_home_deg={"knee": -20.0}, action_scale_rad=dict(SCALE)), ROOT)
    with pytest.raises(ValueError, match="scale_headroom"):
        rh.resolve_residual_map(_auto_ec(scale_headroom=1.2), ROOT)
    with pytest.raises(ValueError, match="symmetric|inside"):
        rh.resolve_residual_map(_ec(action_home_deg={**CROUCH, "knee": 30.0}, action_scale_rad=dict(SCALE)), ROOT)


def test_schema_accepts_auto_and_numbers():
    from config.schema import EnvConfig
    e = EnvConfig(action_home_deg="auto", action_scale_rad="auto", spawn_height_m="auto")
    assert e.spawn_clearance_m == 0.0054 and e.scale_headroom == 0.96
    e = EnvConfig(action_home_deg=dict(CROUCH), action_scale_rad=dict(SCALE), spawn_height_m=0.286)
    assert e.spawn_height_m == 0.286 and dict(e.action_home_deg) == CROUCH
    with pytest.raises(Exception):
        EnvConfig(action_home_deg="automatic")
