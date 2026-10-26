"""track_lin_target: command | reference (CPU, no Isaac). The linear-velocity tracking kernel centred on the
gait table's own base velocity instead of the constant command. Pure function in
envs/isaaclab/rewards/rewards.py + config plumbing. Run from BipedRobot/:
    python -m pytest tests/test_track_lin_reference_target.py -q
"""
import ast
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

TABLE = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
STD_REL, STD_MIN = 0.7, 0.07
# Nominal commands; each is snapped to the nearest baked grid cell of whatever table is present (v1: 0.05 steps,
# v2: vx +-0.02 / 0.06 / 0.10 ..., vy +-0.018 / 0.055 / 0.091 ..., no wz = 0 point). No v1 grid points assumed.
GRID_CMDS = [(0.05, 0.0), (0.1, 0.0), (0.2, 0.0), (0.0, 0.1), (0.1, 0.1)]
STAND_THRESHOLD = 0.05                      # config_gait_reference.yaml reference.stand_threshold


def _table():
    z = np.load(TABLE, allow_pickle=False)
    names = [str(s) for s in z["field_names"]]
    edges = np.concatenate([[0], np.cumsum([int(w) for w in z["field_widths"]])])
    i = names.index("base_lin_vel_b")
    return z, slice(int(edges[i]), int(edges[i + 1]))


def _snap(z, vx, vy):
    """Nearest baked cell (vx, vy, wz ~ 0) -> (index triple, the cell's own vx, vy). The command used by the tests is the
    cell's grid value, so the table's mean base velocity over a cycle IS the command."""
    gi = [int(np.argmin(np.abs(z[k] - v))) for k, v in (("grid_vx", vx), ("grid_vy", vy), ("grid_wz", 0.0))]
    gx, gy = float(z["grid_vx"][gi[0]]), float(z["grid_vy"][gi[1]])
    step = max(float(np.diff(z["grid_vx"]).max()), float(np.diff(z["grid_vy"]).max()))
    assert abs(gx - vx) <= step / 2 + 1e-6 and abs(gy - vy) <= step / 2 + 1e-6, f"no grid cell near ({vx}, {vy})"
    return gi, gx, gy


def _cycle(z, sl, gi):
    """[F, 3] base_lin_vel_b of one cycle at grid cell gi."""
    return torch.as_tensor(z["data"][gi[0], gi[1], gi[2]][:, sl], dtype=torch.float32)


def _cmd(vx, vy, n):
    return torch.tensor([[vx, vy, 0.0]], dtype=torch.float32).repeat(n, 1)


# ---------------- 1. command mode is untouched ----------------

def test_target_equal_to_command_reproduces_the_command_kernel():
    g = torch.Generator().manual_seed(0)
    cmd = torch.randn(64, 3, generator=g) * 0.2
    cmd[:8] = 0.0                                              # zero commands hit the std_min floor
    vel = torch.randn(64, 3, generator=g) * 0.2
    old = R.track_lin_vel_xy_exp_rel(cmd, vel, std_rel=STD_REL, std_min=STD_MIN)
    new = R.track_lin_vel_xy_exp_rel_target(cmd[:, :2], cmd, vel, std_rel=STD_REL, std_min=STD_MIN)
    assert torch.equal(old, new)
    # the old function itself is not changed by the new option (same formula as before the change)
    std = torch.clamp(STD_REL * torch.norm(cmd[:, :2], dim=1), min=STD_MIN)
    ref = torch.exp(-torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1) / torch.square(std))
    assert torch.equal(old, ref)


def test_width_comes_from_the_command_not_the_target():
    cmd = torch.tensor([[0.1, 0.0, 0.0]])
    vel = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[0.3, 0.0]])                        # error 0.3 against std 0.07 (not 0.7 * 0.3)
    got = R.track_lin_vel_xy_exp_rel_target(target, cmd, vel, STD_REL, STD_MIN).item()
    assert got == pytest.approx(float(np.exp(-(0.3 / 0.07) ** 2)))


# ---------------- 2. the real table: a perfect copy vs standing ----------------

@pytest.mark.skipif(not os.path.exists(TABLE), reason="gait_table.npz not baked")
def test_perfect_copy_beats_standing_in_reference_mode():
    z, sl = _table()
    for req in GRID_CMDS:
        gi, vx, vy = _snap(z, *req)
        if np.hypot(vx, vy) < STAND_THRESHOLD:       # the env holds the stand frame there; not a walking target
            continue
        v = _cycle(z, sl, gi)
        n = v.shape[0]
        cmd = _cmd(vx, vy, n)
        # the table's mean is the command (so the old kernel is the only thing fighting the sway); v2 has no wz = 0 cell,
        # the nearest is wz = +-0.1 whose rotation shifts the cycle mean by up to ~2.3 mm/s
        assert v[:, :2].mean(0).numpy() == pytest.approx([vx, vy], abs=5e-3)

        walk_new = R.track_lin_vel_xy_exp_rel_target(v[:, :2], cmd, v, STD_REL, STD_MIN)
        assert torch.allclose(walk_new, torch.ones(n), atol=1e-6)            # perfect copy = 1.0 on every frame
        zero = torch.zeros_like(v)
        stand_new = R.track_lin_vel_xy_exp_rel_target(v[:, :2], cmd, zero, STD_REL, STD_MIN).mean().item()
        stand_old = R.track_lin_vel_xy_exp_rel(cmd, zero, STD_REL, STD_MIN).mean().item()
        if abs(vy) < 0.03:
            # (near-)forward commands: reference mode never pays standing more than the old kernel
            assert stand_new <= stand_old + 0.03, (vx, vy, stand_new, stand_old)   # grid without vy = 0: +0.016 at vy -0.022
        # lateral commands score up to ~0.42 (sway crosses 0; 0.37 on the previous bake): bake-dependent, so the invariant is the
        # relative one -- standing earns at most half of a perfect copy (1.0 on every frame, asserted above)
        assert stand_new <= 0.5 * walk_new.mean().item(), (vx, vy, stand_new)
        assert 1.0 - stand_new >= 0.5, (vx, vy, stand_new)


@pytest.mark.skipif(not os.path.exists(TABLE), reason="gait_table.npz not baked")
def test_old_command_kernel_penalises_the_perfect_copy_documents_the_bug():
    z, sl = _table()

    def old_kernel(gi, vx, vy):
        v = _cycle(z, sl, gi)
        cmd = _cmd(vx, vy, v.shape[0])
        return (R.track_lin_vel_xy_exp_rel(cmd, v, STD_REL, STD_MIN).mean().item(),
                R.track_lin_vel_xy_exp_rel(cmd, torch.zeros_like(v), STD_REL, STD_MIN).mean().item())

    perfect, stand = {}, {}
    for req in GRID_CMDS:
        gi, vx, vy = _snap(z, *req)
        perfect[(vx, vy)], stand[(vx, vy)] = old_kernel(gi, vx, vy)
    # The smallest MOVING cell above the stand threshold (|cmd| >= 0.05; v2: vx 0.02, vy 0.055, |cmd| 0.058): STANDING
    # beats a perfect copy of the table (negative gap) -- the bug that track_lin_target: reference fixes.
    cells = [(float(x), float(y)) for x in z["grid_vx"] for y in z["grid_vy"]
             if np.hypot(float(x), float(y)) >= STAND_THRESHOLD]
    vx, vy = min(cells, key=lambda c: np.hypot(*c))
    gi, _, _ = _snap(z, vx, vy)
    p, s_ = old_kernel(gi, vx, vy)
    # Relative statement of the bug, valid for any sway-y table (not table-specific absolute numbers): the old
    # (command-target) kernel scores a perfect copy of the table at most ~0.6 and clearly below the reference-target
    # kernel's 1.0, and its perfect-minus-standing gap is less than half of the new kernel's gap.
    v = _cycle(z, sl, gi)
    new_p = R.track_lin_vel_xy_exp_rel_target(v[:, :2], _cmd(vx, vy, v.shape[0]), v, STD_REL, STD_MIN).mean().item()
    new_s = R.track_lin_vel_xy_exp_rel_target(v[:, :2], _cmd(vx, vy, v.shape[0]), torch.zeros_like(v), STD_REL,
                                              STD_MIN).mean().item()
    assert new_p == pytest.approx(1.0, abs=1e-5)
    assert p <= 0.7 and p < 0.75 * new_p, (vx, vy, p, new_p)                 # current table: 0.53 (v2 table: 0.26)
    assert (p - s_) < 0.5 * (new_p - new_s), (vx, vy, p, s_, new_p, new_s)   # current: -0.07 vs 0.76
    # The next forward step up (vx 0.1): a perfect copy still scores well under 1, though it now beats standing
    (g_vx, g_vy), = [min(perfect, key=lambda c: abs(c[0] - 0.1) + abs(c[1]))]             # nearest cell to (0.1, 0)
    assert perfect[(g_vx, g_vy)] < 0.8 and stand[(g_vx, g_vy)] < 0.3 and perfect[(g_vx, g_vy)] > stand[(g_vx, g_vy)]
    # and no commanded walking gets near 1.0 at the small commands (current table: 0.2-0.65 at the snapped cells)
    assert max(perfect.values()) < 0.9, perfect


# ---------------- 3. validation + config plumbing ----------------

def test_validate_track_lin_target():
    R.validate_track_lin_target("command", has_reference=False, std_rel=0.0)      # old configs: always fine
    R.validate_track_lin_target("command", has_reference=True, std_rel=0.7)
    R.validate_track_lin_target("reference", has_reference=True, std_rel=0.7)
    with pytest.raises(ValueError, match="expected command | reference"):
        R.validate_track_lin_target("table", has_reference=True, std_rel=0.7)
    with pytest.raises(ValueError, match="gait-table reference"):
        R.validate_track_lin_target("reference", has_reference=False, std_rel=0.7)
    with pytest.raises(ValueError, match="track_lin_std_rel"):
        R.validate_track_lin_target("reference", has_reference=True, std_rel=0.0)


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


def test_defaults_and_plumbing_keep_old_behaviour():
    assert _cfg_defaults()["track_lin_target"] == "command"
    pytest.importorskip("pydantic")
    from config.schema import EnvConfig
    from src.isaaclab import env_config
    assert EnvConfig().track_lin_target == "command"
    assert "track_lin_target" in env_config._SCALAR_FIELDS


def test_gait_reference_config_uses_reference_target_and_config_yaml_does_not():
    pytest.importorskip("yaml")
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    assert ec.track_lin_target == "reference"
    assert ec.track_lin_std_rel > 0.0 and ec.reference["npz_path"]            # what the env requires for reference
    R.validate_track_lin_target(ec.track_lin_target, has_reference=bool(ec.reference["npz_path"]), std_rel=ec.track_lin_std_rel)
    ec0 = Config(os.path.join(ROOT, "config", "config.yaml")).train.env_config
    assert getattr(ec0, "track_lin_target", "command") == "command"


def test_env_wiring_samples_the_reference_once_before_tracking():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    assert "rewards.validate_track_lin_target(" in src
    body = src[src.index("def _get_rewards"):src.index("def _reference_standing")]
    i_sample = body.index("ref = self._sample_reference(")
    i_track = body.index("rewards.track_lin_vel_xy_exp_rel_target(")
    i_imit = body.index("self._imitation_terms(feet_pos, feet_contact_rw, clearance, ref)")
    assert body.index("self._advance_gait_phase(") < i_sample < i_track < i_imit
    assert body.count("_sample_reference(") == 1                               # one sample per step, shared
    assert 'self.robot.data.root_lin_vel_b' in body[i_track:i_track + 400]     # fresh, not the stale cache
    assert "self.gait_ref.field(ref, \"base_lin_vel_b\")[:, :2]" in body[i_track - 200:i_track + 200]
    # command mode keeps its old stale-cache call
    assert "rewards.track_lin_vel_xy_exp_rel(\n                self.commands, self.base_lin_vel_b," in body
