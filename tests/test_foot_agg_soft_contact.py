"""imitation_foot_agg (mean | min), contact_height_mode (hard | soft) and the flat-terrain reset datum
(CPU, no Isaac). Pure functions in envs/isaaclab/rewards/rewards.py + config plumbing. Run from BipedRobot/:
    python -m pytest tests/test_foot_agg_soft_contact.py -q
"""
import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

MM = 1e-3
STD = 0.010


def t(*rows):
    return torch.tensor(rows, dtype=torch.float32)


# ---------------- imitation_foot_height: mean vs min ----------------
# env order [right, left]; the table asks for 10 mm on BOTH feet so the order does not matter here.

def test_foot_height_agg_perfect_half_limp():
    ref = t([0.010, 0.010])
    perfect = t([0.010, 0.010])
    half = t([0.005, 0.005])                  # symmetric 5 mm lift
    limp = t([0.010, 0.0])                    # one foot right, one dragged
    k5, k10 = 0.7788, 0.3679                  # exp(-(5/10)^2), exp(-(10/10)^2)
    for agg_min in (False, True):
        assert R.imitation_foot_height(perfect, ref, STD, agg_min).item() == pytest.approx(1.0)
    assert R.imitation_foot_height(half, ref, STD, False).item() == pytest.approx(k5, abs=1e-3)
    assert R.imitation_foot_height(half, ref, STD, True).item() == pytest.approx(k5, abs=1e-3)     # symmetric: same
    assert R.imitation_foot_height(limp, ref, STD, False).item() == pytest.approx((1.0 + k10) / 2, abs=1e-3)
    assert R.imitation_foot_height(limp, ref, STD, True).item() == pytest.approx(k10, abs=1e-3)
    # the symmetric-vs-limp gap: 0.095 with mean, 0.190 with min (doubled)
    gap = lambda m: (R.imitation_foot_height(half, ref, STD, m) - R.imitation_foot_height(limp, ref, STD, m)).item()
    assert gap(False) == pytest.approx((k5 - (1.0 + k10) / 2), abs=1e-3)
    assert gap(True) == pytest.approx(k5 - k10, abs=1e-3) and gap(True) > 1.9 * gap(False)


def test_foot_height_defaults_reproduce_old_outputs():
    clr, ref = t([0.003, 0.012], [0.0, 0.0], [0.02, 0.001]), t([0.010, 0.0], [0.0, 0.0], [0.010, 0.010])
    h = torch.flip(torch.clamp(clr, min=0.0), [1])
    old = torch.mean(torch.exp(-torch.square((h - ref) / STD)), dim=-1)          # the pre-change formula
    assert torch.equal(R.imitation_foot_height(clr, ref, STD), old)
    assert torch.equal(R.imitation_foot_height(clr, ref, STD, False), old)


# ---------------- soft contact score ----------------

def test_soft_contact_score_unloaded_clearances():
    force = torch.zeros(1, 4, dtype=torch.bool)
    clr = t([0.0, 3 * MM, 6 * MM, 10 * MM])
    c = R.soft_contact_score(force, clr, 0.006)
    assert c.tolist()[0] == pytest.approx([1.0, 0.5, 0.0, 0.0], abs=1e-5)


def test_soft_contact_score_force_wins_and_negative_clearance_clamped():
    c = R.soft_contact_score(torch.tensor([[True, False]]), t([10 * MM, -2 * MM]), 0.006)
    assert c.tolist()[0] == pytest.approx([1.0, 1.0])            # loaded = 1 at any height; below datum = 1
    assert c.dtype == torch.float32 and c.min() >= 0 and c.max() <= 1


def test_soft_contact_score_disabled_threshold_is_force_flag():
    force = torch.tensor([[True, False]])
    for th in (0.0, -1.0):
        assert R.soft_contact_score(force, t([0.0, 0.0]), th).tolist()[0] == [1.0, 0.0]


# ---------------- imitation_contact_match ----------------

def test_contact_match_hard_default_is_old_binary():
    flag = torch.tensor([[True, False], [False, False]])
    ref = t([1.0, 0.0], [1.0, 1.0])
    old = torch.mean(((flag.float() > 0.5) == (ref > 0.5)).float(), dim=-1)
    assert torch.equal(R.imitation_contact_match(flag, ref), old)
    assert torch.equal(R.imitation_contact_match(flag, ref, False, False), old)
    assert R.imitation_contact_match(flag, ref).tolist() == [1.0, 0.0]


def test_contact_match_soft_is_one_minus_abs_diff():
    ref = t([0.0, 1.0])                                          # table [left, right]: left swing, right stance
    c = t([0.5, 1.0])                                            # [left, right]: left half-lifted (3 mm)
    assert R.imitation_contact_match(c, ref, True, False).item() == pytest.approx((0.5 + 1.0) / 2)
    assert R.imitation_contact_match(c, ref, True, True).item() == pytest.approx(0.5)


def test_contact_match_min_aggregation_binary():
    flag = torch.tensor([[True, False]])                         # table [left, right]
    ref = t([1.0, 1.0])                                          # right foot should be down but is not
    assert R.imitation_contact_match(flag, ref, False, False).item() == pytest.approx(0.5)
    assert R.imitation_contact_match(flag, ref, False, True).item() == pytest.approx(0.0)


def test_contact_match_soft_removes_the_threshold_cliff():
    # hard 4 mm: 3.9 mm reads contact, 4.1 mm airborne (full flip); soft 6 mm: smooth
    ref = t([0.0, 0.0])
    hard = lambda c: R.imitation_contact_match(R.feet_contact_with_clearance(torch.zeros(1, 2, dtype=torch.bool), t(c), 0.004), ref).item()
    soft = lambda c: R.imitation_contact_match(R.soft_contact_score(torch.zeros(1, 2, dtype=torch.bool), t(c), 0.006), ref, True).item()
    assert hard([3.9 * MM, 3.9 * MM]) == 0.0 and hard([4.1 * MM, 4.1 * MM]) == 1.0
    assert abs(soft([3.9 * MM, 3.9 * MM]) - soft([4.1 * MM, 4.1 * MM])) < 0.05


# ---------------- feet_slide weighting ----------------

def _hist(f_right, f_left):
    h = torch.zeros(1, 2, 2, 1, 3)
    h[0, :, 0, 0, 2] = f_right
    h[0, :, 1, 0, 2] = f_left
    return h


def _vel(vr, vl):
    v = torch.zeros(1, 2, 3)
    v[0, 0, 0] = vr
    v[0, 1, 0] = vl
    return v


def test_feet_slide_soft_weight_scales_speed():
    h, v = _hist(0.2, 5.0), _vel(0.3, 0.1)                       # right unloaded (<1 N), left loaded
    # c = max(force_flag, soft): right at 3 mm -> 0.5, left loaded -> 1
    soft = R.clearance_contact_soft(t([3 * MM, 0.0]), 0.006)
    assert R.feet_slide(h, v, soft).item() == pytest.approx(0.5 * 0.3 + 1.0 * 0.1)
    airborne = R.clearance_contact_soft(t([10 * MM, 0.0]), 0.006)
    assert R.feet_slide(h, v, airborne).item() == pytest.approx(0.1)
    dragged = R.clearance_contact_soft(t([0.0, 0.0]), 0.006)
    assert R.feet_slide(h, v, dragged).item() == pytest.approx(0.4)
    # a loaded foot is never down-weighted by a small soft value
    assert R.feet_slide(h, v, t([0.0, 0.0])).item() == pytest.approx(0.1)


def test_feet_slide_with_vel_accepts_soft_weight():
    c = R.soft_contact_score(torch.tensor([[False, True]]), t([3 * MM, 0.0]), 0.006)
    assert R.feet_slide_with_vel(c, t([0.3, 0.1])).item() == pytest.approx(0.25)


def test_feet_slide_defaults_unchanged():
    h, v = _hist(0.2, 5.0), _vel(0.3, 0.1)
    assert R.feet_slide(h, v).item() == pytest.approx(0.1)
    assert R.feet_slide(h, v, torch.tensor([[True, False]])).item() == pytest.approx(0.4)      # bool OR as before


# ---------------- reference-table scenario (limp / half / perfect) ----------------

def test_table_cycle_min_widens_the_limp_gap():
    pytest.importorskip("numpy")
    path = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
    if not os.path.exists(path):
        pytest.skip("baked table not present")
    from utils.motion_reference import GaitTableReference
    g = GaitTableReference(path, torch.device("cpu"))
    P = 100
    ref = g.sample(torch.tensor([[0.2, 0.0, 0.0]]).repeat(P, 1), torch.arange(P) / P, torch.zeros(P, dtype=torch.bool))
    fh = g.field(ref, "foot_height")                              # table order [left, right]
    limp = fh.clone()
    limp[:, 1] = 0.0
    half = 0.5 * fh
    res = {}
    for agg_min in (False, True):
        for name, c in (("perfect", fh), ("half", half), ("limp", limp)):
            res[(name, agg_min)] = R.imitation_foot_height(c[:, [1, 0]], fh, STD, agg_min).mean().item()
        assert res[("perfect", agg_min)] == pytest.approx(1.0, abs=1e-5)
    gap = lambda m: res[("half", m)] - res[("limp", m)]           # symmetric half-height minus limp
    assert gap(False) > 0.0 and gap(True) > 1.8 * gap(False)      # min more than doubles the limp penalty


# ---------------- config plumbing ----------------

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


def test_defaults_are_old_behaviour():
    d = _cfg_defaults()
    assert d["contact_height_mode"] == "hard" and d["imitation_foot_agg"] == "mean"
    assert d["contact_height_thresh_m"] is None
    pytest.importorskip("pydantic")
    from config.schema import EnvConfig
    from src.isaaclab import env_config
    e = EnvConfig()
    assert e.contact_height_mode == "hard" and e.imitation_foot_agg == "mean"
    assert "contact_height_mode" in env_config._SCALAR_FIELDS and "imitation_foot_agg" in env_config._SCALAR_FIELDS


def test_gait_reference_config_loads_with_new_keys_and_passes_sign_guard():
    pytest.importorskip("yaml")
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    assert ec.imitation_foot_agg == "min"
    assert ec.contact_height_mode == "soft"
    assert ec.contact_height_thresh_m == pytest.approx(0.006)
    w = dict(ec.reward_weights)
    assert (w["imitation_foot_height"], w["imitation_contact"], w["imitation_foot_pos"], w["feet_slide"]) == (1.5, 0.5, 0.4, -0.7)
    R.validate_reward_weight_signs(w, schedule={k: getattr(ec, k, None) for k in R.SCHEDULED_WEIGHT_SIGN})


def test_env_validates_new_keys_and_seeds_flat_datum():
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    assert 'cfg.imitation_foot_agg not in ("mean", "min")' in src
    assert 'cfg.contact_height_mode not in ("hard", "soft")' in src
    # flat terrain seeds the datum from env_origins z + the flat foot-origin height; rough keeps foot_z
    assert "self.scene.env_origins[fresh][:, 2:3] + self._flat_foot_origin_z" in src
    assert "if self.cfg.use_rough_terrain:" in src
