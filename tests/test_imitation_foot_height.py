"""imitation_foot_height + the height-aware reward contact flag (CPU, no Isaac). Pure functions in
envs/isaaclab/rewards/rewards.py, config plumbing, and that the defaults reproduce the old
behaviour. Run from BipedRobot/:
    python -m pytest tests/test_imitation_foot_height.py -q
"""
import ast
import math
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

MM = 1e-3
STD = 0.004


def t(*rows):
    return torch.tensor(list(rows), dtype=torch.float32)


# ---------------- imitation_foot_height kernel ----------------

def test_height_kernel_values():
    # env order [right, left]; both feet share the same error so the mean equals the per-foot value
    # the foot sits on the ground (clearance 0) while the table asks for dh: the "dragged" case
    for dh, expect in ((0.0, 1.0), (4 * MM, math.exp(-1.0)), (11 * MM, math.exp(-(11 / 4) ** 2))):
        got = R.imitation_foot_height(t([0.0, 0.0]), t([dh, dh]), STD).item()
        assert got == pytest.approx(expect, rel=1e-4, abs=1e-6)
    assert math.exp(-(11 / 4) ** 2) == pytest.approx(0.0005, abs=1e-4)      # the dragged-foot case


def test_height_kernel_is_per_foot_then_mean():
    # one foot exact, one 11 mm low: mean of (1, ~0.0005), NOT exp(-mean(err^2)) of the pair
    got = R.imitation_foot_height(t([0.010, 0.0]), t([0.0, 0.010]), STD)
    # env [right=10mm, left=0]; table [left=0, right=10mm] -> both feet match
    assert got.item() == pytest.approx(1.0, abs=1e-6)
    one_miss = R.imitation_foot_height(t([0.0, 0.0]), t([0.0, 0.011]), STD).item()   # table right = 11 mm
    assert one_miss == pytest.approx(0.5 * (1.0 + math.exp(-(11 / 4) ** 2)), rel=1e-4)
    assert one_miss > 0.49          # the per-foot form keeps the gradient of the matching foot


def test_height_stance_is_one_and_batched():
    out = R.imitation_foot_height(torch.zeros(5, 2), torch.zeros(5, 2), STD)
    assert out.shape == (5,) and torch.allclose(out, torch.ones(5))


def test_negative_clearance_is_clamped_to_zero():
    # a foot below its datum (sinking into the floor) must read 0 clearance, not negative
    a = R.imitation_foot_height(t([-0.003, -0.003]), t([0.0, 0.0]), STD).item()
    assert a == pytest.approx(1.0, abs=1e-6)


def test_feet_order_env_right_left_vs_table_left_right():
    # right foot (env idx 0) lifted 10 mm. Table [left, right]: right swinging -> ref [0, 10 mm].
    env_right_up = t([0.010, 0.0])
    table_right_swing = t([0.0, 0.010])
    table_left_swing = t([0.010, 0.0])
    assert R.imitation_foot_height(env_right_up, table_right_swing, STD).item() == pytest.approx(1.0, abs=1e-6)
    wrong = R.imitation_foot_height(env_right_up, table_left_swing, STD).item()
    assert wrong == pytest.approx(math.exp(-(10 / 4) ** 2), abs=1e-4)      # both feet 10 mm off
    # and the left foot lifted matches the table's left swing
    env_left_up = t([0.0, 0.010])
    assert R.imitation_foot_height(env_left_up, table_left_swing, STD).item() == pytest.approx(1.0, abs=1e-6)


# ---------------- reward-side contact flag ----------------

def b(*rows):
    return torch.tensor(list(rows), dtype=torch.bool)


def test_dragged_unloaded_foot_counts_as_contact():
    force = b([False, False])                       # |F| <= 1 N on both
    clr = t([0.5 * MM, 10 * MM])                    # right dragged at 0.5 mm, left airborne at 10 mm
    out = R.feet_contact_with_clearance(force, clr, 0.002)
    assert out.tolist() == [[True, False]]


def test_loaded_foot_counts_regardless_of_height():
    out = R.feet_contact_with_clearance(b([True, True]), t([0.0, 8 * MM]), 0.002)
    assert out.tolist() == [[True, True]]           # |F| > 1 N wins even with clearance 8 mm


def test_airborne_foot_with_zero_force_is_not_contact():
    out = R.feet_contact_with_clearance(b([False, False]), t([10 * MM, 3 * MM]), 0.002)
    assert out.tolist() == [[False, False]]


def test_flag_boundary_and_negative_clearance():
    out = R.feet_contact_with_clearance(b([False, False, False]).reshape(1, 3),
                                        t([-0.001, 0.0019, 0.002]), 0.002)
    assert out.tolist() == [[True, True, False]]    # strict <, below-datum counts as ground


@pytest.mark.parametrize("thresh", [0.0, -1.0])
def test_disabled_threshold_reproduces_force_only(thresh):
    force = b([False, True], [True, False])
    clr = t([0.0, 0.0], [0.0, 0.0])
    assert torch.equal(R.feet_contact_with_clearance(force, clr, thresh), force)
    assert not R.clearance_contact(clr, thresh).any()


def test_imitation_contact_dragged_foot_no_longer_scores_in_swing():
    # reference swing of the right foot (table [left, right] stance=1: left stance, right swing)
    ref_contact = t([1.0, 0.0])
    force = b([False, True])                        # env [right, left]: right dragged, unloaded
    clr = t([0.4 * MM, 0.0])
    old = R.imitation_contact_match(force[:, [1, 0]], ref_contact).item()
    flag = R.feet_contact_with_clearance(force, clr, 0.002)
    new = R.imitation_contact_match(flag[:, [1, 0]], ref_contact).item()
    # old: dragged foot reads "airborne" = matches the swing (1.0). new: reads contact = mismatch (0.5)
    assert old == pytest.approx(1.0) and new == pytest.approx(0.5)
    # a properly lifted foot still matches
    lifted = R.feet_contact_with_clearance(force, t([10 * MM, 0.0]), 0.002)
    assert R.imitation_contact_match(lifted[:, [1, 0]], ref_contact).item() == pytest.approx(1.0)


# ---------------- feet_slide ----------------

def _hist(f_right, f_left):
    """(N=1, T=2, F=2, M=1, 3) force_matrix_w_history with the given |F| on z."""
    h = torch.zeros(1, 2, 2, 1, 3)
    h[0, :, 0, 0, 2] = f_right
    h[0, :, 1, 0, 2] = f_left
    return h


def _vel(vr, vl):
    v = torch.zeros(1, 2, 3)
    v[0, 0, 0] = vr
    v[0, 1, 0] = vl
    return v


def test_feet_slide_default_is_old_behaviour():
    h, v = _hist(0.2, 5.0), _vel(0.3, 0.1)
    assert R.feet_slide(h, v).item() == pytest.approx(0.1)               # only the loaded left foot
    assert R.feet_slide(h, v, None).item() == pytest.approx(0.1)


def test_feet_slide_penalises_dragged_unloaded_foot_with_extra_contact():
    h, v = _hist(0.2, 5.0), _vel(0.3, 0.1)
    clr = t([0.5 * MM, 0.0])
    extra = R.clearance_contact(clr, 0.002)
    assert R.feet_slide(h, v, extra).item() == pytest.approx(0.4)        # the dragged foot now counts
    airborne = R.clearance_contact(t([10 * MM, 0.0]), 0.002)
    assert R.feet_slide(h, v, airborne).item() == pytest.approx(0.1)     # a lifted foot is free to move


def test_feet_slide_with_vel_gets_flagged_contact():
    flag = (b([False, True]) | R.clearance_contact(t([0.5 * MM, 0.0]), 0.002)).float()
    assert R.feet_slide_with_vel(flag, t([0.3, 0.1])).item() == pytest.approx(0.4)


# ---------------- table field ----------------

def test_table_foot_height_field_is_stance_zero_and_two_wide():
    np = pytest.importorskip("numpy")
    path = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
    if not os.path.exists(path):
        pytest.skip("baked table not present")
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["field_names"]]
    widths = [int(w) for w in z["field_widths"]]
    i = names.index("foot_height")
    assert widths[i] == 2
    sl = slice(sum(widths[:i]), sum(widths[:i + 1]))
    assert float(z["stand"][sl].max()) < 1e-4             # stance baseline ~0.06 mm, no subtraction needed
    fh = z["data"][..., sl]
    assert abs(float(fh.min())) < 1e-4                     # stance ~0: the v2 bake dips to -0.04 mm (IK noise), kernel std is 10 mm
    assert 0.008 < float(fh.max()) < 0.025                 # 10-14 mm swing peaks (max 19.6 mm in some cells)


def test_table_reference_samples_foot_height_with_the_other_fields():
    pytest.importorskip("numpy")
    path = os.path.join(ROOT, "robot_motion_reference", "processed", "gait_table.npz")
    if not os.path.exists(path):
        pytest.skip("baked table not present")
    from utils.motion_reference import GaitTableReference
    g = GaitTableReference(path, torch.device("cpu"))
    cmd = torch.tensor([[0.1, 0.0, 0.0]])
    peak = max(g.field(g.sample(cmd, torch.tensor([p / 40.0]), torch.tensor([False])), "foot_height").max().item()
               for p in range(40))
    assert 0.008 < peak < 0.02
    ref = g.sample(cmd, torch.tensor([0.25]), torch.tensor([False]))      # phase 0.25: right foot mid-stance
    contact = g.field(ref, "contact")[0]                                  # [left, right]
    fh = g.field(ref, "foot_height")[0]
    # the foot that is in the air has the height, the stance foot ~0 (same [left, right] order as contact)
    for side in (0, 1):
        if contact[side] > 0.5:
            assert fh[side].item() < 1e-3


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
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def test_env_cfg_defaults_keep_old_behaviour():
    d = _cfg_defaults()
    assert d["contact_height_thresh_m"] is None                    # force-only flag by default
    assert d["reference_std"]["foot_height"] == 0.004
    assert d["reference_std"]["foot_pos"] == 0.02                  # the old keys are untouched


def test_schema_and_env_config_carry_the_key():
    pytest.importorskip("pydantic")
    from config.schema import EnvConfig
    from src.isaaclab import env_config
    assert EnvConfig().contact_height_thresh_m is None
    assert "contact_height_thresh_m" in env_config._SCALAR_FIELDS


def test_gait_reference_yaml_loads_and_validates():
    pytest.importorskip("yaml")
    from utils import Config
    from src.isaaclab import env_config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    w = ec.reward_weights
    assert (w["imitation_joint_pos"], w["imitation_foot_pos"], w["imitation_orientation"]) == (0.6, 0.4, 0.2)
    assert (w["imitation_contact"], w["imitation_lin_vel"], w["imitation_ang_vel"], w["imitation_joint_vel"]) == (0.5, 0.3, 0.2, 0.1)
    assert w["imitation_foot_height"] == 1.5 and w["swing_foot_height"] == 0.0
    assert ec.contact_height_thresh_m == 0.006               # soft mode ramp width (was 4 mm hard)
    ref = env_config._as_dict(ec.reference)
    env_config._check_keys("reference", ref, env_config._REFERENCE_KEYS)
    assert ref["std"]["foot_height"] == 0.010
    # every reference.std key must exist in the cfg default dict (the env raises on unknown keys)
    assert set(ref["std"]) <= set(_cfg_defaults()["reference_std"])
    # and it really goes through apply_env_config onto a cfg
    cfg = types.SimpleNamespace(rewards={}, reference_std=dict(_cfg_defaults()["reference_std"]),
                                contact_height_thresh_m=None, reference_rsi_prob=0.8)
    env_config._apply_reference(cfg, ec, training=True)
    assert cfg.reference_std["foot_height"] == 0.010
    for field in ("contact_height_thresh_m",):
        assert field in env_config._SCALAR_FIELDS and getattr(ec, field) is not None
        setattr(cfg, field, getattr(ec, field))
    assert cfg.contact_height_thresh_m == 0.006
