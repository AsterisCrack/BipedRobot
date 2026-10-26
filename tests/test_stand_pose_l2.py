"""stand_pose_l2: non-saturating stand-pose anchor, only while the gait reference is standing (CPU, no Isaac).
Pure kernel in envs/isaaclab/rewards/rewards.py + sign guard + config + source-level wiring. Run from BipedRobot/:
    python -m pytest tests/test_stand_pose_l2.py -q
"""
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
# run gaitref_2026-10-21_07-17-11 at command 0: deviation from the stand pose [deg], env order l/r pairs
# (hip_yaw, hip_roll, hip_pitch, knee, ankle_roll, ankle_pitch)
BRACE_DEG = [-28, -18, -4, -13, 33, 15, 26, 24, -22, 20, 2, 19]


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _ones(n):
    return torch.ones(n)


# ---------------- 1. kernel ----------------

def test_zero_when_on_the_reference():
    q = torch.randn(5, 12)
    assert torch.equal(R.stand_pose_l2(q, q.clone(), _ones(5)), torch.zeros(5))


def test_standing_envs_get_the_sum_of_squares_and_walking_envs_zero():
    g = torch.Generator().manual_seed(0)
    q, qr = torch.randn(6, 12, generator=g), torch.randn(6, 12, generator=g)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    out = R.stand_pose_l2(q, qr, mask)
    full = ((q - qr) ** 2).sum(dim=1)
    assert out.shape == (6,)
    assert torch.allclose(out[mask > 0], full[mask > 0], atol=1e-6)
    assert torch.equal(out[mask == 0], torch.zeros(2))            # exactly 0, not ~0


def test_brace_and_two_degree_values():
    brace = torch.deg2rad(torch.tensor([BRACE_DEG], dtype=torch.float32))
    got = R.stand_pose_l2(brace, torch.zeros(1, 12), _ones(1)).item()
    assert got == pytest.approx(sum(math.radians(d) ** 2 for d in BRACE_DEG), rel=1e-5)
    assert got == pytest.approx(1.53, abs=0.05)                  # ~1.5 rad^2: weight -1.0 costs ~1.5 per step there
    two = torch.full((1, 12), math.radians(2.0))
    assert R.stand_pose_l2(two, torch.zeros(1, 12), _ones(1)).item() == pytest.approx(0.015, abs=1e-3)
    # the same brace on a walking env costs nothing
    assert R.stand_pose_l2(brace, torch.zeros(1, 12), torch.zeros(1)).item() == 0.0


def test_does_not_saturate():
    """The point of the term: the gradient never vanishes, unlike imitation_joint_pos (~0 at the brace)."""
    q = torch.zeros(3, 12)
    q[:, 3] = torch.tensor([0.1, 0.4, 0.8])
    out = R.stand_pose_l2(q, torch.zeros(3, 12), _ones(3))
    assert torch.allclose(out, q[:, 3] ** 2, atol=1e-7) and bool((out[1:] > out[:-1]).all())
    k = R.imitation_mse_exp(torch.deg2rad(torch.tensor([BRACE_DEG], dtype=torch.float32)), torch.zeros(1, 12), 0.15).item()
    assert k < 0.01                                               # the existing anchor is flat there


def test_validate_requires_a_reference_for_a_nonzero_weight():
    R.validate_stand_pose_l2(0.0, has_reference=False)
    R.validate_stand_pose_l2(-1.0, has_reference=True)
    with pytest.raises(ValueError, match="stand_pose_l2"):
        R.validate_stand_pose_l2(-1.0, has_reference=False)


# ---------------- 2. sign guard ----------------

def test_sign_is_negative_and_positive_weight_rejected():
    assert R.REWARD_WEIGHT_SIGN["stand_pose_l2"] == -1
    R.validate_reward_weight_signs({"stand_pose_l2": -1.0})
    R.validate_reward_weight_signs({"stand_pose_l2": 0.0})
    with pytest.raises(ValueError, match="stand_pose_l2"):
        R.validate_reward_weight_signs({"stand_pose_l2": 1.0})


# ---------------- 3. configs ----------------

def test_configs_load_with_the_expected_weight():
    pytest.importorskip("yaml")
    from utils import Config
    ec = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml")).train.env_config
    assert ec.reward_weights["stand_pose_l2"] == -1.0
    assert ec.reference["npz_path"]                               # the env needs the table for a nonzero weight
    R.validate_stand_pose_l2(ec.reward_weights["stand_pose_l2"], has_reference=bool(ec.reference["npz_path"]))
    ec0 = Config(os.path.join(ROOT, "config", "config.yaml")).train.env_config
    w0 = dict(getattr(ec0, "reward_weights", None) or {})
    assert w0.get("stand_pose_l2", 0.0) == 0.0                    # absent -> default 0, no reference there
    R.validate_stand_pose_l2(w0.get("stand_pose_l2", 0.0), has_reference=bool(getattr(ec0, "reference", None) and ec0.reference.get("npz_path")))


# ---------------- 4. env wiring (source level) ----------------

def test_env_wiring_uses_the_shared_reference_frame_and_fresh_joint_pos():
    src = _src("envs/isaaclab/biped_env.py")
    assert "rewards.validate_stand_pose_l2(" in src[:src.index("def _get_rewards")]          # at init
    body = src[src.index("def _get_rewards"):src.index("def _reference_standing")]
    assert body.count("_sample_reference(") == 1 and "gait_ref.sample(" not in body        # no second sample
    i_sample = body.index("ref = self._sample_reference(")
    i_call = body.index("rewards.stand_pose_l2(")
    assert i_sample < i_call
    call = body[i_call:i_call + 300]
    assert "self.robot.data.joint_pos" in call                                             # fresh, not the stale cache
    assert 'self.gait_ref.joint_field(ref, "joint_pos")' in call
    assert "ref_standing.float()" in call                                                  # the step's stand mask ...
    assert "ref_standing = self._reference_standing()" in body[:i_sample]                  # ... computed once, before the sample
    assert "ref is not None" in body[i_call - 200:i_call]                                  # no reference -> term is 0
    assert '"stand_pose_l2": r_stand_pose * self.cfg.rewards.get("stand_pose_l2", 0.0)' in body   # weighted, summed, logged


# ---------------- 5. the real table ----------------

@pytest.mark.skipif(not os.path.exists(TABLE), reason="gait_table.npz not baked")
def test_joint_field_follows_the_robot_order_and_stand_frame_is_the_reference_when_standing():
    from utils.motion_reference import GaitTableReference
    g = GaitTableReference(TABLE, torch.device("cpu"))
    names = list(reversed(g.joint_names))                          # a robot order different from the table's
    g.bind_to_robot(names)
    n = 4
    cmd = torch.zeros(n, 3)
    cmd[2:, 0] = 0.1                                               # envs 2, 3 walk
    standing = torch.tensor([True, True, False, False])
    ref = g.sample(cmd, torch.full((n,), 0.3), standing)
    q_ref = g.joint_field(ref, "joint_pos")
    table_q = g.field(ref, "joint_pos")
    for i, name in enumerate(names):
        assert torch.equal(q_ref[:, i], table_q[:, g.joint_names.index(name)])
    # standing envs: exactly the table's stand frame, whatever the phase
    stand_q = g.field(g.stand.reshape(1, -1), "joint_pos")
    assert torch.equal(table_q[0], stand_q[0]) and torch.equal(table_q[1], stand_q[0])
    # at the stand frame the term is 0; away from it, only standing envs pay
    out = R.stand_pose_l2(q_ref + 0.05, q_ref, standing.float())
    assert out[:2].tolist() == pytest.approx([12 * 0.05 ** 2] * 2, rel=1e-5) and out[2:].tolist() == [0.0, 0.0]
