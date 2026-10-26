"""Reward-weight sign guard (CPU, no Isaac): rewards.validate_reward_weight_signs rejects a positive
weight on a penalty term (and a negative one on a reward term), names every offending key, and every
config in config/ passes. Run from BipedRobot/:
    python -m pytest tests/test_reward_sign_guard.py -q
"""
import ast
import glob
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402

CONFIGS = sorted(
    glob.glob(os.path.join(ROOT, "config", "*.yaml")) + glob.glob(os.path.join(ROOT, "config", "ablations", "*.yaml"))
)


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _load_env_config(path):
    from utils import Config
    return Config(path).train.env_config


def _weights_and_schedule(env_conf):
    weights = dict(getattr(env_conf, "reward_weights", None) or {})
    schedule = {k: getattr(env_conf, k, None) for k in R.SCHEDULED_WEIGHT_SIGN}
    return weights, schedule


def test_rejects_positive_foot_flat_and_names_key():
    with pytest.raises(ValueError) as e:
        R.validate_reward_weight_signs({"foot_flat_l2": 1.0, "track_lin_vel_xy_exp": 2.0})
    msg = str(e.value)
    assert "foot_flat_l2" in msg and "<= 0" in msg
    assert "track_lin_vel_xy_exp" not in msg


def test_accepts_negative_and_zero():
    R.validate_reward_weight_signs({"foot_flat_l2": -1.0, "leg_contact_l2": 0.0, "upright": 0.0, "imitation_contact": 0.5})


def test_lists_all_offenders_both_directions_and_schedule():
    with pytest.raises(ValueError) as e:
        R.validate_reward_weight_signs(
            {"foot_flat_l2": 1.0, "dof_torques_l2": 0.001, "track_ang_vel_z_exp": -1.5, "imitation_foot_height": -0.5},
            schedule={"action_rate_weight_start": -0.01, "action_rate_weight_end": 0.1},
        )
    msg = str(e.value)
    for key in ("foot_flat_l2", "dof_torques_l2", "track_ang_vel_z_exp", "imitation_foot_height", "action_rate_weight_end"):
        assert key in msg
    assert "action_rate_weight_start" not in msg
    assert ">= 0" in msg and "<= 0" in msg


def test_schedule_none_and_unknown_keys_ignored():
    R.validate_reward_weight_signs({"some_future_term": 5.0},
                                   schedule={"action_rate_weight_start": -0.01, "action_rate_weight_end": None})


def test_keys_restriction():
    w = {"foot_flat_l2": 1.0, "lin_vel_z_l2": -1.0}
    assert R.reward_weight_sign_errors(w, keys={"lin_vel_z_l2"}) == []
    assert len(R.reward_weight_sign_errors(w)) == 1


def test_every_env_term_is_classified():
    """Every weight key BipedEnv._get_rewards reads (plus the imitation terms) has an expected sign."""
    src = _read("envs/isaaclab/biped_env.py")
    used = set(re.findall(r'self\.cfg\.rewards\.get\("(\w+)"', src))
    used |= set(re.findall(r'"(imitation_\w+)":\s*rewards\.imitation', src))
    assert "foot_flat_l2" in used and "imitation_foot_height" in used
    assert used <= set(R.REWARD_WEIGHT_SIGN), sorted(used - set(R.REWARD_WEIGHT_SIGN))
    assert set(R.REWARD_WEIGHT_SIGN) <= used, sorted(set(R.REWARD_WEIGHT_SIGN) - used)


def test_isaac_env_init_calls_guard():
    tree = ast.parse(_read("envs/isaaclab/biped_env.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BipedEnv")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [c for c in ast.walk(init) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "validate_reward_weight_signs"]
    assert calls, "BipedEnv.__init__ must call rewards.validate_reward_weight_signs"


def test_mujoco_key_set_matches_source_and_guard_is_called():
    src = _read("envs/mujoco/biped_env_v2.py")
    body = src[src.index("def _compute_reward"):]
    used = set(re.findall(r'_w\("(\w+)"', body)) | set(re.findall(r'\bw\.get\("(\w+)"', body))
    m = re.search(r"_MUJOCO_REWARD_KEYS = frozenset\(\{(.*?)\}\)", src, re.S)
    declared = set(re.findall(r'"(\w+)"', m.group(1)))
    assert declared == used, (sorted(declared ^ used))
    assert declared <= set(R.REWARD_WEIGHT_SIGN)
    assert "R.validate_reward_weight_signs" in src


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_repo_config_passes(path):
    weights, schedule = _weights_and_schedule(_load_env_config(path))
    assert weights, "no reward_weights found"
    R.validate_reward_weight_signs(weights, schedule, where=os.path.basename(path))


def test_fixed_gait_reference_config_has_non_positive_foot_flat():
    weights, _ = _weights_and_schedule(_load_env_config(os.path.join(ROOT, "config", "config_gait_reference.yaml")))
    # never the +1.0 sign error that rewarded edge standing in run gaitref_2026-10-19_20-20-42
    assert weights["foot_flat_l2"] <= 0.0
    # E22: the guard is on (runs with 0 stood on their outer foot edges at command 0); strictly negative
    assert weights["foot_flat_l2"] < 0.0


def test_old_run_value_is_rejected_on_real_config():
    weights, schedule = _weights_and_schedule(_load_env_config(os.path.join(ROOT, "config", "config_gait_reference.yaml")))
    weights["foot_flat_l2"] = 1.0           # the value in checkpoints/BipedV2_SAC_gaitref_2026-10-19_20-20-42
    with pytest.raises(ValueError, match="foot_flat_l2"):
        R.validate_reward_weight_signs(weights, schedule)
