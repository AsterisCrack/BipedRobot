"""The observation after step(a) must carry a in its previous_actions slot [36:48].

The Jetson runtime feeds the action it just sent; the sims used to feed the one before it, so
the policy never saw the a_{t-1} that action_rate_l2 penalises. MuJoCo env only (CPU, no Isaac).
Run from BipedRobot/:  python -m pytest tests/test_prev_action_obs.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mujoco = pytest.importorskip("mujoco")
from envs.mujoco.biped_env_v2 import BipedEnvV2  # noqa: E402
from utils import Config  # noqa: E402

PREV = slice(36, 48)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_obs_carries_the_action_just_applied():
    cfg = Config(os.path.join(ROOT, "config", "config_gait_reference.yaml"))
    env = BipedEnvV2(history_size=3, use_history=True, env_config=cfg.train.env_config)
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["critic"][PREV], 0.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(-1.2, 1.2, 12).astype(np.float32)      # out-of-range on purpose: obs gets the clipped one
        obs, *_ = env.step(a)
        expect = np.clip(a, -1.0, 1.0)
        assert np.allclose(obs["critic"][PREV], expect)
        assert np.allclose(obs["actor"][-50:][PREV], expect)    # newest history frame is last
