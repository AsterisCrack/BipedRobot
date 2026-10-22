"""gait_clock_mode: fixed_period (no gait table) in the MuJoCo env. CPU only, no Isaac.

Phase advances by 2*pi*dt/period per step while moving, is held while the command is below the
stand threshold, and speed_proportional is unchanged. Run from BipedRobot/:
    python -m pytest tests/test_fixed_period_clock.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
mujoco = pytest.importorskip("mujoco")
from envs.mujoco.biped_env_v2 import BipedEnvV2, STEP_DT, GAIT_CLOCK_FREQ  # noqa: E402
from utils import Config  # noqa: E402

PHASE = slice(48, 50)
ZERO = np.zeros(12, dtype=np.float32)


def _env(mode, period=0.7):
    cfg = Config(os.path.join(ROOT, "config", "config.yaml"))
    ec = cfg.train.env_config
    ec.reference = None
    ec.gait_clock_mode = mode
    ec.gait_clock_period_s = period
    ec.gait_clock_stand_threshold = 0.05
    env = BipedEnvV2(history_size=1, use_history=False, env_config=ec)
    env.reset(seed=0)
    return env


def _step_phase(env, cmd):
    env.commands[:] = cmd
    before = env.gait_phase
    obs, *_ = env.step(ZERO)
    return before, env.gait_phase, obs


def test_fixed_period_advances_at_constant_rate():
    env = _env("fixed_period")
    want = 2 * math.pi * STEP_DT / 0.7
    for cmd in ((0.1, 0, 0), (0.4, 0, 0), (0.0, 0.1, 0)):
        b, a, _ = _step_phase(env, cmd)
        assert (a - b) % (2 * math.pi) == pytest.approx(want, abs=1e-9)


def test_fixed_period_holds_when_standing_and_obs_matches():
    env = _env("fixed_period")
    b, a, obs = _step_phase(env, (0.0, 0.0, 0.0))
    assert a == b
    b, a, obs = _step_phase(env, (0.04, 0.0, 0.0))
    assert a == b
    b, a, obs = _step_phase(env, (0.0, 0.0, 0.2))      # turning in place counts as moving
    assert a != b
    # the observation after the step shows the same phase the reward saw
    assert np.allclose(obs["critic"][PHASE], [math.sin(a), math.cos(a)], atol=1e-6)


def test_speed_proportional_unchanged():
    env = _env("speed_proportional")
    b, a, _ = _step_phase(env, (0.1, 0, 0))
    assert (a - b) % (2 * math.pi) == pytest.approx(STEP_DT * 2 * math.pi * GAIT_CLOCK_FREQ * 0.1, abs=1e-9)
    b, a, _ = _step_phase(env, (0.0, 0.0, 0.0))
    assert a == b


def test_period_is_configurable():
    env = _env("fixed_period", period=1.0)
    b, a, _ = _step_phase(env, (0.1, 0, 0))
    assert (a - b) % (2 * math.pi) == pytest.approx(2 * math.pi * STEP_DT, abs=1e-9)


def test_cadence_grid_table_clock(tmp_path):
    """A v2 gait table: the phase advances by 2*pi*dt*freq(cmd), freq trilinear in the command
    (Gait Studio's cadence law), still held while standing."""
    from utils.motion_reference import CadenceGrid

    gvx, gvy, gwz = np.array([-0.2, 0.0, 0.2]), np.array([-0.15, 0.0, 0.15]), np.array([-1.2, 0.0, 1.2])
    vx, vy, wz = np.meshgrid(gvx, gvy, gwz, indexing="ij")
    freq = 0.5 + 2.0 * np.abs(vx) + 1.5 * np.abs(vy) + 0.2 * np.abs(wz)
    path = tmp_path / "table.npz"
    np.savez(path, version=np.int32(2), grid_vx=gvx, grid_vy=gvy, grid_wz=gwz,
             freq=freq.astype(np.float32), period=np.float32(2.0))
    cfg = Config(os.path.join(ROOT, "config", "config.yaml"))
    ec = cfg.train.env_config
    ec.reference = {"npz_path": str(path), "stand_threshold": 0.05}
    env = BipedEnvV2(history_size=1, use_history=False, env_config=ec)
    env.reset(seed=0)
    grid = CadenceGrid(gvx, gvy, gwz, freq.astype(np.float32))
    for cmd in ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.05, 0.1, 0.6)):
        b, a, _ = _step_phase(env, cmd)
        assert (a - b) % (2 * math.pi) == pytest.approx(2 * math.pi * STEP_DT * grid.frequency(cmd), abs=1e-6)
    def rate(cmd):
        b, a, _ = _step_phase(env, cmd)
        return (a - b) % (2 * math.pi)
    assert rate((0.2, 0, 0)) > rate((0.1, 0, 0))   # faster command, faster cadence
    b, a, _ = _step_phase(env, (0.0, 0.0, 0.0))
    assert a == b
