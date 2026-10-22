"""Reward-shaping pure functions (CPU, no Isaac): relative tracking kernels, touchdown-paid
feet_air_time, swing-time gate for swing_foot_height. Run from BipedRobot/:
    python -m pytest tests/test_reward_shaping.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
from envs.isaaclab.rewards import rewards as R  # noqa: E402


def _cmd(vx=0.0, vy=0.0, wz=0.0):
    return torch.tensor([[vx, vy, wz]], dtype=torch.float32)


def _vel(vx=0.0, vy=0.0, wz=0.0):
    return torch.tensor([[vx, vy, 0.0]], dtype=torch.float32), torch.tensor([[0.0, 0.0, wz]], dtype=torch.float32)


# ---------------- relative tracking ----------------

@pytest.mark.parametrize("cmd", [0.1, 0.2, 0.4])
def test_lin_rel_standing_still_is_cheap(cmd):
    lin, _ = _vel(0.0)
    r = R.track_lin_vel_xy_exp_rel(_cmd(cmd), lin, 0.7, 0.07).item()
    assert r == pytest.approx(0.13, abs=0.015)
    # the old fixed kernel paid far more at the smallest command
    if cmd == 0.1:
        assert R.track_lin_vel_xy_exp(_cmd(cmd), lin, std=0.2).item() == pytest.approx(0.78, abs=0.01)


@pytest.mark.parametrize("cmd", [0.1, 0.2, 0.4])
def test_lin_rel_half_speed(cmd):
    lin, _ = _vel(0.5 * cmd)
    # at cmd 0.1 the min std (0.07) is exactly rel*cmd, so the number is the same
    assert R.track_lin_vel_xy_exp_rel(_cmd(cmd), lin, 0.7, 0.07).item() == pytest.approx(0.6, abs=0.03)


def test_lin_rel_zero_command_drift():
    lin, _ = _vel(0.05)
    assert R.track_lin_vel_xy_exp_rel(_cmd(0.0), lin, 0.7, 0.07).item() == pytest.approx(0.6, abs=0.01)
    lin0, _ = _vel(0.0)
    assert R.track_lin_vel_xy_exp_rel(_cmd(0.0), lin0, 0.7, 0.07).item() == pytest.approx(1.0)


def test_lin_rel_perfect_and_uses_xy_norm():
    lin, _ = _vel(0.2, 0.2)
    assert R.track_lin_vel_xy_exp_rel(_cmd(0.2, 0.2), lin, 0.7, 0.07).item() == pytest.approx(1.0)
    # standing with a diagonal command: the std follows the xy NORM (0.283), not a component
    r = R.track_lin_vel_xy_exp_rel(_cmd(0.2, 0.2), _vel(0.0)[0], 0.7, 0.07).item()
    assert r == pytest.approx(0.13, abs=0.015)


def test_ang_rel():
    _, ang0 = _vel(wz=0.0)
    # yaw 0.3 ignored: std = 0.21, exp(-0.09/0.0441) = 0.13
    assert R.track_ang_vel_z_exp_rel(_cmd(wz=0.3), ang0, 0.7, 0.15).item() == pytest.approx(0.13, abs=0.015)
    assert R.track_ang_vel_z_exp_rel(_cmd(wz=-0.3), ang0, 0.7, 0.15).item() == pytest.approx(0.13, abs=0.015)
    # half the yaw rate: ~0.6
    _, ang_half = _vel(wz=0.15)
    assert R.track_ang_vel_z_exp_rel(_cmd(wz=0.3), ang_half, 0.7, 0.15).item() == pytest.approx(0.6, abs=0.03)
    # min std floor at zero command: drift 0.1 rad/s -> exp(-(0.1/0.15)^2) = 0.64
    _, ang_d = _vel(wz=0.1)
    assert R.track_ang_vel_z_exp_rel(_cmd(), ang_d, 0.7, 0.15).item() == pytest.approx(0.64, abs=0.01)
    # the old fixed kernel paid 0.85+ for ignoring 0.2 rad/s
    assert R.track_ang_vel_z_exp(_cmd(wz=0.2), ang0, std=0.5).item() > 0.85


def test_rel_is_per_env():
    cmds = torch.tensor([[0.1, 0, 0], [0.4, 0, 0]], dtype=torch.float32)
    lin = torch.zeros(2, 3)
    r = R.track_lin_vel_xy_exp_rel(cmds, lin, 0.7, 0.07)
    assert r.shape == (2,)
    assert r[0].item() == pytest.approx(0.13, abs=0.015) and r[1].item() == pytest.approx(0.13, abs=0.015)


def test_old_tracking_functions_unchanged():
    lin, ang = _vel(0.1, 0.0, 0.2)
    c = _cmd(0.3, 0.0, 0.5)
    err = (0.3 - 0.1) ** 2
    assert R.track_lin_vel_xy_exp(c, lin, std=0.2).item() == pytest.approx(torch.exp(torch.tensor(-err / 0.04)).item(), rel=1e-5)
    assert R.track_ang_vel_z_exp(c, ang, std=0.5).item() == pytest.approx(torch.exp(torch.tensor(-(0.3) ** 2 / 0.25)).item(), rel=1e-5)


# ---------------- touchdown air time ----------------

def _td(first, last_air, other, cmd_x=0.2, amax=0.4, amin=0.15):
    f = torch.tensor([first], dtype=torch.bool)
    la = torch.tensor([last_air], dtype=torch.float32)
    o = torch.tensor([other], dtype=torch.bool)
    return R.feet_air_time_touchdown(f, la, o, _cmd(cmd_x), amax, amin).item()


def test_touchdown_pays_once_at_landing():
    # 0.3 s step: pays 0.3 - 0.15 = 0.15 on the landing step only
    assert _td([True, False], [0.3, 0.0], [True, True]) == pytest.approx(0.15)
    # the following steps: first_contact False (last_air_time stays 0.3 in the sensor) -> 0
    assert _td([False, False], [0.3, 0.0], [True, True]) == 0.0


def test_touchdown_short_flicker_pays_zero():
    for t in (0.07, 0.10, 0.13, 0.15):
        assert _td([True, False], [t, 0.0], [True, True]) == 0.0


def test_touchdown_capped_window():
    assert _td([True, False], [0.7, 0.0], [True, True]) == pytest.approx(0.25)
    assert _td([True, False], [0.4, 0.0], [True, True]) == pytest.approx(0.25)


def test_touchdown_held_foot_forever_pays_zero():
    # a foot held up never lands: first_contact is False on every step, whatever last_air_time says
    total = sum(_td([False, False], [0.0, 0.0], [True, False]) for _ in range(500))
    assert total == 0.0


def test_touchdown_needs_single_stance_and_command():
    assert _td([True, False], [0.3, 0.0], [False, True]) == 0.0      # other foot was not down
    assert _td([True, False], [0.3, 0.0], [True, True], cmd_x=0.0) == 0.0  # standing command
    assert _td([True, False], [0.3, 0.0], [True, True], cmd_x=0.04) == 0.0  # below the 0.05 gate
    assert _td([True, False], [0.3, 0.0], [True, True], cmd_x=0.06) == pytest.approx(0.15)


def test_touchdown_both_feet_same_step_and_batch():
    assert _td([True, True], [0.3, 0.5], [True, True]) == pytest.approx(0.15 + 0.25)
    f = torch.tensor([[True, False], [False, False], [False, True]])
    la = torch.tensor([[0.3, 0.0], [0.3, 0.0], [0.0, 0.2]])
    o = torch.ones(3, 2, dtype=torch.bool)
    c = torch.tensor([[0.2, 0, 0], [0.2, 0, 0], [0.0, 0, 0.5]])  # env 2: yaw-only command -> gated off
    r = R.feet_air_time_touchdown(f, la, o, c, 0.4, 0.15)
    assert r.shape == (3,)
    assert r.tolist() == pytest.approx([0.15, 0.0, 0.0])


# ---------------- swing-time gate ----------------

def test_swing_time_mask():
    t = torch.tensor([[0.0, 0.2], [0.5, 0.51], [3.0, 0.4]])
    m = R.swing_time_mask(t, 0.5)
    assert m.dtype == torch.bool
    assert m.tolist() == [[True, True], [True, False], [False, True]]


def test_swing_foot_height_old_behaviour_and_gate():
    clr = torch.tensor([[0.03, 0.0]])          # right foot 3 cm up, left grounded
    contact = torch.tensor([[False, True]])
    c = _cmd(0.2)
    old = R.swing_foot_height(clr, contact, c, 0.0, 0.02, 0.05)
    assert old.item() == pytest.approx(0.02)   # saturates at max_clearance
    # mask=None and all-True mask equal the old result
    ok = R.swing_foot_height(clr, contact, c, 0.0, 0.02, 0.05, torch.ones(1, 2, dtype=torch.bool))
    assert ok.item() == pytest.approx(0.02)
    # foot held up for 3 s: mask False -> pays 0, no matter the clearance
    held = R.swing_time_mask(torch.tensor([[3.0, 0.0]]), 0.5)
    assert R.swing_foot_height(clr, contact, c, 0.0, 0.02, 0.05, held).item() == 0.0
    # a normal 0.3 s swing still pays
    normal = R.swing_time_mask(torch.tensor([[0.3, 0.0]]), 0.5)
    assert R.swing_foot_height(clr, contact, c, 0.0, 0.02, 0.05, normal).item() == pytest.approx(0.02)
    # command gate still applies
    assert R.swing_foot_height(clr, contact, _cmd(0.0), 0.0, 0.02, 0.05, normal).item() == 0.0


def test_held_leg_total_reward_is_bounded():
    """Simulate holding the right leg up for 400 steps (0.02 s each): swing_foot_height pays only
    during the first 0.5 s (25 steps), and feet_air_time_touchdown pays nothing."""
    contact = torch.tensor([[False, True]])
    clr = torch.tensor([[0.02, 0.0]])
    c = _cmd(0.2)
    total_h, total_a = 0.0, 0.0
    for k in range(400):
        air = torch.tensor([[0.02 * (k + 1), 0.0]])
        total_h += R.swing_foot_height(clr, contact, c, 0.0, 0.02, 0.05, R.swing_time_mask(air, 0.5)).item()
        total_a += R.feet_air_time_touchdown(torch.zeros(1, 2, dtype=torch.bool), air,
                                             torch.tensor([[False, True]]), c, 0.4, 0.15).item()
    assert total_a == 0.0
    assert total_h == pytest.approx(0.02 * 25, abs=0.02 * 1.5)
