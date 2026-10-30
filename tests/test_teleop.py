"""utils/teleop.py: manual-command mapping, speed level, push sampling.

CPU only, no Isaac Sim / GLFW. Run from BipedRobot/:  python -m pytest tests/test_teleop.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.teleop import (DEFAULT_PUSH_RANGE, SpeedLevel, arc_points, axes_to_command,  # noqa: E402
                          deadzone2, push_range, sample_push)

R = {"lin_vel_x": (-0.1, 0.2), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-0.8, 0.8)}


def test_zero_in_zero_out():
    assert np.array_equal(axes_to_command(0, 0, 0, R), np.zeros(3))


@pytest.mark.parametrize("axes,want", [
    ((1, 0, 0), (0.2, 0, 0)), ((-1, 0, 0), (-0.1, 0, 0)),      # forward / back hit their own edge
    ((0, 1, 0), (0, 0.15, 0)), ((0, -1, 0), (0, -0.15, 0)),    # + = left
    ((0, 0, 1), (0, 0, 0.8)), ((0, 0, -1), (0, 0, -0.8)),      # + = counter-clockwise
])
def test_full_deflection_hits_range_edges(axes, want):
    assert np.allclose(axes_to_command(*axes, R), want)


def test_level_scales_and_input_is_clipped():
    assert np.allclose(axes_to_command(1, 0, -1, R, level=0.5), (0.1, 0, -0.4))
    assert np.allclose(axes_to_command(2, 0, 0, R), (0.2, 0, 0))


def test_ellipse_limit_applied():
    box = axes_to_command(1, 1, 0, R, "box")
    ell = axes_to_command(1, 1, 0, R, "ellipse_xy")
    assert np.allclose(box[:2], (0.2, 0.15))
    assert math.isclose((ell[0] / 0.2) ** 2 + (ell[1] / 0.15) ** 2, 1.0)
    assert math.isclose(ell[0] / ell[1], 0.2 / 0.15)            # direction kept


def test_speed_level_clamps_and_snaps():
    lv = SpeedLevel()
    assert lv.pct == 50
    for _ in range(10):
        lv.step(+1)
    assert lv.value == 1.0
    for _ in range(20):
        lv.step(-1)
    assert lv.value == 0.1 and lv.pct == 10
    for _ in range(9):
        lv.step(+1)
    assert lv.value == 1.0                                      # no float drift after many steps


def test_deadzone2():
    assert deadzone2(0.05, 0.05) == (0.0, 0.0)
    x, y = deadzone2(1.0, 0.0)
    assert math.isclose(x, 1.0) and y == 0.0


def test_sample_push_on_ellipse_edge():
    rng = np.random.default_rng(0)
    vr = {"x": (-0.4, 0.4), "y": (-0.2, 0.2), "yaw": (-0.6, 0.6)}
    for _ in range(50):
        dv = sample_push(vr, rng)
        assert math.isclose((dv[0] / 0.4) ** 2 + (dv[1] / 0.2) ** 2, 1.0)
        assert -0.6 <= dv[2] <= 0.6
    dv2 = sample_push(vr, np.random.default_rng(1), scale=2.0)
    assert math.isclose((dv2[0] / 0.8) ** 2 + (dv2[1] / 0.4) ** 2, 1.0)


class _Cfg:
    def __init__(self, events):
        self.events = events


def test_push_range_override_and_default():
    assert push_range(None) == DEFAULT_PUSH_RANGE
    assert push_range(_Cfg(None)) == DEFAULT_PUSH_RANGE
    assert push_range(_Cfg({"push_robot": {"interval_range_s": [1, 2]}})) == DEFAULT_PUSH_RANGE
    got = push_range(_Cfg({"push_robot": {"params": {"velocity_range": {"x": [-0.5, 0.5]}}}}))
    assert got["x"] == (-0.5, 0.5) and got["y"] == DEFAULT_PUSH_RANGE["y"]


def test_arc_points_direction():
    p = arc_points(0.0, math.pi / 2, n=2)
    assert np.allclose(p[0], (1, 0)) and np.allclose(p[-1], (0, 1))   # + span = counter-clockwise


# --------------------------------------------------------------------------- Isaac command sampler
import torch  # noqa: E402

from tests.test_command_limit import _gen, _load_sampler  # noqa: E402


@pytest.fixture(scope="module")
def sampler_mod():
    return _load_sampler()


def test_manual_overrides_sampling_heading_and_reset(sampler_mod):
    g = _gen(sampler_mod, "config.yaml", 4)
    alias = g.commands
    quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1)
    g.set_manual((0.1, -0.05, 0.3))
    g.time_left[:] = 0.0                       # forces a resample inside compute()
    g.compute(quat)
    assert torch.allclose(g.commands, torch.tensor([0.1, -0.05, 0.3]).expand(4, 3))
    g.reset(torch.arange(4))
    g.apply_heading(quat)                      # the env's _resample_commands path
    assert torch.allclose(g.commands, torch.tensor([0.1, -0.05, 0.3]).expand(4, 3))
    assert g.commands is alias                 # written in place: env.commands still aliases it
    g.clear_manual()
    g.reset(torch.arange(4))
    assert not torch.allclose(g.commands, torch.tensor([0.1, -0.05, 0.3]).expand(4, 3))


def test_arrow_and_arc_geometry(sampler_mod):
    from utils.teleop import ARC_SEGMENTS, ARROW_LEN_AT_MAX
    g = _gen(sampler_mod, "config.yaml", 3)
    v_max = max(abs(v) for k in ("lin_vel_x", "lin_vel_y") for v in g.ranges[k])
    pos0 = torch.zeros(3, 3)
    v = torch.tensor([[v_max, 0.0], [0.0, 0.5 * v_max], [0.0, 0.0]])   # full, half, stopped
    pos, quat, sc, idx = g._arrow_instances(pos0, v, 0.3, dot=True)
    tip = pos[:, 1, 0] + g._HEAD_LEN * sc[:, 1, 0] / 2                  # head centre + half head
    assert torch.isclose(tip[0], torch.tensor(ARROW_LEN_AT_MAX))         # length = k * |v|, measured
    assert torch.isclose(pos[1, 1, 1] + g._HEAD_LEN * sc[1, 1, 0] / 2, torch.tensor(ARROW_LEN_AT_MAX / 2))
    assert idx[2, 0] == 2 and idx[0, 0] == 0                             # stopped -> dot prototype
    assert sc[2, 1].max() < 1e-3                                         # ... and no head
    # same scale for the measured arrow (dot=False just hides at rest)
    _, _, sc_m, _ = g._arrow_instances(pos0, v, 0.34, dot=False)
    assert torch.allclose(sc_m[:2], sc[:2]) and sc_m[2].max() < 1e-3

    yaw = torch.zeros(3)
    wz = torch.tensor([1.0, -1.0, 0.0])
    pos, quat, sc, idx = g._arc_instances(pos0, yaw, wz, 0.2)
    assert pos.shape == (3, ARC_SEGMENTS, 3)
    assert pos[0, 0, 1] > 0 and pos[1, 0, 1] < 0                          # +wz sweeps counter-clockwise
    assert (idx[:, -1] == 1).all() and (idx[:, :-1] == 3).all()
    assert sc[2].max() < 1e-3                                            # no yaw command: no arc


def test_visualize_feeds_one_instancer_per_colour(sampler_mod):
    from utils.teleop import ARC_SEGMENTS
    calls = []

    class FakeMarker:
        def visualize(self, **kw):
            calls.append(kw)

    n = 2
    g = _gen(sampler_mod, "config.yaml", n)
    g.set_manual((0.1, 0.0, 0.5))
    g.apply_heading(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
    g.visualize((FakeMarker(), FakeMarker()), torch.zeros(n, 3), torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                torch.zeros(n, 3), torch.zeros(n, 3))
    m = n * (2 + ARC_SEGMENTS)
    for kw in calls:
        assert kw["translations"].shape == (m, 3) and kw["orientations"].shape == (m, 4)
        assert kw["scales"].shape == (m, 3) and kw["marker_indices"].shape == (m,)
    assert len(calls) == 2
