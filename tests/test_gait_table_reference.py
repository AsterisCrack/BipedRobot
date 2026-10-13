"""GaitTableReference sampler. numpy + torch only. Run from BipedRobot/:  python -m pytest tests/test_gait_table_reference.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.motion_reference import GaitTableReference  # noqa: E402

REAL_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "robot_motion_reference", "processed", "gait_table.npz")
DEV = torch.device("cpu")
GRID_VX, GRID_VY, GRID_WZ = np.array([-0.2, 0.0, 0.3]), np.array([-0.1, 0.1]), np.array([-1.0, 0.0, 0.5, 1.0])
F = 8


def _synthetic(tmp_path):
    """D = 5: [lin(vx,vy,wz), frame index, joint a, joint b, joint c]. lin is exactly trilinear."""
    d = np.zeros((len(GRID_VX), len(GRID_VY), len(GRID_WZ), F, 6), dtype=np.float32)
    for i, vx in enumerate(GRID_VX):
        for j, vy in enumerate(GRID_VY):
            for k, wz in enumerate(GRID_WZ):
                d[i, j, k, :, 0] = 2 * vx + 3 * vy - wz + 0.5
                d[i, j, k, :, 1] = np.arange(F)
                d[i, j, k, :, 2:5] = np.array([1.0, 2.0, 3.0])
    d[..., 5] = 7.0
    stand = np.full(6, -1.0, dtype=np.float32)
    path = tmp_path / "t.npz"
    np.savez(path, version=np.int32(1), data=d, stand=stand,
             grid_vx=GRID_VX.astype(np.float32), grid_vy=GRID_VY.astype(np.float32),
             grid_wz=GRID_WZ.astype(np.float32),
             field_names=np.array(["lin", "frame", "joint_pos", "other"]),
             field_widths=np.array([1, 1, 3, 1], dtype=np.int32),
             joint_names=np.array(["a", "b", "c"]), period=np.float32(0.8), dt=np.float32(0.1),
             n_frames=np.int32(F), phase_convention=np.array("p"), style_digest=np.array("x"))
    return str(path)


@pytest.fixture()
def ref(tmp_path):
    return GaitTableReference(_synthetic(tmp_path), DEV)


def _s(ref, cmd, phase=0.0, standing=False):
    c = torch.tensor([cmd], dtype=torch.float32)
    return ref.sample(c, torch.tensor([phase]), torch.tensor([standing]))


def test_trilinear_exact_inside_grid(ref):
    rng = np.random.default_rng(0)
    for _ in range(50):
        cmd = (rng.uniform(-0.2, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(-1.0, 1.0))
        got = ref.field(_s(ref, cmd, 0.25), "lin").item()
        assert got == pytest.approx(2 * cmd[0] + 3 * cmd[1] - cmd[2] + 0.5, abs=1e-5)


def test_phase_is_linear_between_frames(ref):
    assert ref.field(_s(ref, (0, 0, 0), (3 + 0.25) / F), "frame").item() == pytest.approx(3.25, abs=1e-5)


def test_phase_wraps(ref):
    assert ref.field(_s(ref, (0, 0, 0), 1.0), "frame").item() == pytest.approx(0.0, abs=1e-5)
    assert ref.field(_s(ref, (0, 0, 0), (F - 0.5) / F), "frame").item() == pytest.approx((F - 1) / 2, abs=1e-5)
    assert ref.field(_s(ref, (0, 0, 0), 1.25), "frame").item() == pytest.approx(
        ref.field(_s(ref, (0, 0, 0), 0.25), "frame").item(), abs=1e-5)


def test_clamps_to_grid(ref):
    far = ref.field(_s(ref, (5.0, -5.0, 9.0), 0.0), "lin").item()
    edge = ref.field(_s(ref, (0.3, -0.1, 1.0), 0.0), "lin").item()
    assert far == pytest.approx(edge, abs=1e-5)


def test_standing_returns_stand_frame(ref):
    out = _s(ref, (0.1, 0.0, 0.0), 0.3, standing=True)
    assert torch.allclose(out, torch.full_like(out, -1.0))


def test_standing_mask_is_per_env(ref):
    c = torch.zeros(2, 3)
    out = ref.sample(c, torch.tensor([0.0, 0.0]), torch.tensor([True, False]))
    assert out[0, 0].item() == -1.0 and out[1, 0].item() != -1.0


def test_joint_binding_reorders(ref):
    ref.bind_to_robot(["c", "a"])
    out = ref.joint_field(_s(ref, (0, 0, 0), 0.0), "joint_pos")
    assert out.tolist() == [[3.0, 1.0]]


def test_bind_rejects_unknown_joint(ref):
    with pytest.raises(ValueError):
        ref.bind_to_robot(["a", "nope"])


def test_joint_field_requires_bind(ref):
    with pytest.raises(RuntimeError):
        ref.joint_field(_s(ref, (0, 0, 0)), "joint_pos")


def test_missing_file_message():
    with pytest.raises(FileNotFoundError, match="bake"):
        GaitTableReference("/no/such/table.npz", DEV)


@pytest.mark.skipif(not os.path.exists(REAL_TABLE), reason="baked table not present")
def test_real_table_layout_and_symmetry():
    ref = GaitTableReference(REAL_TABLE, DEV)
    assert ref.dim == ref.table.shape[1] == sum(s.stop - s.start for s in ref.slices.values())
    assert ref.period == pytest.approx(ref.n_frames * ref.dt, rel=1e-3)
    for g in ref.grid[1:]:
        assert torch.allclose(g, -g.flip(0), atol=1e-6), "vy/wz grids must be symmetric about 0"

    names = ref.joint_names
    swap = lambda n: ("r_" if n.startswith("l_") else "l_") + n[2:]
    perm = torch.as_tensor([names.index(swap(n)) for n in names])
    data = ref.table.reshape(*ref.shape, ref.n_frames, ref.dim)
    for name in ("joint_pos", "joint_vel"):
        sl = ref.slices[name]
        a = data[..., sl]
        # reference(vx, -vy, -wz, phase + 0.5) is the left/right mirror of reference(vx, vy, wz, phase)
        b = torch.roll(a.flip(1).flip(2), -ref.n_frames // 2, dims=3)[..., perm]
        assert torch.allclose(a, b, atol=1e-5), f"{name} is not left/right symmetric"
