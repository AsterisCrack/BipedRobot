"""Exporting the gait clock of a gait-table run: a v2 table (cadence follows the command) becomes
a cadence_grid clock and bundle format v5; a constant cadence stays fixed_period (v3/v4), so
older runtimes keep working. torch/numpy only, no Isaac. Run from BipedRobot/:
    python -m pytest tests/test_cadence_export.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GVX, GVY, GWZ = np.array([-0.2, 0.0, 0.2]), np.array([-0.15, 0.0, 0.15]), np.array([-1.2, 0.0, 1.2])


def _run(tmp_path, freq=None, period=0.84):
    table = tmp_path / "table.npz"
    keys = dict(version=np.int32(2 if freq is not None else 1), grid_vx=GVX, grid_vy=GVY, grid_wz=GWZ,
                period=np.float32(period), style_digest=np.array("abc123"),
                phase_convention=np.array("phase 0 = right-foot touchdown"))
    if freq is not None:
        keys["freq"] = np.asarray(freq, dtype=np.float32)
    np.savez(table, **keys)
    (tmp_path / "config.yaml").write_text(
        f"train:\n  env_config:\n    reference:\n      npz_path: {table.as_posix()}\n      stand_threshold: 0.05\n")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    os.utime(table, (1, 1))           # table older than the checkpoint: no re-bake warning
    return str(tmp_path), str(ckpt)


def test_cadence_grid_clock_and_v5(tmp_path):
    from src.isaaclab import export_onnx as ex
    vx, vy, wz = np.meshgrid(GVX, GVY, GWZ, indexing="ij")
    freq = 0.5 + 2.0 * np.abs(vx) + 1.5 * np.abs(vy) + 0.2 * np.abs(wz)
    run, ckpt = _run(tmp_path, freq)
    gc = ex._load_gait_clock(run, None, ckpt, 5.0)
    assert gc["mode"] == "cadence_grid"
    assert gc["grid_vx"] == pytest.approx(list(GVX)) and gc["grid_wz"] == pytest.approx(list(GWZ))
    assert np.allclose(np.asarray(gc["freq"]), freq, atol=1e-6)
    assert gc["stand_threshold"] == 0.05 and gc["style_digest"] == "abc123"
    assert ex._format_version({"action_map": "residual"}, gc) == 5
    assert ex._format_version({"action_map": "centered"}, gc) == 5


def test_constant_cadence_stays_fixed_period(tmp_path):
    from src.isaaclab import export_onnx as ex
    run, ckpt = _run(tmp_path, np.full((3, 3, 3), 1.0 / 0.9))
    gc = ex._load_gait_clock(run, None, ckpt, 5.0)
    assert gc["mode"] == "fixed_period" and gc["period"] == pytest.approx(0.9, rel=1e-5)
    assert ex._format_version({"action_map": "residual"}, gc) == 4


def test_v1_table_is_fixed_period(tmp_path):
    from src.isaaclab import export_onnx as ex
    run, ckpt = _run(tmp_path, None, period=0.84)
    gc = ex._load_gait_clock(run, None, ckpt, 5.0)
    assert gc["mode"] == "fixed_period" and gc["period"] == pytest.approx(0.84, rel=1e-5)


def test_implausible_cadence_is_refused(tmp_path):
    from src.isaaclab import export_onnx as ex
    freq = np.full((3, 3, 3), 1.0)
    freq[0, 0, 0] = 9.0
    run, ckpt = _run(tmp_path, freq)
    with pytest.raises(RuntimeError, match="0.2-5 Hz"):
        ex._load_gait_clock(run, None, ckpt, 5.0)
