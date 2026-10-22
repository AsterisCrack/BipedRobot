"""Symmetric stage-1 observation statistics (E15) -- the real BaseIsaacLabWrapper._process_obs with
the real BipedEnv mirror transform. CPU only; Isaac is NOT imported: `isaaclab_tasks` / `envs.isaaclab`
are stubbed just while src/isaaclab/common.py is loaded, and BipedEnv._build_mirror_transform /
mirror_obs are lifted out of biped_env.py with ast and run on a stand-in `self` (V2 joint layout).

Run from BipedRobot/:  python -m pytest tests/test_symmetric_obs_stats.py -q
(torch_rl_algorithms must be importable as a package, as it is for train.py.)
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOL = 1e-5
HIST = 5
PROPRIO, CRITIC = 50, 60
V2_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
V2_SIGNS = [1.0] * 12


def _lift(*names):
    src = open(os.path.join(ROOT, "envs", "isaaclab", "biped_env.py"), encoding="utf-8").read()
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in fns} == set(names)
    ns = {"torch": torch}
    exec(compile(ast.Module(fns, []), "biped_env_methods", "exec"), ns)
    return [ns[n] for n in names]


class FakeBipedEnv:
    """Stand-in carrying the REAL _build_mirror_transform / mirror_obs and the 50/250/60 layout."""
    device = torch.device("cpu")
    num_envs = 64
    obs_proprio_dim = PROPRIO

    def __init__(self, signs=V2_SIGNS):
        self.cfg = SimpleNamespace(observation_space_dim={"policy": PROPRIO * HIST, "critic": CRITIC},
                                   mirror_joint_perm=V2_PERM, mirror_joint_signs=signs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.num_envs, PROPRIO * HIST))
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_envs, 12))
        self._build_mirror_transform()

    unwrapped = property(lambda self: self)


_build, _mirror = _lift("_build_mirror_transform", "mirror_obs")
FakeBipedEnv._build_mirror_transform = _build
FakeBipedEnv.mirror_obs = _mirror


class VideoLike:
    """gymnasium RecordVideo-like wrapper: forwards .unwrapped but NOT mirror_obs."""

    def __init__(self, env):
        self.env = env
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.unwrapped = env.unwrapped


@pytest.fixture(scope="module")
def common():
    """Load src/isaaclab/common.py with the Isaac-only imports stubbed for the duration of the load."""
    mp = pytest.MonkeyPatch()
    try:
        tasks = type(sys)("isaaclab_tasks")
        tasks_utils = type(sys)("isaaclab_tasks.utils")
        tasks_utils.parse_env_cfg = lambda *a, **k: None
        tasks.utils = tasks_utils
        mp.setitem(sys.modules, "isaaclab_tasks", tasks)
        mp.setitem(sys.modules, "isaaclab_tasks.utils", tasks_utils)
        mp.setitem(sys.modules, "envs.isaaclab", type(sys)("envs.isaaclab"))
        spec = importlib.util.spec_from_file_location(
            "_common_under_test", os.path.join(ROOT, "src", "isaaclab", "common.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        mp.undo()
    return mod


def _cfg(normalize=True):
    return SimpleNamespace(train=SimpleNamespace(normalize_obs=normalize))


def _raw_stream(n_steps, seed=0):
    """Asymmetric raw obs: per-dim offsets/scales differ between mirror partners (as foot forces do)."""
    g = torch.Generator().manual_seed(seed)
    pa, pc = torch.linspace(-1, 2, PROPRIO * HIST), torch.linspace(-3, 4, CRITIC)
    sa, sc = torch.linspace(0.5, 3.0, PROPRIO * HIST), torch.linspace(0.3, 5.0, CRITIC)
    for _ in range(n_steps):
        yield {"policy": torch.randn(64, PROPRIO * HIST, generator=g) * sa + pa,
               "critic": torch.randn(64, CRITIC, generator=g) * sc + pc}


def _pairs(env):
    return {"actor": (env._mirror_obs_perm, env._mirror_obs_signs, PROPRIO),
            "critic": (env._mirror_critic_obs_perm, env._mirror_critic_obs_signs, CRITIC)}


def _expand(perm, sign, dim, total):
    """Full-width perm/sign for a history stack (chunk-wise mirror, as mirror_obs does)."""
    k = total // dim
    return (torch.cat([perm + i * dim for i in range(k)]), sign.repeat(k))


def test_real_mirror_is_an_involution():
    env = FakeBipedEnv()
    for key, (perm, sign, _) in _pairs(env).items():
        assert torch.equal(perm[perm], torch.arange(len(perm))), key
        assert torch.equal(sign, sign[perm]), key          # s_i == s_p(i)  <=>  M(M(x)) = x


def test_flag_on_stats_satisfy_pair_relations_and_normalise_commutes(common):
    env = FakeBipedEnv()
    w = common.BaseIsaacLabWrapper(env, _cfg(), symmetric_obs_stats=True)
    for raw in _raw_stream(30):
        w._process_obs(raw)
    for key, (perm, sign, dim) in _pairs(env).items():
        rms = w.obs_scalers[key]
        p, s = _expand(perm, sign, dim, rms.mean.numel())
        assert torch.allclose(rms.mean, s * rms.mean[p], atol=TOL, rtol=0), key
        assert torch.allclose(rms.var, rms.var[p], atol=TOL, rtol=0), key
    # critic foot-force slots 54:60: R/L variances equal, hence mirrored samples correctly scaled
    v = w.obs_scalers["critic"].var
    assert torch.allclose(v[54:57], v[57:60], atol=TOL)
    # (b) normalise-then-mirror == mirror-then-normalise, through the real env mirror
    raw = next(_raw_stream(1, seed=9))
    a = env.mirror_obs({"actor": w.obs_scalers["actor"].normalize(raw["policy"]),
                        "critic": w.obs_scalers["critic"].normalize(raw["critic"])})
    b = env.mirror_obs({"actor": raw["policy"], "critic": raw["critic"]})
    assert torch.allclose(a["actor"], w.obs_scalers["actor"].normalize(b["actor"]), atol=1e-4)
    assert torch.allclose(a["critic"], w.obs_scalers["critic"].normalize(b["critic"]), atol=1e-4)


def test_flag_on_returns_the_same_real_normalised_obs_shape_and_keys(common):
    env = FakeBipedEnv()
    w = common.BaseIsaacLabWrapper(env, _cfg(), symmetric_obs_stats=True)
    raw = next(_raw_stream(1))
    out = w._process_obs(dict(raw))
    assert set(out) == {"actor", "critic"}
    assert out["actor"].shape == raw["policy"].shape and out["critic"].shape == raw["critic"].shape
    assert torch.allclose(out["critic"], w.obs_scalers["critic"].normalize(raw["critic"]))
    assert w.obs_scalers["critic"].count == pytest.approx(1e-4 + 2 * 64)   # real + mirror, equal counts


def test_flag_off_reproduces_old_stats(common):
    env = FakeBipedEnv()
    w_off = common.BaseIsaacLabWrapper(env, _cfg())                       # default: off
    ref = common.RunningMeanStd(shape=(CRITIC,))
    for raw in _raw_stream(10, seed=4):
        w_off._process_obs(dict(raw))
        ref.update(raw["critic"])
    rms = w_off.obs_scalers["critic"]
    assert torch.equal(rms.mean, ref.mean) and torch.equal(rms.var, ref.var) and rms.count == ref.count
    perm, sign, _ = _pairs(env)["critic"]
    assert float((rms.var - rms.var[perm]).abs().max()) > 0.5             # real-only stats are asymmetric


def test_eval_mode_freezes_stats(common):
    env = FakeBipedEnv()
    w = common.BaseIsaacLabWrapper(env, _cfg(), symmetric_obs_stats=True)
    for raw in _raw_stream(3):
        w._process_obs(raw)
    w.set_eval_mode()
    before = {k: (v.mean.clone(), v.var.clone(), v.count) for k, v in w.obs_scalers.items()}
    for raw in _raw_stream(3, seed=5):
        w._process_obs(raw)
    for k, v in w.obs_scalers.items():
        assert torch.equal(v.mean, before[k][0]) and torch.equal(v.var, before[k][1]) and v.count == before[k][2]


def test_video_wrapper_reaches_unwrapped_mirror(common):
    env = FakeBipedEnv()
    w = common.BaseIsaacLabWrapper(VideoLike(env), _cfg(), symmetric_obs_stats=True)
    assert w._mirror_raw_obs is not None
    w._process_obs(next(_raw_stream(1)))
    assert w.obs_scalers["critic"].count == pytest.approx(1e-4 + 2 * 64)


def test_missing_mirror_warns_and_falls_back(common, capsys):
    env = FakeBipedEnv()
    plain = SimpleNamespace(device=env.device, num_envs=env.num_envs, cfg=env.cfg,
                            observation_space=env.observation_space, action_space=env.action_space)
    plain.unwrapped = plain
    w = common.BaseIsaacLabWrapper(plain, _cfg(), symmetric_obs_stats=True)
    assert w._mirror_raw_obs is None and "[WARN] symmetric_obs_stats" in capsys.readouterr().out
    w._process_obs(next(_raw_stream(1)))
    assert w.obs_scalers["critic"].count == pytest.approx(1e-4 + 64)


def test_play_and_probe_paths_do_not_enable_it():
    for rel in ("play.py", "jitter_probe.py", "export_onnx.py"):
        src = open(os.path.join(ROOT, "src", "isaaclab", rel), encoding="utf-8").read()
        assert "symmetric_obs_stats" not in src, rel
    train = open(os.path.join(ROOT, "src", "isaaclab", "train.py"), encoding="utf-8").read()
    assert 'symmetric_obs_stats=bool(getattr(config.train, "symmetry_augmentation", False))' in train
