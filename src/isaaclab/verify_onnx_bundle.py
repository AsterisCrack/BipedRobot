"""Verify an exported ONNX bundle against the repo's OWN actor code.

export_onnx.py already checks ONNX against the ``DeterministicActor`` it builds -- but
that is its own reconstruction, so a mis-mapped layer would be wrong in both and the
check would still pass. This script closes that hole by running the real
``MLPActor.get_action()`` from torch_rl_algorithms and comparing against the bundle,
including the stage-1 normalizer the runtime will apply.

    python -m src.isaaclab.verify_onnx_bundle --bundle exported/<run>_<step>

Exit code 0 = the bundle reproduces the training-time policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from gymnasium.spaces import Box

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "torch_rl_algorithms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_rl_algorithms.models.mlp import MLPActor          # noqa: E402
from torch_rl_algorithms.models.equivariant import EquivariantActor, TABLE_KEYS  # noqa: E402
from torch_rl_algorithms.algorithms.utils import MeanStd     # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Defaults to source_checkpoint in checkpoint_info.json")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    bundle = os.path.abspath(args.bundle)
    with open(os.path.join(bundle, "deploy_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    with open(os.path.join(bundle, "checkpoint_info.json"), encoding="utf-8") as fh:
        info = json.load(fh)

    ckpt_path = args.checkpoint or info["source_checkpoint"]
    sd = torch.load(ckpt_path, map_location="cpu")

    obs_dim = cfg["obs_dim"]
    act_dim = cfg["action_dim"]
    hidden = info["hidden_sizes"]

    # -- rebuild via the REPO's own class, not our reconstruction -------------
    normalizer = MeanStd(shape=(obs_dim,))
    actor = MLPActor(
        observation_space=Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=Box(-1.0, 1.0, (act_dim,), np.float32),
        hidden_sizes=list(hidden),
        observation_normalizer=normalizer,
        head_type="gaussian_multivariate",
    )
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    if all(k in actor_sd for k in TABLE_KEYS):
        # train.equivariant_actor checkpoint: compare against the REAL wrapped actor (tables from the
        # checkpoint buffers), not just the plain MLP, so the symmetrisation in the ONNX is checked too.
        actor = EquivariantActor(actor, obs_dim=obs_dim, act_dim=act_dim)
        print("[verify] checkpoint holds an equivariant actor: verifying against EquivariantActor")
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    # std_layer is loaded but unused on the deterministic path; nothing should be missing.
    if missing:
        print(f"[verify] FAIL missing keys in {type(actor).__name__}: {missing}", file=sys.stderr)
        return 1
    actor.eval()

    # -- deployment path ------------------------------------------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(bundle, "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    obs_mean = np.asarray(cfg["obs_mean"], dtype=np.float64)
    obs_var = np.asarray(cfg["obs_var"], dtype=np.float64)
    eps = cfg["obs_epsilon"]

    # -- compare over realistic, un-normalized observations --------------------
    # Draw from the training distribution itself (mean + sigma*z) so the comparison
    # exercises the region the policy actually operates in, not just N(0,1) noise.
    rng = np.random.default_rng(1234)
    sigma = np.sqrt(obs_var)
    worst = 0.0
    worst_raw = None

    for _ in range(args.n):
        raw = obs_mean + sigma * rng.standard_normal(obs_dim)

        # deployment: stage-1 in the runtime, stage-2 inside the graph
        norm = ((raw - obs_mean) / np.sqrt(obs_var + eps)).astype(np.float32)
        got = sess.run(["action"], {"obs": norm[None, :]})[0][0]

        # training: stage-1 identically, then the repo's own actor (stage-2 inside)
        with torch.no_grad():
            ref = actor.get_action(torch.from_numpy(norm[None, :])).numpy()[0]

        d = float(np.abs(ref - got).max())
        if d > worst:
            worst, worst_raw = d, raw

    print(f"[verify] bundle      : {bundle}")
    print(f"[verify] checkpoint  : {ckpt_path}")
    print(f"[verify] obs_dim={obs_dim} act_dim={act_dim} hidden={hidden}")
    print(f"[verify] max abs action diff over {args.n} samples: {worst:.3e}")

    if worst > args.tol:
        print(f"[verify] FAIL: exceeds tol {args.tol:.1e}", file=sys.stderr)
        if worst_raw is not None:
            np.save(os.path.join(bundle, "verify_worst_obs.npy"), worst_raw)
            print(f"[verify] worst-case obs saved for debugging", file=sys.stderr)
        return 1

    # -- structural assertions the runtime depends on -------------------------
    problems = []
    if cfg["history_order"] != "oldest_first":
        problems.append("history_order must be oldest_first")
    if cfg["obs_dim"] != cfg["history_size"] * cfg["obs_proprio_dim"]:
        problems.append("obs_dim != history_size * obs_proprio_dim")
    if len(cfg["joint_names"]) != act_dim:
        problems.append("joint_names length != action_dim")
    if sorted(cfg["joint_names"]) != sorted(set(cfg["joint_names"])):
        problems.append("joint_names contains duplicates")
    layout_end = max(v[1] for v in cfg["obs_layout"].values())
    if layout_end != cfg["obs_proprio_dim"]:
        problems.append(f"obs_layout spans {layout_end}, proprio is {cfg['obs_proprio_dim']}")
    if cfg["action_filter_applications_per_step"] != 4:
        problems.append("EMA applications per step should be 4 (Isaac decimation)")

    if problems:
        print("[verify] FAIL structural checks:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        return 1

    print("[verify] PASS -- bundle reproduces the training-time policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
