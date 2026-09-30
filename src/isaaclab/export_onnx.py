"""Export a trained SAC actor to a self-contained ONNX deployment bundle.

The output directory can be copied to the Jetson and consumed with nothing but
``onnxruntime`` + ``numpy`` -- no torch, no Isaac Lab, no import from this repo.

    python -m src.isaaclab.export_onnx --checkpoint checkpoints/<run>/step_<N>.pt

Produces::

    <out>/policy.onnx           [1,250] float32 -> [1,12] float32
    <out>/deploy_config.json    everything the runtime shim needs
    <out>/checkpoint_info.json  provenance

WHY NOT src/isaaclab/export.py
------------------------------
That script hardcodes ``obs_proprio_dim = 52`` (the real value is 50), so
``load_state_dict`` raises a size mismatch against any V2 checkpoint -- it has never
run successfully. It also emits TorchScript, which would need libtorch on the robot.

This script instead rebuilds the actor *from the checkpoint's own tensor shapes* and
never consults the config for dimensions, so it cannot drift out of sync.

THE TWO NORMALIZERS
-------------------
Easiest thing to get silently wrong. The observation passes through two trained
normalizers in series:

  stage 1 (outer)  env wrapper, saved as <ckpt>_obs_scalers.pt
                   (x - mean) / sqrt(var + 1e-4)
                   -> lives OUTSIDE the graph, shipped in deploy_config.json

  stage 2 (inner)  actor.torso.observation_normalizer, a MeanStd module
                   (x - _mean) / _std
                   -> nn.Parameter data, captured INTO the ONNX automatically

Both are non-identity. Applying only one produces a policy that looks plausible and
walks straight into the floor.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Canonical Isaac joint order. The runtime permutes by NAME against this list -- the
# Jetson orders joints per-leg (all left, then all right) while Isaac interleaves L/R,
# so a positional copy would silently scramble the observation.
JOINT_NAMES = [
    "l_hip_yaw", "r_hip_yaw",
    "l_hip_roll_joint", "r_hip_roll_joint",
    "l_hip_pitch_joint", "r_hip_pitch_joint",
    "l_knee_joint", "r_knee_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
]

# Named slices of the 50-dim proprio frame, mirroring biped_env._get_observations.
# Shipped so the runtime can assert its own layout rather than assume it.
OBS_LAYOUT = {
    "imu_lin_acc_b": [0, 3],       # body-frame SPECIFIC FORCE (includes +g)
    "imu_ang_vel_b": [3, 6],       # body-frame gyro, rad/s
    "projected_gravity_b": [6, 9], # UNIT gravity vector in body frame
    "commands": [9, 12],           # [v_x, v_y, w_z]
    "joint_pos_rel": [12, 24],     # joint_pos - default_joint_pos (defaults are all 0)
    "joint_vel": [24, 36],         # rad/s
    "previous_actions": [36, 48],  # last clipped action, [-1, 1]
    "gait_phase": [48, 50],        # [sin(phi), cos(phi)]
}


# ---------------------------------------------------------------------------
# Source-of-truth extraction (AST, so we never import Isaac Lab)
# ---------------------------------------------------------------------------

def _literal_from_module(path: str, name: str):
    """Pull a module-level literal assignment out of a .py file without importing it.

    biped_robot.py and biped_env_cfg.py both import isaaclab at module scope, which
    would force this exporter to run inside an Isaac install. Parsing the literal
    keeps the values in sync with their source without that dependency.
    """
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name!r} not found as a literal assignment in {path}")


def _class_attr_from_module(path: str, class_name: str, attr: str):
    """Same idea, for a literal attribute inside a class body."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            else:
                continue
            for t in targets:
                if isinstance(t, ast.Name) and t.id == attr:
                    return ast.literal_eval(stmt.value)
    raise KeyError(f"{class_name}.{attr} not found in {path}")


def _gather_env_constants() -> dict:
    """Read the deployment-relevant constants from their source files."""
    robot_py = os.path.join(_PROJECT_ROOT, "envs/assets/robotV2/biped_robot.py")
    cfg_py = os.path.join(_PROJECT_ROOT, "envs/isaaclab/biped_env_cfg.py")

    limits_deg = _literal_from_module(robot_py, "_JOINT_LIMITS_DEG")
    if len(limits_deg) != 12:
        raise ValueError(f"expected 12 joint limit rows, got {len(limits_deg)}")

    decimation = _class_attr_from_module(cfg_py, "BipedEnvCfg", "decimation")
    alpha = _class_attr_from_module(cfg_py, "BipedEnvCfg", "action_filter_alpha")
    gait_freq = _class_attr_from_module(cfg_py, "BipedEnvCfg", "gait_clock_base_freq")

    return {
        "joint_limits_min": [float(np.radians(lo)) for lo, _ in limits_deg],
        "joint_limits_max": [float(np.radians(hi)) for _, hi in limits_deg],
        "joint_limits_deg": limits_deg,
        "decimation": int(decimation),
        "action_filter_alpha": float(alpha),
        "gait_clock_freq": float(gait_freq),
        # sim.dt is buried in a SimulationCfg(...) call rather than a bare literal;
        # it has been 1/200 for the life of V2 and is asserted against step_dt below.
        "sim_dt": 1.0 / 200.0,
    }


# ---------------------------------------------------------------------------
# Actor reconstruction, driven purely by state_dict shapes
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """Exactly the deterministic inference path, and nothing else.

    Isaac's MLPActor.forward builds a SquashedMultivariateNormalDiag and the caller
    takes ``.loc``, which is ``tanh(distribution.mean)``. Tracing that whole
    construction would drag in the std head (Softplus + clamp) and the torch
    distribution machinery for no benefit, so we emit the collapsed form:

        tanh( mean_layer( ELU_MLP( (x - mean) / std ) ) )

    Note tanh-OF-the-mean, not mean-of-the-tanh -- these differ and only the former
    matches what the policy did during training.
    """

    def __init__(self, hidden_sizes: list[int], in_dim: int, act_dim: int):
        super().__init__()
        # Stage-2 normalizer, kept as buffers so torch.onnx.export folds them in.
        self.register_buffer("norm_mean", torch.zeros(in_dim))
        self.register_buffer("norm_std", torch.ones(in_dim))

        layers: list[nn.Module] = []
        sizes = [in_dim] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ELU()]
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.norm_mean) / self.norm_std
        return torch.tanh(self.mean_layer(self.net(x)))


def build_actor_from_state_dict(sd: dict) -> tuple[DeterministicActor, int, int, list[int]]:
    """Rebuild the actor using only tensor shapes from the checkpoint.

    Deriving dims from the config is what broke the old exporter (it assumed a
    52-dim proprio frame). The checkpoint is authoritative.
    """
    # Torso Linears live at actor.torso.net.{0,2,4}.weight (odd indices are ELU).
    linear_ids = sorted(
        int(k.split(".")[3])
        for k in sd
        if k.startswith("actor.torso.net.") and k.endswith(".weight")
    )
    if not linear_ids:
        raise KeyError("no actor.torso.net.*.weight tensors in checkpoint")

    hidden = [int(sd[f"actor.torso.net.{i}.weight"].shape[0]) for i in linear_ids]
    in_dim = int(sd[f"actor.torso.net.{linear_ids[0]}.weight"].shape[1])
    act_dim = int(sd["actor.mean_layer.0.weight"].shape[0])

    model = DeterministicActor(hidden, in_dim, act_dim)

    with torch.no_grad():
        model.norm_mean.copy_(sd["actor.torso.observation_normalizer._mean"].flatten())
        model.norm_std.copy_(sd["actor.torso.observation_normalizer._std"].flatten())
        for dst, src in enumerate(linear_ids):
            model.net[dst * 2].weight.copy_(sd[f"actor.torso.net.{src}.weight"])
            model.net[dst * 2].bias.copy_(sd[f"actor.torso.net.{src}.bias"])
        model.mean_layer.weight.copy_(sd["actor.mean_layer.0.weight"])
        model.mean_layer.bias.copy_(sd["actor.mean_layer.0.bias"])

    model.eval()
    return model, in_dim, act_dim, hidden


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_history_size(ckpt_dir: str, explicit: str | None) -> int:
    """history_size from the run's own config.yaml, falling back to the repo default.

    Trainer writes the config with pydantic enum tags that yaml.safe_load rejects, so
    read only the scalars we need with a targeted scan instead of a full parse.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(ckpt_dir, "config.yaml"))
    candidates.append(os.path.join(_PROJECT_ROOT, "config/config.yaml"))

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        use_history, size = None, None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("use_history:"):
                    use_history = s.split(":", 1)[1].strip().lower() in ("true", "yes", "1")
                elif s.startswith("history_size:"):
                    try:
                        size = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
        if size is not None:
            return size if (use_history and size > 0) else 1
    raise RuntimeError("could not determine history_size from any config.yaml")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to step_<N>.pt. Defaults to the newest in checkpoints/.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output bundle dir. Defaults to exported/<run>_<step>/.")
    p.add_argument("--config_path", type=str, default=None,
                   help="Override config.yaml used for history_size.")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    # -- resolve checkpoint ---------------------------------------------------
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        root = os.path.join(_PROJECT_ROOT, "checkpoints")
        runs = sorted(
            (os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))),
            key=os.path.getmtime, reverse=True,
        )
        if not runs:
            print(f"[export] no checkpoint dirs under {root}", file=sys.stderr)
            return 1
        steps = [f for f in os.listdir(runs[0])
                 if f.endswith(".pt") and not f.endswith("_obs_scalers.pt")]
        if not steps:
            print(f"[export] no step_*.pt in {runs[0]}", file=sys.stderr)
            return 1
        steps.sort(key=lambda f: int("".join(c for c in f if c.isdigit()) or 0), reverse=True)
        ckpt_path = os.path.join(runs[0], steps[0])
        print(f"[export] auto-selected {ckpt_path}")

    ckpt_path = os.path.abspath(ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    stem = os.path.basename(ckpt_path)[:-3]  # strip .pt

    out_dir = args.out_dir or os.path.join(
        _PROJECT_ROOT, "exported", f"{os.path.basename(ckpt_dir)}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    # -- rebuild actor --------------------------------------------------------
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        print("[export] checkpoint is not a state_dict", file=sys.stderr)
        return 1

    model, in_dim, act_dim, hidden = build_actor_from_state_dict(sd)
    print(f"[export] actor: {in_dim} -> {hidden} -> {act_dim}")

    env = _gather_env_constants()
    history_size = _load_history_size(ckpt_dir, args.config_path)
    proprio_dim = in_dim // history_size

    # -- stage-1 normalizer ---------------------------------------------------
    scalers_path = os.path.join(ckpt_dir, f"{stem}_obs_scalers.pt")
    if not os.path.exists(scalers_path):
        print(f"[export] MISSING {scalers_path}\n"
              f"[export] Without stage-1 normalizer stats the policy CANNOT run.",
              file=sys.stderr)
        return 1
    scalers = torch.load(scalers_path, map_location="cpu")
    key = "actor" if "actor" in scalers else next(iter(scalers))
    obs_mean = scalers[key]["mean"].flatten().double().numpy()
    obs_var = scalers[key]["var"].flatten().double().numpy()
    print(f"[export] stage-1 scaler '{key}': {obs_mean.shape[0]} dims, "
          f"count={float(scalers[key].get('count', -1)):.3g}")

    # -- export ---------------------------------------------------------------
    onnx_path = os.path.join(out_dir, "policy.onnx")
    example = torch.zeros(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, example, onnx_path,
        input_names=["obs"], output_names=["action"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=None,  # static batch 1: the robot runs one policy at a time
    )
    print(f"[export] wrote {onnx_path}")

    # -- validation -----------------------------------------------------------
    failures: list[str] = []

    if in_dim != history_size * proprio_dim:
        failures.append(f"in_dim {in_dim} != history_size {history_size} * proprio {proprio_dim}")
    if proprio_dim != 50:
        failures.append(f"proprio dim is {proprio_dim}, expected 50 "
                        f"(the old export.py assumed 52 and never ran)")
    if act_dim != 12:
        failures.append(f"action dim is {act_dim}, expected 12")
    if obs_mean.shape[0] != in_dim:
        failures.append(f"stage-1 scaler has {obs_mean.shape[0]} dims, actor wants {in_dim}")

    # Index 2 of the newest frame is IMU specific-force Z. A robot standing upright
    # reads ~+9.81 there. If the mean is near 0 the policy was trained on
    # gravity-removed linear acceleration and the Jetson's BNO055 mapping must match.
    newest = (history_size - 1) * proprio_dim
    acc_z_mean = float(obs_mean[newest + 2])
    if not (8.0 < acc_z_mean < 11.5):
        failures.append(
            f"obs_mean[{newest + 2}] (IMU specific-force Z) = {acc_z_mean:.3f}, "
            f"expected ~+9.81. Check the gravity convention before deploying.")
    else:
        print(f"[export] IMU specific-force Z mean = {acc_z_mean:.3f} (expected ~9.81) OK")

    step_dt = env["sim_dt"] * env["decimation"]
    if abs(step_dt - 0.02) > 1e-9:
        failures.append(f"step_dt {step_dt} != 0.02 (50 Hz)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(100):
            x = rng.standard_normal((1, in_dim)).astype(np.float32) * 3.0
            with torch.no_grad():
                ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(["action"], {"obs": x})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
        if worst > 1e-5:
            failures.append(f"ONNX vs PyTorch max abs diff {worst:.3e} > 1e-5")
        else:
            print(f"[export] ONNX vs PyTorch over 100 random inputs: max diff {worst:.3e} OK")
        if not (np.abs(got) <= 1.0 + 1e-6).all():
            failures.append("ONNX output escaped [-1, 1]; tanh missing?")
    except ImportError:
        print("[export] WARNING: onnxruntime not installed, skipping numerical check")

    if failures:
        print("\n[export] VALIDATION FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    # -- bundle ---------------------------------------------------------------
    deploy = {
        "format_version": 1,
        "obs_dim": in_dim,
        "action_dim": act_dim,
        "history_size": history_size,
        "obs_proprio_dim": proprio_dim,
        "obs_layout": OBS_LAYOUT,
        "history_order": "oldest_first",  # newest frame occupies obs[-proprio_dim:]
        "joint_names": JOINT_NAMES,
        "joint_limits_min": env["joint_limits_min"],
        "joint_limits_max": env["joint_limits_max"],
        "default_joint_pos": [0.0] * act_dim,  # every V2 default is 0.0
        "obs_mean": obs_mean.tolist(),
        "obs_var": obs_var.tolist(),
        "obs_epsilon": 1e-4,
        "action_filter_alpha": env["action_filter_alpha"],
        "action_filter_applications_per_step": env["decimation"],
        "step_dt": step_dt,
        "control_hz": round(1.0 / step_dt),
        "gait_clock_freq": env["gait_clock_freq"],
        "command_ranges": {
            "lin_vel_x": [-0.3, 0.5],
            "lin_vel_y": [-0.3, 0.3],
            "ang_vel_z": [-0.3, 0.3],
        },
        "tilt_fault_rad": 0.784,  # matches biped_env._tilt_exceeded
        "notes": {
            "normalizer": "stage 1 (obs_mean/obs_var) applied by runtime; "
                          "stage 2 baked into policy.onnx",
            "action": "tanh already applied in-graph; clip to [-1,1] then "
                      "target = min + (a+1)/2 * (max-min)",
            "ema": "apply action_filter_applications_per_step times per control step",
        },
    }
    with open(os.path.join(out_dir, "deploy_config.json"), "w", encoding="utf-8") as fh:
        json.dump(deploy, fh, indent=2)

    with open(os.path.join(out_dir, "checkpoint_info.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "source_checkpoint": ckpt_path,
            "run": os.path.basename(ckpt_dir),
            "step": "".join(c for c in stem if c.isdigit()),
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "hidden_sizes": hidden,
        }, fh, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"\n[export] bundle ready: {out_dir}  ({size_mb:.1f} MB)")
    print(f"[export] copy to the Jetson under BipedRobotJetson/models/<name>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
