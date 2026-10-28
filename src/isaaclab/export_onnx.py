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

EQUIVARIANT ACTOR (train.equivariant_actor)
-------------------------------------------
A checkpoint trained with the mirror-equivariant actor stores the plain actor as ``actor.net.*`` plus
the mirror tables as ``actor.mirror_*`` buffers. The exported graph then contains the symmetrisation
tanh(1/2 (f(s) + M_a f(M_s s))) (Gather + Mul + Add), so the bundle format and the Jetson runtime are
unchanged: it still maps obs [1,250] -> action [1,12].
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
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
    "joint_pos_rel": [12, 24],     # joint_pos - default_joint_pos (0 except action_map residual: the home pose)
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

    def pre_tanh(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.norm_mean) / self.norm_std
        return self.mean_layer(self.net(x))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.pre_tanh(obs))


class EquivariantDeterministicActor(nn.Module):
    """Deterministic path of torch_rl_algorithms' EquivariantActor (train.equivariant_actor):

        tanh( 1/2 ( f(s) + M_a f(M_s s) ) )        f = pre-tanh mean of the plain actor

    The mirror tables are buffers (obs index/sign over the full history stack, action index/sign), so the
    symmetrisation is part of the ONNX graph (Gather + Mul + Add); the Jetson runtime is unchanged. The
    std head is not needed on the deterministic path.
    """

    def __init__(self, base: DeterministicActor, obs_perm, obs_sign, act_perm, act_sign):
        super().__init__()
        self.base = base
        self.register_buffer("obs_perm", torch.as_tensor(obs_perm, dtype=torch.long).clone())
        self.register_buffer("obs_sign", torch.as_tensor(obs_sign, dtype=torch.float32).clone())
        self.register_buffer("act_perm", torch.as_tensor(act_perm, dtype=torch.long).clone())
        self.register_buffer("act_sign", torch.as_tensor(act_sign, dtype=torch.float32).clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mirrored = obs.index_select(-1, self.obs_perm) * self.obs_sign
        mu_m = self.base.pre_tanh(mirrored)
        mu_m = mu_m.index_select(-1, self.act_perm) * self.act_sign
        return torch.tanh(0.5 * (self.base.pre_tanh(obs) + mu_m))


def build_actor_from_state_dict(sd: dict) -> tuple[nn.Module, int, int, list[int]]:
    """Rebuild the actor using only tensor shapes from the checkpoint.

    Deriving dims from the config is what broke the old exporter (it assumed a
    52-dim proprio frame). The checkpoint is authoritative.

    A checkpoint trained with train.equivariant_actor stores the plain actor under `actor.net.*` plus the
    mirror tables as `actor.mirror_*` buffers; that case returns an EquivariantDeterministicActor.
    """
    equivariant = "actor.mirror_obs_perm" in sd
    pfx = "actor.net." if equivariant else "actor."
    # Torso Linears live at <pfx>torso.net.{0,2,4}.weight (odd indices are ELU).
    linear_ids = sorted(
        int(k.split(".")[len(pfx.split(".")) + 1])
        for k in sd
        if k.startswith(pfx + "torso.net.") and k.endswith(".weight")
    )
    if not linear_ids:
        raise KeyError(f"no {pfx}torso.net.*.weight tensors in checkpoint")

    hidden = [int(sd[f"{pfx}torso.net.{i}.weight"].shape[0]) for i in linear_ids]
    in_dim = int(sd[f"{pfx}torso.net.{linear_ids[0]}.weight"].shape[1])
    act_dim = int(sd[f"{pfx}mean_layer.0.weight"].shape[0])

    base = DeterministicActor(hidden, in_dim, act_dim)

    with torch.no_grad():
        base.norm_mean.copy_(sd[f"{pfx}torso.observation_normalizer._mean"].flatten())
        base.norm_std.copy_(sd[f"{pfx}torso.observation_normalizer._std"].flatten())
        for dst, src in enumerate(linear_ids):
            base.net[dst * 2].weight.copy_(sd[f"{pfx}torso.net.{src}.weight"])
            base.net[dst * 2].bias.copy_(sd[f"{pfx}torso.net.{src}.bias"])
        base.mean_layer.weight.copy_(sd[f"{pfx}mean_layer.0.weight"])
        base.mean_layer.bias.copy_(sd[f"{pfx}mean_layer.0.bias"])

    model: nn.Module = base
    if equivariant:
        tables = [sd[f"actor.{k}"].cpu() for k in ("mirror_obs_perm", "mirror_obs_sign", "mirror_act_perm", "mirror_act_sign")]
        if tables[0].numel() != in_dim or tables[2].numel() != act_dim:
            raise ValueError(f"mirror tables have obs/action length {tables[0].numel()}/{tables[2].numel()}, "
                             f"actor is {in_dim} -> {act_dim}")
        for name, (perm, sign) in (("observation", tables[0:2]), ("action", tables[2:4])):
            if not (torch.equal(perm[perm], torch.arange(perm.numel())) and torch.equal(sign * sign[perm], torch.ones_like(sign))):
                raise ValueError(f"{name} mirror table in the checkpoint is not an involution")
        model = EquivariantDeterministicActor(base, *tables)

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


def _load_policy_contract(ckpt_dir: str, explicit_cfg: str | None, explicit_map: str | None) -> dict:
    """Action map + servo model the checkpoint was TRAINED with, from the run's own config.

    The repo's current config.yaml is deliberately NOT used as a fallback here, unlike
    history_size. Exporting an old checkpoint with today's action_map would stamp an
    absolute-map policy as "centered", and the robot would execute every action wrongly.
    If the run's config can't be found, the caller must say what the map is.
    """
    paths = [p for p in (explicit_cfg, os.path.join(ckpt_dir, "config.yaml")) if p and os.path.exists(p)]
    found: dict = {}
    if paths:
        with open(paths[0], "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.split("#", 1)[0].strip()
                for key in ("action_map", "params_file", "kp_fw", "stall_torque_cap_nm"):
                    if s.startswith(key + ":"):
                        found.setdefault(key, s.split(":", 1)[1].strip().strip("'\""))
    if explicit_map:
        found["action_map"] = explicit_map
    if "action_map" not in found:
        if not paths:
            raise RuntimeError(
                f"no config.yaml beside the checkpoint ({ckpt_dir}); pass --action_map "
                f"absolute|centered explicitly. Guessing would silently mis-map every action.")
        found["action_map"] = "absolute"    # runs from before the key existed
    if found["action_map"] not in ("absolute", "centered", "residual"):
        raise RuntimeError(f"unknown action_map {found['action_map']!r}")

    contract = {"action_map": found["action_map"], "actuator": None}
    if found["action_map"] == "residual":
        # Home pose + per-type scales the run TRAINED with (never the repo's current config).
        if not paths:
            raise RuntimeError("action_map residual needs the run's config.yaml for action_home_deg / "
                               "action_scale_rad; none found beside the checkpoint")
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from utils.residual_action import check_residual_map, home_rad, scale_pair_rad
        home_deg, scale = _scan_dict(paths[0], "action_home_deg"), _scan_dict(paths[0], "action_scale_rad")
        from utils.residual_home import is_auto, resolve_if_auto
        if is_auto(home_deg) or is_auto(scale):
            # A run's saved config.yaml is frozen by train.py and never says auto; a hand-made one might.
            import yaml
            with open(paths[0], "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            env_conf = ((doc.get("train") or {}).get("env_config")) or doc
            auto = resolve_if_auto(env_conf, "export_onnx", _PROJECT_ROOT)
            if auto is None:
                raise RuntimeError(f"{paths[0]}: action_home_deg / action_scale_rad are 'auto' but could not be resolved")
            home_deg = auto[0] if is_auto(home_deg) else home_deg
            scale = auto[1] if is_auto(scale) else scale
        if not home_deg or not scale:
            raise RuntimeError(f"{paths[0]}: action_map residual but action_home_deg / action_scale_rad missing")
        home, (sc, sc_neg) = home_rad(home_deg), scale_pair_rad(scale)      # sc_neg None = symmetric
        robot_py = os.path.join(_PROJECT_ROOT, "envs/assets/robotV2/biped_robot.py")
        limits = np.radians(np.asarray(_literal_from_module(robot_py, "_JOINT_LIMITS_DEG"), dtype=np.float64))
        check_residual_map(home, sc, limits[:, 0], limits[:, 1], scale_neg=sc_neg)
        contract["residual"] = {"home_rad": [float(x) for x in home], "scale_rad": [float(x) for x in sc],
                                "scale_neg_rad": None if sc_neg is None else [float(x) for x in sc_neg]}
    if "params_file" in found:
        pf = found["params_file"]
        full = pf if os.path.isabs(pf) else os.path.join(_PROJECT_ROOT, pf)
        sha = None
        if os.path.exists(full):
            with open(full, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
        contract["actuator"] = {"model": "bam", "params_file": pf, "params_sha256": sha,
                                "kp_fw": float(found.get("kp_fw", 32)),
                                "stall_torque_cap_nm": (None if found.get("stall_torque_cap_nm") in (None, "null", "None")
                                                        else float(found["stall_torque_cap_nm"])),
                                "requires_pid": {"p": int(float(found.get("kp_fw", 32))), "d": 0, "i": 0}}
    return contract


def _scan_block(path: str, block: str) -> dict:
    """Parse ONE block of a run's config.yaml in isolation.

    The trainer writes the file with pydantic enum tags that yaml.safe_load rejects, so the whole
    file cannot be parsed; a plain block (reference, commands) can. Indentation-aware: the block
    is every following line indented deeper than the `<block>:` line. Returns {} if absent.
    """
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    start, indent = None, 0
    for i, raw in enumerate(lines):
        if raw.split("#", 1)[0].strip() == f"{block}:":
            start, indent = i, len(raw) - len(raw.lstrip())
            break
    if start is None:
        return {}
    body = []
    for raw in lines[start + 1:]:
        if raw.strip() and not raw.lstrip().startswith("#") and len(raw) - len(raw.lstrip()) <= indent:
            break
        body.append(raw)
    return yaml.safe_load(textwrap.dedent("\n".join(body))) or {}


def _scan_dict(path: str, key: str) -> dict:
    """A dict-valued key of the run's config.yaml, block style (`key:` + indented lines) or
    inline flow style (`key: {a: 1}`)."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            s = raw.split("#", 1)[0].strip()
            if s.startswith(key + ":") and s[len(key) + 1:].strip():
                return yaml.safe_load(s[len(key) + 1:]) or {}
    return _scan_block(path, key)


def _format_version(contract: dict, gait_clock: dict) -> int:
    """v5 = cadence-grid gait clock (table v2: the clock rate depends on the command); a runtime
    that predates it would run the clock at one rate, so it must refuse v5. v5 may also carry
    the residual map. v4 = residual action map (home pose in default_joint_pos +
    action_scale_rad); a runtime that predates it would treat the action as centered/absolute,
    so it must refuse v4. v6 = asymmetric residual scales (action_scale_pos_rad / action_scale_neg_rad instead of
    action_scale_rad; a v5 runtime would read no scale at all), whatever the gait clock. A symmetric map never gets
    v6, so those bundles stay byte-identical. Otherwise unchanged: v3 for a fixed-period gait clock, else v2."""
    if contract["action_map"] == "residual" and (contract.get("residual") or {}).get("scale_neg_rad") is not None:
        return 6
    if gait_clock["mode"] == "cadence_grid":
        return 5
    if contract["action_map"] == "residual":
        return 4
    return 3 if gait_clock["mode"] == "fixed_period" else 2


def _run_config(ckpt_dir: str, explicit_cfg: str | None) -> str | None:
    """The config the checkpoint was trained with. Never the repo's current config.yaml."""
    for p in (explicit_cfg, os.path.join(ckpt_dir, "config.yaml")):
        if p and os.path.exists(p):
            return p
    return None


def _scan_scalar(path: str, key: str):
    """Value of the first `<key>: value` line in a run's config.yaml (None if absent). Line-based
    for the same reason as _scan_block: the whole file does not parse under safe_load."""
    import re
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            m = re.match(rf"^\s+{re.escape(key)}:\s*([^\s#]+)", raw)
            if m:
                return m.group(1).strip("'\"")
    return None


def _load_gait_clock(ckpt_dir: str, explicit_cfg: str | None, ckpt_path: str, speed_freq: float) -> dict:
    """Gait clock the policy was trained with. With a gait table the clock holds while standing
    and runs at the table's rate: one period (v1 tables, or a v2 table with a constant cadence)
    or the v2 per-command freq grid (cadence_grid, trilinear in the command, clamped to the
    grid). A runtime that advances it any other way would feed the policy a phase it never saw."""
    cfg = _run_config(ckpt_dir, explicit_cfg)
    ref = _scan_block(cfg, "reference") if cfg else {}
    if not ref.get("npz_path"):
        # No table: env_config.gait_clock_mode picks the clock (absent in older runs = speed-proportional).
        if cfg and _scan_scalar(cfg, "gait_clock_mode") == "fixed_period":
            period = float(_scan_scalar(cfg, "gait_clock_period_s") or 0.7)
            thr = float(_scan_scalar(cfg, "gait_clock_stand_threshold") or 0.05)
            if not (0.2 < period < 5.0):
                raise RuntimeError(f"gait_clock_period_s {period} s is outside the 0.2-5 s the Jetson accepts")
            return {"mode": "fixed_period", "period": period, "stand_threshold": thr}
        if cfg and _scan_scalar(cfg, "gait_clock_mode") == "cadence_law":
            # Config-defined cadence (utils/cadence_law.py): exported as the same cadence grid a v2 table gives (v5).
            from utils.cadence_law import cadence_grid
            ranges = (_scan_block(cfg, "commands").get("base_velocity") or {}).get("ranges") or {}
            t_min = float(_scan_scalar(cfg, "gait_clock_period_min_s") or 0.6)
            t_max = float(_scan_scalar(cfg, "gait_clock_period_max_s") or 0.9)
            grid = cadence_grid(ranges, t_min, t_max)
            return {"mode": "cadence_grid",
                    "grid_vx": [float(v) for v in grid.axes[0]], "grid_vy": [float(v) for v in grid.axes[1]],
                    "grid_wz": [float(v) for v in grid.axes[2]], "freq": grid.freq.round(6).tolist(),
                    "period": t_max, "stand_threshold": float(_scan_scalar(cfg, "gait_clock_stand_threshold") or 0.05),
                    "cadence_law": {"period_min_s": t_min, "period_max_s": t_max}}
        return {"mode": "speed_proportional", "freq": speed_freq}
    table = ref["npz_path"]
    full = table if os.path.isabs(table) else os.path.join(_PROJECT_ROOT, table)
    if not os.path.exists(full):
        raise RuntimeError(f"run was trained on gait table {table!r} but {full} is missing; the "
                           f"period (clock rate) cannot be exported without it")
    z = np.load(full, allow_pickle=False)
    if os.path.getmtime(full) > os.path.getmtime(ckpt_path):
        print(f"[export] WARNING: {table} was re-baked after this checkpoint was written; the exported "
              f"clock may differ from the one the policy trained on", file=sys.stderr)
    common = {"stand_threshold": float(ref.get("stand_threshold", 0.05)),
              "table": table, "style_digest": str(z["style_digest"]),
              "phase_convention": str(z["phase_convention"])}
    freq = np.asarray(z["freq"], dtype=np.float64) if "freq" in z else None
    if freq is None or np.ptp(freq) < 1e-9:
        period = float(z["period"]) if freq is None else 1.0 / float(freq.flat[0])
        return {"mode": "fixed_period", "period": period, **common}
    if not (0.2 < freq.min() and freq.max() < 5.0):
        raise RuntimeError(f"gait table {table}: clock frequency {freq.min():.3f}-{freq.max():.3f} Hz is outside "
                           f"the 0.2-5 Hz the Jetson accepts")
    return {"mode": "cadence_grid",
            "grid_vx": [float(v) for v in z["grid_vx"]],
            "grid_vy": [float(v) for v in z["grid_vy"]],
            "grid_wz": [float(v) for v in z["grid_wz"]],
            "freq": freq.round(6).tolist(),        # [nvx][nvy][nwz] cycles/s
            "period": float(z["period"]),          # zero-command cycle, display only
            **common}


_LEGACY_COMMAND_RANGES = {"lin_vel_x": [-0.3, 0.5], "lin_vel_y": [-0.3, 0.3], "ang_vel_z": [-0.3, 0.3]}


def _load_command_ranges(ckpt_dir: str, explicit_cfg: str | None) -> dict:
    """Command ranges the policy was trained on, so the runtime clamps joystick input to them."""
    cfg = _run_config(ckpt_dir, explicit_cfg)
    ranges = ((_scan_block(cfg, "commands").get("base_velocity") or {}).get("ranges") or {}) if cfg else {}
    out = {k: [float(v) for v in ranges[k]] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if k in ranges}
    missing = [k for k in _LEGACY_COMMAND_RANGES if k not in out]
    if missing:
        print(f"[export] WARNING: command ranges for {missing} not found in the run config; using the "
              f"legacy defaults {_LEGACY_COMMAND_RANGES}", file=sys.stderr)
        for k in missing:
            out[k] = _LEGACY_COMMAND_RANGES[k]
    return out


def _load_command_limit(ckpt_dir: str, explicit_cfg: str | None, command_ranges: dict) -> dict:
    """The command limit the policy was trained under: shape (commands.base_velocity.limit_shape of the run
    config; absent = "box", the pre-limit behaviour) plus the ranges. Written to deploy_config.json as an
    optional key (no format bump): a runtime that does not know it keeps clamping per axis, which is a box."""
    from utils.command_limit import validate_shape
    cfg = _run_config(ckpt_dir, explicit_cfg)
    bv = (_scan_block(cfg, "commands").get("base_velocity") or {}) if cfg else {}
    shape = validate_shape(bv.get("limit_shape"))
    return {"shape": shape, **{k: list(command_ranges[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to step_<N>.pt. Defaults to the newest in checkpoints/.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output bundle dir. Defaults to exported/<run>_<step>/.")
    p.add_argument("--config_path", type=str, default=None,
                   help="Override config.yaml used for history_size.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--action_map", choices=["absolute", "centered", "residual"], default=None,
                   help="Only needed when the checkpoint's run config.yaml is missing.")
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
    equivariant = isinstance(model, EquivariantDeterministicActor)
    print(f"[export] actor: {in_dim} -> {hidden} -> {act_dim}" + ("  (mirror-equivariant: symmetrisation baked into the graph)" if equivariant else ""))

    env = _gather_env_constants()
    history_size = _load_history_size(ckpt_dir, args.config_path)
    contract = _load_policy_contract(ckpt_dir, args.config_path, args.action_map)
    print(f"[export] action_map = {contract['action_map']}, actuator = {contract['actuator']}")
    gait_clock = _load_gait_clock(ckpt_dir, args.config_path, ckpt_path, env["gait_clock_freq"])
    command_ranges = _load_command_ranges(ckpt_dir, args.config_path)
    command_limit = _load_command_limit(ckpt_dir, args.config_path, command_ranges)
    print(f"[export] gait_clock = {gait_clock}, command_ranges = {command_ranges}, "
          f"command_limit shape = {command_limit['shape']}")
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
        # float32 through a ~1.9M-parameter MLP: ORT and PyTorch differ by accumulation order,
        # ~1e-5 on a [-1, 1] output. A real export bug (wrong normaliser, missing tanh) is >1e-2.
        if worst > 1e-3:
            failures.append(f"ONNX vs PyTorch max abs diff {worst:.3e} > 1e-3")
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
        # v2 adds action_map and actuator. A runtime that predates them must refuse v2 bundles
        # rather than silently applying the v1 (absolute) map to a centered policy.
        # v3 = a fixed-period gait_clock (table-trained policy). A runtime that only knows
        # gait_clock_freq would clock the policy wrongly, so it must refuse v3 as well. Bundles
        # without a table stay v2.
        # v4 = residual action map, v5 = cadence-grid gait clock, v6 = asymmetric residual scales (see _format_version).
        "format_version": _format_version(contract, gait_clock),
        "action_map": contract["action_map"],
        "actuator": contract["actuator"],
        "obs_dim": in_dim,
        "action_dim": act_dim,
        "history_size": history_size,
        "obs_proprio_dim": proprio_dim,
        "obs_layout": OBS_LAYOUT,
        "history_order": "oldest_first",  # newest frame occupies obs[-proprio_dim:]
        "joint_names": JOINT_NAMES,
        "joint_limits_min": env["joint_limits_min"],
        "joint_limits_max": env["joint_limits_max"],
        # Every V2 default is 0.0 except the residual map's home pose (rad, policy order).
        "default_joint_pos": (contract["residual"]["home_rad"] if contract["action_map"] == "residual"
                              else [0.0] * act_dim),
        **({} if contract["action_map"] != "residual"
           else {"action_scale_rad": contract["residual"]["scale_rad"]} if contract["residual"].get("scale_neg_rad") is None
           else {"action_scale_pos_rad": contract["residual"]["scale_rad"],
                 "action_scale_neg_rad": contract["residual"]["scale_neg_rad"]}),
        "obs_mean": obs_mean.tolist(),
        "obs_var": obs_var.tolist(),
        "obs_epsilon": 1e-4,
        "action_filter_alpha": env["action_filter_alpha"],
        "action_filter_applications_per_step": env["decimation"],
        "step_dt": step_dt,
        "control_hz": round(1.0 / step_dt),
        "gait_clock": gait_clock,
        "gait_clock_freq": env["gait_clock_freq"],   # legacy runtimes; ignored unless gait_clock.mode is speed_proportional
        "command_ranges": command_ranges,
        # Optional (no format bump; absent = "box" = clamp per axis only): the training limit shape, so the
        # runtime scales (vx, vy) onto the ellipse exactly as training/MuJoCo do (utils/command_limit.py).
        "command_limit": command_limit,
        "tilt_fault_rad": 0.784,  # matches biped_env._tilt_exceeded
        "notes": {
            "normalizer": "stage 1 (obs_mean/obs_var) applied by runtime; "
                          "stage 2 baked into policy.onnx",
            "action": "tanh already applied in-graph; clip to [-1,1]. action_map 'absolute': "
                      "target = min + (a+1)/2 * (max-min). 'centered': target = default + "
                      "a * (max - default) for a >= 0, default + a * (default - min) for a < 0. "
                      "'residual' (format 4): target = clip(default_joint_pos + action_scale_rad * a, "
                      "min, max), default_joint_pos being the crouched home pose; joint_pos_rel = "
                      "joint_pos - default_joint_pos, and arming ramps to that pose"
                      + ("" if contract["action_map"] != "residual" or contract["residual"].get("scale_neg_rad") is None else
                         ". format 6: the scale is asymmetric, target = clip(default_joint_pos + "
                         "action_scale_pos_rad * a, min, max) for a >= 0 and clip(default_joint_pos + "
                         "action_scale_neg_rad * a, min, max) for a < 0 (no action_scale_rad key)"),
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
            "equivariant_actor": equivariant,
        }, fh, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"\n[export] bundle ready: {out_dir}  ({size_mb:.1f} MB)")
    print(f"[export] copy to the Jetson under BipedRobotJetson/models/<name>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
