"""policy_gate.py — Pre-deploy gate for an EXPORTED policy bundle (what goes on the robot).

Why it exists: on 2026-10-15 a deployed policy showed a 25 Hz bang-bang on hardware (raw action
flips sign every step, +/-1 each tick). The Jetson rate limiter (max_action_rate per step) then
saturates on every step, so the closed loop no longer behaves like the one that was trained. The
limiter is a safety net; it must RARELY fire. This gate catches such policies before they ship.

It runs the bundle's policy.onnx with the bundle's own normalisation (deploy_config.json) in the
MuJoCo env, using the Jetson limiter semantics (BipedRobotJetson/policy/safety.py limit_action):
    a = clip(raw, -1, 1);  a = prev + clip(a - prev, -rate, rate);  prev = APPLIED action
and the previous_actions obs slot carries the APPLIED action (not the delayed one).

Checks:
  1. Offline gain: spectral radius of d(action)/d(previous_actions) of the newest frame (central
     finite difference, 1e-3) on MuJoCo standing states and, with --trace, on hardware-trace states.
  2. Closed loop with the limiter, extra action delay d in {0,1,2,3}, cmd 0 and walk (0.2, 0, 0):
     falls, limiter-active fraction (steps where ANY joint is limited), raw-action lag-1
     autocorrelation, rms applied delta-a, mean |gyro|, forward velocity (vx = mean body-frame forward velocity after warmup).
     Start state matches the Jetson: zero-filled history, previous action 0, gait phase 0.
  3. Pose sanity and inter-leg contact (from the same closed-loop cases, window 2-6 s):
     per-joint saturation fraction (|raw action| > 0.95) and mean joint_pos_rel; normal contact force
     between ANY geom of a left-leg body and ANY geom of a right-leg body (floor and base excluded),
     per body-pair class; foot-origin and sole-centre lateral separation. Informational only: a
     +-0.10 lateral projected-gravity bias run (cmd 0, 6 s, 1 seed).
  4. PASS/FAIL thresholds (see _verdict). Exit code 0 = PASS, 1 = FAIL.

Usage (from BipedRobot/):
    python -m src.mujoco.policy_gate --bundle exported/<name> [--trace t.npz] [--seeds 3]
                                     [--seconds 8] [--rate 0.5]
    python src/mujoco/policy_gate.py --bundle exported/<name> ...      (also works)

Sweep mode (per-command behaviour, e.g. does the policy walk at 0.1 m/s?):
    python src/mujoco/policy_gate.py --checkpoint checkpoints/<run>/step_N.pt --cmds 0.05 0.1 0.15 0.2 0.3
    python src/mujoco/policy_gate.py --bundle exported/<name> --cmds 0.1 0.2 [--delays 0 1]
  Runs vx commands (vy=wz=0), 10 s episodes by default, and prints falls, achieved vx, vx/cmd,
  step frequency (touchdowns of both feet per second, env contact signal), mean single-support
  duration, max swing clearance, rms delta-a, limiter fraction and a walks / stands-stalls verdict.
  A raw checkpoint (.pt, with config.yaml + optional *_obs_scalers.pt next to it) is evaluated through
  sim2real's model/scaler helpers; the Jetson limiter / applied-action prev-slot semantics are the same.

Nothing is written to the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import deque

import numpy as np

# Project root first (its `config` package must win); torch_rl_algorithms only APPENDED, because it
# has a top-level `config` module that would shadow BipedRobot's.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if not os.path.isdir(os.path.join(_ROOT, "envs")):          # file copied elsewhere: use cwd
    _ROOT = os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TRL = os.path.join(_ROOT, "torch_rl_algorithms")
if _TRL not in sys.path:
    sys.path.append(_TRL)

import mujoco
import onnxruntime as ort

from envs.mujoco.biped_env_v2 import BipedEnvV2
from envs.isaaclab.rewards import rewards as _train_rewards      # the SAME footprint gap training penalises
from utils import Config

DELAYS = (0, 1, 2, 3)
WALK_CMD = (0.2, 0.0, 0.0)
WARMUP_STEPS = 25            # 0.5 s
PREV_SLOT = slice(36, 48)    # previous_actions inside one 50-dim frame
GRAV_SLOT = slice(6, 9)      # projected_gravity inside one 50-dim frame (imu_acc 0:3, gyro 3:6)
FRAME = 50

# Pose-sanity / inter-leg-contact window (control steps of 0.02 s: 2 s .. 6 s)
POSE_T0, POSE_T1 = 2.0, 6.0
SAT_RAW = 0.95               # |raw action| above this counts as saturated
SAT_MAX_FRAC = 0.4           # criterion: max over joints of the saturated fraction
LEG_CONTACT_MAX_N = 1.0      # criterion: mean inter-leg normal force
# Edge stance (hardware 2026-10-25: the 18-55-36 policy stood on its OUTER sole edges at cmd 0; on the real 2 mm EVA
# soles the edge creeps and the robot tipped after 4-12 s). A flat sole spreads its floor contacts across the width
# (table stand: +3 / +39 mm from the ankle axis), an edge stance puts them all on one line (1-4 mm spread).
FOOT_ROLL_MAX_DEG = 6.0      # criterion: mean |sole roll| of the loaded feet at cmd 0
FLAT_SPREAD_MIN_MM = 15.0    # a loaded foot counts as flat when its contacts span >= this laterally (sole ~37 mm)
FLAT_CONTACT_MIN_FRAC = 0.8  # criterion: fraction of loaded-foot steps that are flat at cmd 0
GRAV_BIAS = 0.10             # informational lateral projected-gravity bias
GRAV_SECONDS = 6.0
# Sole centre in the foot link frame (x, lateral, z); +y for the left foot, -y for the right
# (Robot.urdf foot meshes: left spans y = +2.0..+39.3 mm, right -39.3..-2.0 mm).
SOLE_L = np.array([0.0163, 0.0211, -0.0316])
SOLE_R = np.array([0.0163, -0.0211, -0.0316])
# Sole rectangle half extents [x, y] about the sole centre (cfg.foot_sole_half_extents_m; run configs that
# predate E21 do not carry it, so this is the fallback). Used for the footprint gap (rewards.foot_footprint_gap).
SOLE_HALF = (0.0450, 0.0186)
PAIR_CLASSES = ("foot-foot", "ankle-ankle", "calf-calf", "other")
_STEM_CLASS = {"foot_link_1": "foot-foot", "ankle_link_1": "ankle-ankle", "calf_link_1": "calf-calf"}
JOINT_LABELS = ("l_hip_yaw", "r_hip_yaw", "l_hip_roll", "r_hip_roll", "l_hip_pitch", "r_hip_pitch",
                "l_knee", "r_knee", "l_ankle_roll", "r_ankle_roll", "l_ankle_pitch", "r_ankle_pitch")


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class CkptPolicy:
    """Raw .pt checkpoint (checkpoints/<run>/step_N.pt + config.yaml), as in sim2real.py / mj_hw.py."""

    def __init__(self, ckpt: str, config):
        import torch
        import src.mujoco.sim2real as s2r
        self._torch, self._s2r = torch, s2r
        self.dev = torch.device("cpu")
        self.hist = int(config.train.history_size)
        if not config.train.use_history:
            raise RuntimeError("checkpoint was trained without history (unsupported by the gate)")
        self.obs_dim = self.hist * FRAME
        ec = config.train.env_config
        ec.backlash = {**(ec.backlash or {}), "enabled": False}
        env = BipedEnvV2(history_size=self.hist, use_history=True, critic_has_privileged_info=True,
                         env_config=ec, seed=0)
        try:
            self.model = s2r._model_class(config)(env=s2r._EnvSpec(env, self.dev), model_path=ckpt,
                                                  device=self.dev, config=config, use_history=True,
                                                  history_size=self.hist)
            getattr(self.model, "model", self.model).eval()
            self.scaler = s2r._load_obs_scaler(ckpt, self.dev)
            obs, _ = env.reset(seed=0)
            self(obs["actor"], obs["critic"])        # surfaces obs-size / architecture mismatches now
        finally:
            env.close()

    def __call__(self, raw_obs: np.ndarray, critic: np.ndarray | None = None) -> np.ndarray:
        s2r, torch = self._s2r, self._torch
        if critic is None:
            critic = np.zeros(FRAME + 10, np.float32)
        with torch.no_grad():
            a = self.model.step({"actor": s2r._normalize_obs(np.asarray(raw_obs, np.float32), self.scaler, "actor", self.dev),
                                 "critic": s2r._normalize_obs(np.asarray(critic, np.float32), self.scaler, "critic", self.dev)})
        a = a.cpu().numpy() if hasattr(a, "cpu") else a
        return np.clip(np.asarray(a).reshape(-1), -1.0, 1.0)


def load_checkpoint_policy(ckpt: str):
    """(CkptPolicy, Config) for checkpoints/<run>/step_N.pt (path relative to cwd or BipedRobot/)."""
    if not os.path.exists(ckpt):
        ckpt = os.path.join(_ROOT, ckpt)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"run config not found: {cfg_path}")
    try:
        config = Config(cfg_path)
    except Exception:                      # old run configs hold !!python/object/apply:...ObsType tags
        import tempfile
        txt = re.sub(r"!!python/object/apply:[\w.]+\s*\n\s*- (\w+)", r"\1",
                     open(cfg_path, encoding="utf-8").read())
        with tempfile.TemporaryDirectory() as td:          # sanitised copy; nothing written to the repo
            tmp = os.path.join(td, "config.yaml")
            open(tmp, "w", encoding="utf-8").write(txt)
            config = Config(tmp)
    return CkptPolicy(ckpt, config), config


class Policy:
    """policy.onnx + stage-1 normalisation from deploy_config.json."""

    def __init__(self, bundle: str):
        self.cfg = json.load(open(os.path.join(bundle, "deploy_config.json")))
        self.sess = ort.InferenceSession(os.path.join(bundle, "policy.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.mean = np.asarray(self.cfg["obs_mean"], np.float64)
        self.std = np.sqrt(np.asarray(self.cfg["obs_var"], np.float64) + float(self.cfg["obs_epsilon"]))
        self.obs_dim = int(self.cfg["obs_dim"])
        self.hist = int(self.cfg["history_size"])

    def __call__(self, raw_obs: np.ndarray) -> np.ndarray:
        x = ((np.asarray(raw_obs, np.float64) - self.mean) / self.std).astype(np.float32)[None]
        return np.clip(self.sess.run(["action"], {"obs": x})[0].reshape(-1), -1.0, 1.0)


def _run_config_path(bundle: str) -> str:
    """checkpoints/<run>/config.yaml from deploy_config model_info, else from the bundle name."""
    cfg = json.load(open(os.path.join(bundle, "deploy_config.json")))
    mi = cfg.get("model_info") or {}
    src = mi.get("checkpoint") or mi.get("source_checkpoint") or mi.get("checkpoint_path")
    run_dir = os.path.dirname(os.path.abspath(src)) if src else None
    if not run_dir or not os.path.exists(os.path.join(run_dir, "config.yaml")):
        name = os.path.basename(os.path.normpath(bundle))
        m = re.match(r"(.+?)_step_\d+$", name)
        run_dir = os.path.join(_ROOT, "checkpoints", m.group(1) if m else name)
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run config not found: {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Offline gain
# ---------------------------------------------------------------------------

def spectral_radius(policy: Policy, obs: np.ndarray, h: float = 1e-3) -> float:
    """Spectral radius of d action / d previous_actions (newest frame), central finite difference."""
    base = (policy.hist - 1) * FRAME + PREV_SLOT.start
    J = np.zeros((12, 12))
    for j in range(12):
        up, dn = obs.copy(), obs.copy()
        up[base + j] += h
        dn[base + j] -= h
        J[:, j] = (policy(up) - policy(dn)) / (2 * h)
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def trace_states(trace_path: str, hist: int, n: int = 15) -> list[np.ndarray]:
    """~n stacked observations (oldest first) from consecutive finite rows of the trace's `frame`."""
    fr = np.asarray(np.load(trace_path)["frame"], np.float32)
    ok = np.isfinite(fr).all(axis=1)
    ends = [i for i in range(hist - 1, len(fr)) if ok[i - hist + 1:i + 1].all()]
    if not ends:
        return []
    pick = np.unique(np.linspace(0, len(ends) - 1, min(n, len(ends))).astype(int))
    return [fr[ends[k] - hist + 1:ends[k] + 1].reshape(-1) for k in pick]


# ---------------------------------------------------------------------------
# 2. Closed loop
# ---------------------------------------------------------------------------

def leg_sides(model):
    """(side, stem) per body id. side: 0 = left leg, 1 = right leg, -1 = world / base / other.
    Membership is by MJCF body name (l_* / r_* links), stem is the name without the l_/r_ prefix."""
    side = np.full(model.nbody, -1, dtype=int)
    stem = [""] * model.nbody
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("l_"):
            side[b], stem[b] = 0, n[2:]
        elif n.startswith("r_"):
            side[b], stem[b] = 1, n[2:]
    return side, stem


def inter_leg_contact(env, side, stem) -> dict:
    """Sum of NORMAL contact force [N] between any left-leg geom and any right-leg geom, per
    body-pair class (foot-foot, ankle-ankle, calf-calf, other). Floor and base contacts are
    excluded (their bodies have side -1)."""
    out = dict.fromkeys(PAIR_CLASSES, 0.0)
    m, d = env.model, env.data
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if side[b1] < 0 or side[b2] < 0 or side[b1] == side[b2]:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        cls = _STEM_CLASS.get(stem[b1], "other") if stem[b1] == stem[b2] else "other"
        out[cls] += abs(float(f6[0]))
    return out


def foot_separations(env) -> tuple[float, float]:
    """(foot-origin, sole-centre) lateral separation, left minus right, along body Y, metres."""
    R = np.asarray(env._world_to_body_rot(), np.float64)
    xp, xm = env.data.xpos, env.data.xmat
    pr, pl = xp[env._r_foot_body].copy(), xp[env._l_foot_body].copy()
    cr = pr + xm[env._r_foot_body].reshape(3, 3) @ SOLE_R
    cl = pl + xm[env._l_foot_body].reshape(3, 3) @ SOLE_L
    return float((R @ (pl - pr))[1]), float((R @ (cl - cr))[1])


def foot_flatness(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per foot [right, left]: sole roll [deg] (training foot_flat_roll_l2 geometry: foot-body z = sole normal, its
    component along the base lateral axis projected horizontally = sin(roll)), the lateral spread [mm] of that foot's
    floor contacts along the same axis (nan with < 2 contacts), and the loaded flag (env._in_contact)."""
    m, d = env.model, env.data
    by = env._world_to_body_rot().T[:, 1]
    lat = np.array([by[0], by[1], 0.0]) / max(float(np.linalg.norm(by[:2])), 1e-9)
    roll, spread = np.full(2, np.nan), np.full(2, np.nan)
    for k, b in enumerate((env._r_foot_body, env._l_foot_body)):
        n = d.xmat[b].reshape(3, 3)[:, 2]
        roll[k] = np.degrees(np.arcsin(np.clip(float(n @ lat), -1.0, 1.0)))
        ys = [float(np.asarray(d.contact[i].pos) @ lat) for i in range(d.ncon)
              if env._floor_geom in (d.contact[i].geom1, d.contact[i].geom2)
              and b in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])]
        if len(ys) >= 2:
            spread[k] = (max(ys) - min(ys)) * 1000.0
    return roll, spread, np.asarray(env._in_contact, bool).copy()


def footprint_params(ec) -> tuple[list, list]:
    """(sole centre offset [x, lateral magnitude, z], half extents [x, y]) from the run's env_config, else the defaults."""
    off = getattr(ec, "foot_sole_offset_m", None)
    half = getattr(ec, "foot_sole_half_extents_m", None)
    return ([float(v) for v in (off if off is not None else (SOLE_L[0], SOLE_L[1], SOLE_L[2]))],
            [float(v) for v in (half if half is not None else SOLE_HALF)])


def foot_footprint_gap(env, off, half) -> float:
    """Signed horizontal-plane gap [m] between the two sole rectangles, from the foot BODY pose in MuJoCo, with the
    function training uses (rewards.foot_footprint_gap). Negative = the footprints overlap."""
    import torch
    xp, xq = env.data.xpos, env.data.xquat

    def t(v):
        return torch.as_tensor(np.asarray(v, np.float32)[None])
    lb, rb = env._l_foot_body, env._r_foot_body
    return float(_train_rewards.foot_footprint_gap(t(xp[lb]), t(xq[lb]), t(xp[rb]), t(xq[rb]), off, half)[0])


class FootSlipProbe:
    """Per-step foot slip / drag quantities of the MuJoCo env (E25), for calibration and the pose report.

    Feeds the foot LINK-body pose / velocity (data.xpos, xquat, mj_objectVelocity flg_local=0 = [w; v] at the body origin,
    world axes) to the SAME rewards.py functions the Isaac env uses (foot_sole_slip_speed, foot_slip, foot_drag), with
    load = env._in_contact (|F| > 1 N). Clearances are listed per variant below.
    Also measured, per step (feet ordered [right, left]):
      true       force-weighted planar speed of the foot at its MuJoCo floor contact points (mj_jac at the contact
                 position; the floor is static), summed over the loaded feet. The ground truth for foot_slip.
      slip       rewards.foot_slip (new term).        old_origin  sum over loaded feet of |v_origin,xy| (rolling included).
      old_isaac  what Isaac feet_slide measures: the foot COM velocity (robot.data.body_lin_vel_w) x the soft contact
                 score max(loaded, clamp(1 - clearance_origin / thresh, 0, 1)).
      drag       rewards.foot_drag as the Isaac env computes it: clearance = lowest sole CORNER height minus its value at
                 the last loaded step (datum emulated here).   drag_mesh  same with the lowest foot MESH vertex above
                 the floor.   drag_origin  with the swing_foot_height clearance (foot ORIGIN height minus its last
                 in-contact height) -- NOT what the env uses: the origin rises 18 mm when the foot pitches.
      dragged    the Diagnostics/dragged_foot_frac count: unloaded, origin clearance < thresh, |v_com,xy| > 0.05 m/s.
    All speeds in m/s. update() after every env.step(); series() -> array [T, len(NAMES)].
    """
    NAMES = ("true", "slip", "old_origin", "old_isaac", "drag", "drag_mesh", "dragged", "n_loaded",
             "slip_per_loaded", "clr_sole_min_mm",
             # raw per-foot [right, left] inputs, so a drag height other than h can be evaluated offline
             "slipf_r", "slipf_l", "clr_sole_r", "clr_sole_l", "clr_orig_r", "clr_orig_l", "loaded_r", "loaded_l",
             "clr_corner_r", "clr_corner_l", "drag_origin")

    def __init__(self, env, off, half, h_drag: float = 0.005, thresh: float = 0.004):
        import torch
        self._t = torch
        self.env, self.h, self.thr = env, h_drag, thresh
        self.off, self.half = [float(v) for v in off], [float(v) for v in half]
        m = env.model
        self.bodies = (env._r_foot_body, env._l_foot_body)
        self.geoms = (set(env._r_foot_geoms), set(env._l_foot_geoms))
        self.verts = []                                           # per foot: [(geom id, vertices (n, 3) in the geom frame)]
        for gs in (env._r_foot_geoms, env._l_foot_geoms):
            vs = []
            for g in gs:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(m.geom_dataid[g])
                    a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                    vs.append((g, np.asarray(m.mesh_vert[a:a + n], np.float64)))
            self.verts.append(vs)
        self.datum = np.full(2, -self.off[2], np.float64)         # origin height of a foot flat on the floor
        self.zdatum = np.zeros(2, np.float64)                     # lowest sole corner height of a foot flat on the floor
        self.rows: list[list[float]] = []
        self._res = np.zeros(6)

    def _sole_min_z(self) -> np.ndarray:
        d, out = self.env.data, np.zeros(2)
        for i, vs in enumerate(self.verts):
            out[i] = min(float((d.geom_xpos[g][2] + (d.geom_xmat[g].reshape(3, 3) @ v.T)[2]).min()) for g, v in vs)
        return out

    def _contact_slip(self) -> np.ndarray:
        """(2,) force-weighted planar speed at the floor contact points of each foot (0 if no contact force)."""
        m, d, env = self.env.model, self.env.data, self.env
        num, den = np.zeros(2), np.zeros(2)
        qvel, jac, f6 = env._qvel(), np.zeros((3, m.nv)), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != env._floor_geom and c.geom2 != env._floor_geom:
                continue
            other = c.geom2 if c.geom1 == env._floor_geom else c.geom1
            for k in (0, 1):
                if other in self.geoms[k]:
                    mujoco.mj_contactForce(m, d, i, f6)
                    mujoco.mj_jac(m, d, jac, None, c.pos, self.bodies[k])
                    num[k] += abs(f6[0]) * float(np.linalg.norm((jac @ qvel)[:2]))
                    den[k] += abs(f6[0])
        return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

    def update(self) -> None:
        torch, m, d, env = self._t, self.env.model, self.env.data, self.env
        mujoco.mj_kinematics(m, d)          # refresh the derived pose / velocity from the CURRENT qpos / qvel
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        pos = np.stack([d.xpos[b] for b in self.bodies]).astype(np.float64)
        quat = np.stack([d.xquat[b] for b in self.bodies]).astype(np.float64)
        vel, ang, vcom = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))
        for k, b in enumerate(self.bodies):
            # mjOBJ_XBODY = the body FRAME origin (d.xpos); mjOBJ_BODY would be the COM (xipos), which is what
            # Isaac's robot.data.body_lin_vel_w holds. flg_local = 0: world axes.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, self._res, 0)
            ang[k], vel[k] = self._res[:3], self._res[3:]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._res, 0)
            vcom[k] = self._res[3:]
        loaded = env._in_contact.astype(np.float64)

        def t(a):
            return torch.as_tensor(np.asarray(a, np.float64)[None])
        slip_f, z_low = _train_rewards.foot_sole_slip_speed(t(pos), t(quat), t(vel), t(ang), self.off, self.half)
        slip = float(_train_rewards.foot_slip(slip_f, t(loaded))[0])
        z_low = z_low[0].numpy().astype(np.float64)
        self.zdatum = np.where(loaded > 0, z_low, self.zdatum)    # the env datum: lowest sole corner at the last loaded step
        clr_c = z_low - self.zdatum
        drag = float(_train_rewards.foot_drag(slip_f, t(clr_c), t(loaded), self.h)[0])
        clr_sole = self._sole_min_z()
        drag_m = float(_train_rewards.foot_drag(slip_f, t(clr_sole), t(loaded), self.h)[0])
        self.datum = np.where(loaded > 0, pos[:, 2], self.datum)  # swing_foot_height datum: origin height at the last loaded step
        clr_o = pos[:, 2] - self.datum
        drag_o = float(_train_rewards.foot_drag(slip_f, t(clr_o), t(loaded), self.h)[0])
        soft = np.maximum(loaded, np.clip(1.0 - clr_o / self.thr, 0.0, 1.0))
        vcom_xy = np.linalg.norm(vcom[:, :2], axis=1)
        dragged = float(((loaded == 0) & (clr_o < self.thr) & (vcom_xy > 0.05)).sum())
        n_l = float(loaded.sum())
        self.rows.append([float((self._contact_slip() * loaded).sum()), slip,
                          float((np.linalg.norm(vel[:, :2], axis=1) * loaded).sum()), float((soft * vcom_xy).sum()),
                          drag, drag_m, dragged, n_l, slip / n_l if n_l else np.nan, float(clr_sole.min() * 1000.0),
                          *[float(v) for v in slip_f[0].numpy()], *[float(v) for v in clr_sole],
                          *[float(v) for v in clr_o], *[float(v) for v in loaded], *[float(v) for v in clr_c], drag_o])

    def series(self) -> np.ndarray:
        return np.asarray(self.rows, np.float64).reshape(-1, len(self.NAMES))


def foot_slip_summary(series: np.ndarray, sl: slice) -> dict:
    """Window means (mm/s) of a FootSlipProbe series: per-step sums over both feet (as the reward terms) and
    the mean over loaded feet (true_per_loaded / slip_per_loaded)."""
    s = series[sl]
    if len(s) < 10:
        return {}
    ix = {n: i for i, n in enumerate(FootSlipProbe.NAMES)}
    nl = s[:, ix["n_loaded"]]
    out = {k: float(1000.0 * s[:, ix[k]].mean()) for k in ("true", "slip", "old_origin", "old_isaac", "drag", "drag_mesh", "drag_origin")}
    ok = nl > 0
    out["true_per_loaded"] = float(1000.0 * (s[ok, ix["true"]] / nl[ok]).mean()) if ok.any() else float("nan")
    out["slip_per_loaded"] = float(1000.0 * np.nanmean(s[:, ix["slip_per_loaded"]])) if ok.any() else float("nan")
    out["dragged_frac"] = float(s[:, ix["dragged"]].mean() / 2.0)
    return out


def _window(n: int) -> slice:
    return slice(int(round(POSE_T0 / 0.02)), min(n, int(round(POSE_T1 / 0.02))))


def run_case(policy: Policy, config, cmd, delay: int, seed: int, rate: float, steps: int,
             collect: list | None = None, legacy_prev_slot: bool = False,
             legacy_imu: bool = False, grav_bias=(0.0, 0.0)) -> dict:
    """One MuJoCo episode with the Jetson limiter and `delay` extra steps of action delay.

    grav_bias = (x, y) is added to the observed projected gravity (slots 6:9) of every frame pushed
    into the policy's history, the way a mis-calibrated IMU mount would present it."""
    ec = config.train.env_config
    ec.backlash = {**(ec.backlash or {}), "enabled": False}      # same as the scratch hardware sims
    np.random.seed(seed)
    env = BipedEnvV2(history_size=policy.hist, use_history=True,
                     critic_has_privileged_info=True, env_config=ec, seed=seed)
    if legacy_imu:               # same as sim2real.py --legacy-imu
        from envs.mujoco.biped_env_v2 import PHYSICS_DT
        env.imu_acc_dt = PHYSICS_DT
    obs, _ = env.reset(seed=seed)
    eff = env.set_command(*cmd)
    if not np.allclose(eff, cmd, atol=1e-9):
        print(f"[policy_gate] command {tuple(cmd)} limited to ({eff[0]:.4f}, {eff[1]:.4f}, {eff[2]:.4f}) "
              f"by the config's command limit")

    # Jetson start(): history ZERO-filled, gait phase 0, previous action 0. Drop the reset frame
    # (random gait phase / command) and push the first frame normally, so history = 4 zero frames
    # + this frame, as on the robot.
    env.gait_phase = 0.0
    env._history.clear()
    for _ in range(policy.hist):
        env._history.append(np.zeros(FRAME, dtype=np.float32))
    ob0 = env._build_obs()
    o, crit = ob0["actor"].copy(), ob0["critic"].copy()
    ckpt = isinstance(policy, CkptPolicy)
    bias = np.array([grav_bias[0], grav_bias[1], 0.0], np.float32)
    biased = bool(np.any(bias))
    if biased:
        o[-FRAME:][GRAV_SLOT] += bias
        env._history[-1][GRAV_SLOT] += bias
    side, stem = leg_sides(env.model)
    leg_t, sep_t, jrel_t, gap_t, flat_t = [], [], [], [], []
    fp_off, fp_half = footprint_params(ec)
    probe = FootSlipProbe(env, fp_off, fp_half, h_drag=float(getattr(ec, "foot_drag_height_m", 0.005)),
                          thresh=float(getattr(ec, "contact_height_thresh_m", None) or 0.004))

    foot_ids = (env._r_foot_body, env._l_foot_body)
    ground_z = np.array([env.data.xpos[b][2] for b in foot_ids])
    cprev = np.ones(2, bool)                     # reset: both feet assumed down
    n_td, clear_max, ss_runs, ss_cur = 0, 0.0, [], 0

    prev = np.zeros(12)
    fifo = deque([np.zeros(12)] * (1 + delay), maxlen=1 + delay)
    raws, dA, gyro, vx, lim_n, fell = [], [], [], [], 0, None
    for i in range(steps):
        if collect is not None and i >= WARMUP_STEPS and i % 30 == 0:
            collect.append(o.copy())
        raw = policy(o, crit) if ckpt else policy(o)
        a = prev + np.clip(raw - prev, -rate, rate) if rate > 0 else raw   # rate <= 0 = no limiter (as on the Jetson)
        lim_n += int(np.max(np.abs(a - raw)) > 1e-6)
        raws.append(raw)
        dA.append(a - prev)
        prev = a
        fifo.append(a)
        obs, _, term, _, _ = env.step(fifo[0])
        o, crit = obs["actor"].copy(), obs["critic"].copy()
        # contact bookkeeping (env._in_contact = [right, left], force > 1 N)
        c = env._in_contact.copy()
        fz = np.array([env.data.xpos[b][2] for b in foot_ids])
        ground_z = np.where(c, fz, ground_z)
        if i >= WARMUP_STEPS:
            n_td += int((~cprev & c).sum())
            clear_max = max(clear_max, float(np.max(np.where(c, 0.0, fz - ground_z))))
            if c.sum() == 1:
                ss_cur += 1
            else:
                if ss_cur:
                    ss_runs.append(ss_cur)
                ss_cur = 0
        cprev = c
        # env.step() writes the DELAYED action into the new frame's previous_actions slot; the
        # Jetson feeds back the APPLIED one (a), so patch the newest frame and the history copy.
        if not legacy_prev_slot:     # --legacy-prev-slot keeps the delayed one (old mj_hw.py)
            o[-FRAME:][PREV_SLOT] = a
            env._history[-1][PREV_SLOT] = a
            env.prev_actions[:] = a
        if biased:                   # patched when pushed, so every stored history frame carries it
            o[-FRAME:][GRAV_SLOT] += bias
            env._history[-1][GRAV_SLOT] += bias
        leg_t.append(inter_leg_contact(env, side, stem))
        sep_t.append(foot_separations(env))
        gap_t.append(foot_footprint_gap(env, fp_off, fp_half))
        flat_t.append(foot_flatness(env))
        probe.update()
        jrel_t.append(np.asarray(env._jpos(), np.float64) - np.asarray(env._default_joint_pos, np.float64))
        gyro.append(np.linalg.norm(env.data.qvel[3:6]))
        vx.append(float((env._world_to_body_rot() @ env.data.qvel[:3])[0]))
        if term:
            fell = (i + 1) * 0.02
            break
    raws, dA = np.array(raws), np.array(dA)
    w = WARMUP_STEPS
    t_eval = max(len(raws) - w, 1) * 0.02
    step_hz = n_td / t_eval if len(raws) > w else np.nan
    ss_dur = float(np.mean(ss_runs)) * 0.02 if ss_runs else np.nan
    ac = np.nan
    if len(raws) > w + 10:
        r = raws[w:] - raws[w:].mean(axis=0)
        den = (r ** 2).sum(axis=0)
        num = (r[1:] * r[:-1]).sum(axis=0)
        good = den > 1e-12
        ac = float(np.mean(num[good] / den[good])) if good.any() else 0.0
    env.close()
    sl = _window(len(raws))
    pose = dict(sat=np.full(12, np.nan), jrel_deg=np.full(12, np.nan), leg_mean={k: np.nan for k in PAIR_CLASSES},
                leg_frac={k: np.nan for k in PAIR_CLASSES}, leg_total=np.nan, leg_total_frac=np.nan,
                sep_origin_min_mm=np.nan, sep_sole_mean_mm=np.nan, foot_gap_mean_mm=np.nan, foot_overlap_frac=np.nan,
                slip_true_mms=np.nan, drag_mms=np.nan, foot_roll_deg=np.nan, flat_contact_frac=np.nan)
    foot_series = probe.series()
    if sl.stop - sl.start >= 10:          # fell before the window -> stays nan (falls are reported separately)
        pose["sat"] = (np.abs(raws[sl]) > SAT_RAW).mean(axis=0)
        pose["jrel_deg"] = np.degrees(np.mean(jrel_t[sl], axis=0))
        tot = np.array([sum(c.values()) for c in leg_t[sl]])
        for k in PAIR_CLASSES:
            v = np.array([c[k] for c in leg_t[sl]])
            pose["leg_mean"][k], pose["leg_frac"][k] = float(v.mean()), float((v > 1.0).mean())
        pose["leg_total"], pose["leg_total_frac"] = float(tot.mean()), float((tot > 1.0).mean())
        seps = np.array(sep_t[sl]) * 1000.0
        pose["sep_origin_min_mm"], pose["sep_sole_mean_mm"] = float(seps[:, 0].min()), float(seps[:, 1].mean())
        gaps = np.array(gap_t[sl])
        pose["foot_gap_mean_mm"], pose["foot_overlap_frac"] = float(gaps.mean() * 1000.0), float((gaps <= 0.0).mean())
        fs = foot_slip_summary(foot_series, sl)
        pose["slip_true_mms"], pose["drag_mms"] = fs.get("true_per_loaded", np.nan), fs.get("drag", np.nan)
        rolls = np.array([f[0] for f in flat_t[sl]]); spreads = np.array([f[1] for f in flat_t[sl]])
        loaded = np.array([f[2] for f in flat_t[sl]])
        if loaded.any():
            pose["foot_roll_deg"] = float(np.abs(rolls[loaded]).mean())
            pose["flat_contact_frac"] = float((np.nan_to_num(spreads[loaded], nan=0.0) >= FLAT_SPREAD_MIN_MM).mean())
    return dict(fell=fell, lim=lim_n / len(raws), ac=ac, pose=pose,
                da=float(np.sqrt((dA[w:] ** 2).mean())) if len(dA) > w else np.nan,
                gyro=float(np.mean(gyro[w:])) if len(gyro) > w else np.nan,
                vx=float(np.mean(vx[w:])) if len(vx) > w else np.nan,
                step_hz=step_hz, ss=ss_dur, clear=clear_max * 1000.0, foot_series=foot_series, foot_window=sl)


def _nm(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------
# 3. Verdict
# ---------------------------------------------------------------------------

AC_MIN_DA = 0.1     # autocorr criterion only applies when rms applied delta-a is at least this


def _ac_ok(r: dict) -> bool:
    return r["da"] < AC_MIN_DA or r["ac"] > 0


def _verdict(rho_med: float, cases: dict) -> list[tuple[str, bool]]:
    """Checks over the cases that were run (a --cases filter skips the others)."""
    def sel(cmds, ds, f):
        return all(f(cases[(k, d)]) for k in cmds for d in ds if (k, d) in cases)
    both = ("cmd0", "walk")
    out = [("median spectral radius < 1.0" + (" (marginal: 1.0-1.2)" if 1.0 <= rho_med < 1.2 else ""),
            rho_med < 1.0),
           ("limiter fraction < 0.05 (d<=1)", sel(both, (0, 1), lambda r: r["lim"] < 0.05)),
           (f"raw lag-1 autocorr > 0 when rms da >= {AC_MIN_DA} (d<=1)", sel(both, (0, 1), _ac_ok)),
           ("no falls at cmd0 (d<=2)", sel(("cmd0",), (0, 1, 2), lambda r: r["falls"] == 0))]
    if any(k == "walk" for k, _ in cases):
        out.append(("walk vx >= 0.1 (d=0,1)", sel(("walk",), (0, 1), lambda r: r["vx"] >= 0.1)))
    # Pose sanity / inter-leg contact, evaluated at the smallest delay that was run (nan = fell early = FAIL)
    d0 = min((d for _, d in cases), default=None)
    if d0 is not None:
        if ("cmd0", d0) in cases and "sat_max" in cases[("cmd0", d0)]:
            sat = cases[("cmd0", d0)]["sat_max"]
            out.append((f"no saturated posture (max joint |raw|>{SAT_RAW} fraction < {SAT_MAX_FRAC}, cmd0 {POSE_T0:g}-{POSE_T1:g}s)",
                        bool(np.isfinite(sat) and sat < SAT_MAX_FRAC)))
        legs = [cases[(k, d0)]["leg_total"] for k in both if (k, d0) in cases and "leg_total" in cases[(k, d0)]]
        if legs:
            out.append((f"inter-leg contact < {LEG_CONTACT_MAX_N:g} N mean (legs not bracing on each other)",
                        bool(all(np.isfinite(x) and x < LEG_CONTACT_MAX_N for x in legs))))
        if ("cmd0", d0) in cases and "foot_roll_deg" in cases[("cmd0", d0)]:
            r0 = cases[("cmd0", d0)]
            out.append((f"feet flat at cmd0 (mean |sole roll| < {FOOT_ROLL_MAX_DEG:g} deg, contacts span >= "
                        f"{FLAT_SPREAD_MIN_MM:g} mm on >= {100 * FLAT_CONTACT_MIN_FRAC:.0f}% of loaded-foot steps: no edge stance)",
                        bool(np.isfinite(r0["foot_roll_deg"]) and r0["foot_roll_deg"] < FOOT_ROLL_MAX_DEG
                             and np.isfinite(r0["flat_contact_frac"]) and r0["flat_contact_frac"] >= FLAT_CONTACT_MIN_FRAC)))
    return out


def _case_pose(rs: list[dict]) -> dict:
    """Aggregate the per-seed pose/contact stats of one (cmd, d) case."""
    P = [r["pose"] for r in rs]
    with np.errstate(all="ignore"):
        sat = np.nanmean([p["sat"] for p in P], axis=0) if P else np.full(12, np.nan)
        jrel = np.nanmean([p["jrel_deg"] for p in P], axis=0) if P else np.full(12, np.nan)
    return dict(sat=sat, sat_max=float(np.max(sat)) if np.isfinite(sat).any() else float("nan"), jrel_deg=jrel,
                leg_mean={k: _nm([p["leg_mean"][k] for p in P]) for k in PAIR_CLASSES},
                leg_frac={k: _nm([p["leg_frac"][k] for p in P]) for k in PAIR_CLASSES},
                leg_total=_nm([p["leg_total"] for p in P]), leg_total_frac=_nm([p["leg_total_frac"] for p in P]),
                sep_origin_min_mm=_nm([p["sep_origin_min_mm"] for p in P]),
                sep_sole_mean_mm=_nm([p["sep_sole_mean_mm"] for p in P]),
                foot_gap_mean_mm=_nm([p["foot_gap_mean_mm"] for p in P]),
                foot_overlap_frac=_nm([p["foot_overlap_frac"] for p in P]),
                slip_true_mms=_nm([p["slip_true_mms"] for p in P]), drag_mms=_nm([p["drag_mms"] for p in P]),
                foot_roll_deg=_nm([p["foot_roll_deg"] for p in P]), flat_contact_frac=_nm([p["flat_contact_frac"] for p in P]))


def print_pose_report(cases: dict, out=print) -> None:
    """Pose-sanity / inter-leg-contact tables for every case that was run (informational + criteria)."""
    for (name, d), r in cases.items():
        out(f"\nPose / contact: {name} d={d}, window {POSE_T0:g}-{POSE_T1:g}s")
        if not np.isfinite(r["sat_max"]):
            out("  (fell before the window: no pose statistics)")
            continue
        out("  joint        " + " ".join(f"{n:>11s}" for n in JOINT_LABELS[:6]))
        out("  sat frac     " + " ".join(f"{v:11.2f}" for v in r["sat"][:6]))
        out("  pos_rel deg  " + " ".join(f"{v:11.1f}" for v in r["jrel_deg"][:6]))
        out("  joint        " + " ".join(f"{n:>12s}" for n in JOINT_LABELS[6:]))
        out("  sat frac     " + " ".join(f"{v:12.2f}" for v in r["sat"][6:]))
        out("  pos_rel deg  " + " ".join(f"{v:12.1f}" for v in r["jrel_deg"][6:]))
        out("  inter-leg normal force:  " + "  ".join(
            f"{k} {r['leg_mean'][k]:.2f} N ({100 * r['leg_frac'][k]:.0f}% >1N)" for k in PAIR_CLASSES))
        out(f"  inter-leg total {r['leg_total']:.2f} N mean, {100 * r['leg_total_frac']:.0f}% of steps > 1 N | "
            f"foot-origin lateral sep min {r['sep_origin_min_mm']:.0f} mm | sole-centre sep mean "
            f"{r['sep_sole_mean_mm']:.0f} mm")
        out(f"  foot footprint gap (the training foot_proximity quantity) mean {r['foot_gap_mean_mm']:.1f} mm, "
            f"{100 * r['foot_overlap_frac']:.0f}% of steps <= 0 (overlap)")
        out(f"  foot slip (report only, E25): true contact-point slip {r['slip_true_mms']:.1f} mm/s (force-weighted MuJoCo "
            f"contacts, mean over loaded feet) | foot_drag {r['drag_mms']:.1f} mm/s (training term, summed over feet)")
        out(f"  sole flatness: mean |roll| {r['foot_roll_deg']:.1f} deg of the loaded feet, contacts span >= "
            f"{FLAT_SPREAD_MIN_MM:g} mm on {100 * r['flat_contact_frac']:.0f}% of loaded-foot steps (edge stance ~0%)")


def gravity_bias_info(policy, config, cmd, delay, rate, legacy_prev_slot, legacy_imu, out=print) -> None:
    """Informational (no PASS/FAIL): +-GRAV_BIAS lateral projected-gravity bias, 1 seed, GRAV_SECONDS."""
    steps = int(round(GRAV_SECONDS / 0.02))
    res = []
    for sgn in (+1.0, -1.0):
        r = run_case(policy, config, cmd, delay, 0, rate, steps, legacy_prev_slot=legacy_prev_slot,
                     legacy_imu=legacy_imu, grav_bias=(0.0, sgn * GRAV_BIAS))
        res.append(f"{sgn * GRAV_BIAS:+.2f}: " + ("no fall" if r["fell"] is None else f"FELL at {r['fell']:.1f}s"))
    out(f"\nInfo (not a criterion): lateral gravity bias, cmd 0, d={delay}, 1 seed, {GRAV_SECONDS:g}s -> "
        + " | ".join(res))


SWEEP_SECONDS = 10.0


def sweep(policy, config, cmds, delays, seeds, seconds, rate, legacy_prev_slot=False,
          legacy_imu=False, out=print) -> list[dict]:
    """Per-command closed-loop sweep; returns one row dict per (cmd, delay) and prints a table."""
    steps = int(round(seconds / 0.02))
    rows = []
    for cmd in cmds:
        for d in delays:
            rs = [run_case(policy, config, (cmd, 0.0, 0.0), d, s, rate, steps,
                           legacy_prev_slot=legacy_prev_slot, legacy_imu=legacy_imu)
                  for s in range(seeds)]
            falls = sum(r["fell"] is not None for r in rs)
            row = dict(cmd=cmd, d=d, falls=falls, n=seeds, vx=_nm([r["vx"] for r in rs]),
                       step_hz=_nm([r["step_hz"] for r in rs]), ss=_nm([r["ss"] for r in rs]),
                       clear=_nm([r["clear"] for r in rs]), da=_nm([r["da"] for r in rs]),
                       lim=_nm([r["lim"] for r in rs]),
                       fell_t=_nm([r["fell"] for r in rs if r["fell"] is not None]))
            row["ratio"] = row["vx"] / cmd if cmd else float("nan")
            row["walks"] = bool(falls == 0 and row["ratio"] >= 0.6 and row["step_hz"] > 0.5)
            rows.append(row)
    out(f"{'cmd':>5s} {'d':>1s} {'falls':>5s} {'vx':>6s} {'vx/cmd':>6s} {'step Hz':>7s} {'SS s':>5s} "
        f"{'clr mm':>6s} {'rms da':>6s} {'lim':>5s}  verdict")
    for r in rows:
        out(f"{r['cmd']:5.2f} {r['d']:>1d} {r['falls']:>2d}/{r['n']:<2d} {r['vx']:6.3f} {r['ratio']:6.2f} "
            f"{r['step_hz']:7.2f} {r['ss']:5.2f} {r['clear']:6.1f} {r['da']:6.3f} {r['lim']:5.2f}  "
            + ("walks" if r["walks"] else "stands/stalls"))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-deploy gate for an exported policy bundle")
    p.add_argument("--bundle", default=None, help="exported/<name> (policy.onnx + deploy_config.json)")
    p.add_argument("--checkpoint", default=None,
                   help="raw checkpoints/<run>/step_N.pt (uses the run's config.yaml + obs scaler)")
    p.add_argument("--cmds", type=float, nargs="+", default=None,
                   help="SWEEP mode: forward-velocity commands (vy=wz=0), e.g. 0.05 0.1 0.15 0.2 0.3")
    p.add_argument("--delays", type=int, nargs="+", default=None,
                   help="extra action delays (steps); gate default 0..3, sweep default 0 only")
    p.add_argument("--trace", default=None, help="hardware trace .npz with a `frame` array")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seconds", type=float, default=None, help="episode length (default 8; sweep 10)")
    p.add_argument("--rate", type=float, default=0.5, help="limiter max |delta a| per step (0 = off)")
    p.add_argument("--cases", choices=("all", "cmd0", "walk"), default="all",
                   help="restrict closed-loop cases (verdict checks of skipped cases are ignored)")
    p.add_argument("--legacy-imu", action="store_true",
                   help="old Isaac accelerometer (motion part 4x too large), as sim2real.py --legacy-imu; "
                        "only for checkpoints trained before 2026-10-11")
    p.add_argument("--legacy-prev-slot", action="store_true",
                   help="policy sees the DELAYED action in its previous_actions slot (old mj_hw.py); "
                        "default is the Jetson semantics (applied action)")
    args = p.parse_args(argv)
    if bool(args.bundle) == bool(args.checkpoint):
        p.error("give exactly one of --bundle / --checkpoint")
    if args.seconds is None:
        args.seconds = SWEEP_SECONDS if args.cmds else 8.0

    if args.checkpoint:
        bundle = args.checkpoint
        policy, config = load_checkpoint_policy(bundle)
    else:
        bundle = args.bundle
        if not os.path.exists(bundle):
            bundle = os.path.join(_ROOT, bundle)
        policy = Policy(bundle)
        config = Config(_run_config_path(bundle))
        assert policy.obs_dim == policy.hist * FRAME,             f"bundle obs_dim {policy.obs_dim} != history_size {policy.hist} * {FRAME}"
        if int(config.train.history_size) != policy.hist:
            print(f"[gate] WARNING: run config history_size {config.train.history_size} != bundle {policy.hist}")
    steps = int(round(args.seconds / 0.02))
    print("[gate] sim caveats: backlash DISABLED (sim is optimistic); IMU accel = "
          + ("LEGACY 4x motion term" if args.legacy_imu else "step-averaged (post 2026-10-11)")
          + "; history zero-filled, gait phase 0 at start")
    print(f"[gate] {'checkpoint' if args.checkpoint else 'bundle'} {os.path.basename(os.path.normpath(bundle))} | rate {args.rate} | "
          f"{args.seeds} seeds x {args.seconds:g}s"
          + (" | LEGACY prev slot (delayed action)" if args.legacy_prev_slot else ""))

    if args.cmds:
        print(f"[sweep] cmds {args.cmds} | delays {args.delays or [0]} | step Hz = touchdowns of both feet per s "
              f"(env contact signal); SS = mean single-support run; clr = max swing foot height above last "
              f"stance height; walks = no fall, vx/cmd >= 0.6, step freq > 0.5 Hz", flush=True)
        sweep(policy, config, args.cmds, args.delays or [0], args.seeds, args.seconds, args.rate,
              args.legacy_prev_slot, args.legacy_imu)
        return 0

    cases, stand = {}, []
    for name, cmd in (("cmd0", (0.0, 0.0, 0.0)), ("walk", WALK_CMD)):
        if args.cases not in ("all", name):
            continue
        for d in (args.delays if args.delays is not None else DELAYS):
            rs = [run_case(policy, config, cmd, d, s, args.rate, steps,
                           collect=stand if (len(stand) == 0 and d == 0 and s == 0) else None,
                           legacy_prev_slot=args.legacy_prev_slot, legacy_imu=args.legacy_imu)
                  for s in range(args.seeds)]
            falls = [r["fell"] for r in rs if r["fell"] is not None]
            cases[(name, d)] = dict(falls=len(falls), fell_t=_nm(falls), lim=_nm([r["lim"] for r in rs]),
                                    ac=_nm([r["ac"] for r in rs]), da=_nm([r["da"] for r in rs]),
                                    gyro=_nm([r["gyro"] for r in rs]), vx=_nm([r["vx"] for r in rs]),
                                    **_case_pose(rs))
            print(f"[gate] {name} d={d} done", flush=True)

    # 1. offline gain
    rho_sim = [spectral_radius(policy, s) for s in stand[:10]]
    rho_tr = [spectral_radius(policy, s) for s in trace_states(args.trace, policy.hist)] if args.trace else []
    print("\nOffline gain: spectral radius of d(action)/d(prev_action), newest frame")
    for lab, v in (("MuJoCo standing", rho_sim), ("hardware trace", rho_tr)):
        if v:
            print(f"  {lab:16s} n={len(v):2d}  median {np.median(v):.2f}  max {np.max(v):.2f}")
    rho_all = rho_sim + rho_tr
    rho_med = float(np.median(rho_all)) if rho_all else float("nan")

    # 2. table
    print(f"\n{'cmd':5s} {'d':>1s} {'falls':>7s} {'fall t':>6s} {'lim':>5s} {'ac1':>6s} "
          f"{'rms da':>6s} {'|gyro|':>6s} {'vx':>6s}")
    for (name, d), r in cases.items():
        ft = f"{r['fell_t']:.1f}s" if r["falls"] else "-"
        vx = r["vx"] if name == "walk" else float("nan")
        ac = f"{r['ac']:6.2f}" if r["da"] >= AC_MIN_DA else "n/a(s)"
        print(f"{name:5s} {d:>1d} {r['falls']:>3d}/{args.seeds:<3d} {ft:>6s} {r['lim']:5.2f} {ac:>6s} "
              f"{r['da']:6.3f} {r['gyro']:6.2f} {vx:6.3f}")

    # 2b. pose sanity / inter-leg contact (same cases) + informational gravity-bias run
    print_pose_report(cases)
    if args.cases in ("all", "cmd0") and cases:
        gravity_bias_info(policy, config, (0.0, 0.0, 0.0), min(d for _, d in cases), args.rate,
                          args.legacy_prev_slot, args.legacy_imu)

    # 3. verdict
    checks = _verdict(rho_med, cases)
    print(f"\nGate (median spectral radius over all states = {rho_med:.2f}):")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    ok_all = all(ok for _, ok in checks)
    print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
