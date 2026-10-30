"""record_reel.py -- real-time 4K video of one BipedRobotV2 walking in MuJoCo (offscreen).

The MuJoCo counterpart of src/isaaclab/record_reel.py's follow shot: side view a bit up, following the
robot along its spawn heading, one frame per 50 Hz policy step written at 50 fps (real time). The policy
runs exactly as in sim2real.py (deterministic, the checkpoint's obs scaler). Only the looks change: a
copy of the robot XML in a temp dir gets a gradient sky and a 10 cm checker floor (no physics change).

Usage (from BipedRobot/):
  python -m src.mujoco.record_reel --checkpoint <run>/step_N.pt --vx 0.1 --out videos/reel/mujoco.mp4
  --vertical renders 2160x3840 instead of 3840x2160; --preview N writes PNG stills every N frames.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# sim2real first: it imports config / utils before putting torch_rl_algorithms (which shadows them) on sys.path
from src.mujoco.sim2real import _EnvSpec, _load_obs_scaler, _model_class, _normalize_obs
from envs.mujoco.biped_env_v2 import _XML_DEFAULT, STEP_DT, BipedEnvV2
from utils import Config
from utils.residual_home import freeze_residual_map

_VISUAL = """
  <visual>
    <global offwidth="3840" offheight="3840"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>"""
_ASSETS = """
    <texture name="reel_sky" type="skybox" builtin="gradient" rgb1="0.62 0.76 0.92" rgb2="0.96 0.97 1.0" width="512" height="3072"/>
    <texture name="reel_grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.36" rgb2="0.24 0.26 0.30" width="512" height="512" mark="edge" markrgb="0.42 0.44 0.48"/>
    <material name="reel_floor" texture="reel_grid" texrepeat="5 5" texuniform="true" reflectance="0.03"/>"""


def _dressed_xml(src: str) -> str:
    """Copy of the robot XML with a sky and a checker floor; meshdir made absolute so it loads from a temp dir."""
    text = open(src, encoding="utf-8").read()
    mesh_dir = os.path.dirname(os.path.abspath(src)).replace("\\", "/")
    text, n = re.subn(r"<compiler([^>]*?)\s*/>", rf'<compiler\1 meshdir="{mesh_dir}" />{_VISUAL}', text, count=1)
    assert n == 1, "no <compiler/> element"
    text, n = re.subn(r"<asset>", "<asset>" + _ASSETS, text, count=1)
    assert n == 1, "no <asset> element"
    text, n = re.subn(r'(<geom name="floor"[^>]*?)rgba="[^"]*"', r'\1material="reel_floor"', text, count=1)
    assert n == 1, "floor geom not found"
    out = os.path.join(tempfile.mkdtemp(prefix="reel_mjcf_"), "robot_v2.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


def main():
    p = argparse.ArgumentParser(description="Record a 4K MuJoCo reel video of BipedRobotV2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_path", default=None, help="Default: the checkpoint dir's config.yaml")
    p.add_argument("--vx", type=float, default=0.1)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--settle_s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--out", default=os.path.join("videos", "reel", "mujoco.mp4"))
    p.add_argument("--preview", type=int, default=0)
    p.add_argument("--crf", type=int, default=14)
    # same geometry as the Isaac follow shot (defaults landscape / vertical: 0.9 / 0.7, 0.55 / 0.62)
    p.add_argument("--follow_dist", type=float, default=None)
    p.add_argument("--follow_height", type=float, default=None)
    p.add_argument("--follow_lookat_z", type=float, default=0.14)
    p.add_argument("--follow_tau", type=float, default=0.5)
    args = p.parse_args()
    if args.follow_dist is None:
        args.follow_dist = 0.7 if args.vertical else 0.9
    if args.follow_height is None:
        args.follow_height = 0.62 if args.vertical else 0.55
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_path = args.config_path or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "config.yaml")
    config = Config(cfg_path)
    freeze_residual_map(config, warn_unfrozen=True)
    env_conf = getattr(config.train, "env_config", None)
    env = BipedEnvV2(xml_path=_dressed_xml(_XML_DEFAULT), render_mode=None,
                     history_size=config.train.history_size, use_history=config.train.use_history,
                     critic_has_privileged_info=(getattr(config.train, "critic_obs", "privileged") == "privileged"),
                     env_config=env_conf, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_class(config)(env=_EnvSpec(env, device), model_path=args.checkpoint, device=device,
                                 config=config, use_history=config.train.use_history,
                                 history_size=config.train.history_size)
    getattr(model, "model", model).eval()
    normalize = getattr(config.train, "normalize_obs", False)
    scaler = _load_obs_scaler(args.checkpoint, device) if normalize else None

    def act(obs):
        a = _normalize_obs(obs["actor"], scaler, "actor", device) if normalize else \
            torch.from_numpy(obs["actor"]).float().unsqueeze(0).to(device)
        c = obs.get("critic", obs["actor"])
        c = _normalize_obs(c, scaler, "critic", device) if normalize else \
            torch.from_numpy(c).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model.step({"actor": a, "critic": c})
        return np.asarray(action.cpu().numpy() if isinstance(action, torch.Tensor) else action).squeeze()

    obs, _ = env.reset(seed=args.seed)
    eff = env.set_command(args.vx, args.vy, args.wz)
    print(f"[reel] config={cfg_path}  command={np.round(eff, 3)}")
    for _ in range(int(round(args.settle_s / STEP_DT))):
        obs, _, term, _, _ = env.step(act(obs))
        assert not term, "robot fell while settling"

    width, height = (2160, 3840) if args.vertical else (3840, 2160)
    # vertical field of view matching the Isaac camera: 60 deg across the long side
    env.model.vis.global_.fovy = 60.0 if args.vertical else math.degrees(2 * math.atan(math.tan(math.radians(30)) * 9 / 16))
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    base = env._base_body_id
    w, x, y, z = env.data.xquat[base]
    psi = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cam.azimuth = math.degrees(psi) + 90.0            # look along the robot's left: it walks left-to-right
    cam.elevation = -math.degrees(math.atan2(args.follow_height - args.follow_lookat_z, args.follow_dist))
    cam.distance = math.hypot(args.follow_dist, args.follow_height - args.follow_lookat_z)
    p_cam = env.data.xpos[base][:2].copy()
    alpha = 1.0 - math.exp(-STEP_DT / args.follow_tau)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None
    if not args.preview:
        writer = imageio.get_writer(args.out, fps=1.0 / STEP_DT, codec="libx264", quality=None,
                                    pixelformat="yuv420p", macro_block_size=8,
                                    output_params=["-crf", str(args.crf), "-preset", "slow", "-movflags", "+faststart"])
    n_frames = int(round(args.seconds / STEP_DT))
    print(f"[reel] {n_frames} frames at {1 / STEP_DT:.0f} fps, {width}x{height}")
    x0 = env.data.xpos[base][:2].copy()
    falls = 0
    for i in range(n_frames):
        obs, _, term, _, _ = env.step(act(obs))
        if term:
            falls += 1
            print(f"[reel] WARNING: fell at frame {i}; resetting")
            obs, _ = env.reset()
            env.set_command(args.vx, args.vy, args.wz)
        p_cam += alpha * (env.data.xpos[base][:2] - p_cam)
        cam.lookat[:] = (p_cam[0], p_cam[1], args.follow_lookat_z)
        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()
        if writer is not None:
            writer.append_data(frame)
        elif i % args.preview == 0:
            imageio.imwrite(args.out.replace(".mp4", f"_f{i:04d}.png"), frame)
        if i % 100 == 0:
            print(f"[reel] frame {i}/{n_frames}", flush=True)
    if writer is not None:
        writer.close()
    d = np.linalg.norm(env.data.xpos[base][:2] - x0)
    print(f"[reel] travelled {d:.3f} m in {args.seconds:.1f} s -> {d / args.seconds:.3f} m/s; falls {falls}")
    print(f"[reel] done: {args.out}")
    renderer.close()


if __name__ == "__main__":
    main()
