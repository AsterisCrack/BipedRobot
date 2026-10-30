"""record_reel.py -- real-time 4K videos of BipedRobotV2 in Isaac Lab (headless, offscreen).

One frame per policy step (50 Hz) written at 50 fps, so playback is real time no matter how slowly
the 4K frames render. Unlike play.py, a fallen robot is reset on its own (the env's auto-reset):
play.py calls env.reset() on any termination, which would respawn every robot at once.

Shots:
  crowd   static high camera framing the whole env grid
  follow  side view of env 0, a bit up, following it along its spawn heading

Usage (from BipedRobot/):
  # step-0 policy (fresh SAC actor, sampled like training), 300 robots
  python -m src.isaaclab.record_reel --task BipedRobotV2 --shot crowd --untrained --num_envs 300 \
      --checkpoint <run>/step_N.pt --out videos/reel/untrained.mp4
  # trained, random commands
  python -m src.isaaclab.record_reel --task BipedRobotV2 --shot crowd --num_envs 300 --checkpoint <ckpt>
  # one robot walking forward at 0.1 m/s
  python -m src.isaaclab.record_reel --task BipedRobotV2 --shot follow --num_envs 1 --vx 0.1 --checkpoint <ckpt>

  --vertical renders 2160x3840 instead of 3840x2160. --preview N writes PNG stills every N frames and
  no video (for framing the camera).
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record reel videos of the Biped Robot in Isaac Lab")
parser.add_argument("--task", type=str, default="BipedRobotV2")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Trained checkpoint. With --untrained only its config.yaml / obs layout is used.")
parser.add_argument("--config_path", type=str, default=None, help="Default: the checkpoint dir's config.yaml")
parser.add_argument("--untrained", action="store_true",
                    help="Fresh, randomly initialised actor sampled stochastically (what training step 0 runs), "
                         "on the training env (pushes, DR).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--shot", choices=("crowd", "follow"), default="crowd")
parser.add_argument("--vx", type=float, default=None, help="Fixed command for every env (default: random, as in play)")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--settle_s", type=float, default=None,
                    help="Simulated seconds before recording starts (default 0 untrained, 2 trained)")
parser.add_argument("--vertical", action="store_true", help="2160x3840 (9:16) instead of 3840x2160")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--preview", type=int, default=0, help="Write a PNG every N frames instead of a video")
parser.add_argument("--crf", type=int, default=14)
parser.add_argument("--focal_length", type=float, default=None,
                    help="Camera focal length, mm (aperture 20.955 mm wide). Default 18.15 (60 deg across the "
                         "width) landscape, 32.25 vertical (60 deg along the height)")
# crowd camera: spherical around the grid centre, distance in units of the grid half-diagonal.
# Defaults (landscape / vertical): dist 1.75 / 1.7, elev 38 / 50, shift 1.0 / 1.5.
parser.add_argument("--crowd_dist", type=float, default=None)
parser.add_argument("--crowd_elev", type=float, default=None, help="deg above the horizon")
parser.add_argument("--crowd_azim", type=float, default=-120.0, help="deg, camera position around the centre")
parser.add_argument("--crowd_shift", type=float, default=None,
                    help="m, aim this far from the grid centre toward the camera (perspective puts the centre low)")
# follow camera, metres
# 9:16 needs a steeper camera to keep the horizon out of frame (defaults landscape / vertical: 0.9 / 0.7, 0.55 / 0.62)
parser.add_argument("--follow_dist", type=float, default=None, help="sideways distance from the robot")
parser.add_argument("--follow_height", type=float, default=None, help="camera height above the ground")
parser.add_argument("--follow_lookat_z", type=float, default=0.14)
parser.add_argument("--follow_lead", type=float, default=0.0, help="look this far ahead of the robot")
parser.add_argument("--follow_tau", type=float, default=0.5, help="camera smoothing time constant, s")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
for _k, _land, _vert in (("crowd_dist", 1.75, 1.7), ("crowd_elev", 38.0, 50.0), ("crowd_shift", 1.0, 1.5),
                         ("follow_dist", 0.9, 0.7), ("follow_height", 0.55, 0.62)):
    if getattr(args_cli, _k) is None:
        setattr(args_cli, _k, _vert if args_cli.vertical else _land)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_trl_root = os.path.join(_project_root, "torch_rl_algorithms")
if _trl_root not in sys.path:
    sys.path.insert(0, _trl_root)

import imageio.v2 as imageio
import numpy as np
import torch

from torch_rl_algorithms.algorithms.sac.model import SAC
from utils import Config
from utils.residual_home import freeze_residual_map
from src.isaaclab.common import BaseIsaacLabWrapper
from src.isaaclab.env_config import build_env_cfg
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg

BipedEnvCfg = {"BipedRobot": BipedRobotEnvCfg, "BipedRobotV2": BipedRobotV2EnvCfg}[args_cli.task]


class Wrapper(BaseIsaacLabWrapper):
    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        return self._process_obs(obs), terminated, truncated


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    cfg_path = args_cli.config_path or os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "config.yaml")
    config = Config(cfg_path)
    freeze_residual_map(config, warn_unfrozen=True)
    print(f"[reel] config={cfg_path}  untrained={args_cli.untrained}  shot={args_cli.shot}")

    env_cfg = build_env_cfg(config, BipedEnvCfg, training=args_cli.untrained, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if not args_cli.untrained:
        env_cfg.episode_length_s = 1.0e6        # no synchronized time-out respawn mid-video
    if args_cli.shot == "follow":
        # spawn at the env origin facing +x
        env_cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    width, height = (2160, 3840) if args_cli.vertical else (3840, 2160)
    env_cfg.viewer.resolution = (width, height)
    env_cfg.viewer.origin_type = "world"

    env = BipedEnv(cfg=env_cfg, render_mode="rgb_array")
    wrapped = Wrapper(env, config)
    device = env.device

    model = SAC(env=wrapped, model_path=None if args_cli.untrained else args_cli.checkpoint, device=device,
                config=config, use_history=config.train.use_history, history_size=config.train.history_size)
    actor = model.model.actor
    if not args_cli.untrained:
        wrapped.load(args_cli.checkpoint)
        wrapped.set_eval_mode()
        model.model.eval()
    # untrained: the obs scalers start fresh and keep updating, as they do at training step 0

    if args_cli.vx is not None:
        env.command_generator.set_manual((args_cli.vx, args_cli.vy, args_cli.wz))

    def act(obs):
        with torch.no_grad():
            if args_cli.untrained:
                return actor.forward(obs["actor"]).rsample()
            return model.step(obs)

    # --- camera
    step_dt = env.step_dt
    origins = env.scene.env_origins.cpu().numpy()
    cam_state = {}

    def set_camera(first=False):
        if args_cli.shot == "crowd":
            if first:
                c = origins.mean(axis=0)
                r = float(np.linalg.norm(origins[:, :2] - c[:2], axis=1).max()) + 0.5
                d = args_cli.crowd_dist * r
                el, az = math.radians(args_cli.crowd_elev), math.radians(args_cli.crowd_azim)
                look = c + args_cli.crowd_shift * np.array([math.cos(az), math.sin(az), 0.0])
                eye = look + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
                env.sim.set_camera_view(eye.tolist(), look.tolist())
                print(f"[reel] crowd camera eye={np.round(eye, 2)} lookat={np.round(look, 2)} (grid radius {r:.2f} m)")
            return
        p = env.robot.data.root_pos_w[0].cpu().numpy()
        if first:
            psi = _yaw(env.robot.data.root_quat_w[0].cpu().numpy())
            cam_state["f"] = np.array([math.cos(psi), math.sin(psi), 0.0])
            cam_state["l"] = np.array([-math.sin(psi), math.cos(psi), 0.0])
            cam_state["p"] = p[:2].copy()
        a = 1.0 - math.exp(-step_dt / args_cli.follow_tau)
        cam_state["p"] += a * (p[:2] - cam_state["p"])
        ground = np.array([cam_state["p"][0], cam_state["p"][1], 0.0])
        look = ground + cam_state["f"] * args_cli.follow_lead + np.array([0.0, 0.0, args_cli.follow_lookat_z])
        # camera on the robot's right, looking left: it walks left-to-right across the frame
        eye = ground - cam_state["l"] * args_cli.follow_dist + np.array([0.0, 0.0, args_cli.follow_height])
        env.sim.set_camera_view(eye.tolist(), look.tolist())

    obs = wrapped.start()
    settle_s = args_cli.settle_s if args_cli.settle_s is not None else (0.0 if args_cli.untrained else 2.0)
    for _ in range(int(round(settle_s / step_dt))):
        obs, _, _ = wrapped.step(act(obs))
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera(stage.GetPrimAtPath(env_cfg.viewer.cam_prim_path))
    # Kit's viewport camera lives in the session layer: an edit in the root layer is a weaker opinion and is ignored
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        cam.GetFocalLengthAttr().Set(args_cli.focal_length or (32.25 if args_cli.vertical else 18.15))
    set_camera(first=True)
    # The RTX renderer returns black/half-loaded frames at first (and streams the ground material).
    for _ in range(60):
        env.render()
    print(f"[reel] camera focal {cam.GetFocalLengthAttr().Get():.2f} mm, aperture "
          f"{cam.GetHorizontalApertureAttr().Get():.3f} x {cam.GetVerticalApertureAttr().Get():.3f} mm", flush=True)

    n_frames = int(round(args_cli.seconds / step_dt))
    fps = 1.0 / step_dt
    out = args_cli.out or os.path.join("videos", "reel", f"{args_cli.shot}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = None
    if not args_cli.preview:
        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                                    macro_block_size=8,
                                    output_params=["-crf", str(args_cli.crf), "-preset", "slow",
                                                   "-movflags", "+faststart"])
    print(f"[reel] {n_frames} frames at {fps:.0f} fps, {width}x{height} -> "
          f"{out if writer else 'previews next to ' + out}")

    v = torch.linalg.norm(env.commands[:, :2], dim=1)
    print(f"[reel] commands at start: |v_xy| min {v.min():.3f} mean {v.mean():.3f} m/s, "
          f"{int((v < 0.05).sum())}/{env.num_envs} envs below 0.05 m/s", flush=True)
    x0 = env.robot.data.root_pos_w[:, :2].clone()
    resets = 0
    for i in range(n_frames):
        obs, term, trunc = wrapped.step(act(obs))
        resets += int((term | trunc).sum())
        set_camera()
        frame = env.render()        # the only render this step (no RTX sensors, so step() does not)
        if writer is not None:
            writer.append_data(frame)
        elif i % args_cli.preview == 0:
            imageio.imwrite(out.replace(".mp4", f"_f{i:04d}.png"), frame)
        if i % 100 == 0:
            print(f"[reel] frame {i}/{n_frames}  resets so far {resets}", flush=True)

    if writer is not None:
        writer.close()
    if env.num_envs == 1:
        d = (env.robot.data.root_pos_w[0, :2] - x0[0]).cpu().numpy()
        print(f"[reel] env 0 travelled {np.linalg.norm(d):.3f} m in {args_cli.seconds:.1f} s "
              f"-> {np.linalg.norm(d) / args_cli.seconds:.3f} m/s; resets {resets}", flush=True)
    else:
        print(f"[reel] resets during recording: {resets} over {env.num_envs} envs", flush=True)
    print(f"[reel] done: {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
