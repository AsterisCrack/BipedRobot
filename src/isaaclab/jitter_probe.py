"""Jitter probe: is the chatter in the network or in the simulator?

Runs a checkpoint deterministically, headless, on flat terrain, with one fixed command, and
reports per-joint chatter numbers. Toggle the simulator suspects to see what each contributes:

    python src/isaaclab/jitter_probe.py --task BipedRobotV2 --checkpoint <run>/step_N.pt
    python src/isaaclab/jitter_probe.py ... --no-backlash --no-action-noise --alpha 0.2

Metrics (averaged over envs, after a 2 s settle):
  da_rms      rms change of the policy action per 20 ms step, on the [-1, 1] scale (network chatter)
  dq_hf       fraction of joint-velocity power above 8 Hz (what you see as buzzing)
  sat         fraction of servo steps at the duty ceiling
  wb_rms      base angular velocity rms [rad/s] (body wobble)
  vx          mean forward speed achieved [m/s]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="BipedRobotV2")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--config_path", type=str, default=None, help="defaults to the checkpoint dir's config.yaml")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--cmd", type=float, nargs=3, default=(0.1, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
parser.add_argument("--no-backlash", action="store_true")
parser.add_argument("--no-action-noise", action="store_true")
parser.add_argument("--alpha", type=float, default=None, help="override action_filter_alpha")
parser.add_argument("--legacy-imu", action="store_true",
                    help="old 4x accelerometer, for checkpoints trained before the StepAveragedImu fix")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, _root)
# Appended, not prepended: torch_rl_algorithms has its own top-level `config` module, which
# would shadow this repo's `config` package.
sys.path.append(os.path.join(_root, "torch_rl_algorithms"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_rl_algorithms.algorithms.sac.model import SAC  # noqa: E402
from utils import Config  # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402
from src.isaaclab.common import BaseIsaacLabWrapper  # noqa: E402
from src.isaaclab.env_config import build_env_cfg  # noqa: E402
from envs.isaaclab.biped_env import BipedEnv  # noqa: E402
from envs.isaaclab.biped_env_cfg import BipedEnvCfg, BipedRobotV2EnvCfg  # noqa: E402


class Wrapper(BaseIsaacLabWrapper):
    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        obs, rew, term, trunc, info = self.env.step(actions)
        return self._process_obs(obs), rew, term, trunc, info


def main():
    ckpt = os.path.abspath(args.checkpoint)
    cfg_path = args.config_path or os.path.join(os.path.dirname(ckpt), "config.yaml")
    config = Config(cfg_path)
    freeze_residual_map(config, warn_unfrozen=True)
    config.train.use_rough_terrain = False

    env_cfg = build_env_cfg(config, BipedRobotV2EnvCfg if args.task == "BipedRobotV2" else BipedEnvCfg,
                            training=False, num_envs=args.num_envs)
    bv = env_cfg.commands["base_velocity"]
    vx, vy, wz = args.cmd
    bv["ranges"].update(lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz))
    bv["limit_shape"] = "box"   # the ranges are collapsed onto the probe command: an ellipse would scale it
    for k in ("rel_standing_envs", "rel_heading_envs", "rel_turn_in_place_envs"):
        if k in bv:
            bv[k] = 0.0
    bv["resampling_time_range"] = (1e6, 1e6)
    if args.no_backlash:
        for a in env_cfg.scene.robot.actuators.values():
            if hasattr(a, "backlash_enabled"):
                a.backlash_enabled, a.backlash_nominal_rad = False, 0.0
    if args.no_action_noise:
        env_cfg.action_noise_model = None
    if args.alpha is not None:
        env_cfg.action_filter_alpha = args.alpha
    if args.legacy_imu:
        env_cfg.imu_step_averaged = False

    env = BipedEnv(cfg=env_cfg)
    wenv = Wrapper(env, config)
    model = SAC(env=wenv, model_path=ckpt, device=env.device, config=config,
                use_history=config.train.use_history, history_size=config.train.history_size)
    wenv.load(ckpt)
    wenv.set_eval_mode()
    getattr(model, "model", model).eval()

    obs = wenv.start()
    n = int(args.seconds / env.step_dt)
    settle = int(2.0 / env.step_dt)
    acts, dq, wb, vxs, sat = [], [], [], [], []
    with torch.no_grad():
        for i in range(n):
            a = model.step(obs)
            obs, _, term, trunc, _ = wenv.step(a)
            if i < settle:
                continue
            acts.append(env.actions.clone())
            dq.append(env.robot.data.joint_vel.clone())
            wb.append(torch.norm(env.robot.data.root_ang_vel_b, dim=-1))
            vxs.append(env.robot.data.root_lin_vel_b[:, 0].clone())
            if env.bam_actuators:
                sat.append(env.bam_actuators[0].duty_saturated.float().mean(dim=1))
    A = torch.stack(acts)                      # [T, N, J]
    Q = torch.stack(dq).cpu().numpy()
    da = (A[1:] - A[:-1]).pow(2).mean(dim=(0, 1)).sqrt().cpu().numpy()          # per joint
    spec = np.abs(np.fft.rfft(Q - Q.mean(0), axis=0)) ** 2                     # [F, N, J]
    f = np.fft.rfftfreq(Q.shape[0], d=env.step_dt)
    hf = (spec[f > 8.0].sum(0) / np.maximum(spec[f > 0.0].sum(0), 1e-12)).mean(0)
    tag = " ".join(f"{k}" for k, on in (("no-backlash", args.no_backlash), ("no-action-noise", args.no_action_noise),
                                        (f"alpha={args.alpha}", args.alpha is not None), ("legacy-imu", args.legacy_imu)) if on) or "baseline"
    print(f"\nPROBE [{tag}] cmd={tuple(args.cmd)}  {A.shape[1]} envs x {A.shape[0]} steps")
    print(f"PROBE   da_rms mean {da.mean():.3f}   dq_hf mean {hf.mean():.2f}   "
          f"sat {torch.stack(sat).mean().item() if sat else float('nan'):.3f}   "
          f"wb_rms {torch.stack(wb).pow(2).mean().sqrt().item():.3f}   vx {torch.stack(vxs).mean().item():+.3f}")
    print("PROBE   per joint (da_rms / dq_hf):")
    for j, name in enumerate(env.robot.joint_names):
        print(f"PROBE     {name:22s} {da[j]:.3f} / {hf[j]:.2f}")
    sys.stdout.flush()
    os._exit(0)   # Isaac's shutdown can hang for minutes; results are already printed


if __name__ == "__main__":
    main()
