"""sim2real.py — Run an Isaac Lab-trained policy in MuJoCo (no Isaac Lab dependency).

Usage:
    # Auto-detect latest checkpoint:
    python src/mujoco/sim2real.py

    # Specify checkpoint and walk forward:
    python src/mujoco/sim2real.py --checkpoint checkpoints/BipedV2_SAC_xxx/model_50000.pt \\
        --vx 0.3

    # Headless evaluation (no window):
    python src/mujoco/sim2real.py --checkpoint <path> --headless --num_episodes 5

Arguments:
    --checkpoint     Path to .pt checkpoint. If omitted, latest in checkpoints/ is used.
    --config_path    Path to the training config YAML (default: config/config.yaml).
    --task           Robot variant (default: BipedRobotV2 — only V2 supported here).
    --seed           Random seed (default: 42).
    --headless       Disable the MuJoCo viewer window.
    --num_episodes   Episodes to run before exiting (0 = run forever, default: 0).
    --render_fps     Target render FPS when not headless (default: 50).
    --vx             Fixed forward velocity command m/s (default: random each episode).
    --vy             Fixed lateral velocity command m/s (default: random).
    --wz             Fixed yaw-rate command rad/s (default: random).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Project root on sys.path so imports resolve from any working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config.schema import ModelType
from envs.mujoco.biped_env_v2 import BipedEnvV2
from utils import Config

# RL algorithm imports
_TRL = os.path.join(_ROOT, "torch_rl_algorithms")
if _TRL not in sys.path:
    sys.path.insert(0, _TRL)

from torch_rl_algorithms.algorithms.sac.model  import SAC
from torch_rl_algorithms.algorithms.ddpg.model import DDPG
from torch_rl_algorithms.algorithms.d4pg.model import D4PG
from torch_rl_algorithms.algorithms.mpo.model  import MPO
from torch_rl_algorithms.algorithms.ppo.model  import PPO

try:
    from torch_rl_algorithms.algorithms.utils import RunningMeanStd
    _HAS_RMS = True
except ImportError:
    _HAS_RMS = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="MuJoCo sim-to-real inference for BipedRobotV2")
    p.add_argument("--checkpoint",   type=str,   default=None,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--config_path",  type=str,   default="config/config.yaml",
                   help="Training config YAML.")
    p.add_argument("--task",         type=str,   default="BipedRobotV2")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--headless",     action="store_true", default=False,
                   help="Run without rendering window.")
    p.add_argument("--num_episodes", type=int,   default=0,
                   help="Episodes to run (0 = infinite).")
    p.add_argument("--render_fps",   type=int,   default=50)
    p.add_argument("--vx",  type=float, default=None, help="Fixed vx command (m/s).")
    p.add_argument("--vy",  type=float, default=None, help="Fixed vy command (m/s).")
    p.add_argument("--wz",  type=float, default=None, help="Fixed wz command (rad/s).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers (copied from play.py)
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path: str | None, config) -> str | None:
    """Return explicit path, or auto-detect latest checkpoint under checkpoints/."""
    if path and os.path.exists(path):
        return path

    model_name = getattr(config.train, "model_name", "")
    roots = [
        os.path.join(_ROOT, "checkpoints"),
        os.path.join(_ROOT, "checkpoints", model_name),
    ]
    for ckpt_root in roots:
        if not os.path.isdir(ckpt_root):
            continue
        dirs = sorted(
            [os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root)
             if os.path.isdir(os.path.join(ckpt_root, d))],
            key=os.path.getmtime, reverse=True,
        )
        for d in dirs:
            files = [f for f in os.listdir(d) if f.endswith(".pt")]
            if not files:
                continue

            def _step(fn):
                try:
                    return int(fn.split("_")[1].replace(".pt", ""))
                except Exception:
                    return 0

            files.sort(key=_step, reverse=True)
            ckpt = os.path.join(d, files[0])
            print(f"[sim2real] Auto-detected checkpoint: {ckpt}")
            return ckpt

    return None


# ---------------------------------------------------------------------------
# Model selection (same pattern as play.py)
# ---------------------------------------------------------------------------

def _model_class(config):
    mt = config.train.model
    mapping = {
        ModelType.SAC: SAC, ModelType.DDPG: DDPG, ModelType.D4PG: D4PG,
        ModelType.MPO: MPO, ModelType.PPO:  PPO,
    }
    if mt in mapping:
        return mapping[mt]
    s = str(mt).lower()
    for k, v in [("sac", SAC), ("ddpg", DDPG), ("d4pg", D4PG), ("mpo", MPO), ("ppo", PPO)]:
        if k in s:
            return v
    raise ValueError(f"Unknown model type: {mt}")


# ---------------------------------------------------------------------------
# Obs-scaler loading (saved by BaseIsaacLabWrapper.save())
# ---------------------------------------------------------------------------

def _load_obs_scaler(checkpoint_path: str, device: torch.device):
    """Load RunningMeanStd obs scaler if it was saved alongside the checkpoint."""
    if not _HAS_RMS:
        return None
    scaler_path = checkpoint_path.rstrip(".pt") + "_obs_scalers.pt"
    # Also try the exact pattern used by common.py: path + "_obs_scalers.pt"
    alt_path = checkpoint_path + "_obs_scalers.pt"
    for sp in (scaler_path, alt_path):
        if os.path.exists(sp):
            state = torch.load(sp, map_location=device)
            print(f"[sim2real] Loaded obs scaler from {sp}")
            return state   # dict: {"actor": state_dict, "critic": state_dict}
    return None


def _normalize_obs(obs_np: np.ndarray, scaler_state: dict, key: str,
                   device: torch.device) -> torch.Tensor:
    """Normalise a single observation array using a pre-loaded RunningMeanStd state."""
    t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    if scaler_state is None or key not in scaler_state:
        return t
    rms = RunningMeanStd(shape=obs_np.shape, device=device)
    rms.load_state_dict(scaler_state[key])
    return rms.normalize(t)


# ---------------------------------------------------------------------------
# Minimal env-spec shim (lets model constructors inspect spaces without Isaac Lab)
# ---------------------------------------------------------------------------

class _EnvSpec:
    """Lightweight shim that exposes spaces and device to RL model constructors."""
    def __init__(self, env: BipedEnvV2, device: torch.device):
        self.device           = device
        self.num_envs         = 1
        self.observation_space = env.observation_space
        self.action_space      = env.action_space


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ config
    cfg_path = args.config_path
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        print(f"[sim2real] Config not found: {args.config_path}")
        sys.exit(1)
    config = Config(cfg_path)

    # ------------------------------------------------------------------ checkpoint
    ckpt = _resolve_checkpoint(args.checkpoint, config)
    if ckpt is None:
        print("[sim2real] No checkpoint found. Use --checkpoint <path>.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sim2real] Device: {device}")

    # ------------------------------------------------------------------ env
    env_conf = getattr(config.train, "env_config", None)
    render_mode = None if args.headless else "human"

    env = BipedEnvV2(
        render_mode=render_mode,
        history_size=config.train.history_size,
        use_history=config.train.use_history,
        critic_has_privileged_info=(
            getattr(config.train, "critic_obs", "privileged") == "privileged"
        ),
        env_config=env_conf,
        seed=args.seed,
    )

    # ------------------------------------------------------------------ model
    ModelClass = _model_class(config)
    env_spec   = _EnvSpec(env, device)
    model = ModelClass(
        env=env_spec,
        model_path=ckpt,
        device=device,
        config=config,
        use_history=config.train.use_history,
        history_size=config.train.history_size,
    )
    model_obj = getattr(model, "model", model)
    if hasattr(model_obj, "eval"):
        model_obj.eval()
    print(f"[sim2real] Loaded {type(model).__name__} from {ckpt}")

    # ------------------------------------------------------------------ scaler
    normalize_obs = getattr(config.train, "normalize_obs", False)
    scaler_state  = _load_obs_scaler(ckpt, device) if normalize_obs else None

    # ------------------------------------------------------------------ helpers
    def _get_action(obs_dict: dict) -> np.ndarray:
        actor_np = obs_dict["actor"]
        if normalize_obs:
            actor_t = _normalize_obs(actor_np, scaler_state, "actor", device)
        else:
            actor_t = torch.from_numpy(actor_np).float().unsqueeze(0).to(device)

        # Build obs in whatever format model.step() expects
        if isinstance(env.observation_space, type(env.observation_space)) and hasattr(
            env.observation_space, "spaces"
        ):
            critic_np = obs_dict.get("critic", obs_dict["actor"])
            if normalize_obs:
                critic_t = _normalize_obs(critic_np, scaler_state, "critic", device)
            else:
                critic_t = torch.from_numpy(critic_np).float().unsqueeze(0).to(device)
            obs_in = {"actor": actor_t, "critic": critic_t}
        else:
            obs_in = actor_t

        with torch.no_grad():
            action = model.step(obs_in)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return np.asarray(action).squeeze()

    # ------------------------------------------------------------------ loop
    episode   = 0
    step_time = 1.0 / args.render_fps

    print("[sim2real] Starting. Press Ctrl-C to quit.")
    print(f"           Command: vx={args.vx}  vy={args.vy}  wz={args.wz}  "
          f"({'fixed' if args.vx is not None else 'random'} per episode)")

    try:
        while True:
            obs, _ = env.reset()
            # Apply fixed command if specified
            if args.vx is not None or args.vy is not None or args.wz is not None:
                env.set_command(
                    args.vx if args.vx is not None else env.commands[0],
                    args.vy if args.vy is not None else env.commands[1],
                    args.wz if args.wz is not None else env.commands[2],
                )

            ep_reward  = 0.0
            ep_steps   = 0
            terminated = False
            t_ep_start = time.time()

            while not terminated:
                t0 = time.time()
                action = _get_action(obs)
                obs, reward, terminated, _, _ = env.step(action)
                ep_reward += reward
                ep_steps  += 1

                if not args.headless:
                    env.render()
                    elapsed = time.time() - t0
                    sleep   = step_time - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            ep_time = time.time() - t_ep_start
            cmd_str = f"vx={env.commands[0]:.2f} vy={env.commands[1]:.2f} wz={env.commands[2]:.2f}"
            print(
                f"[sim2real] Episode {episode + 1:4d} | "
                f"steps={ep_steps:4d} | reward={ep_reward:7.2f} | "
                f"time={ep_time:.1f}s | {cmd_str}"
            )

            episode += 1
            if args.num_episodes > 0 and episode >= args.num_episodes:
                break

    except KeyboardInterrupt:
        print("\n[sim2real] Interrupted by user.")
    finally:
        env.close()
        print("[sim2real] Done.")


if __name__ == "__main__":
    main()
