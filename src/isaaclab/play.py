import os
import sys
import torch
import numpy as np
import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play Biped Robot in Isaac Lab")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BipedRobot", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--config_path", type=str, default=None,
                    help="Training config. Default: the checkpoint dir's config.yaml (the run's own), "
                         "else config/config.yaml")
parser.add_argument("--pushes", action="store_true",
                    help="Keep the training pushes (events.push_robot) on during playback, to watch "
                         "push recovery. Off by default: play runs the nominal, unperturbed robot.")
parser.add_argument("--teleop", action="store_true",
                    help="Drive the command from the keyboard (W/S fwd/back, A/D left/right, Q/E turn, "
                         "Z/X speed -/+) or a gamepad (sticks, LB/RB speed). The camera follows the robot "
                         "and episodes don't time out.")
parser.add_argument("--push_scale", type=float, default=1.0,
                    help="Space / pad A random push size, x the training push range edge. The push works "
                         "with or without --teleop.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher reinitializes sys.path; re-add project paths after it launches.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_trl_root = os.path.join(_project_root, "torch_rl_algorithms")
if _trl_root not in sys.path:
    sys.path.insert(0, _trl_root)

from torch_rl_algorithms.algorithms.sac.model import SAC
from torch_rl_algorithms.algorithms.ddpg.model import DDPG
from torch_rl_algorithms.algorithms.d4pg.model import D4PG
from torch_rl_algorithms.algorithms.mpo.model import MPO
from torch_rl_algorithms.algorithms.ppo.model import PPO
from config.schema import ModelType
from utils import Config
from utils.residual_home import freeze_residual_map
from src.isaaclab.common import BaseIsaacLabWrapper
from src.isaaclab.env_config import build_env_cfg, describe as describe_env_cfg
from envs.isaaclab.biped_env import BipedEnv
from envs.isaaclab.biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
from envs.isaaclab.biped_env_cfg import BipedRobotV2EnvCfg

if args_cli.task == "BipedRobot":
    BipedEnvCfg = BipedRobotEnvCfg
elif args_cli.task == "BipedRobotV2":
    BipedEnvCfg = BipedRobotV2EnvCfg
else:
    raise ValueError(f"Unknown task: {args_cli.task}")

class IsaacLabWrapper(BaseIsaacLabWrapper):
    """Inference wrapper - step() returns a 5-tuple matching the Gymnasium interface."""

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        obs, rew, terminated, truncated, info = self.env.step(actions)
        return self._process_obs(obs), rew, terminated, truncated, info

class _Hud:
    """A small omni.ui window with the teleop state; falls back to printing when the text changes."""

    def __init__(self):
        self._label = None
        self._last = None
        try:
            import omni.ui as ui
            self._window = ui.Window("Teleop", width=460, height=60)
            with self._window.frame:
                self._label = ui.Label("", style={"font_size": 18})
        except Exception as e:
            print(f"[play] no omni.ui HUD ({e}); printing the teleop state instead")

    def set(self, text: str, console_text: str) -> None:
        if self._label is not None:
            self._label.text = text
        elif console_text != self._last:
            print(f"[teleop] {console_text}")
        self._last = console_text


def _make_teleop_tick(env, config):
    from src.isaaclab.teleop import IsaacTeleop
    from utils.teleop import axes_to_command, hud_text, push_range, sample_push

    teleop = IsaacTeleop()
    gen = env.command_generator
    ranges, shape = env._cmd_full_ranges, gen.limit_shape
    push_vr = push_range(getattr(config.train, "env_config", None))
    rng = np.random.default_rng(args_cli.seed)
    hud = _Hud()
    state = {"t_push": None, "cmd": None}
    if args_cli.teleop:
        print("[play] teleop: W/S fwd/back  A/D left/right  Q/E turn  Z/X speed -/+  Space push  "
              "(pad: sticks, LB/RB speed, A push). Keys need the Isaac Sim window focused.")
    print(f"[play] push: Space / pad A, range {push_vr} x {args_cli.push_scale}")

    def tick():
        axes = teleop.poll()
        if args_cli.teleop:
            state["cmd"] = axes_to_command(*axes, ranges, shape, teleop.level.value)
            gen.set_manual(state["cmd"])
        if teleop.push_requested():
            dv = sample_push(push_vr, rng, args_cli.push_scale)
            env.apply_push(dv)
            state["t_push"] = time.time()
            print(f"[play] push dv=({dv[0]:+.2f}, {dv[1]:+.2f}) m/s  dwz={dv[2]:+.2f} rad/s")
        shown = state["cmd"] if args_cli.teleop else env.commands[0].tolist()
        age = None if state["t_push"] is None else time.time() - state["t_push"]
        # console fallback: only the teleop command (a random heading-mode wz changes every step)
        console = (hud_text(teleop.level, np.round(shown, 2), True) if args_cli.teleop
                   else f"speed {teleop.level.pct}%")
        hud.set(hud_text(teleop.level, shown, args_cli.teleop, age), console)

    return tick


def _latest_checkpoint() -> str | None:
    """Newest step_N.pt in the newest folder under checkpoints/ (not its step_N_obs_scalers.pt)."""
    checkpoints_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    if not os.path.exists(checkpoints_root):
        print(f"No checkpoint directories found in {checkpoints_root}")
        return None
    dirs = [os.path.join(checkpoints_root, d) for d in os.listdir(checkpoints_root)
            if os.path.isdir(os.path.join(checkpoints_root, d))]
    if not dirs:
        print(f"No checkpoint directories found in {checkpoints_root}")
        return None
    latest_dir = max(dirs, key=os.path.getmtime)
    print(f"Searching for checkpoints in: {latest_dir}")
    files = [f for f in os.listdir(latest_dir) if f.endswith(".pt") and not f.endswith("_obs_scalers.pt")]
    if not files:
        print(f"No .pt files found in {latest_dir}")
        return None

    def get_step_num(filename):
        try:
            return int(filename.split("_")[1].replace(".pt", ""))
        except (IndexError, ValueError):
            return 0

    checkpoint_path = os.path.join(latest_dir, max(files, key=get_step_num))
    print(f"Auto-selected latest checkpoint: {checkpoint_path}")
    return checkpoint_path


def play():
    # Checkpoint first: without --config_path the config comes from the checkpoint's own run folder,
    # so the auto-selected checkpoint has to be known before the config is loaded.
    checkpoint_path = args_cli.checkpoint or _latest_checkpoint()
    if not checkpoint_path:
        print("No checkpoint provided. Please provide a checkpoint path using --checkpoint")
        return

    # Load config. Without --config_path, use the run's own config so a gait-table checkpoint
    # gets its reference block (fixed-period clock) and its actor type (equivariant_actor)
    # instead of config/config.yaml's.
    if args_cli.config_path is None:
        run_cfg = (os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "config.yaml")
                   if checkpoint_path else None)
        args_cli.config_path = run_cfg if run_cfg and os.path.exists(run_cfg) else "config/config.yaml"
    print(f"[play] task={args_cli.task}  config={args_cli.config_path}")
    if not os.path.exists(args_cli.config_path):
        rel_path = os.path.join(os.path.dirname(__file__), "../../../", args_cli.config_path)
        if os.path.exists(rel_path):
            args_cli.config_path = rel_path
        else:
            print(f"Config file not found: {args_cli.config_path}")
            return

    config = Config(args_cli.config_path)
    freeze_residual_map(config, warn_unfrozen=True)     # a run's own config.yaml is already numbers; a repo config may say auto
    
    # Same builder as train.py, with pushes and physics DR off: playback runs the NOMINAL
    # robot through the identical observation/actuation pipeline (action map, servo model,
    # sensor lag, noise). The old hand-copied block here ignored most of those, so play was
    # evaluating a different env from the one the policy trained in.
    env_cfg = build_env_cfg(config, BipedEnvCfg, training=False,
                            num_envs=args_cli.num_envs if args_cli.num_envs is not None else 1,
                            pushes=args_cli.pushes)
    if args_cli.teleop:
        env_cfg.episode_length_s = 1.0e6            # no time-out reset while driving
        env_cfg.viewer.origin_type = "asset_root"   # camera follows env 0's robot
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (1.2, 1.2, 0.7)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    print("[play] env contract:", describe_env_cfg(env_cfg))

    env = BipedEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video or args_cli.headless else None)
    wrapped_env = IsaacLabWrapper(env, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    use_history = config.train.use_history
    history_size = config.train.history_size
    

    # Select model class based on config
    model_type = config.train.model
    if model_type == ModelType.SAC:
        ModelClass = SAC
    elif model_type == ModelType.DDPG:
        ModelClass = DDPG
    elif model_type == ModelType.D4PG:
        ModelClass = D4PG
    elif model_type == ModelType.MPO:
        ModelClass = MPO
    elif model_type == ModelType.PPO:
        ModelClass = PPO
    else:
        # Fallback
        if str(model_type).lower() == "sac":
            ModelClass = SAC
        elif str(model_type).lower() == "ddpg":
            ModelClass = DDPG
        elif str(model_type).lower() == "d4pg":
            ModelClass = D4PG
        elif str(model_type).lower() == "mpo":
            ModelClass = MPO
        elif str(model_type).lower() == "ppo":
            ModelClass = PPO
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model = ModelClass(
        env=wrapped_env,
        model_path=checkpoint_path,
        device=device,
        config=config,
        use_history=use_history,
        history_size=history_size
    )

    # Load obs-normalisation statistics saved during training and freeze them.
    if checkpoint_path and hasattr(wrapped_env, "load"):
        wrapped_env.load(checkpoint_path)
    if hasattr(wrapped_env, "set_eval_mode"):
        wrapped_env.set_eval_mode()

    # Switch neural-network layers to eval mode (BatchNorm uses running stats, Dropout disabled)
    _net = getattr(model, "model", model)
    if hasattr(_net, "eval"):
        _net.eval()

    # Teleop / push (GUI only). The push works without --teleop too, to watch recovery under random commands.
    tick = None
    if env.sim.has_gui():
        tick = _make_teleop_tick(env, config)
    elif args_cli.teleop:
        print("[play] --teleop needs the GUI (drop --headless); running random commands")

    # Play loop
    obs = wrapped_env.start()

    print("Starting playback...")
    _step_count = 0
    _history_size = getattr(env_cfg, "history_size", 0) if getattr(env_cfg, "use_history", False) else 0

    while simulation_app.is_running():
        if tick is not None:
            tick()      # a new command is applied in this step's _pre_physics_step: the next obs carries it

        # For the first history_size steps after each reset, use stochastic rsample()
        # to match SAC training behaviour (policy was trained with rsample, not .mean).
        # The zero-padded history produces extreme obs at episode start that cause
        # saturated deterministic actions; stochastic sampling is more robust here.
        if _step_count < _history_size and hasattr(model, "model") and hasattr(model.model, "actor"):
            obs_t = obs.get("actor", next(iter(obs.values()))) if isinstance(obs, dict) else obs
            if not isinstance(obs_t, torch.Tensor):
                obs_t = torch.from_numpy(obs_t).float()
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            obs_t = obs_t.to(device)
            with torch.no_grad():
                dist = model.model.actor.forward(obs_t)
                action = dist.rsample() if hasattr(dist, "rsample") else model.step(obs)
        else:
            action = model.step(obs)

        _step_count += 1

        # Step environment
        obs, rew, terminated, truncated, info = wrapped_env.step(action)

        # Reset if needed
        if terminated.any() or truncated.any():
            obs = wrapped_env.start()
            _step_count = 0   # restart warmup on each episode reset
            
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    play()
