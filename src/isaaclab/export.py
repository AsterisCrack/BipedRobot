"""
Export a trained BipedRobotV2 policy for real-robot deployment.

Outputs:
  <out_dir>/actor.pt          — TorchScript actor (runs without Python/Isaac Lab)
  <out_dir>/obs_normalizer.npz — Running mean/std for observation pre-processing
  <out_dir>/config.yaml       — Config snapshot used to train this checkpoint

Usage:
  python src/isaaclab/export.py --checkpoint checkpoints/<run>/model.pt \
                                 --out_dir deploy/
"""
import os
import sys
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export BipedRobotV2 policy to TorchScript")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--out_dir",    type=str, default="deploy",  help="Output directory")
parser.add_argument("--config_path", type=str, default=None,    help="Config yaml (auto-detected if omitted)")
args = parser.parse_args()

import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_trl_root = os.path.join(_project_root, "torch_rl_algorithms")
if _trl_root not in sys.path:
    sys.path.insert(0, _trl_root)

from utils import Config

# ---------------------------------------------------------------------------
# Resolve config
# ---------------------------------------------------------------------------
config_path = args.config_path
if config_path is None:
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    for name in os.listdir(ckpt_dir):
        if name.endswith(".yaml") or name.endswith(".yml"):
            config_path = os.path.join(ckpt_dir, name)
            break
if config_path is None:
    config_path = os.path.join(_project_root, "BipedRobot", "config", "config.yaml")
    print(f"[WARN] Config not found near checkpoint; falling back to {config_path}")

config = Config(config_path)
device = torch.device("cpu")  # Export on CPU for portability

# ---------------------------------------------------------------------------
# Reconstruct model (no environment needed — we just load weights)
# ---------------------------------------------------------------------------
from torch_rl_algorithms.models.networks import ActorTwinCriticWithTargets
import gymnasium.spaces as spaces

use_history  = bool(getattr(config.train, "use_history",  False))
history_size = int(getattr(config.train, "history_size",  0))

# Obs / action dims must match training. Phase+contact means 52-dim base.
obs_proprio_dim = 52
obs_priv_dim    = 10
actor_dim = obs_proprio_dim * history_size if (use_history and history_size > 0) else obs_proprio_dim
critic_dim = obs_proprio_dim + obs_priv_dim   # no history on critic

actor_space  = spaces.Dict({
    "actor":  spaces.Box(-np.inf, np.inf, shape=(actor_dim,),  dtype=np.float32),
    "critic": spaces.Box(-np.inf, np.inf, shape=(critic_dim,), dtype=np.float32),
})
action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

model = ActorTwinCriticWithTargets(
    actor_space, action_space,
    actor_type="gaussian_multivariate",
    device=device,
    use_history=use_history,
    history_size=history_size,
    config=config,
)

ckpt = torch.load(args.checkpoint, map_location=device)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)
model.eval()

# ---------------------------------------------------------------------------
# Export actor as TorchScript
# ---------------------------------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)

class ActorWrapper(torch.nn.Module):
    """Thin wrapper: takes raw obs tensor → returns deterministic action tensor."""
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.actor(obs)
        # Deterministic: use the mode (loc) of the squashed Gaussian
        return dist.loc if hasattr(dist, "loc") else dist.sample()

actor_wrapper = ActorWrapper(model.actor)
example_input = torch.zeros(1, actor_dim)
scripted = torch.jit.trace(actor_wrapper, example_input)
actor_out = os.path.join(args.out_dir, "actor.pt")
scripted.save(actor_out)
print(f"[OK] Actor exported → {actor_out}")

# ---------------------------------------------------------------------------
# Export observation normalizer stats (if present)
# ---------------------------------------------------------------------------
normalizer_out = os.path.join(args.out_dir, "obs_normalizer.npz")
saved_normalizer = False

scalers_path = args.checkpoint.replace(".pt", "_obs_scalers.pt")
if os.path.exists(scalers_path):
    scalers = torch.load(scalers_path, map_location=device)
    arrays = {}
    for k, sd in scalers.items():
        arrays[f"{k}_mean"] = sd["mean"].cpu().numpy()
        arrays[f"{k}_var"]  = sd["var"].cpu().numpy()
    np.savez(normalizer_out, **arrays)
    saved_normalizer = True
    print(f"[OK] Obs normalizer exported → {normalizer_out}")

if not saved_normalizer:
    print("[WARN] No _obs_scalers.pt found — normalizer not exported. "
          "Run with normalize_obs=True during training to enable this.")

# ---------------------------------------------------------------------------
# Copy config snapshot
# ---------------------------------------------------------------------------
import shutil
config_out = os.path.join(args.out_dir, "config.yaml")
shutil.copy2(config_path, config_out)
print(f"[OK] Config snapshot → {config_out}")

# ---------------------------------------------------------------------------
# Smoke-test: run a forward pass on CPU
# ---------------------------------------------------------------------------
with torch.no_grad():
    dummy = torch.zeros(1, actor_dim)
    out = scripted(dummy)
print(f"[OK] Smoke test passed — output shape: {tuple(out.shape)}")
print()
print("Deployment checklist:")
print("  1. Copy actor.pt and obs_normalizer.npz to the robot.")
print("  2. Load with: model = torch.jit.load('actor.pt')")
print("  3. Pre-process obs: (obs - mean) / sqrt(var + 1e-4)")
print(f"  4. Actor input dim: {actor_dim}  |  output (action) dim: 12")
