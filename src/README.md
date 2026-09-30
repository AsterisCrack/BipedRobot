# Scripts Reference

Every runnable script in this directory, with all arguments documented.

---

## MuJoCo Training (`src/mujoco/`)

### `train.py` - Train or resume

```bash
python src/mujoco/train.py [--config PATH] [--checkpoint PATH]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/final/train_config_d4pg.yaml` | Path to YAML config file |
| `--checkpoint` | None | Path to a `trainer_state/` directory to resume from |

**Examples:**
```bash
# Fresh training with SAC
python src/mujoco/train.py --config config/final/train_config_sac.yaml

# Resume from checkpoint
python src/mujoco/train.py --config config/final/train_config_sac.yaml \
    --checkpoint checkpoints/sac/my_run/trainer_state/
```

The algorithm, architecture, and environment are all set in the YAML config. Checkpoints are saved to `checkpoints/<model_name>/step_<N>.pt` and full trainer state to `checkpoints/<model_name>/trainer_state/`.

---

### `test_model.py` - Evaluate a trained model

```bash
python src/mujoco/test_model.py [--config PATH] [--checkpoint PATH] [--episodes N] [--episode_length N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/final/train_config_sac.yaml` | Config used during training |
| `--checkpoint` | `checkpoints_final/sac/step_30000000.pt` | Path to `.pt` weights file |
| `--episodes` | `10` | Number of evaluation episodes |
| `--episode_length` | `1000` | Max steps per episode |

```bash
python src/mujoco/test_model.py \
    --config config/final/train_config_sac.yaml \
    --checkpoint checkpoints/sac/my_run/step_5000000.pt \
    --episodes 5
```

Opens a GLFW rendering window. Close with the window button or Ctrl+C.

---

### `train_lstm.py` - Train with LSTM backbone (MPO)

```bash
python src/mujoco/train_lstm.py [--config PATH]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/final/train_config_mpo.yaml` | Path to YAML config file |

This script forces the actor and critic to use an LSTM backbone regardless of what `network_type` is set to in the config. Sequence length is taken from `train.history_size` (defaults to 2 if not set). Uses MPO as the algorithm.

```bash
python src/mujoco/train_lstm.py --config config/final/train_config_mpo.yaml
```

---

## Isaac Lab Training (`src/isaaclab/`)

Isaac Lab must be installed. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### `train.py` - Train or resume

```bash
python src/isaaclab/train.py --task TASK --config_path PATH [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `BipedRobot` | Environment: `BipedRobot` (V1) or `BipedRobotV2` (V2) |
| `--config_path` | `config/train_config.yaml` | Path to YAML config file |
| `--num_envs` | from config | Override number of parallel environments |
| `--checkpoint` | None | Path to `.pt` file to resume from |
| `--headless` | False | Disable GUI (faster, for remote/cluster training) |
| `--video` | False | Record video every `--video_interval` steps |
| `--video_length` | `200` | Video length in steps |
| `--video_interval` | `2000` | Steps between video recordings |

```bash
# Train with GUI
python src/isaaclab/train.py --task BipedRobot \
    --config_path config/isaac/config_working.yaml

# Train headless (server / cluster)
python src/isaaclab/train.py --task BipedRobot \
    --config_path config/isaac/config_working.yaml \
    --headless

# Resume from checkpoint
python src/isaaclab/train.py --task BipedRobot \
    --config_path config/isaac/config_working.yaml \
    --checkpoint checkpoints/my_run/step_500000.pt

# Override number of envs (e.g. for debugging)
python src/isaaclab/train.py --task BipedRobot \
    --config_path config/isaac/config_working.yaml \
    --num_envs 64
```

---

### `play.py` - Evaluate / inference

```bash
python src/isaaclab/play.py --task TASK --config_path PATH --checkpoint PATH [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `BipedRobot` | Environment task |
| `--config_path` | `config/train_config.yaml` | Path to YAML config |
| `--checkpoint` | None | Path to `.pt` weights file |
| `--num_envs` | from config | Number of environments |
| `--video` | False | Record video |
| `--video_length` | `200` | Video length in steps |

```bash
python src/isaaclab/play.py --task BipedRobot \
    --config_path config/isaac/config_working.yaml \
    --checkpoint checkpoints/my_run/step_1000000.pt
```

---

## Deployment (`src/isaaclab/`)

### `export_onnx.py` - Export a checkpoint to a standalone ONNX bundle

```bash
python -m src.isaaclab.export_onnx [--checkpoint PATH] [--out_dir PATH] [--config_path PATH] [--opset N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | newest in `checkpoints/` | Path to a `step_<N>.pt` weights file |
| `--out_dir` | `exported/<run>_<step>/` | Output bundle directory |
| `--config_path` | run's own `config.yaml` | Override for reading `history_size` |
| `--opset` | `17` | ONNX opset version |

Rebuilds the actor from the checkpoint's own tensor shapes (never from config, which is what
broke the old `export.py` - it hardcoded a 52-dim observation and never ran successfully
against a V2 checkpoint), traces the deterministic path
`tanh(mean_layer(torso(obs)))`, and writes a directory that needs nothing but `onnxruntime` +
`numpy` to run - no torch, no Isaac Lab, no import from this repo:

```
<out>/policy.onnx           [1,250] float32 -> [1,12] float32, stage-2 normalizer baked in
<out>/deploy_config.json    stage-1 normalizer stats, joint limits, timing, obs layout
<out>/checkpoint_info.json  source checkpoint, step, git sha, export timestamp
```

There are two trained observation normalizers in series - stage 1 (`obs_mean`/`obs_var` in
`deploy_config.json`) is applied by the runtime; stage 2 is inside `policy.onnx` as baked-in
constants. Both are required; skipping either produces a policy that looks plausible and
walks into the floor. The export validates this automatically (see below) and fails loudly
rather than writing a bad bundle.

```bash
python -m src.isaaclab.export_onnx --checkpoint checkpoints/my_run/step_5000000.pt
```

**Built-in checks, run every export:**
- ONNX output vs PyTorch on 100 random inputs, max abs diff must be < 1e-5
- `obs_dim == history_size * 50` (catches a stale/mismatched `history_size`)
- IMU specific-force channel mean is ~+9.81 (catches a gravity-convention error)
- `step_dt == 0.02` (50 Hz)

---

### `verify_onnx_bundle.py` - Independent verification against the training actor

```bash
python -m src.isaaclab.verify_onnx_bundle --bundle PATH [--checkpoint PATH] [--n N] [--tol TOL]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--bundle` | *(required)* | Path to an exported bundle directory |
| `--checkpoint` | `source_checkpoint` from `checkpoint_info.json` | Override the reference weights |
| `--n` | `200` | Number of random observations to compare |
| `--tol` | `1e-5` | Max allowed action difference |

`export_onnx.py`'s self-check compares ONNX against its *own* reconstruction of the actor - a
mis-mapped layer would be wrong in both and still pass. This script closes that hole by
loading the checkpoint into the repo's real `torch_rl_algorithms.models.mlp.MLPActor` and
comparing its `get_action()` output against the ONNX bundle, on observations drawn from the
checkpoint's own trained distribution (`obs_mean + sigma * randn`) rather than arbitrary noise.

```bash
python -m src.isaaclab.verify_onnx_bundle --bundle exported/my_run_step_5000000
```

Also asserts the structural invariants the Jetson runtime depends on: `history_order`,
`obs_dim == history_size * obs_proprio_dim`, `joint_names` length and uniqueness, `obs_layout`
tiling the full proprio frame, and the EMA filter's `action_filter_applications_per_step == 4`
(Isaac applies it once per decimation substep, not once per control step).

Copy the bundle directory to `BipedRobotJetson/models/<name>/` to deploy - see that repo's
README for the control-side setup.

---

## Developer Tools (`src/tools/`)

These scripts do not train - they are for debugging, analysis, and validation.

### `try_symmetry.py` - Symmetry validation (no Isaac Lab required)

```bash
python src/tools/try_symmetry.py [--checkpoint PATH]
```

Runs the policy with mirrored inputs and checks that the mirrored output matches the forward-pass output. Useful for verifying that a trained policy has learned left-right symmetric behaviour.

---

### `plot_symmetry_logs.py` - Plot symmetry error logs

```bash
python src/tools/plot_symmetry_logs.py [--log_dir PATH]
```

Reads CSV files written by the symmetry validators and plots error over time with matplotlib.

---

### `src/tools/isaaclab/debug_env.py` - Interactive joint controller

Requires Isaac Lab.

```bash
python src/tools/isaaclab/debug_env.py --task BipedRobot [--num_envs 1]
```

Opens a Tkinter window with sliders for each of the 12 joints. Lets you pose the robot manually, inspect root height, and observe contact forces. Useful for understanding the robot's range of motion and verifying the simulation setup.

---

### `src/tools/isaaclab/test_ik.py` - Interactive IK test

Requires Isaac Lab.

```bash
python src/tools/isaaclab/test_ik.py --task BipedRobot [--num_envs 1]
```

Opens a Tkinter window with 6 sliders per foot (3 position + 3 orientation). Solves IK and drives the robot to the target foot positions in real time. Used for validating the IK utility in `utils/ik_utils.py`.

---

### `src/tools/isaaclab/try_symmetry.py` - Symmetry validator with TensorBoard

Requires Isaac Lab.

```bash
python src/tools/isaaclab/try_symmetry.py --task BipedRobot \
    --checkpoint checkpoints/my_run/step_1000000.pt
```

Runs the policy, computes left-right symmetry error at each step, and logs it to TensorBoard. More detailed than the standalone version - shows per-joint asymmetry curves.

---

## Common Patterns

**Pick a different algorithm** - set `train.model` in the config:
```yaml
train:
  model: "d4pg"   # sac | ddpg | d4pg | mpo | ppo
```

**Pick a different architecture** - set `model.actor_config.network_type`:
```yaml
model:
  actor_config:
    network_type: "transformer"  # mlp | cnn | lstm | transformer
```

**Disable GUI rendering during MuJoCo eval:**
```python
# test_model.py creates the env with render_mode="human" by default.
# Edit MujocoEnv(render_mode=None) in test_model.py to skip the window.
```

**Monitor training:**
```bash
tensorboard --logdir runs/
```
