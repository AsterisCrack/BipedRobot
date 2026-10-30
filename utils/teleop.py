"""Manual command control (keyboard / gamepad) and the command-arrow geometry, shared by Isaac play.py and
MuJoCo sim2real.py. Pure numpy: the input devices live in src/isaaclab/teleop.py (carb) and
src/mujoco/teleop.py (GLFW), which both feed axes_to_command().

Bindings are the Jetson web UI's (BipedRobotJetson/web/static/js/control.js):
  W/S +vx/-vx   A/D +vy/-vy (left = +y)   Q/E +wz/-wz (Q turns left)   Z/X speed level -/+   Space push
  pad: left stick vx/vy, right stick x wz, LB/RB speed level -/+, A push

Full deflection x speed level reaches the TRAINING config's command range edge on that side; the result
is then limited exactly as in training (utils/command_limit.py), so the config stays the source of truth.
"""
from __future__ import annotations

import math

import numpy as np

from utils.command_limit import DEFAULT_RANGES, limit_command, validate_shape

# Mirrors BipedEnvCfg.events["push_robot"] (envs/isaaclab/biped_env_cfg.py); a config's
# env_config.events.push_robot.params.velocity_range overrides it, as in build_env_cfg.
DEFAULT_PUSH_RANGE = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.6, 0.6)}

DEADZONE = 0.12

# Command visualisation, the same in both sims. Command (green) and measured (blue) share one scale.
STOP_THRESH = 0.01          # |v| (m/s) or |wz| (rad/s) below this: a dot / no arc
ARROW_LEN_AT_MAX = 0.4      # m, arrow length at the command range's top speed
ARC_SPAN_AT_MAX = 1.5 * math.pi   # rad of arc at the range's top |wz|
ARC_RADIUS = 0.18           # m (the measured arc is drawn ARC_RADIUS_GAP further out)
ARC_RADIUS_GAP = 0.04
ARC_SEGMENTS = 12


def _edge(v: float, lo: float, hi: float) -> float:
    """v in [-1, 1] -> the range edge on v's side, scaled by |v|."""
    return v * hi if v >= 0 else -v * lo


def axes_to_command(fwd: float, lat: float, turn: float, ranges, shape: str = "box",
                    level: float = 1.0) -> np.ndarray:
    """(fwd, lat, turn) in [-1, 1] (+ = forward, left, counter-clockwise) -> the (vx, vy, wz) command."""
    r = {k: tuple(ranges.get(k, DEFAULT_RANGES[k])) for k in DEFAULT_RANGES}
    a = np.clip([fwd, lat, turn], -1.0, 1.0) * float(level)
    cmd = np.array([_edge(a[0], *r["lin_vel_x"]), _edge(a[1], *r["lin_vel_y"]),
                    _edge(a[2], *r["ang_vel_z"])], dtype=np.float64)
    return limit_command(cmd, r, validate_shape(shape))


class SpeedLevel:
    """Fraction of the command range a held key / full stick gives, in 10 % steps."""

    def __init__(self, value: float = 0.5, lo: float = 0.1, hi: float = 1.0, step: float = 0.1):
        self.lo, self.hi, self._step = lo, hi, step
        self.value = self._snap(value)

    def _snap(self, v: float) -> float:
        return round(min(max(v, self.lo), self.hi), 1)

    def step(self, direction: int) -> float:
        self.value = self._snap(self.value + direction * self._step)
        return self.value

    @property
    def pct(self) -> int:
        return int(round(self.value * 100))


def hud_text(level: SpeedLevel, cmd, teleop: bool, push_age_s: float | None = None) -> str:
    s = f"speed {level.pct}% | vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}"
    s += " | teleop" if teleop else " | random cmd"
    if push_age_s is not None and push_age_s < 1.0:
        s += " | push!"
    return s


def deadzone1(v: float, dz: float = DEADZONE) -> float:
    m = abs(v)
    if m < dz:
        return 0.0
    return math.copysign(min(1.0, (m - dz) / (1.0 - dz)), v)


def deadzone2(x: float, y: float, dz: float = DEADZONE) -> tuple[float, float]:
    """Radial deadzone, rescaled so the output still starts at 0 and reaches 1."""
    m = math.hypot(x, y)
    if m < dz:
        return 0.0, 0.0
    k = min(1.0, (m - dz) / (1.0 - dz)) / m
    return x * k, y * k


def push_range(env_config) -> dict:
    """The training push range: env_config.events.push_robot.params.velocity_range, else the default."""
    events = getattr(env_config, "events", None) if env_config else None
    try:
        vr = events["push_robot"]["params"]["velocity_range"]
    except (KeyError, TypeError):
        vr = None
    out = dict(DEFAULT_PUSH_RANGE)
    if vr:
        out.update({k: tuple(v) for k, v in dict(vr).items() if k in out})
    return out


def sample_push(vel_range: dict, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """(dvx, dvy, dwz), world frame: a random horizontal direction at the edge of the push range's
    (x, y) ellipse (a draw inside it is often too small to see), yaw uniform in its range."""
    th = rng.uniform(0.0, 2.0 * math.pi)
    ax = max(abs(v) for v in vel_range["x"])
    ay = max(abs(v) for v in vel_range["y"])
    dwz = rng.uniform(*vel_range["yaw"])
    return float(scale) * np.array([ax * math.cos(th), ay * math.sin(th), dwz])


def vis_scales(ranges) -> tuple[float, float]:
    """(m of arrow per m/s, rad of arc per rad/s) for this command range."""
    r = {k: tuple(ranges.get(k, DEFAULT_RANGES[k])) for k in DEFAULT_RANGES}
    v_max = max(abs(v) for k in ("lin_vel_x", "lin_vel_y") for v in r[k]) or 1.0
    w_max = max(abs(v) for v in r["ang_vel_z"]) or 1.0
    return ARROW_LEN_AT_MAX / v_max, ARC_SPAN_AT_MAX / w_max


def arc_points(yaw0: float, span: float, n: int = ARC_SEGMENTS) -> np.ndarray:
    """n + 1 unit-circle points (n+1, 2) from angle yaw0, sweeping `span` (signed: + = counter-clockwise)."""
    t = yaw0 + np.linspace(0.0, span, n + 1)
    return np.stack([np.cos(t), np.sin(t)], axis=-1)
