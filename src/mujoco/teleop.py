"""Keyboard + gamepad input for the MuJoCo viewer window (GLFW, no extra dependency).

Polled once per control step; BaseEnv.render() already calls glfw.poll_events(). Bindings and the
command mapping are in utils/teleop.py. Keys need the viewer window focused; the gamepad does not.
"""
from __future__ import annotations

import glfw
import numpy as np

from utils.teleop import SpeedLevel, deadzone1, deadzone2

_HOLD = {glfw.KEY_W: (0, +1), glfw.KEY_S: (0, -1), glfw.KEY_A: (1, +1), glfw.KEY_D: (1, -1),
         glfw.KEY_Q: (2, +1), glfw.KEY_E: (2, -1)}


class GlfwTeleop:
    def __init__(self, window, level: SpeedLevel | None = None):
        self.window = window
        self.level = level or SpeedLevel()
        self._prev = {}          # edge detection: name -> was pressed
        self._push = False
        self._pad_name = None

    def _edge(self, name: str, pressed: bool) -> bool:
        hit = pressed and not self._prev.get(name, False)
        self._prev[name] = pressed
        return hit

    def _gamepad(self):
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_is_gamepad(jid):
                ok, state = glfw.get_gamepad_state(jid)
                if ok:
                    name = glfw.get_gamepad_name(jid)
                    if name != self._pad_name:
                        self._pad_name = name
                        print(f"[teleop] gamepad: {name.decode() if isinstance(name, bytes) else name}")
                    return state
        return None

    def poll(self) -> tuple[float, float, float]:
        """Read the devices: returns (fwd, lat, turn) in [-1, 1]; updates the speed level and the push flag."""
        axes = np.zeros(3)
        key = lambda k: self.window is not None and glfw.get_key(self.window, k) == glfw.PRESS
        for k, (i, sign) in _HOLD.items():
            if key(k):
                axes[i] += sign
        lv_down = self._edge("z", key(glfw.KEY_Z))
        lv_up = self._edge("x", key(glfw.KEY_X))
        push = self._edge("space", key(glfw.KEY_SPACE))

        st = self._gamepad()
        if st is not None:
            lx, ly = deadzone2(st.axes[glfw.GAMEPAD_AXIS_LEFT_X], -st.axes[glfw.GAMEPAD_AXIS_LEFT_Y])  # y-down
            rx = deadzone1(st.axes[glfw.GAMEPAD_AXIS_RIGHT_X])
            axes += (ly, -lx, -rx)          # stick right = move right (-y), turn clockwise (-z)
            b = st.buttons
            lv_down |= self._edge("lb", bool(b[glfw.GAMEPAD_BUTTON_LEFT_BUMPER]))
            lv_up |= self._edge("rb", bool(b[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER]))
            push |= self._edge("a", bool(b[glfw.GAMEPAD_BUTTON_A]))

        if lv_down != lv_up:
            print(f"[teleop] speed {self.level.step(-1 if lv_down else +1) * 100:.0f}%")
        self._push |= push
        return tuple(np.clip(axes, -1.0, 1.0))

    def push_requested(self) -> bool:
        hit, self._push = self._push, False
        return hit
