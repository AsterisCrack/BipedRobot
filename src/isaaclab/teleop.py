"""Keyboard + gamepad input in the Isaac Sim window (carb events), for play.py.

Same interface as src/mujoco/teleop.py (poll() -> (fwd, lat, turn), push_requested(), level); bindings
and the command mapping are in utils/teleop.py. Isaac Lab's Se2Keyboard / Se2Gamepad are not reused:
their bindings are arrows/numpad and stick-right = +y, not the Jetson's WASD/QE and stick-right = -y.
Keys need the Isaac Sim window focused.
"""
from __future__ import annotations

import weakref

import carb
import carb.input
import numpy as np
import omni

from utils.teleop import SpeedLevel, deadzone1, deadzone2

_HOLD = {"W": (0, +1), "S": (0, -1), "A": (1, +1), "D": (1, -1), "Q": (2, +1), "E": (2, -1)}


def _free_space_key() -> None:
    """Space is Kit's global Play/Pause hotkey (omni.kit.widget.toolbar): pressing it to push would also
    pause the sim. Deregister it for this session (deregister, not disable: nothing is saved)."""
    try:
        import omni.kit.hotkeys.core as hk
        reg = hk.get_hotkey_registry()
        for h in list(reg.get_all_hotkeys_for_key("SPACE")):
            reg.deregister_hotkey(h)
            print(f"[teleop] released Space from Kit hotkey {h.action_text!r}")
    except Exception as e:          # no hotkeys extension: nothing is bound to Space then
        print(f"[teleop] could not check Kit hotkeys for Space ({e}); if Space pauses the sim, use pad A")


class IsaacTeleop:
    def __init__(self, level: SpeedLevel | None = None):
        self.level = level or SpeedLevel()
        self._held = set()
        self._stick = np.zeros((2, 3))      # (positive, negative) x (fwd, lat, turn); carb sends each half
        self._push = False
        self._level_steps = 0
        self._pad_pressed = set()

        _free_space_key()
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = appwindow.get_keyboard()
        self._gamepad = appwindow.get_gamepad(0)
        proxy = weakref.proxy(self)
        self._kb_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, lambda e, *a, obj=proxy: obj._on_key(e))
        self._pad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad, lambda e, *a, obj=proxy: obj._on_pad(e))

        G = carb.input.GamepadInput
        # stick half -> (direction 0 = +, 1 = -, axis). Stick right = move right (-y) / turn clockwise (-z).
        self._stick_map = {
            G.LEFT_STICK_UP: (0, 0), G.LEFT_STICK_DOWN: (1, 0),
            G.LEFT_STICK_LEFT: (0, 1), G.LEFT_STICK_RIGHT: (1, 1),
            G.RIGHT_STICK_LEFT: (0, 2), G.RIGHT_STICK_RIGHT: (1, 2),
        }
        self._buttons = {G.A: "push", G.LEFT_SHOULDER: "slower", G.RIGHT_SHOULDER: "faster"}

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._kb_sub)
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._pad_sub)
        except Exception:
            pass

    # ------------------------------------------------------------------ carb callbacks
    def _on_key(self, event) -> bool:
        name = event.input.name
        T = carb.input.KeyboardEventType
        if event.type == T.KEY_PRESS:       # KEY_REPEAT is ignored: one level step / push per press
            self._held.add(name)
            if name == "SPACE":
                self._push = True
            elif name == "Z":
                self._level_steps -= 1
            elif name == "X":
                self._level_steps += 1
        elif event.type == T.KEY_RELEASE:
            self._held.discard(name)
        return True

    def _on_pad(self, event) -> bool:
        if event.input in self._stick_map:
            d, axis = self._stick_map[event.input]
            self._stick[d, axis] = float(event.value)
        elif event.input in self._buttons:
            pressed = float(event.value) > 0.5
            what = self._buttons[event.input]
            if pressed and what not in self._pad_pressed:
                if what == "push":
                    self._push = True
                else:
                    self._level_steps += 1 if what == "faster" else -1
            (self._pad_pressed.add if pressed else self._pad_pressed.discard)(what)
        return True

    # ------------------------------------------------------------------ polled by play.py
    def poll(self) -> tuple[float, float, float]:
        """(fwd, lat, turn) in [-1, 1]; applies the speed-level presses since the last poll."""
        axes = np.zeros(3)
        for k, (i, sign) in _HOLD.items():
            if k in self._held:
                axes[i] += sign
        raw = self._stick[0] - self._stick[1]
        lx, ly = deadzone2(-raw[1], raw[0])         # deadzone in stick space (x right, y up)
        axes += (ly, -lx, deadzone1(raw[2]))
        if self._level_steps:
            for _ in range(abs(self._level_steps)):
                self.level.step(1 if self._level_steps > 0 else -1)
            self._level_steps = 0
            print(f"[teleop] speed {self.level.pct}%")
        return tuple(np.clip(axes, -1.0, 1.0))

    def push_requested(self) -> bool:
        hit, self._push = self._push, False
        return hit
