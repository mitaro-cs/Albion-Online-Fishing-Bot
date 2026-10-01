"""Mouse/keyboard output.

Windows uses SendInput directly (hardware scan codes, which games read more
reliably than virtual-key events). Other platforms fall back to pynput.
"""
from __future__ import annotations

import sys
import threading
import time


class Input:
    """Interface shared by the real backends and the simulator."""

    def __init__(self):
        self._held = False
        self._lock = threading.Lock()

    @property
    def held(self) -> bool:
        return self._held

    def down(self) -> None:
        with self._lock:
            if not self._held:
                self._down()
                self._held = True

    def up(self) -> None:
        with self._lock:
            if self._held:
                self._up()
                self._held = False

    def release(self) -> None:
        """Unconditional release — safe to call from any thread at any time."""
        with self._lock:
            self._up()
            self._held = False

    def click(self, hold_s: float = 0.06) -> None:
        self.down()
        time.sleep(hold_s)
        self.up()

    # backend hooks
    def _down(self) -> None: raise NotImplementedError
    def _up(self) -> None: raise NotImplementedError
    def move(self, x: int, y: int) -> None: raise NotImplementedError
    def position(self) -> tuple[int, int]: raise NotImplementedError
    def key(self, name: str) -> None: raise NotImplementedError


_VK = {
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "backspace": 0x08,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}


def vk_code(name: str) -> int | None:
    name = name.strip().lower()
    if len(name) == 1 and name.isalnum():
        return ord(name.upper())
    return _VK.get(name)


class WinInput(Input):
    def __init__(self):
        super().__init__()
        import ctypes
        from ctypes import wintypes

        ulong_ptr = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._ct, self._INPUT, self._MI, self._KI = ctypes, INPUT, MOUSEINPUT, KEYBDINPUT
        self._user32 = ctypes.windll.user32
        self._user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self._point = wintypes.POINT

    def _send(self, inp) -> None:
        self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp))

    def _mouse(self, flags: int) -> None:
        inp = self._INPUT(type=0)
        inp.u.mi = self._MI(0, 0, 0, flags, 0, 0)
        self._send(inp)

    def _down(self) -> None:
        self._mouse(0x0002)  # MOUSEEVENTF_LEFTDOWN

    def _up(self) -> None:
        self._mouse(0x0004)  # MOUSEEVENTF_LEFTUP

    def move(self, x: int, y: int) -> None:
        self._user32.SetCursorPos(int(x), int(y))

    def position(self) -> tuple[int, int]:
        pt = self._point()
        self._user32.GetCursorPos(self._ct.byref(pt))
        return pt.x, pt.y

    def key(self, name: str) -> None:
        vk = vk_code(name)
        if vk is None:
            raise ValueError(f"unknown key {name!r}")
        scan = self._user32.MapVirtualKeyW(vk, 0)
        for flags in (0x0008, 0x0008 | 0x0002):  # KEYEVENTF_SCANCODE, then | KEYUP
            inp = self._INPUT(type=1)
            inp.u.ki = self._KI(0, scan, flags, 0, 0)
            self._send(inp)
            time.sleep(0.04)


class PynputInput(Input):
    def __init__(self):
        super().__init__()
        from pynput import keyboard, mouse
        self._m = mouse.Controller()
        self._k = keyboard.Controller()
        self._btn = mouse.Button.left
        self._keys = keyboard.Key

    def _down(self) -> None:
        self._m.press(self._btn)

    def _up(self) -> None:
        self._m.release(self._btn)

    def move(self, x: int, y: int) -> None:
        self._m.position = (int(x), int(y))

    def position(self) -> tuple[int, int]:
        x, y = self._m.position
        return int(x), int(y)

    def key(self, name: str) -> None:
        name = name.strip().lower()
        k = name if len(name) == 1 else getattr(self._keys, {"escape": "esc"}.get(name, name), None)
        if k is None:
            raise ValueError(f"unknown key {name!r}")
        self._k.press(k)
        time.sleep(0.04)
        self._k.release(k)


def system_input() -> Input:
    return WinInput() if sys.platform == "win32" else PynputInput()
