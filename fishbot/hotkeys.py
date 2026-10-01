"""Global hotkeys (work while the game has focus) via pynput."""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


def _name(key) -> str:
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    return str(getattr(key, "name", "") or "").lower()


class Hotkeys:
    def __init__(self):
        self._bindings: dict[str, callable] = {}
        self._down: set[str] = set()
        self._listener = None
        self._lock = threading.Lock()
        self.ok = False

    def bind(self, bindings: dict[str, callable]) -> None:
        with self._lock:
            self._bindings = {k.strip().lower(): fn for k, fn in bindings.items() if k}

    def start(self) -> bool:
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(on_press=self._press, on_release=self._release, daemon=True)
            self._listener.start()
            self.ok = True
        except Exception as e:  # no display server, missing permissions, …
            log.warning("global hotkeys unavailable: %s", e)
            self.ok = False
        return self.ok

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _press(self, key) -> None:
        name = _name(key)
        if name in self._down:  # ignore auto-repeat while held
            return
        self._down.add(name)
        with self._lock:
            fn = self._bindings.get(name)
        if fn:
            try:
                fn()
            except Exception:
                log.exception("hotkey %s failed", name)

    def _release(self, key) -> None:
        self._down.discard(_name(key))
