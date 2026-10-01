"""Screen capture through mss (DXGI-free, ~1 ms for small regions)."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from .config import Region


class Screen:
    """Interface shared by the real screen and the simulator."""

    def grab(self, region: Region) -> np.ndarray:  # BGR uint8
        raise NotImplementedError

    def monitors(self) -> list[dict]:
        raise NotImplementedError

    def grab_monitor(self, index: int) -> tuple[np.ndarray, dict]:
        mons = self.monitors()
        mon = mons[min(max(index, 1), len(mons) - 1)] if len(mons) > 1 else mons[0]
        r = Region(mon["left"], mon["top"], mon["width"], mon["height"])
        return self.grab(r), mon


class MssScreen(Screen):
    """mss handles are not thread-safe, so every thread gets its own."""

    def __init__(self):
        import mss  # fail early if missing
        self._mss = mss
        self._local = threading.local()

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = self._local.sct = self._mss.mss()
        return sct

    def grab(self, region: Region) -> np.ndarray:
        shot = self._sct().grab(region.as_mss())
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

    def monitors(self) -> list[dict]:
        return [dict(m) for m in self._sct().monitors]
