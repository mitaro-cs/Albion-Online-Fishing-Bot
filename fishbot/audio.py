"""Hear the bite: watch the *game's own* sound for a sudden splash.

Windows only. Instead of recording the speakers, it reads the peak meter of the
Albion Online audio session (the same meter as in Windows' Volume Mixer) via
pycaw, so music, Discord or a browser never trigger it. A jump of
``sensitivity``× above the recent background marks an onset.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class OnsetDetector:
    def __init__(self, sensitivity: float = 4.0, floor: float = 120.0, history: int = 150):
        self.sensitivity = sensitivity
        self.floor = floor
        self.levels: deque[float] = deque(maxlen=history)  # ~3 s of 20 ms blocks
        self.onsets: deque[float] = deque(maxlen=64)
        self._armed = True

    def feed(self, samples: np.ndarray, t: float) -> bool:
        x = samples.astype(np.float32)
        level = float(np.sqrt(np.mean(np.diff(x) ** 2))) if x.size > 1 else 0.0
        hit = False
        if len(self.levels) >= 40:
            background = float(np.median(self.levels))
            loud = level > max(self.floor, background * self.sensitivity)
            if loud and self._armed:
                self.onsets.append(t)
                hit = True
            self._armed = not loud  # one onset per sound, re-arm once it fades
        if not hit:
            self.levels.append(level)
        return hit

    def onset_after(self, t: float) -> bool:
        return any(o > t for o in self.onsets)

    def feed_level(self, level: float, t: float) -> bool:
        """Same as feed() for an already measured loudness (0..1 peak meter)."""
        hit = False
        if len(self.levels) >= 40:
            background = float(np.median(self.levels))
            loud = level > max(self.floor, background * self.sensitivity)
            if loud and self._armed:
                self.onsets.append(t)
                hit = True
            self._armed = not loud
        if not hit:
            self.levels.append(level)
        return hit


class AudioWatcher:
    """Polls the game's audio-session peak meter ~100 times a second."""

    PROCESS = "albion"

    def __init__(self, sensitivity: float = 4.0):
        self.detector = OnsetDetector(sensitivity, floor=0.04, history=300)
        self.ok = False      # attached to the game's audio session right now
        self.error = ""
        self._stop = threading.Event()

    @staticmethod
    def now() -> float:
        return time.monotonic()

    def onset_after(self, t: float) -> bool:
        return self.ok and self.detector.onset_after(t)

    def set_sensitivity(self, value: float) -> None:
        self.detector.sensitivity = value

    def start(self) -> None:
        if sys.platform != "win32":
            self.error = "audio is Windows-only"
            return
        threading.Thread(target=self._run, name="fishbot-audio", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _find_meter(self):
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is not None and self.PROCESS in proc.name().lower():
                return session._ctl.QueryInterface(IAudioMeterInformation)
        return None

    def _run(self) -> None:
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception as e:
            self.error = f"pycaw unavailable: {e}"
            log.warning("bite sound detection unavailable: %s", e)
            return
        meter, next_look = None, 0.0
        try:
            while not self._stop.is_set():
                now = self.now()
                if meter is None and now >= next_look:
                    try:
                        meter = self._find_meter()
                    except Exception as e:
                        self.error = str(e)
                    next_look = now + 3.0
                    if meter is not None:
                        log.info("listening to the game's own sound for bites")
                self.ok = meter is not None
                if meter is not None:
                    try:
                        self.detector.feed_level(float(meter.GetPeakValue()), now)
                    except Exception:  # game closed or its session went away: look again
                        meter = None
                self._stop.wait(0.01)
        finally:
            self.ok = False
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass
