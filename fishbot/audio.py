"""Hear the bite: watch the *game's own* sound for the bite splash.

Windows only. Instead of recording the speakers, it reads the peak meter of the
Albion Online audio session (the same meter as in Windows' Volume Mixer) via
pycaw, so music, Discord or a browser never trigger it. A jump of
``sensitivity``× above the recent background marks an onset.

The game itself is noisy too (music, frogs, other players), so the bot learns
what the bite sounds like: after every real bite (the reel minigame showed up)
it keeps the loudness envelope of the sound that came with it. Once a few are
known, only onsets with the same shape count — and even then only together
with the float going under.
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


STEP = 0.01      # envelope resolution, s
PRE, POST = 8, 40  # samples before / after the onset kept in a print
MIN_POST = 12    # judge a sound once this much of it (120 ms) has been heard
MATCH = 0.65     # envelope similarity that counts as "the same sound"
KEEP = 5         # prints remembered (newest win)


def similarity(a, b) -> float:
    """Shape similarity of two loudness envelopes over their common length: 1 is identical.

    Both are scaled to the same peak (the game's volume and the camera distance change how
    loud a sound is, not its shape); a mean gap of a quarter of the peak or more scores 0.
    """
    n = min(len(a), len(b))
    if n < PRE + MIN_POST:
        return 0.0
    x, y = np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)
    px, py = float(x.max()), float(y.max())
    if px <= 0 or py <= 0:
        return 0.0
    return max(0.0, 1.0 - float(np.abs(x / px - y / py).mean()) / 0.25)


def bite_print(prints: list) -> np.ndarray | None:
    """The typical bite envelope: element-wise median of the remembered ones (a stray one is outvoted)."""
    good = [p for p in prints if len(p) == PRE + POST]
    if len(good) < 2:
        return None
    return np.median(np.asarray(good, np.float64), axis=0)


class SoundSource:
    """Loudness history of the game's sound, its onsets and bite-print matching."""

    ok = False

    def __init__(self, sensitivity: float = 4.0):
        self.detector = OnsetDetector(sensitivity, floor=0.04, history=300)
        self._t: deque[float] = deque(maxlen=2000)   # ~20 s at 100 Hz
        self._lv: deque[float] = deque(maxlen=2000)
        self._lock = threading.Lock()

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def set_sensitivity(self, value: float) -> None:
        self.detector.sensitivity = value

    def _push(self, t: float, level: float) -> None:
        with self._lock:
            self._t.append(t)
            self._lv.append(level)
        self.detector.feed_level(level, t)

    def _sync(self) -> None:
        """Bring the history up to now (the simulator renders its sound lazily)."""

    def envelope(self, onset: float, full: bool = False) -> tuple[np.ndarray, float] | None:
        """Normalised loudness around an onset (as much as has been heard) and its peak."""
        with self._lock:
            ts, lv = np.fromiter(self._t, np.float64), np.fromiter(self._lv, np.float64)
        if ts.size < 2 or ts[0] > onset - PRE * STEP:
            return None
        n = int((min(ts[-1], onset + (POST - 1) * STEP) - onset) / STEP + 1e-6) + 1
        if n < (POST if full else MIN_POST):
            return None
        env = np.interp(onset + (np.arange(PRE + n) - PRE) * STEP, ts, lv)
        peak = float(env.max())
        return (env / peak, peak) if peak > 0 else None

    def bite_heard(self, since: float, prints: list) -> bool:
        """A sound since ``since`` that matches the learned bite (False until it is learned)."""
        self._sync()
        ref = bite_print(prints)
        if not self.ok or ref is None:
            return False
        for onset in [o for o in list(self.detector.onsets) if o > since]:
            got = self.envelope(onset)
            if got is not None and similarity(got[0], ref) >= MATCH:
                return True
        return False

    def remember(self, around: float, prints: list) -> list | None:
        """After a real bite: add the loudest sound near ``around`` to the prints."""
        self._sync()
        best = None
        for onset in [o for o in list(self.detector.onsets) if around - 0.6 <= o <= around + 0.3]:
            got = self.envelope(onset, full=True)
            if got is not None and (best is None or got[1] > best[1]):
                best = got
        if best is None:
            return None
        return [list(p) for p in prints[-(KEEP - 1):]] + [[round(float(v), 3) for v in best[0]]]


class AudioWatcher(SoundSource):
    """Polls the game's audio-session peak meter ~100 times a second."""

    PROCESS = "albion"

    def __init__(self, sensitivity: float = 4.0):
        super().__init__(sensitivity)
        self.ok = False      # attached to the game's audio session right now
        self.error = ""
        self._stop = threading.Event()

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
                        self._push(now, float(meter.GetPeakValue()))
                    except Exception:  # game closed or its session went away: look again
                        meter = None
                self._stop.wait(0.01)
        finally:
            self.ok = False
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass
