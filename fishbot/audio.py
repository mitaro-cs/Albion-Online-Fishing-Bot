"""Hear the bite: watch the game's sound output for a sudden splash.

Windows only — records what the speakers play (WASAPI loopback, no microphone)
via PyAudioWPatch. Each 20 ms block is high-passed (first difference, so music
bass and ambience matter less) and its loudness compared with the recent
background; a jump of ``sensitivity``× marks an onset.
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


class AudioWatcher:
    def __init__(self, sensitivity: float = 4.0):
        self.detector = OnsetDetector(sensitivity)
        self.ok = False
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
            self.error = "audio capture is Windows-only"
            return
        threading.Thread(target=self._run, name="fishbot-audio", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self.error = "PyAudioWPatch is not installed"
            return
        pa = stream = None
        try:
            pa = pyaudio.PyAudio()
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not speakers.get("isLoopbackDevice"):
                for dev in pa.get_loopback_device_info_generator():
                    if speakers["name"] in dev["name"]:
                        speakers = dev
                        break
            rate = int(speakers["defaultSampleRate"])
            channels = max(1, int(speakers["maxInputChannels"]))
            block = rate // 50
            stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                             input_device_index=speakers["index"], frames_per_buffer=block)
            self.ok = True
            log.info("listening to %s for bite sounds", speakers["name"])
            while not self._stop.is_set():
                raw = stream.read(block, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).mean(axis=1)
                self.detector.feed(data, self.now())
        except Exception as e:  # no output device, exclusive mode, …: just go without sound
            self.error = str(e)
            log.warning("bite sound detection unavailable: %s", e)
        finally:
            self.ok = False
            try:
                if stream is not None:
                    stream.close()
                if pa is not None:
                    pa.terminate()
            except Exception:
                pass
