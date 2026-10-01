"""Quick setup: calibrate from inside the game with one hotkey, no screenshots.

Step 1  hover the floating bobber        → F7  (tight crop found by local contrast)
Step 2  hover the LEFT end of the reel bar  → F7
Step 3  hover the RIGHT end of the reel bar → F7  (marker found by its motion)
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .capture import Screen
from .config import Region

STEPS = ("bobber", "bar_left", "bar_right")


class SetupError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _monitor_for(screen: Screen, x: int, y: int) -> dict:
    mons = screen.monitors()
    for m in mons[1:] or mons:
        if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
            return m
    return mons[1] if len(mons) > 1 else mons[0]


def find_bobber(screen: Screen, x: int, y: int) -> tuple[np.ndarray, Region]:
    """Crop the most detailed patch near the cursor and size a search area around it."""
    mon = _monitor_for(screen, x, y)
    s = int(np.clip(round(mon["height"] * 0.028), 18, 48))  # ~30 px at 1080p
    r = 2 * s
    area = Region(x - r, y - r, 2 * r, 2 * r)
    patch = screen.grab(area)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    score = cv2.boxFilter(grad, -1, (s, s))
    # prefer patches near the cursor: people point roughly, not exactly
    yy, xx = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    score *= np.exp(-(((xx - r) ** 2 + (yy - r) ** 2) / (2 * (r * 0.6) ** 2)))
    lo, hi = s // 2, 2 * r - s // 2
    sub = score[lo:hi, lo:hi]
    _, _, _, (cx, cy) = cv2.minMaxLoc(sub)
    cx, cy = cx + lo, cy + lo
    tpl = patch[cy - s // 2:cy - s // 2 + s, cx - s // 2:cx - s // 2 + s].copy()
    if tpl.shape[0] < 8 or float(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).std()) < 6:
        raise SetupError("bobber_flat")
    ax, ay = area.left + cx, area.top + cy
    w, h = int(mon["width"] * 0.26), int(mon["height"] * 0.22)
    left = int(np.clip(ax - w // 2, mon["left"], mon["left"] + mon["width"] - w))
    top = int(np.clip(ay - h // 2, mon["top"], mon["top"] + mon["height"] - h))
    return tpl, Region(left, top, w, h)


def find_marker(screen: Screen, x0: int, x1: int, y: int, frames: int = 14,
                interval: float = 0.045, sleep=time.sleep) -> tuple[np.ndarray, Region]:
    """Watch the bar strip for a moment; the marker is what moves."""
    left, right = sorted((int(x0), int(x1)))
    if right - left < 60:
        raise SetupError("bar_narrow")
    band = Region(left, int(y) - 50, right - left, 100)
    shots = []
    for i in range(frames):
        shots.append(screen.grab(band))
        if i < frames - 1:
            sleep(interval)
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in shots]).astype(np.int16)
    background = np.median(grays, axis=0)
    diff = np.abs(grays[-1] - background).astype(np.uint8)
    mask = cv2.morphologyEx((diff > 24).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise SetupError("marker_still")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, area = (int(v) for v in stats[i])
    if area < 12 or bw >= band.width * 0.5:
        raise SetupError("marker_still")
    pad = 2
    tx0, ty0 = max(0, bx - pad), max(0, by - pad)
    tx1, ty1 = min(band.width, bx + bw + pad), min(band.height, by + bh + pad)
    tpl = shots[-1][ty0:ty1, tx0:tx1].copy()
    reel = Region(left, band.top + ty0 - 4, right - left, (ty1 - ty0) + 8)
    return tpl, reel
