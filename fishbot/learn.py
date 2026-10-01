"""Zero-setup learning: find the bobber and the reel bar by what changes on screen.

* Bobber — compare the water around the cast point before and after the cast;
  the new compact object is the bobber.
* Reel bar — compare the screen before and after the hook click; the new wide,
  thin shape is the bar. Inside it the marker is the column range that stands
  out from the bar's own background.

Frames are medians of a few shots, so animated water mostly cancels out.
"""
from __future__ import annotations

import cv2
import numpy as np


def median_frame(frames: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)


def find_new_object(before: list[np.ndarray], after: list[np.ndarray], near: tuple[int, int],
                    scale: float = 1.0) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Template + bbox (x, y, w, h) of the compact object that appeared, preferring ones near ``near``."""
    ref, now = median_frame(before), median_frame(after)
    noise = 0
    if len(before) >= 2:  # how much the water alone moves between shots
        noise = float(np.percentile(np.abs(_gray(before[0]) - _gray(before[-1])), 99))
    diff = np.abs(_gray(now) - _gray(ref)).astype(np.uint8)
    mask = (diff > max(28.0, noise * 1.3)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drops the thin fishing line
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lo, hi = 30 * scale * scale, 6000 * scale * scale
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not lo <= area <= hi or not 0.25 <= w / max(h, 1) <= 4:
            continue
        strength = float(diff[y:y + h, x:x + w].mean())
        dist = float(np.hypot(cents[i][0] - near[0], cents[i][1] - near[1]))
        score = strength * np.sqrt(area) / (1 + dist / (220 * scale))
        if score > best_score:
            best, best_score = (x, y, w, h), score
    if best is None:
        return None
    x, y, w, h = best
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(now.shape[1], x + w + pad), min(now.shape[0], y + h + pad)
    tpl = now[y0:y1, x0:x1].copy()
    if tpl.shape[0] < 6 or tpl.shape[1] < 6 or float(_gray(tpl).std()) < 4:
        return None
    return tpl, (x0, y0, x1 - x0, y1 - y0)


def find_new_bar(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bbox of a wide, thin UI element that appeared between two frames."""
    H, W = before.shape[:2]
    diff = np.abs(_gray(after) - _gray(before)).astype(np.uint8)
    best, best_score = None, 0.0
    for thresh in (40, 28):  # strict first; looser if the bar is low-contrast
        mask = (diff > thresh).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 25), np.uint8))  # join along the bar only
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if w < 0.06 * W or w < 4 * h or not 5 <= h <= 0.15 * H:
                continue
            fill = area / float(w * h)                 # a bar is solid, water noise is not
            centred = 1 - abs(cents[i][0] - W / 2) / W  # game UI sits near the middle
            greenish = float((green_mask(after[y:y + h, x:x + w]) > 0).mean())
            score = area * fill * centred * (1 + 20 * min(greenish, 0.25))  # the reel bar carries a green zone
            if score > best_score:
                best, best_score = (x, y, w, h), score
        if best:
            return best
    return None


def find_marker(strip: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The column range inside the bar that differs most from the bar's background."""
    g = _gray(strip).astype(np.float32)
    H, W = g.shape
    background = np.median(g, axis=1, keepdims=True)           # per-row colour of the track
    green = green_mask(strip) > 0
    g = np.where(green, np.broadcast_to(background, g.shape), g)  # the green zone is not the marker
    dev = np.abs(g - background).sum(axis=0)
    dev = np.convolve(dev, np.ones(3) / 3, mode="same")
    edge = max(2, int(W * 0.015))                               # bar frame / rounded ends
    dev[:edge] = dev[-edge:] = 0
    p = int(np.argmax(dev))
    peak = float(dev[p])
    if peak <= 0 or peak < 3 * float(np.median(dev[edge:-edge]) + 1):
        return None
    half = peak * 0.45
    x0 = p
    while x0 > edge and dev[x0 - 1] > half:
        x0 -= 1
    x1 = p
    while x1 < W - edge - 1 and dev[x1 + 1] > half:
        x1 += 1
    x0, x1 = max(0, x0 - 2), min(W, x1 + 3)
    if not 3 <= x1 - x0 <= W * 0.3:
        return None
    rows = np.where(np.abs(g[:, x0:x1] - background).mean(axis=1) > 12)[0]
    y0, y1 = (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, H)
    y0, y1 = max(0, y0 - 2), min(H, y1 + 2)
    if y1 - y0 < 4:
        y0, y1 = 0, H
    return strip[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def green_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 70, 60), (90, 255, 255))


def green_span(strip: np.ndarray, frac: float = 0.15) -> tuple[float, float] | None:
    """The green "safe" zone on the reel bar as (start, end), 0..1 across the bar."""
    cols = (green_mask(strip) > 0).mean(axis=0) > frac
    if not cols.any():
        return None
    gap = max(3, int(cols.size * 0.08))  # the bobber sits on the band and splits the green
    cols = cv2.morphologyEx(cols.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))[0] > 0
    best = (0, 0)
    start = None
    for i, on in enumerate(np.append(cols, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    w = cols.size
    if best[1] - best[0] < max(3, w * 0.02):
        return None
    return best[0] / w, best[1] / w


def find_green_bar(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Albion's minigame band: a wide strip with a green middle and red/orange chevron ends.
    Returns the band's bbox (x, y, w, h) in frame coordinates."""
    H, W = frame.shape[:2]
    green = cv2.morphologyEx(green_mask(frame), cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w >= 0.04 * W and w >= 2.5 * h and 8 <= h <= max(0.12 * H, 70) and area / (w * h) > 0.6 and area > best_area:
            best, best_area = (x, y, w, h), area
    if best is None:
        return None
    x, y, w, h = best
    hsv = cv2.cvtColor(frame[y:y + h], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # chevron ends: bright red/orange/yellow (grass and water are much darker or bluer)
    chevron = (((hue <= 32) | (hue >= 165)) & (sat > 120) & (val > 140)).mean(axis=0) > 0.25
    reach = int(w * 0.5)
    x0, x1 = x, x + w
    while x0 > max(0, x - reach) and chevron[x0 - 1]:
        x0 -= 1
    while x1 < min(W, x + w + reach) and chevron[x1]:
        x1 += 1
    return x0, y, x1 - x0, h


def find_float(img: np.ndarray) -> tuple[float, float, int, int, float] | None:
    """The white body of the bobber riding on the band: (cx, cy, w, h, area)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 190), (179, 70, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 6:
        return None
    return float(cents[i][0]), float(cents[i][1]), int(stats[i, 2]), int(stats[i, 3]), float(area)
