"""Object finders (template / HSV colour) and preview annotation helpers."""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


class VisionError(RuntimeError):
    """Raised when calibration data is missing or unusable."""

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


@dataclass
class Match:
    x: float      # centre, frame coordinates
    y: float
    w: int
    h: int
    score: float  # 0..1 (colour finder reports 1.0 when present)
    area: float


class TemplateFinder:
    def __init__(self, template: np.ndarray, grayscale: bool = True, name: str = "template"):
        self.gray = grayscale
        tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if grayscale else template
        if tpl.shape[0] < 3 or tpl.shape[1] < 3 or float(tpl.std()) < 1.5:
            raise VisionError("template_flat", name=name)
        self.tpl = np.ascontiguousarray(tpl)
        self.h, self.w = tpl.shape[:2]

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.gray else frame

    def _scores(self, frame: np.ndarray) -> np.ndarray | None:
        img = self._prep(frame)
        if img.shape[0] < self.h or img.shape[1] < self.w:
            return None
        res = cv2.matchTemplate(img, self.tpl, cv2.TM_CCOEFF_NORMED)
        # perfectly flat patches divide by zero → nan/inf on some builds
        return np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def find(self, frame: np.ndarray) -> Match | None:
        res = self._scores(frame)
        if res is None:
            return None
        _, best, _, loc = cv2.minMaxLoc(res)
        return Match(loc[0] + self.w / 2, loc[1] + self.h / 2, self.w, self.h, float(best), self.w * self.h)

    def find_all(self, frame: np.ndarray, threshold: float, limit: int = 8) -> list[Match]:
        res = self._scores(frame)
        out: list[Match] = []
        if res is None:
            return out
        for _ in range(limit):
            _, best, _, (x, y) = cv2.minMaxLoc(res)
            if best < threshold:
                break
            out.append(Match(x + self.w / 2, y + self.h / 2, self.w, self.h, float(best), self.w * self.h))
            # non-maximum suppression: blank out a template-sized neighbourhood
            res[max(0, y - self.h):y + self.h, max(0, x - self.w):x + self.w] = -1.0
        return out


class ColorFinder:
    """Largest blob inside an HSV range. Hue ranges may wrap (lo_h > hi_h) for reds."""

    def __init__(self, lo, hi, min_area: int):
        self.lo, self.hi, self.min_area = list(lo), list(hi), min_area
        self.kernel = np.ones((3, 3), np.uint8)

    def mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        (h0, s0, v0), (h1, s1, v1) = self.lo, self.hi
        if h0 <= h1:
            m = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
        else:
            m = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1)) | cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)

    def find(self, frame: np.ndarray) -> Match | None:
        n, _, stats, cents = cv2.connectedComponentsWithStats(self.mask(frame), connectivity=8)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < self.min_area:
            return None
        return Match(float(cents[i][0]), float(cents[i][1]), int(stats[i, cv2.CC_STAT_WIDTH]),
                     int(stats[i, cv2.CC_STAT_HEIGHT]), 1.0, float(area))


class BarFinder:
    """Albion reel minigame: the bobber (white float) on the green/red band.

    The search region spans the band plus the space above it (the bobber sticks up).
    ``zone`` holds the green span of the last frame; no green means the minigame is over.
    """

    def __init__(self):
        self.zone: tuple[float, float] | None = None

    def find(self, frame: np.ndarray) -> Match | None:
        from .learn import find_float, green_span
        h = frame.shape[0]
        band = frame[int(h * 0.45):]             # lower part of the region is the band itself
        self.zone = green_span(band, 0.3)
        if self.zone is None:
            return None
        f = find_float(frame)
        if f is None:
            return None
        cx, cy, w, hh, area = f
        return Match(cx, cy, w, hh, 1.0, area)


def make_finder(method: str, template: np.ndarray | None, cfg, name: str):
    if method == "bar":
        return BarFinder()
    if method == "color":
        return ColorFinder(cfg.hsv_lo, cfg.hsv_hi, cfg.min_area)
    if template is None:
        raise VisionError("template_missing", name=name)
    return TemplateFinder(template, cfg.grayscale, name)


# ── preview rendering ───────────────────────────────────────────────────────

# UI palette in BGR
YELLOW = (99, 230, 245)
GREEN = (160, 231, 110)
RED = (107, 107, 255)
BLUE = (255, 168, 106)
PINK = (217, 122, 255)
WHITE = (236, 236, 236)


def draw_box(img: np.ndarray, m: Match, color, label: str | None = None) -> None:
    x0, y0 = int(m.x - m.w / 2), int(m.y - m.h / 2)
    cv2.rectangle(img, (x0, y0), (x0 + m.w, y0 + m.h), color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def draw_reel(img: np.ndarray, x: float | None, target: float, deadband: float, holding: bool) -> None:
    h, w = img.shape[:2]
    lo, hi = int((target - deadband / 2) * w), int((target + deadband / 2) * w)
    band = img.copy()
    cv2.rectangle(band, (lo, 0), (max(lo + 1, hi), h), GREEN, -1)
    cv2.addWeighted(band, 0.18, img, 0.82, 0, img)
    if x is not None:
        px = int(x * (w - 1))
        cv2.line(img, (px, 0), (px, h), PINK if holding else BLUE, 2, cv2.LINE_AA)


def to_data_url(img: np.ndarray, max_w: int = 720, quality: int = 82) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
