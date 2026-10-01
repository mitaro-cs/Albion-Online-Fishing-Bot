"""What did we catch? Read the loot banner the game shows at the top of the screen.

After a catch the game pops up a banner: item name (gold) and "Получено N шт."
(received N pcs). It is found as the new, large change in the top part of the
screen compared with a shot taken during the minigame, read with Windows OCR,
and tallied by name. Without OCR the banner image itself is the "name" and
identical banners are grouped by image similarity.
"""
from __future__ import annotations

import re
import threading

import cv2
import numpy as np

from .ocr import read_lines

QTY = re.compile(r"(\d+)")


def top_area(mon) -> tuple[int, int, int, int]:
    """Where loot banners appear: upper middle of the game screen."""
    return (mon.left + int(mon.width * 0.22), mon.top + int(mon.height * 0.03),
            int(mon.width * 0.56), int(mon.height * 0.32))


def find_banner(before: np.ndarray, after: np.ndarray) -> np.ndarray | None:
    """Crop of the banner that appeared between two shots of the top area."""
    diff = np.abs(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
                  - cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)).astype(np.uint8)
    mask = cv2.morphologyEx((diff > 30).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = after.shape[:2]
    best, area = None, 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if w >= 0.2 * W and h >= 0.06 * H and w > 1.5 * h and a > area:
            best, area = (x, y, w, h), a
    if best is None:
        return None
    x, y, w, h = best
    return after[y:y + h, x:x + w].copy()


def text_lines(banner: np.ndarray) -> list[np.ndarray]:
    """Split the banner into its text lines (bright glyphs on the dark plate)."""
    hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
    ink = ((hsv[..., 2] > 170) & ((hsv[..., 1] < 70) | ((hsv[..., 0] >= 12) & (hsv[..., 0] <= 38)))).astype(np.uint8)
    rows = ink.mean(axis=1) > 0.02
    lines, start = [], None
    for i, on in enumerate(np.append(rows, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= 6:
                cols = np.where(ink[start:i].any(axis=0))[0]
                x0, x1 = max(0, cols[0] - 4), min(banner.shape[1], cols[-1] + 5)
                lines.append(banner[max(0, start - 3):i + 3, x0:x1].copy())
            start = None
    return lines


def signature(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (96, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


class CatchLog:
    """Tally of caught items for the current session."""

    def __init__(self):
        self.items: list[dict] = []  # {name, image, sig, count}
        self.lock = threading.Lock()
        self.rev = 0

    def reset(self) -> None:
        with self.lock:
            self.items.clear()
            self.rev += 1

    def add(self, banner: np.ndarray) -> dict:
        lines = text_lines(banner)
        title = lines[0] if lines else banner
        name, qty = None, 1
        texts = read_lines(banner)
        if texts:
            texts = [t.strip() for t in texts if t.strip()]
            qty_line = next((t for t in texts if QTY.search(t)), "")
            words = [t for t in texts if t != qty_line]
            name = words[0] if words else None
            m = QTY.search(qty_line)
            qty = int(m.group(1)) if m else 1
        sig = signature(title)
        with self.lock:
            for item in self.items:
                same = (name and item["name"] == name) or (not name and not item["name"]
                                                          and float((item["sig"] * sig).mean()) > 0.85)
                if same:
                    item["count"] += qty
                    break
            else:
                item = {"name": name, "image": title, "sig": sig, "count": qty}
                self.items.append(item)
            self.rev += 1
            return item

    def summary(self) -> list[dict]:
        from .vision import to_data_url
        with self.lock:
            return [{"name": i["name"], "count": i["count"],
                     "image": None if i["name"] else to_data_url(i["image"], 260)} for i in self.items]
