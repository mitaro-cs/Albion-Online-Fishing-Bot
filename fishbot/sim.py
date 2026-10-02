"""A small fishing simulator that mimics the game's loop on a virtual screen.

It powers ``--demo`` (try the UI without the game) and the end-to-end tests:
the real engine drives it through the same Screen/Input interfaces it uses in
the game, so casting, bite detection and the reel controller are exercised
for real — only the pixels are synthetic.
"""
from __future__ import annotations

import math
import random
import threading

import cv2
import numpy as np

from .audio import SoundSource
from .capture import Screen
from .config import Config, ProfileStore, Region
from .controls import Input
from .engine import Clock

W, H = 1600, 900
WATER = Region(160, 110, 1280, 470)
BAR = Region(600, 640, 400, 28)
MARKER_W, MARKER_H = 18, 22


class SimGame:
    def __init__(self, clock: Clock | None = None, seed: int | None = None):
        self.clock = clock or Clock()
        self.rng = random.Random(seed)
        self.lock = threading.RLock()
        self.cursor = (W // 2, 420)
        self.held = False
        self.t = self.clock.now()
        self.state = "idle"        # idle charging flying floating biting reel reward
        self.state_t = self.t
        self.bobber = (0.0, 0.0)
        self.bite_at = 0.0
        self.landing_seed = self.bite_seed = 0
        self.landed_at = -1.0
        self.spot_index: int | None = None
        self.x = self.v = self.progress = self.duration = 0.0
        self.fish = (1.0, 1.0, 0.0, 0.0)
        self.caught = self.escaped = 0
        self.bar_at = (BAR.left, BAR.top)  # where the minigame bar is drawn (UI scale / resolution)
        self.skin = 0        # 1: a different float look (another rod, night light)
        self.early = 0       # clicks before the fish took the bait ("Too early!" in the game)
        self.nibbles = False  # the float twitches a few times before the real bite
        self.surf = False     # the float drifts by a shore where foam rolls in and out
        self.sounds: list[tuple[float, str]] = []  # game sounds: (start, kind)
        self.ambient = 0.0   # mean seconds between unrelated game sounds (frogs, music); 0 = none
        self._next_ambient = 0.0
        self._twitch = (0.0, 0.0, 0.0)  # (start, end, depth px)
        self.invert = False  # True: holding pushes the marker left
        self.green = True    # the marker must be kept inside a moving green zone
        self.decoy = False   # a static UI panel ("A" key hint) that pops up with the minigame
        self.loot = ("", 0.0)  # (item name, shown until) — the "you received" banner
        self.fish_names = ["Common Rudd", "Brook Trout", "River Perch"]
        self.zone_phase = 0.0
        self.outside = 0.0
        self.spots = [self._new_spot() for _ in range(3)]
        self.bg = self._background()
        nrng = np.random.default_rng(seed)
        self.noise = [nrng.integers(0, 7, (H, W, 3), dtype=np.uint8) for _ in range(3)]

    # ── world ───────────────────────────────────────────────────────────────

    def _new_spot(self) -> dict:
        for _ in range(50):
            x = self.rng.uniform(WATER.left + 80, WATER.left + WATER.width - 80)
            y = self.rng.uniform(WATER.top + 60, WATER.top + WATER.height - 60)
            if all((x - s["x"]) ** 2 + (y - s["y"]) ** 2 > 220 ** 2 for s in getattr(self, "spots", [])):
                break
        return {"x": x, "y": y, "fish": self.rng.randint(3, 5), "respawn": 0.0}

    def _background(self) -> np.ndarray:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        b = 92 + 40 * yy + 10 * np.sin(xx * 9 + yy * 5)
        g = 70 + 26 * yy + 8 * np.sin(xx * 7 - yy * 4)
        r = 30 + 10 * yy + 0 * xx
        img = np.dstack([b, g, r]).astype(np.uint8)
        rng = np.random.default_rng(7)
        for _ in range(260):  # long soft wave streaks
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            cv2.ellipse(img, (x, y), (int(rng.integers(18, 70)), 2), 0, 0, 360, (150, 120, 70), 1, cv2.LINE_AA)
        shore = WATER.top + WATER.height + 70
        img[shore:] = (38, 52, 60)
        cv2.circle(img, (W // 2, shore + 90), 22, (60, 90, 120), -1, cv2.LINE_AA)
        return cv2.GaussianBlur(img, (3, 3), 0)

    # ── input events ────────────────────────────────────────────────────────

    def press(self) -> None:
        with self.lock:
            self.advance()
            self.held = True
            if self.state == "idle":
                self._set("charging")
            elif self.state == "floating":
                if self.spot_index is not None and self.t < self.bite_at:
                    self.early += 1          # "Too early!"
                self._set("idle")            # reeled the empty line in
            elif self.state == "biting":
                self._start_reel()

    def release(self) -> None:
        with self.lock:
            self.advance()
            self.held = False
            if self.state == "charging":
                if self.t - self.state_t >= 0.1 and WATER.contains(*self.cursor):
                    self.bobber = (self.cursor[0] + self.rng.uniform(-6, 6), self.cursor[1] + self.rng.uniform(-6, 6))
                    self._set("flying")
                else:
                    self._set("idle")

    def _set(self, state: str) -> None:
        self.state, self.state_t = state, self.t

    def zone(self) -> tuple[float, float]:
        """Green zone (centre, width) the marker has to stay in — it drifts along the bar."""
        if not self.green:
            return 0.5, 1.0
        # like the real game: a wide green middle (it may drift a little), chevron ends outside it
        return 0.5 + 0.05 * math.sin(0.8 * (self.t - self.state_t) + self.zone_phase), 0.56

    def _start_reel(self) -> None:
        self.x, self.v, self.progress, self.outside = 0.5, 0.0, 0.0, 0.0
        self.zone_phase = self.rng.uniform(0, 6.28)
        self.duration = self.rng.uniform(3.5, 6.0)
        self.fish = (self.rng.uniform(1.0, 1.8), self.rng.uniform(0.4, 0.9),
                     self.rng.uniform(0, 6.28), self.rng.uniform(0, 6.28))
        self._set("reel")

    # ── physics ─────────────────────────────────────────────────────────────

    def advance(self) -> None:
        with self.lock:
            now = self.clock.now()
            while self.t < now:
                dt = min(0.004, now - self.t)
                self.t += dt
                self._step(dt)

    def _step(self, dt: float) -> None:
        t, age = self.t, self.t - self.state_t
        for s in self.spots:
            if s["fish"] <= 0 and t >= s["respawn"]:
                s.update(self._new_spot())
        if self.ambient and t >= self._next_ambient:
            if self._next_ambient:
                self.sounds.append((t, self.rng.choice(("croak", "chime"))))
            self._next_ambient = t + self.rng.expovariate(1 / self.ambient)
        if self.state == "flying" and age > 0.7:
            self.landing_seed, self.landed_at = self.rng.randrange(1 << 30), t
            self.spot_index = self._spot_at(self.bobber)
            self.bite_at = t + self.rng.uniform(2.0, 7.0)
            self.sounds.append((t, "land"))  # the float lands with a splash
            self._set("floating")
        elif self.state == "floating" and self.spot_index is not None and t >= self.bite_at:
            self.bite_seed = self.rng.randrange(1 << 30)
            self.sounds.append((t, "bite"))
            self._set("biting")
        elif self.state == "floating" and self.nibbles and t > self._twitch[1] + 0.6 and self.rng.random() < dt * 0.8:
            # a nibble: the float jerks down a little and pops back, with a soft plop
            self._twitch = (t, t + self.rng.uniform(0.1, 0.3), self.rng.uniform(4.0, 8.0))
            self.sounds.append((t, "nibble"))
        elif self.state == "biting" and age > 1.2:
            # like the game: the fish lets go, and another one bites a while later
            self.bite_at = t + self.rng.uniform(4.0, 12.0)
            self._set("floating")
        elif self.state == "reel":
            a1, a2, p1, p2 = self.fish
            force = a1 * math.sin(1.7 * t + p1) + a2 * math.sin(4.3 * t + p2)
            push = 3.0 if self.held else -3.0
            acc = (-push if self.invert else push) + force - 1.5 * self.v
            self.v += acc * dt
            self.x += self.v * dt
            zc, zw = self.zone()
            if abs(self.x - zc) <= zw / 2:
                self.progress += dt
                self.outside = 0.0
            else:
                self.outside += dt  # line tension builds while the fish is out of the green
            if self.x <= 0.0 or self.x >= 1.0 or self.outside > 1.5:
                self.escaped += 1
                self._set("idle")
            elif self.progress >= self.duration:
                self.caught += 1
                self.loot = (self.rng.choice(self.fish_names), t + 2.5)
                spot = self.spots[self.spot_index] if self.spot_index is not None else None
                if spot:
                    spot["fish"] -= 1
                    if spot["fish"] <= 0:
                        spot["respawn"] = t + 8.0
                        spot["x"] = spot["y"] = -999
                self._set("reward")
        elif self.state == "reward" and age > 0.6:
            self._set("idle")

    def _spot_at(self, p) -> int | None:
        for i, s in enumerate(self.spots):
            if s["fish"] > 0 and (p[0] - s["x"]) ** 2 + (p[1] - s["y"]) ** 2 <= 45 ** 2:
                return i
        return None

    # ── rendering ───────────────────────────────────────────────────────────

    def render(self, r: Region) -> np.ndarray:
        with self.lock:
            self.advance()
            x0, y0 = max(0, r.left), max(0, r.top)
            x1, y1 = min(W, r.left + r.width), min(H, r.top + r.height)
            img = np.zeros((r.height, r.width, 3), np.uint8)
            if x1 <= x0 or y1 <= y0:
                return img
            view = img[y0 - r.top:y1 - r.top, x0 - r.left:x1 - r.left]
            view[:] = self.bg[y0:y1, x0:x1]
            k = int(self.t * 12) % len(self.noise)
            cv2.add(view, self.noise[k][y0:y1, x0:x1], dst=view)
            ox, oy = r.left, r.top
            for s in self.spots:
                if s["fish"] > 0:
                    draw_spot(img, s["x"] - ox, s["y"] - oy, self.t)
            age = self.t - self.state_t
            bx, by = self.bobber[0] - ox, self.bobber[1] - oy
            if self.state == "flying":
                f = min(1.0, age / 0.7)
                sx, sy = self.cursor[0] - ox, H - 160 - oy
                cv2.circle(img, (int(sx + (bx - sx) * f), int(sy + (by - sy) * f - 120 * math.sin(math.pi * f))),
                           5, (235, 235, 235), -1, cv2.LINE_AA)
            if self.surf and self.state in ("floating", "biting"):
                # a band of surf beside the float, slowly rolling in and out (no bite)
                k = 0.5 + 0.5 * math.sin(self.t * 2.2)
                for i in range(7):
                    sx = int(bx - 30 + 9 * i)
                    sy = int(by + 16 - 6 * k + 2 * math.sin(i))
                    c = int(140 + 100 * k)
                    cv2.ellipse(img, (sx, sy), (6, 3), 0, 0, 360, (c, c, c), -1, cv2.LINE_AA)
            if self.state == "floating":
                s0, s1, depth = self._twitch
                dip = depth if s0 <= self.t < s1 else 0.0
                draw_bobber(img, bx, by + 2 * math.sin(self.t * 3.9) + dip, self.skin)
                if self.t - self.landed_at < 0.4:  # it lands with a splash
                    draw_splash(img, bx, by, (self.t - self.landed_at) / 0.4, self.landing_seed)
            elif self.state == "biting":
                # the bite: a burst of foam and bubbles hits the float, which stays where it is
                draw_bobber(img, bx, by + 2 * math.sin(self.t * 3.9) + 2, self.skin)
                draw_splash(img, bx, by, min(1.0, age / 0.8), self.bite_seed)
            elif self.state == "reel":
                draw_bar(img, self.bar_at[0] - ox, self.bar_at[1] - oy, self.x, self.progress / self.duration,
                         self.zone() if self.green else None)
                if self.decoy:
                    x0, y0 = 500 - ox, 433 - oy
                    cv2.rectangle(img, (x0, y0), (x0 + 252, y0 + 49), (140, 90, 40), -1)
                    cv2.putText(img, "A", (x0 + 10, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 250, 250), 3, cv2.LINE_AA)
            if self.loot[0] and self.t < self.loot[1]:
                draw_loot(img, W // 2 - ox, int(H * 0.12) - oy, self.loot[0])
            if self.state == "reward":
                cv2.putText(img, "+1", (int(W / 2 - ox - 14), int(H - 240 - oy - age * 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (99, 230, 245), 2, cv2.LINE_AA)
            return img


def draw_bobber(img, x: float, y: float, skin: int = 0) -> None:
    x, y = int(round(x)), int(round(y))
    cv2.ellipse(img, (x, y + 3), (9, 3), 0, 0, 360, (60, 40, 20), -1, cv2.LINE_AA)
    if skin:  # a long yellow quill float instead of the round red one
        cv2.rectangle(img, (x - 3, y - 14), (x + 3, y + 4), (20, 20, 20), -1)
        cv2.rectangle(img, (x - 2, y - 13), (x + 2, y - 4), (40, 220, 240), -1)
        cv2.rectangle(img, (x - 2, y - 4), (x + 2, y + 3), (240, 240, 240), -1)
        return
    cv2.circle(img, (x, y), 7, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (6, 6), 0, 180, 360, (60, 60, 230), -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (6, 6), 0, 0, 180, (240, 240, 240), -1, cv2.LINE_AA)
    cv2.line(img, (x, y - 7), (x, y - 12), (20, 20, 20), 2, cv2.LINE_AA)


def draw_splash(img, x: float, y: float, phase: float, seed: int) -> None:
    """Foam and bubble rings around the float (a bite, or the float landing); fades as ``phase`` → 1."""
    rng = random.Random(seed)
    fade = 1.0 - 0.6 * phase
    for _ in range(14):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(11, 26) * (0.7 + 0.5 * phase)
        r = int(rng.uniform(2, 5) * (1 + phase))
        c = int(150 + 100 * fade)
        cv2.circle(img, (int(x + d * math.cos(a)), int(y + 0.7 * d * math.sin(a))), r, (c, c, c), -1 if rng.random() < 0.5 else 1,
                   cv2.LINE_AA)


def draw_spot(img, x: float, y: float, t: float) -> None:
    for i in range(6):
        a = i * math.pi / 3 + 0.3
        px, py = int(x + 16 * math.cos(a)), int(y + 8 * math.sin(a))
        cv2.circle(img, (px, py), 3, (230, 220, 200), 1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(y)), (24, 11), 0, 0, 360, (200, 185, 150), 1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), 2 + int(1.5 * (1 + math.sin(t * 5))), (240, 235, 225), -1, cv2.LINE_AA)


def draw_bar(img, x: float, y: float, pos: float, progress: float, zone=None) -> None:
    """Albion-style minigame: chevron band with red ends and a green middle, the bobber riding on
    it, and a blue progress bar with a fish below."""
    x, y = int(x), int(y)
    W_, H_ = BAR.width, BAR.height
    cv2.rectangle(img, (x, y), (x + W_ - 1, y + H_ - 1), (40, 125, 235), -1)            # orange
    for i in range(0, int(W_ * 0.22), 9):                                               # red/yellow chevrons
        colour = (40, 45, 210) if i < W_ * 0.1 else (40, 190, 240)
        for x0, d in ((x + i, 1), (x + W_ - 1 - i, -1)):
            pts = np.array([[x0, y], [x0 + d * 6, y + H_ // 2], [x0, y + H_ - 1]], np.int32)
            cv2.polylines(img, [pts], False, colour, 3, cv2.LINE_AA)
    zc, zw = zone if zone is not None else (0.5, 0.56)
    z0, z1 = int(x + (zc - zw / 2) * W_), int(x + (zc + zw / 2) * W_)
    cv2.rectangle(img, (max(x, z0), y), (min(x + W_ - 1, z1), y + H_ - 1), (60, 165, 75), -1)  # green
    # progress bar
    py = y + H_ + 6
    cv2.rectangle(img, (x, py), (x + W_ - 1, py + 14), (150, 90, 40), -1)
    fx = x + 12 + int((W_ - 40) * min(1.0, progress))
    cv2.ellipse(img, (fx, py + 7), (9, 4), 0, 0, 360, (250, 235, 215), -1, cv2.LINE_AA)
    cv2.line(img, (fx + 10, py + 7), (x + W_ - 14, py + 7), (230, 210, 190), 1)
    cv2.fillPoly(img, [np.array([[x + W_ - 12, py + 2], [x + W_ - 3, py + 7], [x + W_ - 12, py + 12]], np.int32)],
                 (40, 160, 245))
    # the bobber is the marker
    mx = int(x + MARKER_W // 2 + pos * (W_ - MARKER_W))
    cy = y + H_ // 2 + 3
    feather = np.array([[mx - 2, cy - 6], [mx - 8, y - 14], [mx, y - 6], [mx + 5, y - 13], [mx + 3, cy - 6]], np.int32)
    cv2.fillPoly(img, [feather], (40, 150, 250))
    cv2.circle(img, (mx, cy), 7, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.ellipse(img, (mx, cy), (6, 6), 0, 180, 360, (50, 50, 225), -1, cv2.LINE_AA)
    cv2.ellipse(img, (mx, cy), (6, 6), 0, 0, 180, (245, 245, 245), -1, cv2.LINE_AA)


def draw_loot(img, cx: int, cy: int, name: str) -> None:
    """Loot banner like the game's: metal bars, dark plate, gold item name, white quantity line."""
    w, h = 520, 96
    x0, y0 = cx - w // 2, cy - h // 2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (22, 24, 28), -1)
    for yy in (y0, y0 + h):
        cv2.rectangle(img, (x0 - 20, yy - 3), (x0 + w + 20, yy + 3), (150, 160, 170), -1)
    (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(img, name, (cx - tw // 2, y0 + 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (90, 205, 245), 2, cv2.LINE_AA)
    sub = "Received 1 pcs."
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, sub, (cx - sw // 2, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


class SimScreen(Screen):
    def __init__(self, game: SimGame):
        self.game = game

    def grab(self, region: Region) -> np.ndarray:
        return self.game.render(region)

    def monitors(self) -> list[dict]:
        mon = {"left": 0, "top": 0, "width": W, "height": H}
        return [mon, dict(mon)]


def sound_level(kind: str, dt: float) -> float:
    """Loudness envelope of each simulated game sound, ``dt`` seconds after it starts."""
    if dt < 0 or dt > 1.2:
        return 0.0
    if kind == "bite":     # sharp splash, then the slosh of the float being dragged under
        return 0.9 * math.exp(-dt / 0.08) + 0.5 * math.exp(-((dt - 0.22) / 0.05) ** 2)
    if kind == "nibble":   # soft short plop
        return 0.35 * math.exp(-dt / 0.04)
    if kind == "land":
        return 0.7 * math.exp(-dt / 0.25)
    if kind == "croak":    # frog: two quick croaks
        return 0.6 * (math.exp(-((dt - 0.05) / 0.035) ** 2) + math.exp(-((dt - 0.19) / 0.035) ** 2))
    return 0.5 * min(1.0, dt / 0.12) * math.exp(-max(0.0, dt - 0.12) / 0.3)  # chime / music swell


class SimAudio(SoundSource):
    """The game's sound meter for the simulator: renders the sounds it played, 100 times a second."""

    ok = True

    def __init__(self, game: SimGame):
        super().__init__()
        self.game = game
        self._at = game.clock.now()
        self._k = 0

    def now(self) -> float:
        return self.game.clock.now()

    def _sync(self) -> None:
        self.game.advance()
        end = self.now()
        while self._at < end:
            self._at += 0.01
            self._k += 1
            noise = 0.004 * ((self._k * 7919) % 13) / 13
            level = 0.02 + noise + sum(sound_level(k, self._at - t0) for t0, k in self.game.sounds[-12:])
            self._push(self._at, level)


class SimInput(Input):
    def __init__(self, game: SimGame):
        super().__init__()
        self.game = game
        self.keys: list[str] = []

    def _down(self) -> None:
        self.game.press()

    def _up(self) -> None:
        if self.game.held:
            self.game.release()

    def move(self, x: int, y: int) -> None:
        self.game.cursor = (int(x), int(y))

    def position(self) -> tuple[int, int]:
        return self.game.cursor

    def key(self, name: str) -> None:
        self.keys.append(name)


def make_templates() -> dict[str, np.ndarray]:
    """Crop sprite templates the same way a user would in the calibrator."""
    canvas = np.zeros((120, 200, 3), np.uint8)
    canvas[:] = (110, 82, 33)
    bob = canvas.copy()
    draw_bobber(bob, 100, 60)
    spot = canvas.copy()
    draw_spot(spot, 100, 60, 0.0)
    bar = np.zeros((BAR.height + 40, BAR.width + 40, 3), np.uint8)
    draw_bar(bar, 20, 20, 0.5, 0.0, (0.12, 0.1))  # zone away from the marker, like a user's crop
    mx = 20 + MARKER_W // 2 + int(0.5 * (BAR.width - MARKER_W))
    return {
        "bobber": bob[60 - 13:60 + 9, 100 - 10:100 + 10].copy(),
        "spot": spot[60 - 14:60 + 14, 100 - 27:100 + 27].copy(),
        "marker": bar[20:20 + BAR.height, mx - MARKER_W // 2:mx + MARKER_W // 2].copy(),
    }


def demo_config() -> Config:
    cfg = Config()
    cfg.cast.target = "auto"
    cfg.cast.points = [[560, 300], [1040, 300], [800, 470]]
    cfg.regions.bobber = Region(WATER.left, WATER.top, WATER.width, WATER.height)
    cfg.regions.water = Region(WATER.left, WATER.top, WATER.width, WATER.height)
    cfg.regions.reel = Region(BAR.left, BAR.top - BAR.height, BAR.width, 2 * BAR.height + 1)
    cfg.reel.method = "bar"
    cfg.system.learn_version = 3
    cfg.system.require_focus = False
    cfg.system.sound = False
    cfg.system.start_delay_s = 1.0
    cfg.bite.settle_ms = 600
    cfg.bite.bite_timeout_s = 12
    cfg.session.cooldown_ms = 600
    cfg.session.cooldown_jitter_ms = 400
    return cfg


def prepare_profile(store: ProfileStore, name: str = "Demo") -> str:
    if not store.exists(name):
        store.save(name, demo_config())
        for tpl, img in make_templates().items():
            store.save_template(name, tpl, img)
    return name
