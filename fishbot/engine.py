"""The fishing state machine.

    countdown → [actions] → cast → land → bite → hook → reel → cooldown → …

Runs on its own thread. Every wait goes through ``_sleep``/``_tick`` so stop,
pause, focus loss and the failsafe corner are honoured within ~50 ms, and the
mouse button is always released on the way out (``finally``).
"""
from __future__ import annotations

import logging
import random
import statistics
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .capture import Screen
from .config import Config, Region
from .controller import ReelController
from .catches import CatchLog, find_banner, top_area
from .learn import (find_float, find_green_bar, find_marker, find_new_bar, find_new_object, find_reel_band,
                    green_span)
from .controls import Input
from .system import beep, foreground_title
from .vision import (BarFinder, BLUE, GREEN, PINK, RED, YELLOW, Match, SplashMeter, TemplateFinder, VisionError,
                     draw_box, draw_reel, make_finder)


class Clock:
    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, dt: float) -> None:
        if dt > 0:
            time.sleep(dt)


class _Stop(BaseException):
    """Unwinds the worker; BaseException so broad ``except Exception`` can't swallow it."""


class _Pause(BaseException):
    """Unwinds the current cycle back to the main loop."""


MESSAGES = {  # English fallback, used by the CLI; the UI translates codes itself
    "start": "Session started",
    "stopped": "Stopped",
    "countdown": "Starting in {s}s — switch to the game",
    "cast": "Cast #{n} at ({x}, {y}), power {ms} ms",
    "bobber": "Bobber settled",
    "bite": "Bite after {s}s ({why})",
    "reel_start": "Reeling…",
    "caught": "Caught! Reel took {s}s",
    "fail": "Missed: {reason}",
    "rotate": "Switching fishing spot",
    "spot": "Fishing spot found (score {score})",
    "spot_none": "No fishing spot visible",
    "limit_catches": "Catch limit reached ({n})",
    "limit_time": "Session time limit reached ({n} min)",
    "break": "Taking a break for {min} min",
    "break_end": "Break over",
    "action": "Pressed {key} {label}",
    "action_error": "Action key {key} failed: {error}",
    "paused": "Paused",
    "resumed": "Resumed",
    "focus_lost": "Game window not focused — waiting",
    "focus_back": "Game window focused again",
    "failsafe": "Failsafe corner — stopping",
    "error": "Error: {error}",
    "learn_bobber": "Learned the bobber ({w}×{h})",
    "learn_bobber_fail": "Couldn't spot the bobber — keep the cursor over open water",
    "learn_bar": "Learned the reel bar ({w} px) and marker",
    "learn_bar_fail": "Couldn't spot the reel bar — retrying on the next fish",
    "learn_hold": "Holding the button moves the marker {way}",
    "learn_sound": "Learned the bite sound",
    "relocate": "Found the bobber in a new place",
    "relearn_bobber": "Lost the bobber — learning it again",
    "bait": "Used bait ({key})",
    "relearn_bar": "The learned reel bar never moves — learning it again",
    "recast": "Nothing landed on the water — casting again",
    "loot": "{name} — {count} total",
    "vision_error": "Calibration problem: {code} {name}",
}


def format_event(e: dict) -> str:
    try:
        return MESSAGES.get(e["code"], e["code"]).format(**e["params"])
    except (KeyError, IndexError, ValueError):
        return f"{e['code']} {e['params']}"


@dataclass
class Stats:
    casts: int = 0
    bites: int = 0
    hooks: int = 0
    catches: int = 0
    escaped: int = 0
    misses: int = 0
    fail_streak: int = 0
    best_streak: int = 0
    win_streak: int = 0
    bite_sum: float = 0.0
    reel_sum: float = 0.0
    last_reel_s: float = 0.0
    fps: float = 0.0


class Engine:
    def __init__(self, screen: Screen, inp: Input, clock: Clock | None = None,
                 focus=foreground_title, rng: random.Random | None = None, on_event=None):
        self.screen, self.inp = screen, inp
        self.clock = clock or Clock()
        self.rng = rng or random.Random()
        self._focus_fn = focus
        self.on_event = on_event
        self.on_learn = None
        self.audio = None  # AudioWatcher (or a stand-in in tests)
        self._bite_t: float | None = None  # when the float went under for the last hooked bite
        self.debug_dir = None  # Path for failure snapshots
        self._escapes = 0
        self._bobber_misses = 0  # casts in a row where the bobber wasn't seen
        self._no_bites = 0       # casts in a row without a bite
        self._loot_due = False
        self._bait_at: float | None = None
        self._bait_catches = 0
        self.reel_target = None
        self.catches = CatchLog()
        self._loot_ref = None
        self.last_trigger = ""
        self.last_bite: dict = {}
        self._learn_ref = None
        self._per_cast = False      # the float is learned on every cast (auto-learn), not kept
        self._landed: tuple | None = None  # (x, y, Match) where the float was just found, area coordinates
        self._nothing_landed = False  # the last cast put nothing new on the water
        self._landings: deque[tuple[float, float]] = deque(maxlen=7)  # where real floats landed, from the aim
        self._landed_at: tuple[float, float] | None = None  # this cast's float, screen coordinates
        self._last_try = True
        self._float_known = False   # announced "learned the float" once this session

        self._lock = threading.RLock()
        self._cfg = Config()
        self._templates: dict[str, np.ndarray] = {}
        self._dirty = True
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()

        self.status = "idle"        # idle | running | paused
        self.stage = "idle"
        self.stage_ends: float | None = None
        self.pause_reason = ""
        self.stats = Stats()
        self.reel_x: float | None = None
        self.logs: deque[dict] = deque(maxlen=400)
        self._log_id = 0

        self._preview: tuple[np.ndarray, str, float] | None = None
        self._preview_at = 0.0
        self.preview_wanted_until = 0.0
        self._target: tuple[int, int] | None = None
        self._point_index = 0
        self._avoid: list[tuple[float, float, float]] = []
        self._next_focus_check = 0.0
        self._next_failsafe_check = 0.0
        self._active_acc = 0.0
        self._active_t0: float | None = None
        self._next_break = 0.0
        self._action_last: dict[int, float] = {}

    # ── public control (any thread) ─────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def configure(self, cfg: Config, templates: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._cfg, self._templates, self._dirty = cfg, dict(templates), True

    def validate(self) -> None:
        """Raise VisionError if the current calibration can't run."""
        with self._lock:
            cfg, tpl = self._cfg, self._templates
        self._build(cfg, tpl)

    def start(self) -> None:
        if self.running and self._stop.is_set():
            self.join(3)  # a quick Stop → Start: let the old worker finish first
        if self.running:
            self.resume()
            return
        sysc = self._cfg.system
        if sysc.auto_learn and sysc.relearn_on_start:
            self._relearn_all()
        self.validate()
        self._stop.clear()
        self._pause.clear()
        self.stats = Stats()
        self.catches.reset()
        self._active_acc, self._active_t0 = 0.0, None
        self._action_last.clear()
        self._bait_at = None
        self._avoid.clear()
        self._point_index = 0
        self._float_known = False
        self._landings.clear()  # the player may stand elsewhere now
        self.status = "running"
        self._thread = threading.Thread(target=self._main, name="fishbot-engine", daemon=True)
        self._thread.start()

    def pause(self, reason: str = "user") -> None:
        if self.running and not self._pause.is_set():
            self.pause_reason = reason
            self._pause.set()
            self.inp.release()

    def resume(self) -> None:
        self.pause_reason = ""
        self._pause.clear()

    def toggle(self) -> None:
        if not self.running or self._stop.is_set():
            self.start()
        elif self._pause.is_set():
            self.resume()
        else:
            self.pause()

    def stop(self) -> None:
        self._stop.set()
        self.inp.release()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def reset_stats(self) -> None:
        self.stats = Stats()
        self._active_acc = 0.0
        self._next_break = 0.0
        if self._active_t0 is not None:
            self._active_t0 = self.clock.now()

    def active_seconds(self) -> float:
        extra = (self.clock.now() - self._active_t0) if self._active_t0 is not None else 0.0
        return self._active_acc + extra

    def snapshot(self, since: int = 0) -> dict:
        s = self.stats
        active = self.active_seconds()
        hours = active / 3600
        return {
            "status": self.status,
            "stage": self.stage,
            "remaining": max(0.0, self.stage_ends - self.clock.now()) if self.stage_ends else None,
            "pause_reason": self.pause_reason,
            "hold": self.inp.held,
            "reel_x": self.reel_x,
            "target": list(self._target) if self._target else None,
            "stats": {
                **{k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(s).items()},
                "active_s": round(active, 1),
                "per_hour": round(s.catches / hours, 1) if hours > 0.01 else 0.0,
                "success": round(100 * s.catches / s.casts, 1) if s.casts else 0.0,
                "avg_bite": round(s.bite_sum / s.bites, 2) if s.bites else 0.0,
                "avg_reel": round(s.reel_sum / s.catches, 2) if s.catches else 0.0,
            },
            "logs": self._logs_since(since),
            "catch_rev": self.catches.rev,
        }

    def _logs_since(self, since: int) -> list[dict]:
        with self._lock:
            return [e for e in self.logs if e["id"] > since]

    def take_preview(self) -> tuple[np.ndarray, str, float] | None:
        self.preview_wanted_until = time.monotonic() + 2.5
        with self._lock:
            return self._preview

    def probe(self, kind: str) -> dict:
        """One-shot detection on the live screen, used by the UI's Test buttons."""
        with self._lock:
            cfg, tpl = self._cfg, self._templates
        if kind == "spot":
            region = cfg.regions.water
            if not region.ok:
                raise VisionError("region_missing", name="water")
            if "spot" not in tpl:
                raise VisionError("template_missing", name="spot")
            frame = self.screen.grab(region)
            matches = TemplateFinder(tpl["spot"], True, "spot").find_all(frame, cfg.cast.spot_threshold)
            for m in matches:
                draw_box(frame, m, YELLOW, f"{m.score:.2f}")
            best = matches[0] if matches else None
            return {"kind": kind, "found": bool(matches), "score": best.score if best else 0.0,
                    "count": len(matches), "image": frame}
        sub, region, name = (cfg.bite, cfg.regions.bobber, "bobber") if kind == "bobber" else \
                            (cfg.reel, cfg.regions.reel, "marker")
        if not region.ok:
            raise VisionError("region_missing", name=kind if kind == "bobber" else "reel")
        finder = make_finder(sub.method, tpl.get(name), sub, name)
        frame = self.screen.grab(region)
        m = finder.find(frame)
        found = self._present(m, sub)
        if m is not None:
            draw_box(frame, m, GREEN if found else RED, f"{m.score:.2f}")
        return {"kind": kind, "found": found, "score": m.score if m else 0.0, "image": frame}

    # ── worker ──────────────────────────────────────────────────────────────

    def _main(self) -> None:
        self._active_t0 = self.clock.now()
        self._emit("start")
        reason = "stopped"
        countdown, backoff = True, 0.0
        try:
            while True:
                try:
                    if backoff:
                        self._sleep(backoff)
                        backoff = 0.0
                    if countdown:
                        self._countdown()
                        countdown = False
                    self._tick()
                    cfg = self._refresh()
                    self._session_gate(cfg)
                    self._run_actions(cfg)
                    self._use_bait(cfg)
                    self._cycle(cfg)
                    self._cooldown(cfg)
                except _Pause:
                    self.inp.release()
                    self._wait_resume()
                    countdown = True
                except VisionError:
                    raise
                except Exception as e:  # transient: log, count as a miss, back off, keep going
                    self.inp.release()
                    self._emit("error", "error", error=f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                    self._fail(self._cfg, "error")
                    backoff = 1.0
        except _Stop:
            pass
        except VisionError as e:
            reason = "error"
            self._emit("vision_error", "error", code=e.code, name=e.params.get("name", ""))
        except Exception as e:  # pragma: no cover - last-resort guard
            reason = "error"
            self._emit("error", "error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            self.inp.release()
            if self._active_t0 is not None:
                self._active_acc += self.clock.now() - self._active_t0
                self._active_t0 = None
            self.status, self.stage, self.stage_ends, self.reel_x = "idle", "idle", None, None
            self._emit("stopped")
            if self._cfg.system.sound:
                beep("error" if reason == "error" else "info")

    def _refresh(self) -> Config:
        with self._lock:
            if self._dirty:
                self._build(self._cfg, self._templates)
                self._dirty = False
            return self._cfg

    def _build(self, cfg: Config, tpl: dict) -> None:
        r = cfg.regions
        learn = cfg.system.auto_learn
        bobber = marker = None  # None = learn it on the fly
        if r.bobber.ok and (cfg.bite.method == "color" or "bobber" in tpl):
            bobber = make_finder(cfg.bite.method, tpl.get("bobber"), cfg.bite, "bobber")
        elif not learn:
            if not r.bobber.ok:
                raise VisionError("region_missing", name="bobber")
            make_finder(cfg.bite.method, tpl.get("bobber"), cfg.bite, "bobber")
        if r.reel.ok and (cfg.reel.method in ("color", "bar") or "marker" in tpl):
            marker = make_finder(cfg.reel.method, tpl.get("marker"), cfg.reel, "marker")
        elif not learn:
            if not r.reel.ok:
                raise VisionError("region_missing", name="reel")
            make_finder(cfg.reel.method, tpl.get("marker"), cfg.reel, "marker")
        for f, region, name in ((bobber, r.bobber, "bobber"), (marker, r.reel, "marker")):
            if isinstance(f, TemplateFinder) and (f.w > region.width or f.h > region.height):
                raise VisionError("template_too_big", name=name)
        spot = None
        if cfg.cast.target == "auto":
            if not r.water.ok:
                raise VisionError("region_missing", name="water")
            if "spot" not in tpl:
                raise VisionError("template_missing", name="spot")
            spot = TemplateFinder(tpl["spot"], True, "spot")
        if cfg.cast.target == "points" and not cfg.cast.points:
            raise VisionError("points_missing", name="cast")
        old = getattr(self, "_marker", None)
        if isinstance(marker, BarFinder) and isinstance(old, BarFinder) and old.ends is not None                 and r.reel == getattr(self, "_marker_region", None):
            marker.ends = old.ends  # same bar: keep how its ends look (a settings change rebuilds the finders)
        self._marker_region = r.reel
        self._bobber, self._marker, self._spot = bobber, marker, spot

    # ── one fishing cycle ───────────────────────────────────────────────────

    def _cycle(self, cfg: Config) -> None:
        for attempt in range(3):
            if cfg.system.auto_learn:
                # find the float afresh on every cast: where it landed (the cast power varies) and how
                # it looks in the light right now — following an old picture locked onto rocks and glints
                self._bobber = self._landed = None
                self._per_cast = True
            self._nothing_landed = False
            self._landed_at = None
            self._cast(cfg, retry=attempt > 0)
            learned = self._bobber is not None
            self._last_try = attempt == 2
            base = self._wait_bobber(cfg)
            if base is not None or learned or not self._nothing_landed:
                break
            # nothing new on the water: the game ignored the press (still busy putting the last fish
            # away) or it reeled in a line that was already out — either way, just cast again
            self._emit("recast", "warn")
        if base is None:
            if learned or not self._nothing_landed:
                # the line is out even though the float never settled where we look: reel it in, or
                # the next "cast" press only pulls it back and fishing goes out of step. (When nothing
                # landed at all there is no line to reel in: a click would cast a short one instead.)
                self._click()
                self._sleep(0.8)
            self._bobber_misses += learned
            if self._bobber_misses >= 2:  # it looks different now (light, skin): learn it again
                self._forget_bobber()
            return self._fail(cfg, "no_bobber")
        self._bobber_misses = 0
        waited = self._wait_bite(cfg, base)
        if waited is None:
            self._click()  # reel the empty line in before recasting
            self._sleep(0.8)
            self._no_bites += 1
            if self._no_bites >= 3:  # tracking something that never sinks (a rock, a reflection)
                self._forget_bobber()
            return self._fail(cfg, "no_bite")
        self._no_bites = 0
        self.stats.bites += 1
        self.stats.bite_sum += waited
        self._emit("bite", s=round(waited, 1), why=self.last_trigger, **self.last_bite)
        self._hook(cfg)
        result, took = self._reel(cfg)
        if result != "no_game":
            self._learn_sound()
            if self._landed_at and self._target:  # a real bite: that was our float
                self._landings.append((self._landed_at[0] - self._target[0], self._landed_at[1] - self._target[1]))
        if result == "caught":
            s = self.stats
            s.catches += 1
            s.reel_sum += took
            s.last_reel_s = round(took, 2)
            s.fail_streak = 0
            self._escapes = 0
            s.win_streak += 1
            s.best_streak = max(s.best_streak, s.win_streak)
            self._avoid.clear()
            self._emit("caught", "success", s=round(took, 1))
            self._loot_due = True  # the banner is looked for during the pause before the next cast
        else:
            if result == "escaped":
                self.stats.escaped += 1
                self._escapes += 1
                if self._escapes >= 2 and cfg.system.auto_learn and self._cfg.reel.hold_moves != "auto":
                    self._adopt_cfg({"reel": {"hold_moves": "auto"}})  # maybe learned it wrong: re-check
                    self._escapes = 0
            self._fail(cfg, result)

    def _cast(self, cfg: Config, retry: bool = False) -> None:
        self._set_stage("cast")
        c = cfg.cast
        target = self._target if retry and self._target else self._pick_target(cfg)
        if target is not None:
            j = c.aim_jitter_px
            x = int(target[0] + self.rng.uniform(-j, j))
            y = int(target[1] + self.rng.uniform(-j, j))
            self.inp.move(x, y)
            self._sleep(self.rng.uniform(0.08, 0.16))
        else:
            x, y = self.inp.position()
        self._target = (x, y)
        if self._bobber is None:
            self._learn_ref = self._snap_water(x, y)
        hold = max(0.1, (c.power_ms + self.rng.uniform(-c.power_jitter_ms, c.power_jitter_ms)) / 1000)
        if not retry:  # a press the game ignored is not another cast
            self.stats.casts += 1
            self._emit("cast", n=self.stats.casts, x=x, y=y, ms=int(hold * 1000))
        self.inp.down()
        try:
            self._sleep(hold)
        finally:
            self.inp.up()

    def _pick_target(self, cfg: Config) -> tuple[int, int] | None:
        c = cfg.cast
        if c.target == "auto" and self._spot is not None:
            now = self.clock.now()
            self._avoid = [a for a in self._avoid if now - a[2] < 180]
            r = cfg.regions.water
            frame = self._grab(r)
            for m in self._spot.find_all(frame, c.spot_threshold):
                x, y = r.left + m.x, r.top + m.y
                if all((x - ax) ** 2 + (y - ay) ** 2 > 60 ** 2 for ax, ay, _ in self._avoid):
                    self._emit("spot", score=round(m.score, 2))
                    return int(x), int(y)
            self._emit("spot_none", "warn")
        if c.target in ("points", "auto") and c.points:
            p = c.points[self._point_index % len(c.points)]
            return int(p[0]), int(p[1])
        return None

    def _wait_bobber(self, cfg: Config) -> Match | None:
        self._set_stage("land")
        fresh = self._bobber is None
        if fresh:
            cfg = self._learn_bobber(cfg)
            if cfg is None:
                return None
        b, region = cfg.bite, cfg.regions.bobber
        t0 = self.clock.now()
        deadline = t0 + b.appear_timeout_s
        first = last_seen = None
        looked_wide = fresh  # just found where it landed: nothing to look for elsewhere
        settle = min(b.settle_ms, 500) if fresh else b.settle_ms  # the landing splash is already over
        samples: list[Match] = []
        frames: deque[np.ndarray] = deque(maxlen=5)
        while self.clock.now() < deadline:
            if first is None and not looked_wide and self.clock.now() - t0 > min(2.5, b.appear_timeout_s / 2):
                looked_wide = True  # not where it used to land: the player moved or aims elsewhere
                moved = self._relocate_bobber(cfg)
                if moved is not None:
                    region = moved
            t = self.clock.now()
            frame = self._grab(region)
            # just learned: it is where it landed — a look-alike elsewhere in the area is not it
            m = self._find_near(frame, *self._landed) if fresh and self._landed else self._bobber.find(frame)
            present = self._present(m, b)
            self._show(frame, "bobber", lambda img: draw_box(img, m, GREEN if present else RED,
                                                             f"{m.score:.2f}") if m else None)
            if present:
                if first is None:
                    first, samples = t, []
                last_seen = t
                samples.append(m)
                frames.append(frame)
                if t - first >= settle / 1000 and len(samples) >= 3:
                    tail = samples[-5:]
                    base = Match(statistics.median(s.x for s in tail), statistics.median(s.y for s in tail),
                                 m.w, m.h, statistics.median(s.score for s in tail), statistics.median(s.area for s in tail))
                    if fresh or base.score < 0.7:  # learned mid-splash, or the light changed: retake it
                        base = self._refine_bobber(list(frames), base) or base
                    self._emit("bobber")
                    return base
            elif first is not None and t - last_seen > 0.4:
                first = None  # lost it during the landing splash — start settling again
            self._tick()
            self._pace(t, cfg.system.idle_fps)
        return None

    def _bar_moved(self) -> bool:
        """Is a reel bar visible somewhere other than the learned area?"""
        mon, old = self._monitor_region(), self._cfg.regions.reel
        top = mon.top + mon.height // 3
        shot = self._grab(Region(mon.left, top, mon.width, mon.height - mon.height // 3))
        band = find_reel_band(shot) or find_green_bar(shot)
        if band is None:
            return False
        bx, by, bw, bh = band
        cx, cy = mon.left + bx + bw / 2, top + by + bh / 2
        return not (old.left <= cx <= old.left + old.width and old.top - bh <= cy <= old.top + old.height + bh)

    def _find_near(self, frame: np.ndarray, bx: float, by: float, base: Match) -> Match | None:
        """Best match of the float in a small window around where it floats (frame coordinates)."""
        if not isinstance(self._bobber, TemplateFinder):
            return self._bobber.find(frame)
        mx, my = max(base.w, 12), int(2.5 * base.h)
        x0, y0 = int(max(0, bx - base.w / 2 - mx)), int(max(0, by - base.h / 2 - base.h))
        x1, y1 = int(min(frame.shape[1], bx + base.w / 2 + mx)), int(min(frame.shape[0], by + base.h / 2 + my))
        m = self._bobber.find(frame[y0:y1, x0:x1])
        if m is None:
            return None
        return Match(m.x + x0, m.y + y0, m.w, m.h, m.score, m.area)

    def _refine_bobber(self, frames: list[np.ndarray], base: Match) -> Match | None:
        """Retake the float's picture from the settled float (no landing splash, current light)."""
        if not isinstance(self._bobber, TemplateFinder) or len(frames) < 3:
            return None
        med = np.median(np.stack(frames), axis=0).astype(np.uint8)
        x0, y0 = int(round(base.x - base.w / 2)), int(round(base.y - base.h / 2))
        tpl = med[max(0, y0):y0 + base.h, max(0, x0):x0 + base.w].copy()
        if tpl.shape[:2] != (base.h, base.w):
            return None
        try:
            finder = TemplateFinder(tpl, self._cfg.bite.grayscale, "bobber")
        except VisionError:
            return None
        found = [finder.find(f) for f in frames]
        scores = [f.score for f in found if f is not None]
        if len(scores) < len(frames) or min(scores) < 0.7:
            return None  # not steady enough to trust
        self._bobber = finder
        # the same float on another patch of water matches a little less well: leave room for that
        threshold = round(min(0.62, max(0.5, min(scores) * 0.6)), 2)
        self._adopt("bobber", tpl, {"bite": {"threshold": threshold}}, save=not self._per_cast)
        last = found[-1]
        return Match(base.x, base.y, base.w, base.h, statistics.median(scores), base.area or last.area)

    def _wide_area(self, x: int, y: int) -> Region:
        """The part of the screen a cast aimed at (x, y) can land in."""
        mon = self._monitor_region()
        w, h = int(mon.width * 0.6), int(mon.height * 0.55)
        left = int(min(max(x - w // 2, mon.left), mon.left + mon.width - w))
        top = int(min(max(y - h // 2, mon.top), mon.top + mon.height - h))
        return Region(left, top, w, h)

    def _relocate_bobber(self, cfg: Config) -> Region | None:
        """Look for the known bobber all around the cast point and move the search area there."""
        if self._target is None or not isinstance(self._bobber, TemplateFinder):
            return None
        area = self._wide_area(*self._target)
        m = self._bobber.find(self._grab(area))
        if m is None or m.score < cfg.bite.threshold:
            return None
        old, mon = cfg.regions.bobber, self._monitor_region()
        left = int(min(max(area.left + m.x - old.width / 2, mon.left), mon.left + mon.width - old.width))
        top = int(min(max(area.top + m.y - old.height / 2, mon.top), mon.top + mon.height - old.height))
        region = Region(left, top, old.width, old.height)
        self._adopt_cfg({"regions": {"bobber": asdict(region)}})
        self._emit("relocate")
        return region

    def _forget_bobber(self) -> None:
        self._bobber_misses = self._no_bites = 0
        if not self._cfg.system.auto_learn or self._bobber is None:
            return
        self._bobber = None
        self._emit("relearn_bobber", "warn")
        with self._lock:
            self._templates.pop("bobber", None)
        self._adopt_cfg({"regions": {"bobber": asdict(Region())}}, forget="bobber")

    def _wait_bite(self, cfg: Config, base: Match) -> float | None:
        """Hook on the bite: the burst of foam and bubbles that hits the float.

        That is how a bite looks in Albion — the float itself barely sinks, so "the float went
        under" is no signal (and its picture matching a little worse is no bite either: that was
        what hooked too early). The share of bright foam in a ring around the float is compared
        with its own calm level on this cast; a jump well above it for ``confirm_ms`` is the bite.
        The learned bite sound at that moment shortens the wait to a single frame. Waves, the
        float's bobbing, fish swimming by, music and other game sounds never hook on their own.
        """
        cfg = self._cfg  # may hold a freshly learned bobber area
        self._set_stage("bite", cfg.bite.bite_timeout_s)
        b, region = cfg.bite, cfg.regions.bobber
        t0 = self.clock.now()
        deadline, calm_until = t0 + b.bite_timeout_s, t0 + 0.3  # the landing splash is over by now
        bx, by = base.x, base.y
        # the ring is sized by the float (a picture that caught some splash too is larger than it)
        meter = SplashMeter(min(max(base.w, base.h), 40 * self._monitor_region().height / 1080))
        foam_since: float | None = None
        audio = self.audio if b.use_sound else None
        self._bite_t = None
        log = logging.getLogger("fishbot.bite")
        still: deque[tuple[float, float, float]] = deque(maxlen=int(5 * cfg.system.idle_fps))
        while self.clock.now() < deadline:
            t = self.clock.now()
            frame = self._grab(region)
            m = self._find_near(frame, bx, by, base)
            if m is not None and self._present(m, b) and abs(m.x - bx) < base.w and abs(m.y - by) < base.h:
                bx = 0.85 * bx + 0.15 * m.x  # follow the float as it drifts with the waves
                by = 0.85 * by + 0.15 * m.y
            if m is not None:
                still.append((m.x, m.y, m.score))
                if (self._per_cast and len(still) == still.maxlen and min(s for _, _, s in still) > 0.97
                        and np.ptp([x for x, _, _ in still]) < 0.5 and np.ptp([y for _, y, _ in still]) < 0.5):
                    log.info("the tracked float never moves: it is not the float")
                    return None  # a float always bobs; this is a rock, a reflection or a piece of UI
            box = (m.x - m.w / 2, m.y - m.h / 2, m.w, m.h) if m is not None and self._present(m, b) else None
            share = meter.share(frame, bx, by, box)
            limit = meter.threshold()
            foam = t >= calm_until and share is not None and meter.feed(t, share)
            if foam:
                foam_since = foam_since or t
            else:
                foam_since = None
            heard = foam and audio is not None and audio.bite_heard(t - 0.6, b.sound_prints)
            log.debug("t=%.2f share=%s limit=%s score=%.2f", t - t0, share and round(share, 3),
                      limit and round(limit, 3), m.score if m else 0.0)
            self._show(frame, "bobber", lambda img: self._draw_bite(img, m, base, by, foam))
            if foam_since is not None and (heard or t - foam_since >= b.confirm_ms / 1000):
                self.last_trigger = "sound" if heard else "splash"
                self._bite_t = foam_since
                self.last_bite = {"share": round(share, 3), "limit": round(limit, 3)}
                log.info("bite: foam %.3f over a limit of %.3f (calm %d frames)", share, limit, len(meter.calm))
                return t - t0
            self._tick()
            self._pace(t, cfg.system.idle_fps)
        return None

    def _hook(self, cfg: Config) -> None:
        self._set_stage("hook")
        b = cfg.bite
        self._sleep((b.hook_delay_ms + self.rng.uniform(0, b.hook_jitter_ms)) / 1000)
        if self._marker is None:
            self._learn_ref = self.screen.grab(self._monitor_region())
        self._click()

    def _click(self) -> None:
        self.inp.down()
        try:
            self._sleep(self.rng.uniform(0.04, 0.09))
        finally:
            self.inp.up()

    def _reel(self, cfg: Config, moved: bool = False) -> tuple[str, float]:
        self._set_stage("reel")
        try:  # the top of the screen without a loot banner, to spot the banner later
            self._loot_ref = (top_area(self._monitor_region()), None)
            self._loot_ref = (self._loot_ref[0], self._grab(Region(*self._loot_ref[0])))
        except Exception:
            self._loot_ref = None
        if self._marker is None:
            cfg = self._learn_reel(cfg)
            if cfg is None:
                return "no_game", 0.0
        rc, region = cfg.reel, cfg.regions.reel
        ctrl = ReelController(rc) if rc.hold_moves != "auto" else None
        t0 = self.clock.now()
        appear_by = t0 + rc.appear_timeout_s
        started = last_seen = last_x = last_zone = shown = None
        frames, result, frame = 0, None, None
        xs: list[float] = []
        learned = cfg.system.auto_learn
        self.reel_target = None
        rlog = logging.getLogger("fishbot.reel")
        try:
            while True:
                t = self.clock.now()
                frame = self._grab(region)
                m = self._marker.find(frame)
                if isinstance(self._marker, BarFinder) and self._marker.ends is None:
                    # a bar from the profile: learn how its ends look once it stopped zooming in
                    if self._marker.present:
                        shown = shown or t
                        if t - shown >= 0.3:
                            self._marker.remember(frame)
                    else:
                        shown = None
                if self._present(m, rc):
                    span = max(1.0, frame.shape[1] - (m.w if rc.method == "template" else 0))
                    x = (m.x - (m.w / 2 if rc.method == "template" else 0)) / span
                    x = min(1.0, max(0.0, x))
                    if started is None:
                        started = t
                        self.stats.hooks += 1
                        self._emit("reel_start")
                    if ctrl is None:  # first minigame: find out which way holding pushes the marker
                        rc = replace(rc, hold_moves=self._probe_hold(region, rc, span,
                                                                     m.w if rc.method == "template" else 0))
                        ctrl = ReelController(rc)
                        t = self.clock.now()  # the probe took a while: don't take that for the end
                    last_seen, last_x, self.reel_x = t, x, x
                    # keep the marker in the middle of the green zone, wherever it is
                    zone = self._marker.zone if isinstance(self._marker, BarFinder) else green_span(frame)
                    last_zone = zone or last_zone
                    tgt = band = None
                    if zone is not None:
                        W = frame.shape[1]
                        off = m.w / 2 if rc.method == "template" else 0
                        tgt = min(1.0, max(0.0, ((zone[0] + zone[1]) / 2 * W - off) / span))
                        band = max(0.02, (zone[1] - zone[0]) * W / span * 0.3)
                    self.reel_target = tgt
                    xs.append(x)
                    if (learned and rc.method == "template" and len(xs) >= 40 and max(xs) - min(xs) < 0.006
                            and t - started > 1.2):  # "marker" never moves: we learned the wrong thing
                        self._forget_bar()
                        return "no_game", 0.0
                    hold = ctrl.update(x, t, tgt, band)
                    if hold:
                        self.inp.down()
                    else:
                        self.inp.up()
                    rlog.debug("t=%.2f x=%.3f target=%s zone=%s hold=%d", t - t0, x, tgt and round(tgt, 3),
                               zone and tuple(round(z, 2) for z in zone), hold)
                elif started is None:
                    if not moved and learned and t - t0 > 1.0 and self._bar_moved():
                        # the minigame is up, just not where it was (resolution, UI scale): learn it now
                        self._forget_bar()
                        return self._reel(self._cfg, moved=True)
                    if t > appear_by:
                        return "no_game", 0.0
                elif isinstance(self._marker, BarFinder) and self._marker.present:
                    last_seen = t  # the band is still up, the float just wasn't made out this frame
                elif t - last_seen > (rc.end_confirm_ms / 1000 if t - started > 1.0 else 1.0):
                    break  # the band itself is gone: the minigame is over (it zooms in for a moment first)
                if started is not None and t - started > rc.max_duration_s:
                    result = "timeout"
                    break
                self._show(frame, "reel", lambda img, rc=rc: draw_reel(img, self.reel_x, self.reel_target or rc.target, rc.deadband,
                                                                self.inp.held))
                frames += 1
                self._tick()
                self._pace(t, rc.fps)
        finally:
            self.inp.up()
            self.reel_x = None
            elapsed = self.clock.now() - t0
            if frames and elapsed > 0:
                self.stats.fps = round(frames / elapsed, 1)
        took = (last_seen - started) if started is not None else 0.0
        if result:
            return result, took
        # the fish gets away once the float is dragged out of the green into a red end
        lo, hi = (last_zone[0] - 0.02, last_zone[1] + 0.02) if last_zone else (0.08, 0.92)
        outcome = "escaped" if last_x is not None and (last_x < lo or last_x > hi) else "caught"
        logging.getLogger("fishbot.reel").info("minigame over after %.1fs, marker last at %s: %s", took,
                                               last_x and round(last_x, 3), outcome)
        if outcome == "escaped" and frame is not None:
            self._debug("escaped", frame)
        return outcome, took

    def _read_loot(self, budget: float) -> None:
        """Find the "you received" banner and tally it (OCR runs off the fishing thread)."""
        if not self._loot_ref or self._loot_ref[1] is None:
            return
        (x, y, w, h), before = self._loot_ref
        end = self.clock.now() + min(2.0, budget)
        while self.clock.now() < end:
            self._sleep(0.15)
            banner = find_banner(before, self._grab(Region(x, y, w, h)))
            if banner is not None:
                self._sleep(0.25)  # let its fade-in finish
                later = find_banner(before, self._grab(Region(x, y, w, h)))
                banner = later if later is not None else banner
                threading.Thread(target=self._tally, args=(banner,), daemon=True).start()
                return

    def _tally(self, banner) -> None:
        try:
            item = self.catches.add(banner)
            self._emit("loot", "success", name=item["name"] or "?", count=item["count"])
        except Exception:
            traceback.print_exc()

    def _learn_sound(self) -> None:
        """The minigame showed up, so that was a real bite: remember what it sounded like."""
        cfg = self._cfg
        if self.audio is None or self._bite_t is None or not cfg.bite.use_sound or not cfg.system.auto_learn:
            return
        prints = self.audio.remember(self._bite_t, cfg.bite.sound_prints)
        if prints is not None:
            first = len(cfg.bite.sound_prints) < 2 <= len(prints)
            self._adopt_cfg({"bite": {"sound_prints": prints}})
            if first:
                self._emit("learn_sound", "success")

    def _probe_hold(self, region, rc, span: float, mw: int) -> str:
        """Release, then hold, and compare how the marker accelerates (velocity alone is fooled by inertia)."""
        def accel(hold: bool, dur: float) -> float | None:
            (self.inp.down if hold else self.inp.up)()
            pts, end = [], self.clock.now() + dur
            while self.clock.now() < end:
                m = self._marker.find(self._grab(region))
                if self._present(m, rc):
                    pts.append((self.clock.now(), (m.x - mw / 2) / span))
                self.clock.sleep(1 / 90)
            if len(pts) < 5:
                return None
            ts, xs = np.array(pts).T
            return float(2 * np.polyfit(ts - ts[0], xs, 2)[0])
        samples = []
        for _ in range(2):  # release/hold twice: a fish tug can't fake both
            r, h = accel(False, 0.14), accel(True, 0.14)
            if r is not None and h is not None:
                samples.append(h - r)
        self.inp.up()
        logging.getLogger("fishbot.reel").info("hold probe: %s", [round(v, 2) for v in samples])
        if not samples or abs(float(np.mean(samples))) < 0.4 or len(set(np.sign(samples))) > 1:
            return "right"  # couldn't tell: common default for now, ask again next fish
        way = "right" if np.mean(samples) > 0 else "left"
        self._adopt_cfg({"reel": {"hold_moves": way}})
        self._emit("learn_hold", way=way)
        return way

    # ── zero-setup learning ─────────────────────────────────────────────────

    def _monitor_region(self) -> Region:
        x, y = self.inp.position()
        mons = self.screen.monitors()
        for m in mons[1:] or mons:
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return Region(m["left"], m["top"], m["width"], m["height"])
        m = mons[1] if len(mons) > 1 else mons[0]
        return Region(m["left"], m["top"], m["width"], m["height"])

    def _snap_water(self, x: int, y: int):
        region = self._wide_area(x, y)  # the cast can land well past the cursor
        left, top = region.left, region.top
        if self._landings:  # it lands about where it landed before: look there first (not at a neighbour's)
            x = x + statistics.median(dx for dx, _ in self._landings)
            y = y + statistics.median(dy for _, dy in self._landings)
        frames = []
        for _ in range(8):  # ~0.6 s: long enough to see how far the water around moves on its own
            frames.append(self._grab(region))
            self._sleep(0.08)
        return region, frames, (x - left, y - top), self._monitor_region().height / 1080

    def _learn_bobber(self, cfg: Config) -> Config | None:
        region, before, near, scale = self._learn_ref
        self._sleep(max(1.8, cfg.bite.settle_ms / 1000))  # let it fly, land and the landing splash settle
        after = []
        for _ in range(5):
            after.append(self._grab(region))
            self._sleep(0.08)
        # the float lands around the aim point (nearer or farther with the cast power), never far off
        found = find_new_object(before, after, near, scale, reach=0.25 * self._monitor_region().height)
        self._nothing_landed = found is None
        if found is None:
            if self._last_try:
                self._emit("learn_bobber_fail", "warn")
                self._debug("bobber-after", after[-1])
            return None
        tpl, (bx, by, bw, bh) = found
        # search only around where it landed: faster and no look-alikes elsewhere
        mon = self._monitor_region()
        sw, sh = max(bw * 8, int(mon.width * 0.22)), max(bh * 8, int(mon.height * 0.2))
        cx, cy = region.left + bx + bw // 2, region.top + by + bh // 2
        area = Region(int(min(max(cx - sw // 2, mon.left), mon.left + mon.width - sw)),
                      int(min(max(cy - sh // 2, mon.top), mon.top + mon.height - sh)), sw, sh)
        self._landed = (float(cx - area.left), float(cy - area.top), Match(cx - area.left, cy - area.top, bw, bh, 1.0, 0))
        self._landed_at = (float(cx), float(cy))
        logging.getLogger("fishbot.bite").info("float found at (%d, %d), %d×%d", cx, cy, bw, bh)
        shot = after[-1].copy()
        cv2.rectangle(shot, (bx, by), (bx + bw, by + bh), (99, 230, 245), 1)
        self._debug("float", shot)
        self._bobber = TemplateFinder(tpl, cfg.bite.grayscale, "bobber")
        # how well does it match itself frame to frame? set the threshold from that
        scores = [self._bobber.find(f).score for f in after]
        threshold = round(min(0.62, max(0.45, min(scores) * 0.75)), 2)
        cfg = self._adopt("bobber", tpl, {"regions": {"bobber": asdict(area)},
                                          "bite": {"method": "template", "threshold": threshold}},
                          save=not self._per_cast or not self._float_known)
        if not self._float_known:
            self._float_known = True
            self._emit("learn_bobber", "success", w=tpl.shape[1], h=tpl.shape[0])
        return cfg

    def _learn_reel(self, cfg: Config) -> Config | None:
        mon = self._monitor_region()
        before = after = self._learn_ref
        rejected: list[tuple[int, int]] = []
        deadline = self.clock.now() + max(2.5, cfg.reel.appear_timeout_s)
        last = None
        while self.clock.now() < deadline:
            self._sleep(0.04 if last is not None else 0.12)  # a band in sight: look again soon
            after = self._grab(mon)
            # Albion's band: red chevron ends, see-through green middle, bobber on top, blue progress bar under
            band = find_reel_band(after) or find_green_bar(after)
            # it zooms in when it appears: learn it once it stopped growing, not mid-animation
            steady = (band is not None and last is not None and abs(band[0] - last[0]) <= 3
                      and abs(band[2] - last[2]) <= max(3, 0.03 * band[2]))
            last = band
            if steady:
                bx, by, bw, bh = band
                top = max(0, by - bh)
                region = Region(mon.left + bx, mon.top + top, bw, by + bh - top + 1)
                finder = BarFinder()
                shot = self._grab(region)
                finder.remember(shot)
                if find_float(after[top:by + bh + 1, bx:bx + bw]) is not None and finder.find(shot):
                    self._marker = finder
                    self._adopt_cfg({"regions": {"reel": asdict(region)}, "reel": {"method": "bar"}}, forget="marker")
                    cfg = self._cfg
                    self._emit("learn_bar", "success", w=bw)
                    shot = after.copy()
                    cv2.rectangle(shot, (bx, top), (bx + bw, by + bh), (99, 230, 245), 2)
                    self._debug("bar-learned", shot)
                    return cfg
            if band is not None:
                continue  # Albion's band is in sight, just not steady yet: look again
            bar = find_new_bar(before, after) if before is not None else None
            if bar is None:
                continue
            bx, by, bw, bh = bar
            found = find_marker(after[by:by + bh, bx:bx + bw])
            if found is None:
                continue
            tpl, _ = found
            region = Region(mon.left + bx, mon.top + by, bw, bh)
            if any(abs(bx - rx) < 12 and abs(by - ry) < 12 for rx, ry in rejected):
                continue
            try:
                finder = TemplateFinder(tpl, cfg.reel.grayscale, "marker")
            except VisionError:
                continue
            xs, scores = [], []
            for _ in range(8):  # the real marker moves; a static icon or text does not
                self._sleep(0.05)
                mm = finder.find(self._grab(region))
                if mm is not None:
                    xs.append(mm.x)
                    scores.append(mm.score)
            if len(xs) < 5 or max(xs) - min(xs) < 3:
                rejected.append((bx, by))
                self._debug("bar-static", after)
                continue
            self._marker = finder
            threshold = round(min(0.62, max(0.4, min(scores) * 0.75)), 2)  # it changes background as it moves
            cfg = self._adopt("marker", tpl, {"regions": {"reel": asdict(region)},
                                              "reel": {"method": "template", "threshold": threshold}})
            self._emit("learn_bar", "success", w=bw)
            shot = after.copy()
            cv2.rectangle(shot, (bx, by), (bx + bw, by + bh), (99, 230, 245), 2)
            self._debug("bar-learned", shot)
            return cfg
        self._emit("learn_bar_fail", "warn")
        self._debug("bar-before", before)
        self._debug("bar-after", after)
        return None

    def _relearn_all(self) -> None:
        """A fresh Start: the player may stand elsewhere, use another rod or another UI scale —
        forget everything learned and learn it again while fishing (nothing to do by hand)."""
        self._bobber = self._marker = None
        self._bobber_misses = self._no_bites = 0
        self._escapes = 0
        with self._lock:
            self._templates.pop("bobber", None)
            self._templates.pop("marker", None)
        fresh = Config()
        self._adopt_cfg({}, forget="marker")
        self._adopt_cfg({"regions": {"bobber": asdict(Region()), "reel": asdict(Region())},
                         "bite": {"method": "template", "threshold": fresh.bite.threshold, "sound_prints": []},
                         "reel": {"method": "bar", "hold_moves": fresh.reel.hold_moves}}, forget="bobber")
        self._dirty = True

    def _forget_bar(self) -> None:
        self._marker = None
        self._emit("relearn_bar", "warn")
        with self._lock:
            self._templates.pop("marker", None)
        self._adopt_cfg({"regions": {"reel": asdict(Region())}}, forget="marker")

    def _adopt_cfg(self, patch: dict, forget: str | None = None) -> None:
        with self._lock:
            self._cfg = self._cfg.merged(patch)
        if self.on_learn:
            try:
                self.on_learn(forget, None, patch)
            except Exception:
                traceback.print_exc()

    def _debug(self, tag: str, img) -> None:
        """Keep a few snapshots of what went wrong so the user can send them."""
        if not self.debug_dir:
            return
        try:
            from .config import write_image
            d = self.debug_dir
            d.mkdir(parents=True, exist_ok=True)
            if img.shape[1] > 1600:
                img = cv2.resize(img, (1600, int(img.shape[0] * 1600 / img.shape[1])), interpolation=cv2.INTER_AREA)
            write_image(d / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}.png", img)
            for old in sorted(d.glob("*.png"))[:-12]:
                old.unlink(missing_ok=True)
        except Exception:
            traceback.print_exc()

    def _adopt(self, name: str, tpl, patch: dict, save: bool = True) -> Config:
        """Use a learned template now and hand it to the owner for saving."""
        with self._lock:
            self._templates[name] = tpl
            self._cfg = self._cfg.merged(patch)
            cfg = self._cfg
        if self.on_learn and save:
            try:
                self.on_learn(name, tpl, patch)
            except Exception:
                traceback.print_exc()
        return cfg

    # ── session plumbing ────────────────────────────────────────────────────

    def _fail(self, cfg: Config, reason: str) -> None:
        s = self.stats
        if reason in ("no_bobber", "no_bite", "no_game"):
            s.misses += 1
        s.fail_streak += 1
        s.win_streak = 0
        self._emit("fail", "warn", reason=reason)
        if cfg.cast.target != "cursor" and s.fail_streak % cfg.cast.rotate_after == 0:
            self._point_index += 1
            if self._target:
                self._avoid.append((*self._target, self.clock.now()))
            self._emit("rotate")

    def _session_gate(self, cfg: Config) -> None:
        s = cfg.session
        if s.max_catches and self.stats.catches >= s.max_catches:
            self._emit("limit_catches", "success", n=s.max_catches)
            raise _Stop
        active = self.active_seconds()
        if s.max_minutes and active >= s.max_minutes * 60:
            self._emit("limit_time", "success", n=s.max_minutes)
            raise _Stop
        if not s.break_every_min:
            self._next_break = 0.0
            return
        if not self._next_break:
            self._next_break = active + s.break_every_min * 60 * self.rng.uniform(0.85, 1.15)
        if active >= self._next_break:
            dur = s.break_minutes * 60 * self.rng.uniform(0.8, 1.25)
            # schedule the next one first: pausing + resuming mid-break skips the rest of it
            self._next_break = active + dur + s.break_every_min * 60 * self.rng.uniform(0.85, 1.15)
            self._set_stage("break", dur)
            self._emit("break", min=round(dur / 60, 1))
            self._sleep(dur)
            self._emit("break_end")

    def _run_actions(self, cfg: Config) -> None:
        now = self.clock.now()
        for i, a in enumerate(cfg.session.actions):
            if not a.enabled or not a.key:
                continue
            last = self._action_last.get(i)
            if last is not None and now - last < a.every_min * 60:
                continue
            self._set_stage("action")
            self._action_last[i] = now
            try:
                self.inp.key(a.key)
                self._emit("action", key=a.key.upper(), label=a.label)
            except Exception as e:
                self._emit("action_error", "warn", key=a.key, error=str(e))
            self._sleep(1.5)

    def _cooldown(self, cfg: Config) -> None:
        s = cfg.session
        dur = (s.cooldown_ms + self.rng.uniform(0, s.cooldown_jitter_ms)) / 1000
        if self._loot_due:
            dur += 1.0  # the character puts the fish away first; a cast before that is ignored
        self._set_stage("cooldown", dur)
        end = self.clock.now() + dur
        if self._loot_due:  # read the catch banner within the pause, not on top of it
            self._loot_due = False
            self._read_loot(max(dur, 0.5))
        self._sleep(max(0.0, end - self.clock.now()))

    def _use_bait(self, cfg: Config) -> None:
        """Bait in the potion slot: use it at the start, then again once it runs out."""
        b = cfg.bait
        if not b.enabled or not b.key:
            return
        now, n = self.clock.now(), self.stats.catches
        self._bait_catches = min(self._bait_catches, n)  # stats were reset
        due = (self._bait_at is None
               or (b.every_catches and n - self._bait_catches >= b.every_catches)
               or (b.every_min and now - self._bait_at >= b.every_min * 60))
        if not due:
            return
        self._set_stage("bait")
        self._bait_at, self._bait_catches = now, n
        try:
            self.inp.key(b.key)
            self._emit("bait", key=b.key.upper())
        except Exception as e:
            self._emit("action_error", "warn", key=b.key, error=str(e))
        self._sleep(b.delay_ms / 1000)

    def _countdown(self) -> None:
        delay = self._cfg.system.start_delay_s
        if delay > 0:
            self._set_stage("countdown", delay)
            self._emit("countdown", s=delay)
            self._sleep(delay)

    def _wait_resume(self) -> None:
        if self._active_t0 is not None:
            self._active_acc += self.clock.now() - self._active_t0
            self._active_t0 = None
        self.status = "paused"
        self._set_stage("paused")
        self._emit("paused" if self.pause_reason != "focus" else "focus_lost", "warn")
        while True:
            if self._stop.is_set():
                raise _Stop
            if not self._pause.is_set():
                break
            if self.pause_reason == "focus" and self._focused(self._cfg):
                self._pause.clear()
                self.pause_reason = ""
                self._emit("focus_back")
                break
            self.clock.sleep(0.1)
        self.status = "running"
        self._active_t0 = self.clock.now()
        self._emit("resumed")

    # ── helpers ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._stop.is_set():
            raise _Stop
        cfg = self._cfg
        now = self.clock.now()
        if cfg.system.failsafe and now >= self._next_failsafe_check:
            self._next_failsafe_check = now + 0.1
            try:
                x, y = self.inp.position()
            except Exception:
                x = y = 99
            if abs(x) <= 1 and abs(y) <= 1:
                self._emit("failsafe", "warn")
                raise _Stop
        if cfg.system.require_focus and now >= self._next_focus_check and not self._pause.is_set():
            self._next_focus_check = now + 0.25
            if not self._focused(cfg):
                self.pause_reason = "focus"
                self._pause.set()
        if self._pause.is_set():
            raise _Pause

    def _focused(self, cfg: Config) -> bool:
        title = self._focus_fn() if self._focus_fn else None
        return title is None or cfg.system.window_title.lower() in title.lower()

    def _sleep(self, dt: float) -> None:
        end = self.clock.now() + dt
        while True:
            self._tick()
            rem = end - self.clock.now()
            if rem <= 0:
                return
            self.clock.sleep(min(rem, 0.05))

    def _pace(self, t_start: float, fps: int) -> None:
        self.clock.sleep(1.0 / fps - (self.clock.now() - t_start))

    def _grab(self, region) -> np.ndarray:
        for attempt in range(3):  # BitBlt can fail transiently (UAC prompt, lock screen)
            try:
                return self.screen.grab(region)
            except Exception:
                if attempt == 2:
                    raise
                self._sleep(0.05)
        raise AssertionError("unreachable")

    @staticmethod
    def _present(m: Match | None, sub) -> bool:
        return m is not None and m.score >= sub.threshold

    def _set_stage(self, stage: str, duration: float | None = None) -> None:
        self.stage = stage
        self.stage_ends = self.clock.now() + duration if duration else None

    def report(self, err: VisionError) -> None:
        """Surface a calibration problem in the log (e.g. Start pressed via hotkey)."""
        self._emit("vision_error", "error", code=err.code, name=err.params.get("name", ""))

    def _emit(self, code: str, level: str = "info", **params) -> None:
        with self._lock:
            self._log_id += 1
            event = {"id": self._log_id, "ts": time.time(), "t": round(self.clock.now(), 3), "level": level, "code": code, "params": params}
            self.logs.append(event)
        logging.getLogger("fishbot.events").info(format_event(event))
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _show(self, frame: np.ndarray, kind: str, draw) -> None:
        now = time.monotonic()
        if now > self.preview_wanted_until or now - self._preview_at < 1 / 15:
            return
        self._preview_at = now
        img = frame.copy()
        draw(img)
        with self._lock:
            self._preview = (img, kind, now)

    @staticmethod
    def _draw_bite(img: np.ndarray, m: Match | None, base: Match, by: float, alarm: bool) -> None:
        import cv2
        y = int(by)
        cv2.line(img, (0, y), (img.shape[1], y), YELLOW, 1, cv2.LINE_AA)
        if m is not None:
            draw_box(img, m, PINK if alarm else BLUE, f"{m.score:.2f}")
