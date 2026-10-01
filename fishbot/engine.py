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
from .learn import find_float, find_green_bar, find_marker, find_new_bar, find_new_object, green_span
from .controls import Input
from .system import beep, foreground_title
from .vision import (BarFinder, BLUE, GREEN, PINK, RED, YELLOW, Match, TemplateFinder, VisionError, draw_box,
                     draw_reel, make_finder)


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
    "bite": "Bite after {s}s",
    "reel_start": "Reeling…",
    "caught": "Caught! Reel took {s}s",
    "fail": "Missed: {reason}",
    "rotate": "Switching fishing spot",
    "spot": "Fishing spot found (score {score})",
    "spot_none": "No fishing spot visible",
    "fail_streak": "Too many misses in a row ({n}) — stopping",
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
    "relearn_bar": "The learned reel bar never moves — learning it again",
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
        self.debug_dir = None  # Path for failure snapshots
        self._escapes = 0
        self.reel_target = None
        self.catches = CatchLog()
        self._loot_ref = None
        self.last_trigger = ""
        self._learn_ref = None

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
        self.validate()
        self._stop.clear()
        self._pause.clear()
        self.stats = Stats()
        self.catches.reset()
        self._active_acc, self._active_t0 = 0.0, None
        self._action_last.clear()
        self._avoid.clear()
        self._point_index = 0
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
        self._bobber, self._marker, self._spot = bobber, marker, spot

    # ── one fishing cycle ───────────────────────────────────────────────────

    def _cycle(self, cfg: Config) -> None:
        self._cast(cfg)
        base = self._wait_bobber(cfg)
        if base is None:
            return self._fail(cfg, "no_bobber")
        waited = self._wait_bite(cfg, base)
        if waited is None:
            self._click()  # reel the empty line in before recasting
            self._sleep(0.8)
            return self._fail(cfg, "no_bite")
        self.stats.bites += 1
        self.stats.bite_sum += waited
        self._emit("bite", s=round(waited, 1))
        self._hook(cfg)
        result, took = self._reel(cfg)
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
            self._read_loot()
        else:
            if result == "escaped":
                self.stats.escaped += 1
                self._escapes += 1
                if self._escapes >= 2 and cfg.system.auto_learn and self._cfg.reel.hold_moves != "auto":
                    self._adopt_cfg({"reel": {"hold_moves": "auto"}})  # maybe learned it wrong: re-check
                    self._escapes = 0
            self._fail(cfg, result)

    def _cast(self, cfg: Config) -> None:
        self._set_stage("cast")
        c = cfg.cast
        target = self._pick_target(cfg)
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
        if self._bobber is None:
            cfg = self._learn_bobber(cfg)
            if cfg is None:
                return None
        b, region = cfg.bite, cfg.regions.bobber
        deadline = self.clock.now() + b.appear_timeout_s
        first = last_seen = None
        samples: list[Match] = []
        while self.clock.now() < deadline:
            t = self.clock.now()
            frame = self._grab(region)
            m = self._bobber.find(frame)
            present = self._present(m, b)
            self._show(frame, "bobber", lambda img: draw_box(img, m, GREEN if present else RED,
                                                             f"{m.score:.2f}") if m else None)
            if present:
                if first is None:
                    first, samples = t, []
                last_seen = t
                samples.append(m)
                if t - first >= b.settle_ms / 1000 and len(samples) >= 3:
                    tail = samples[-5:]
                    self._emit("bobber")
                    return Match(statistics.median(s.x for s in tail), statistics.median(s.y for s in tail),
                                 m.w, m.h, m.score, statistics.median(s.area for s in tail))
            elif first is not None and t - last_seen > 0.4:
                first = None  # lost it during the landing splash — start settling again
            self._tick()
            self._pace(t, cfg.system.idle_fps)
        return None

    def _wait_bite(self, cfg: Config, base: Match) -> float | None:
        """React to the bite itself: any of score drop, dip, vanish or a splash of motion."""
        cfg = self._cfg  # may hold a freshly learned bobber area
        self._set_stage("bite", cfg.bite.bite_timeout_s)
        b, region = cfg.bite, cfg.regions.bobber
        t0 = self.clock.now()
        deadline = t0 + b.bite_timeout_s
        streak, by, score_ref = 0, base.y, base.score
        energies: deque[float] = deque(maxlen=45)
        prev = None
        audio = self.audio if b.use_sound else None
        heard_after = (audio.now() + 1.0) if audio else 0.0  # ignore the landing splash
        while self.clock.now() < deadline:
            t = self.clock.now()
            frame = self._grab(region)
            m = self._bobber.find(frame)
            present = self._present(m, b)
            lost = not present or (b.method == "color" and m.area < base.area * 0.35)
            dip = present and (m.y - by) >= b.dip_px
            drop = present and b.method == "template" and m.score < score_ref - 0.22
            # splash: sudden motion right around the bobber compared to the usual waves
            x0, y0 = int(max(0, base.x - 2.5 * base.w)), int(max(0, base.y - 2.5 * base.h))
            x1, y1 = int(base.x + 2.5 * base.w), int(base.y + 2.5 * base.h)
            patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
            splash = False
            if prev is not None and prev.shape == patch.shape and patch.size:
                e = float(np.abs(patch - prev).mean())
                if len(energies) >= 20:
                    med = float(np.median(energies))
                    mad = float(np.median(np.abs(np.asarray(energies) - med)))
                    splash = e > med + 6 * mad + 3.0
                if not splash:
                    energies.append(e)
            prev = patch
            if audio is not None and audio.onset_after(heard_after):
                self.last_trigger = "sound"  # the bite splash is audible before it's visible
                return self.clock.now() - t0
            if lost or dip or drop or splash:
                streak += 1
            else:
                streak = 0
                by = 0.92 * by + 0.08 * m.y  # follow slow drift with the waves
                score_ref = 0.95 * score_ref + 0.05 * m.score
            self._show(frame, "bobber", lambda img: self._draw_bite(img, m, base, by, streak > 0))
            if streak >= b.confirm_frames:
                self.last_trigger = "lost" if lost else "dip" if dip else "drop" if drop else "splash"
                return self.clock.now() - t0
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

    def _reel(self, cfg: Config) -> tuple[str, float]:
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
        started = last_seen = last_x = None
        frames, result, frame = 0, None, None
        xs: list[float] = []
        learned = cfg.system.auto_learn
        self.reel_target = None
        try:
            while True:
                t = self.clock.now()
                frame = self._grab(region)
                m = self._marker.find(frame)
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
                        t = self.clock.now()
                    last_seen, last_x, self.reel_x = t, x, x
                    # keep the marker in the middle of the green zone, wherever it is
                    zone = self._marker.zone if isinstance(self._marker, BarFinder) else green_span(frame)
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
                    if ctrl.update(x, t, tgt, band):
                        self.inp.down()
                    else:
                        self.inp.up()
                elif started is None:
                    if t > appear_by:
                        return "no_game", 0.0
                elif t - last_seen > rc.end_confirm_ms / 1000:
                    break
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
        outcome = "escaped" if last_x is not None and (last_x < 0.03 or last_x > 0.97) else "caught"
        if outcome == "escaped" and frame is not None:
            self._debug("escaped", frame)
        return outcome, took

    def _read_loot(self) -> None:
        """Find the "you received" banner and tally it (OCR runs off the fishing thread)."""
        if not self._loot_ref or self._loot_ref[1] is None:
            return
        (x, y, w, h), before = self._loot_ref
        end = self.clock.now() + 2.0
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
        mon = self._monitor_region()
        w, h = int(mon.width * 0.6), int(mon.height * 0.55)  # the cast can land well past the cursor
        left = int(min(max(x - w // 2, mon.left), mon.left + mon.width - w))
        top = int(min(max(y - h // 2, mon.top), mon.top + mon.height - h))
        region = Region(left, top, w, h)
        frames = []
        for _ in range(3):
            frames.append(self._grab(region))
            self._sleep(0.06)
        return region, frames, (x - left, y - top), mon.height / 1080

    def _learn_bobber(self, cfg: Config) -> Config | None:
        region, before, near, scale = self._learn_ref
        self._sleep(max(1.4, cfg.bite.settle_ms / 1000))  # let it fly and land
        after = []
        for _ in range(5):
            after.append(self._grab(region))
            self._sleep(0.08)
        found = find_new_object(before, after, near, scale)
        if found is None:
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
        self._bobber = TemplateFinder(tpl, cfg.bite.grayscale, "bobber")
        # how well does it match itself frame to frame? set the threshold from that
        scores = [self._bobber.find(f).score for f in after]
        threshold = round(min(0.62, max(0.45, min(scores) * 0.75)), 2)
        cfg = self._adopt("bobber", tpl, {"regions": {"bobber": asdict(area)},
                                          "bite": {"method": "template", "threshold": threshold}})
        self._emit("learn_bobber", "success", w=tpl.shape[1], h=tpl.shape[0])
        return cfg

    def _learn_reel(self, cfg: Config) -> Config | None:
        mon = self._monitor_region()
        before = after = self._learn_ref
        rejected: list[tuple[int, int]] = []
        deadline = self.clock.now() + max(2.5, cfg.reel.appear_timeout_s)
        while self.clock.now() < deadline:
            self._sleep(0.12)
            after = self._grab(mon)
            band = find_green_bar(after)  # Albion's band: green middle, red chevron ends, bobber on top
            if band is not None:
                bx, by, bw, bh = band
                top = max(0, by - bh)
                region = Region(mon.left + bx, mon.top + top, bw, by + bh - top + 1)
                finder = BarFinder()
                if find_float(after[top:by + bh + 1, bx:bx + bw]) is not None and finder.find(self._grab(region)):
                    self._marker = finder
                    self._adopt_cfg({"regions": {"reel": asdict(region)}, "reel": {"method": "bar"}}, forget="marker")
                    cfg = self._cfg
                    self._emit("learn_bar", "success", w=bw)
                    shot = after.copy()
                    cv2.rectangle(shot, (bx, top), (bx + bw, by + bh), (99, 230, 245), 2)
                    self._debug("bar-learned", shot)
                    return cfg
            bar = find_new_bar(before, after)
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

    def _adopt(self, name: str, tpl, patch: dict) -> Config:
        """Use a learned template now and hand it to the owner for saving."""
        with self._lock:
            self._templates[name] = tpl
            self._cfg = self._cfg.merged(patch)
            cfg = self._cfg
        if self.on_learn:
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
        if s.fail_streak >= cfg.session.max_fail_streak:
            self._emit("fail_streak", "error", n=s.fail_streak)
            raise _Stop

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
        self._set_stage("cooldown", dur)
        self._sleep(dur)

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
            event = {"id": self._log_id, "ts": time.time(), "level": level, "code": code, "params": params}
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
