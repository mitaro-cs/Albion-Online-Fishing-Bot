"""Backend API consumed by the web UI (see server.py for the transport)."""
from __future__ import annotations

import base64
import functools
import logging
import sys
import threading
import time
from dataclasses import asdict

import cv2
import numpy as np

from . import __version__
from .capture import Screen
from .config import TEMPLATES, Config, ProfileStore, Region, schema
from .controls import Input
from .engine import Engine
from .hotkeys import Hotkeys
from .updater import Updater
from .setup import STEPS, SetupError, find_bobber, find_marker
from .system import foreground_title, open_path
from .vision import VisionError, to_data_url

log = logging.getLogger(__name__)
PUBLIC: set[str] = set()


def endpoint(fn):
    """Expose a method to the UI and turn exceptions into ``{ok: false, error}``."""
    PUBLIC.add(fn.__name__)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_gone", False):  # uninstalled: never recreate files
            return {"ok": False, "error": "offline"}
        try:
            out = fn(self, *args, **kwargs)
            return {"ok": True, **(out or {})}
        except VisionError as e:
            return {"ok": False, "error": e.code, "params": e.params}
        except (ValueError, TypeError, KeyError, FileNotFoundError) as e:
            return {"ok": False, "error": "invalid", "message": str(e)}
        except Exception as e:
            log.exception("api %s failed", fn.__name__)
            return {"ok": False, "error": "internal", "message": f"{type(e).__name__}: {e}"}
    return wrapper


class Api:
    def __init__(self, store: ProfileStore, screen: Screen, inp: Input, demo: bool = False,
                 focus=foreground_title, profile: str | None = None, hotkeys: bool = True, audio: bool = False):
        self._store, self._screen, self._demo = store, screen, demo
        self._engine = Engine(screen, inp, focus=focus)
        self._engine.on_learn = self._learned
        self._engine.debug_dir = store.root / "debug"
        self._lock = threading.RLock()
        self._capture: tuple[np.ndarray, dict] | None = None
        self._window = None
        settings = store.settings()
        name = profile or settings.get("profile")
        self._profile = name if name and store.exists(name) else store.list()[0]
        self._hotkeys = Hotkeys()
        self._setup = {"active": False, "done": False, "step": 0, "error": "", "busy": False, "left": None, "rev": 0}
        self._mark_lock = threading.Lock()
        if hotkeys:
            self._hotkeys.start()
        self._load()
        self.quit = threading.Event()
        self._updater = Updater(self._cfg.system.auto_update)
        self._updater.start()
        from .audio import AudioWatcher
        self._audio = AudioWatcher(self._cfg.bite.sound_sensitivity)
        if audio:
            self._audio.start()
            self._engine.audio = self._audio

    # ── internals ───────────────────────────────────────────────────────────

    @property
    def engine(self) -> Engine:
        return self._engine

    def attach_window(self, window) -> None:
        self._window = window

    def shutdown(self) -> None:
        self._engine.stop()
        self._engine.join(2)
        self._engine.inp.release()
        self._hotkeys.stop()
        self._audio.stop()
        self._updater.finish()

    LEARN_VERSION = 3  # bump when what auto-learn stores changes, so old guesses get relearned

    def _load(self) -> None:
        with self._lock:
            self._cfg = self._store.load(self._profile)
            self._templates = self._store.templates(self._profile)
            sysc = self._cfg.system
            if sysc.auto_learn and sysc.learn_version < self.LEARN_VERSION:
                # earlier versions could mistake other UI for the reel bar: learn it again
                self._store.delete_template(self._profile, "marker")
                self._templates = self._store.templates(self._profile)
                self._cfg = self._cfg.merged({"regions": {"reel": asdict(Region())},
                                              "reel": {"method": "bar", "hold_moves": "auto"},
                                              "system": {"learn_version": self.LEARN_VERSION}})
                self._store.save(self._profile, self._cfg)
            self._engine.configure(self._cfg, self._templates)
            self._bind_hotkeys()
            settings = self._store.settings()
            settings["profile"] = self._profile
            self._store.save_settings(settings)

    def _bind_hotkeys(self) -> None:
        s = self._cfg.system
        self._hotkeys.bind({s.hotkey_toggle: self._hotkey_toggle, s.hotkey_stop: self._engine.stop,
                            s.hotkey_mark: self._hotkey_mark})

    def _hotkey_mark(self) -> None:
        if self._setup["active"]:
            threading.Thread(target=self._mark, name="fishbot-setup", daemon=True).start()

    def _mark(self) -> None:
        """One quick-setup step, using the current cursor position."""
        if not self._mark_lock.acquire(blocking=False):
            return
        st = self._setup
        try:
            st["busy"], st["error"] = True, ""
            x, y = self._engine.inp.position()
            step = STEPS[st["step"]]
            if step == "bobber":
                tpl, region = find_bobber(self._screen, x, y)
                self._save_setup("bobber", tpl, {"regions": {"bobber": asdict(region)}, "bite": {"method": "template"}})
            elif step == "bar_left":
                st["left"] = (x, y)
            else:
                lx, ly = st["left"]
                region = self._green_bar_near(lx, x, (ly + y) // 2)
                if region is not None:  # Albion's green/red band: no template needed
                    with self._lock:
                        self._store.delete_template(self._profile, "marker")
                        self._templates = self._store.templates(self._profile)
                        self._commit(self._cfg.merged({"regions": {"reel": asdict(region)}, "reel": {"method": "bar"}}))
                else:
                    tpl, region = find_marker(self._screen, lx, x, (ly + y) // 2)
                    self._save_setup("marker", tpl, {"regions": {"reel": asdict(region)}, "reel": {"method": "template"}})
            st["step"] += 1
            if st["step"] >= len(STEPS):
                st["active"], st["done"] = False, True
        except SetupError as e:
            st["error"] = e.code
        except Exception as e:  # never kill the hotkey thread
            log.exception("quick setup step failed")
            st["error"] = f"internal: {e}"
        finally:
            st["busy"] = False
            st["rev"] += 1
            self._mark_lock.release()

    def _learned(self, name: str, tpl, patch: dict) -> None:
        """The engine figured out the bobber / reel bar by itself: keep it in the profile."""
        if tpl is None:  # a config change, or "forget this template"
            with self._lock:
                if name:
                    self._store.delete_template(self._profile, name)
                    self._templates = self._store.templates(self._profile)
                self._commit(self._cfg.merged(patch))
            return
        self._save_setup(name, tpl, patch)

    def _green_bar_near(self, x0: int, x1: int, y: int):
        from .learn import find_green_bar
        left, right = sorted((int(x0), int(x1)))
        pad = max(40, (right - left) // 4)
        area = Region(left - pad, int(y) - 120, right - left + 2 * pad, 240)
        band = find_green_bar(self._screen.grab(area))
        if band is None:
            return None
        bx, by, bw, bh = band
        top = max(0, by - bh)
        return Region(area.left + bx, area.top + top, bw, by + bh - top + 1)

    def _save_setup(self, name: str, tpl, patch: dict) -> None:
        with self._lock:
            self._store.save_template(self._profile, name, tpl)
            self._templates = self._store.templates(self._profile)
            self._commit(self._cfg.merged(patch))

    def _hotkey_toggle(self) -> None:
        try:
            self._engine.toggle()
        except VisionError as e:
            self._engine.report(e)

    def _commit(self, cfg: Config) -> None:
        with self._lock:
            self._rev = getattr(self, "_rev", 0) + 1
            self._cfg = cfg
            self._store.save(self._profile, cfg)
            self._engine.configure(cfg, self._templates)
            self._bind_hotkeys()
            if hasattr(self, "_audio"):
                self._audio.set_sensitivity(cfg.bite.sound_sensitivity)

    def _thumbs(self) -> dict:
        return {t: (to_data_url(self._templates[t], 240) if t in self._templates else None) for t in TEMPLATES}

    def _abs_rect(self, rect: dict) -> Region:
        if self._capture is None:
            raise ValueError("no capture — take a screenshot first")
        img, mon = self._capture
        h, w = img.shape[:2]
        x = int(np.clip(round(float(rect["x"])), 0, w - 1))
        y = int(np.clip(round(float(rect["y"])), 0, h - 1))
        rw = int(np.clip(round(float(rect["w"])), 1, w - x))
        rh = int(np.clip(round(float(rect["h"])), 1, h - y))
        return Region(mon["left"] + x, mon["top"] + y, rw, rh)

    # ── endpoints ───────────────────────────────────────────────────────────

    def _apply(self, patch: dict) -> dict:
        with self._lock:
            self._commit(self._cfg.merged(patch))
            return {"config": self._cfg.to_dict()}

    def _bootstrap(self) -> dict:
        return {
            "version": __version__,
            "demo": self._demo,
            "native": self._window is not None,
            "platform": sys.platform,
            "schema": schema(),
            "config": self._cfg.to_dict(),
            "profile": self._profile,
            "profiles": self._store.list(),
            "templates": self._thumbs(),
            "ui": self._store.settings().get("ui", {}),
            "hotkeys": self._hotkeys.ok,
            "sound": self._audio.ok,
            "monitors": len(self._screen.monitors()) - 1,
            "data_dir": str(self._store.root.resolve()),
        }

    def _switch(self, name: str) -> dict:
        if not self._store.exists(name):
            raise FileNotFoundError(name)
        self._halt()
        self._profile = name
        self._load()
        return self._bootstrap()

    def _halt(self) -> None:
        self._engine.stop()
        self._engine.join(2)

    @endpoint
    def bootstrap(self):
        return self._bootstrap()

    @endpoint
    def state(self, since: int = 0):
        return {**self._engine.snapshot(int(since)), "cfg_rev": getattr(self, "_rev", 0)}

    @endpoint
    def start(self):
        self._engine.start()

    @endpoint
    def pause(self):
        self._engine.pause()

    @endpoint
    def resume(self):
        self._engine.resume()

    @endpoint
    def toggle(self):
        self._engine.toggle()

    @endpoint
    def stop(self):
        self._engine.stop()

    @endpoint
    def setup_start(self):
        self._halt()
        self._setup.update(active=True, done=False, step=0, error="", left=None)
        self._setup["rev"] += 1
        return self._setup_view()

    @endpoint
    def setup_cancel(self):
        self._setup.update(active=False, error="")
        self._setup["rev"] += 1

    @endpoint
    def setup_state(self):
        return self._setup_view()

    @endpoint
    def setup_mark(self):
        """Run the current step at the cursor (same as pressing the hotkey)."""
        self._mark()
        return self._setup_view()

    def _setup_view(self) -> dict:
        st = {k: v for k, v in self._setup.items() if k != "left"}
        return {"setup": {**st, "key": self._cfg.system.hotkey_mark.upper(), "hotkeys": self._hotkeys.ok},
                "config": self._cfg.to_dict(), "templates": self._thumbs()}

    @endpoint
    def update_state(self):
        return {"update": dict(self._updater.state)}

    @endpoint
    def update_install(self):
        """Close now; the downloaded version replaces the exe and starts again."""
        if not self._updater.ready:
            raise ValueError("no update downloaded")
        self._updater.restart = True

        def close():
            time.sleep(0.3)
            win = self._window
            if win is not None:
                win.destroy()
            self.quit.set()
        threading.Thread(target=close, daemon=True).start()

    @endpoint
    def uninstall(self):
        """Remove data/ (and the exe when frozen), then close the app."""
        from .uninstall import run
        self._halt()
        self._updater.restart = False
        self._updater.state["status"] = "off"  # don't swap in a pending update afterwards
        self._hotkeys.stop()
        logging.shutdown()
        result = run(self._store)
        self._gone = True

        def close():
            time.sleep(0.4)
            if self._window is not None:
                self._window.destroy()
            self.quit.set()
        threading.Thread(target=close, daemon=True).start()
        return result

    @endpoint
    def forget(self):
        """Drop the learned bobber / reel bar so the bot learns them again (new spot, new UI scale)."""
        self._halt()
        with self._lock:
            for name in ("bobber", "marker"):
                self._store.delete_template(self._profile, name)
            self._templates = self._store.templates(self._profile)
            cfg = Config.from_dict(self._cfg.to_dict())
            cfg.regions.bobber = Region()
            cfg.regions.reel = Region()
            cfg.bite.threshold = Config().bite.threshold
            cfg.bite.method = cfg.reel.method = "template"
            self._commit(cfg)
            return {"config": cfg.to_dict(), "templates": self._thumbs()}

    @endpoint
    def catches(self):
        return {"items": self._engine.catches.summary()}

    @endpoint
    def reset_stats(self):
        self._engine.reset_stats()
        self._engine.catches.reset()

    @endpoint
    def check(self):
        self._engine.validate()

    @endpoint
    def update(self, patch: dict):
        if not isinstance(patch, dict):
            raise ValueError("patch must be an object")
        return self._apply(patch)

    @endpoint
    def reset_section(self, section: str):
        with self._lock:
            if section not in ("cast", "bite", "reel", "session", "system"):
                raise ValueError(section)
            cfg = Config.from_dict(self._cfg.to_dict())
            fresh = getattr(Config(), section)
            if section == "cast":
                fresh.points = cfg.cast.points  # keep calibrated points
            setattr(cfg, section, fresh)
            self._commit(cfg)
            return {"config": cfg.to_dict()}

    @endpoint
    def capture(self, delay: float = 0.0):
        win = self._window
        delay = float(min(max(delay, 0.0), 10.0))
        hidden = False
        if win is not None:
            try:
                win.hide()
                hidden = True
                delay = max(delay, 0.45)  # let the compositor drop our window
            except Exception:
                log.warning("could not hide the window for capture", exc_info=True)
        try:
            time.sleep(delay)
            img, mon = self._screen.grab_monitor(self._cfg.system.monitor)
        finally:
            if hidden:
                win.show()
        with self._lock:
            self._capture = (img, mon)
        h, w = img.shape[:2]
        _, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return {"image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
                "width": w, "height": h, "left": mon["left"], "top": mon["top"]}

    @endpoint
    def set_region(self, name: str, rect: dict):
        if name not in ("bobber", "reel", "water"):
            raise ValueError(name)
        r = self._abs_rect(rect)
        return self._apply({"regions": {name: asdict(r)}})

    @endpoint
    def save_template(self, name: str, rect: dict):
        r = self._abs_rect(rect)
        img, mon = self._capture
        x, y = r.left - mon["left"], r.top - mon["top"]
        crop = img[y:y + r.height, x:x + r.width].copy()
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            raise ValueError("selection too small")
        with self._lock:
            self._store.save_template(self._profile, name, crop)
            self._templates = self._store.templates(self._profile)
            patch = {}
            if name == "bobber" and not self._cfg.regions.bobber.ok:
                # sensible default search area around the float
                patch = {"regions": {"bobber": {"left": r.left - 140, "top": r.top - 90,
                                                "width": r.width + 280, "height": r.height + 180}}}
            self._commit(self._cfg.merged(patch))
            return {"templates": self._thumbs(), "config": self._cfg.to_dict()}

    @endpoint
    def clear_template(self, name: str):
        with self._lock:
            self._store.delete_template(self._profile, name)
            self._templates = self._store.templates(self._profile)
            self._engine.configure(self._cfg, self._templates)
            return {"templates": self._thumbs()}

    @endpoint
    def add_point(self, x: float, y: float):
        r = self._abs_rect({"x": x, "y": y, "w": 1, "h": 1})
        points = [*self._cfg.cast.points, [r.left, r.top]][-16:]
        return self._apply({"cast": {"points": points}})

    @endpoint
    def preview(self):
        shot = self._engine.take_preview()
        if shot is None:
            return {"image": None}
        img, kind, at = shot
        return {"image": to_data_url(img, 760), "kind": kind, "age": round(time.monotonic() - at, 2)}

    @endpoint
    def probe(self, kind: str):
        if kind not in ("bobber", "reel", "spot"):
            raise ValueError(kind)
        out = self._engine.probe(kind)
        out["image"] = to_data_url(out["image"], 760)
        return out

    @endpoint
    def profile_switch(self, name: str):
        return self._switch(name)

    @endpoint
    def profile_create(self, name: str, copy: bool = True):
        return self._switch(self._store.create(name, self._profile if copy else None))

    @endpoint
    def profile_rename(self, name: str):
        self._halt()
        self._profile = self._store.rename(self._profile, name)
        self._load()
        return self._bootstrap()

    @endpoint
    def profile_delete(self):
        self._halt()
        self._store.delete(self._profile)
        self._profile = self._store.list()[0]
        self._load()
        return self._bootstrap()

    @endpoint
    def save_ui(self, prefs: dict):
        if not isinstance(prefs, dict):
            raise ValueError("prefs must be an object")
        settings = self._store.settings()
        settings["ui"] = prefs
        self._store.save_settings(settings)

    @endpoint
    def open_debug(self):
        d = self._store.root / "debug"
        d.mkdir(parents=True, exist_ok=True)
        open_path(str(d.resolve()))

    @endpoint
    def open_folder(self):
        open_path(str(self._store.root.resolve()))
