"""Typed, self-validating configuration plus on-disk profile storage.

Every tunable lives in one dataclass field whose metadata carries its range,
step, unit or allowed options. The same metadata drives JSON validation and
the UI controls (see ``schema()``), so limits are defined exactly once.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def num(default, lo, hi, step=None, unit: str = ""):
    return field(default=default, metadata={"min": lo, "max": hi, "step": step, "unit": unit})


def choice(default: str, *options: str):
    return field(default=default, metadata={"options": options})


def hsv(*value: int):
    return field(default_factory=lambda: list(value), metadata={"kind": "hsv"})


# ── sections ────────────────────────────────────────────────────────────────

@dataclass
class Region:
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def ok(self) -> bool:
        return self.width >= 4 and self.height >= 4

    def as_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height


@dataclass
class CastCfg:
    target: str = choice("cursor", "cursor", "points", "auto")
    points: list = field(default_factory=list, metadata={"kind": "points"})
    rotate_after: int = num(2, 1, 20)
    power_ms: int = num(450, 100, 4000, 10, "ms")
    power_jitter_ms: int = num(120, 0, 1000, 10, "ms")
    aim_jitter_px: int = num(6, 0, 60, 1, "px")
    spot_threshold: float = num(0.6, 0.3, 0.99, 0.01)


@dataclass
class BiteCfg:
    method: str = choice("template", "template", "color")
    threshold: float = num(0.7, 0.3, 0.99, 0.01)
    settle_ms: int = num(1200, 0, 5000, 50, "ms")
    dip_px: int = num(6, 1, 60, 1, "px")
    confirm_frames: int = num(2, 1, 10)
    appear_timeout_s: float = num(6.0, 1, 30, 0.5, "s")
    bite_timeout_s: float = num(35.0, 5, 180, 1, "s")
    hook_delay_ms: int = num(140, 0, 1500, 10, "ms")
    hook_jitter_ms: int = num(80, 0, 1000, 10, "ms")
    use_sound: bool = True
    sound_sensitivity: float = num(4.0, 1.5, 20.0, 0.5)
    grayscale: bool = True
    hsv_lo: list = hsv(0, 120, 120)
    hsv_hi: list = hsv(10, 255, 255)
    min_area: int = num(12, 1, 5000, 1, "px")


@dataclass
class ReelCfg:
    method: str = choice("template", "template", "color")
    threshold: float = num(0.65, 0.3, 0.99, 0.01)
    control: str = choice("predictive", "predictive", "hysteresis")
    target: float = num(0.5, 0.1, 0.9, 0.01)
    deadband: float = num(0.06, 0.0, 0.4, 0.01)
    lookahead_ms: int = num(90, 0, 400, 5, "ms")
    edge_guard: float = num(0.12, 0.0, 0.45, 0.01)
    min_toggle_ms: int = num(20, 0, 200, 5, "ms")
    hold_moves: str = choice("auto", "auto", "right", "left")
    fps: int = num(90, 15, 240, 5, "fps")
    appear_timeout_s: float = num(3.0, 0.5, 15, 0.5, "s")
    end_confirm_ms: int = num(350, 50, 3000, 50, "ms")
    max_duration_s: float = num(40.0, 5, 180, 1, "s")
    grayscale: bool = True
    hsv_lo: list = hsv(20, 80, 180)
    hsv_hi: list = hsv(40, 255, 255)
    min_area: int = num(10, 1, 5000, 1, "px")


@dataclass
class Action:
    key: str = "1"
    every_min: float = num(30.0, 0.5, 600, 0.5, "min")
    enabled: bool = True
    label: str = ""


@dataclass
class SessionCfg:
    cooldown_ms: int = num(1400, 0, 15000, 50, "ms")
    cooldown_jitter_ms: int = num(600, 0, 10000, 50, "ms")
    max_catches: int = num(0, 0, 100000)
    max_minutes: int = num(0, 0, 1440, 5, "min")
    break_every_min: int = num(0, 0, 600, 5, "min")
    break_minutes: float = num(3.0, 0.5, 120, 0.5, "min")
    max_fail_streak: int = num(12, 2, 200)
    actions: list = field(default_factory=list, metadata={"item": Action, "limit": 8})


@dataclass
class SystemCfg:
    monitor: int = num(1, 1, 8)
    window_title: str = "Albion Online"
    require_focus: bool = True
    failsafe: bool = True
    sound: bool = True
    auto_update: bool = True
    auto_learn: bool = True
    start_delay_s: float = num(3.0, 0, 30, 0.5, "s")
    idle_fps: int = num(30, 5, 120, 5, "fps")
    hotkey_toggle: str = "f8"
    hotkey_stop: str = "f9"
    hotkey_mark: str = "f7"


@dataclass
class Regions:
    bobber: Region = field(default_factory=Region)
    reel: Region = field(default_factory=Region)
    water: Region = field(default_factory=Region)


@dataclass
class Config:
    cast: CastCfg = field(default_factory=CastCfg)
    bite: BiteCfg = field(default_factory=BiteCfg)
    reel: ReelCfg = field(default_factory=ReelCfg)
    session: SessionCfg = field(default_factory=SessionCfg)
    system: SystemCfg = field(default_factory=SystemCfg)
    regions: Regions = field(default_factory=Regions)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "Config":
        return _build(cls, data)

    def merged(self, patch: dict) -> "Config":
        return _build(Config, _deep_merge(self.to_dict(), patch or {}))


# ── validation ──────────────────────────────────────────────────────────────

def _build(cls, data: Any):
    base = cls()
    if not isinstance(data, dict):
        return base
    for f in fields(cls):
        if f.name not in data:
            continue
        current = getattr(base, f.name)
        try:
            if is_dataclass(current):
                value = _build(type(current), data[f.name])
            else:
                value = _coerce(data[f.name], current, f.metadata)
        except (TypeError, ValueError, OverflowError):
            continue  # keep the default for malformed values
        setattr(base, f.name, value)
    return base


def _coerce(value: Any, default: Any, meta) -> Any:
    kind = meta.get("kind")
    if kind == "hsv":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError
        h, s, v = (int(round(float(c))) for c in value)
        return [min(max(h, 0), 179), min(max(s, 0), 255), min(max(v, 0), 255)]
    if kind == "points":
        if not isinstance(value, list):
            raise ValueError
        pts = []
        for p in value[:16]:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
        return pts
    if "item" in meta:
        if not isinstance(value, list):
            raise ValueError
        return [_build(meta["item"], v) for v in value[: meta.get("limit", 16)] if isinstance(v, dict)]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, (bool, int, float)):
            return bool(value)
        raise ValueError
    if isinstance(default, str):
        out = str(value).strip()[:120]
        options = meta.get("options")
        if options and out not in options:
            raise ValueError
        return out
    if isinstance(default, (int, float)):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError
        lo, hi = meta.get("min"), meta.get("max")
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return int(round(x)) if isinstance(default, int) else round(x, 4)
    return value


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def schema() -> dict:
    """Field ranges/options for the UI, derived from the dataclass metadata."""
    def describe(cls):
        out = {}
        base = cls()
        for f in fields(cls):
            value = getattr(base, f.name)
            if is_dataclass(value):
                out[f.name] = describe(type(value))
                continue
            meta = {k: v for k, v in f.metadata.items() if k != "item" and v not in (None, "")}
            if "options" in meta:
                meta["options"] = list(meta["options"])
            if "item" in f.metadata:
                meta["item"] = describe(f.metadata["item"])
            meta["type"] = type(value).__name__
            meta["default"] = value
            out[f.name] = meta
        return out
    return describe(Config)


# ── storage ─────────────────────────────────────────────────────────────────

TEMPLATES = ("bobber", "marker", "spot")
_NAME_RE = re.compile(r"[^\w\- ]+", re.UNICODE)


def clean_name(name: str) -> str:
    out = " ".join(_NAME_RE.sub("", str(name or "")).split()).strip(" .-_")[:40]
    return out or "Profile"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(5):  # Windows: AV scanners may briefly lock the target
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread cannot open non-ASCII paths on Windows; decode from bytes instead."""
    import cv2
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def write_image(path: Path, img: np.ndarray) -> None:
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    write_atomic(path, buf.tobytes())


class ProfileStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "profiles"
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.list():
            self.save("Default", Config())

    # settings.json keeps the active profile and UI preferences
    def settings(self) -> dict:
        try:
            data = json.loads((self.root / "settings.json").read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_settings(self, data: dict) -> None:
        write_atomic(self.root / "settings.json", json.dumps(data, ensure_ascii=False, indent=2).encode())

    def list(self) -> list[str]:
        return sorted((p.name for p in self.dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=str.lower)

    def path(self, name: str) -> Path:
        return self.dir / clean_name(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / "config.json").is_file()

    def load(self, name: str) -> Config:
        try:
            data = json.loads((self.path(name) / "config.json").read_text("utf-8"))
        except (OSError, ValueError):
            data = {}
        return Config.from_dict(data)

    def save(self, name: str, cfg: Config) -> None:
        blob = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2).encode()
        write_atomic(self.path(name) / "config.json", blob)

    def unique(self, name: str) -> str:
        base = clean_name(name)
        out, i = base, 2
        taken = {n.lower() for n in self.list()}
        while out.lower() in taken:
            out = f"{base} {i}"
            i += 1
        return out

    def create(self, name: str, copy_from: str | None = None) -> str:
        name = self.unique(name)
        if copy_from and self.exists(copy_from):
            shutil.copytree(self.path(copy_from), self.path(name))
        else:
            self.save(name, Config())
        return name

    def rename(self, old: str, new: str) -> str:
        new = clean_name(new)
        if new.lower() != clean_name(old).lower():
            new = self.unique(new)
        os.replace(self.path(old), self.path(new))
        return new

    def delete(self, name: str) -> None:
        if len(self.list()) <= 1:
            raise ValueError("cannot delete the last profile")
        shutil.rmtree(self.path(name))

    def template_path(self, name: str, tpl: str) -> Path:
        if tpl not in TEMPLATES:
            raise ValueError(f"unknown template {tpl!r}")
        return self.path(name) / f"{tpl}.png"

    def templates(self, name: str) -> dict[str, np.ndarray]:
        out = {}
        for tpl in TEMPLATES:
            p = self.template_path(name, tpl)
            if p.is_file():
                img = read_image(p)
                if img is not None and img.size:
                    out[tpl] = img
        return out
