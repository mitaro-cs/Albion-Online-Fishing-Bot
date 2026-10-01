"""Reel controller: decides whether the mouse button should be held.

The marker position ``x`` is normalised to 0..1 across the reel bar. Holding
the button pushes the marker one way (``hold_moves``); releasing lets it drift
back. A plain hysteresis controller oscillates because of capture + input
latency, so the predictive mode extrapolates the marker ``lookahead_ms`` into
the future using a smoothed velocity estimate and acts on that instead.
"""
from __future__ import annotations

from .config import ReelCfg


class ReelController:
    def __init__(self, cfg: ReelCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.x: float | None = None
        self.t: float | None = None
        self.v = 0.0
        self.hold = False
        self.toggled_at = -1e9

    def predicted(self) -> float:
        if self.x is None:
            return self.cfg.target
        if self.cfg.control != "predictive":
            return self.x
        return self.x + self.v * self.cfg.lookahead_ms / 1000.0

    def update(self, x: float, t: float) -> bool:
        cfg = self.cfg
        if self.x is not None and self.t is not None and t > self.t:
            raw = (x - self.x) / (t - self.t)
            self.v = 0.55 * raw + 0.45 * self.v  # EMA keeps single-frame jitter out
        self.x, self.t = x, t

        p = self.predicted()
        half = cfg.deadband / 2
        if x < cfg.edge_guard:
            push_right, forced = True, True
        elif x > 1.0 - cfg.edge_guard:
            push_right, forced = False, True
        elif p < cfg.target - half:
            push_right, forced = True, False
        elif p > cfg.target + half:
            push_right, forced = False, False
        else:
            return self.hold  # inside the deadband: keep the current state

        want = push_right if cfg.hold_moves != "left" else not push_right
        if want != self.hold and (forced or (t - self.toggled_at) * 1000.0 >= cfg.min_toggle_ms):
            self.hold = want
            self.toggled_at = t
        return self.hold
