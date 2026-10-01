import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishbot.config import Region  # noqa: E402
from fishbot.engine import Engine  # noqa: E402
from fishbot.sim import SimGame, SimInput, SimScreen, demo_config, make_templates  # noqa: E402


class FakeClock:
    """Virtual time: sleeping advances it instantly, so minutes of fishing run in seconds."""

    def __init__(self):
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt if dt > 0 else 0.0005  # every frame costs a little time


class Rig:
    def __init__(self, seed: int = 1, focus=None):
        self.clock = FakeClock()
        self.game = SimGame(self.clock, seed=seed)
        # one rich fishing spot under a single cast point keeps the search area small (fast tests)
        self.game.spots = [{"x": 800.0, "y": 300.0, "fish": 10 ** 6, "respawn": 0.0}]
        self.inp = SimInput(self.game)
        self.events: list[dict] = []
        self.engine = Engine(SimScreen(self.game), self.inp, clock=self.clock, focus=focus,
                             rng=random.Random(seed), on_event=self.events.append)
        cfg = demo_config()
        cfg.cast.target = "points"
        cfg.cast.points = [[800, 300]]
        cfg.cast.aim_jitter_px = 4
        cfg.regions.bobber = Region(650, 200, 300, 200)
        cfg.system.start_delay_s = 0.5
        self.cfg = cfg
        self.templates = make_templates()
        self.engine.configure(cfg, self.templates)

    def codes(self) -> list[str]:
        return [e["code"] for e in self.events]

    def wait(self, pred, timeout: float = 60.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached in time")


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.engine.stop()
    r.engine.join(5)
