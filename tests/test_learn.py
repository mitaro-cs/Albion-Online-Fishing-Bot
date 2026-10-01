"""Zero setup: a fresh profile with nothing calibrated learns everything while fishing."""
import random

from conftest import FakeClock
from fishbot.api import Api
from fishbot.config import Config, ProfileStore
from fishbot.engine import Engine
from fishbot.sim import BAR, SimGame, SimInput, SimScreen


def fresh_rig(seed=4):
    clock = FakeClock()
    game = SimGame(clock, seed=seed)
    game.spots = [{"x": 800.0, "y": 300.0, "fish": 10 ** 6, "respawn": 0.0}]
    game.cursor = (800, 300)                      # player points at the water, nothing else
    events = []
    eng = Engine(SimScreen(game), SimInput(game), clock=clock, focus=None, rng=random.Random(seed),
                 on_event=events.append)
    cfg = Config()                                 # defaults: no regions, no templates
    cfg.system.sound = False
    cfg.system.start_delay_s = 0.5
    cfg.session.max_catches = 4
    eng.configure(cfg, {})
    return eng, game, events


def test_learns_bobber_and_bar_then_fishes():
    eng, game, events = fresh_rig()
    eng.validate()                                 # nothing calibrated, still allowed
    eng.start()
    eng.join(120)
    codes = [e["code"] for e in events]
    assert "learn_bobber" in codes and "learn_bar" in codes, codes
    assert codes.count("learn_bobber") == 1 and codes.count("learn_bar") == 1   # learned once, then reused
    assert eng.stats.catches == 4, codes
    reel = eng._cfg.regions.reel
    assert abs(reel.left - BAR.left) <= 6 and abs(reel.width - BAR.width) <= 12
    assert eng._cfg.regions.bobber.ok and "bobber" in eng._templates and "marker" in eng._templates


def test_learned_data_is_saved_to_profile(tmp_path):
    clock = FakeClock()
    game = SimGame(clock, seed=6)
    game.spots = [{"x": 800.0, "y": 300.0, "fish": 10 ** 6, "respawn": 0.0}]
    game.cursor = (800, 300)
    store = ProfileStore(tmp_path)
    api = Api(store, SimScreen(game), SimInput(game), focus=None, hotkeys=False)
    try:
        eng = api.engine
        eng.clock = clock
        eng.rng = random.Random(6)
        api.update({"system": {"sound": False, "start_delay_s": 0.5}, "session": {"max_catches": 2}})
        assert api.start()["ok"]
        eng.join(120)
        assert eng.stats.catches == 2
        saved = store.load(api._profile)
        assert saved.regions.bobber.ok and saved.regions.reel.ok
        assert set(store.templates(api._profile)) >= {"bobber", "marker"}
    finally:
        api.shutdown()
