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
    assert eng._cfg.regions.bobber.ok and "bobber" in eng._templates
    assert eng._cfg.reel.method == "bar"            # the green/red band + bobber, no template needed


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
        assert "bobber" in store.templates(api._profile) and saved.reel.method == "bar"
    finally:
        api.shutdown()


def test_finds_out_which_way_holding_pushes():
    for invert, expected in ((True, "left"), (False, "right")):
        eng, game, events = fresh_rig(seed=5)
        game.invert = invert
        eng.start()
        eng.join(120)
        assert eng.stats.catches == 4, (invert, [e["code"] for e in events])
        assert eng._cfg.reel.hold_moves == expected
        assert [e["code"] for e in events].count("learn_hold") == 1


def test_ignores_static_panel_and_heals_a_wrong_bar():
    eng, game, events = fresh_rig(seed=7)
    game.decoy = True
    eng.start()
    eng.join(150)
    assert eng.stats.catches == 4, [e["code"] for e in events]
    reel = eng._cfg.regions.reel
    assert abs(reel.left - BAR.left) <= 6, reel          # the real bar, not the "A" panel

    # a profile that already learned the panel (like v1.2.0 could): the bot notices and relearns
    from fishbot.config import Region
    eng2, game2, events2 = fresh_rig(seed=8)
    game2.decoy = True
    panel = Region(500, 433, 252, 49)
    a_icon = game2.render(Region(500, 433, 1, 1))  # warm up renderer
    game2._start_reel()
    a_icon = game2.render(Region(505, 440, 30, 38))
    game2._set("idle")
    cfg = eng2._cfg.merged({"regions": {"reel": {"left": 505, "top": 440, "width": panel.width - 10, "height": 38}},
                            "reel": {"method": "template"}})
    eng2.configure(cfg, {"marker": a_icon})
    eng2.start()
    eng2.join(200)
    codes = [e["code"] for e in events2]
    assert "relearn_bar" in codes and eng2.stats.catches == 4, codes


def test_tallies_the_loot_banner_by_item():
    eng, game, events = fresh_rig(seed=9)
    eng._cfg.session.max_catches = 6
    eng.start()
    eng.join(200)
    import time
    time.sleep(0.5)  # tallies are added off the fishing thread
    items = eng.catches.items
    assert sum(i["count"] for i in items) == eng.stats.catches == 6
    assert len(items) <= len(game.fish_names)
    assert [e["code"] for e in events].count("loot") == 6
