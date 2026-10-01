"""Quick setup on the simulator: hover + hotkey instead of screenshots."""
import math

from fishbot.api import Api
from fishbot.config import ProfileStore
from fishbot.sim import BAR, SimGame, SimInput, SimScreen


class WobbleGame(SimGame):
    """Keeps the minigame alive with the marker swinging, like a fish fighting."""

    def _step(self, dt):
        if self.state == "reel":
            self.x = 0.5 + 0.32 * math.sin(self.t * 7)
            return
        super()._step(dt)


def test_quick_setup_calibrates_from_cursor(tmp_path):
    game = WobbleGame(seed=2)
    game.spots = []
    api = Api(ProfileStore(tmp_path), SimScreen(game), SimInput(game), focus=None, hotkeys=False)
    try:
        api.update({"system": {"auto_learn": False}})
        assert api.check()["error"] == "region_missing"
        assert api.setup_start()["setup"]["step"] == 0

        game.bobber = (700.0, 330.0)
        game._set("floating")
        game.cursor = (704, 334)                       # roughly on the bobber
        r = api.setup_mark()
        assert r["setup"]["error"] == "" and r["setup"]["step"] == 1
        assert r["templates"]["bobber"]
        bob = r["config"]["regions"]["bobber"]
        assert bob["left"] < 700 < bob["left"] + bob["width"] and bob["top"] < 330 < bob["top"] + bob["height"]
        assert api.probe("bobber")["found"]

        game._start_reel()
        game.cursor = (BAR.left + 1, BAR.top + 14)     # left end of the bar
        assert api.setup_mark()["setup"]["step"] == 2
        game.cursor = (BAR.left + BAR.width - 2, BAR.top + 12)
        r = api.setup_mark()
        assert r["setup"]["error"] == "", r["setup"]["error"]
        assert r["setup"]["done"] and not r["setup"]["active"]
        reel = r["config"]["regions"]["reel"]
        assert r["config"]["reel"]["method"] == "bar"
        assert abs(reel["left"] - BAR.left) <= 3 and abs(reel["width"] - BAR.width) <= 5
        assert reel["top"] < BAR.top and reel["top"] + reel["height"] >= BAR.top + BAR.height - 2
        assert api.probe("reel")["found"]
        assert api.check()["ok"]
    finally:
        api.shutdown()


def test_quick_setup_reports_still_marker(tmp_path):
    game = SimGame(seed=3)
    api = Api(ProfileStore(tmp_path), SimScreen(game), SimInput(game), focus=None, hotkeys=False)
    try:
        api.setup_start()
        api._setup["step"] = 1
        game.cursor = (300, 700)
        api.setup_mark()
        game.cursor = (900, 700)                       # no minigame on screen: nothing moves
        r = api.setup_mark()
        assert r["setup"]["error"] == "marker_still" and r["setup"]["step"] == 2
    finally:
        api.shutdown()
