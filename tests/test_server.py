import json
import urllib.request
from urllib.error import HTTPError

import pytest

from fishbot.api import Api
from fishbot.config import ProfileStore
from fishbot.server import serve
from fishbot.sim import SimGame, SimInput, SimScreen, prepare_profile


@pytest.fixture
def server(tmp_path):
    store = ProfileStore(tmp_path)
    game = SimGame(seed=1)
    api = Api(store, SimScreen(game), SimInput(game), demo=True, focus=None,
              profile=prepare_profile(store), hotkeys=False)
    httpd, url = serve(api)
    base, token = url.split("/?token=")
    yield api, base, token
    httpd.shutdown()
    api.shutdown()


def post(base, method, *args, token="", host=None):
    req = urllib.request.Request(f"{base}/api/{method}", data=json.dumps({"args": list(args)}).encode(),
                                 headers={"X-Token": token, "Content-Type": "application/json"})
    if host:
        req.add_header("Host", host)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_requires_token(server):
    _, base, token = server
    with pytest.raises(HTTPError) as e:
        post(base, "bootstrap", token="wrong")
    assert e.value.code == 403
    assert post(base, "bootstrap", token=token)["ok"]


def test_rejects_foreign_host(server):
    _, base, token = server
    with pytest.raises(HTTPError) as e:
        post(base, "bootstrap", token=token, host="evil.example")
    assert e.value.code == 403


def test_unknown_method_and_private_attrs(server):
    _, base, token = server
    for name in ("shutdown", "_commit", "engine"):
        with pytest.raises(HTTPError) as e:
            post(base, name, token=token)
        assert e.value.code == 404


def test_static_files_and_traversal(server):
    _, base, _ = server
    with urllib.request.urlopen(f"{base}/", timeout=5) as r:
        assert b"Albion Fishing Bot" in r.read()
    with urllib.request.urlopen(f"{base}/fonts/manrope-latin.woff2", timeout=5) as r:
        assert r.headers["Content-Type"] == "font/woff2"
    with pytest.raises(HTTPError) as e:
        urllib.request.urlopen(f"{base}/../config.py", timeout=5)
    assert e.value.code == 404


def test_config_update_is_validated_and_persisted(server):
    api, base, token = server
    r = post(base, "update", {"reel": {"target": 5, "control": "hysteresis"}}, token=token)
    assert r["ok"] and r["config"]["reel"]["target"] == 0.9 and r["config"]["reel"]["control"] == "hysteresis"
    assert api._store.load(api._profile).reel.control == "hysteresis"


def test_calibration_flow(server):
    _, base, token = server
    shot = post(base, "capture", token=token)
    assert shot["ok"] and shot["image"].startswith("data:image/png")
    r = post(base, "set_region", "reel", {"x": 600, "y": 640, "w": 400, "h": 28}, token=token)
    assert r["config"]["regions"]["reel"] == {"left": 600, "top": 640, "width": 400, "height": 28}
    r = post(base, "save_template", "spot", {"x": 10, "y": 10, "w": 30, "h": 30}, token=token)
    assert r["ok"] and r["templates"]["spot"].startswith("data:image/jpeg")
    r = post(base, "add_point", 500, 300, token=token)
    assert [500, 300] in r["config"]["cast"]["points"]
    r = post(base, "probe", "reel", token=token)
    assert r["ok"] and "score" in r


def test_errors_are_reported_not_raised(server):
    _, base, token = server
    assert post(base, "update", {"system": {"auto_learn": False}, "reel": {"method": "template"}}, token=token)["ok"]
    assert post(base, "clear_template", "marker", token=token)["ok"]
    r = post(base, "start", token=token)
    assert not r["ok"] and r["error"] == "template_missing" and r["params"]["name"] == "marker"
    r = post(base, "set_region", "nope", {}, token=token)
    assert not r["ok"] and r["error"] == "invalid"


def test_profiles(server):
    _, base, token = server
    r = post(base, "profile_create", "Lake", True, token=token)
    assert r["ok"] and r["profile"] == "Lake" and "Demo" in r["profiles"]
    r = post(base, "profile_rename", "River", token=token)
    assert r["profile"] == "River"
    r = post(base, "profile_delete", token=token)
    assert r["ok"] and "River" not in r["profiles"]


def test_uninstall_removes_data_and_blocks_writes(server):
    api, base, token = server
    root = api._store.root
    assert root.exists()
    r = post(base, "uninstall", token=token)
    assert r["ok"] and not r["scheduled"]
    assert not root.exists()
    assert not post(base, "save_ui", {"lang": "ru"}, token=token)["ok"]
    assert not root.exists()


def test_old_profile_takes_the_new_settings(tmp_path):
    """A profile from 1.7.x: what it learned is dropped and the bite settings tuned in the game apply."""
    from fishbot.config import Config
    store = ProfileStore(tmp_path)
    old = Config()
    old.system.learn_version = 3
    old.bite.confirm_ms, old.bite.bite_timeout_s, old.reel.hold_moves = 200, 35, "auto"
    old.bite.sound_prints = [[0.1] * 48]
    old.regions.bobber.width = old.regions.bobber.height = 100
    store.save("Default", old)
    game = SimGame(seed=1)
    api = Api(store, SimScreen(game), SimInput(game), focus=None, profile="Default", hotkeys=False)
    try:
        cfg = store.load("Default")
        fresh = Config()
        assert cfg.system.learn_version == Api.LEARN_VERSION and cfg.system.relearn_on_start
        assert cfg.bite.confirm_ms == fresh.bite.confirm_ms and cfg.bite.bite_timeout_s == fresh.bite.bite_timeout_s
        assert cfg.reel.hold_moves == "right" and not cfg.bite.sound_prints and not cfg.regions.bobber.ok
    finally:
        api.shutdown()
