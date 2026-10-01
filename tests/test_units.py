import cv2
import numpy as np
import pytest

from fishbot.config import Config, ProfileStore, ReelCfg, read_image, schema, write_image
from fishbot.controller import ReelController
from fishbot.sim import draw_bobber, make_templates
from fishbot.vision import ColorFinder, TemplateFinder, VisionError


# ── config ──────────────────────────────────────────────────────────────────

def test_config_roundtrip_and_clamping():
    cfg = Config.from_dict({
        "cast": {"power_ms": 99999, "target": "nonsense", "points": [[1, 2], "bad", [3.6, 4.4]]},
        "bite": {"threshold": "0.8", "hsv_lo": [500, -3, 20.4]},
        "reel": {"fps": "fast", "grayscale": "false"},
        "unknown": 1,
    })
    assert cfg.cast.power_ms == 4000
    assert cfg.cast.target == "cursor"          # invalid option keeps the default
    assert cfg.cast.points == [[1, 2], [4, 4]]
    assert cfg.bite.threshold == 0.8
    assert cfg.bite.hsv_lo == [179, 0, 20]
    assert cfg.reel.fps == ReelCfg().fps        # unparsable number keeps the default
    assert cfg.reel.grayscale is False
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_merge_is_partial():
    cfg = Config().merged({"reel": {"target": 0.4}, "regions": {"reel": {"left": 5, "width": 100, "height": 20}}})
    assert cfg.reel.target == 0.4 and cfg.reel.deadband == ReelCfg().deadband
    assert cfg.regions.reel.ok and cfg.regions.reel.left == 5


def test_schema_exposes_ranges():
    s = schema()
    assert s["cast"]["power_ms"]["min"] == 100 and s["cast"]["power_ms"]["unit"] == "ms"
    assert s["reel"]["control"]["options"] == ["predictive", "hysteresis"]
    assert s["session"]["actions"]["item"]["every_min"]["max"] == 600


def test_profile_store(tmp_path):
    store = ProfileStore(tmp_path / "данные")       # non-ASCII path on purpose
    assert store.list() == ["Default"]
    name = store.create("Мост / Bridge!!")
    assert name == "Мост Bridge"
    assert store.create(name) != name             # unique names
    renamed = store.rename(name, "Lake")
    assert "Lake" in store.list()
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    write_image(store.template_path(renamed, "bobber"), img)
    assert np.array_equal(read_image(store.template_path(renamed, "bobber")), img)
    assert "bobber" in store.templates(renamed)
    store.delete(renamed)
    assert "Lake" not in store.list()
    with pytest.raises(ValueError):
        for n in store.list():
            store.delete(n)


# ── controller ──────────────────────────────────────────────────────────────

def ctrl(**kw):
    cfg = ReelCfg(**{"min_toggle_ms": 0, **kw})
    return ReelController(cfg)


def test_controller_pushes_towards_target():
    c = ctrl(control="hysteresis")
    assert c.update(0.3, 0.0) is True     # left of target → hold (moves right)
    assert c.update(0.7, 0.1) is False    # right of target → release
    assert c.update(0.5, 0.2) is False    # inside deadband → keep state


def test_controller_predicts_overshoot():
    c = ctrl(control="predictive", lookahead_ms=200)
    c.update(0.30, 0.00)
    # moving right fast: still left of target now, but predicted past it → release early
    assert c.update(0.42, 0.05) is False


def test_controller_edge_guard_and_inversion():
    c = ctrl(hold_moves="left", edge_guard=0.15)
    assert c.update(0.05, 0.0) is False   # near left edge: must push right → release (hold moves left)
    assert c.update(0.95, 0.1) is True


def test_controller_min_toggle_interval():
    c = ReelController(ReelCfg(control="hysteresis", min_toggle_ms=100, edge_guard=0.0))
    assert c.update(0.3, 0.000) is True
    assert c.update(0.7, 0.020) is True    # too soon to flip
    assert c.update(0.7, 0.150) is False


# ── vision ──────────────────────────────────────────────────────────────────

def scene(points):
    img = np.full((240, 320, 3), (110, 82, 33), np.uint8)
    img += np.random.default_rng(1).integers(0, 6, img.shape, dtype=np.uint8)
    for x, y in points:
        draw_bobber(img, x, y)
    return img


def test_template_finder_locates_sprite():
    finder = TemplateFinder(make_templates()["bobber"])
    m = finder.find(scene([(200, 120)]))
    assert m.score > 0.9
    assert abs(m.x - 200) <= 2 and abs(m.y - 118) <= 3


def test_find_all_suppresses_neighbours():
    finder = TemplateFinder(make_templates()["bobber"])
    found = finder.find_all(scene([(60, 60), (250, 170)]), 0.8)
    assert len(found) == 2
    assert {round(m.x / 50) for m in found} == {1, 5}


def test_flat_template_rejected():
    with pytest.raises(VisionError):
        TemplateFinder(np.zeros((10, 10, 3), np.uint8))


def test_color_finder_wraps_hue():
    img = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(img, (30, 30), 8, (20, 20, 230), -1)     # red: hue ≈ 0/179
    finder = ColorFinder([170, 120, 120], [10, 255, 255], 20)
    m = finder.find(img)
    assert m is not None and abs(m.x - 30) < 1.5 and m.area > 150
    assert ColorFinder([40, 120, 120], [80, 255, 255], 20).find(img) is None
