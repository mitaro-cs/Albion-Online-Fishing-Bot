"""End-to-end: the real engine plays the simulator through the Screen/Input interfaces."""
import pytest

from conftest import Rig
from fishbot.config import Action, Region
from fishbot.vision import VisionError


def run_until_stopped(rig: Rig, timeout: float = 90) -> None:
    rig.engine.start()
    rig.engine.join(timeout)
    assert not rig.engine.running, "engine did not stop"


def test_catches_fish_reliably(rig):
    rig.cfg.session.max_catches = 6
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig)
    s = rig.engine.stats
    assert s.catches == 6
    assert rig.game.caught == 6
    assert s.escaped == 0 and rig.game.escaped == 0
    assert s.casts <= 7  # at most one stray miss
    assert "limit_catches" in rig.codes()
    assert not rig.inp.held


def test_hysteresis_mode_also_lands_fish():
    rig = Rig(seed=5)
    rig.cfg.reel.control = "hysteresis"
    rig.cfg.session.max_catches = 3
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig)
    assert rig.engine.stats.catches == 3


def test_validate_reports_missing_calibration(rig):
    rig.engine.configure(rig.cfg, {})
    with pytest.raises(VisionError) as e:
        rig.engine.start()
    assert e.value.code == "template_missing"
    rig.cfg.regions.reel = Region()
    rig.engine.configure(rig.cfg, rig.templates)
    with pytest.raises(VisionError) as e:
        rig.engine.validate()
    assert e.value.code == "region_missing"


def test_stop_releases_mouse_mid_reel(rig):
    rig.engine.start()
    rig.wait(lambda: rig.engine.stage == "reel" and rig.inp.held)
    rig.engine.stop()
    rig.engine.join(5)
    assert not rig.engine.running
    assert not rig.inp.held
    assert not rig.game.held


def test_pause_and_resume(rig):
    rig.engine.start()
    rig.wait(lambda: rig.engine.stats.catches >= 1)
    rig.engine.pause()
    rig.wait(lambda: rig.engine.status == "paused")
    assert not rig.inp.held
    before = rig.engine.stats.casts
    rig.engine.resume()
    rig.wait(lambda: rig.engine.stats.casts > before and rig.engine.status == "running")
    rig.wait(lambda: rig.engine.stats.catches >= 2)


def test_auto_pause_on_focus_loss():
    title = {"value": "Albion Online Client"}
    rig = Rig(focus=lambda: title["value"])
    rig.cfg.system.require_focus = True
    rig.engine.configure(rig.cfg, rig.templates)
    rig.engine.start()
    rig.wait(lambda: rig.engine.stats.casts >= 1)
    title["value"] = "Discord"
    rig.wait(lambda: rig.engine.status == "paused")
    assert rig.engine.pause_reason == "focus"
    assert "focus_lost" in rig.codes()
    title["value"] = "Albion Online Client"
    rig.wait(lambda: "focus_back" in rig.codes() and rig.engine.status == "running")
    rig.engine.stop()
    rig.engine.join(5)


def test_rotates_points_and_stops_after_fail_streak():
    rig = Rig()
    rig.game.spots = [{"x": 800.0, "y": 300.0, "fish": 0, "respawn": 1e12}]  # empty water: never bites
    rig.cfg.cast.points = [[760, 280], [840, 320]]
    rig.cfg.cast.rotate_after = 1
    rig.cfg.bite.bite_timeout_s = 5
    rig.cfg.session.max_fail_streak = 4
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig)
    codes = rig.codes()
    assert codes.count("rotate") >= 3
    assert "fail_streak" in codes
    targets = {(e["params"]["x"] // 40, e["params"]["y"] // 40) for e in rig.events if e["code"] == "cast"}
    assert len(targets) >= 2  # really alternated between the two points


def test_timed_actions_and_time_limit(rig):
    rig.cfg.session.actions = [Action(key="2", every_min=0.5, enabled=True, label="food")]
    rig.cfg.session.max_minutes = 2
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig, timeout=120)
    assert "limit_time" in rig.codes()
    assert rig.inp.keys[0] == "2"
    assert 3 <= len(rig.inp.keys) <= 5  # on start, then every ~30 s over 2 min


def test_breaks_are_scheduled(rig):
    rig.cfg.session.break_every_min = 5
    rig.cfg.session.break_minutes = 0.5
    rig.cfg.session.max_minutes = 12
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig, timeout=180)
    codes = rig.codes()
    assert codes.count("break") >= 1
    assert codes.count("break") == codes.count("break_end")


def test_snapshot_shape(rig):
    snap = rig.engine.snapshot()
    assert snap["status"] == "idle"
    assert {"casts", "catches", "per_hour", "success", "avg_bite", "avg_reel", "active_s"} <= snap["stats"].keys()


def test_quick_stop_then_start_restarts(rig):
    rig.engine.start()
    rig.wait(lambda: rig.engine.stats.casts >= 1)
    rig.engine.stop()
    rig.engine.start()  # immediately, while the old worker may still be unwinding
    rig.wait(lambda: rig.engine.status == "running" and rig.engine.stats.casts >= 1)
    assert rig.engine.running
    rig.engine.stop()
    rig.engine.join(5)
    assert not rig.inp.held
