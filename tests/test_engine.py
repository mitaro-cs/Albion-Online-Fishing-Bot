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
    rig.cfg.system.auto_learn = False  # with auto-learn on, missing calibration is learned instead
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


def test_rotates_points_and_keeps_fishing_after_misses():
    rig = Rig()
    rig.game.spots = [{"x": 800.0, "y": 300.0, "fish": 0, "respawn": 1e12}]  # empty water: never bites
    rig.cfg.cast.points = [[760, 280], [840, 320]]
    rig.cfg.cast.rotate_after = 1
    rig.cfg.bite.bite_timeout_s = 5
    rig.engine.configure(rig.cfg, rig.templates)
    rig.engine.start()
    rig.wait(lambda: rig.engine.stats.fail_streak >= 15)
    assert rig.engine.running  # misses never stop the loop
    rig.engine.stop()
    rig.engine.join(5)
    codes = rig.codes()
    assert codes.count("rotate") >= 10
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


def test_never_hooks_early_on_nibbles_or_stray_sounds():
    """The float twitches before the bite and the game plays unrelated sounds: neither may hook."""
    from fishbot.sim import SimAudio
    rig = Rig(seed=7)
    rig.game.nibbles = True
    rig.game.ambient = 1.5
    rig.engine.audio = SimAudio(rig.game)
    rig.cfg.session.max_catches = 8
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig, timeout=120)
    assert rig.game.early == 0
    assert rig.engine.stats.catches == 8
    assert rig.engine.stats.casts <= 9


def _fish_until(rig, n, timeout=120):
    rig.wait(lambda: rig.engine.stats.catches >= n, timeout=timeout)


def test_finds_the_bobber_again_after_the_player_moves():
    """After a restart the player stands elsewhere: the old search area no longer has the bobber."""
    rig = Rig(seed=21)
    rig.engine.start()
    _fish_until(rig, 2)
    rig.game.spots = [{"x": 1150.0, "y": 430.0, "fish": 10 ** 6, "respawn": 0.0}]
    rig.cfg.cast.points = [[1150, 430]]
    rig.engine.configure(rig.cfg, rig.templates)   # old bobber area (650..950, 200..400) is now wrong
    _fish_until(rig, 5)
    rig.engine.stop()
    rig.engine.join(5)
    assert "relocate" in rig.codes()
    r = rig.engine._cfg.regions.bobber
    assert r.left <= 1150 <= r.left + r.width and r.top <= 430 <= r.top + r.height


def test_relearns_a_bobber_that_looks_different():
    rig = Rig(seed=22)
    rig.engine.start()
    _fish_until(rig, 2)
    rig.game.skin = 1  # another float (or night light): the learned picture stops matching
    _fish_until(rig, 5, timeout=150)
    rig.engine.stop()
    rig.engine.join(5)
    codes = rig.codes()
    assert "relearn_bobber" in codes and "learn_bobber" in codes


def test_uses_bait_at_start_and_when_it_runs_out():
    rig = Rig(seed=23)
    rig.cfg.bait.enabled = True
    rig.cfg.bait.key = "1"
    rig.cfg.bait.every_catches = 3
    rig.cfg.bait.every_min = 0
    rig.cfg.session.max_catches = 7
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig)
    assert rig.inp.keys == ["1", "1", "1"]  # start, after 3 fish, after 6 fish
    casts = [i for i, e in enumerate(rig.events) if e["code"] == "cast"]
    baits = [i for i, e in enumerate(rig.events) if e["code"] == "bait"]
    assert baits[0] < casts[0]  # used before the first cast, with the line in


def test_next_cast_follows_the_pause_setting():
    rig = Rig(seed=24)
    rig.cfg.session.cooldown_ms = 0
    rig.cfg.session.cooldown_jitter_ms = 0
    rig.cfg.session.max_catches = 3
    rig.engine.configure(rig.cfg, rig.templates)
    run_until_stopped(rig)
    gaps = []
    for i, e in enumerate(rig.events):
        if e["code"] == "caught":
            nxt = next((x for x in rig.events[i:] if x["code"] == "cast"), None)
            if nxt:
                gaps.append(nxt["t"] - e["t"])
    assert gaps and max(gaps) < 1.0, gaps  # no hidden wait for the loot banner


def test_relearns_the_bar_when_the_minigame_moves():
    rig = Rig(seed=25)
    rig.engine.start()
    _fish_until(rig, 2)
    rig.game.bar_at = (420, 560)  # another resolution / UI scale: the bar is somewhere else now
    _fish_until(rig, 5, timeout=150)
    rig.engine.stop()
    rig.engine.join(5)
    codes = rig.codes()
    assert "relearn_bar" in codes and "learn_bar" in codes
    assert abs(rig.engine._cfg.regions.reel.left - 420) <= 8
