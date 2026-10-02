import numpy as np

from conftest import Rig
from fishbot.audio import OnsetDetector


def test_onset_detector_hears_splash_over_ambience():
    rng = np.random.default_rng(0)
    det = OnsetDetector(sensitivity=4.0)
    t = 0.0
    for _ in range(100):                                   # 2 s of quiet ambience
        assert not det.feed(rng.normal(0, 150, 960), t)
        t += 0.02
    assert det.feed(rng.normal(0, 3000, 960), t)           # splash
    assert not det.feed(rng.normal(0, 3000, 960), t + 0.02)  # same sound: one onset only
    assert det.onset_after(t - 0.01) and not det.onset_after(t + 0.01)


def test_music_swell_is_not_a_bite():
    det = OnsetDetector(sensitivity=4.0)
    t = np.arange(960) / 48000
    for i in range(200):  # slow low tone getting gradually louder
        amp = 2000 + 20 * i
        assert not det.feed(amp * np.sin(2 * np.pi * 110 * (t + i * 0.02)), i * 0.02)


def test_engine_learns_the_bite_sound_and_uses_it():
    from fishbot.sim import SimAudio
    rig = Rig(seed=3)
    rig.game.ambient = 2.0
    rig.engine.audio = SimAudio(rig.game)
    rig.cfg.session.max_catches = 6
    rig.engine.configure(rig.cfg, rig.templates)
    learned = []
    rig.engine.on_learn = lambda name, tpl, patch: learned.append(patch)
    triggers = []
    orig = rig.engine._hook
    rig.engine._hook = lambda cfg: (triggers.append(rig.engine.last_trigger), orig(cfg))
    rig.engine.start()
    rig.engine.join(120)
    assert rig.engine.stats.catches == 6 and rig.game.early == 0
    prints = rig.engine._cfg.bite.sound_prints
    assert len(prints) >= 2 and "learn_sound" in rig.codes()
    assert any("sound_prints" in p.get("bite", {}) for p in learned)  # handed over to be saved
    # the splash at the float is what hooks; the sound may only confirm it, never hook on its own
    assert set(triggers) <= {"splash", "sound"} and triggers[:2] != ["sound", "sound"]


def test_bite_print_rejects_other_game_sounds():
    from fishbot.audio import MATCH, bite_print, similarity
    from fishbot.sim import SimAudio
    rig = Rig(seed=4)
    audio = SimAudio(rig.game)
    t = rig.clock.now() + 1.0
    plan = [(t + i, k) for i, k in enumerate(["bite", "croak", "bite", "chime", "nibble", "bite", "land"])]
    rig.game.sounds.extend(plan)
    rig.clock.t = t + len(plan) + 1
    audio._sync()
    prints = []
    for at, kind in plan:
        if kind == "bite":
            prints = audio.remember(at, prints)
    ref = bite_print(prints)
    assert ref is not None and len(prints) == 3
    for at, kind in plan:
        onset = min(audio.detector.onsets, key=lambda o: abs(o - at))
        score = similarity(audio.envelope(onset, full=True)[0], ref)
        assert (score >= MATCH) == (kind == "bite"), (kind, score)


def test_peak_meter_onset():
    det = OnsetDetector(sensitivity=4.0, floor=0.04, history=300)
    for i in range(100):
        assert not det.feed_level(0.02 + 0.005 * (i % 3), i * 0.01)   # ambience / music bed
    assert det.feed_level(0.4, 1.0)                                    # splash
    assert not det.feed_level(0.35, 1.01)
