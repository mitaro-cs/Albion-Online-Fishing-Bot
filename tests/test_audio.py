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


class SimAudio:
    """Plays a 'splash' when the simulated fish bites."""

    def __init__(self, game, clock):
        self.game, self.clock = game, clock

    def now(self):
        return self.clock.now()

    def onset_after(self, t):
        self.game.advance()
        return self.game.state == "biting" and self.game.state_t > t


def test_engine_hooks_on_the_sound():
    rig = Rig(seed=3)
    rig.engine.audio = SimAudio(rig.game, rig.clock)
    rig.cfg.session.max_catches = 3
    rig.engine.configure(rig.cfg, rig.templates)
    triggers = []
    orig = rig.engine._hook
    rig.engine._hook = lambda cfg: (triggers.append(rig.engine.last_trigger), orig(cfg))
    rig.engine.start()
    rig.engine.join(90)
    assert rig.engine.stats.catches == 3
    assert triggers and all(t == "sound" for t in triggers)
