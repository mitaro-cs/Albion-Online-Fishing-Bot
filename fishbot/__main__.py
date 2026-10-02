"""Entry point: ``python -m fishbot [--demo] [--browser] [--cli]``."""
from __future__ import annotations

import argparse
import os
import logging
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__
from .system import enable_dpi_awareness


def open_store(data: str | None):
    """Where profiles live.

    * ``--data DIR``: a plain folder (development, tests, portable use).
    * Windows exe: the registry (HKCU\\Software\\AlbionFishingBot) for profiles and settings, and
      %LOCALAPPDATA%\\AlbionFishingBot for the log and debug snapshots — nothing next to the exe.
    * From source: ./data.
    """
    from .config import ProfileStore, RegistryBackend
    if data:
        return ProfileStore(Path(data))
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        files = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "AlbionFishingBot"
        backend = RegistryBackend()
        fresh = not backend.children("profiles")
        store = ProfileStore(files, backend)
        legacy = Path(sys.executable).resolve().parent / "data"
        if fresh and (legacy / "profiles").is_dir():  # move older portable data in once
            store.import_from(ProfileStore(legacy))
        return store
    return ProfileStore(Path("data"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fishbot", description="Albion Online fishing bot")
    ap.add_argument("--demo", action="store_true", help="run against the built-in simulator (no game needed)")
    ap.add_argument("--browser", action="store_true", help="open the UI in the default browser")
    ap.add_argument("--cli", action="store_true", help="no UI: start fishing right away, Ctrl+C to quit")
    ap.add_argument("--profile", help="profile to load")
    ap.add_argument("--data", help="keep profiles in this folder instead of the default storage")
    ap.add_argument("--port", type=int, default=0, help="UI server port (default: random free port)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--uninstall", action="store_true", help="delete the data folder (and the exe) and exit")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    store = open_store(args.data)
    if sys.stdout is None or sys.stderr is None:  # windowed exe: no console, log to a file instead
        sys.stdout = sys.stderr = open(store.root / "fishbot.log", "a", encoding="utf-8", buffering=1)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    enable_dpi_awareness()
    if args.uninstall:
        from .uninstall import run
        logging.shutdown()
        print(run(store))
        return 0

    from .api import Api

    if args.demo:
        from .sim import SimGame, SimInput, SimScreen, prepare_profile
        game = SimGame()
        game.nibbles, game.ambient = True, 4.0  # nibbles and stray game sounds, like the real thing
        screen, inp, focus = SimScreen(game), SimInput(game), None
        profile = args.profile or prepare_profile(store)
    else:
        from .capture import MssScreen
        from .controls import system_input
        from .system import foreground_title
        screen, inp, focus, profile = MssScreen(), system_input(), foreground_title, args.profile

    api = Api(store, screen, inp, demo=args.demo, focus=focus, profile=profile, audio=not args.demo)
    if args.demo:
        from .sim import SimAudio
        api.engine.audio = SimAudio(game)
    try:
        if args.cli:
            return run_cli(api)
        from .server import serve
        httpd, url = serve(api, args.port)
        try:
            from .gui import run_window
            if args.browser or not run_window(api, url, args.debug, str(store.root.resolve() / "webview")):
                webbrowser.open(url)
                print(f"UI: {url}\nPress Ctrl+C to quit.")
                while not api.quit.wait(1):
                    pass
        finally:
            httpd.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        api.shutdown()
    return 0


def run_cli(api) -> int:
    from .engine import format_event
    from .vision import VisionError

    def show(e):
        print(time.strftime("%H:%M:%S", time.localtime(e["ts"])), f"{e['level']:<7}", format_event(e), flush=True)

    s = api._cfg.system
    print(f"Hotkeys: {s.hotkey_toggle.upper()} pause/resume, {s.hotkey_stop.upper()} stop. Ctrl+C to quit.")
    api.engine.on_event = show
    try:
        api.engine.start()
    except VisionError as e:
        print(f"Calibration needed: {e.code} {e.params} — run without --cli to calibrate.", file=sys.stderr)
        return 2
    while api.engine.running:
        time.sleep(0.25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
