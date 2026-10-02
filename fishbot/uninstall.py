"""Remove everything the bot put on disk.

Footprint: the ``data/`` folder (profiles, templates, settings, log, WebView
cache), update leftovers next to the exe, and the exe itself. No registry keys,
no AppData. The exe can't delete itself while running, so a detached script
waits for this process (and the PyInstaller bootloader parent) to exit first.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .updater import clean_env


def run(store) -> dict:
    """Wipe the profiles (registry or folder) now; schedule the files folder and exe for after exit."""
    frozen = bool(getattr(sys, "frozen", False))
    data = Path(store.root).resolve()
    store.backend.drop_tree() if store.in_registry else None
    if sys.platform != "win32" or not frozen:
        shutil.rmtree(data, ignore_errors=True)  # from source: only our data, never the user's checkout
        return {"scheduled": False, "removed": [str(data)]}
    exe = Path(sys.executable).resolve()
    extra = [exe.with_name(exe.name + s) for s in (".new", ".new.part")]
    legacy = exe.parent / "data"  # folder used by versions before the registry
    script = Path(tempfile.gettempdir()) / f"fishbot-uninstall-{os.getpid()}.bat"
    # retry until our process (and the PyInstaller parent) let go of the files
    lines = ["@echo off", "chcp 65001 >nul", "set n=0", ":retry", "ping -n 2 127.0.0.1 >nul",
             f'rmdir /s /q "{data}" 2>nul', f'rmdir /s /q "{legacy}" 2>nul']
    lines += [f'del /f /q "{f}" 2>nul' for f in [*extra, exe]]
    lines += ["set /a n+=1",
              f'if exist "{exe}" if %n% lss 90 goto retry',
              f'if exist "{data}" if %n% lss 90 goto retry',
              f'rmdir "{exe.parent}" 2>nul',  # only if now empty
              'del "%~f0"']
    script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    subprocess.Popen(["cmd", "/c", str(script)], creationflags=flags, close_fds=True, env=clean_env())
    return {"scheduled": True, "removed": [str(data), str(exe), "HKCU\\" + "Software\\AlbionFishingBot"]}
