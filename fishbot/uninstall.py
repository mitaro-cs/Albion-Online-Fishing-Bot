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


def targets(data_dir: Path) -> dict:
    frozen = bool(getattr(sys, "frozen", False))
    exe = Path(sys.executable).resolve() if frozen else None
    extra = [exe.with_name(exe.name + s) for s in (".new", ".new.part")] if exe else []
    return {"data": Path(data_dir).resolve(), "exe": exe, "extra": extra, "frozen": frozen}


def run(data_dir: Path) -> dict:
    """Schedule removal; the caller must exit the app right after."""
    t = targets(data_dir)
    if sys.platform != "win32" or not t["frozen"]:
        # running from source: only the data we created; the code is the user's checkout
        shutil.rmtree(t["data"], ignore_errors=True)
        return {"scheduled": False, "removed": [str(t["data"])]}
    pids = {os.getpid(), os.getppid()}
    script = Path(tempfile.gettempdir()) / f"fishbot-uninstall-{os.getpid()}.bat"
    lines = ["@echo off", ":wait"]
    for pid in pids:
        lines.append(f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul && (timeout /t 1 /nobreak >nul & goto wait)')
    lines.append(f'rmdir /s /q "{t["data"]}" 2>nul')
    for f in [*t["extra"], t["exe"]]:
        lines.append(f'del /f /q "{f}" 2>nul')
    lines.append(f'rmdir "{t["exe"].parent}" 2>nul')  # only if now empty
    lines.append('del "%~f0"')
    script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    subprocess.Popen(["cmd", "/c", str(script)], creationflags=flags, close_fds=True)
    return {"scheduled": True, "removed": [str(t["data"]), str(t["exe"])]}
