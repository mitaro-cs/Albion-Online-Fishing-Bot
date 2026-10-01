"""Self-update for the Windows exe from GitHub Releases.

Flow: on start (frozen exe only) check the latest release → download the new
exe next to the current one as ``*.new`` → verify its SHA-256 against the
digest GitHub publishes → swap it in once this process exits (a tiny batch
script waits for our PID, replaces the file and optionally restarts).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from . import __version__

log = logging.getLogger(__name__)
REPO = "mitaro-cs/Albion-Online-Fishing-Bot"
ASSET = "AlbionFishingBot.exe"
API = f"https://api.github.com/repos/{REPO}/releases/latest"


def parse_version(tag: str) -> tuple[int, ...]:
    parts = []
    for p in tag.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def is_newer(tag: str, current: str = __version__) -> bool:
    return parse_version(tag) > parse_version(current)


def _get(url: str, timeout: float = 15):
    req = urllib.request.Request(url, headers={"User-Agent": f"AlbionFishingBot/{__version__}",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - fixed https URLs


def latest_release() -> dict | None:
    with _get(API) as r:
        data = json.loads(r.read())
    for a in data.get("assets", []):
        if a.get("name") == ASSET:
            return {"tag": data.get("tag_name", ""), "url": a["browser_download_url"], "size": a.get("size", 0),
                    "sha256": (a.get("digest") or "").removeprefix("sha256:").lower(),
                    "notes": data.get("html_url", "")}
    return None


def download(info: dict, dest: Path, progress=None) -> None:
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    done = 0
    with _get(info["url"], timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress and info.get("size"):
                progress(done / info["size"])
    if info.get("sha256") and h.hexdigest() != info["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError("checksum mismatch")
    if info.get("size") and done != info["size"]:
        tmp.unlink(missing_ok=True)
        raise ValueError("incomplete download")
    os.replace(tmp, dest)


def apply(new_exe: Path, exe: Path, restart: bool) -> None:
    """Hand off to a detached script that swaps the exe after we exit."""
    script = Path(tempfile.gettempdir()) / f"fishbot-update-{os.getpid()}.bat"
    lines = [
        "@echo off", "chcp 65001 >nul", "set n=0", ":retry", "ping -n 2 127.0.0.1 >nul",
        f'move /y "{new_exe}" "{exe}" >nul 2>nul',
        "set /a n+=1",
        f'if exist "{new_exe}" if %n% lss 90 goto retry',
    ]
    if restart:
        lines.append(f'start "" "{exe}"')
    lines.append('del "%~f0"')
    script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    subprocess.Popen(["cmd", "/c", str(script)], creationflags=flags, close_fds=True)


class Updater:
    """Background check + download; the UI polls ``state``."""

    def __init__(self, enabled: bool):
        self.exe = Path(sys.executable).resolve()
        self.enabled = enabled and getattr(sys, "frozen", False) and sys.platform == "win32"
        self.state = {"status": "off" if not self.enabled else "idle", "current": __version__,
                      "latest": None, "progress": 0.0, "error": ""}
        self.new_exe = self.exe.with_name(self.exe.name + ".new")
        self.restart = False

    @property
    def ready(self) -> bool:
        return self.state["status"] == "ready" and self.new_exe.is_file()

    def start(self) -> None:
        if self.enabled:
            threading.Thread(target=self._run, name="fishbot-update", daemon=True).start()

    def _run(self) -> None:
        st = self.state
        try:
            st["status"] = "checking"
            info = latest_release()
            if not info or not is_newer(info["tag"]):
                st["status"] = "latest"
                return
            st.update(status="downloading", latest=info["tag"])
            download(info, self.new_exe, progress=lambda p: st.update(progress=round(p, 3)))
            st["status"] = "ready"
            log.info("update %s downloaded", info["tag"])
        except Exception as e:  # offline, rate-limited, … — never bother the user
            log.warning("update check failed: %s", e)
            st.update(status="error", error=str(e))

    def finish(self) -> None:
        """Called on exit: swap in the downloaded exe (and relaunch if asked)."""
        if self.ready:
            try:
                apply(self.new_exe, self.exe, self.restart)
            except Exception:
                log.exception("could not apply the update")
