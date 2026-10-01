"""Small OS helpers: DPI awareness, foreground window title, notification beep."""
from __future__ import annotations

import os
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"


def enable_dpi_awareness() -> None:
    """Make screen and cursor coordinates physical pixels (must run before any capture)."""
    if not IS_WINDOWS:
        return
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


def foreground_title() -> str | None:
    """Title of the focused window, or None when the platform can't tell."""
    if not IS_WINDOWS:
        return None
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def beep(kind: str = "info") -> None:
    if not IS_WINDOWS:
        return
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONHAND if kind == "error" else winsound.MB_OK)
    except (ImportError, RuntimeError):
        pass


def open_path(path: str) -> None:
    if IS_WINDOWS:
        os.startfile(path)  # noqa: S606 - local folder opened on user request
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
