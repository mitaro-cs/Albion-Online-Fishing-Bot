"""Native window (pywebview → WebView2 on Windows) around the local UI server."""
from __future__ import annotations

import logging

from .api import Api

log = logging.getLogger(__name__)


def run_window(api: Api, url: str, debug: bool = False) -> bool:
    """Blocks until the window closes. Returns False if no native webview is available."""
    try:
        import webview
    except ImportError:
        log.warning("pywebview is not installed — opening the UI in your browser instead")
        return False
    try:
        window = webview.create_window(
            "Albion Fishing Bot", url, width=1440, height=900, min_size=(1024, 680),
            background_color="#0c0c0e", text_select=False)
        api.attach_window(window)
        webview.start(debug=debug)
        return True
    except Exception:
        log.exception("native window failed — falling back to the browser")
        api.attach_window(None)
        return False
