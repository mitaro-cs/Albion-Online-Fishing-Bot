"""Loopback HTTP transport for the UI — the same path for the app window and a browser.

Bound to 127.0.0.1 only; every API call must carry the per-run token and a
loopback Host header (blocks other web pages and DNS-rebinding tricks).
"""
from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import PUBLIC, Api

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent / "web"


def _json_default(o):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(type(o).__name__)


TYPES = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".svg": "image/svg+xml"}


def serve(api: Api, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    token = secrets.token_urlsafe(18)

    class Handler(BaseHTTPRequestHandler):
        server_version = "fishbot"

        def log_message(self, fmt, *args):  # keep the console clean
            log.debug(fmt, *args)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
            target = (WEB / rel).resolve()
            if WEB not in target.parents or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self):
            if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Token", "").encode(), token.encode()):
                return self._send(403, b'{"ok":false,"error":"forbidden"}', "application/json")
            name = self.path.removeprefix("/api/")
            if name not in PUBLIC:
                return self._send(404, b'{"ok":false,"error":"unknown"}', "application/json")
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 1 << 20)
                payload = json.loads(self.rfile.read(length) or b"{}")
                args = payload.get("args", []) if isinstance(payload, dict) else []
                if not isinstance(args, list):
                    raise ValueError
            except ValueError:
                return self._send(400, b'{"ok":false,"error":"bad_request"}', "application/json")
            try:
                result = getattr(api, name)(*args)
            except TypeError as e:  # wrong arity from the client
                result = {"ok": False, "error": "invalid", "message": str(e)}
            body = json.dumps(result, ensure_ascii=False, default=_json_default).encode()
            self._send(200, body, "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="fishbot-http", daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}/?token={token}"
