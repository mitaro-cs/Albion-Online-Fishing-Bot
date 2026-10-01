import hashlib
import io
import json

import pytest

from fishbot import updater


def test_version_compare():
    assert updater.is_newer("v1.1.0", "1.0.0")
    assert updater.is_newer("v1.10.0", "1.9.3")
    assert not updater.is_newer("v1.0.0", "1.0.0")
    assert not updater.is_newer("v0.9", "1.0.0")


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_release(monkeypatch, blob: bytes, digest: str):
    meta = {"tag_name": "v9.0.0", "html_url": "x", "assets": [
        {"name": "other.zip", "browser_download_url": "nope"},
        {"name": updater.ASSET, "browser_download_url": "https://dl/exe", "size": len(blob), "digest": f"sha256:{digest}"}]}
    monkeypatch.setattr(updater, "_get", lambda url, timeout=15: Resp(json.dumps(meta).encode() if url == updater.API else blob))


def test_download_verifies_checksum(tmp_path, monkeypatch):
    blob = b"MZ" + bytes(range(256)) * 300
    fake_release(monkeypatch, blob, hashlib.sha256(blob).hexdigest())
    info = updater.latest_release()
    assert info["tag"] == "v9.0.0" and info["url"] == "https://dl/exe"
    seen = []
    updater.download(info, tmp_path / "new.exe", progress=seen.append)
    assert (tmp_path / "new.exe").read_bytes() == blob and seen[-1] == 1.0


def test_download_rejects_tampered_file(tmp_path, monkeypatch):
    fake_release(monkeypatch, b"evil", "0" * 64)
    with pytest.raises(ValueError):
        updater.download(updater.latest_release(), tmp_path / "new.exe")
    assert not (tmp_path / "new.exe").exists() and not list(tmp_path.iterdir())


def test_disabled_outside_frozen_exe():
    u = updater.Updater(True)
    assert u.state["status"] == "off" and not u.ready
    u.finish()  # no-op
