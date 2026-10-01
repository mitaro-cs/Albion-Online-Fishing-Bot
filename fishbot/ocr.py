"""Text recognition with the OCR engine built into Windows 10/11 (no downloads, offline).

Uses the Windows.Media.Ocr API through PyWinRT with the user's display languages
(so Russian item names work on a Russian Windows). Anywhere else, or if the
engine is missing, ``read_lines`` returns None and callers fall back to images.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)
_engine = None
_failed = False
_lock = threading.Lock()


def _get_engine():
    global _engine, _failed
    if _engine is not None or _failed or sys.platform != "win32":
        return _engine
    try:
        from winrt.windows.media.ocr import OcrEngine
        _engine = OcrEngine.try_create_from_user_profile_languages()
        if _engine is None:
            raise RuntimeError("no OCR language installed")
    except Exception as e:
        _failed = True
        log.warning("Windows OCR unavailable, using images instead: %s", e)
    return _engine


def prepare(img: np.ndarray) -> np.ndarray:
    """Light text on a dark game banner → dark text on white, upscaled: what OCR reads best."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if gray.mean() < 128:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)


async def _recognize(engine, bgra: np.ndarray) -> list[str]:
    from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.storage.streams import DataWriter
    h, w = bgra.shape[:2]
    writer = DataWriter()
    writer.write_bytes(bytes(np.ascontiguousarray(bgra).tobytes()))
    bitmap = SoftwareBitmap(BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.PREMULTIPLIED)
    bitmap.copy_from_buffer(writer.detach_buffer())
    result = await engine.recognize_async(bitmap)
    return [line.text for line in result.lines]


def read_lines(img: np.ndarray) -> list[str] | None:
    with _lock:
        engine = _get_engine()
        if engine is None:
            return None
        try:
            return asyncio.run(_recognize(engine, prepare(img)))
        except Exception as e:
            log.warning("OCR failed: %s", e)
            return None
