import asyncio
import io

import piexif
import pytest
from PIL import Image

from tools.fake_image_detector.checks.exif_check import EXIFCheck


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


def _jpeg_with_exif(zeroth_ifd: dict) -> bytes:
    exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}})
    buf = io.BytesIO()
    piexif.insert(exif_bytes, _jpeg_bytes(), buf)
    return buf.getvalue()


@pytest.fixture()
def check():
    return EXIFCheck()


def test_non_jpeg_is_skipped(check):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    result = run(check.run(buf.getvalue(), {}))
    assert result.skipped is True


def test_jpeg_without_exif_returns_low_confidence_pass(check):
    result = run(check.run(_jpeg_bytes(), {}))
    assert result.passed is True
    assert result.skipped is False
    assert "NO_EXIF_DATA" in result.flags
    assert result.confidence == pytest.approx(0.2)


def test_editing_software_fails(check):
    result = run(check.run(_jpeg_with_exif({piexif.ImageIFD.Software: b"Adobe Photoshop 2024"}), {}))
    assert result.passed is False
    assert result.fake_score == 1.0
    assert "EDITING_SOFTWARE_DETECTED" in result.flags
    assert "photoshop" in result.signals["software"]


def test_camera_exif_passes(check):
    result = run(check.run(
        _jpeg_with_exif({
            piexif.ImageIFD.Make: b"NIKON CORPORATION",
            piexif.ImageIFD.Model: b"NIKON D850",
        }),
        {},
    ))
    assert result.passed is True
    assert result.fake_score == 0.0
    assert result.skipped is False
    assert result.signals.get("make") == "NIKON CORPORATION"
    assert result.signals.get("model") == "NIKON D850"
