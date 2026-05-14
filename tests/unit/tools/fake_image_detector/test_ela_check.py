import asyncio
import io

from PIL import Image

from tools.fake_image_detector.checks.ela_check import ELACheck


def run(coro):
    return asyncio.run(coro)


def _transparent_png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_transparent_png_is_processed_without_skip():
    check = ELACheck()
    result = run(check.run(_transparent_png_bytes(), {"input_type": "face"}))

    assert result.skipped is False
    assert result.error is None
    assert "ela_mean" in result.signals
    assert "ela_max" in result.signals
    assert result.normalized_signals is not None
    assert result.normalized_signals.category == "manipulation"
    assert result.normalized_signals.manipulation_score == result.fake_score
