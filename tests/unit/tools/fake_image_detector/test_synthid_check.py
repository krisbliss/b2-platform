import asyncio
import io
import sys
from unittest.mock import MagicMock

from PIL import Image

from tools.fake_image_detector.checks.synthid_check import VertexSynthIDCheck


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, prediction_obj: object):
        self.predictions = [prediction_obj]


def _install_mock_vertex(prediction_dict: dict):
    mock_client = MagicMock()
    mock_client.predict.return_value = _FakeResponse(object())

    mock_aiplatform = MagicMock()
    mock_aiplatform.PredictionServiceClient.return_value = mock_client

    mock_json_format = MagicMock()
    mock_json_format.MessageToDict.return_value = prediction_dict

    sys.modules["google.cloud.aiplatform_v1"] = mock_aiplatform
    sys.modules["google.protobuf.json_format"] = mock_json_format


def _remove_mock_vertex() -> None:
    sys.modules.pop("google.cloud.aiplatform_v1", None)
    sys.modules.pop("google.protobuf.json_format", None)


def test_synthid_detected_sets_flag_and_score(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_SYNTHID_ENDPOINT_ID", "123456789")
    _install_mock_vertex(
        {
            "watermark_likelihood": 0.93,
            "confidence": 0.88,
            "watermark_detected": True,
        }
    )

    try:
        result = run(VertexSynthIDCheck().run(_jpeg_bytes(), {}))
    finally:
        _remove_mock_vertex()

    assert result.check == "synthid"
    assert result.skipped is False
    assert result.passed is False
    assert result.fake_score == 0.93
    assert result.confidence == 0.88
    assert "SYNTHID_WATERMARK_DETECTED" in result.flags


def test_missing_endpoint_skips(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("VERTEX_SYNTHID_ENDPOINT_ID", raising=False)

    result = run(VertexSynthIDCheck().run(_jpeg_bytes(), {}))

    assert result.skipped is True
    assert result.passed is True
    assert result.confidence == 0.0