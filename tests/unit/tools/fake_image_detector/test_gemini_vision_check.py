import asyncio
import io
import json
import os
import sys
from unittest.mock import MagicMock

from PIL import Image

from tools.fake_image_detector.checks.gemini_vision_check import GeminiVisionCheck


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _install_mock_genai(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    sys.modules["google.genai"] = mock_genai
    sys.modules["google.genai.types"] = MagicMock()
    return mock_client


def _remove_mock_genai() -> None:
    sys.modules.pop("google.genai", None)
    sys.modules.pop("google.genai.types", None)


class TestGeminiVisionCheck:
    def test_check_id(self):
        assert GeminiVisionCheck.check_id == "gemini_vision"

    def test_detects_synthetic_image(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.87,
            "confidence": 0.87,
            "signals": ["unnatural skin texture", "bilateral symmetry"],
            "flags": ["GAN_ARTIFACTS", "UNNATURAL_TEXTURE"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is False
        assert result.skipped is False
        assert result.fake_score == 0.87
        assert result.confidence == 0.87
        assert "GAN_ARTIFACTS" in result.flags
        assert "EDITING_ARTIFACTS" in result.flags
        assert result.human_escalate is False
        assert result.signals["is_deceptive"] is True
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "synthetic"
        assert result.normalized_signals.synthetic_score == 0.87

    def test_possible_stock_flag_triggers_hard_escalation(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.7,
            "confidence": 0.8,
            "signals": ["looks like stock imagery"],
            "flags": ["STOCK_PHOTO_INDICATORS"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert "POSSIBLE_STOCK" in result.flags
        assert result.human_escalate is True
        assert any("POSSIBLE_STOCK" in reason for reason in result.escalation_reasons)

    def test_passes_real_image(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.05,
            "confidence": 0.9,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is True
        assert result.skipped is False
        assert result.fake_score == 0.05
        assert result.confidence == 0.9
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "synthetic"
        assert result.normalized_signals.synthetic_score == 0.05

    def test_markdown_wrapped_json_is_parsed(self):
        payload = json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.75,
            "confidence": 0.75,
            "signals": ["diffusion artifacts"],
            "flags": ["DIFFUSION_ARTIFACTS"],
        })
        _install_mock_genai(f"```json\n{payload}\n```")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is False
        assert result.confidence == 0.75

    def test_json_parse_failure_fails_closed(self):
        _install_mock_genai("sorry, I cannot analyze this image")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 1.0
        assert result.confidence == 1.0
        assert "GEMINI_PARSE_ERROR" in result.flags
        assert "JSON parse error" in result.error

    def test_api_exception_fails_closed(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 1.0
        assert result.confidence == 1.0
        assert "CHECK_RUNTIME_ERROR" in result.flags
        assert "quota exceeded" in result.error

    def test_missing_project_fails_closed(self):
        env_backup = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        sys.modules["google.genai"] = MagicMock()
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiVisionCheck(project=None).run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()
            if env_backup is not None:
                os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 1.0
        assert result.confidence == 1.0
        assert "CHECK_RUNTIME_ERROR" in result.flags
        assert "GOOGLE_CLOUD_PROJECT" in result.error

    def test_confidence_clamped_above_one(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "confidence": 1.5,
            "signals": [],
            "flags": [],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.0,
            "confidence": -0.3,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.confidence == 0.0  # clamp(-0.3) = 0.0
        assert result.fake_score == 0.0

    def test_document_context_is_sanitized_in_prompt_and_signals(self):
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.1,
            "confidence": 0.8,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(),
                {"doc_type": "passport<script>alert(1)</script>", "country": "u s-1; DROP"},
            ))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.signals["doc_type"] == "passportscriptalert1script"
        assert result.signals["country"] == "US1"
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "document_authenticity"
        assert result.normalized_signals.document_type == "passportscriptalert1script"
        assert result.normalized_signals.country_code == "US1"
        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "passportscriptalert1script" in prompt
        assert "from US1" in prompt

    def test_json_parse_failure_includes_bounded_raw_snippet(self):
        _install_mock_genai("not json at all")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert "GEMINI_PARSE_ERROR" in result.flags
        assert "raw_snippet='not json at all'" in result.error
