import asyncio
import io
import json
import os
import sys
from unittest.mock import MagicMock

from PIL import Image

from tools.fake_image_detector.checks.gemini_extract_check import GeminiExtractCheck


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


class TestGeminiExtractCheckSkip:
    def test_check_id(self):
        assert GeminiExtractCheck.check_id == "gemini_extract"

    def test_skips_when_no_doc_type(self):
        result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {}))
        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0

    def test_skips_when_doc_type_has_no_field_list(self):
        result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "unknown_type"}))
        assert result.skipped is True

    def test_skips_when_project_not_set(self):
        env_backup = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        sys.modules["google.genai"] = MagicMock()
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiExtractCheck(project=None).run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
            if env_backup is not None:
                os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup
        assert result.skipped is True
        assert "GOOGLE_CLOUD_PROJECT" in result.error

    def test_skips_on_api_error(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
        assert result.skipped is True
        assert "quota exceeded" in result.error

    def test_skips_on_json_parse_error(self):
        _install_mock_genai("Sorry, I cannot extract fields from this image.")
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
        assert result.skipped is True
        assert "JSON parse error" in result.error


class TestGeminiExtractCheckExtraction:
    def test_extracts_passport_fields_and_stores_in_context(self):
        payload = json.dumps({
            "full_name": "Max Mustermann",
            "document_number": "C01X00T47",
            "date_of_birth": "1985-03-15",
            "expiry_date": "2030-03-14",
            "nationality": "DEU",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "passport"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert result.skipped is True
        assert result.passed is True
        extracted = context.get("extracted_fields", {})
        assert extracted["full_name"] == "Max Mustermann"
        assert extracted["document_number"] == "C01X00T47"
        assert result.signals["extracted"]["date_of_birth"] == "1985-03-15"

    def test_null_values_excluded_from_extracted_fields(self):
        payload = json.dumps({
            "full_name": "Anna Schmidt",
            "document_number": None,
            "date_of_birth": "1990-07-22",
            "expiry_date": None,
            "nationality": "DEU",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "passport"}
        try:
            run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        extracted = context.get("extracted_fields", {})
        assert "document_number" not in extracted
        assert "expiry_date" not in extracted
        assert extracted["full_name"] == "Anna Schmidt"

    def test_extracts_bank_statement_iban(self):
        payload = json.dumps({
            "account_holder": "Max Mustermann",
            "iban": "DE89370400440532013000",
            "account_number": "0532013000",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "bank_statement"}
        try:
            run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert context["extracted_fields"]["iban"] == "DE89370400440532013000"

    def test_extracts_birth_certificate_fields(self):
        payload = json.dumps({
            "full_name": "Emma Müller",
            "date_of_birth": "2000-01-10",
            "place_of_birth": "Berlin",
            "signed_by": "Standesamt Berlin",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "birth_certificate"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        extracted = context.get("extracted_fields", {})
        assert extracted["signed_by"] == "Standesamt Berlin"
        assert result.signals["doc_type"] == "birth_certificate"

    def test_markdown_wrapped_json_is_parsed(self):
        payload = json.dumps({"full_name": "Test User", "date_of_death": "2024-05-01", "place_of_death": "Hamburg", "signed_by": "Amt"})
        _install_mock_genai(f"```json\n{payload}\n```")
        context = {"doc_type": "death_certificate"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert result.skipped is True
        assert context["extracted_fields"]["full_name"] == "Test User"
